from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
from ipaddress import ip_address
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

import httpx
import openai

from liveweb_arena.core.agent_protocol import BROWSER_ACTIONS, FunctionCallingProtocol
from liveweb_arena.core.models import BrowserObservation, BrowserAction, CompositeTask, TrajectoryStep
from liveweb_arena.core.task_manager import TaskManager
from liveweb_arena.plugins import get_all_plugins


THINK_SCAFFOLD = """Before deciding the next browser action, reason explicitly in this order:
1. Restate the immediate sub-goal.
2. Summarize the most relevant current page evidence.
3. Compare 1-3 candidate next actions.
4. Commit to the single best next action."""

SYSTEM_PROMPT_VARIANTS: dict[str, str] = {
    "base": "",
    "think_brief": (
        "\n## Additional Decision Rules\n"
        "- Before every action, briefly reason about the immediate sub-goal and the most relevant page evidence.\n"
        "- Use that reasoning to choose the next browser action.\n"
    ),
    "think_structured": (
        "\n## Additional Decision Rules\n"
        "- Before every action, explicitly reason in this order: sub-goal, page evidence, candidate actions, final choice.\n"
        "- Prefer actions that are directly grounded in the current page and task goal.\n"
        "- If the current page already contains the needed evidence, act on it instead of exploring broadly.\n"
    ),
    "action_only_strict": (
        "\n## Additional Decision Rules\n"
        "- Be extremely terse and action-oriented.\n"
        "- Avoid unnecessary exploration and prefer the shortest valid next action.\n"
        "- If the current page is enough, act immediately.\n"
    ),
}

GENERIC_THINK_PATTERNS = [
    re.compile(r"\b(i need to|let me|i should|first,|next,|then,|carefully)\b", re.I),
    re.compile(r"\bthe task is asking\b", re.I),
    re.compile(r"\bI will browse\b", re.I),
]


@dataclass
class StepSample:
    sample_id: str
    task_id: int
    seed: int
    num_subtasks: int
    templates: list[list[str]]
    step_id: int
    step_num: int
    task_goal: str
    system_prompt: str
    user_prompt: str
    current_observation: dict[str, Any]
    reference_action: dict[str, Any]
    history: list[dict[str, Any]]
    metadata: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def jaccard_similarity(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def normalize_url_for_compare(url: str) -> str:
    text = normalize_whitespace(url).lower()
    text = re.sub(r"^https?://", "", text)
    return text.rstrip("/")


def normalize_action(action: dict[str, Any] | None) -> Optional[dict[str, Any]]:
    if not action:
        return None
    action_type = action.get("action_type")
    params = dict(action.get("params") or {})
    if action_type == "stop":
        final = params.get("final")
        if isinstance(final, dict):
            params = {"answers": dict(final.get("answers") or {})}
    return {"action_type": action_type, "params": params}


def action_signature(action: dict[str, Any] | None) -> str:
    norm = normalize_action(action)
    if not norm:
        return "NONE"
    return json.dumps(norm, sort_keys=True, ensure_ascii=False)


def is_executable_action(action: dict[str, Any] | None) -> bool:
    norm = normalize_action(action)
    if not norm:
        return False
    action_type = norm["action_type"]
    spec = BROWSER_ACTIONS.get(action_type)
    if not spec:
        return False
    params = norm.get("params") or {}
    required = spec["parameters"].get("required", [])
    return all(key in params and params[key] not in (None, "", {}) for key in required)


def exact_action_match(reference_action: dict[str, Any], predicted_action: dict[str, Any] | None) -> bool:
    return action_signature(reference_action) == action_signature(predicted_action)


def normalized_action_match(reference_action: dict[str, Any], predicted_action: dict[str, Any] | None) -> bool:
    ref = normalize_action(reference_action)
    pred = normalize_action(predicted_action)
    if ref is None or pred is None:
        return False
    if ref["action_type"] != pred["action_type"]:
        return False

    r_params = ref["params"]
    p_params = pred["params"]
    if ref["action_type"] == "goto":
        return normalize_url_for_compare(r_params.get("url", "")) == normalize_url_for_compare(p_params.get("url", ""))
    if ref["action_type"] == "stop":
        return json.dumps(r_params, sort_keys=True, ensure_ascii=False) == json.dumps(
            p_params, sort_keys=True, ensure_ascii=False
        )
    return json.dumps(r_params, sort_keys=True, ensure_ascii=False) == json.dumps(
        p_params, sort_keys=True, ensure_ascii=False
    )


def grounding_score(action: dict[str, Any] | None, observation: dict[str, Any]) -> float:
    norm = normalize_action(action)
    if not norm:
        return 0.0
    obs_text = " ".join(
        [
            observation.get("url", ""),
            observation.get("title", ""),
            observation.get("accessibility_tree", ""),
        ]
    ).lower()
    if norm["action_type"] == "goto":
        url = normalize_url_for_compare((norm.get("params") or {}).get("url", ""))
        if not url:
            return 0.0
        return 1.0 if any(part and part in obs_text for part in url.split("/")) else 0.25
    joined = json.dumps(norm.get("params") or {}, ensure_ascii=False).lower()
    tokens = [tok for tok in tokenize(joined) if len(tok) > 2]
    if not tokens:
        return 0.0
    overlap = sum(1 for tok in tokens if tok in obs_text)
    return overlap / len(tokens)


def observation_overlap_ratio(think_text: str, observation: dict[str, Any]) -> float:
    think_tokens = set(tokenize(think_text))
    if not think_tokens:
        return 0.0
    obs_tokens = set(
        tokenize(
            " ".join(
                [
                    observation.get("url", ""),
                    observation.get("title", ""),
                    observation.get("accessibility_tree", ""),
                ]
            )
        )
    )
    if not obs_tokens:
        return 0.0
    return len(think_tokens & obs_tokens) / len(think_tokens)


def prompt_generic_ratio(think_text: str) -> float:
    tokens = tokenize(think_text)
    if not tokens:
        return 0.0
    generic_hits = 0
    for pattern in GENERIC_THINK_PATTERNS:
        if pattern.search(think_text):
            generic_hits += 1
    return min(1.0, generic_hits / max(1, len(GENERIC_THINK_PATTERNS)))


def label_think_category(think_text: str, observation: dict[str, Any]) -> str:
    text = think_text.lower()
    obs_text = (observation.get("accessibility_tree") or "").lower()
    if any(k in text for k in ["compare", "option", "candidate", "best action"]):
        return "candidate_action_comparison"
    if any(k in text for k in ["summary", "current page", "page shows", "visible", "on this page"]):
        return "page_state_summary"
    if any(k in text for k in ["goal", "sub-goal", "need to answer", "need to find"]):
        return "subgoal_planning"
    if any(k in text for k in ["button", "link", "textbox", "selector", "element"]) or any(
        tok in obs_text for tok in tokenize(text)
    ):
        return "element_grounding"
    if any(k in text for k in ["maybe", "however", "double-check", "verify", "not sure"]):
        return "self_correction"
    if prompt_generic_ratio(think_text) > 0.5:
        return "generic_reasoning"
    return "task_restatement"


def crop_observation_prompt(user_prompt: str, max_accessibility_chars: int) -> str:
    pattern = re.compile(r"(### Accessibility Tree\s+```)(.*?)(```)", re.DOTALL)
    match = pattern.search(user_prompt)
    if not match:
        return user_prompt
    tree = match.group(2)
    cropped_tree = tree[:max_accessibility_chars]
    return user_prompt[: match.start(2)] + cropped_tree + user_prompt[match.end(2) :]


def build_action_user_prompt(user_prompt: str, think_text: str | None, scaffold: bool = False) -> str:
    if not think_text:
        return user_prompt
    scaffold_text = (
        "\nUse the reasoning trace below as a candidate plan. If it conflicts with the page, trust the page."
        if scaffold
        else "\nCandidate reasoning trace:"
    )
    return (
        f"{user_prompt}\n"
        f"{scaffold_text}\n"
        f"<think>\n{think_text.strip()}\n</think>\n"
        "Now output exactly one valid tool call for the next action."
    )


def build_think_user_prompt(user_prompt: str, scaffold: bool = False) -> str:
    extra = (
        "\n" + THINK_SCAFFOLD + "\nReturn only the reasoning trace. Do not output a tool call."
        if scaffold
        else "\nThink through the next browser action. Return only a short reasoning trace. Do not output a tool call."
    )
    return f"{user_prompt}{extra}"


def apply_system_prompt_variant(system_prompt: str, variant: str) -> str:
    suffix = SYSTEM_PROMPT_VARIANTS.get(variant)
    if suffix is None:
        raise ValueError(f"Unknown system prompt variant: {variant}")
    return f"{system_prompt}{suffix}" if suffix else system_prompt


def extract_reasoning_text(message: Any, visible_content: str = "") -> str:
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        return normalize_whitespace(str(reasoning))
    content = getattr(message, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if str(item.get("type")).lower() in {"reasoning", "reasoning_content", "thinking"}:
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return normalize_whitespace(" ".join(parts))
    content_text = visible_content or (content if isinstance(content, str) else "")
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", content_text or "", re.DOTALL | re.IGNORECASE)
    if think_match:
        return normalize_whitespace(think_match.group(1))
    return normalize_whitespace(content_text or "")


def difficulty_bucket(num_subtasks: int, step_index: int, total_steps: int) -> str:
    if num_subtasks <= 1 and step_index <= 1:
        return "easy"
    if num_subtasks == 2 and step_index <= max(1, total_steps // 2):
        return "medium"
    return "hard"


async def reconstruct_task_metadata(
    task_id: int,
    seed: int,
    num_subtasks: int,
    templates: list[list[str]],
) -> tuple[str, str]:
    manager = TaskManager(get_all_plugins())
    tuple_templates = [tuple(item) for item in templates]
    task = await manager.generate_composite_task(
        seed=seed,
        num_subtasks=num_subtasks,
        templates=tuple_templates,
    )
    protocol = FunctionCallingProtocol()
    system_prompt = protocol.build_system_prompt(task)
    return task.combined_intent, system_prompt


def serialize_history(trajectory: list[dict[str, Any]], upto_step_index: int) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for step in trajectory[:upto_step_index]:
        history.append(
            {
                "step_num": step["step_num"],
                "action": step.get("action"),
                "action_result": step.get("action_result", ""),
                "observation": step.get("observation", {}),
                "prompt": step.get("prompt"),
                "raw_response": step.get("raw_response"),
            }
        )
    return history


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_variant_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
        by_variant_mode[f"{row.get('system_prompt_variant', 'base')}::{row['mode']}"].append(row)

    def _summarize_bucket(mode_rows: list[dict[str, Any]]) -> dict[str, Any]:
        parse_success = [1.0 if row["metrics"]["parse_success"] else 0.0 for row in mode_rows]
        executable = [1.0 if row["metrics"]["executable_action"] else 0.0 for row in mode_rows]
        exact = [1.0 if row["metrics"]["exact_match"] else 0.0 for row in mode_rows]
        normalized = [1.0 if row["metrics"]["normalized_match"] else 0.0 for row in mode_rows]
        action_type_acc = [1.0 if row["metrics"]["action_type_match"] else 0.0 for row in mode_rows]
        grounding = [row["metrics"]["grounding_score"] for row in mode_rows]
        think_lengths = [row["think"]["length_chars"] for row in mode_rows if row.get("think")]
        obs_overlap = [row["think"]["observation_overlap_ratio"] for row in mode_rows if row.get("think")]
        generic_ratio = [row["think"]["generic_ratio"] for row in mode_rows if row.get("think")]
        think_categories = Counter(row["think"]["category"] for row in mode_rows if row.get("think"))
        think_texts = [row["think"]["text"] for row in mode_rows if row.get("think", {}).get("text")]
        unique_ratio = len(set(think_texts)) / len(think_texts) if think_texts else 0.0

        return {
            "count": len(mode_rows),
            "parse_success_rate": mean(parse_success) if parse_success else 0.0,
            "executable_action_rate": mean(executable) if executable else 0.0,
            "exact_match_rate": mean(exact) if exact else 0.0,
            "normalized_match_rate": mean(normalized) if normalized else 0.0,
            "action_type_accuracy": mean(action_type_acc) if action_type_acc else 0.0,
            "mean_grounding_score": mean(grounding) if grounding else 0.0,
            "average_think_length": mean(think_lengths) if think_lengths else 0.0,
            "unique_think_ratio": unique_ratio,
            "mean_observation_overlap_ratio": mean(obs_overlap) if obs_overlap else 0.0,
            "mean_prompt_generic_ratio": mean(generic_ratio) if generic_ratio else 0.0,
            "think_category_distribution": dict(think_categories),
        }

    mode_summary: dict[str, Any] = {mode: _summarize_bucket(mode_rows) for mode, mode_rows in by_mode.items()}
    variant_mode_summary: dict[str, Any] = {
        key: _summarize_bucket(mode_rows) for key, mode_rows in by_variant_mode.items()
    }

    paired = defaultdict(dict)
    for row in rows:
        paired[row["sample_id"]][row["mode"]] = row

    coupling: dict[str, Any] = {
        "real_vs_empty_exact_delta": 0.0,
        "real_vs_shuffled_exact_delta": 0.0,
        "same_history_different_think_different_action_rate": 0.0,
    }
    real_empty = []
    real_shuffle = []
    different_action = []
    for mode_map in paired.values():
        if "with_think_raw" in mode_map and "empty_think" in mode_map:
            real_empty.append(
                float(mode_map["with_think_raw"]["metrics"]["exact_match"])
                - float(mode_map["empty_think"]["metrics"]["exact_match"])
            )
        if "with_think_raw" in mode_map and "shuffled_think" in mode_map:
            real_shuffle.append(
                float(mode_map["with_think_raw"]["metrics"]["exact_match"])
                - float(mode_map["shuffled_think"]["metrics"]["exact_match"])
            )
        sampled = mode_map.get("sampled_think")
        if sampled and sampled.get("sampled_actions"):
            signatures = {action_signature(item.get("predicted_action")) for item in sampled["sampled_actions"]}
            different_action.append(1.0 if len(signatures) > 1 else 0.0)
    if real_empty:
        coupling["real_vs_empty_exact_delta"] = mean(real_empty)
    if real_shuffle:
        coupling["real_vs_shuffled_exact_delta"] = mean(real_shuffle)
    if different_action:
        coupling["same_history_different_think_different_action_rate"] = mean(different_action)

    by_variant: dict[str, dict[str, float]] = defaultdict(dict)
    for key, bucket in variant_mode_summary.items():
        variant, mode = key.split("::", 1)
        by_variant[variant][mode] = bucket["exact_match_rate"]

    return {
        "modes": mode_summary,
        "variant_modes": variant_mode_summary,
        "coupling": coupling,
        "system_prompt_variants": by_variant,
        "count": len(rows),
    }


def render_markdown_report(
    *,
    title: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    sample_examples: list[dict[str, Any]],
) -> str:
    lines = [f"# {title}", "", "## 实验设置", ""]
    lines.append(f"- 样本数: `{config.get('num_samples')}`")
    lines.append(f"- 模型: `{config.get('model')}`")
    lines.append(f"- 服务: `{config.get('base_url')}`")
    lines.append(f"- 模式: `{', '.join(config.get('modes', []))}`")
    lines.append("")
    lines.append("## 主要结果")
    lines.append("")
    lines.append("| mode | parse | exec | exact | normalized | type_acc | grounding | think_len | overlap | generic |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for mode, mode_summary in sorted(summary.get("modes", {}).items()):
        lines.append(
            "| {mode} | {parse:.3f} | {exec:.3f} | {exact:.3f} | {norm:.3f} | {type_acc:.3f} | {ground:.3f} | {think_len:.1f} | {overlap:.3f} | {generic:.3f} |".format(
                mode=mode,
                parse=mode_summary["parse_success_rate"],
                exec=mode_summary["executable_action_rate"],
                exact=mode_summary["exact_match_rate"],
                norm=mode_summary["normalized_match_rate"],
                type_acc=mode_summary["action_type_accuracy"],
                ground=mode_summary["mean_grounding_score"],
                think_len=mode_summary["average_think_length"],
                overlap=mode_summary["mean_observation_overlap_ratio"],
                generic=mode_summary["mean_prompt_generic_ratio"],
            )
        )
    lines.append("")
    variant_summary = summary.get("variant_modes", {})
    if variant_summary:
        lines.append("## System Prompt Ablation")
        lines.append("")
        lines.append("| variant::mode | exact | normalized | type_acc | grounding |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for key, bucket in sorted(variant_summary.items()):
            lines.append(
                "| {key} | {exact:.3f} | {norm:.3f} | {type_acc:.3f} | {ground:.3f} |".format(
                    key=key,
                    exact=bucket["exact_match_rate"],
                    norm=bucket["normalized_match_rate"],
                    type_acc=bucket["action_type_accuracy"],
                    ground=bucket["mean_grounding_score"],
                )
            )
        lines.append("")
    coupling = summary.get("coupling", {})
    lines.append("## Coupling 指标")
    lines.append("")
    lines.append(f"- 真实 think vs 空 think exact-match 差值: `{coupling.get('real_vs_empty_exact_delta', 0.0):.3f}`")
    lines.append(f"- 真实 think vs 打乱 think exact-match 差值: `{coupling.get('real_vs_shuffled_exact_delta', 0.0):.3f}`")
    lines.append(
        f"- 同一 history 下不同 think 导致不同 action 的比例: `{coupling.get('same_history_different_think_different_action_rate', 0.0):.3f}`"
    )
    lines.append("")
    lines.append("## 典型样例")
    lines.append("")
    for example in sample_examples[:5]:
        lines.append(f"### {example['sample_id']} / {example['mode']}")
        lines.append("")
        lines.append(f"- 任务: `{example['metadata'].get('templates')}`")
        lines.append(f"- reference action: `{json.dumps(example['reference_action'], ensure_ascii=False)}`")
        lines.append(f"- predicted action: `{json.dumps(example.get('predicted_action'), ensure_ascii=False)}`")
        if example.get("think", {}).get("text"):
            lines.append(f"- think: `{example['think']['text'][:400]}`")
        lines.append(f"- metrics: `{json.dumps(example['metrics'], ensure_ascii=False)}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def create_openai_client(base_url: str, api_key: str, timeout_s: int = 120) -> openai.AsyncOpenAI:
    timeout_config = httpx.Timeout(connect=30.0, read=timeout_s, write=30.0, pool=30.0)
    hostname = urlparse(base_url).hostname or ""
    trust_env = True
    try:
        if hostname in {"localhost", "127.0.0.1"} or ip_address(hostname).is_loopback:
            trust_env = False
    except ValueError:
        trust_env = hostname not in {"localhost"}
    http_client = httpx.AsyncClient(timeout=timeout_config, trust_env=trust_env)
    return openai.AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_config, max_retries=0, http_client=http_client)


def build_reasoning_extra_body(enable_thinking: bool, separate_reasoning: bool, reasoning_effort: str | None) -> dict[str, Any]:
    extra_body: dict[str, Any] = {"request_id": f"think-ablation-{random.randint(0, 2**31-1):x}"}
    extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    extra_body["separate_reasoning"] = separate_reasoning
    if enable_thinking:
        if reasoning_effort:
            extra_body["reasoning"] = {"effort": reasoning_effort}
    else:
        extra_body["reasoning"] = {"enabled": False}
    return extra_body


async def fetch_reasoning_trace(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    seed: int | None,
    max_tokens: int,
    reasoning_effort: str | None = "low",
    enable_thinking: bool = True,
) -> dict[str, Any]:
    client = create_openai_client(base_url=base_url, api_key=api_key, timeout_s=180)
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_tokens=max_tokens,
            max_completion_tokens=max_tokens,
            extra_body=build_reasoning_extra_body(
                enable_thinking=enable_thinking,
                separate_reasoning=True,
                reasoning_effort=reasoning_effort,
            ),
        )
        choice = response.choices[0]
        message = choice.message
        visible_content = message.content if isinstance(message.content, str) else ""
        reasoning_text = extract_reasoning_text(message, visible_content)
        usage = response.usage.model_dump() if response.usage else None
        return {
            "text": reasoning_text,
            "raw_content": visible_content,
            "request_id": getattr(response, "id", None),
            "usage": usage,
        }
    finally:
        await client.close()


async def fetch_action_response(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    seed: int | None,
    max_tokens: int,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    client = create_openai_client(base_url=base_url, api_key=api_key, timeout_s=180)
    protocol = FunctionCallingProtocol()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=protocol.get_tools(),
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            max_completion_tokens=max_tokens,
            extra_body=build_reasoning_extra_body(
                enable_thinking=enable_thinking,
                separate_reasoning=True,
                reasoning_effort=None,
            ),
        )
        choice = response.choices[0]
        tool_calls = choice.message.tool_calls or []
        content = choice.message.content or ""
        parsed_tool_calls = [
            {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tool_calls
        ]
        action = protocol.parse_response(content, parsed_tool_calls)
        usage = response.usage.model_dump() if response.usage else None
        return {
            "parsed_action": {"action_type": action.action_type, "params": action.params} if action else None,
            "tool_calls": parsed_tool_calls,
            "raw_content": content,
            "usage": usage,
            "request_id": getattr(response, "id", None),
        }
    finally:
        await client.close()
