#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.think_ablation.common import (
    SYSTEM_PROMPT_VARIANTS,
    action_signature,
    apply_system_prompt_variant,
    build_action_user_prompt,
    build_think_user_prompt,
    crop_observation_prompt,
    exact_action_match,
    fetch_action_response,
    fetch_reasoning_trace,
    grounding_score,
    is_executable_action,
    jaccard_similarity,
    label_think_category,
    normalized_action_match,
    observation_overlap_ratio,
    prompt_generic_ratio,
    read_jsonl,
    render_markdown_report,
    summarize_results,
    write_jsonl,
)


SUPPORTED_MODES = [
    "no_think_raw",
    "with_think_raw",
    "with_think_scaffold",
    "sampled_think",
    "empty_think",
    "shuffled_think",
    "with_think_cropped",
]


def _metrics_for_prediction(sample: dict[str, Any], predicted_action: dict[str, Any] | None) -> dict[str, Any]:
    reference_action = sample["reference_action"]
    return {
        "parse_success": predicted_action is not None,
        "executable_action": is_executable_action(predicted_action),
        "exact_match": exact_action_match(reference_action, predicted_action),
        "normalized_match": normalized_action_match(reference_action, predicted_action),
        "action_type_match": (
            predicted_action is not None
            and predicted_action.get("action_type") == reference_action.get("action_type")
        ),
        "grounding_score": grounding_score(predicted_action, sample["current_observation"]),
    }


async def _run_no_think(
    sample: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_action_tokens: int,
    seed: int | None,
    crop_chars: int | None = None,
    system_prompt_variant: str = "base",
    repeat_index: int = 0,
) -> dict[str, Any]:
    user_prompt = sample["user_prompt"]
    if crop_chars:
        user_prompt = crop_observation_prompt(user_prompt, crop_chars)
    system_prompt = apply_system_prompt_variant(sample["system_prompt"], system_prompt_variant)
    response = await fetch_action_response(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        seed=None if seed is None else seed + repeat_index,
        max_tokens=max_action_tokens,
        enable_thinking=False,
    )
    predicted_action = response["parsed_action"]
    return {
        "mode": "no_think_raw",
        "sample_id": sample["sample_id"],
        "task_id": sample["task_id"],
        "step_id": sample["step_id"],
        "system_prompt_variant": system_prompt_variant,
        "repeat_index": repeat_index,
        "metadata": sample["metadata"],
        "reference_action": sample["reference_action"],
        "predicted_action": predicted_action,
        "metrics": _metrics_for_prediction(sample, predicted_action),
        "raw": response,
        "think": {
            "text": "",
            "length_chars": 0,
            "observation_overlap_ratio": 0.0,
            "generic_ratio": 0.0,
            "category": "no_think",
        },
    }


async def _run_with_think(
    sample: dict[str, Any],
    *,
    mode: str,
    base_url: str,
    api_key: str,
    model: str,
    think_temperature: float,
    action_temperature: float,
    max_think_tokens: int,
    max_action_tokens: int,
    seed: int | None,
    scaffold: bool = False,
    injected_think: str | None = None,
    crop_chars: int | None = None,
    system_prompt_variant: str = "base",
    repeat_index: int = 0,
) -> dict[str, Any]:
    user_prompt = sample["user_prompt"]
    if crop_chars:
        user_prompt = crop_observation_prompt(user_prompt, crop_chars)
    system_prompt = apply_system_prompt_variant(sample["system_prompt"], system_prompt_variant)
    if injected_think is None:
        think_prompt = build_think_user_prompt(user_prompt, scaffold=scaffold)
        think_result = await fetch_reasoning_trace(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=think_prompt,
            temperature=think_temperature,
            top_p=0.95,
            seed=None if seed is None else seed + repeat_index,
            max_tokens=max_think_tokens,
            reasoning_effort="low",
            enable_thinking=True,
        )
        think_text = think_result["text"]
    else:
        think_result = {"text": injected_think, "raw_content": injected_think, "usage": None, "request_id": None}
        think_text = injected_think

    action_prompt = build_action_user_prompt(user_prompt, think_text, scaffold=scaffold)
    action_result = await fetch_action_response(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=action_prompt,
        temperature=action_temperature,
        seed=None if seed is None else seed + repeat_index,
        max_tokens=max_action_tokens,
        enable_thinking=False,
    )
    predicted_action = action_result["parsed_action"]
    return {
        "mode": mode,
        "sample_id": sample["sample_id"],
        "task_id": sample["task_id"],
        "step_id": sample["step_id"],
        "system_prompt_variant": system_prompt_variant,
        "repeat_index": repeat_index,
        "metadata": sample["metadata"],
        "reference_action": sample["reference_action"],
        "predicted_action": predicted_action,
        "metrics": _metrics_for_prediction(sample, predicted_action),
        "raw": {"think": think_result, "action": action_result},
        "think": {
            "text": think_text,
            "length_chars": len(think_text or ""),
            "observation_overlap_ratio": observation_overlap_ratio(think_text or "", sample["current_observation"]),
            "generic_ratio": prompt_generic_ratio(think_text or ""),
            "category": label_think_category(think_text or "", sample["current_observation"]),
        },
    }


async def _run_sampled_think(
    sample: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    think_temperature: float,
    action_temperature: float,
    max_think_tokens: int,
    max_action_tokens: int,
    seed: int | None,
    sample_count: int,
    system_prompt_variant: str = "base",
    repeat_index: int = 0,
) -> dict[str, Any]:
    sampled_actions: list[dict[str, Any]] = []
    think_texts: list[str] = []
    for i in range(sample_count):
        result = await _run_with_think(
            sample,
            mode="sampled_think_member",
            base_url=base_url,
            api_key=api_key,
            model=model,
            think_temperature=think_temperature,
            action_temperature=action_temperature,
            max_think_tokens=max_think_tokens,
            max_action_tokens=max_action_tokens,
            seed=None if seed is None else seed + repeat_index * 1000 + i + 1,
            scaffold=False,
            system_prompt_variant=system_prompt_variant,
            repeat_index=repeat_index,
        )
        sampled_actions.append(
            {
                "think_text": result["think"]["text"],
                "predicted_action": result["predicted_action"],
                "metrics": result["metrics"],
            }
        )
        think_texts.append(result["think"]["text"])

    majority_action = None
    if sampled_actions:
        signatures: dict[str, tuple[int, dict[str, Any] | None]] = {}
        for item in sampled_actions:
            sig = action_signature(item["predicted_action"])
            count, _ = signatures.get(sig, (0, item["predicted_action"]))
            signatures[sig] = (count + 1, item["predicted_action"])
        majority_action = max(signatures.values(), key=lambda item: item[0])[1]

    pairwise_similarity: list[float] = []
    for i in range(len(think_texts)):
        for j in range(i + 1, len(think_texts)):
            pairwise_similarity.append(jaccard_similarity(think_texts[i], think_texts[j]))

    result = {
        "mode": "sampled_think",
        "sample_id": sample["sample_id"],
        "task_id": sample["task_id"],
        "step_id": sample["step_id"],
        "system_prompt_variant": system_prompt_variant,
        "repeat_index": repeat_index,
        "metadata": sample["metadata"],
        "reference_action": sample["reference_action"],
        "predicted_action": majority_action,
        "metrics": _metrics_for_prediction(sample, majority_action),
        "raw": {},
        "think": {
            "text": think_texts[0] if think_texts else "",
            "length_chars": sum(len(text) for text in think_texts) / max(1, len(think_texts)),
            "observation_overlap_ratio": mean(
                observation_overlap_ratio(text, sample["current_observation"]) for text in think_texts
            )
            if think_texts
            else 0.0,
            "generic_ratio": mean(prompt_generic_ratio(text) for text in think_texts) if think_texts else 0.0,
            "category": "sampled_think",
            "self_consistency": sum(pairwise_similarity) / max(1, len(pairwise_similarity)) if pairwise_similarity else 1.0,
        },
        "sampled_actions": sampled_actions,
    }
    return result


async def main_async(args: argparse.Namespace) -> None:
    samples = read_jsonl(args.dataset)
    if args.max_samples:
        samples = samples[: args.max_samples]

    rng = random.Random(args.seed)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    system_prompt_variants = [variant.strip() for variant in args.system_prompt_variants.split(",") if variant.strip()]
    for mode in modes:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode: {mode}")
    for variant in system_prompt_variants:
        if variant not in SYSTEM_PROMPT_VARIANTS:
            raise ValueError(f"Unsupported system prompt variant: {variant}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_results.jsonl"
    if raw_path.exists():
        raw_path.unlink()

    results: list[dict[str, Any]] = []
    think_pool_by_action: dict[str, list[str]] = {}
    with_think_cache: dict[str, dict[str, Any]] = {}

    for sample in samples:
        action_type = sample["reference_action"]["action_type"]
        think_pool_by_action.setdefault(action_type, [])
        new_rows: list[dict[str, Any]] = []
        for system_prompt_variant in system_prompt_variants:
            for repeat_index in range(args.repeats):
                if "no_think_raw" in modes:
                    new_rows.append(
                        await _run_no_think(
                            sample,
                            base_url=args.base_url,
                            api_key=args.api_key,
                            model=args.model,
                            temperature=args.action_temperature,
                            max_action_tokens=args.max_action_tokens,
                            seed=args.seed,
                            system_prompt_variant=system_prompt_variant,
                            repeat_index=repeat_index,
                        )
                    )

                if "with_think_raw" in modes:
                    res = await _run_with_think(
                        sample,
                        mode="with_think_raw",
                        base_url=args.base_url,
                        api_key=args.api_key,
                        model=args.model,
                        think_temperature=args.think_temperature,
                        action_temperature=args.action_temperature,
                        max_think_tokens=args.max_think_tokens,
                        max_action_tokens=args.max_action_tokens,
                        seed=args.seed,
                        scaffold=False,
                        system_prompt_variant=system_prompt_variant,
                        repeat_index=repeat_index,
                    )
                    new_rows.append(res)
                    with_think_cache[f"{sample['sample_id']}::{system_prompt_variant}::{repeat_index}"] = res
                    if res["think"]["text"]:
                        think_pool_by_action[action_type].append(res["think"]["text"])

                if "with_think_scaffold" in modes:
                    new_rows.append(
                        await _run_with_think(
                            sample,
                            mode="with_think_scaffold",
                            base_url=args.base_url,
                            api_key=args.api_key,
                            model=args.model,
                            think_temperature=args.think_temperature,
                            action_temperature=args.action_temperature,
                            max_think_tokens=args.max_think_tokens,
                            max_action_tokens=args.max_action_tokens,
                            seed=args.seed,
                            scaffold=True,
                            system_prompt_variant=system_prompt_variant,
                            repeat_index=repeat_index,
                        )
                    )

                if "sampled_think" in modes:
                    new_rows.append(
                        await _run_sampled_think(
                            sample,
                            base_url=args.base_url,
                            api_key=args.api_key,
                            model=args.model,
                            think_temperature=args.think_temperature,
                            action_temperature=args.action_temperature,
                            max_think_tokens=args.max_think_tokens,
                            max_action_tokens=args.max_action_tokens,
                            seed=args.seed,
                            sample_count=args.sample_count,
                            system_prompt_variant=system_prompt_variant,
                            repeat_index=repeat_index,
                        )
                    )

                if "empty_think" in modes:
                    new_rows.append(
                        await _run_with_think(
                            sample,
                            mode="empty_think",
                            base_url=args.base_url,
                            api_key=args.api_key,
                            model=args.model,
                            think_temperature=args.think_temperature,
                            action_temperature=args.action_temperature,
                            max_think_tokens=args.max_think_tokens,
                            max_action_tokens=args.max_action_tokens,
                            seed=args.seed,
                            injected_think="",
                            system_prompt_variant=system_prompt_variant,
                            repeat_index=repeat_index,
                        )
                    )

                if "shuffled_think" in modes:
                    pool = think_pool_by_action.get(action_type) or []
                    shuffled_think = None
                    if pool:
                        shuffled_think = rng.choice(pool)
                    else:
                        cached = with_think_cache.get(
                            f"{sample['sample_id']}::{system_prompt_variant}::{repeat_index}"
                        )
                        shuffled_think = cached["think"]["text"] if cached else ""
                    new_rows.append(
                        await _run_with_think(
                            sample,
                            mode="shuffled_think",
                            base_url=args.base_url,
                            api_key=args.api_key,
                            model=args.model,
                            think_temperature=args.think_temperature,
                            action_temperature=args.action_temperature,
                            max_think_tokens=args.max_think_tokens,
                            max_action_tokens=args.max_action_tokens,
                            seed=args.seed,
                            injected_think=shuffled_think,
                            system_prompt_variant=system_prompt_variant,
                            repeat_index=repeat_index,
                        )
                    )

                if "with_think_cropped" in modes:
                    new_rows.append(
                        await _run_with_think(
                            sample,
                            mode="with_think_cropped",
                            base_url=args.base_url,
                            api_key=args.api_key,
                            model=args.model,
                            think_temperature=args.think_temperature,
                            action_temperature=args.action_temperature,
                            max_think_tokens=args.max_think_tokens,
                            max_action_tokens=args.max_action_tokens,
                            seed=args.seed,
                            scaffold=False,
                            crop_chars=args.crop_observation_chars,
                            system_prompt_variant=system_prompt_variant,
                            repeat_index=repeat_index,
                        )
                    )

        results.extend(new_rows)
        with raw_path.open("a") as f:
            for row in new_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize_results(results)
    config = {
        "dataset": str(args.dataset),
        "num_samples": len(samples),
        "base_url": args.base_url,
        "model": args.model,
        "modes": modes,
        "system_prompt_variants": system_prompt_variants,
        "seed": args.seed,
        "sample_count": args.sample_count,
        "repeats": args.repeats,
        "think_temperature": args.think_temperature,
        "action_temperature": args.action_temperature,
        "max_think_tokens": args.max_think_tokens,
        "max_action_tokens": args.max_action_tokens,
    }
    (args.output_dir / "summary.json").write_text(json.dumps({"config": config, "summary": summary}, ensure_ascii=False, indent=2))
    report = render_markdown_report(
        title="Qwen3-32B Think Ablation",
        config=config,
        summary=summary,
        sample_examples=results,
    )
    (args.output_dir / "REPORT.md").write_text(report)
    print(json.dumps({"config": config, "summary": summary}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run step-level think ablations on liveweb-arena teacher states.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--modes",
        type=str,
        default="no_think_raw,with_think_raw,with_think_scaffold,sampled_think,empty_think,shuffled_think,with_think_cropped",
    )
    parser.add_argument(
        "--system-prompt-variants",
        type=str,
        default="base,think_brief,think_structured,action_only_strict",
    )
    parser.add_argument("--max-samples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--think-temperature", type=float, default=0.6)
    parser.add_argument("--action-temperature", type=float, default=0.0)
    parser.add_argument("--max-think-tokens", type=int, default=256)
    parser.add_argument("--max-action-tokens", type=int, default=128)
    parser.add_argument("--crop-observation-chars", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
