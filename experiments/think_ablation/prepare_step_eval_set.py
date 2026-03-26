#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liveweb_arena.core.agent_protocol import FunctionCallingProtocol
from liveweb_arena.core.models import BrowserObservation

from experiments.think_ablation.common import (
    StepSample,
    difficulty_bucket,
    read_jsonl,
    reconstruct_task_metadata,
    serialize_history,
    write_jsonl,
)


def _iter_step_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    trajectory = row.get("trajectory") or []
    total_steps = len(trajectory)
    for step_index, step in enumerate(trajectory):
        action = step.get("action")
        prompt = step.get("prompt")
        if not action or not prompt:
            continue
        candidates.append(
            {
                "task_id": row["task_id"],
                "seed": row["seed"],
                "num_subtasks": row["num_subtasks"],
                "templates": row["templates"],
                "step_id": step_index,
                "step_num": step["step_num"],
                "current_observation": step["observation"],
                "reference_action": action,
                "history": serialize_history(trajectory, step_index),
                "user_prompt": prompt,
                "metadata": {
                    "action_type": action.get("action_type"),
                    "difficulty": difficulty_bucket(row["num_subtasks"], step_index + 1, total_steps + 1),
                    "step_position": "early"
                    if step_index <= 1
                    else ("late" if step_index >= max(2, total_steps - 1) else "mid"),
                    "templates": row["templates"],
                    "score": row.get("score"),
                    "trajectory_steps": row.get("trajectory_steps"),
                },
            }
        )

    final_answer = row.get("final_answer")
    if trajectory and final_answer:
        last_obs = trajectory[-1]["observation"]
        last_history = serialize_history(trajectory, len(trajectory))
        protocol = FunctionCallingProtocol()
        prompt = protocol.build_step_prompt(
            BrowserObservation(
                url=last_obs.get("url", ""),
                title=last_obs.get("title", ""),
                accessibility_tree=last_obs.get("accessibility_tree", ""),
            ),
            [],
            current_step=len(trajectory) + 1,
            max_steps=30,
        )
        candidates.append(
            {
                "task_id": row["task_id"],
                "seed": row["seed"],
                "num_subtasks": row["num_subtasks"],
                "templates": row["templates"],
                "step_id": len(trajectory),
                "step_num": len(trajectory) + 1,
                "current_observation": last_obs,
                "reference_action": {"action_type": "stop", "params": {"final": {"answers": final_answer}}},
                "history": last_history,
                "user_prompt": prompt,
                "metadata": {
                    "action_type": "stop",
                    "difficulty": difficulty_bucket(row["num_subtasks"], len(trajectory) + 1, len(trajectory) + 1),
                    "step_position": "late",
                    "templates": row["templates"],
                    "score": row.get("score"),
                    "trajectory_steps": row.get("trajectory_steps"),
                },
            }
        )
    return candidates


def _select_candidates(candidates: list[dict[str, Any]], max_samples: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        grouped[(item["metadata"]["action_type"], item["metadata"]["difficulty"])].append(item)

    for bucket in grouped.values():
        rng.shuffle(bucket)

    keys = sorted(grouped.keys())
    selected: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int]] = set()
    while len(selected) < max_samples:
        progressed = False
        for key in keys:
            bucket = grouped[key]
            while bucket:
                item = bucket.pop()
                dedupe_key = (item["task_id"], item["step_id"])
                if dedupe_key in seen_keys:
                    continue
                selected.append(item)
                seen_keys.add(dedupe_key)
                progressed = True
                break
            if len(selected) >= max_samples:
                break
        if not progressed:
            break
    return selected


async def _attach_task_metadata(samples: list[dict[str, Any]]) -> list[StepSample]:
    task_cache: dict[tuple[int, int], tuple[str, str]] = {}
    keys = {(sample["task_id"], sample["seed"]) for sample in samples}
    for task_id, seed in keys:
        exemplar = next(sample for sample in samples if sample["task_id"] == task_id and sample["seed"] == seed)
        task_cache[(task_id, seed)] = await reconstruct_task_metadata(
            task_id=task_id,
            seed=seed,
            num_subtasks=exemplar["num_subtasks"],
            templates=exemplar["templates"],
        )

    result: list[StepSample] = []
    for sample in samples:
        task_goal, system_prompt = task_cache[(sample["task_id"], sample["seed"])]
        result.append(
            StepSample(
                sample_id=f"task{sample['task_id']}_step{sample['step_id']}",
                task_id=sample["task_id"],
                seed=sample["seed"],
                num_subtasks=sample["num_subtasks"],
                templates=sample["templates"],
                step_id=sample["step_id"],
                step_num=sample["step_num"],
                task_goal=task_goal,
                system_prompt=system_prompt,
                user_prompt=sample["user_prompt"],
                current_observation=sample["current_observation"],
                reference_action=sample["reference_action"],
                history=sample["history"],
                metadata=sample["metadata"],
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a step-level think ablation eval set from teacher trajectories.")
    parser.add_argument(
        "--teacher-trajectories",
        type=Path,
        default=Path("/data/liveweb_teacher_runs/teacher_dataset_runtimepool_formal_replay_20260324_1555/teacher_trajectories.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.teacher_trajectories)
    all_candidates: list[dict[str, Any]] = []
    for row in rows:
        all_candidates.extend(_iter_step_candidates(row))

    selected = _select_candidates(all_candidates, args.max_samples, args.seed)
    samples = asyncio.run(_attach_task_metadata(selected))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "step_eval_set.jsonl"
    write_jsonl(out_path, [sample.__dict__ for sample in samples])

    action_counts = Counter(sample.metadata["action_type"] for sample in samples)
    difficulty_counts = Counter(sample.metadata["difficulty"] for sample in samples)
    summary = {
        "teacher_trajectories": str(args.teacher_trajectories),
        "num_rows": len(rows),
        "num_candidates": len(all_candidates),
        "num_selected": len(samples),
        "action_type_counts": dict(action_counts),
        "difficulty_counts": dict(difficulty_counts),
        "seed": args.seed,
        "max_samples": args.max_samples,
    }
    (args.output_dir / "step_eval_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
