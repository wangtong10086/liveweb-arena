# Think Ablation

这个目录用于在 `liveweb-arena` 已收集的 teacher 轨迹上做 `Qwen3-32B base` 的 step-level think 消融实验。

## 目录结构

- `prepare_step_eval_set.py`
  - 从 teacher 轨迹构造 step-level evaluation set
- `run_ablation.py`
  - 对 step 样本运行 `no-think / with-think / sampled-think / intervention-think` 等模式
- `common.py`
  - 共享的数据结构、指标、OpenAI-compatible 推理辅助函数

## 数据来源

默认输入：

`/data/liveweb_teacher_runs/teacher_dataset_runtimepool_formal_replay_20260324_1555/teacher_trajectories.jsonl`

## 最小使用方式

先构造 step 数据集：

```bash
uv run python experiments/think_ablation/prepare_step_eval_set.py \
  --output-dir /home/xmyf/liveweb-arena/tmp/think_ablation_smoke \
  --max-samples 24
```

再运行小规模消融：

```bash
uv run python experiments/think_ablation/run_ablation.py \
  --dataset /home/xmyf/liveweb-arena/tmp/think_ablation_smoke/step_eval_set.jsonl \
  --output-dir /home/xmyf/liveweb-arena/tmp/think_ablation_run_small \
  --base-url http://127.0.0.1:31003/v1 \
  --api-key local-liveweb-bench \
  --model /home/xmyf/Qwen3-32B \
  --max-samples 6 \
  --sample-count 3
```

## 输出

`run_ablation.py` 会输出：

- `raw_results.jsonl`
- `summary.json`
- `REPORT.md`

## 当前已完成的真实小规模实验

真实实验输出目录：

- `/home/xmyf/liveweb-arena/tmp/think_ablation_run_small`

其中：

- `summary.json`：结构化指标汇总
- `REPORT.md`：简短 markdown 报告

## 备注

- 本实验默认优先使用离线 step 数据，不需要真实 browser rollout。
- 对本地 `127.0.0.1` OpenAI-compatible 服务，代码会禁用系统代理，避免本地请求绕行代理。
