# InfiniDepth HAMMER Evaluation

这个目录是在当前 InfiniDepth 外部项目中适配的 HAMMER 评估入口。原始评估 pipeline 仍通过根目录下的 `run_bs_eval_pipeline` 软链接保留为参考来源，最终可运行脚本放在本目录。

## 适配范围

- 数据集固定为 HAMMER，默认 JSONL 为 `data/HAMMER/test.jsonl`。
- 模型固定为 `InfiniDepth_DepthSensor`，默认 encoder 为 `vitl16`。
- 输入使用 HAMMER sample 中的 `rgb` 和指定 `raw-type` 对应的 raw depth：`d435_depth`、`l515_depth` 或 `tof_depth`。
- `infer.py` 输出每个 sample 的 `HxW float32` `.npy`，内容是 metric depth，单位为 meter。
- `eval.py` 继续复用原 pipeline 的 GT depth 读取、valid mask、固定指标和 CSV/JSON 保存逻辑。

这里没有把评估代码改造成通用 adapter 框架，也没有改动项目训练代码。

## 运行

默认使用 uv 创建的本地 `.venv`：

```bash
./evaluation/run_eval.sh ckpts/infinidepth_depthsensor.ckpt
```

完整参数：

```text
./evaluation/run_eval.sh [model_path=ckpts/infinidepth_depthsensor.ckpt] [raw_type=d435] [encoder=vitl16] [cleanup_npy=false]
```

常用环境变量：

```text
DATASET_PATH          HAMMER JSONL 路径，默认 data/HAMMER/test.jsonl
OUTPUT_DIR            基础输出根目录，默认 evaluation/output
INPUT_SIZE            InfiniDepth 输入尺寸，默认 768x1024
BATCH_SIZE            兼容参数，当前适配器逐样本推理，建议 1
NUM_WORKERS           兼容参数，当前适配器单进程读取，建议 0
MAX_SAMPLES           最多评估样本数，默认 0 表示全部
SAVE_VIS              是否保存可视化，默认 true
PYTHON_BIN            Python 可执行文件，默认优先使用 ./.venv/bin/python
```

示例：

```bash
DATASET_PATH=data/HAMMER/test.jsonl \
OUTPUT_DIR=evaluation/output \
MAX_SAMPLES=1 \
./evaluation/run_eval.sh ckpts/infinidepth_depthsensor.ckpt d435 vitl16 false
```

## 模型与输出

当前适配的是 README 推荐的 RGB + depth sensor metric depth 路径：

- 模型类：`InfiniDepth_DepthSensor`
- 默认 checkpoint：`ckpts/infinidepth_depthsensor.ckpt`
- 默认 encoder：`vitl16`
- 输入 RGB：PIL 按 RGB 读取，resize 到 `INPUT_SIZE`
- 输入 raw depth：复用 `InfiniDepth.utils.io_utils.load_depth`
- 模型内部输入 depth：按官方推理逻辑转成 disparity prompt
- 输出：模型返回 metric depth，保存为原图尺寸 `.npy`

没有默认启用 alignment。HAMMER 的 raw depth 作为 DepthSensor prompt 使用，而不是用 GT depth 对预测做后处理对齐。

## 输出目录

`OUTPUT_DIR` 是基础输出根目录。每次运行会创建时间戳子目录，格式为 `YYYY-mm-dd_HH-MM-SS`：

```text
evaluation/output/<timestamp>/
  args.json
  eval_args.json
  predictions/
    <scene>#<sample>.npy
  visualizations/
    <scene>#<sample>_promptda_vis.jpg
  all_metrics_<timestamp>_False.csv
  mean_metrics_<timestamp>_False.json
```

默认会保存可视化图片。设置 `SAVE_VIS=false` 时，`visualizations/` 目录会创建但不会写入图片。`cleanup_npy=true` 时只删除 `predictions/*.npy`，保留指标、元数据和可视化。

## Smoke Check

本机可做轻量检查：

```bash
.venv/bin/python -B -m py_compile evaluation/infer.py evaluation/eval.py
./evaluation/run_eval.sh --help
.venv/bin/python -B evaluation/infer.py --help
.venv/bin/python -B evaluation/eval.py --help
```

## 已知限制

- InfiniDepth 模型初始化代码要求 CUDA；MacBook 本地只能做参数解析、import 和数据路径 smoke check，不能完整跑模型推理。
- 当前适配不支持 RGB-only `InfiniDepth`。RGB-only 输出是 relative depth，不适合在没有 alignment 的情况下直接做 HAMMER metric depth 评估。
- `batch_size` 保留为兼容参数，当前实现优先保证正确性，逐样本推理。
