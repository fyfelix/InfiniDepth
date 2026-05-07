# InfiniDepth HAMMER / ClearPose / DREDS / TransCG Evaluation

这个目录是 InfiniDepth 的 RGB + depth sensor metric depth 评估入口，已按 `eval_pipeline_cdm` 的新版导出结构整理，但模型加载和推理链路仍固定使用本项目的 `InfiniDepth_DepthSensor`。

```text
evaluation/
  dataset.py
  infer.py
  eval.py
  run_hammer.sh
  run_clearpose.sh
  run_dreds.sh
  run_transcg.sh
  requirements.txt
  utils/
    metric.py
    img_utils.py
```

## 数据约定

- HAMMER JSONL 每行是单样本，使用 `rgb`、`depth`、`depth-range`，并按 `raw-type` 选择 `d435_depth`、`l515_depth` 或 `tof_depth`。GT depth 为 16-bit PNG 毫米值，`depth_scale=1000.0`。
- ClearPose JSONL 每行是 sequence，使用 `rgb`、`rgb-suffix`、`raw_depth-suffix`、`depth-suffix`、`depth-range` 展开帧；固定 `raw-type=d435`，`depth_scale=1000.0`。
- DREDS JSONL 沿用 ClearPose 式 sequence schema，支持 `test_std_catknown.jsonl` 和 `test_std_catnovel.jsonl`；raw / GT depth 为 EXR float meter，`depth_scale=1.0`。
- TransCG JSONL 每行是单样本，直接使用 `rgb`、`d435_depth`、`depth`，可选 `sample_name` 和 `depth-range`；固定 `raw-type=d435`，raw / GT depth 按 uint16 PNG 毫米值读取，`depth_scale=1000.0`。缺少 `depth-range` 时默认 `[0.1, 6.0]`。
- 样本命名统一由 `dataset.py` 生成：HAMMER 为 `scene#stem`，ClearPose 和 DREDS 为 `dir1#dir2#stem`，TransCG 优先使用 JSONL 内 `sample_name`，否则回退到最后两级目录名 `dir1_dir2`。

## 四条运行路线

默认优先使用项目根目录 `.venv/bin/python`，也可以通过 `PYTHON_BIN` 覆盖。

### HAMMER

```bash
DATASET_PATH=data/HAMMER/test.jsonl \
OUTPUT_DIR=/tmp/infinidepth_hammer_eval \
MAX_SAMPLES=1 \
bash evaluation/run_hammer.sh ckpts/infinidepth_depthsensor.ckpt vitl16 d435 false
```

参数：

```text
bash evaluation/run_hammer.sh [model_path=ckpts/infinidepth_depthsensor.ckpt] [encoder=vitl16] [camera_type=d435] [cleanup_npy=false]
```

`camera_type` 支持 `d435`、`l515`、`tof`。

### ClearPose

```bash
DATASET_PATH=data/clearpose/test.jsonl \
OUTPUT_DIR=/tmp/infinidepth_clearpose_eval \
bash evaluation/run_clearpose.sh ckpts/infinidepth_depthsensor.ckpt vitl16 false
```

参数：

```text
bash evaluation/run_clearpose.sh [model_path=ckpts/infinidepth_depthsensor.ckpt] [encoder=vitl16] [cleanup_npy=false]
```

ClearPose 固定按 `raw-type=d435` 运行。

### DREDS

```bash
DREDS_KNOWN_JSONL=data/DREDS/test_std_catknown.jsonl \
DREDS_NOVEL_JSONL=data/DREDS/test_std_catnovel.jsonl \
OUTPUT_ROOT=/tmp/infinidepth_dreds_eval \
bash evaluation/run_dreds.sh ckpts/infinidepth_depthsensor.ckpt vitl16 all false
```

参数：

```text
bash evaluation/run_dreds.sh [model_path=ckpts/infinidepth_depthsensor.ckpt] [encoder=vitl16] [variant=all] [cleanup_npy=false]
```

`variant` 支持 `catknown`、`catnovel`、`all`。`all` 会顺序运行 known 和 novel，此时只能使用 `OUTPUT_ROOT`；单 variant 可用 `OUTPUT_DIR` 指定唯一输出目录。

### TransCG

TransCG 固定按 `raw-type=d435`，默认 JSONL 为 `data/TransCG/transcg/dc_testset_d435_005ratio.jsonl`：

```bash
DATASET_PATH=data/TransCG/transcg/dc_testset_d435_005ratio.jsonl \
OUTPUT_DIR=/tmp/infinidepth_transcg_eval \
MAX_SAMPLES=1 \
bash evaluation/run_transcg.sh ckpts/infinidepth_depthsensor.ckpt vitl16 false
```

参数：

```text
bash evaluation/run_transcg.sh [model_path=ckpts/infinidepth_depthsensor.ckpt] [encoder=vitl16] [cleanup_npy=false]
```

说明：

- TransCG JSONL 每行直接包含 `rgb`、`d435_depth`、`depth`，相对路径以 JSONL 所在目录为根。
- 如果行内有 `sample_name`，推理和评估会优先使用它作为 `predictions/<sample_name>.npy` 文件名。
- TransCG raw / GT depth 按 D435 风格 uint16 PNG 毫米深度读取，因此 `dataset.py` 会把 `depth_scale` 设为 `1000.0`。
- 如果 JSONL 行内没有 `depth-range`，默认评估有效范围为 `[0.1, 6.0]`。

## 环境变量

```text
DATASET_PATH          HAMMER / ClearPose / TransCG JSONL 路径
DREDS_KNOWN_JSONL     DREDS catknown JSONL 路径，默认 data/DREDS/test_std_catknown.jsonl
DREDS_NOVEL_JSONL     DREDS catnovel JSONL 路径，默认 data/DREDS/test_std_catnovel.jsonl
OUTPUT_DIR            单数据集或单 DREDS variant 输出目录
OUTPUT_ROOT           DREDS all 模式默认输出根目录
INPUT_SIZE            InfiniDepth 输入尺寸，默认 768x1024
BATCH_SIZE            兼容参数，当前适配器逐样本推理，默认 1
NUM_WORKERS           兼容参数，当前适配器单进程读取，默认 0
MAX_SAMPLES           最多推理/评估样本数，默认 0 表示全部
SAVE_VIS              是否保存可视化，默认 true
ENABLE_NOISE_FILTER   是否对 raw depth prompt 做严格过滤，默认 false
PROMPT_SAMPLES        最大 raw-depth prompt 采样点数，默认 1500
PYTHON_BIN            Python 可执行文件
```

未设置 `OUTPUT_DIR` / `OUTPUT_ROOT` 时，默认写到 checkpoint 同级目录：

```text
<checkpoint_dir>/hammer_<checkpoint_stub>_data_<camera_type>/
<checkpoint_dir>/clearpose_<checkpoint_stub>_data_d435/
<checkpoint_dir>/dreds_catknown_<checkpoint_stub>/
<checkpoint_dir>/dreds_catnovel_<checkpoint_stub>/
<checkpoint_dir>/transcg_<jsonl_stub>_<checkpoint_stub>_data_d435/
```

## 输出结构

```text
<output_dir>/
  args.json
  eval_args.json
  predictions/
    <sample>.npy
  visualizations/
    <sample>_promptda_vis.jpg
  all_metrics_<timestamp>_False.csv
  mean_metrics_<timestamp>_False.json
```

`infer.py` 默认把预测写入 `predictions/`，可视化写入 `visualizations/`。`eval.py` 默认读取 `predictions/*.npy`，如果找不到会 fallback 到旧版根目录 `<output_dir>/*.npy`。`cleanup_npy=true` 时只删除 `predictions/*.npy`，保留指标、元数据和可视化。

## 关键实现约定

- `infer.py` 保留 `build_model()`、`load_image()`、`load_depth()` 和 InfiniDepth disparity prompt 推理方式，不使用 CDM 的 `RGBDDepth`。
- `dataset.py` 提供 `detect_dataset_kind()`、`load_test_dataset()`、`sample_name_for_dataset()` 和 `sample_name_for_sample()`，用于统一 infer/eval 的数据集分发和命名。
- DREDS 允许 prediction shape 与 GT shape 不一致，`eval.py` 会用 nearest resize 对齐；HAMMER / ClearPose / TransCG 遇到 shape mismatch 会报错。
- `OPENCV_IO_ENABLE_OPENEXR=1` 会在 Python 导入 OpenCV 前设置，`run_dreds.sh` 也会在启动 Python 前导出该变量。

## Smoke Check

```bash
PYTHONPYCACHEPREFIX=/tmp/infinidepth-pycache .venv/bin/python -m py_compile \
  evaluation/dataset.py evaluation/infer.py evaluation/eval.py
bash -n evaluation/run_hammer.sh
bash -n evaluation/run_clearpose.sh
bash -n evaluation/run_dreds.sh
bash -n evaluation/run_transcg.sh
bash evaluation/run_hammer.sh --help
bash evaluation/run_clearpose.sh --help
bash evaluation/run_dreds.sh --help
bash evaluation/run_transcg.sh --help
PYTHONPYCACHEPREFIX=/tmp/infinidepth-pycache .venv/bin/python -B evaluation/infer.py --help
PYTHONPYCACHEPREFIX=/tmp/infinidepth-pycache .venv/bin/python -B evaluation/eval.py --help
git diff --check
```

完整推理需要真实 checkpoint、真实数据和 CUDA 环境。Mac 本地通常只适合做 CLI、import、factory 和静态检查。
