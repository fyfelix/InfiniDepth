# InfiniDepth iBims 官方评估

`evaluation_ibims/` 是当前 InfiniDepth 项目的 iBims 官方评估适配目录。它只消费
已有 synthetic raw depth manifest，使用 `InfiniDepth_DepthSensor` 推理并保存
iBims official evaluator 需要的 `*_results.mat`。

本目录不包含 raw depth 生成或 mask 校验脚本；这些数据应提前由现有 iBims
pipeline 生成。

## 前置条件

iBims 数据集目录默认是：

```text
data/ibims1
```

运行前需要已经存在 synthetic manifest：

```text
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_easy.jsonl
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_medium.jsonl
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_hard.jsonl
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_extreme.jsonl
```

完整官方评估还需要数据集自带文件：

```text
data/ibims1/imagelist.txt
data/ibims1/ibims1_core_mat/
data/ibims1/evaluation_scripts/evaluate_ibims.py
```

当前项目推理依赖按根目录 `INSTALL.md` 安装。完整 official evaluator 还需要
`scipy`、`scikit-image` 和 `scikit-learn`；项目 `requirements.txt` 已覆盖这些依赖。

## 一站式运行

默认使用本地 `.venv/bin/python`：

```bash
./evaluation_ibims/run_all.sh
```

等价于：

```bash
./evaluation_ibims/run_all.sh checkpoints/depth/infinidepth_depthsensor.ckpt
```

服务器 conda 环境示例：

```bash
PYTHON_BIN=/path/to/conda/env/bin/python \
./evaluation_ibims/run_all.sh checkpoints/depth/infinidepth_depthsensor.ckpt
```

额外参数会透传给 `evaluation_ibims/run_all.py`：

```bash
./evaluation_ibims/run_all.sh checkpoints/depth/infinidepth_depthsensor.ckpt \
  --levels easy \
  --max-samples 1 \
  --skip-eval
```

## Python 入口

```bash
.venv/bin/python evaluation_ibims/run_all.py \
  --model-path checkpoints/depth/infinidepth_depthsensor.ckpt \
  --ibims-root data/ibims1 \
  --levels easy medium hard extreme \
  --encoder vitl16 \
  --input-size 768x1024
```

常用参数：

```text
--run-dir <dir>            指定输出根目录
--max-samples <N>          每个 difficulty 只跑前 N 个样本，默认 0 表示全部
--prompt-samples <N>       sparse raw-depth prompt 采样点数，默认 1500
--enable-noise-filter      采样 prompt 前启用当前项目的 strict depth noise filter
--depth-scale <scale>      覆盖 manifest depth_scale，默认使用每行 depth_scale
--skip-infer               跳过推理，使用 --run-dir 下已有 predictions
--skip-eval                跳过官方评估，只生成 MAT prediction
```

## 输出结构

默认输出目录：

```text
evaluation_ibims/output/ibims_<model_stem>_<YYYYMMDD_HHMMSS>/
```

主要内容：

```text
run_args.json
predictions/<level>/<sample>_results.mat
predictions/<level>/infer_args.json
official_eval/<level>/workspace/
official_eval/<level>/official_eval_stdout.txt
eval_summary.csv
eval_summary.json
```

每个 prediction MAT 包含变量 `pred_depths`：

```text
shape: 480x640
dtype: float32
unit: meter
invalid prediction: NaN
```

## 数据处理约定

- RGB 使用当前项目 `load_image` 读取，保留 PIL/EXIF-safe RGB 路径，并 resize 到 `--input-size`。
- iBims synthetic raw depth 不使用项目默认 PNG 毫米换算；它按 manifest 中的
  `depth_scale` 解码，默认 `65535 / 50`，单位 meter。
- raw depth 中非有限值、`<= depth-range min`、`>= depth-range max` 的点会置为无效。
- full raw depth 作为 `gt_depth` 输入模型，sparse sampled raw depth 作为 prompt；
  official GT 只由 iBims evaluator 使用，不参与推理。
- 模型输入 depth 按当前项目推理约定转为 disparity。
- 不做 prediction/GT alignment。

## Smoke Check

本机无 CUDA 时可做轻量检查：

```bash
PYTHONPYCACHEPREFIX=/tmp/cdm_pycache .venv/bin/python -m py_compile evaluation_ibims/*.py
bash -n evaluation_ibims/run_all.sh
.venv/bin/python evaluation_ibims/infer_to_mat.py --help
.venv/bin/python evaluation_ibims/run_all.py --help
.venv/bin/python evaluation_ibims/eval_official.py --help
git diff --check
```

完整推理需要 CUDA，因为当前项目 `InfiniDepth_DepthSensor` 初始化会强制使用 CUDA。
