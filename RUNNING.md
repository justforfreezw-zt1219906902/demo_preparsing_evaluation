# OpenScan 运行指南

## 1. 配置环境

项目自动读取根目录 `.env`：

```dotenv
OPENSCAN_DATASET_DIR=/path/to/dataset
OPENSCAN_U2NET_MODEL=/path/to/U2Net_v1.onnx
OPENSCAN_REFERENCE_MESH=/path/to/reference.stl
OPENSCAN_RECONSTRUCTION_MESH=/path/to/reconstructed.obj
```

数据集目录中，CSV 和图片位于同一级：

```text
dataset/
├── positions.csv
├── default_0_1.jpg
└── ...
```

## 2. 默认配置来源

唯一默认配置是 `configs/default.yaml`。CLI 总是先读取该文件。使用
`--config my-config.yaml` 时，用户文件递归覆盖默认值。

```bash
.venv/bin/python main.py all --config my-config.yaml
```

## 3. 裁剪与 full-resolution

```yaml
crop:
  enabled: true
  mode: manual
  roi_xyxy: [0.25, 0.18, 0.72, 0.88]
  margin_ratio: 0.08
  resize:
    enabled: true
    output_size: [1200, 1000]
```

`roi_xyxy` 是相对于原始图片的归一化坐标 `[x0, y0, x1, y1]`。

普通运行会裁剪并缩放：

```bash
.venv/bin/python main.py preprocess
```

保留裁剪，但不缩放裁剪结果：

```bash
.venv/bin/python main.py preprocess --full-resolution
```

`--full-resolution` 不会关闭裁剪。

## 4. 一次运行完整流程

```bash
.venv/bin/python main.py all --full-resolution
```

顺序为：

```text
验证 → 解析裁剪 → U2Net 分割 → 对象区域质量分析
→ 可选图像增强 → RGBA → Preview → 姿态元数据 → 可用时进行网格对比
```

没有配置外部重建网格时，网格比较自动跳过。

## 5. 分阶段运行

```bash
.venv/bin/python main.py validate
.venv/bin/python main.py quality
.venv/bin/python main.py preprocess --full-resolution
.venv/bin/python main.py export --full-resolution
.venv/bin/python main.py compare
.venv/bin/python main.py report
```

处理过程会输出轻量进度日志。

## 6. 输出

默认预处理输出：

```text
output/processed/
├── rgba/
├── previews/
└── crop_transforms.json
```

默认不会写入独立 RGB、Mask 或 Edge。需要调试 Mask 时设置：

```yaml
debug:
  save_masks: true
```

PyTorch3D 交接目录：

```text
output/exports/pytorch3d/
├── rgba/
├── metadata.json
└── dataset_manifest.csv
```

RGBA 是主要重建输入，其中 alpha 保留柔和抗锯齿边缘。

## 7. 分割模式

默认：

```yaml
mask:
  mode: u2net
  model_env_var: OPENSCAN_U2NET_MODEL
```

可选模式：

```yaml
mask:
  mode: background_subtraction
```

或提供外部 PNG Mask：

```yaml
mask:
  mode: external
  external_dir: masks
```

外部 Mask 放在 `DATASET/masks/`，文件名需与原图主文件名一致。

## 8. 可选预处理

以下开关相互独立：

- `preprocessing.clahe.enabled`
- `preprocessing.highlight_suppression.enabled`
- `preprocessing.sharpen.enabled`
- `preprocessing.brightness_normalization.enabled`

## 9. 测试

```bash
.venv/bin/python -m pytest -q
```

项目不包含 PyTorch3D 重建代码，不修改原始图片，也不执行自由尺度对齐。
