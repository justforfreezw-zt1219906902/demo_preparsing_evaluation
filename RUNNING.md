# OpenScan 工具运行指南

程序入口为项目根目录下的 `main.py`。程序自动读取同目录的 `.env`，原始数据集只读，所有结果写入项目的 `output/`。

## 1. 环境配置

编辑 `.env`：

```dotenv
# 包含 positions.csv 和原始图片的数据集目录
OPENSCAN_DATASET_DIR=/path/to/dataset

# 用于对比的参考 STL
OPENSCAN_REFERENCE_MESH=/path/to/reference.stl

# 外部重建流程生成的模型；尚未生成时可以留空
OPENSCAN_RECONSTRUCTION_MESH=/path/to/reconstructed.obj
```

数据集采用扁平结构，CSV 和图片位于同一级目录：

```text
dataset/
├── positions.csv
├── default_0_1.jpg
├── default_0_2.jpg
└── ...
```

## 2. 运行完整流程

进入项目目录：

```bash
cd /Users/zhaowei/PyCharmMiscProject/xolo_3d_reconstrcuction/demo_preparing_and_evaluation
```

使用完整原始画布和原始图片尺寸运行所有阶段：

```bash
.venv/bin/python main.py all --full-resolution
```

完整流程为：

```text
数据验证
→ 图片质量分析
→ Mask 生成
→ 原图分辨率裁剪和预处理
→ RGB / Mask / RGBA / Edge 输出
→ OpenScan 姿态转换
→ PyTorch3D 数据集导出
→ 网格对比
→ HTML 报告
```

如果 `OPENSCAN_RECONSTRUCTION_MESH` 为空或文件不存在，程序会跳过网格对比，其余阶段仍正常完成。

## 3. 分阶段运行

### 验证数据集

检查 CSV、缺失图片、损坏图片、重复记录和图片尺寸：

```bash
.venv/bin/python main.py validate
```

验证结果只显示在终端，不保存运行历史。

### 图片质量分析

```bash
.venv/bin/python main.py quality
```

输出：

```text
output/quality_report.csv
```

### Mask 与图片预处理

保留完整原始画布和原始图片尺寸：

```bash
.venv/bin/python main.py preprocess --full-resolution
```

使用配置中的输出尺寸，默认为 `1200×900`：

```bash
.venv/bin/python main.py preprocess
```

输出目录：

```text
output/processed/
├── rgb/
├── masks/
├── rgba/
├── edges/
├── previews/
└── crop_transforms.json
```

建议检查 `output/processed/previews/`。对于自动 Mask 不准确的图片，可以提供外部单通道 PNG Mask，并在 YAML 中设置：

```yaml
mask:
  mode: external
```

外部 Mask 放置方式：

```text
dataset/
└── masks/
    ├── default_0_1.png
    ├── default_0_2.png
    └── ...
```

### 导出 PyTorch3D 数据集

全分辨率导出：

```bash
.venv/bin/python main.py export --full-resolution
```

配置尺寸导出：

```bash
.venv/bin/python main.py export
```

输出：

```text
output/exports/pytorch3d/
├── images/
├── masks/
├── edges/
├── metadata.json
└── dataset_manifest.csv
```

`metadata.json` 中的姿态来自 OpenScan 指令角度，标记为 `openscan_commanded`。这些角度用于外部重建初始化，不应视为标定后的真实相机姿态。

### 对比参考 STL 和重建模型

确保 `.env` 中已经配置：

```dotenv
OPENSCAN_REFERENCE_MESH=/path/to/reference.stl
OPENSCAN_RECONSTRUCTION_MESH=/path/to/reconstructed.obj
```

运行：

```bash
.venv/bin/python main.py compare
```

也可以在命令中临时指定文件：

```bash
.venv/bin/python main.py compare \
  --reference /path/to/reference.stl \
  --reconstruction /path/to/reconstructed.obj
```

输出：

```text
output/evaluation/
├── metrics.json
├── metrics.csv
├── meshes/
├── views/
├── distance_histogram.png
└── report.html
```

对比不会执行自由缩放。只有在配置中启用时才执行旋转和平移刚性对齐。

### 生成总报告

```bash
.venv/bin/python main.py report
```

输出：

```text
output/report.html
```

## 4. 使用其他配置文件

所有命令均支持 `--config`：

```bash
.venv/bin/python main.py all \
  --full-resolution \
  --config configs/default.yaml
```

主要参数位于 `configs/default.yaml`，包括：

- 质量分类阈值
- Mask 模式和形态学参数
- 裁剪边距和输出尺寸
- CLAHE 开关
- 高光抑制开关
- 锐化开关
- 亮度归一化开关
- 网格采样数量
- 热力图毫米范围
- 刚性对齐开关

## 5. 临时指定其他数据集

无需修改 `.env`，可以使用：

```bash
.venv/bin/python main.py all \
  --dataset /path/to/another/dataset \
  --full-resolution
```

## 6. 查看帮助

```bash
.venv/bin/python main.py --help
.venv/bin/python main.py all --help
.venv/bin/python main.py compare --help
```

## 7. 运行测试

```bash
.venv/bin/python -m pytest -q
```

## 8. 注意事项

- 原始图片不会被修改。
- 项目不执行 PyTorch3D 重建，只负责准备输入数据和评价外部重建结果。
- `--full-resolution` 会关闭裁剪，输出与原图具有完全相同的宽高，因此会显著增加运行时间和磁盘占用。
- 自动 Mask 不依赖大型神经网络，但反光金属、蓝色反射和物体与转台接触区域可能需要人工检查。
- 最终重建前建议检查 `output/processed/previews/`。
- 参考 STL 可能被外部重建项目用作初始化几何，因此评价结果不代表独立的绝对测量精度。
