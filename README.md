# FS-Det: Fine-Grained Detail Enhancement and Scene-Level Semantic Integration for Efficient Small Object Detection in Remote Sensing

This repository is built on top of Ultralytics YOLO and contains the implementation of FS-Det for small object detection in remote sensing imagery.

Status: under review.

## Environment setup

We recommend creating a fresh Python environment and installing the project in editable mode.

```bash
conda create -n fsdet python=3.10 -y
conda activate fsdet
pip install -e .
```

Core dependencies are managed in [pyproject.toml](pyproject.toml).

## Dataset preparation

The training scripts use dataset YAML files under [ultralytics/cfg/datasets/](ultralytics/cfg/datasets/). Before training, update the `path` field in the corresponding dataset YAML so that it points to your local dataset root.

- VisDrone: [ultralytics/cfg/datasets/VisDrone.yaml](ultralytics/cfg/datasets/VisDrone.yaml)
- AI-TODv2: [ultralytics/cfg/datasets/AI-TODv2.yaml](ultralytics/cfg/datasets/AI-TODv2.yaml)

Example:

```yaml
path: /path/to/your/dataset
```

## Model configuration

The default model config used by both training scripts is [ultralytics/cfg/models/v8/yolov8_propose.yaml](ultralytics/cfg/models/v8/yolov8_propose.yaml).

Default pretrained weights:

```text
premodel/yolov8s.pt
```

Please place the pretrained checkpoint at that path, or override it with `--premodel`.

## Training

### VisDrone

The default entry for VisDrone training is [train.py](train.py).

```bash
python train.py
```

Useful overrides:

```bash
python train.py \
  --data ultralytics/cfg/datasets/VisDrone.yaml \
  --config ultralytics/cfg/models/v8/yolov8_propose.yaml \
  --premodel premodel/yolov8s.pt \
  --batch_size 8 \
  --imgsz 640 \
  --epochs 300 \
  --device 0 \
  --project output_dir/visdrone \
  --name exp
```

### AI-TODv2

The default entry for AI-TODv2 training is [train2.py](train2.py).

```bash
python train2.py
```

Useful overrides:

```bash
python train2.py \
  --data ultralytics/cfg/datasets/AI-TODv2.yaml \
  --config ultralytics/cfg/models/v8/yolov8_propose.yaml \
  --premodel premodel/yolov8s.pt \
  --batch_size 8 \
  --imgsz 800 \
  --epochs 300 \
  --device 0 \
  --save_period 5 \
  --project output_dir/aitodv2 \
  --name exp
```

## Common arguments

The main training options exposed by the scripts are:

- `--data`: dataset YAML path
- `--config`: model YAML path
- `--premodel`: pretrained checkpoint path
- `--batch_size`: batch size
- `--imgsz`: input image size
- `--epochs`: number of training epochs
- `--device`: GPU id or `cpu`
- `--workers`: dataloader workers
- `--optimizer`: optimizer type, such as `SGD`, `Adam`, or `AdamW`
- `--project`: output root directory
- `--name`: experiment name
- `--resume`: resume training from a checkpoint
- `--amp`: enable mixed precision training
- `--save_period`: checkpoint save interval in epochs (available in [train2.py](train2.py))

## Outputs

Training results are saved to the directory specified by `--project/--name`, for example:

```text
output_dir/visdrone/exp
```

The trained weights are typically written under:

```text
weights/best.pt
weights/last.pt
```
