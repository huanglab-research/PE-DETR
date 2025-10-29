# PE-DETR: Physics-Enhanced DETR for Object Detection

This is the implementation of the paper: **Physics-Enhanced DETR for Object Detection**

> ⚠ **Note:** This is a research implementation based on DINO with physics-enhanced modules for improved object detection performance.

## Key Features
Physics-enhanced object detection with improved performance
- **Physics Integration**. Incorporates optical flow and physics-based features for enhanced detection.
- **Multi-Scale Processing**. Supports multi-scale physics heads with fusion capabilities.
- **Flexible Architecture**. Compatible with various DETR variants (Deformable DETR, DINO, H-DETR).
- **Efficient Design**. Maintains computational efficiency while adding physics enhancements.

## Datasets
Experiments on **人工标注的中尺度涡数据集**:
中尺度涡旋检测数据集 (Mesoscale Eddy Detection Dataset)

|#|Datasets|Description|Download|
|---|----|-----|-----|
|1|中尺度涡数据集|人工标注的中尺度涡旋检测数据集|Contact authors for access|

> **Note:** 本数据集为人工标注的中尺度涡旋检测数据集，包含反气旋和气旋两类涡旋的标注信息。

## Environment
```bash
# Python 3.7+, PyTorch 1.9+, CUDA 11.1+
pip install -r requirements.txt
```

## Installation

1. Clone this repository
```bash
git clone https://github.com/your-repo/PE-DETR
cd PE-DETR
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Compile CUDA operators
```bash
cd models/dino/ops
python setup.py build install
python test.py  # Verify installation
cd ../../..
```

## Training/Resume Training

### Single GPU Training
```bash
python main.py \
    -c config/DINO/DINO_4scale_cocoeast.py \
    --coco_path /path/to/eddy_dataset \
    --use_physics \
    --phys_channels 32 \
    --use_optflow_proxy \
    --use_phys_query_init \
    --use_ms_physics
```

### Multi-GPU Training
```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py \
    -c config/DINO/DINO_4scale_cocoeast.py \
    --coco_path /path/to/eddy_dataset \
    --use_physics \
    --phys_channels 32 \
    --use_optflow_proxy \
    --use_phys_query_init \
    --use_ms_physics
```

### Swin Transformer Backbone
```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py \
    -c config/DINO/DINO_4scale_swinT.py \
    --coco_path /path/to/eddy_dataset \
    --use_physics \
    --phys_channels 32 \
    --use_ms_physics
```

## Test/Evaluation

### Standard Evaluation
```bash
python main.py \
    --eval \
    -c config/DINO/DINO_4scale_cocoeast.py \
    --coco_path /path/to/eddy_dataset \
    --resume /path/to/checkpoint.pth \
    --use_physics \
    --use_ms_physics
```

### Benchmark Performance
```bash
python main.py \
    --eval \
    -c config/DINO/DINO_4scale_cocoeast.py \
    --coco_path /path/to/eddy_dataset \
    --resume /path/to/checkpoint.pth \
    --benchmark \
    --benchmark_only \
    --use_physics \
    --use_ms_physics
```

## Configuration Options

### Physics Module Parameters
- `--use_physics`: Enable physics feature branch
- `--phys_channels`: Number of physics feature channels (default: 32)
- `--use_optflow_proxy`: Use optical flow as physics proxy
- `--use_phys_query_init`: Initialize queries with physics+visual cues
- `--use_ms_physics`: Use multi-scale physics heads with fusion (default: enabled)

### Model Configuration
- `--enc_scale`: Encoder scale parameter
- `--num_expansion`: Number of expansion layers
- `--backbone`: Backbone architecture (resnet50, swin_T_224_1k)

## Model Architecture
The PE-DETR framework integrates physics-enhanced modules into the DINO architecture for mesoscale eddy detection:

- **Physics Feature Branch**: Processes optical flow and physics-based features for eddy detection
- **Multi-Scale Physics Heads**: Handles features at different scales for various eddy sizes
- **Query Initialization**: Uses physics+visual cues for better query initialization
- **Fusion Modules**: Combines physics and visual features effectively
- **Eddy Classification**: Detects and classifies anticyclonic and cyclonic eddies

## License & Acknowledgment

We are very grateful for these excellent works: [DINO](https://github.com/IDEA-Research/DINO), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [DETR](https://github.com/facebookresearch/detr). Please follow their respective licenses for usage and redistribution. Thanks for their awesome works.

## 📬 Contact

Feel free to contact us if you have any questions:
- Author: [your-email@domain.com](mailto:your-email@domain.com)
- Institution: [Institution Name]

---

## Citation

If you find our work helpful for your research, please consider citing:

```BibTeX
@article{pe_detr_2024,
  title={PE-DETR: Physics-Enhanced DETR for Object Detection},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```