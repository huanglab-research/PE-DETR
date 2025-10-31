# PE-DETR: A Physics-Enhanced Detection Transformer for Mesoscale Eddy Detection

This is the implementation of the paper: **PE-DETR: A Physics-Enhanced Detection Transformer for Mesoscale Eddy Detection**

> ⚠ **Note:** The source code is currently incomplete and will be fully released once the manuscript is accepted by the journal.

## Model Framework
![PE-DETR Framework](figs/pe.jpg)

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
cd models/pedetr/ops
python setup.py build install
python test.py  # Verify installation
cd ../../..
```

## Dataset Preparation

**Train Set:** Manually annotated mesoscale eddy dataset (Aug20, 2016–Feb9, 2021, 1,635 samples, ≈81.8%)

**Test Sets:** Manually annotated mesoscale eddy dataset (Feb10, 2021–Feb9, 2022, 365 samples, ≈18.2%)

## Training/Resume Training

### Single GPU Training
```bash
python main.py \
    -c config/PEDETR_eddy.py \
    --coco_path /path/to/eddy_dataset \
    --use_physics \
    --use_phys_query_init \
    --use_ms_physics
```

### Multi-GPU Training
```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py \
    -c config/PEDETR_eddy.py \
    --coco_path /path/to/eddy_dataset \
    --use_physics \
    --use_phys_query_init \
    --use_ms_physics
```


## Acknowledgment

We appreciate the great work of [Lite-DETR](https://github.com/IDEA-Research/Lite-DETR), [DETR](https://github.com/facebookresearch/detr), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), etc. Please refer to the original repo for more usage and documents.

## 📬 Contact

Feel free to contact me if there is any question. (Yunqi Wang: wangyunqi@stu.ouc.edu.cn)

---
 