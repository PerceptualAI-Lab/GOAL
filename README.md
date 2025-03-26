# GOAL⚽: Global-local Object Alignment Learning
Implement of paper: [GOAL: Global-local Object Alignment Learning](https://arxiv.org/abs/2503.17782)

## 🔍 Project Page

Visit our project page for additional information and interactive examples:
* **[https://perceptualai-lab.github.io/GOAL/](https://perceptualai-lab.github.io/GOAL/)**


## 🐳 Docker

Our implementation is also available as a Docker image:
* [Docker Hub](https://hub.docker.com/repository/docker/qkenr0804/goal/general)

```bash
# Pull the image
docker pull qkenr0804/goal:goal
```

## 🏋️ Fine-tuned Weights

Download our fine-tuned weights from the links below:

* 🔍 ViT-Base16 Model: GOAL method fine-tuned with DOCCI
    * [Download link](https://drive.google.com/file/d/19M1QvrnQqRtE0i8Zr0qHZvawL8446cTW/view?usp=drive_link)

* 🔍 ViT-Base16 Model: GOAL method fine-tuned with DCI
    * [Download link](https://drive.google.com/file/d/1zvT1yzds45f-jdVNAR1JadQ3D5bbcgKB/view?usp=drive_link)

* 🔍 ViT-Large14 Model: GOAL method fine-tuned with DOCCI
    * [Download link](https://drive.google.com/file/d/10RpCjDTK9PlOnMhb_TvgAOYXtzR68Xbc/view?usp=drive_link)

* 🔍 ViT-Large14 Model: GOAL method fine-tuned with DCI
    * [Download link](https://drive.google.com/file/d/1jw-b2MqLRLCCHumMLrqS_BHcy8ao0EPE/view?usp=drive_link)

## 📊 Datasets

Please download the datasets from the links below:

* DOCCI Dataset
    * [Download link](https://google.github.io/docci/)

* DCI Dataset
    * [Download link](https://github.com/facebookresearch/DCI)

For our newly proposed evaluation protocols on DCI test set and ShareGPT4V test set, please refer to the JSON files available in the `datasets` folder of this repository.

## 🚀 Training

You can fine-tuning the CLIP with GOAL method in goal_loss_finetuning.py

You can adjust datasets, ouput path, ... in get_args_parser()

```bash
python goal_loss_finetuning.py
```

## 📈 Evaluatation

Use your fine-tunned weight

You can evaluate retreival score using retrival_goal.py

You can evaluate mAP score about global+local test set using mAP_goal_jointtset.py

```bash
python retrieval_goal.py --ckpt <path/to/your/weight>
```


```bash
python mAP_goal_jointtest.py --ckpt <path/to/your/weight>
```

## 👁️ Visualization

You can extract the attention map with you custum weight using visualization_attentionmap.py

![visualization attention map example](./images/image5.PNG)

```bash
python visualization_attentionmap.py --image_path <path/to/your/image> --output_path <path/to/your/output> --model L --ckpt <path/to/your/weight>
```

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{Hyungyu_2025_CVPR,
  author={Hyungyu Choi, Young Kyun Jang, Chanho Eom},
  title={GOAL: Global-local Object Alignment Learning},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}

