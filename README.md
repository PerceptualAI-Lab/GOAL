# GOAL⚽: Global-local Object Alignment Learning

## 🔍 **[Project Page](https://perceptualai-lab.github.io/GOAL/)**

Implement of paper: [GOAL: Global-local Object Alignment Learning](https://arxiv.org/abs/2503.17782)

## 🐳 Docker

Our implementation is also available as a Docker image:
* [Docker Hub](https://hub.docker.com/r/username/goal-project)

```bash
# Pull the image
docker pull username/goal-project:latest

# Run the container
docker run --gpus all -it username/goal-project:latest
```

## 📊 Datasets

Please download the datasets from the links below:

* DOCCI Dataset
    * [Download link](https://google.github.io/docci/)

* DCI Dataset
    * [Download link](https://github.com/facebookresearch/DCI)

## 🚀 Train

You can fine-tuning the CLIP with GOAL method in goal_bbox_local_token_align_only_max_pair.py

You can adjust datasets, ouput path, ... in get_args_parser()

```bash
python goal_bbox_local_token_align_only_max_pair.py
```

## 📈 Evaluate

Use your fine-tunned weight

You can evaluate retreival score using retrival_goal.py

You can evaluate mAP score about global+local test set using mAP_goal_jointtset.py

```bash
python retrieval_goal.py --ckpt <path/to/your/weight>
```


```bash
python mAP_goal_jointtest.py --ckpt <path/to/your/weight>
```

## 👁️ Visualize

You can extract the attention map with you custum weight using visualization_attentionmap_longtestset.py

![visualization attention map example](./images/image5.PNG)

```bash
python visualization_attentionmap_longtestset.py --image_path <path/to/your/image> --output_path <path/to/your/output> --model L --ckpt <path/to/your/weight>
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

