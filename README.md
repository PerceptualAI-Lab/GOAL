# GOAL: Global-local Object Alignment Learning

Implement of paper: [GOAL: Global-local Object Alignment Learning](https://drive.google.com/file/d/1E-g68yzf9-BBYe-C1zOxTrai7oFL7ifx/view?usp=sharing)

## Datasets

Please download the datasets from the links below:

* DOCCI Dataset
    * [Download link](https://google.github.io/docci/)

* DCI Dataset
    * [Download link](https://github.com/facebookresearch/DCI)

## Train

You can fine-tuning the CLIP with GOAL method in goal_bbox_local_token_align_only_max_pair.py

You can adjust datasets, ouput path, ... in get_args_parser()

```bash
python goal_bbox_local_token_align_only_max_pair.py
```

## Evaluate

Use your fine-tunned weight

You can evaluate retreival score using retrival_goal.py

You can evaluate mAP score about global+local test set using mAP_goal_jointtset.py

```bash
python retrieval_goal.py --ckpt <path/to/your/weight>
```


```bash
python mAP_goal_jointtest.py --ckpt <path/to/your/weight>
```

## Visualize

You can extract the attention map with you custum weight using visualization_attentionmap_longtestset.py

![visualization attention map example](./images/image5.PNG)

```bash
python visualization_attentionmap_longtestset.py --image_path <path/to/your/image> --output_path <path/to/your/output> --model L --ckpt <path/to/your/weight>
```

<!-- ## Dependencies
* Python >= 3.5
* PyTorch >= 0.4.0
* torchvision
* scipy
* numpy
* scikit_learn

## Current Result
| Re-Ranking | backbone | mAP | rank1 | rank3 | rank5 | rank10 |
|------------|----------|-----|-------|-------|-------|---------|
| yes        | resnet50 | 94.33| 95.58 | 97.54 | 97.92 | 98.46  |
| no         | resnet50 | 86.15| 94.95 | 97.42 | 98.07 | 98.93  |

## Data
The data structure would look like: -->