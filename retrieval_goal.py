import json
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
from pathlib import Path
import lightning as L
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
from utils.func import *
from utils.transforms import *

# Hyperparameters
micro_batch_size = 32
devices = 1
num_workers = 1

class QueryLoader(Dataset):
    def __init__(self, data_list, processor):
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)
    
    def _load_image(self, id: int):
        return Image.open(self.data_list[id]["filename"]).convert("RGB")

    def _load_target(self, id: int):
        return self.data_list[id]["caption"]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self._load_image(idx)
        caption = self._load_target(idx)
        data = self.processor(images=image, text=caption, return_tensors="pt", truncation = True, padding = "max_length", max_length=args.new_max_token)
    
        return data.pixel_values[0], data.input_ids[0]
    

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 1번 인덱스만 사용
    fabric = L.Fabric(
        accelerator="cuda", 
        devices=devices,
        precision="bf16-mixed"  # "32"에서 "bf16-mixed"로 변경
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if args.model=='L-336':
        args.model = 'openai/clip-vit-large-patch14-336'
    elif args.model=='L':
        args.model = 'openai/clip-vit-large-patch14'
    elif args.model=='B':
        args.model = 'openai/clip-vit-base-patch16'
    elif args.model=='G':
        args.model = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
        
    with fabric.device:
        processor = transformers.AutoProcessor.from_pretrained(args.model)
        model = transformers.AutoModel.from_pretrained(args.model).bfloat16()
        longclip_pos_embeddings(model, args.new_max_token)
        if args.ckpt:  # ckpt가 제공된 경우에만 로드
            model.load_state_dict(torch.load(args.ckpt), strict=False)

    if args.dataset == 'docci':
        query_list = 'datasets/docci_test.json'
    elif args.dataset =='coco':
        query_list = 'datasets/coco_test.json'
    elif args.dataset =='flickr30k':
        query_list = 'datasets/flickr30k_test.json'    
    elif args.dataset =='DCI':
        query_list = 'datasets/DCI_test.json'
    elif args.dataset =='urban':
        query_list = 'datasets/urban_dataset_test.json'
    elif args.dataset =='sharegpt4v':
        query_list = 'datasets/sharegpt4v_test.json'

    with open(query_list) as f:
        query_list = json.loads(f.read())

    args.query_list = query_list

    query_dataset = QueryLoader(query_list, processor)
    query_loader = DataLoader(query_dataset, num_workers=num_workers, batch_size=micro_batch_size, shuffle=False, drop_last=False, pin_memory=False)
    query_loader = fabric.setup_dataloaders(query_loader)

    model.eval().to(fabric.device)
    test(fabric, model, query_loader)

def compute_AP_and_recall_at_Ks(similarity, label_matrix, Ks):
    # Sort gallery indices based on similarity
    sorted_indices = torch.argsort(similarity, descending=True)
    # Initialize results
    results = {K: {'AP': 0.0, 'recall': 0.0, 'relevant_items': 0} for K in Ks}
    total_relevant_items = label_matrix.sum().item()
    for i, idx in enumerate(sorted_indices):
        if label_matrix[idx]:
            for K in Ks:
                if i < K:
                    results[K]['relevant_items'] += 1
                    precision = results[K]['relevant_items'] / (i + 1)
                    results[K]['AP'] += precision
                    results[K]['recall'] = results[K]['relevant_items'] / total_relevant_items
    for K in Ks:
        results[K]['AP'] /= total_relevant_items
    return results


@torch.no_grad()
def test(fabric: L.Fabric, model: torch.nn.Module, query_loader) -> torch.Tensor:
    fabric.print("Testing ...")
    
    images = torch.tensor([], dtype=torch.float32).to(fabric.device)
    texts = torch.tensor([], dtype=torch.float32).to(fabric.device)

    for samples in query_loader:
        image, text = samples

        x = model(pixel_values=image, input_ids=text)

        x_i = F.normalize(x.image_embeds)
        x_t = F.normalize(x.text_embeds)

        images = torch.cat((images,x_i), dim=0)
        texts = torch.cat((texts,x_t), dim=0)

    # Calculate cosine similarity
    similarity = torch.mm(images, texts.t())
    # Image to Text (I2T)
    sorted_indices_i2t = torch.argsort(similarity, descending=True)
    correct_indices = torch.arange(images.shape[0]).to(fabric.device)
    ranks_i2t = (sorted_indices_i2t == correct_indices[:, None]).nonzero(as_tuple=True)[1]
    # Text to Image (T2I)
    sorted_indices_t2i = torch.argsort(similarity.t(), descending=True)
    ranks_t2i = (sorted_indices_t2i == correct_indices[:, None]).nonzero(as_tuple=True)[1]
    # Calculate recall at different ranks for I2T
    recall_i2t_1 = (ranks_i2t < 1).float().mean().item() * 100
    recall_i2t_5 = (ranks_i2t < 5).float().mean().item() * 100
    recall_i2t_25 = (ranks_i2t < 25).float().mean().item() * 100
    recall_i2t_50 = (ranks_i2t < 50).float().mean().item() * 100
    # Calculate recall at different ranks for T2I
    recall_t2i_1 = (ranks_t2i < 1).float().mean().item() * 100
    recall_t2i_5 = (ranks_t2i < 5).float().mean().item() * 100
    recall_t2i_25 = (ranks_t2i < 25).float().mean().item() * 100
    recall_t2i_50 = (ranks_t2i < 50).float().mean().item() * 100
    # Print recall percentages for T2I
    fabric.print(f"Text-to-Image: {recall_t2i_1:.2f} & {recall_t2i_5:.2f} & {recall_t2i_25:.2f} & {recall_t2i_50:.2f}")
    # Print recall percentages for I2T
    fabric.print(f"Image-to-Text: {recall_i2t_1:.2f} & {recall_i2t_5:.2f} & {recall_i2t_25:.2f} & {recall_i2t_50:.2f}")



if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='urban')
    parser.add_argument('--new_max_token', default=248, type=int)
    parser.add_argument("--model", type=str, default='B')
    parser.add_argument("--ckpt", type=str, default='')
    args = parser.parse_args()

    main(args)