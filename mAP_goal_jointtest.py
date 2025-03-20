import json
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import sys
from pathlib import Path
import lightning as L
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, Dataset, Subset
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
        self.image_to_segs, self.org_images, self.filename_to_caption = self._create_mappings()

    def _create_mappings(self):
        image_to_segs = {}
        org_images = set()
        filename_to_caption = {}
        for item in self.data_list:
            filename = item['filename']
            filename_to_caption[filename] = item['caption']
            if 'segment_with_background' in filename:
                org_filename = get_org_filename(filename)
                if org_filename not in image_to_segs:
                    image_to_segs[org_filename] = []
                image_to_segs[org_filename].append(item)
            else:
                org_images.add(filename)
                image_to_segs[filename] = []
        return image_to_segs, org_images, filename_to_caption

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
        data = self.processor(images=image, text=caption, return_tensors="pt", truncation=True, padding="max_length", max_length=args.new_max_token)
        return data.pixel_values[0], data.input_ids[0], self.data_list[idx]["filename"]

def process_chunk(fabric, model, data_loader):
    images = []
    texts = []
    filenames = []
    for samples in data_loader:
        image, text, filename = samples
        with torch.no_grad():
            x = model(pixel_values=image.to(fabric.device), input_ids=text.to(fabric.device))
        x_i = F.normalize(x.image_embeds)
        x_t = F.normalize(x.text_embeds)
        images.append(x_i)
        texts.append(x_t)
        filenames.extend(filename)
    return torch.cat(images), torch.cat(texts), filenames

def compute_similarity(images, texts):
    return torch.mm(images, texts.t())

def compute_ap(ranks, relevant_items, k=None):
    if not relevant_items:
        return 0.0
    score = 0.0
    num_hits = 0.0
    for i, item in enumerate(ranks[:k] if k else ranks):
        if item in relevant_items:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / len(relevant_items)

def get_org_filename(filename):
    if 'segment_with_background' in filename:
        parts = filename.split('/')
        org_filename = parts[-2].split('_results')[0]
        if not org_filename.endswith('.jpg'):
            org_filename += '.jpg'
        return org_filename
    return filename

def get_relevant_items(filename, all_filenames, image_to_segs, org_images, filename_to_caption):
    org_filename = get_org_filename(filename)
    
    if filename in org_images:
        # Query is an org image
        relevant_items = [all_filenames.index(seg['filename']) for seg in image_to_segs[filename]]
    else:
        # Query is a seg image
        relevant_items = []
        if org_filename in all_filenames:
            relevant_items.append(all_filenames.index(org_filename))
        if filename in all_filenames:
            relevant_items.append(all_filenames.index(filename))

    return relevant_items

def get_relevant_captions(filename, image_to_segs, org_images, filename_to_caption):
    org_filename = get_org_filename(filename)
    
    if filename in org_images:
        # Query is an org image
        relevant_captions = [filename_to_caption[filename]]  # 원본 이미지의 캡션 추가
        relevant_captions.extend([seg['caption'] for seg in image_to_segs[filename]])
    else:
        # Query is a seg image
        relevant_captions = [filename_to_caption[filename]]
        if org_filename in filename_to_caption:
            relevant_captions.append(filename_to_caption[org_filename])

    return relevant_captions

def get_relevant_items_for_text(query_caption, all_filenames, image_to_segs, org_images, filename_to_caption):
    relevant_items = []
    for filename, caption in filename_to_caption.items():
        if caption == query_caption:
            if filename in org_images:
                # 쿼리가 org 이미지의 캡션인 경우
                relevant_items.append(all_filenames.index(filename))  # org 이미지
                relevant_items.extend([all_filenames.index(seg['filename']) for seg in image_to_segs[filename]])  # seg 이미지들
            else:
                # 쿼리가 seg 이미지의 캡션인 경우
                relevant_items.append(all_filenames.index(filename))  # seg 이미지
                org_filename = get_org_filename(filename)
                if org_filename in all_filenames:
                    relevant_items.append(all_filenames.index(org_filename))  # org 이미지
    return list(set(relevant_items))  # 중복 제거

@torch.no_grad()
def test(fabric: L.Fabric, model: torch.nn.Module, query_loader, k=None) -> torch.Tensor:
    fabric.print("Testing ...")
    
    chunk_size = 5000
    dataset_size = len(query_loader.dataset)
    all_images = []
    all_texts = []
    all_filenames = []

    for start_idx in range(0, dataset_size, chunk_size):
        end_idx = min(start_idx + chunk_size, dataset_size)
        chunk_dataset = Subset(query_loader.dataset, range(start_idx, end_idx))
        chunk_loader = DataLoader(chunk_dataset, batch_size=query_loader.batch_size, shuffle=False, num_workers=query_loader.num_workers)
        
        chunk_images, chunk_texts, chunk_filenames = process_chunk(fabric, model, chunk_loader)
        
        all_images.append(chunk_images)
        all_texts.append(chunk_texts)
        all_filenames.extend(chunk_filenames)
        
        torch.cuda.empty_cache()

    all_images = torch.cat(all_images)
    all_texts = torch.cat(all_texts)

    similarity = compute_similarity(all_images, all_texts)

    image_to_segs = query_loader.dataset.image_to_segs
    org_images = query_loader.dataset.org_images
    filename_to_caption = query_loader.dataset.filename_to_caption
    mAP_i2t = 0.0
    mAP_t2i = 0.0

    # Image to Text
    for i, filename in enumerate(all_filenames):
        relevant_captions = get_relevant_captions(filename, image_to_segs, org_images, filename_to_caption)

        i2t_ranks = torch.argsort(similarity[i], descending=True).tolist()
        i2t_relevant = [idx for idx, fn in enumerate(all_filenames) if filename_to_caption[fn] in relevant_captions]
        mAP_i2t += compute_ap(i2t_ranks, i2t_relevant, k)

    # Text to Image
    unique_captions = set(filename_to_caption.values())
    for caption in unique_captions:
        relevant_items = get_relevant_items_for_text(caption, all_filenames, image_to_segs, org_images, filename_to_caption)
        
        caption_index = all_filenames.index(next(filename for filename, cap in filename_to_caption.items() if cap == caption))
        t2i_ranks = torch.argsort(similarity[:, caption_index], descending=True).tolist()
        mAP_t2i += compute_ap(t2i_ranks, relevant_items, k)

    mAP_i2t /= len(all_filenames)
    mAP_t2i /= len(unique_captions)

    fabric.print(f"mAP@{k if k else 'all'} - Text-to-Image: {mAP_t2i:.4f} & Image-to-Text: {mAP_i2t:.4f}")


def main(args):
    fabric = L.Fabric(
        accelerator="cuda", 
        devices=devices,
        precision="bf16-mixed"
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if args.model == 'L-336':
        args.model = 'openai/clip-vit-large-patch14-336'
    elif args.model == 'L':
        args.model = 'openai/clip-vit-large-patch14'
    elif args.model == 'B':
        args.model = 'openai/clip-vit-base-patch16'
    elif args.model == 'G':
        args.model = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
        
    with fabric.device:
        processor = transformers.AutoProcessor.from_pretrained(args.model)
        model = transformers.AutoModel.from_pretrained(args.model).bfloat16()
        longclip_pos_embeddings(model, args.new_max_token)
        model.load_state_dict(torch.load(args.ckpt), strict=False)

    if args.dataset == 'docci':
        query_list = 'datasets/docci_test_joint_sim_max_1:1.json'
    elif args.dataset == 'DCI':
        query_list = 'datasets/DCI_test_joint_sim_max_1:1.json'

    with open(query_list) as f:
        query_list = json.loads(f.read())

    args.query_list = query_list

    query_dataset = QueryLoader(query_list, processor)
    query_loader = DataLoader(query_dataset, num_workers=num_workers, batch_size=micro_batch_size, shuffle=False, drop_last=False, pin_memory=False)
    query_loader = fabric.setup_dataloaders(query_loader)

    model.eval().to(fabric.device)
    test(fabric, model, query_loader, args.k)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='docci')
    parser.add_argument('--new_max_token', default=248, type=int)
    parser.add_argument("--model", type=str, default='B')
    parser.add_argument("--ckpt", type=str, default='')
    parser.add_argument("--k", type=int, default=10, help="Limit rank calculation to top K results. Use None for all ranks.")
    args = parser.parse_args()

    main(args)