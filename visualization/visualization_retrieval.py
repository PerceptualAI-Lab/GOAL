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
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn.functional as F
from utils.func import *
import matplotlib.pyplot as plt
import cv2
import textwrap

def collate_fn(batch):
    # Find the maximum length in the batch
    max_text_length = max(len(item['input_ids']) for item in batch)
    
    # Lists to store all batch items
    pixel_values = []
    input_ids = []
    attention_masks = []
    image_paths = []
    captions = []
    
    for item in batch:
        # Process image
        pixel_values.append(item['pixel_values'])
        
        # Pad text
        curr_len = len(item['input_ids'])
        padding_len = max_text_length - curr_len
        
        padded_ids = torch.cat([
            item['input_ids'],
            torch.zeros(padding_len, dtype=torch.long)
        ])
        padded_mask = torch.cat([
            item['attention_mask'],
            torch.zeros(padding_len, dtype=torch.long)
        ])
        
        input_ids.append(padded_ids)
        attention_masks.append(padded_mask)
        image_paths.append(item['image_path'])
        captions.append(item['caption'])
    
    # Create batch
    return {
        'pixel_values': torch.stack(pixel_values),
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'image_path': image_paths,
        'caption': captions
    }

class JsonGalleryDataset(Dataset):
    def __init__(self, json_path, processor, max_length=248):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['filename']
        caption = item['caption']
        
        try:
            image = Image.open(image_path).convert("RGB")
            # Process both image and text
            processed_image = self.processor(images=image, return_tensors="pt")
            processed_text = self.processor(
                text=caption,
                return_tensors="pt",
                padding='max_length',
                max_length=self.max_length,
                truncation=True
            )
            
            return {
                'pixel_values': processed_image['pixel_values'].squeeze(0),
                'input_ids': processed_text['input_ids'].squeeze(0),
                'attention_mask': processed_text['attention_mask'].squeeze(0),
                'image_path': image_path,
                'caption': caption
            }
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self.__getitem__(0)

def compute_similarities(model, query_features, gallery_features):
    query_features = F.normalize(query_features, dim=-1)
    gallery_features = F.normalize(gallery_features, dim=-1)
    similarities = torch.mm(query_features, gallery_features.t())
    return similarities

def process_query(model, processor, query, device, is_image=False, max_length=248):
    with torch.no_grad():
        if is_image:
            image = Image.open(query).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
            features = model.get_image_features(pixel_values)
        else:
            inputs = processor(
                text=query,
                return_tensors="pt",
                padding='max_length',
                max_length=max_length,
                truncation=True
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            features = model.get_text_features(input_ids, attention_mask)
    return features

def visualize_results(query, results, output_path, is_image_query=False):
    # Use only one result
    n_results = 1
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    fig, axes = plt.subplots(1, 1 + n_results, figsize=(10, 4))  # Adjust figure size
    
    if is_image_query:
        # For image query
        query_img = Image.open(query)
        axes[0].imshow(query_img)
        axes[0].set_title("Image query", fontsize=12, pad=10)
        
        # Show result (caption only)
        img_path, similarity, caption = results[0]  # Use only the first result
        axes[1].set_facecolor('white')
        wrapped_text = textwrap.fill(caption, width=40)
        
        # Display text without a text box
        axes[1].text(0.5, 0.5, wrapped_text, 
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=axes[1].transAxes,
                     wrap=True,
                     fontsize=10)
        # axes[1].set_title("Recall@1", fontsize=12, pad=10)
            
    else:
        # For text query
        axes[0].set_facecolor('white')
        wrapped_text = textwrap.fill(query, width=40)
        
        # Display text without a text box
        axes[0].text(0.5, 0.5, wrapped_text, 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes[0].transAxes,
                    wrap=True,
                    fontsize=10)
        axes[0].set_title("Text query", fontsize=12, pad=10)
        
        # Show result (image only)
        img_path, similarity, caption = results[0]  # Use only the first result
        img = Image.open(img_path)
        axes[1].imshow(img)
        # axes[1].set_title("Recall@1", fontsize=12, pad=10)
    
    # Set common style for all subplots
    for ax in axes:
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    
    plt.savefig(output_path, 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.2)
    plt.close()
    
def main(args):
    fabric = L.Fabric(
        accelerator="cuda", 
        devices=1,
        precision="bf16-mixed"
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    # Model configuration
    if args.model == 'L-336':
        model_name = 'openai/clip-vit-large-patch14-336'
    elif args.model == 'L':
        model_name = 'openai/clip-vit-large-patch14'
    elif args.model == 'B':
        model_name = 'openai/clip-vit-base-patch32'
    elif args.model == 'G':
        model_name = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
    
    # Load model and processor
    processor = transformers.CLIPProcessor.from_pretrained(model_name)
    model = transformers.CLIPModel.from_pretrained(model_name).bfloat16()
    
    # First extend position embeddings
    longclip_pos_embeddings(model, args.new_max_token)
    
    # Then load weights
    if args.ckpt:
        state_dict = torch.load(args.ckpt, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(fabric.device)
    model.eval()

    # Prepare gallery dataset
    dataset = JsonGalleryDataset(args.gallery_json, processor, args.new_max_token)
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn
    )

    # Check if query is an image or text
    is_image_query = args.query.endswith(('.jpg', '.jpeg', '.png'))
    
    # Extract gallery features
    gallery_features = []
    gallery_items = []
    
    with torch.no_grad():
        for batch in dataloader:
            if is_image_query:
                # For image query, extract text features
                input_ids = batch['input_ids'].to(fabric.device)
                attention_mask = batch['attention_mask'].to(fabric.device)
                features = model.get_text_features(input_ids, attention_mask)
            else:
                # For text query, extract image features
                pixel_values = batch['pixel_values'].to(fabric.device)
                features = model.get_image_features(pixel_values)
            
            gallery_features.append(features)
            
            for path, caption in zip(batch['image_path'], batch['caption']):
                gallery_items.append({
                    'image_path': path,
                    'caption': caption
                })
    
    gallery_features = torch.cat(gallery_features, dim=0)

    # Extract query features
    query_features = process_query(model, processor, args.query, fabric.device, is_image_query, args.new_max_token)

    # Calculate similarities and get only the top result
    similarities = compute_similarities(model, query_features, gallery_features)
    top_k = 1  # Get only one result
    top_k_similarities, top_k_indices = torch.topk(similarities[0].float(), k=top_k)
    
    # Organize results
    results = []
    top_k_similarities_np = top_k_similarities.cpu().numpy()
    top_k_indices_np = top_k_indices.cpu().numpy()
    
    for sim, idx in zip(top_k_similarities_np, top_k_indices_np):
        item = gallery_items[idx]
        results.append((
            item['image_path'],
            sim,
            item['caption']
        ))

    # Visualize results
    visualize_results(
        args.query,
        results,
        args.output_path,
        is_image_query
    )

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, 
                       help="Path to query image or text query")
    parser.add_argument("--gallery_json", type=str, required=True,
                       help="Path to JSON file containing gallery information")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save visualization")
    parser.add_argument("--model", type=str, default='L',
                       choices=['L-336', 'L', 'B', 'G'])
    parser.add_argument("--ckpt", type=str, default='',
                       help="Path to custom checkpoint")
    parser.add_argument("--new_max_token", type=int, default=248,
                       help="Maximum number of text tokens")
    
    args = parser.parse_args()
    main(args)