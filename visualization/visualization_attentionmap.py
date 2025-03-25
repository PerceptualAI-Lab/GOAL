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
import matplotlib.pyplot as plt
import cv2

# Hyperparameters
micro_batch_size = 1
devices = 1
num_workers = 1

class ImageAttentionLoader(Dataset):
    def __init__(self, image_path, processor):
        self.image_path = image_path
        self.processor = processor

    def __len__(self):
        return 1
    
    def _load_image(self):
        return Image.open(self.image_path).convert("RGB")
    
    def __getitem__(self, idx):
        image = self._load_image()
        self.original_size = image.size
        # Process image without return_tensors
        data = self.processor(images=image)
        # Convert to tensor manually
        pixel_values = torch.tensor(data['pixel_values'])
        # Ensure correct shape [C, H, W]
        if len(pixel_values.shape) > 3:
            pixel_values = pixel_values.squeeze()
        return pixel_values, np.array(image)

def get_attention_maps(model, image_input):
    print("Input shape before processing:", image_input.shape)
    
    # Ensure correct shape and device
    if len(image_input.shape) == 3:
        image_input = image_input.unsqueeze(0)  # Add batch dimension
    
    image_input = image_input.to(model.device)
    print("Input shape after processing:", image_input.shape)
    
    outputs = model.vision_model(pixel_values=image_input, output_attentions=True)
    
    # Get attention maps from the last layer
    attention_maps = outputs.attentions[-1]
    print("Attention maps shape:", attention_maps.shape)
    
    # Resize attention maps (average across heads)
    attention_maps = attention_maps.mean(dim=1)
    
    # Use only patch attention excluding CLS token
    attention_maps = attention_maps[:, 1:, 1:]
    
    # Move to CPU and convert to float32
    attention_maps = attention_maps.squeeze().cpu().float()
    print("Final attention maps shape:", attention_maps.shape)
    
    return attention_maps

def visualize_attention(image, attention_maps, output_path, model_name):
    # Calculate image patch size
    if 'patch14' in model_name:
        patch_size = 14
    elif 'patch32' in model_name:
        patch_size = 32
    else:
        patch_size = 16
    
    # Resize attention maps
    h = int(np.sqrt(attention_maps.shape[0]))
    attention_maps = attention_maps.reshape(h, h)
    
    # Upsample attention maps to original image size
    attention_maps = F.interpolate(
        attention_maps.unsqueeze(0).unsqueeze(0),
        size=image.shape[:2],
        mode='bicubic'
    ).squeeze().numpy()
    
    # Normalize
    attention_maps = (attention_maps - attention_maps.min()) / (attention_maps.max() - attention_maps.min())
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Attention map
    plt.subplot(1, 3, 2)
    plt.imshow(attention_maps, cmap='jet')
    plt.title('Self-Attention Map')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(attention_maps, cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    fabric = L.Fabric(
        accelerator="cuda",
        devices=devices,
        precision="bf16-mixed"
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if args.model=='L-336':
        args.model = 'openai/clip-vit-large-patch14-336'
    elif args.model=='L':
        args.model = 'openai/clip-vit-large-patch14'
    elif args.model=='B':
        args.model = 'openai/clip-vit-base-patch32'
    elif args.model=='G':
        args.model = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
        
    with fabric.device:
        processor = transformers.AutoProcessor.from_pretrained(args.model)
        model = transformers.AutoModel.from_pretrained(args.model, output_attentions=True).bfloat16()
        longclip_pos_embeddings(model, args.new_max_token)
        if args.ckpt:
            model.load_state_dict(torch.load(args.ckpt), strict=False)

    # Create data loader
    dataset = ImageAttentionLoader(args.image_path, processor)
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=False)

    # Set model to evaluation mode
    model.eval()
    model = model.to(fabric.device)
    
    # Generate and visualize attention maps
    with torch.no_grad():
        for batch in dataloader:
            image_input, original_image = batch
            print("Original input shape:", image_input.shape)
            attention_maps = get_attention_maps(model, image_input)
            
            # Calculate average attention score across all patches
            avg_attention = attention_maps.mean(dim=0)
            
            visualize_attention(
                original_image[0],
                avg_attention,
                args.output_path,
                args.model
            )
            break

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save visualization")
    parser.add_argument('--new_max_token', default=248, type=int)
    parser.add_argument("--model", type=str, default='L')
    parser.add_argument("--ckpt", type=str, default='')
    args = parser.parse_args()

    main(args)