import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import argparse
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import lightning as L
import transformers
import torch.nn.functional as F
import shutil
import time
import numpy as np
from utils.func import *
from utils.transforms import *
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
import shutil
import math
import random
import wandb

torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

def clip_loss(sim):
    gt = torch.arange(len(sim), dtype=torch.long, device=sim.device)
    return (torch.nn.CrossEntropyLoss()(sim, gt) + torch.nn.CrossEntropyLoss()(sim.t(), gt)) / 2.0

def get_patch_tokens_from_bbox(patch_tokens, bbox, b, org_size, image_size=224, patch_size=16):
    # Get original dimensions
    org_width = bbox['width'][b].item()
    org_height = bbox['height'][b].item()
    
    # Scale coordinates to image_size
    x1 = int(round(bbox['x1'][b].item() * image_size / org_width))
    y1 = int(round(bbox['y1'][b].item() * image_size / org_height))
    x2 = int(round(bbox['x2'][b].item() * image_size / org_width))
    y2 = int(round(bbox['y2'][b].item() * image_size / org_height))
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, image_size-1))
    y1 = max(0, min(y1, image_size-1))
    x2 = max(0, min(x2, image_size))
    y2 = max(0, min(y2, image_size))
    
    # Convert to patch indices (include any patch that the bbox touches)
    patch_x1 = x1 // patch_size
    patch_y1 = y1 // patch_size
    patch_x2 = (x2 + patch_size - 1) // patch_size
    patch_y2 = (y2 + patch_size - 1) // patch_size
    
    # Get indices of patches
    num_patches = (image_size // patch_size)
    print(num_patches)
    indices = []
    for i in range(patch_y1, patch_y2):
        for j in range(patch_x1, patch_x2):
            indices.append(i * num_patches + j + 1)
    
    # Extract and pool relevant patch tokens
    relevant_tokens = patch_tokens[:, indices, :]
    pooled_tokens = torch.mean(relevant_tokens, dim=1)
    
    return pooled_tokens

def get_text_tokens_from_segment(text_tokens, org_text, seg_text, processor):
    """
    Args:
        text_tokens: (B, L, D) tensor of text tokens - tokens of the entire original text
        org_text: original text string
        seg_text: segment text string
        processor: CLIP processor
    Returns:
        pooled_tokens: (B, D) tensor of pooled text tokens from the relevant segment
    """
    # Preprocess text
    org_text = ' '.join(org_text.split()).strip()
    seg_text = ' '.join(seg_text.split()).strip()

    # Split org_text into sentences
    sentences = org_text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]

    # Find location of seg_text
    seg_pos = org_text.find(seg_text)
    current_pos = 0
    sent_idx = -1

    # Find position by sentence
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if sent == seg_text:
            seg_pos = current_pos
            sent_idx = i
            break
        current_pos += len(sent) + 2

    assert seg_pos != -1, f"Segment text not found in original text"

    # Tokenize segment text
    seg_tokens = processor(text=seg_text,
                         return_tensors="pt",
                         padding=False,
                         truncation=False)
    seg_token_length = len(seg_tokens.input_ids[0]) - 2  # Exclude CLS, EOS tokens

    if sent_idx != -1:
        # Calculate token index based on sentence position
        text_before = '. '.join(sentences[:sent_idx]) + ('. ' if sent_idx > 0 else '')
        tokens_before = processor(text=text_before,
                                return_tensors="pt",
                                padding=False,
                                truncation=False)
        start_idx = len(tokens_before.input_ids[0])
    else:
        # Calculate token index based on string position
        text_before = org_text[:seg_pos]
        tokens_before = processor(text=text_before,
                                return_tensors="pt",
                                padding=False,
                                truncation=False)
        start_idx = len(tokens_before.input_ids[0])

    # Adjust range considering maximum token length
    max_length = text_tokens.shape[1]  # 248
    if start_idx >= max_length:
        # If segment is beyond the maximum length,
        # extract tokens from the end with enough space for the segment
        end_idx = max_length - 1
        start_idx = max(1, end_idx - seg_token_length)  # Start after CLS token
    else:
        # Normal case within range
        end_idx = min(start_idx + seg_token_length, max_length - 1)

    # Extract tokens
    relevant_tokens = text_tokens[:, start_idx:end_idx, :]

    # Handle case with no tokens extracted
    if relevant_tokens.shape[1] == 0:
        # Fallback: use tokens from the beginning
        relevant_tokens = text_tokens[:, 1:min(1 + seg_token_length, max_length), :]

    # Pool tokens
    pooled_tokens = torch.mean(relevant_tokens, dim=1)

    return pooled_tokens

class DLoader(Dataset):
    def __init__(self, data_list, processor, new_max_token):
        self.data_list = data_list
        self.processor = processor
        self.new_max_token = new_max_token

    def __len__(self):
        return len(self.data_list)
    
    def _load_image(self, name):
        img = Image.open(name).convert("RGB")
        return img, img.size  # Return original image size as well
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        item = self.data_list[idx]
        org_image = self._load_image(item["original_filename"])[0]  # Size info not needed
        org_caption = item["original_caption"]
        
        # Always select the segment with the highest similarity score
        segment = max(item["segment"], key=lambda x: x["similarity_score"])
        
        seg_image = self._load_image(segment["filename"])[0]
        seg_caption = segment["caption"]
        bbox = segment["bbox_coordinates"]

        org_data = self.processor(images=org_image, text=org_caption, return_tensors="pt", 
                                truncation=True, padding="max_length", max_length=self.new_max_token)
        seg_data = self.processor(images=seg_image, text=seg_caption, return_tensors="pt",
                                truncation=True, padding="max_length", max_length=self.new_max_token)

        return (org_data.pixel_values[0], org_data.input_ids[0], 
                seg_data.pixel_values[0], seg_data.input_ids[0],
                bbox, org_caption, seg_caption)

def main(args):
    wandb.init(project="CLIP_Training", config=args)   
    
    fabric = L.Fabric(
        accelerator="cuda", 
        devices=args.world_size,
        strategy="ddp",
        precision="bf16"
    )

    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    with open(args.dataset) as f:
        train_list = json.load(f)
    
    with fabric.device:
        processor = transformers.AutoProcessor.from_pretrained(args.model)
        model = transformers.CLIPModel.from_pretrained(args.model)
        longclip_pos_embeddings(model, args.new_max_token)
        print_trainable_parameters(fabric, model)

    dataset_train = DLoader(train_list, processor, args.new_max_token)
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=True,
    )

    train_loader = fabric.setup_dataloaders(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
    
    train(fabric, model, optimizer, train_loader, processor)
    
def train(fabric: L.Fabric, model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_loader, processor) -> None:
    iter = 0
    total_iter = len(train_loader) * args.epochs
    
    # Define MSE Loss
    mse_loss = torch.nn.MSELoss()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_loss_org = 0.0
        epoch_loss_seg = 0.0
        epoch_loss_patch = 0.0
        epoch_loss_text = 0.0
        
        for i, samples in enumerate(train_loader):
            # Cosine LR
            lr = (args.init_lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * iter / total_iter)) + args.min_lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            org_image, org_text, seg_image, seg_text, bbox, org_caption, seg_caption = samples
            
            # Get all embeddings including patch tokens and sequence tokens
            outputs = model(pixel_values=torch.cat((org_image, seg_image), dim=0),
                        input_ids=torch.cat((org_text, seg_text), dim=0),
                        output_hidden_states=True)
            # print(model.text_model.embeddings.position_embedding.weight.requires_grad)
            # Get patch tokens and text tokens
            vision_outputs = model.vision_model(torch.cat((org_image, seg_image), dim=0), output_hidden_states=True)
            text_outputs = model.text_model(torch.cat((org_text, seg_text), dim=0), output_hidden_states=True)

            # Split embeddings for org and seg
            batch_size = org_image.shape[0]
            org_image_embeds, seg_image_embeds = outputs.image_embeds[:batch_size], outputs.image_embeds[batch_size:]
            org_text_embeds, seg_text_embeds = outputs.text_embeds[:batch_size], outputs.text_embeds[batch_size:]

            # Get patch tokens and text tokens from the last hidden states
            org_patch_tokens = vision_outputs.hidden_states[-1][:batch_size]  # (B, N, D)
            org_text_tokens = text_outputs.hidden_states[-1][:batch_size]     # (B, L, D)
            
            # Original CLIP loss
            eps = 1e-8
            x_i = batch_align(fabric, F.normalize(outputs.image_embeds + eps))
            x_t = batch_align(fabric, F.normalize(outputs.text_embeds + eps))
            x_i_org, x_i_seg = x_i.chunk(2)
            x_t_org, x_t_seg = x_t.chunk(2)
            
            # Compute original losses
            sim_org = model.logit_scale.exp() * x_i_org @ x_t_org.t()
            loss_org = clip_loss(sim_org)
            sim_seg = model.logit_scale.exp() * x_i_seg @ x_t_seg.t()
            loss_seg = clip_loss(sim_seg)
            
            # Compute patch-level alignment loss
            patch_pooled = []
            for b in range(batch_size):
                pooled = get_patch_tokens_from_bbox(org_patch_tokens[b:b+1], 
                                                bbox, 
                                                b,
                                                None,  
                                                image_size=args.image_size, 
                                                patch_size=16)
                patch_pooled.append(pooled)

            patch_pooled = torch.cat(patch_pooled, dim=0)
            patch_pooled = model.vision_model.post_layernorm(patch_pooled)
            patch_pooled = model.visual_projection(patch_pooled)
            patch_pooled = F.normalize(patch_pooled + eps, dim=-1)
            seg_image_embeds = F.normalize(seg_image_embeds + eps, dim=-1)
            
            # Compute patch alignment loss with cosine similarity directly
            sim_patch = patch_pooled @ seg_image_embeds.t()  # Remove logit_scale
            patch_diag = torch.diag(sim_patch)
            loss_patch = mse_loss(patch_diag, torch.ones_like(patch_diag))
            
            # Compute text-level alignment loss
            text_pooled = []
            for b in range(batch_size):
                #print(f"\nBatch {b} Text Sequences:")
                
                # All token IDs for org_text
                org_tokens = processor(text=org_caption[b], 
                                    return_tensors="pt", 
                                    padding=False, 
                                    truncation=False)
                org_token_ids = org_tokens.input_ids[0]
                
                # All token IDs for seg_text
                seg_tokens = processor(text=seg_caption[b], 
                                    return_tensors="pt", 
                                    padding=False, 
                                    truncation=False)
                seg_token_ids = seg_tokens.input_ids[0]
                
                # Decode token IDs to text
                org_tokens_text = processor.tokenizer.convert_ids_to_tokens(org_token_ids)
                seg_tokens_text = processor.tokenizer.convert_ids_to_tokens(seg_token_ids)
               
                start_idx = len(processor(text=org_caption[b][:org_caption[b].find(seg_caption[b])], 
                                        return_tensors="pt", 
                                        padding=False, 
                                        truncation=False).input_ids[0])
                
                end_idx = start_idx + len(seg_tokens.input_ids[0]) - 2 
                
                pooled = get_text_tokens_from_segment(org_text_tokens[b:b+1], 
                                                    org_caption[b], 
                                                    seg_caption[b],
                                                    processor)
                text_pooled.append(pooled)
            text_pooled = torch.cat(text_pooled, dim=0)
            
            text_pooled = model.text_model.final_layer_norm(text_pooled)
            text_pooled = model.text_projection(text_pooled)
            text_pooled = F.normalize(text_pooled + eps, dim=-1)

            seg_text_embeds = F.normalize(seg_text_embeds + eps, dim=-1)
            
            # Compute text alignment loss with cosine similarity directly
            sim_text = text_pooled @ seg_text_embeds.t() 
            text_diag = torch.diag(sim_text)
            loss_text = mse_loss(text_diag, torch.ones_like(text_diag))
            
            # Total loss
            loss = loss_org + 0.5 * loss_seg + loss_patch + loss_text
            
            epoch_loss += loss.item()
            epoch_loss_org += loss_org.item()
            epoch_loss_seg += loss_seg.item()
            epoch_loss_patch += loss_patch.item()
            epoch_loss_text += loss_text.item()
            
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            if fabric.global_rank == 0:
                wandb.log({
                    "iter": iter,
                    "lr": lr,
                    "loss": loss.item(),
                    "loss_org": loss_org.item(),
                    "loss_seg": loss_seg.item(),
                    "loss_patch": loss_patch.item(),
                    "loss_text": loss_text.item(),
                    "epoch": epoch,
                    "progress": (iter / total_iter) * 100,
                    "batch_size": args.batch_size,
                    "logit_scale": model.logit_scale.exp().item(),
                    "patch_similarity": patch_diag.mean().item(),  
                    "text_similarity": text_diag.mean().item(),    
                })
            
            fabric.print(f"epoch {epoch} iter {iter} ({(iter/total_iter)*100:.4f}%) lr {lr:.6f} "
                        f"loss {loss.item():.4f} (org: {loss_org.item():.4f}, seg: {loss_seg.item():.4f}, "
                        f"patch: {loss_patch.item():.4f}, text: {loss_text.item():.4f} "
                        f"patch_sim: {patch_diag.mean().item():.4f}, text_sim: {text_diag.mean().item():.4f})")
            iter += 1
            
        # Calculate and log epoch averages
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_loss_org = epoch_loss_org / len(train_loader)
        avg_epoch_loss_seg = epoch_loss_seg / len(train_loader)
        avg_epoch_loss_patch = epoch_loss_patch / len(train_loader)
        avg_epoch_loss_text = epoch_loss_text / len(train_loader)
        
        if fabric.global_rank == 0:
            wandb.log({
                "epoch": epoch,
                "avg_epoch_loss": avg_epoch_loss,
                "avg_epoch_loss_org": avg_epoch_loss_org,
                "avg_epoch_loss_seg": avg_epoch_loss_seg,
                "avg_epoch_loss_patch": avg_epoch_loss_patch,
                "avg_epoch_loss_text": avg_epoch_loss_text,
            })
            
        # Save model weights
        save_path = os.path.join(args.output_dir, 
                               f"GOAL_12_{os.path.splitext(os.path.basename(args.model))[0]}_"
                               f"{os.path.splitext(os.path.basename(args.dataset))[0]}_{epoch+1}_{args.image_size}.pth")
        
        fabric.barrier()
        if fabric.global_rank == 0:
            model_state_dict = model.state_dict()
            cpu_state_dict = {k: v.cpu() for k, v in model_state_dict.items()}
            torch.save(cpu_state_dict, save_path)
            fabric.print(f"Model saved to {save_path}")
        fabric.barrier()
        

def get_args_parser():
    parser = argparse.ArgumentParser('CLIP Training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--new_max_token', default=248, type=int)
    parser.add_argument('--dataset', default='datasets/DCI_segment_sim_bbox_del_org.json', type=str)
    parser.add_argument('--model', default='openai/clip-vit-base-patch16', type=str)
    parser.add_argument('--weight_decay', type=float, default=0.05) 
    parser.add_argument('--init_lr', type=float, default=5e-6, metavar='LR') 
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR')
    parser.add_argument('--output_dir', default='finetune_out_DCI/goal_bbox_local_token_align_batch16_only_max_pair_base16_patch16',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_interval', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--wandb_project', type=str, default='CLIP_Training', help='wandb project name')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')

    return parser

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)