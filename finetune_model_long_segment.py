import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
import wandb

class DLoader(Dataset):
    def __init__(self, data_list, processor):
        self.data_list = data_list
        print(f"Filtered data list length: {len(self.data_list)}")
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
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # GPU 1번 인덱스만 사용

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

    train_list = args.dataset

    with open(train_list) as f:
        train_list = json.load(f)
    
    with fabric.device:
        
        processor = transformers.AutoProcessor.from_pretrained(args.model)
        model = transformers.CLIPModel.from_pretrained(args.model).bfloat16()
        longclip_pos_embeddings(model, args.new_max_token)
        print_trainable_parameters(fabric, model)

    dataset_train = DLoader(train_list, processor)
    
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
    
    train(fabric, model, optimizer, train_loader)

def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
) -> None:
    
    iter = 0
    total_iter = len(train_loader) * args.epochs

    start_time = time.time()
    
    # Initialize wandb
    if fabric.global_rank == 0:
        wandb.init(project="clip-training", config=args)
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for i, samples in enumerate(train_loader):

            # Cosine LR
            lr = (args.init_lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * iter / total_iter)) + args.min_lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            image, text = samples

            x = model(pixel_values=image, input_ids=text)
  
            x_i = batch_align(fabric, F.normalize(x.image_embeds))
            x_t = batch_align(fabric, F.normalize(x.text_embeds))
        
            sim = model.logit_scale.exp()*x_i@x_t.t()
            loss = clip_loss(sim)

            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            fabric.print(f"epoch {epoch} iter {iter} ({(iter/total_iter)*100:.4f}%) lr {lr:.6f} loss {loss.item():.4f}")
            
            # Log metrics to wandb
            if fabric.global_rank == 0:
                wandb.log({
                    "iter": iter,
                    "lr": lr,
                    "loss": loss.item(),
                    "epoch": epoch,
                    "progress": (iter / total_iter) * 100,
                    "batch_size": args.batch_size,
                    "logit_scale": model.logit_scale.exp().item(),
                })
            
            iter += 1

        # 에포크 종료 시 평균 손실 계산 및 기록
        avg_epoch_loss = epoch_loss / len(train_loader)
        if fabric.global_rank == 0:
            wandb.log({
                "epoch": epoch,
                "avg_epoch_loss": avg_epoch_loss,
            })
        
        # 에포크가 끝날 때마다 가중치 저장
        save_path = os.path.join(args.output_dir, f"Only_segment_train_{os.path.splitext(os.path.basename(args.model))[0]}_{os.path.splitext(os.path.basename(args.dataset))[0]}_{epoch+1}_{args.image_size}.pth")
        
        # 모든 프로세스가 동기화되도록 합니다
        fabric.barrier()
        
        # 랭크 0에서만 저장합니다
        if fabric.global_rank == 0:
            # 모델의 state_dict를 얻습니다
            model_state_dict = model.state_dict()
            
            # state_dict를 CPU로 이동시키고 저장합니다
            cpu_state_dict = {k: v.cpu() for k, v in model_state_dict.items()}
            
            torch.save(cpu_state_dict, save_path)
            fabric.print(f"Model saved to {save_path}")
        
        # 다시 한 번 모든 프로세스가 동기화되도록 합니다
        fabric.barrier()

    end_time = time.time()
    total_time = end_time - start_time
    fabric.print(f'Total training time: {total_time} seconds')
        

def get_args_parser():
    parser = argparse.ArgumentParser('CLIP Training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--new_max_token', default=248, type=int)
    parser.add_argument('--dataset', default='datasets/DCI_segment_only_sim_max_del_org.json', type=str)
    parser.add_argument('--model', default='openai/clip-vit-base-patch16', type=str)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--init_lr', type=float, default=5e-6, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR')
    parser.add_argument('--output_dir', default='finetune_out_DCI/only_segment_sim_max_long_batch16_base16',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_interval', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
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