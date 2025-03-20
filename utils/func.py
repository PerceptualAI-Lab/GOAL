import torch
import cv2
from PIL import Image
import numpy as np

def interpolate_pos_embeddings(model, new_image_size):
    vision_model = model.vision_model
    patch_size = vision_model.config.patch_size
    num_patches = (new_image_size // patch_size) ** 2 + 1
    # Extract and interpolate positional embeddings
    pos_embeddings = vision_model.embeddings.position_embedding.weight
    pos_embeddings = pos_embeddings.unsqueeze(0).permute(0, 2, 1)  # Convert to 1xCxN format
    pos_embeddings = torch.nn.functional.interpolate(
        pos_embeddings, size=(num_patches), mode='nearest'
    ).squeeze(0).permute(1, 0)  # Convert back to NxC format
    pos_embeddings = pos_embeddings.contiguous()  # Ensure contiguous
    vision_model.embeddings.position_embedding.weight = torch.nn.Parameter(pos_embeddings)
    # Set position_ids
    if hasattr(vision_model.embeddings, 'position_ids'):
        vision_model.embeddings.position_ids = torch.arange(0, num_patches).unsqueeze(0)
    else:
        vision_model.register_buffer('position_ids', torch.arange(0, num_patches).unsqueeze(0))

def interpolate_text_pos_embeddings(model, new_max_token):
    text_model = model.text_model
    # Extract and interpolate positional embeddings
    pos_embeddings = text_model.embeddings.position_embedding.weight
    pos_embeddings = pos_embeddings.unsqueeze(0).permute(0, 2, 1)  # Convert to 1xCxN format
    
    # Interpolate the position embeddings to the new maximum token length
    pos_embeddings = torch.nn.functional.interpolate(
        pos_embeddings, size=(new_max_token), mode='nearest'
    ).squeeze(0).permute(1, 0)  # Convert back to NxC format
    
    pos_embeddings = pos_embeddings.contiguous()  # Ensure contiguous
    text_model.embeddings.position_embedding.weight = torch.nn.Parameter(pos_embeddings)
    
    # Set position_ids if the model uses them
    if hasattr(text_model.embeddings, 'position_ids'):
        text_model.embeddings.position_ids = torch.arange(0, new_max_token).unsqueeze(0)
    else:
        text_model.register_buffer('position_ids', torch.arange(0, new_max_token).unsqueeze(0))

def longclip_pos_embeddings(model, new_max_token):
    text_model = model.text_model
    # Extract positional embeddings
    pos_embeddings_pre = text_model.embeddings.position_embedding.weight
    length, dim = pos_embeddings_pre.shape
    keep_len = 20
    new_length = 4*length - 3*keep_len
    if new_length < new_max_token:
        raise ValueError("new_max_token is too large")
    pos_embeddings_new = torch.zeros([new_max_token, dim], dtype=pos_embeddings_pre.dtype)
    for i in range(keep_len):
        pos_embeddings_new[i] = pos_embeddings_pre[i]
    for i in range(length-1-keep_len):
        pos_embeddings_new[4*i + keep_len] = pos_embeddings_pre[i + keep_len]
        pos_embeddings_new[4*i + 1 + keep_len] = 3*pos_embeddings_pre[i + keep_len]/4 + 1*pos_embeddings_pre[i+1+keep_len]/4
        pos_embeddings_new[4*i + 2+keep_len] = 2*pos_embeddings_pre[i+keep_len]/4 + 2*pos_embeddings_pre[i+1+keep_len]/4
        pos_embeddings_new[4*i + 3+keep_len] = 1*pos_embeddings_pre[i+keep_len]/4 + 3*pos_embeddings_pre[i+1+keep_len]/4
    pos_embeddings_new[4*length -3*keep_len - 4] = pos_embeddings_pre[length-1] + 0*(pos_embeddings_pre[length-1] - pos_embeddings_pre[length-2])/4
    pos_embeddings_new[4*length -3*keep_len - 3] = pos_embeddings_pre[length-1] + 1*(pos_embeddings_pre[length-1] - pos_embeddings_pre[length-2])/4
    pos_embeddings_new[4*length -3*keep_len - 2] = pos_embeddings_pre[length-1] + 2*(pos_embeddings_pre[length-1] - pos_embeddings_pre[length-2])/4
    pos_embeddings_new[4*length -3*keep_len - 1] = pos_embeddings_pre[length-1] + 3*(pos_embeddings_pre[length-1] - pos_embeddings_pre[length-2])/4
    text_model.embeddings.position_embedding.weight = torch.nn.Parameter(pos_embeddings_new)
    # Set position_ids if the model uses them
    if hasattr(text_model.embeddings, 'position_ids'):
        text_model.embeddings.position_ids = torch.arange(0, new_max_token).unsqueeze(0)
    else:
        text_model.register_buffer('position_ids', torch.arange(0, new_max_token).unsqueeze(0))


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def batch_align(fabric, x):
    x = fabric.all_gather(x, sync_grads=True)
    return x.view(x.shape[0]*x.shape[1], -1)

cls_criterion = torch.nn.CrossEntropyLoss()

def clip_loss(logits):
    gt = torch.arange(len(logits),dtype=torch.long, device=logits.device)
    return (cls_criterion(logits, gt) + cls_criterion(logits.t(), gt))/2.0

def print_trainable_parameters(fabric, model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    fabric.print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    fabric.print('Memory load of model: {} bytes'.format(torch.cuda.memory_allocated()))
