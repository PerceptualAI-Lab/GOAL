import requests
from io import BytesIO
from PIL import Image
import numpy as np
from transformers import pipeline, CLIPProcessor, CLIPModel
import cv2
import torch
import json
import os
from tqdm import tqdm

# Load SAM model
generator = pipeline("mask-generation", device=1, points_per_batch=128)

# Load CLIP model
model_name = "openai/clip-vit-large-patch14-336"
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)

# Maintain original background
def keep_original_background(image, mask):
    masked_image = np.array(image)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked_image = masked_image * mask_3d + (1 - mask_3d) * np.array(image)
    return Image.fromarray(np.uint8(masked_image))

# Image resizing function
def resize_image_if_needed(image_path, max_size=(2048, 1536)):
    with Image.open(image_path) as img:
        if img.size[0] > 4000 or img.size[1] > 3000:
            img.thumbnail(max_size, Image.LANCZOS)
            return img.copy()
        return img.copy()

# Image-caption matching using CLIP
def get_clip_similarity(images, texts):
    try:
        inputs = clip_processor(text=texts, images=images, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        return outputs.logits_per_image.cpu().detach().numpy()
    except Exception as e:
        print(f"Error in CLIP processing: {str(e)}")
        return None

# Simple sentence tokenization
def tokenize_caption(text):
    return [s.strip() for s in text.split('.') if s.strip()]

# Set base paths
matching_results_path = "matching_results"
output_base_dir = "segment_with_background_DCI_train_set_max_0.01"
os.makedirs(output_base_dir, exist_ok=True)

# Find folders starting with train_sa
train_folders = [f for f in os.listdir(matching_results_path) if f.startswith('train_sa_')]

# Process each train folder
for train_folder in tqdm(train_folders, desc="Processing folders"):
    folder_path = os.path.join(matching_results_path, train_folder)
    
    # Find JSON files (sa_xxxxxx_result.json)
    json_files = [f for f in os.listdir(folder_path) if f.endswith('_result.json')]
    if not json_files:
        continue
    
    json_path = os.path.join(folder_path, json_files[0])
    
    # Load JSON file
    try:
        with open(json_path, 'r') as f:
            annotation = json.load(f)
    except Exception as e:
        print(f"Error loading JSON for {train_folder}: {str(e)}")
        continue

    image_filename = annotation['original_image']
    image_path = os.path.join(folder_path, image_filename)
    caption = annotation['extra_caption']

    # Set directory to save results
    output_dir = os.path.join(output_base_dir, f"{image_filename}_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Skip already processed images
    if os.path.exists(os.path.join(output_dir, 'matched_dataset_train.json')):
        continue

    # Load and segment the image
    try:
        original_image = Image.open(image_path)
        resized_image = resize_image_if_needed(image_path)
        
        # Run SAM on resized image
        outputs = generator(resized_image, points_per_batch=128)
        
        # Resize masks to original image size
        if original_image.size != resized_image.size:
            for i in range(len(outputs['masks'])):
                mask = Image.fromarray(outputs['masks'][i])
                mask = mask.resize(original_image.size, Image.NEAREST)
                outputs['masks'][i] = np.array(mask)
        
        image = original_image
    except Exception as e:
        print(f"Error processing {image_filename}: {str(e)}")
        continue

    # Filter and create segment images
    min_area_ratio = 0.01
    max_area_ratio = 0.8
    total_area = image.size[0] * image.size[1]
    segmented_images = [image]
    segmented_image_paths = ["original_image.jpg"]

    for i, mask in enumerate(outputs['masks']):
        segment_area = np.sum(mask)
        area_ratio = segment_area / total_area
        if min_area_ratio <= area_ratio <= max_area_ratio:
            masked_image = keep_original_background(image, mask)
            
            y, x = np.where(mask)
            y_min, y_max, x_min, x_max = y.min(), y.max(), x.min(), x.max()
            cropped_image = np.array(masked_image)[y_min:y_max+1, x_min:x_max+1]
            
            segmented_images.append(Image.fromarray(np.uint8(cropped_image)))
            segmented_image_paths.append(f"{image_filename.split('.')[0]}_max_{i+1}.png")

    # Tokenize captions
    tokenized_captions = tokenize_caption(caption)

    # Calculate similarity for all sentence-image pairs
    similarity_matrix = get_clip_similarity(segmented_images, tokenized_captions)

    if similarity_matrix is None:
        print(f"Skipping image {image_filename} due to CLIP processing error")
        continue

    matched_data = []
    saved_images = set()

    # Match each sentence with the image that has the highest similarity
    for j, caption in enumerate(tokenized_captions):
        i = np.argmax(similarity_matrix[:, j])
        similarity_score = float(similarity_matrix[i, j])
        matched_image_path = segmented_image_paths[i]
        
        matched_data.append({
            "caption": caption,
            "matched_image_path": matched_image_path,
            "similarity_score": similarity_score
        })
        
        saved_images.add(i)

    # Save original image
    original_image_path = os.path.join(output_dir, "original_image.jpg")
    image.save(original_image_path)

    # Save matched segment images
    for i in saved_images:
        if i != 0:  # Skip original image
            output_file_path = os.path.join(output_dir, segmented_image_paths[i])
            segmented_images[i].save(output_file_path)

    # Save results
    json_output_path = os.path.join(output_dir, 'matched_dataset_train.json')
    with open(json_output_path, 'w') as f:
        json.dump(matched_data, f, indent=2)

print("All train set images have been processed.")