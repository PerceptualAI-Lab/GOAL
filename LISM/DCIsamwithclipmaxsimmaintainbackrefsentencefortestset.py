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

# SAM 모델 로드
generator = pipeline("mask-generation", device=2, points_per_batch=128)

# CLIP 모델 로드
model_name = "openai/clip-vit-large-patch14-336"
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)

# 원본 배경 유지
def keep_original_background(image, mask):
    masked_image = np.array(image)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked_image = masked_image * mask_3d + (1 - mask_3d) * np.array(image)
    return Image.fromarray(np.uint8(masked_image))

# 이미지 리사이징 함수
def resize_image_if_needed(image_path, max_size=(2048, 1536)):
    with Image.open(image_path) as img:
        if img.size[0] > 4000 or img.size[1] > 3000:
            img.thumbnail(max_size, Image.LANCZOS)
            return img.copy()
        return img.copy()

# CLIP을 사용한 이미지-캡션 매칭
def get_clip_similarity(images, texts):
    try:
        inputs = clip_processor(text=texts, images=images, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        return outputs.logits_per_image.cpu().detach().numpy()
    except Exception as e:
        print(f"Error in CLIP processing: {str(e)}")
        return None

# 간단한 문장 토큰화
def tokenize_caption(text):
    return [s.strip() for s in text.split('.') if s.strip()]

# 기본 경로 설정
matching_results_path = "matching_results"
output_base_dir = "segment_with_background_DCI_test_set_max_0.01"
os.makedirs(output_base_dir, exist_ok=True)

# train_sa로 시작하는 폴더들 찾기
train_folders = [f for f in os.listdir(matching_results_path) if f.startswith('test_sa_')]

# 각 train 폴더에 대해 처리
for train_folder in tqdm(train_folders, desc="Processing folders"):
    folder_path = os.path.join(matching_results_path, train_folder)
    
    # JSON 파일 찾기 (sa_xxxxxx_result.json)
    json_files = [f for f in os.listdir(folder_path) if f.endswith('_result.json')]
    if not json_files:
        continue
    
    json_path = os.path.join(folder_path, json_files[0])
    
    # JSON 파일 로드
    try:
        with open(json_path, 'r') as f:
            annotation = json.load(f)
    except Exception as e:
        print(f"Error loading JSON for {train_folder}: {str(e)}")
        continue

    image_filename = annotation['original_image']
    image_path = os.path.join(folder_path, image_filename)
    caption = annotation['extra_caption']

    # 결과를 저장할 디렉토리 설정
    output_dir = os.path.join(output_base_dir, f"{image_filename}_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미 처리된 이미지는 건너뛰기
    if os.path.exists(os.path.join(output_dir, 'matched_dataset_test.json')):
        continue

    # 이미지 로드 및 세그멘테이션
    try:
        original_image = Image.open(image_path)
        resized_image = resize_image_if_needed(image_path)
        
        # 리사이즈된 이미지로 SAM 실행
        outputs = generator(resized_image, points_per_batch=128)
        
        # 원본 이미지 크기로 마스크 리사이즈
        if original_image.size != resized_image.size:
            for i in range(len(outputs['masks'])):
                mask = Image.fromarray(outputs['masks'][i])
                mask = mask.resize(original_image.size, Image.NEAREST)
                outputs['masks'][i] = np.array(mask)
        
        image = original_image
    except Exception as e:
        print(f"Error processing {image_filename}: {str(e)}")
        continue

    # 필터링 및 세그먼트 이미지 생성
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

    # 캡션 토큰화
    tokenized_captions = tokenize_caption(caption)

    # 모든 문장-이미지 쌍에 대한 유사도 계산
    similarity_matrix = get_clip_similarity(segmented_images, tokenized_captions)

    if similarity_matrix is None:
        print(f"Skipping image {image_filename} due to CLIP processing error")
        continue

    matched_data = []
    saved_images = set()

    # 모든 문장에 대해 가장 높은 유사도를 가진 이미지와 매칭
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

    # 원본 이미지 저장
    original_image_path = os.path.join(output_dir, "original_image.jpg")
    image.save(original_image_path)

    # 매칭된 세그먼트 이미지 저장
    for i in saved_images:
        if i != 0:  # 원본 이미지는 건너뛰기
            output_file_path = os.path.join(output_dir, segmented_image_paths[i])
            segmented_images[i].save(output_file_path)

    # 결과 저장
    json_output_path = os.path.join(output_dir, 'matched_dataset_test.json')
    with open(json_output_path, 'w') as f:
        json.dump(matched_data, f, indent=2)

print("모든 test set 이미지 처리가 완료되었습니다.")