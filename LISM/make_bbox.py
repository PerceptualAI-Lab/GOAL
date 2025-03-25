import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def find_crop_position_cv2(original_img, crop_img):
    """
    OpenCV를 사용하여 크롭된 이미지의 위치를 빠르게 찾습니다
    """
    # OpenCV로 이미지 읽기
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
    # 템플릿 매칭 수행
    result = cv2.matchTemplate(original_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    
    # 좌표 계산
    x, y = max_loc
    h, w = crop_img.shape[:2]
    
    return {
        "x1": int(x),
        "y1": int(y),
        "x2": int(x + w),
        "y2": int(y + h),
        "width": int(w),
        "height": int(h)
    }

def process_folder(folder_path):
    folder_path = Path(folder_path)
    
    # 원본 이미지 로드
    original_img = cv2.imread(str(folder_path / "original_image.jpg"))
    if original_img is None:
        raise Exception(f"Cannot read original image in {folder_path}")
    
    # JSON 파일 로드
    json_path = folder_path / "matched_dataset_train.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 이미 처리된 이미지 추적
    processed_images = {}
    
    # 각 엔트리 처리
    for entry in tqdm(data, desc=f"Processing {folder_path.name}"):
        crop_image_name = entry["matched_image_path"]
        
        # 이미 처리된 이미지면 저장된 좌표 사용
        if crop_image_name in processed_images:
            entry["bbox_coordinates"] = processed_images[crop_image_name]
            continue
            
        try:
            # 크롭된 이미지 로드
            crop_img = cv2.imread(str(folder_path / crop_image_name))
            if crop_img is None:
                print(f"Warning: Cannot read crop image {crop_image_name}")
                continue
                
            # 좌표 찾기
            coords = find_crop_position_cv2(original_img, crop_img)
            
            # 결과 저장
            entry["bbox_coordinates"] = coords
            processed_images[crop_image_name] = coords
            
        except Exception as e:
            print(f"Error processing {crop_image_name}: {str(e)}")
    
    # 업데이트된 JSON 저장
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_all_folders(base_path):
    """
    train_*.jpg_results 패턴의 모든 폴더 처리
    """
    base_path = Path(base_path)
    folders = list(base_path.glob("sa_*.jpg_results"))
    
    print(f"Found {len(folders)} folders to process")
    
    for folder in tqdm(folders, desc="Processing folders"):
        try:
            process_folder(folder)
        except Exception as e:
            print(f"Error processing folder {folder.name}: {str(e)}")

if __name__ == "__main__":
    # 데이터셋 폴더 경로 지정
    dataset_path = "segment_with_background_DCI_train_set_max_0.01"
    process_all_folders(dataset_path)