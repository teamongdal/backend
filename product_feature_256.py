import json
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn


def load_detection_model(model_path, device):
    """df2matchrcnn 모델 로드 (DeepFashion2용 Mask R-CNN)
       model_path: 체크포인트 파일 경로 (예: "df2matchrcnn.pth")
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    else:
        model_state_dict = checkpoint

    # DeepFashion2의 경우, 배경을 포함해 14개의 클래스로 설정되어 있음
    model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=14)
    new_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def extract_feature_for_crop_df2matchrcnn(cropped_image, detection_model, device):
    # 전처리: 일반적인 ResNet/df2matchrcnn 전처리와 동일하게 진행
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        # df2matchrcnn의 backbone을 통해 여러 해상도의 feature map을 반환받음
        features = detection_model.backbone(img_tensor)  # features: OrderedDict
        # 보통 가장 낮은 해상도의 feature map 선택 (예: 키 "0")
        feature_map = list(features.values())[0]  # shape: [N, C, H, W]
        # adaptive average pooling을 통해 [N, C, 1, 1]로 축소
        pooled = F.adaptive_avg_pool2d(feature_map, (1, 1))
        # flatten하여 [N, C] 형태로 변환
        vector = pooled.view(pooled.size(0), -1)
        # 필요한 경우 projection_layer를 통해 차원 변환 (여기서는 사용하지 않음)
    return vector.squeeze().cpu().numpy()

def compute_feature_df2matchrcnn(image, detection_model, device):
    feat = extract_feature_for_crop_df2matchrcnn(image, detection_model, device)
    norm = np.linalg.norm(feat)
    if norm == 0:
        norm = 1
    return feat.astype(np.float32) / norm

def find_image(image_path):
    """이미지 파일 경로에서 PIL 이미지 객체 반환"""
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return None

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    # 입력 JSON 파일 경로 (예시)
    # input_json_path = "raw_db_cat&color.json"
    # input_json_path = "myeongpum_similar_cat&color.json"
    input_json_path = "myeongpum_cat&color&clip&pattern.json"
    
    # 출력 npy 파일 경로 (예시)
    # output_npy_path = "raw_db_df2matchrcnn_256.npy"
    # output_npy_path = "myeongpum_similar_df2matchrcnn_256.npy"
    output_npy_path = "myeongpum_df2matchrcnn_256.npy"

    # df2matchrcnn 체크포인트 파일 경로 (실제 파일 경로로 수정하세요)
    model_checkpoint = "df2matchrcnn"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # df2matchrcnn 모델 로드
    detection_model = load_detection_model(model_checkpoint, device)
    
    json_data = load_json(input_json_path)
    all_features = []
    
    for item in json_data:
        image_path = item.get("product_images_1")
        if not image_path:
            continue
        # 확장자가 없으면 .jpg 추가
        if not image_path.lower().endswith(".jpg"):
            image_path += ".jpg"
        
        image = find_image(image_path)
        if image is None:
            continue
        
        print(f"Processing image: {image_path}")
        # 각 cloth 객체의 box 좌표에 대해 feature vector 추출
        for cloth in item.get("clothes", []):
            box = cloth.get("box")
            if not box or len(box) != 4:
                print(f"INVALID IMAGE: {image_path}")
                continue
            # crop 영역 추출
            x1, y1, x2, y2 = map(int, box)
            cropped = image.crop((x1, y1, x2, y2))
            feature_vector = compute_feature_df2matchrcnn(cropped, detection_model, device)
            all_features.append(feature_vector)
    
    # numpy 배열로 변환 후 npy 파일로 저장
    np.save(output_npy_path, np.array(all_features))
    print(f"Extracted features saved to {output_npy_path}")
