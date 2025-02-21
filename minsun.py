import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.ops as ops
from io import BytesIO

CATEGORIES = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve dress",
    "vest", "sling", "short sleeve dress", "long sleeve dress", "vest dress", "sling dress"
]

def load_model(model_path, device):
    """DF2MatchRCNN 모델 로드 및 Feature Extractor 설정"""
    # weights_only=True로 보안 경고 완화 (모델 파일이 신뢰할 수 있다고 가정)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model_state_dict = checkpoint["model_state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    
    model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=14)
    model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    model.eval()
    return model

def extract_roi_features(image_tensor, model, device, detected_boxes):
    """Faster R-CNN의 feature map에서 ROI Align을 이용하여 특징 벡터 추출"""
    with torch.no_grad():
        features = model.backbone(image_tensor.to(device))
    # 가장 해상도가 높은 feature map 선택 (보통 '0' 키)
    feature_map = features["0"]  # shape: (1, C, H_feat, W_feat)
    _, _, H_feat, _ = feature_map.shape
    _, _, H_img, _ = image_tensor.shape
    # 이미지와 feature map의 높이 비율을 spatial_scale로 사용
    spatial_scale = H_feat / H_img

    # ROI Align을 위해 detected_boxes를 텐서로 변환 (원본 이미지 좌표)
    boxes = torch.tensor(detected_boxes, dtype=torch.float32, device=device)
    aligned_features = ops.roi_align(
        feature_map,
        [boxes],             # RoIs는 리스트 형태로 전달
        output_size=(7, 7),
        spatial_scale=spatial_scale,
        sampling_ratio=2
    )  # 결과: (N, C, 7, 7)
    pooled_features = aligned_features.mean(dim=[2, 3])  # (N, C)
    return pooled_features.cpu().numpy()

def extract_clothes(image, model, device, score_threshold=0.5):
    """
    이미지에서 검출된 객체(의류)의 ROI 특징 벡터들을 추출한 후,
    각 ROI 벡터의 평균을 계산해 하나의 대표 벡터(query vector)로 반환합니다.
    """
    transform = T.Compose([T.ToTensor()])
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
    
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    
    # 점수가 높은 ROI만 선택
    selected_boxes = [box for box, score in zip(boxes, scores) if score >= score_threshold]
    if len(selected_boxes) == 0:
        # 검출된 ROI가 없으면 0 벡터 반환
        return np.zeros((1, 256), dtype=np.float32)
    
    roi_features = extract_roi_features(input_tensor, model, device, selected_boxes)
    # 여러 ROI 벡터들의 평균을 계산하여 하나의 query vector로 만듭니다.
    query_vector = roi_features.mean(axis=0, keepdims=True)  # shape: (1, 256)
    return query_vector

def detect_objects(image, model, device, score_threshold=0.5):
    """
    이미지에서 객체(의류)를 검출하여 가장 높은 점수의 bounding box, score, label을 반환합니다.
    """
    transform = T.Compose([T.ToTensor()])
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
    
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    
    # 점수가 높은 객체만 필터링
    detections = [(box, score, label) for box, score, label in zip(boxes, scores, labels) if score >= score_threshold]
    if len(detections) == 0:
        return None, None, None
    # 최고 점수의 detection 선택
    detections.sort(key=lambda x: x[1], reverse=True)
    best_box, best_score, best_label = detections[0]
    return best_box, best_score, best_label

def extract_color(image, box):
    """
    주어진 이미지(PIL.Image)와 bounding box로부터 평균 RGB 색상 벡터를 추출합니다.
    """
    x1, y1, x2, y2 = map(int, box)
    cropped = image.crop((x1, y1, x2, y2))
    np_img = np.array(cropped)
    avg_color = np_img.mean(axis=(0,1))  # RGB 평균
    return avg_color

def minsun_model(image_file) -> tuple:
    """
    image_file: UploadFile (FastAPI UploadFile) 또는 파일과 유사한 객체
    Returns:
      feature_vector: ROI들의 평균 특징 벡터 (numpy array, shape: (1, 256))
      bounding_box: 가장 높은 점수의 bounding box (array: [x1, y1, x2, y2])
      color_vector: 해당 bounding box 영역의 평균 RGB 색상 (array: [R, G, B])
      output_category: 해당 객체의 카테고리 (string)
    """
    # 이미지 파일을 읽어 PIL 이미지로 변환
    image_bytes = image_file.file.read()
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "df2matchrcnn"  # 모델 파일 경로 (필요에 따라 수정)
    model = load_model(model_path, device)
    
    # 객체 검출: 가장 높은 점수의 detection
    best_box, best_score, best_label = detect_objects(pil_image, model, device, score_threshold=0.5)
    if best_box is None:
        # 검출된 객체가 없으면 모두 None 반환
        return None, None, None, None
    
    # 특징 벡터 추출: 전체 ROI의 평균 벡터 반환
    feature_vector = extract_clothes(pil_image, model, device, score_threshold=0.5)
    bounding_box = best_box
    # 색상 벡터 추출: best_box 영역의 평균 색상
    color_vector = extract_color(pil_image, best_box)
    # 카테고리: 라벨(1부터 시작)을 CATEGORIES와 매핑 (없으면 "unknown")
    output_category = CATEGORIES[int(best_label) - 1] if (int(best_label) - 1) < len(CATEGORIES) else "unknown"
    
    return feature_vector, bounding_box, color_vector, output_category
