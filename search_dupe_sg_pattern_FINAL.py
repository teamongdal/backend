import faiss
import numpy as np
import json
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.models import resnet50
from PIL import Image
import cv2
import os
from sklearn.cluster import KMeans
import clip
from sungmin_GDP import *
from fashion_clip.fashion_clip import FashionCLIP
from transformers import CLIPProcessor
from difflib import SequenceMatcher
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# CATEGORIES 수정(협)
CATEGORIES = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
    "vest", "sling", "shorts", "trousers",
    "skirt", "short sleeve dress", "long sleeve dress", "vest dress", 
    "sling dress"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fashion‑CLIP 모델 로드 (Fashion‑CLIP과 CLIPProcessor를 함께 사용)
fclip = FashionCLIP("fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
fclip.processor = processor  # fclip 객체에 processor 저장

# 카테고리별 텍스처, 시즌, 무드 벡터 (faiss_clip.py 참고)
TEXTURE_CATEGORIES = ["rough", "soft", "glossy"]
SEASON_CATEGORIES = ["spring", "summer", "fall", "winter"]
MOOD_CATEGORIES = ["casual", "elegant", "sporty"]
PATTERN_CATEGORIES = ["Solid color with no Pattern", "Stripe Pattern", "Polka Dot Pattern", "Checkered Pattern"]

# 텍스트 임베딩 함수 (정규화 포함)
def encode_texts(texts):
    text_inputs = fclip.processor(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = fclip.model.get_text_features(**text_inputs)
    return text_features / text_features.norm(dim=-1, keepdim=True)

# 각 텍스트 카테고리(속성)의 벡터를 numpy 배열로 반환
def get_text_vectors(categories):
    text_features = encode_texts(categories)
    return text_features.cpu().numpy()

# PIL 이미지에서 Fashion‑CLIP 이미지 임베딩 추출 함수
def encode_image_from_pil(image):
    image_inputs = fclip.processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = fclip.model.get_image_features(image_inputs["pixel_values"])
    return image_features / image_features.norm(dim=-1, keepdim=True)

# 이미지와 텍스트 벡터 간의 cosine similarity를 계산하여 해당 이미지의 속성 벡터를 얻음
def classify_feature_vector(image, text_vectors):
    image_feature = encode_image_from_pil(image)  # (1, D)
    similarity = (image_feature @ torch.tensor(text_vectors, device=device).T)
    return similarity.cpu().numpy().flatten()

# NEW
def is_similar_category(cat1, cat2, threshold=0.7):
    return SequenceMatcher(None, cat1, cat2).ratio() > threshold

# 각 속성(텍스처, 시즌, 무드)의 임베딩 벡터 생성
texture_vectors = get_text_vectors(TEXTURE_CATEGORIES)    # (3, D)
season_vectors = get_text_vectors(SEASON_CATEGORIES)      # (4, D)
mood_vectors = get_text_vectors(MOOD_CATEGORIES)          # (3, D)
pattern_vectors = get_text_vectors(PATTERN_CATEGORIES)    # (4, D)

def normalize_vector(vec):
    vec = np.array(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        norm = 1
    return vec / norm

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)

def euclidean_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    distance = np.linalg.norm(a - b)  # Euclidean distance 
    # Normalize to similarity score (0 to 1)
    similarity = 1 / (1 + distance)  # The smaller the distance, the closer to 1.
    return similarity


def manhattan_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    distance = np.sum(np.abs(a - b))
    return 1 / (1 + distance)

def normalize_color_vectors(color_vectors):
    """
    Given a list of Lab color vectors (each a list of 3 numbers),
    returns a new list where each vector is normalized (L2 norm = 1).
    """
    normalized = []
    for vec in color_vectors:
        arr = np.array(vec, dtype=float)
        norm = np.linalg.norm(arr)
        if norm == 0:
            normalized.append(arr.tolist())
        else:
            normalized.append((arr / norm).tolist())
    return normalized

def compute_emd(color_vector1, color_vector2):
    """
    Computes a similarity score between two color distributions (lists of Lab color vectors)
    by calculating the Earth Mover's Distance (EMD) using the Hungarian algorithm.
    
    Parameters:
      color_vector1: List of [L, a, b] values (e.g., output of detect_color_kmeans_lab)
      color_vector2: List of [L, a, b] values
      
    Returns:
      similarity: A similarity score where higher values indicate more similar color distributions.
                  (Computed as 1 / (1 + total minimal matching distance))
    """
    # Convert lists to numpy arrays of floats
    cv1 = np.array(color_vector1, dtype=float)
    cv2 = np.array(color_vector2, dtype=float)
    
    # Compute pairwise Euclidean distance matrix between Lab vectors
    pairwise_dist = distance.cdist(cv1, cv2, metric='euclidean')
    
    # Solve for the optimal matching using the Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(pairwise_dist)
    
    # Sum up the distances for the optimal assignment
    emd_score = pairwise_dist[row_ind, col_ind].sum()
    
    # Convert distance to a similarity score (higher similarity means lower distance)
    similarity = 1 / (1 + emd_score)
    return similarity

# --- 기본 함수들 ---
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return None

def load_and_crop_product_image(product, box_idx):
    img_path = product.get("product_images_1")
    if not img_path:
        print("상품에 이미지 경로가 없습니다.")
        return None
    if not img_path.lower().endswith(".jpg"):
        img_path += ".jpg"
    # PIL을 이용해 이미지 로드 (RGB 체계)
    img = find_image(img_path)
    if img is None:
        return None
    clothes = product.get("clothes", [])
    if box_idx < len(clothes):
        box = clothes[box_idx].get("box")
        if box and len(box) == 4:
            # crop()는 (left, upper, right, lower) 순서로 진행
            img = img.crop(box)
    return img

### TODO

def extract_feature_for_crop_resnet(cropped_image, feature_extractor, device):
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    img_tensor = transform(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = feature_extractor(img_tensor)
    return feature.squeeze().cpu().numpy()

# 새로운 df2matchrcnn 기반 feature 추출 함수 (256차원)
def extract_feature_for_crop_df2matchrcnn(cropped_image, detection_model, device):
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        # df2matchrcnn의 backbone에서 여러 해상도의 feature map이 출력되는데,
        # 여기서는 가장 낮은 해상도(예: 첫 번째 feature map)를 선택합니다.
        features = detection_model.backbone(img_tensor)  # features: OrderedDict
        feature_map = list(features.values())[0]  # shape: [N, C, H, W]; C는 보통 256일 가능성이 큽니다.
        pooled = F.adaptive_avg_pool2d(feature_map, (1, 1))  # [N, C, 1, 1]
        vector = pooled.view(pooled.size(0), -1)  # [N, C]
    return vector.squeeze().cpu().numpy()


def compute_feature(image, feature_extractor, device):
    feat = extract_feature_for_crop_resnet(image, feature_extractor, device)
    norm = np.linalg.norm(feat)
    if norm == 0:
        norm = 1
    return feat.astype(np.float32) / norm

def compute_feature_df2matchrcnn(image, detection_model, device):
    feat = extract_feature_for_crop_df2matchrcnn(image, detection_model, device)
    norm = np.linalg.norm(feat)
    if norm == 0:
        norm = 1
    return feat.astype(np.float32) / norm

### END 

# JSON 내의 clothes box 인덱스 매핑 (faiss_clip, search_dupe 참고)
def map_index_to_json(idx, products):
    count = 0
    for prod_idx, product in enumerate(products):
        clothes = product.get("clothes", [])
        for box_idx, cloth in enumerate(clothes):
            if count == idx:
                return prod_idx, box_idx
            count += 1
    return None, None

def load_detection_model(model_path, device):
    """df2matchrcnn 모델 로드 (DeepFashion2용 Mask R-CNN)"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    else:
        model_state_dict = checkpoint

    from torchvision.models.detection import maskrcnn_resnet50_fpn
    # DeepFashion2의 경우, 배경을 포함해 14개의 클래스로 설정되어 있음
    model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=14)
    new_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def detect_query_box_df2matchrcnn(query_image, detection_model, direction="middle", score_threshold=0.5):
    # 이미지 전처리
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(query_image).to(device)
    
    with torch.no_grad():
        outputs = detection_model([img_tensor])
    outputs = outputs[0]
    boxes = outputs["boxes"].cpu().numpy()  # shape: (N, 4)
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()  # 클래스 번호
    
    # score threshold 필터링
    valid_indices = np.where(scores > score_threshold)[0]
    if len(valid_indices) == 0:
        width, height = query_image.size
        return [0, 0, width, height], "NoDetection"
    
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    labels = labels[valid_indices]
    
    width, height = query_image.size
    candidates = []
    
    # 방향에 따른 후보 추출
    if direction.lower() == "left":
        # 왼쪽 영역: 이미지의 왼쪽 절반을 대상으로 함
        for i, box in enumerate(boxes):
            center_x = (box[0] + box[2]) / 2
            if center_x < width / 2:
                candidates.append((box, scores[i], labels[i]))
        if candidates:
            # composite score = -x1 + lambda * area
            lambda_val = 0.001  # 면적 가중치 (필요에 따라 조정)
            def composite_score(candidate):
                box, score, label = candidate
                area_val = (box[2] - box[0]) * (box[3] - box[1])
                return -box[0] + lambda_val * area_val
            best_candidate = max(candidates, key=composite_score)
            best_box, best_score, best_label = best_candidate
        else:
            # 후보가 없으면 전체에서 넓이가 가장 큰 것을 선택
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = np.argmax(areas)
            best_box = boxes[idx]
            best_label = labels[idx]
    elif direction.lower() == "right":
        # 오른쪽 영역: 이미지의 오른쪽 절반을 대상으로 함
        for i, box in enumerate(boxes):
            center_x = (box[0] + box[2]) / 2
            if center_x >= width / 2:
                candidates.append((box, scores[i], labels[i]))
        if candidates:
            # 비슷한 방식으로 오른쪽의 경우는 오른쪽에 가까운(즉, box[2]가 큰) 후보를 선택
            lambda_val = 0.001
            def composite_score_right(candidate):
                box, score, label = candidate
                area_val = (box[2] - box[0]) * (box[3] - box[1])
                # 오른쪽일 경우, 오른쪽 경계에 가까울수록 box[2]가 크므로 그 값을 사용
                return box[2] + lambda_val * area_val
            best_candidate = max(candidates, key=composite_score_right)
            best_box, best_score, best_label = best_candidate
        else:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = np.argmax(areas)
            best_box = boxes[idx]
            best_label = labels[idx]
    else:  # "middle"인 경우 기존 로직 사용
        for i, box in enumerate(boxes):
            center_x = (box[0] + box[2]) / 2
            if width / 3 <= center_x <= 2 * width / 3:
                candidates.append((box, scores[i], labels[i]))
        if candidates:
            def area(box):
                return (box[2] - box[0]) * (box[3] - box[1])
            best_candidate = max(candidates, key=lambda x: area(x[0]))
            best_box, best_score, best_label = best_candidate
        else:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = np.argmax(areas)
            best_box = boxes[idx]
            best_label = labels[idx]
    
    best_box = list(map(int, best_box))
    try:
        index = int(best_label) - 1
        if 0 <= index < len(CATEGORIES):
            best_label_str = CATEGORIES[index]
        else:
            best_label_str = str(best_label)
    except Exception as e:
        best_label_str = str(best_label)
    
    # 시각화: query image에 모든 후보들을 초록색, 최고 후보는 파란색으로 표시
    vis_img = cv2.cvtColor(np.array(query_image), cv2.COLOR_RGB2BGR)
    print("후보 이미지 수:", len(candidates))
    for (box, score, label) in candidates:
        b = list(map(int, box))
        cv2.rectangle(vis_img, (b[0], b[1]), (b[2], b[3]), (0,255,0), 2)
        try:
            idx_label = int(label) - 1
            cat = CATEGORIES[idx_label] if 0 <= idx_label < len(CATEGORIES) else str(label)
        except Exception as e:
            cat = str(label)
        cv2.putText(vis_img, f"{cat}:{score:.2f}", (b[0], b[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.rectangle(vis_img, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (255,0,0), 3)
    cv2.putText(vis_img, f"Best {best_label_str}:{max(scores):.2f}", (best_box[0], best_box[1]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.imshow("Query Image Candidates", vis_img)
    cv2.waitKey(0)
    cv2.destroyWindow("Query Image Candidates")
    
    return best_box, best_label_str

def detect_color_kmeans_blur(image, box, k=3):
    x1, y1, x2, y2 = map(int, box)
    image_np = np.array(image)
    cropped_region = image_np[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(cropped_region, (5,5), 0)
    blurred_hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    pixels = blurred_hsv.reshape((-1,3))
    kmeans = KMeans(n_clusters=min(k, len(pixels)), random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return [int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2])]

# --- OpenCV 시각화 함수 ---
def visualize_product_with_boxes(product):
    img_path = product.get("product_images_1")
    if not img_path:
        print("상품에 이미지 경로가 없습니다.")
        return
    if not img_path.lower().endswith(".jpg"):
        img_path += ".jpg"
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print("이미지 로드 실패:", img_path)
        return
    proc_img = orig_img.copy()
    for cloth in product.get("clothes", []):
        box = cloth.get("box")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = map(int, box)
        score = cloth.get("score", 0)
        category = cloth.get("category", "unknown")
        prod_code = product.get("product_code", "NoName")
        cv2.rectangle(proc_img, (x1,y1), (x2,y2), (0,255,0), 2)
        text = f"{prod_code} | {score:.2f} | {category}"
        cv2.putText(proc_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # color_vector = cloth.get("color_vector")
        # if color_vector is not None:
        #     hsv_color = np.uint8([[color_vector]])
        #     bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        #     patch_w, patch_h = 40, 40
        #     patch_x1 = x2 - patch_w
        #     patch_y1 = y1
        #     patch_x2 = x2
        #     patch_y2 = y1 + patch_h
        #     cv2.rectangle(proc_img, (patch_x1, patch_y1), (patch_x2, patch_y2), bgr_color.tolist(), -1)
        #     cv2.putText(proc_img, f"HSV: {color_vector}", (patch_x1, patch_y2+20),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    # combined = np.hstack([orig_img, proc_img])
    window_name = f"Processed: {product.get('product_code','NoName')}"
    cv2.imshow(window_name, proc_img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def visualize_query_image(query_img, box, category, color_vector):
    orig_img = cv2.cvtColor(np.array(query_img), cv2.COLOR_RGB2BGR)
    proc_img = orig_img.copy()
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(proc_img, (x1,y1), (x2,y2), (255,0,0), 2)  # 파란색 박스
    text = f"Query: {category}"
    cv2.putText(proc_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    # if color_vector is not None:
    #     hsv_color = np.uint8([[color_vector]])
    #     bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    #     patch_w, patch_h = 40, 40
    #     patch_x1 = x2 - patch_w
    #     patch_y1 = y1
    #     patch_x2 = x2
    #     patch_y2 = y1 + patch_h
    #     cv2.rectangle(proc_img, (patch_x1, patch_y1), (patch_x2, patch_y2), bgr_color.tolist(), -1)
    #     cv2.putText(proc_img, f"HSV: {color_vector}", (patch_x1, patch_y2+20),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    # combined = np.hstack([orig_img, proc_img])
    cv2.imshow("Query Image", proc_img)
    cv2.waitKey(0)
    cv2.destroyWindow("Query Image")


# --- 설정: 파일 경로 및 변수 ---
# # MYEONGPUM_JSON = "k3_256_myeongpum_test_cat&color&clip&pattern.json"
# MYEONGPUM_JSON = "256_myeongpum_test_cat&color&clip&pattern.json"
# # MYEONGPUM_JSON = "k3_myeongpum_test_cat&color&clip&pattern.json"
# # MYEONGPUM_JSON = "myeongpum_test_cat&color&clip&pattern.json"
# MYEONGPUM_FEATURES_NPY = "myeongpum_test_df2matchrcnn_256.npy"
# # MYEONGPUM_FEATURES_NPY = "myeongpum_test_2048.npy"

# DUPE_JSON = "ver3_cat&color&clip&pattern.json"
# # DUPE_JSON = "updated_ver3_cat&color&clip&pattern.json"
# # DUPE_JSON = "k3_ver3_cat&color&clip&pattern.json"
# # DUPE_JSON = "ver2_cat&color&clip&pattern.json"
# DUPE_FEATURES_NPY = "ver3_df2matchrcnn_256.npy"
# # DUPE_FEATURES_NPY = "myeongpum_similar_df2matchrcnn_256.npy"
# # DUPE_FEATURES_NPY = "ver2_2048.npy"


# MYEONGPUM_JSON = "256_myeongpum_test_cat&color&clip&pattern.json"
MYEONGPUM_JSON = "myeongpum_cat&color&clip&pattern.json"
# MYEONGPUM_S1_FEATURES_NPY = "myeongpum_test_df2matchrcnn_256.npy"
MYEONGPUM_S1_FEATURES_NPY = "myeongpum_df2matchrcnn_256.npy"
# MYEONGPUM_FEATURES_NPY = "256_myeongpum_test_2048.npy"
MYEONGPUM_FEATURES_NPY = "myeongpum_2048.npy"

DUPE_JSON = "ver2_cat&color&clip&pattern.json"
DUPE_FEATURES_NPY = "ver2_2048.npy"

# DEBUGGING -> set to input parameter for final main.py
# QUERY_IMAGE_PATH = "manual_blazer_0001_01.jpg"
# QUERY_IMAGE_PATH = "manual_knit_0001_01.jpg" # BAD
# QUERY_IMAGE_PATH = "manual_knit_0002_01.jpg" # BAD
# QUERY_IMAGE_PATH = "manual_collarshirts_0001_01.jpg"
# QUERY_IMAGE_PATH = "manual_trucker_0007_01.jpg"

QUERY_IMAGE_PATH = "highlight_0001_0001.png"
# QUERY_IMAGE_PATH = "manual_leather_0001_01.jpg"

# QUERY_IMAGE_PATH = "manual_maxionepiece_0001_01.jpg"


### Dupe Recommendation before applying GDP ###
TOP_K = 8

# TODO
threshold_myeongpum = 0.61

# # --- 모델 로드 ---
### ResNet50 ###
# resnet_model = resnet50(pretrained=True)
# resnet_model.fc = torch.nn.Identity()
# resnet_model.to(device)
# resnet_model.eval()

### RoboFlow ###
# API_KEY = "XpCdCNt4CFykx5vyIApu"

if __name__ == "__main__":

    # 1. Query 이미지 로드 및 feature 추출
    query_img = find_image(QUERY_IMAGE_PATH)
    if query_img is None:
        exit(1)
    print("")
    print("Using query image:", QUERY_IMAGE_PATH)   
    print("")


    # 2. 객체 탐지
    df2_model = load_detection_model("df2matchrcnn", device)  # df2matchrcnn checkpoint 경로 필요
    direction = "left"  # "left", "right", "middle" 중 선택
    box_df2, label_df2 = detect_query_box_df2matchrcnn(query_img, df2_model, direction, score_threshold=0.5)
    crop_img = query_img.crop(box_df2)
    final_label = label_df2

    final_color = detect_color_kmeans_blur(query_img, box_df2)
    visualize_query_image(query_img, box_df2, final_label, final_color)

    final_color = np.array(final_color, dtype=float)
    norm_final = np.linalg.norm(final_color) or 1
    final_color = final_color / norm_final

    query_feature = compute_feature_df2matchrcnn(crop_img, df2_model, device)

    # 3. 후보 상품 선택 (myeongpum_test 단계) - combined similarity 방식 적용
    myeongpum_products = load_json(MYEONGPUM_JSON)
    if not os.path.exists(MYEONGPUM_S1_FEATURES_NPY):
        print(f"{MYEONGPUM_S1_FEATURES_NPY} 파일이 존재하지 않습니다.")
        exit(1)
    
    # NORMALIZE MYEONGPUM_FEATURES
    myeongpum_features = np.load(MYEONGPUM_S1_FEATURES_NPY, allow_pickle=True).astype(np.float32)
    norms = np.linalg.norm(myeongpum_features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    myeongpum_features_normalized = myeongpum_features / norms

    # print("")
    # print(f"{MYEONGPUM_FEATURES_NPY}: ")
    # for i in range(10):
    #     print(myeongpum_features[i])
    # print("")
    
    # disallowed는 그대로 사용
    disallowed = {"skirt", "shorts", "trousers"}
    # 객체 탐지에서 얻은 최종 카테고리 (대소문자 무시)
    query_category = final_label.lower()

    # allowed mapping: clothes 내의 category가 query_category와 동일한 항목만 선택
    # 기존: allowed mapping: clothes 내의 category가 query_category와 동일한 항목만 선택
    allowed_global_indices = []
    allowed_mapping = []  # 각 인덱스에 대해 (prod_idx, box_idx)
    global_idx = 0
    for prod_idx, product in enumerate(myeongpum_products):
        for box_idx, cloth in enumerate(product.get("clothes", [])):
            cat = cloth.get("category", "").lower()
            # If the query category is either "long sleeve top" or "long sleeve outwear"
            # this also prevents disallowed cateogries from being recommended
            if query_category in {"long sleeve top", "long sleeve outwear"}:
                if cat in {"long sleeve top", "long sleeve outwear"}:
                    allowed_global_indices.append(global_idx)
                    allowed_mapping.append((prod_idx, box_idx))
            elif query_category in {"short sleeve top", "short sleeve outwear"}:
                if cat in {"short sleeve top", "short sleeve outwear"}:
                    allowed_global_indices.append(global_idx)
                    allowed_mapping.append((prod_idx, box_idx))
            else:
                if cat == query_category:
                    allowed_global_indices.append(global_idx)
                    allowed_mapping.append((prod_idx, box_idx))
            global_idx += 1

    # 후보 상품 선택: combined similarity 방식으로 점수를 계산
    # 하이퍼파라미터 설정 (추천 상품 검색과 동일)
    # alpha = 0.3     # feature similarity
    # beta = 0.4      # color similarity
    # gamma = 0.4     # texture similarity
    # delta = 0     # season similarity
    # epsilon = 0   # mood similarity
    # zeta = 0      # category similarity

    alpha = 0.4     # feature similarity
    beta = 0.1      # color similarity
    eta = 0.2       # pattern similarity

    # query_feature는 앞서 추출한 값 사용.
    # query 색상은 detect_color_kmeans_blur()에서 얻은 final_color 사용.
    # query의 texture, season, mood는 객체 탐지된 영역 crop_img에서 추출.
    # query_texture = normalize_vector(classify_feature_vector(crop_img, texture_vectors))
    # query_season = normalize_vector(classify_feature_vector(crop_img, season_vectors))
    # query_mood = normalize_vector(classify_feature_vector(crop_img, mood_vectors))
    # # query_category = normalize_vector(classify_feature_vector(crop_img, pattern_vectors))
    # query_pattern = normalize_vector(classify_feature_vector(crop_img, pattern_vectors))
    
    query_texture = classify_feature_vector(crop_img, texture_vectors)
    query_season = classify_feature_vector(crop_img, season_vectors)
    query_mood = classify_feature_vector(crop_img, mood_vectors)
    # query_category = normalize_vector(classify_feature_vector(crop_img, pattern_vectors))
    query_pattern = classify_feature_vector(crop_img, pattern_vectors)

    candidate_scores = []
    candidate_details = []  # (global_idx, prod_idx, box_idx)
    allowed_mapping = []
    global_idx = 0

    for prod_idx, product in enumerate(myeongpum_products):
        for box_idx, cloth in enumerate(product.get("clothes", [])):
            cat = cloth.get("category", "").lower()
            # 카테고리 비교: sleeve 유형은 그대로, 그 외는 유사도 함수를 활용
            if query_category in {"long sleeve top", "long sleeve outwear"}:
                allowed = cat in {"long sleeve top", "long sleeve outwear"}
            elif query_category in {"short sleeve top", "short sleeve outwear"}:
                allowed = cat in {"short sleeve top", "short sleeve outwear"}
            else:
                allowed = is_similar_category(cat, query_category)
            
            if allowed:
                allowed_mapping.append((prod_idx, box_idx))
                # feature similarity: query_feature와 후보 feature는 이미 정규화되어 있다고 가정
                candidate_feat = myeongpum_features_normalized[global_idx]
                feat_sim = cosine_similarity(query_feature, candidate_feat)
                
                # 색상 유사도: 각 색상 벡터를 정규화 후 유클리드 기반 유사도 계산
                candidate_color = cloth.get("color_vector")
                if candidate_color is None:
                    color_sim = 0
                else:
                    candidate_color = np.array(candidate_color, dtype=float)
                    norm_candidate = np.linalg.norm(candidate_color) or 1
                    candidate_color = candidate_color / norm_candidate
                    color_sim = euclidean_similarity(final_color, candidate_color)
                
                # 패턴 유사도: 각 패턴 벡터를 정규화 후 맨해튼 기반 유사도 계산
                candidate_pattern = cloth.get("pattern_vector")
                pattern_sim = manhattan_similarity(query_pattern, candidate_pattern)
                
                combined = (alpha * feat_sim + beta * color_sim + eta * pattern_sim)

                candidate_scores.append(combined)
                candidate_details.append((global_idx, prod_idx, box_idx))
            global_idx += 1
    
    best_idx = np.argmax(candidate_scores)
    selected_global_idx, selected_prod_idx, selected_box_idx = candidate_details[best_idx]
    
    # candidate_product = myeongpum_products[selected_prod_idx]
    # candidate_sim = candidate_scores[best_idx]
    # candidate_img = load_and_crop_product_image(candidate_product, selected_box_idx)
    # candidate_box_idx = selected_box_idx  # 후보의 clothes 인덱스

    step1_myeongpum = myeongpum_products[selected_prod_idx]
    step1_sim = candidate_scores[best_idx]
    step1_img = load_and_crop_product_image(step1_myeongpum, selected_box_idx)
    step1_box_idx = selected_box_idx

    print("Candidate from myeongpum_test:")
    print("상품명:", step1_myeongpum.get("product_name"))
    print("상품 코드:", step1_myeongpum.get("product_code"))
    print("category_sub:", step1_myeongpum.get("category_sub"))
    print("유사도:", step1_sim)

    # ADDED
    # if step1_sim < threshold_myeongpum:
    #     print("No Product Found")
    #     exit(1)

    visualize_product_with_boxes(step1_myeongpum)

#     # query_category는 최종 감지된 category (소문자)
#     for i in range(len(allowed_global_indices)):
#         global_idx = allowed_global_indices[i]
#         prod_idx, box_idx = allowed_mapping[i]
#         candidate_feat = myeongpum_features_normalized[global_idx]
#         # candidate_feat = myeongpum_features_normalized[my_idx]

# #        feat_sim = cosine_similarity(query_feature, candidate_feat)

#         # feat_sim DEBUG START
#         if myeongpum_products[prod_idx].get("product_code") == QUERY_IMAGE_PATH.replace("_01.jpg", ""):
#             feat_sim = 1
#         else:
#             feat_sim = 0
#         # feat_sim DEBUG END

#         # Bring myeongpum details
#         product = myeongpum_products[prod_idx]
#         cloth = product.get("clothes", [])[box_idx]
        
#         # 색상 유사도 계산
#         # candidate_color = cloth.get("color_vector")
#         # if candidate_color is None:
#         #     color_sim = 0
#         # else:
#         #     final_color_normalized = normalize_vector(final_color)
#         #     candidate_color_normalized = normalize_vector(candidate_color)
#         #     # candidate_color = np.array(candidate_color, dtype=float)
#         #     # norm_candidate = np.linalg.norm(candidate_color)
#         #     # if norm_candidate == 0:
#         #     #     norm_candidate = 1
#         #     # candidate_color = candidate_color / norm_candidate
#         #     color_sim = cosine_similarity(final_color_normalized, candidate_color_normalized)
        
#         # feat_sim DEBUG START
#         if myeongpum_products[prod_idx].get("product_code") == QUERY_IMAGE_PATH.replace("_01.jpg", ""):
#             color_sim = 1
#         else:
#             color_sim = 0
#         # feat_sim DEBUG END

#         # 텍스처 유사도 계산
#         candidate_texture = cloth.get("texture_vector")
#         if candidate_texture is None:
#             texture_sim = 0
#         else:
#             candidate_texture = np.array(candidate_texture, dtype=float)
#             norm_candidate = np.linalg.norm(candidate_texture)
#             if norm_candidate == 0:
#                 norm_candidate = 1
#             candidate_texture = candidate_texture / norm_candidate
#             texture_sim = cosine_similarity(query_texture, candidate_texture)
#         # 시즌 유사도 계산
#         candidate_season = cloth.get("season_vector")
#         if candidate_season is None:
#             season_sim = 0
#         else:
#             candidate_season = np.array(candidate_season, dtype=float)
#             norm_candidate = np.linalg.norm(candidate_season)
#             if norm_candidate == 0:
#                 norm_candidate = 1
#             candidate_season = candidate_season / norm_candidate
#             season_sim = cosine_similarity(query_season, candidate_season)
#         # 무드 유사도 계산
#         candidate_mood = cloth.get("mood_vector")
#         if candidate_mood is None:
#             mood_sim = 0
#         else:
#             candidate_mood = np.array(candidate_mood, dtype=float)
#             norm_candidate = np.linalg.norm(candidate_mood)
#             if norm_candidate == 0:
#                 norm_candidate = 1
#             candidate_mood = candidate_mood / norm_candidate
#             mood_sim = cosine_similarity(query_mood, candidate_mood)
#         # category similarity (clothes 내 category 비교)
#         candidate_category = cloth.get("category", "").lower()
#         category_sim = 1 if candidate_category == query_category else 0

#         combined = (alpha * feat_sim +
#                     beta * color_sim +
#                     gamma * texture_sim +
#                     delta * season_sim +
#                     epsilon * mood_sim +
#                     zeta * category_sim)
        
#         candidate_scores.append(combined)
#         candidate_details.append((global_idx, prod_idx, box_idx))
    
#     # 후보 중 결합 유사도가 가장 높은 것을 선택
#     # 후보 중 결합 유사도가 가장 높은 것을 선택 (Top 10)
#     sorted_indices = np.argsort(candidate_scores)[::-1]
    
#     # DEBUG TOP 10 MYEONGPUM
#     # top_k = min(10, len(candidate_scores))

#     # print("Top {} candidates from myeongpum_test:".format(top_k))
#     # for rank in range(top_k):
#     #     idx = sorted_indices[rank]
#     #     selected_global_idx, selected_prod_idx, selected_box_idx = candidate_details[idx]
#     #     candidate_product = myeongpum_products[selected_prod_idx]
#     #     candidate_sim = candidate_scores[idx]
#     #     candidate_img = find_image(candidate_product.get("product_images_1"))
#     #     print("Rank {}:".format(rank + 1))
#     #     print("  상품명:", candidate_product.get("product_name"))
#     #     print("  상품 코드:", candidate_product.get("product_code"))
#     #     print("  category_sub:", candidate_product.get("category_sub"))
#     #     print("  유사도:", candidate_sim)
#     #     visualize_product_with_boxes(candidate_product)

#     # END DEBUG

#     best_idx = np.argmax(candidate_scores)
#     selected_global_idx, selected_prod_idx, selected_box_idx = candidate_details[best_idx]
    
#     step1_myeongpum = myeongpum_products[selected_prod_idx]
#     step1_sim = candidate_scores[best_idx]
#     step1_img = find_image(step1_myeongpum.get("product_images_1"))
#     step1_box_idx = selected_box_idx

#     print("")
#     print("Final Candidate from myeongpum_test:")
#     print("상품명:", step1_myeongpum.get("product_name"))
#     print("상품 코드:", step1_myeongpum.get("product_code"))
#     print("category_sub:", step1_myeongpum.get("category_sub"))
#     print("유사도:", step1_sim)
#     # print("NUMPY FEAT: ", myeongpum_features[selected_global_idx]) # DEBUG
#     print("")
#     visualize_product_with_boxes(step1_myeongpum)

    







    # --------------------------
    # 4. 추천 상품 검색 (ver2 단계) – allowed clothes 항목만 비교
    # --------------------------
    # candidate_product의 category_sub 기준으로 allowed 항목 필터링
    dupe_products = load_json(DUPE_JSON)

    if not os.path.exists(DUPE_FEATURES_NPY):
        print(f"{DUPE_FEATURES_NPY} 파일이 존재하지 않습니다.")
        exit(1)

    # NORMALIZE MYEONGPUM_FEATURES (For STEP 2)
    myeongpum_features = np.load(MYEONGPUM_FEATURES_NPY, allow_pickle=True).astype(np.float32)
    norms = np.linalg.norm(myeongpum_features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    myeongpum_features_normalized = myeongpum_features / norms

    # NORMALIZE FEATURES
    dupe_features = np.load(DUPE_FEATURES_NPY, allow_pickle=True).astype(np.float32)
    norms = np.linalg.norm(dupe_features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    dupe_features_normalized = dupe_features /  norms

    # use step1_myeongpum result to obtain parameter values (and normalize)
    step1_cloth = step1_myeongpum["clothes"][step1_box_idx]
    # step1_pattern의 pattern vector
    step1_pattern = step1_cloth.get("pattern_vector")
   # step1_myeongpum의 category_sub
    step1_cat_sub = step1_myeongpum.get("category_sub", "").lower()

    # allowed mapping for dupe_products based on candidate_product's category_sub
    allowed_global_indices_v1 = []
    allowed_mapping_v1 = []  # 각 인덱스에 대해 (prod_idx, box_idx)
    global_idx_v1 = 0

    for prod_idx, product in enumerate(dupe_products):
        product_cat_sub = product.get("category_sub", "")
        # 제품 레벨의 category_sub가 candidate와 일치하지 않으면 건너뜁니다.
        if product_cat_sub != step1_cat_sub:
            global_idx_v1 += len(product.get("clothes", []))
            continue
        # 일치하는 경우, 해당 상품의 모든 clothes 항목을 allowed mapping에 추가합니다.
        for box_idx, cloth in enumerate(product.get("clothes", [])):
            cat = cloth.get("category", "").lower()
            # skip disallowed categories
            if cat not in disallowed:
                #if candidate_product.get("product_code") == "manual_leather_0001":
                #     allowed_global_indices_v1.append(global_idx_v1)
                #     allowed_mapping_v1.append((prod_idx, box_idx))
                # else:
                dupe_pattern = cloth.get("pattern_vector")
                # Skip if query_pattern != product_pattern
                if step1_pattern is not None and dupe_pattern is not None:
                    if np.argmax(step1_pattern) == np.argmax(dupe_pattern):
                    # if np.argmax(query_pattern) == np.argmax(prod_pattern) and prod_pattern[np.argmax(prod_pattern)] > 0.24:
                        allowed_global_indices_v1.append(global_idx_v1)
                        allowed_mapping_v1.append((prod_idx, box_idx))
            global_idx_v1 += 1

    if len(allowed_global_indices_v1) == 0:
        print("추천 대상에서 허용된 항목이 없습니다.")
        exit(1)

    # REMOVE -- NONE OF THESE ARE BEING USED
    # allowed_features_v1는 allowed_global_indices_v1에 해당하는 feature만 포함
    allowed_features_v1 = dupe_features_normalized[allowed_global_indices_v1]
    index_v1_allowed = faiss.IndexFlatIP(allowed_features_v1.shape[1])
    index_v1_allowed.add(allowed_features_v1)

    # 후보 상품 이미지 feature는 이미 candidate_feat로 추출됨
    # try:
    #     color_vec = candidate_product["clothes"][candidate_box_idx]["color_vector"]
    #     query_color_rep = np.array(color_vec, dtype=float)
    #     print(f"QUERY COLOR: {query_color_rep}")
    #     norm_color = np.linalg.norm(query_color_rep)
    #     if norm_color == 0:
    #         norm_color = 1
    #     query_color_rep_norm = query_color_rep / norm_color
    # except Exception as e:
    #     query_color_rep_norm = np.zeros(3)

    # Retrieve the precomputed attribute vectors from the myeongpum's JSON entry
    # ADDED NORMALIZATION @sgyi22
    step1_color = step1_cloth.get("color_vector")
    
    step1_color_normalized = normalize_vector(step1_color)
    # step1_color_normalized = normalize_color_vectors(step1_color)
    
    step1_myeongpum_feat = myeongpum_features[selected_global_idx]
    # global_idx -> selected_global_idx change updated @sgyi22
    step1_feat_normalized = myeongpum_features_normalized[selected_global_idx]
    step1_texture_normalized = normalize_vector(step1_cloth.get("texture_vector"))
    step1_season_normalized  = normalize_vector(step1_cloth.get("season_vector"))
    step1_mood_normalized    = normalize_vector(step1_cloth.get("mood_vector"))
    step1_pattern_normalized = normalize_vector(step1_cloth.get("pattern_vector"))

    # Define hyperparameters (including new weight for pattern similarity)
    # ROLLBACK
    alpha = 0.4     # feature similarity
    beta  = 0.2     # color similarity
    gamma = 0.2     # pattern similarity
    delta = 0.05    # season similarity  
    epsilon = 0.1   # mood similarity
    zeta  = 0.05    # texture similarity
    # alpha = 0.3     # feature similarity
    # beta  = 0.4     # color similarity
    # gamma = 0.3     # pattern similarity
    # delta = 0.0    # season similarity  
    # epsilon = 0.0   # mood similarity
    # zeta  = 0.0   # texture similarity
    
    allowed_combined_scores = []
    candidate_details_debug = []

    # REMOVE
    # candidate_category = candidate_product["clothes"][candidate_box_idx].get("category")
    for idx in range(len(allowed_global_indices_v1)):
        global_idx_v1 = allowed_global_indices_v1[idx]
        prod_idx, box_idx = allowed_mapping_v1[idx]
        
        # feat similarity 계산
        # print("PROD FEATURE:", dupe_features[global_idx_v1])
        
        feat_sim = cosine_similarity(step1_feat_normalized.flatten(), dupe_features_normalized[global_idx_v1])
        dupe = dupe_products[prod_idx]
        dupe_cloth = dupe.get("clothes", [])[box_idx]
        
        # color similarity 계산
        dupe_color = dupe_cloth.get("color_vector")
        dupe_color_normalized = normalize_vector(dupe_color)
        # dupe_color_normalized = normalize_color_vectors(dupe_color)
        # (1) cosine similarity method
        color_sim = cosine_similarity(step1_color_normalized, dupe_color_normalized)
        # print(step1_color_normalized)
        # print(dupe_color_normalized)
        # color_sim = compute_emd(step1_color_normalized, dupe_color_normalized)
        # (2) euclidean distance method
        # color_sim = euclidean_similarity(step1_color_normalized, dupe_color_normalized)
        
        # TO REMOVE
        # if dupe_color is None:
        #     color_sim = 0
        # else:
        #     dupe_color = np.array(dupe_color, dtype=float)
            
        #     print(f"PROD COLOR: {dupe_color}")
            
        #     norm_dupe = np.linalg.norm(dupe_color)
        #     if norm_dupe == 0:
        #         norm_dupe = 1
        #     dupe_color_norm = dupe_color / norm_dupe

        # (2) euclidean distance method
        # if dupe_color is None:
        #     color_sim = 0
        # else:
        #     dupe_color = np.array(dupe_color, dtype=float)
        #     norm_dupe = np.linalg.norm(dupe_color)
        #     if norm_dupe == 0:
        #         norm_dupe = 1
        #     dupe_color_norm = dupe_color / norm_dupe
        #     # Compute Euclidean distance between the normalized dupe and dupeuct color vectors.
        #     d = np.linalg.norm(dupe_color_rep_norm - dupe_color_norm)
        #     # Convert distance to similarity (maximum distance between two unit vectors in 3D is 2)
        #     color_sim = 1 - (d / 2)
        # END REMOVE

        # texture similarity 계산
        dupe_texture = dupe_cloth.get("texture_vector")
        dupe_texture_normalized = normalize_vector(dupe_texture)
        texture_sim = cosine_similarity(step1_texture_normalized, dupe_texture_normalized)
        # if dupe_texture is None:
        #     texture_sim = 0
        # else:
        #     dupe_texture = np.array(dupe_texture, dtype=float)
        #     norm_dupe = np.linalg.norm(dupe_texture)
        #     if norm_dupe == 0:
        #         norm_dupe = 1
        #     dupe_texture = dupe_texture / norm_dupe
        #     texture_sim = cosine_similarity(dupe_texture, dupe_texture)
        
        # season similarity 계산
        dupe_season = dupe_cloth.get("season_vector")
        dupe_season_normalized = normalize_vector(dupe_season)
        season_sim = cosine_similarity(step1_season_normalized, dupe_season_normalized)
        # if dupe_season is None:
        #     season_sim = 0
        # else:
        #     dupe_season = np.array(dupe_season, dtype=float)
        #     norm_dupe = np.linalg.norm(dupe_season)
        #     if norm_dupe == 0:
        #         norm_dupe = 1
        #     dupe_season = dupe_season / norm_dupe
        #     season_sim = cosine_similarity(dupe_season, dupe_season)
        
        # mood similarity 계산
        dupe_mood = dupe_cloth.get("mood_vector")
        dupe_mood_normalized = normalize_vector(dupe_mood)
        mood_sim = cosine_similarity(step1_mood_normalized, dupe_mood_normalized)
        # if dupe_mood is None:
        #     mood_sim = 0
        # else:
        #     dupe_mood = np.array(dupe_mood, dtype=float)
        #     norm_dupe = np.linalg.norm(dupe_mood)
        #     if norm_dupe == 0:
        #         norm_dupe = 1
        #     dupe_mood = dupe_mood / norm_dupe
        #     mood_sim = cosine_similarity(dupe_mood, dupe_mood)
        
        # category similarity 계산 -- REMOVE
        # dupe_cat = dupe_cloth.get("category")
        # category_sim = 1 if dupe_cat == candidate_category else 0
        
        # pattern similarity 계산
        dupe_pattern = dupe_cloth.get("pattern_vector")
        dupe_pattern_normalized = normalize_vector(dupe_pattern)
        # if (dupe_pattern is not None) and (dupe_pattern is not None):
        if (step1_pattern_normalized is not None) and (dupe_pattern_normalized is not None):
            # step1_pattern_arr = np.array(step1_pattern, dtype=float)
            step1_pattern_arr = np.array(step1_pattern_normalized, dtype=float)
            dupe_pattern_arr = np.array(dupe_pattern_normalized, dtype=float)
            # Check if the highest value indices match; if not, skip this candidate.
            pattern_sim = step1_pattern_arr[np.argmax(dupe_pattern_arr)] * dupe_pattern_arr[np.argmax(dupe_pattern_arr)]
        else:
            pattern_sim = 0


        dupe_code = dupe.get("product_code")
        # Compute the weighted combined similarity
        combined = (alpha * feat_sim +
                    beta  * color_sim +
                    gamma  * pattern_sim +
                    delta * season_sim +
                    epsilon * mood_sim +
                    zeta * texture_sim)

        # Print detailed similarity scores for debugging
        print(f"  database (dupe_code= {dupe_code}, dupe_idx={prod_idx}, box_idx={box_idx}):")
        print(f"  myeongpum color: {step1_color}")
        print(f"  dupe color: {dupe_color}")
        print(f"  Feature similarity: {feat_sim:.4f} (weighted: {alpha * feat_sim:.4f})")
        print(f"  Color similarity:   {color_sim:.4f} (weighted: {beta  * color_sim:.4f})")
        print(f"  Texture similarity: {texture_sim:.4f} (weighted: {zeta * texture_sim:.4f})")
        print(f"  Season similarity:  {season_sim:.4f} (weighted: {delta * season_sim:.4f})")
        print(f"  Mood similarity:    {mood_sim:.4f} (weighted: {epsilon * mood_sim:.4f})")
        print(f"  Pattern similarity: {pattern_sim:.4f} (weighted: {gamma * pattern_sim:.4f})")
        print(f"  Combined score:     {combined:.4f}\n")

        allowed_combined_scores.append(combined)
        
        candidate_details_debug.append({
            "prod_code": dupe_code,
            "box_idx": box_idx,
            "feat_sim": feat_sim,
            "color_sim": color_sim,
            "texture_sim": texture_sim,
            "season_sim": season_sim,
            "mood_sim": mood_sim,
            "pattern_sim": pattern_sim,
            "combined": combined
        })

    # sort "scores" and add to "mapping"
    sorted_indices = np.argsort(allowed_combined_scores)[::-1]
    mapping = []
    seen_codes = set()
    for idx in sorted_indices:
        prod_idx, box_idx = allowed_mapping_v1[idx]
        dupe_code = dupe_products[prod_idx].get("product_code")
        if dupe_code in seen_codes:
            continue
        seen_codes.add(dupe_code)
        # mapping에는 allowed_combined_scores의 인덱스(idx)를 저장합니다.
        mapping.append((idx, prod_idx, box_idx))
        if len(mapping) >= TOP_K:
            break

    if len(mapping) == 0:
        print("추천 대상에서 허용된 항목이 없습니다.")
        exit(1)

    # Start Golden Dupe Price Function (GDP) @ sgyi22
    print("Starting GDP Function:\n")
    
    myeongpum_code = step1_myeongpum.get("product_code")
    print(f"Myeongpum Product Code: {myeongpum_code}:\n")
    myeongpum_price = get_final_price(step1_myeongpum.get("final_price"))
    weighted_scores = {}
    for score_idx, p_idx, b_idx in mapping:
        dupe_code = dupe_products[p_idx].get("product_code")
        dupe_price = get_final_price(dupe_products[p_idx].get("final_price"))
        dc_ratio_value = ((myeongpum_price - dupe_price) / myeongpum_price) * 100
        selected_weight = continuous_bump_function(dc_ratio_value)[0]
        
        original_score = allowed_combined_scores[score_idx]
        # if dupe_code == "manual_blazer_0004": # Checkered-Zara
        #     selected_weight = 1.5
        new_score = original_score * selected_weight
        weighted_scores[(score_idx, p_idx, b_idx)] = new_score

        product = dupe_products[p_idx]
        print(f"Golden Dupe Price (GDP) for {dupe_code}: {dc_ratio_value:.2f}%")
        print("Product Name:", product.get("product_name"))
        print("Category Sub:", product.get("category_sub"))
        print("Product Final Price:", product.get("final_price"))
        # Print detailed similarity scores for debugging
        details = candidate_details_debug[score_idx]
        print(f"  Feature similarity: {details['feat_sim']:.4f} (weighted: {alpha * details['feat_sim']:.4f})")
        print(f"  Color similarity:   {details['color_sim']:.4f} (weighted: {beta * details['color_sim']:.4f})")
        print(f"  Texture similarity: {details['texture_sim']:.4f} (weighted: {zeta * details['texture_sim']:.4f})")
        print(f"  Season similarity:  {details['season_sim']:.4f} (weighted: {delta * details['season_sim']:.4f})")
        print(f"  Mood similarity:    {details['mood_sim']:.4f} (weighted: {epsilon * details['mood_sim']:.4f})")
        print(f"  Pattern similarity: {details['pattern_sim']:.4f} (weighted: {gamma * details['pattern_sim']:.4f})")
        print(f"Combined score: {details['combined']:.4f}")
        print(f"Selected Weight for {dupe_code}: {selected_weight:.2f}")
        print(f"Original Score: {original_score:.4f} -> Weighted Score: {new_score:.4f}\n")
        
        # CURRENT RANK PRINT #
        product = dupe_products[p_idx]
        img_path = product.get("product_images_1")
        if not img_path:
            continue
        if not img_path.lower().endswith(".jpg"):
            img_path += ".jpg"
        prod_img = cv2.imread(img_path)
        if prod_img is None:
            continue

        # allowed clothes 항목 찾기 -- Not Needed (Already Filtered)
        allowed_cloth = None
        for cloth in product.get("clothes", []):
            if cloth.get("category", "").lower() not in disallowed:
                allowed_cloth = cloth
                break
        if allowed_cloth is None:
            continue

        box = allowed_cloth.get("box")
        score = allowed_cloth.get("score", 0)
        prod_cat = allowed_cloth.get("category", "unknown")
        # prod_catsub = product.get("category_sub")
        prod_code = product.get("product_code", "NoName")

        if box and len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(prod_img, (x1, y1), (x2, y2), (0,255,0), 2)
            text = f"{prod_code} | {score:.2f} | {prod_cat}"
            cv2.putText(prod_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            color_vector = allowed_cloth.get("color_vector")
            # if color_vector is not None:
            #     hsv_color = np.uint8([[color_vector]])
            #     bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            #     patch_w, patch_h = 40, 40
            #     patch_x1 = x2 - patch_w
            #     patch_y1 = y1
            #     patch_x2 = x2
            #     patch_y2 = y1 + patch_h
            #     cv2.rectangle(prod_img, (patch_x1, patch_y1), (patch_x2, patch_y2), bgr_color.tolist(), -1)
            #     cv2.putText(prod_img, f"HSV: {color_vector}", (patch_x1, patch_y2+20),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        # Resize the image before displaying
        scale_factor = 0.4
        display_img = cv2.resize(prod_img, None, fx=scale_factor, fy=scale_factor)

        window_name = f"Recommended: {prod_code}"
        cv2.imshow(window_name, display_img)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        
    # Resort using weighted scores after GDP
    sorted_weighted_mapping = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
    reordered_mapping = [(score_idx, p_idx, b_idx) for (score_idx, p_idx, b_idx), _ in sorted_weighted_mapping]
    
    # 최중 (GDP 반영) 추천 상품 시각화 (allowed 항목 기준 -- 필요없음)
    for score_idx, p_idx, b_idx in reordered_mapping[:4]:
        product = dupe_products[p_idx]
        img_path = product.get("product_images_1")
        if not img_path:
            continue
        if not img_path.lower().endswith(".jpg"):
            img_path += ".jpg"
        prod_img = cv2.imread(img_path)
        if prod_img is None:
            continue

        # allowed clothes 항목 찾기 -- Not Needed (Already Filtered)
        allowed_cloth = None
        for cloth in product.get("clothes", []):
            if cloth.get("category", "").lower() not in disallowed:
                allowed_cloth = cloth
                break
        if allowed_cloth is None:
            continue

        box = allowed_cloth.get("box")
        score = allowed_cloth.get("score", 0)
        prod_cat = allowed_cloth.get("category", "unknown")
        # prod_catsub = product.get("category_sub")
        prod_code = product.get("product_code", "NoName")

        details = candidate_details_debug[score_idx]

        print("---")
        print("추천 상품명:", product.get("product_name"))
        print("카테고리:", product.get("category_sub"))
        print("추천 상품 코드:", prod_code)
        print("최종 가격:", product.get("final_price"))
        # print("유사도:", allowed_combined_scores[score_idx])
        # Print detailed similarity scores for debugging
        # print(f"  Feature similarity: {details['feat_sim']:.4f} (weighted: {alpha * details['feat_sim']:.4f})")
        # print(f"  Color similarity:   {details['color_sim']:.4f} (weighted: {beta * details['color_sim']:.4f})")
        # print(f"  Texture similarity: {details['texture_sim']:.4f} (weighted: {zeta * details['texture_sim']:.4f})")
        # print(f"  Season similarity:  {details['season_sim']:.4f} (weighted: {delta * details['season_sim']:.4f})")
        # print(f"  Mood similarity:    {details['mood_sim']:.4f} (weighted: {epsilon * details['mood_sim']:.4f})")
        # print(f"  Pattern similarity: {details['pattern_sim']:.4f} (weighted: {gamma * details['pattern_sim']:.4f})")
        print(f"점수 (GDP 적용): {weighted_scores[(score_idx, p_idx, b_idx)]:.4f}")
        print("---")

        if box and len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(prod_img, (x1, y1), (x2, y2), (0,255,0), 2)
            text = f"{prod_code} | {score:.2f} | {prod_cat}"
            cv2.putText(prod_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # color_vector = allowed_cloth.get("color_vector")
            # if color_vector is not None:
            #     hsv_color = np.uint8([[color_vector]])
            #     bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            #     patch_w, patch_h = 40, 40
            #     patch_x1 = x2 - patch_w
            #     patch_y1 = y1
            #     patch_x2 = x2
            #     patch_y2 = y1 + patch_h
            #     cv2.rectangle(prod_img, (patch_x1, patch_y1), (patch_x2, patch_y2), bgr_color.tolist(), -1)
            #     cv2.putText(prod_img, f"HSV: {color_vector}", (patch_x1, patch_y2+20),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        # Resize the image before displaying
        scale_factor = 0.4
        display_img = cv2.resize(prod_img, None, fx=scale_factor, fy=scale_factor)

        window_name = f"Recommended: {prod_code}"
        cv2.imshow(window_name, display_img)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()
