# 🔬 SemiSA: SAT Image Segmentation with Fine-tuned SAM

**논문**: *Optimizing Scanning Acoustic Tomography Image Segmentation With Segment Anything Model for Semiconductor Devices*  
**IEEE Transactions on Semiconductor Manufacturing, Vol. 37, No. 4, November 2024**

---

## 📋 전체 파이프라인
1. 환경 설치 (SAM, 라이브러리)
2. SAM ViT-B 사전학습 모델 다운로드
3. 데이터셋 준비 (커스텀 SAT 이미지 or 데모용 샘플)
4. 데이터 전처리 (Butterworth 필터, 노이즈 제거, 1024×1024 리사이즈)
5. 이미지 어노테이션 생성
6. SemiSA 모델 정의 (SAM ViT-B fine-tuning)
7. 학습 (Dice Loss + Cross-Entropy Loss, Adam optimizer)
8. 평가 (DSC, IoU 메트릭)
9. 추론 및 시각화

> ⚠️ **GPU 런타임 필수**: 런타임 → 런타임 유형 변경 → GPU (T4 권장)

---

## 1️⃣ 환경 설치

```python
# GPU 확인
!nvidia-smi
import torch
print(f'PyTorch 버전: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
```

```python
# 필요한 라이브러리 설치
!pip install -q git+https://github.com/facebookresearch/segment-anything.git
!pip install -q opencv-python-headless monai scipy scikit-image matplotlib tqdm
```

```python
# SAM ViT-B 사전학습 가중치 다운로드 (~375MB)
import os
if not os.path.exists('sam_vit_b_01ec64.pth'):
    !wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    print('✅ SAM ViT-B 체크포인트 다운로드 완료')
else:
    print('✅ 이미 다운로드됨')
```

---

## 2️⃣ 라이브러리 임포트

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import cv2
import os
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import butter, filtfilt
from skimage import filters
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 디바이스: {DEVICE}')
```

---

## 3️⃣ 데이터 전처리 (논문 Section III-B)

논문에서 사용한 전처리 파이프라인:
- Butterworth band-pass 필터로 노이즈 제거
- OpenCV fastNlMeansDenoising
- 1024×1024 리사이즈
- 0~255 정규화
- 데이터 어그멘테이션 (flip, rotate, mosaic 등)

```python
# ──────────────────────────────────────────────
# 논문 Section III-B: 이미지 전처리 함수
# ──────────────────────────────────────────────

def butterworth_bandpass(data, lowcut, highcut, fs, order=4):
    """
    Butterworth band-pass 필터 (논문 참조 [12])
    SAT 초음파 신호의 노이즈 제거에 사용
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))
    if low >= high:
        return data
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)


def preprocess_sat_image(image, target_size=1024):
    """
    논문 Section III-B의 전처리 파이프라인
    1. Grayscale 변환
    2. Butterworth 필터 적용 (SAT 신호 노이즈 제거)
    3. fastNlMeansDenoising (OpenCV)
    4. 1024×1024 리사이즈
    5. 0~255 정규화
    6. RGB 변환 (SAM 입력 형식)
    """
    # Grayscale 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Butterworth 필터 적용
    gray_float = gray.astype(np.float64)
    filtered = butterworth_bandpass(gray_float, lowcut=1e5, highcut=5e6, fs=2e7)
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)

    # OpenCV fastNlMeansDenoising (논문 참조 [13])
    denoised = cv2.fastNlMeansDenoising(filtered, h=10, templateWindowSize=7, searchWindowSize=21)

    # 1024×1024 리사이즈 (zero padding으로 비율 유지)
    h, w = denoised.shape
    if h != target_size or w != target_size:
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(denoised, (new_w, new_h))
        padded = np.zeros((target_size, target_size), dtype=np.uint8)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        denoised = padded

    # 0~255 정규화
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)

    # RGB 변환 (SAM은 3채널 RGB 입력)
    rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    return rgb


def augment_image(image, mask):
    """
    논문 Section III-B: 데이터 어그멘테이션
    - Translation, Flipping, Rotation, Scaling, Shearing
    """
    h, w = image.shape[:2]

    # 랜덤 좌우 반전
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    # 랜덤 상하 반전
    if random.random() > 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    # 랜덤 회전 (-30 ~ +30도)
    if random.random() > 0.5:
        angle = random.uniform(-30, 30)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        mask = cv2.warpAffine(mask, M, (w, h))

    # 랜덤 스케일링 (0.8 ~ 1.2배)
    if random.random() > 0.5:
        scale = random.uniform(0.8, 1.2)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, (w, h))
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return image, mask


print('✅ 전처리 함수 정의 완료')
```

---

## 4️⃣ 데모 데이터셋 생성

**실제 SAT 데이터가 있는 경우**: 아래 `DEMO_MODE = False`로 변경하고 데이터 경로를 지정하세요.  
**데모 모드**: 합성 이미지로 파이프라인 동작을 검증합니다.

```python
# ──────────────────────────────────────────────
# 데이터셋 설정
# ──────────────────────────────────────────────

DEMO_MODE = True           # True: 합성 데이터 생성 / False: 실제 데이터 사용
DATA_DIR = './sat_dataset'  # 실제 데이터 경로 (DEMO_MODE=False일 때)
IMAGE_SIZE = 1024
BATCH_SIZE = 2             # Colab 메모리에 맞게 조정 (논문은 16, T4 GPU는 2~4 권장)
NUM_EPOCHS = 50            # 논문은 10,000 epoch (데모는 50으로 단축)
LEARNING_RATE = 1e-5       # 논문과 동일
TRAIN_RATIO = 0.8          # 논문과 동일


def generate_demo_sat_image(size=1024, sample_type='flip_chip'):
    """
    SAT 이미지를 흉내낸 합성 데이터 생성 (데모용)
    실제 사용 시에는 TSAM-400 시스템으로 스캔한 .bmp 파일을 사용
    """
    img = np.zeros((size, size), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    if sample_type == 'flip_chip':
        bg_noise = np.random.normal(30, 10, (size, size)).clip(0, 255).astype(np.uint8)
        img = bg_noise.copy()
        rows, cols = 8, 8
        spacing = size // (rows + 1)
        for r in range(rows):
            for c in range(cols):
                cy = (r + 1) * spacing
                cx = (c + 1) * spacing
                radius = 20
                brightness = random.randint(160, 220)
                cv2.circle(img, (cx, cy), radius, int(brightness), -1)
                cv2.circle(mask, (cx, cy), radius, 255, -1)

    elif sample_type == 'wafer':
        noise = np.random.normal(50, 15, (size, size)).clip(0, 255).astype(np.uint8)
        img = noise.copy()
        center = (size // 2, size // 2)
        radius = int(size * 0.45)
        cv2.circle(img, center, radius, 180, -1)
        cv2.circle(mask, center, radius, 255, -1)

    elif sample_type == 'mlcc':
        noise = np.random.normal(40, 12, (size, size)).clip(0, 255).astype(np.uint8)
        img = noise.copy()
        for _ in range(16):
            x = random.randint(50, size - 150)
            y = random.randint(50, size - 100)
            w = random.randint(40, 80)
            h = random.randint(25, 50)
            brightness = random.randint(150, 210)
            cv2.rectangle(img, (x, y), (x+w, y+h), int(brightness), -1)
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img_rgb, mask


# 데모 데이터셋 생성
if DEMO_MODE:
    os.makedirs('./sat_dataset/images', exist_ok=True)
    os.makedirs('./sat_dataset/masks', exist_ok=True)

    sample_types = ['flip_chip', 'wafer', 'mlcc']
    n_samples = 30

    for i in range(n_samples):
        stype = sample_types[i % len(sample_types)]
        img, mask = generate_demo_sat_image(size=1024, sample_type=stype)
        img_processed = preprocess_sat_image(img, target_size=1024)
        cv2.imwrite(f'./sat_dataset/images/{i:04d}_{stype}.png', img_processed)
        cv2.imwrite(f'./sat_dataset/masks/{i:04d}_{stype}.png', mask)

    print(f'✅ 데모 데이터 {n_samples}장 생성 완료')

    # 샘플 시각화
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    for idx, stype in enumerate(sample_types):
        img, mask = generate_demo_sat_image(size=256, sample_type=stype)
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'{stype} - 이미지', fontsize=11)
        axes[idx, 0].axis('off')
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title(f'{stype} - 마스크 (GT)', fontsize=11)
        axes[idx, 1].axis('off')
    plt.suptitle('SAT 이미지 샘플 (데모)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./sample_data.png', dpi=100, bbox_inches='tight')
    plt.show()
    print('✅ 샘플 시각화 저장')
```

---

## 5️⃣ Dataset 클래스 정의

```python
class SATDataset(Dataset):
    """
    SAT 이미지 데이터셋
    - 논문에서 사용한 6가지 반도체 타입 지원
    - Bounding box 프롬프트 자동 생성 (GT 마스크 기반 + 랜덤 perturbation 0~20px)
    """
    def __init__(self, image_dir, mask_dir, target_size=256, augment=True, perturbation=20):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.augment = augment
        self.perturbation = perturbation  # 논문: 0~20px 랜덤 perturbation

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.bmp'))
        ])
        print(f'데이터셋 로드: {len(self.image_files)}장')

        self.transform = ResizeLongestSide(target_size)

    def __len__(self):
        return len(self.image_files)

    def get_bounding_box(self, mask, perturbation=20):
        """
        GT 마스크에서 Bounding Box 추출 + 랜덤 perturbation 적용
        논문 Section III-D-2: '0 to 20 pixels random perturbation'
        """
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.array([0, 0, mask.shape[1], mask.shape[0]])

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        h, w = mask.shape
        x_min = max(0, x_min - random.randint(0, perturbation))
        y_min = max(0, y_min - random.randint(0, perturbation))
        x_max = min(w, x_max + random.randint(0, perturbation))
        y_max = min(h, y_max + random.randint(0, perturbation))

        return np.array([x_min, y_min, x_max, y_max])

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        mask_name = fname

        img = cv2.imread(os.path.join(self.image_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.mask_dir, mask_name)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        if self.augment:
            img, mask = augment_image(img, mask)

        img_resized = self.transform.apply_image(img)
        img_tensor = torch.as_tensor(img_resized, dtype=torch.float32).permute(2, 0, 1)

        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        img_tensor = (img_tensor - pixel_mean) / pixel_std

        h, w = img_tensor.shape[1], img_tensor.shape[2]
        padh = self.target_size - h
        padw = self.target_size - w
        img_tensor = F.pad(img_tensor, (0, padw, 0, padh))

        mask_binary = (mask > 127).astype(np.float32)
        mask_resized = cv2.resize(mask_binary, (self.target_size, self.target_size),
                                  interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.as_tensor(mask_resized, dtype=torch.float32).unsqueeze(0)

        bbox = self.get_bounding_box(
            mask_resized,
            perturbation=self.perturbation if self.augment else 0
        )
        bbox_tensor = torch.as_tensor(bbox, dtype=torch.float32)

        return img_tensor, mask_tensor, bbox_tensor, fname


print('✅ Dataset 클래스 정의 완료')
```

---

## 6️⃣ SemiSA 모델 정의 (논문 Section III-D-1)

- **Image Encoder**: SAM ViT-B (frozen)
- **Prompt Encoder**: Bounding Box 전용 (fine-tuning)
- **Mask Decoder**: fine-tuning 대상

```python
class SemiSA(nn.Module):
    """
    SemiSA 모델 - 논문의 Figure 2 아키텍처 구현

    구성:
    - Image Encoder: SAM ViT-B (가중치 고정 = 효율적 fine-tuning)
    - Prompt Encoder: Bounding Box 프롬프트만 사용 (Point/Text 제거)
    - Mask Decoder: 완전 fine-tuning
    """
    def __init__(self, sam_checkpoint, model_type='vit_b', device='cuda'):
        super().__init__()
        self.device = device

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam = self.sam.to(device)

        # Image Encoder 고정
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False

        # Prompt Encoder 고정
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False

        # Mask Decoder만 학습 가능
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True

        total = sum(p.numel() for p in self.sam.parameters())
        trainable = sum(p.numel() for p in self.sam.parameters() if p.requires_grad)
        print(f'전체 파라미터: {total:,}')
        print(f'학습 가능 파라미터: {trainable:,} ({100*trainable/total:.1f}%)')

    def forward(self, images, bboxes, image_size):
        """
        Forward pass
        Args:
            images: (B, 3, H, W) - 전처리된 이미지
            bboxes: (B, 4) - [x1, y1, x2, y2] bounding box
            image_size: 원본 이미지 크기 (H, W)
        """
        batch_size = images.shape[0]
        pred_masks = []
        iou_preds = []

        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(images)

        for i in range(batch_size):
            box = bboxes[i].unsqueeze(0)
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=None,
                    boxes=box,
                    masks=None
                )

            low_res_masks, iou_pred = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            upscaled = self.sam.postprocess_masks(
                low_res_masks,
                input_size=image_size,
                original_size=image_size
            )
            pred_masks.append(upscaled)
            iou_preds.append(iou_pred)

        return torch.cat(pred_masks, dim=0), torch.cat(iou_preds, dim=0)


print('✅ SemiSA 모델 클래스 정의 완료')
```

---

## 7️⃣ 손실 함수 정의 (논문 Eq. 1-3)

$$L = L_D + L_C$$
$$L_D = 1 - \frac{2 \sum G_i R_i}{\sum G_i^2 + \sum R_i^2}$$
$$L_C = -\sum G_i \log(R_i)$$

```python
class DiceLoss(nn.Module):
    """논문 Equation (1): Dice Loss"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / \
               (pred.pow(2).sum() + target.pow(2).sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """논문 Equation (3): L = L_Dice + L_CrossEntropy"""
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        L_D = self.dice_loss(pred, target)
        L_C = self.bce_loss(pred, target)
        return L_D + L_C, L_D.item(), L_C.item()


print('✅ 손실 함수 정의 완료')
```

---

## 8️⃣ 평가 메트릭 정의 (논문 Eq. 4-5)

$$DSC = \frac{2|G \cap R|}{|G| + |R|}$$
$$IoU = \frac{|G \cap R|}{|G \cup R|}$$

```python
def compute_dsc(pred_mask, gt_mask, threshold=0.5):
    """논문 Equation (4): Dice Similarity Coefficient"""
    pred = (torch.sigmoid(pred_mask) > threshold).float()
    target = gt_mask.float()
    intersection = (pred * target).sum()
    dsc = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
    return dsc.item()


def compute_iou(pred_mask, gt_mask, threshold=0.5):
    """논문 Equation (5): Intersection over Union"""
    pred = (torch.sigmoid(pred_mask) > threshold).float()
    target = gt_mask.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection) / (union + 1e-8)
    return iou.item()


print('✅ 평가 메트릭 정의 완료')
```

---

## 9️⃣ 데이터로더 준비

```python
IMG_SIZE = 1024   # SAM ViT-B는 내부적으로 1024×1024 pos_embed 사용 (반드시 1024 고정)
BATCH_SIZE = 1    # T4 기준 1~2 권장 (OOM나면 1로)

image_dir = './sat_dataset/images'
mask_dir  = './sat_dataset/masks'

all_files = sorted(os.listdir(image_dir))
random.seed(42)
random.shuffle(all_files)

split = int(len(all_files) * TRAIN_RATIO)
train_files = all_files[:split]
val_files   = all_files[split:]
print(f'학습: {len(train_files)}장 / 검증: {len(val_files)}장')

import shutil
for split_name, files in [('train', train_files), ('val', val_files)]:
    os.makedirs(f'./sat_dataset/{split_name}/images', exist_ok=True)
    os.makedirs(f'./sat_dataset/{split_name}/masks', exist_ok=True)
    for f in files:
        src_img  = os.path.join(image_dir, f)
        src_mask = os.path.join(mask_dir, f)
        dst_img  = f'./sat_dataset/{split_name}/images/{f}'
        dst_mask = f'./sat_dataset/{split_name}/masks/{f}'
        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)
        if os.path.exists(src_mask) and not os.path.exists(dst_mask):
            shutil.copy2(src_mask, dst_mask)

train_dataset = SATDataset('./sat_dataset/train/images',
                           './sat_dataset/train/masks',
                           target_size=1024, augment=True)
val_dataset   = SATDataset('./sat_dataset/val/images',
                           './sat_dataset/val/masks',
                           target_size=1024, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=1,
                          shuffle=False, num_workers=2)

print(f'✅ DataLoader 준비 완료')
```

---

## 🔟 모델 초기화 및 학습 루프

논문 Section III-D-2: Adam optimizer, lr=1e-5

```python
# 모델 초기화
model = SemiSA(
    sam_checkpoint='sam_vit_b_01ec64.pth',
    model_type='vit_b',
    device=DEVICE
)

optimizer = Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

criterion = CombinedLoss()
print('✅ 모델, Optimizer, Loss 함수 초기화 완료')
```

```python
# ──────────────────────────────────────────────
# 학습 루프
# ──────────────────────────────────────────────
history = {
    'train_loss': [], 'val_loss': [],
    'val_dsc': [], 'val_iou': []
}
best_dsc = 0.0
image_size_tuple = (IMG_SIZE, IMG_SIZE)

print(f'🚀 학습 시작 (총 {NUM_EPOCHS} epochs)\n')

for epoch in range(NUM_EPOCHS):
    # ── Training ──
    model.train()
    train_losses = []

    for imgs, masks, bboxes, _ in train_loader:
        imgs   = imgs.to(DEVICE)
        masks  = masks.to(DEVICE)
        bboxes = bboxes.to(DEVICE)

        optimizer.zero_grad()
        pred_masks, _ = model(imgs, bboxes, image_size_tuple)

        pred_masks_resized = F.interpolate(
            pred_masks, size=(IMG_SIZE, IMG_SIZE),
            mode='bilinear', align_corners=False
        )

        loss, l_d, l_c = criterion(pred_masks_resized, masks)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # ── Validation ──
    model.eval()
    val_losses, val_dscs, val_ious = [], [], []

    with torch.no_grad():
        for imgs, masks, bboxes, _ in val_loader:
            imgs   = imgs.to(DEVICE)
            masks  = masks.to(DEVICE)
            bboxes = bboxes.to(DEVICE)

            pred_masks, _ = model(imgs, bboxes, image_size_tuple)
            pred_masks_resized = F.interpolate(
                pred_masks, size=(IMG_SIZE, IMG_SIZE),
                mode='bilinear', align_corners=False
            )

            loss, _, _ = criterion(pred_masks_resized, masks)
            dsc = compute_dsc(pred_masks_resized, masks)
            iou = compute_iou(pred_masks_resized, masks)

            val_losses.append(loss.item())
            val_dscs.append(dsc)
            val_ious.append(iou)

    avg_train_loss = np.mean(train_losses)
    avg_val_loss   = np.mean(val_losses)
    avg_dsc        = np.mean(val_dscs)
    avg_iou        = np.mean(val_ious)

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_dsc'].append(avg_dsc)
    history['val_iou'].append(avg_iou)

    if avg_dsc > best_dsc:
        best_dsc = avg_dsc
        torch.save(model.state_dict(), './semisa_best.pth')

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1:4d}/{NUM_EPOCHS}] '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | '
              f'DSC: {avg_dsc*100:.2f}% | '
              f'IoU: {avg_iou*100:.2f}%')

print(f'\n🏆 학습 완료! 최고 DSC: {best_dsc*100:.2f}%')
```

---

## 1️⃣1️⃣ 학습 곡선 시각화 (논문 Figure A.2 재현)

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history['train_loss'], label='Train Loss', color='steelblue', linewidth=2)
axes[0].plot(history['val_loss'],   label='Val Loss',   color='tomato',    linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (Dice + Cross-Entropy)')
axes[0].set_title('학습/검증 손실 곡선', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot([d*100 for d in history['val_dsc']], color='mediumseagreen', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('DSC (%)')
axes[1].set_title('Dice Similarity Coefficient', fontweight='bold')
axes[1].axhline(y=best_dsc*100, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_dsc*100:.2f}%')
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot([i*100 for i in history['val_iou']], color='darkorange', linewidth=2)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('IoU (%)')
axes[2].set_title('Intersection over Union', fontweight='bold')
axes[2].grid(alpha=0.3)

plt.suptitle('SemiSA 학습 결과', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('./training_curves.png', dpi=100, bbox_inches='tight')
plt.show()
print('✅ 학습 곡선 저장: training_curves.png')
```

---

## 1️⃣2️⃣ 추론 및 결과 시각화 (논문 Figure 7, 8 재현)

```python
def visualize_segmentation(model, dataset, num_samples=3, device='cuda'):
    """
    논문 Figure 7, 8과 동일한 형식의 세그멘테이션 결과 시각화
    (a) Ground Truth  (b) 예측 마스크  (c) 오버레이
    """
    model.eval()
    image_size_tuple = (IMG_SIZE, IMG_SIZE)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for row, idx in enumerate(indices):
        img_tensor, mask_tensor, bbox_tensor, fname = dataset[idx]

        img_t   = img_tensor.unsqueeze(0).to(device)
        bbox_t  = bbox_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_masks, iou_pred = model(img_t, bbox_t, image_size_tuple)

        pred_resized = F.interpolate(
            pred_masks, size=(IMG_SIZE, IMG_SIZE),
            mode='bilinear', align_corners=False
        )
        pred_binary = (torch.sigmoid(pred_resized) > 0.5).squeeze().cpu().numpy()
        gt_mask     = mask_tensor.squeeze().numpy()

        dsc = compute_dsc(pred_resized, mask_tensor.unsqueeze(0).to(device))
        iou = compute_iou(pred_resized, mask_tensor.unsqueeze(0).to(device))

        mean = np.array([123.675, 116.28, 103.53])
        std  = np.array([58.395, 57.12, 57.375])
        img_vis = img_tensor.permute(1, 2, 0).numpy() * std + mean
        img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)

        # (a) Ground Truth
        axes[row, 0].imshow(img_vis)
        gt_overlay = np.zeros((*gt_mask.shape, 4), dtype=np.uint8)
        gt_overlay[gt_mask > 0.5] = [0, 255, 0, 100]
        axes[row, 0].imshow(gt_overlay)
        axes[row, 0].set_title(f'(a) Ground Truth\n{fname[:20]}', fontsize=9)
        axes[row, 0].axis('off')

        # (b) SemiSA 예측
        axes[row, 1].imshow(img_vis)
        pred_overlay = np.zeros((*pred_binary.shape, 4), dtype=np.uint8)
        pred_overlay[pred_binary] = [255, 50, 50, 120]
        axes[row, 1].imshow(pred_overlay)
        axes[row, 1].set_title(
            f'(b) SemiSA 예측\nDSC: {dsc*100:.1f}% | IoU: {iou*100:.1f}%', fontsize=9
        )
        axes[row, 1].axis('off')

        # (c) 오차 맵 (TP: 초록, FP: 빨강, FN: 파랑)
        diff_map = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        gt_bool   = gt_mask > 0.5
        pred_bool = pred_binary.astype(bool)
        diff_map[gt_bool & pred_bool]  = [0, 200, 0]    # TP
        diff_map[~gt_bool & pred_bool] = [255, 0, 0]    # FP
        diff_map[gt_bool & ~pred_bool] = [0, 0, 255]    # FN
        axes[row, 2].imshow(diff_map)
        axes[row, 2].set_title('(c) 오차 맵\n🟢TP 🔴FP 🔵FN', fontsize=9)
        axes[row, 2].axis('off')

    plt.suptitle('SemiSA 세그멘테이션 결과 시각화', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./segmentation_results.png', dpi=100, bbox_inches='tight')
    plt.show()
    print('✅ 세그멘테이션 결과 저장: segmentation_results.png')


# 최적 모델 로드 후 시각화
model.load_state_dict(torch.load('./semisa_best.pth', map_location=DEVICE))
visualize_segmentation(model, val_dataset, num_samples=3, device=DEVICE)
```

---

## 1️⃣3️⃣ 최종 성능 평가 (논문 Table III 재현)

```python
def evaluate_model(model, dataloader, device, image_size):
    """전체 검증 세트에 대한 DSC, IoU 평가"""
    model.eval()
    all_dsc, all_iou = [], []
    image_size_tuple = (image_size, image_size)

    with torch.no_grad():
        for imgs, masks, bboxes, fnames in tqdm(dataloader, desc='평가 중'):
            imgs   = imgs.to(device)
            masks  = masks.to(device)
            bboxes = bboxes.to(device)

            pred_masks, _ = model(imgs, bboxes, image_size_tuple)
            pred_resized = F.interpolate(
                pred_masks, size=(image_size, image_size),
                mode='bilinear', align_corners=False
            )

            dsc = compute_dsc(pred_resized, masks)
            iou = compute_iou(pred_resized, masks)
            all_dsc.append(dsc)
            all_iou.append(iou)

    return np.mean(all_dsc), np.std(all_dsc), np.mean(all_iou), np.std(all_iou)


dsc_mean, dsc_std, iou_mean, iou_std = evaluate_model(
    model, val_loader, DEVICE, IMG_SIZE
)

print('\n' + '='*60)
print('📊 SemiSA 최종 평가 결과 (논문 Table III 형식)')
print('='*60)
print(f'  DSC (Dice Similarity Coefficient): {dsc_mean*100:.2f}% ± {dsc_std*100:.2f}%')
print(f'  IoU (Intersection over Union):     {iou_mean*100:.2f}% ± {iou_std*100:.2f}%')
print('='*60)
print()
print('📌 논문 기준값 (Table III 평균):')
print('   DSC: 96.09% ~ 98.22% (타입별 상이)')
print('   IoU: 92.98% ~ 96.12% (타입별 상이)')
print()
print('⚠️  데모 데이터로 학습 시 논문 수치와 차이가 있을 수 있습니다.')
print('   실제 TSAM-400 SAT 데이터 사용 시 논문 수준의 성능을 기대할 수 있습니다.')
```

---

## 1️⃣4️⃣ 실제 이미지로 추론 (Inference Only)

학습된 모델로 새로운 SAT 이미지를 세그멘테이션합니다.

```python
def inference_single_image(model, image_path, bbox=None, device='cuda', image_size=256):
    """
    단일 이미지 추론
    Args:
        image_path: 이미지 경로
        bbox: [x1, y1, x2, y2] 바운딩 박스 (None이면 전체 이미지)
    """
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        print('❌ 이미지를 불러올 수 없습니다')
        return

    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_processed = preprocess_sat_image(img_rgb, target_size=image_size)

    transform = ResizeLongestSide(image_size)
    img_resized = transform.apply_image(img_processed)
    img_tensor = torch.as_tensor(img_resized, dtype=torch.float32).permute(2, 0, 1)

    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    pixel_std  = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
    img_tensor = (img_tensor - pixel_mean) / pixel_std

    h, w = img_tensor.shape[1], img_tensor.shape[2]
    img_tensor = F.pad(img_tensor, (0, image_size - w, 0, image_size - h))

    if bbox is None:
        bbox = [0, 0, image_size, image_size]
    bbox_tensor = torch.as_tensor(bbox, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred_masks, iou_pred = model(
            img_tensor.unsqueeze(0).to(device),
            bbox_tensor.to(device),
            (image_size, image_size)
        )

    pred_resized = F.interpolate(
        pred_masks, size=(image_size, image_size),
        mode='bilinear', align_corners=False
    )
    pred_binary = (torch.sigmoid(pred_resized) > 0.5).squeeze().cpu().numpy()

    mean = np.array([123.675, 116.28, 103.53])
    std  = np.array([58.395, 57.12, 57.375])
    img_vis = img_tensor.permute(1, 2, 0).numpy() * std + mean
    img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_vis)
    if bbox:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                   linewidth=2, edgecolor='yellow', facecolor='none')
        axes[0].add_patch(rect)
    axes[0].set_title('입력 이미지 + BBox', fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(pred_binary, cmap='gray')
    axes[1].set_title(f'예측 마스크\nIoU Score: {iou_pred.item():.3f}', fontweight='bold')
    axes[1].axis('off')

    overlay = img_vis.copy()
    overlay[pred_binary] = overlay[pred_binary] * 0.5 + np.array([255, 100, 0]) * 0.5
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title('세그멘테이션 오버레이', fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return pred_binary


# 검증 세트 첫 번째 이미지로 테스트
sample_img_path = f'./sat_dataset/val/images/{val_files[0]}'
print(f'추론 이미지: {val_files[0]}')
pred_mask = inference_single_image(
    model, sample_img_path,
    bbox=None,
    device=DEVICE,
    image_size=IMG_SIZE
)
print('✅ 추론 완료')
```

---

## 1️⃣5️⃣ 모델 저장 및 다운로드

```python
import zipfile

torch.save(model.state_dict(), './semisa_final.pth')

with zipfile.ZipFile('./semisa_results.zip', 'w') as zf:
    for f in ['./semisa_best.pth', './semisa_final.pth',
              './training_curves.png', './segmentation_results.png',
              './sample_data.png']:
        if os.path.exists(f):
            zf.write(f)

print('✅ 결과물 압축 완료: semisa_results.zip')

from google.colab import files
files.download('./semisa_results.zip')
print('✅ 다운로드 시작')
```

---

## 📝 실제 SAT 데이터 사용 가이드

### 1. 데이터 준비

```
sat_dataset/
├── images/
│   ├── flip_chip_001.bmp
│   ├── wafer6_001.bmp
│   └── ...
└── masks/
    ├── flip_chip_001.png  (같은 파일명, 이진 마스크)
    └── ...
```

### 2. 파라미터 조정

```python
DEMO_MODE  = False
IMAGE_SIZE = 1024   # 논문 원래 크기
BATCH_SIZE = 4      # GPU VRAM에 따라 조정 (A100: 16, T4: 2~4)
NUM_EPOCHS = 10000  # 논문과 동일
```

### 3. 논문 GitHub 소스 참고

- https://github.com/ThuHa96/SemiSA

### 4. LabelMe로 어노테이션

```bash
pip install labelme
labelme  # GUI 실행 후 SAT 이미지 어노테이션
```

### 5. 논문에서 보고한 성능 (Table III)

| 데이터 타입 | DSC (%) | IoU (%) |
|---|---|---|
| Flip Chip | **97.34** | **93.04** |
| Power Semiconductor | **96.09** | **93.67** |
| Wafer 6-inch | **98.22** | **96.12** |
| Wafer 12-inch | **97.74** | **95.57** |
| Transistor | **96.89** | **92.98** |
| MLCC | **97.15** | **93.87** |
