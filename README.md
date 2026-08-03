# AIC-2026

Sản phẩm vòng sơ tuyển AI Challenge 2026.

Project này chạy trên Python 3.11.

## Tiến độ hiện tại

- Đã hoàn thiện khung pipeline tiền xử lý dữ liệu: quét keyframes, lập chỉ mục frame vào SQLite, nối ma trận CLIP và trích xuất object/vocab vào `data/processed/`.
- Đã hoàn thiện khung truy xuất: `QueryProcessor` mã hóa truy vấn bằng OpenCLIP và trích xuất `E_q`, còn `SearchEngine` kết hợp điểm CLIP với điểm object theo chiến lược top-candidate hai bước.
- `debug.py` đã có thể dùng làm script kiểm tra truy vấn tương tác; với prompt thử `a man wearing a hat and shirt standing next to cattle inside a building`, hệ thống đã được đưa vào trạng thái chạy thử để rà luồng truy xuất end-to-end.
- Phần cần tiếp tục theo dõi là mức độ đầy đủ của dữ liệu sinh ra trong `data/processed/` và độ ổn định khi chạy nhiều truy vấn liên tiếp trong chế độ tương tác.

## Cấu trúc thư mục

### `data/`

| Thư mục | Mô tả |
| --- | --- |
| `data/keyframes/` | Chứa ảnh keyframe của từng video, tổ chức theo cây thư mục `data/keyframes/<video_id>/<keyframe_id>.jpg`. |
| `data/map-keyframes/` | Chứa các file CSV ánh xạ keyframe với thời gian và frame gốc, theo `data/map-keyframes/<video_id>.csv`. |
| `data/clip-features/` | Chứa ma trận NumPy đặc trưng CLIP cho từng video, theo `data/clip-features/<video_id>.npy`. |
| `data/metadata/` | Chứa metadata cấp video như tiêu đề, kênh đăng tải, thời lượng, ngày đăng và liên kết YouTube. |
| `data/objects/` | Chứa kết quả phát hiện object theo từng keyframe, theo `data/objects/<video_id>/<keyframe_id>.json`. |
| `data/processed/` | Thư mục đầu ra sau khi chạy pipeline tiền xử lý, nơi lưu các artifact dùng cho truy xuất dữ liệu. |

```bash
               ┌────────────────────────┐
               │    data/metadata/      │  (Thông tin YouTube / Video gốc)
               │   <video_id>.json      │
               └───────────┬────────────┘
                           │ 1 - N
                           ▼
┌───────────────────────────────────────────────────┐
│                 data/keyframes/                   │  (Ảnh keyframe của từng video)
│     <video_id>/<keyframe_id>.jpg                  │
└───────────┬───────────────────────────┬───────────┘
            │                           │
            │ 1 - 1                     │ 1 - 1
            ▼                           ▼
┌─────────────────────────┐  ┌────────────────────────────┐
│  data/clip-features/    │  │    data/objects/           │
│     <video_id>.npy      │  │ <video_id>/<keyframe>.json │
│ (Vector CLIP cho từng   │  │ (Nhãn, điểm tin cậy,       │
│  keyframe)              │  │  hộp giới hạn)             │
└─────────────────────────┘  └────────────────────────────┘
            ▲
            │ cùng thứ tự keyframe
            │
┌───────────────────────────────────────────────────┐
│               data/map-keyframes/                 │  (Ánh xạ keyframe với frame gốc)
│                <video_id>.csv                     │
└───────────────────────────────────────────────────┘
```

### Cách tạo mock file cho keyframes

Trong terminal, đứng tại thư mục `AIC-2026/data/keyframes/` và chạy lệnh sau:

```bash
xargs -I {} -P 8 touch "{}" < manifest.txt
```

Lưu ý quá trình chạy tốn nhiều thời gian. Cần chờ đến khi terminal tự ngắt, hoặc số lượng mock file đúng bằng 177322, hoặc số lượng folder con đúng bằng 873.

## Hướng dẫn cài đặt

Để lấy dự án về máy, hãy chạy các lệnh dưới đây trong terminal, đứng tại thư mục dự án.

### Bước 1: Clone repository về máy

```bash
git clone https://github.com/shiroihanakatou/AIC-2026.git
cd AIC-2026
```

### Bước 2: Tạo môi trường ảo bằng conda

#### Cách 1: Dùng môi trường có sẵn

```bash
conda env create -f configs/environment.yml
conda activate aic-2026
```

#### Cách 2: Tự tạo môi trường

```bash
conda create -n ten-moi-truong python=3.11 -y
conda activate ten-moi-truong
pip install -r configs/requirements.txt
```

### Bước 3: Kiểm tra tính tương thích

#### Cấp độ 1

```bash
pip check
conda doctor
```

#### Cấp độ 2

```bash
python -c "
import torch
import torchvision
import open_clip
import faiss
import pandas as pd
import numpy as np
import cv2
import PIL
import fastapi
import uvicorn

print('=== KIỂM TRA IMPORT THÀNH CÔNG ===')
print('PyTorch Version :', torch.__version__)
print('CUDA Available  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name     :', torch.cuda.get_device_name(0))
    print('CUDA Version    :', torch.version.cuda)
print('OpenCV Version  :', cv2.__version__)
print('FAISS Version   :', faiss.__version__)
print('OpenCLIP Version:', open_clip.__version__)
print('===================================')
"
```

#### Cấp độ 3

Mở một file Python để dán đoạn code dưới đây và chạy thử trong môi trường ảo.

```python
import torch
import open_clip
import faiss
import numpy as np
import cv2
from PIL import Image

print("1. Đang khởi tạo mô hình CLIP...")
# Load model CLIP nhẹ để test
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"2. Mô hình đã load thành công trên thiết bị: {device}")

print("3. Test xử lý Ảnh giả lập...")
# Tạo 1 ảnh ngẫu nhiên bằng OpenCV
dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
pil_img = Image.fromarray(dummy_img)
image_input = preprocess(pil_img).unsqueeze(0).to(device)

print("4. Test Trích xuất Vector (Feature Extraction)...")
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)

# Chuyển vector sang dạng Numpy float32 cho FAISS
vector_np = image_features.cpu().numpy().astype(np.float32)

print("5. Test nạp Vector vào FAISS Index...")
dimension = vector_np.shape[1]
# Tạo FAISS Index đơn giản
index = faiss.IndexFlatIP(dimension)
index.add(vector_np)

print(f"-> Hoàn tất! FAISS đã lưu {index.ntotal} vector. Môi trường hoạt động 100% hoàn hảo!")
```
