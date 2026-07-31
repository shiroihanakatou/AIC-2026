# AIC-2026

Sản phẩm vòng sơ tuyển AI Challenge 2026.

Project này chạy trên Python 3.11.

## Tiến độ dự án

> Cập nhật lần cuối: 29-07-2026

### Tổng quan trạng thái

Dự án hiện đang ở giai đoạn xây dựng nền tảng dữ liệu và kho lưu trữ vector cho hệ thống truy xuất keyframe/video. Các thành phần cốt lõi đã có thể chạy được và tạo ra dữ liệu đã xử lý ở thư mục `data/processed/`.

### Bảng theo dõi

| Hạng mục | Trạng thái | Ghi chú |
| --- | --- | --- |
| Khởi tạo cấu trúc dự án | ✅ Hoàn thành | README, cấu trúc thư mục và môi trường Python đã được chuẩn bị. |
| Loader dữ liệu từ `data/` | ✅ Hoàn thành | Hỗ trợ đọc song song keyframes, metadata, objects và CLIP features. |
| Xây dựng pipeline tiền xử lý | ✅ Hoàn thành | Dữ liệu được ghép thành bản ghi phẳng để phục vụ lưu trữ và truy vấn. |
| Tạo FAISS Vector Database | ✅ Hoàn thành | Đã có file index và map ID ở `data/processed/`. |
| Tạo SQLite Metadata Database | ✅ Hoàn thành | Đã có file `metadata.db` chứa thông tin keyframe và object. |
| Kiểm tra pipeline sau khi đóng gói | ✅ Hoàn thành | Có script kiểm tra dữ liệu đã đóng gói và tương tác với cả FAISS lẫn SQLite. |
| Retrieval engine / semantic search | ⏳ Đang chờ phát triển | Chưa có module truy vấn nâng cao cho hệ thống tìm kiếm thực tế. |
| API services | ⏳ Đang chờ phát triển | Chưa có endpoint phục vụ cho frontend hoặc hệ thống truy vấn. |
| UX/UI | ⏳ Đang chờ phát triển | Chưa có giao diện người dùng. |
| Agent / LLM workflow | ⏳ Đang chờ phát triển | Các module `sources/agent/` vẫn chưa được triển khai. |

### Mốc phát triển tiếp theo

1. Hoàn thiện module retrieval để thực hiện tìm kiếm vector/query trên dữ liệu đã đóng gói.
2. Xây dựng API services để expose kết quả truy vấn cho UI.
3. Thiết kế giao diện người dùng cho trải nghiệm tìm kiếm và xem kết quả.
4. Mở rộng sang agent/LLM để hỗ trợ truy vấn tự nhiên và phân tích ngữ cảnh.

## Cấu trúc thư mục

### Nhóm `sources/`

| Thư mục | Mô tả |
| --- | --- |
| `sources/agent/` | Sub-system dành cho Trợ lý AI (LLM Planner, Query Expansion, Conversational Agent phục vụ KISC/VideoQA). |
| `sources/database/` | Quản lý kết nối, lưu trữ và truy vấn Vector Database cũng như Metadata Database. |
| `sources/ingestion/` | Chứa các script đọc, làm sạch và chuẩn hóa dữ liệu từ `data/` (keyframes, clipfeatures, metadata, objects). |
| `sources/retrieval/` | Lõi thuật toán tìm kiếm (Search engine, Cosine similarity, Fusion logic, Re-ranking). |
| `sources/services/` | Dựng các endpoint API phục vụ dữ liệu cho UI. |
| `sources/uxui/` | Mã nguồn giao diện người dùng (Streamlit / Gradio / React). |

### Nhóm `data/`

| Thư mục | Mô tả |
| --- | --- |
| `data/keyframes/` | Chứa ảnh keyframe của từng video, tổ chức theo cây thư mục `data/keyframes/<video_id>/<keyframe_id>.jpg`. |
| `data/map-keyframes/` | Chứa các file CSV ánh xạ keyframe với thời gian và frame gốc, theo `data/map-keyframes/<video_id>.csv`. |
| `data/clip-features/` | Chứa ma trận NumPy đặc trưng CLIP cho từng video, theo `data/clip-features/<video_id>.npy`. |
| `data/metadata/` | Chứa metadata cấp video như tiêu đề, kênh đăng tải, thời lượng, ngày đăng và liên kết YouTube. |
| `data/objects/` | Chứa kết quả phát hiện object theo từng keyframe, theo `data/objects/<video_id>/<keyframe_id>.json`. |
| `data/processed/` | Thư mục đầu ra sau khi chạy pipeline tiền xử lý, nơi lưu các artifact như FAISS index, ID map và SQLite database dùng cho truy xuất dữ liệu. |

Phần `data/` được rút gọn từ nội dung trong từng file `.gitkeep`, tập trung vào mô tả tổng quát vai trò của mỗi thư mục và quan hệ giữa keyframes, map-keyframes, clip-features, metadata, objects và thư mục đầu ra `processed`.

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
