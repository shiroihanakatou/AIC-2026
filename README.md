# AIC-2026
Sản phẩm vòng sơ tuyển AI Challenge 2026

Chạy Python 3.11

Lấy dự án về máy:
Chạy các đoạn code dưới trong terminal, đứng tại thư mục dự án
Bước 1: Clone repo về máy
```
git clone https://github.com/shiroihanakatou/AIC-2026.git
cd AIC-2026
```
Bước 2: Tạo môi trường ảo bằng conda
    Cách 1: Dùng môi trường có sẵn
```
conda env create -f configs/environment.yml
conda activate aic-2026
```
    Cách 2: Tự tạo môi trường
```
conda create -n ten-moi-truong python=3.11 -y
conda activate ten-moi-truong
pip install -r configs/requirements.txt
```
Bước 3: Kiểm tra tính tương thích
    Cấp độ 1:
```
pip check
conda doctor
```
    Cấp độ 2:
```
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
    Cấp độ 3:
Mở một file để dán đoạn này và chạy thử (chạy trong môi trường ảo)
```
python -c "
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
"
```
