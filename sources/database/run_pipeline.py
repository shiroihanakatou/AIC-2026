import time
import logging
from pathlib import Path
from tqdm import tqdm

from frame_indexer import get_ordered_frame_list, build_frames_db
from clip_processor import process_clip_features
from object_processor import process_objects

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Khởi tạo đường dẫn động
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    processed_dir = data_dir / "processed"
    
    # Tự động tạo thư mục nếu chưa tồn tại
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("🚀 BẮT ĐẦU CHẠY PIPELINE TIỀN XỬ LÝ DỮ LIỆU AIC-2026")
    start_time = time.time()
    
    try:
        # BƯỚC 1: INDEXING
        logging.info("--- BƯỚC 1: Quét thư mục và tạo Frame Index ---")
        with tqdm(total=1, desc="Indexing Frames") as pbar:
            ordered_frames = get_ordered_frame_list(data_dir)
            db_path = processed_dir / "frames_info.db"
            build_frames_db(ordered_frames, db_path)
        logging.info(f"Đã lập chỉ mục {len(ordered_frames)} keyframes và lưu DB.")
        
        # BƯỚC 2: CLIP FEATURES
        logging.info("--- BƯỚC 2: Chuẩn hóa L2 & Nối Ma Trận CLIP ---")
        with tqdm(total=1, desc="Processing CLIP") as pbar:
            clip_master_path = processed_dir / "clip_master.npy"
            process_clip_features(ordered_frames, data_dir, clip_master_path)
        logging.info("Đã lưu ma trận CLIP Master.")
        
        # BƯỚC 3: OBJECTS FILTERING & VOCAB EXTRACTING
        logging.info("--- BƯỚC 3: Lọc Objects và Trích xuất Vocab ---")
        with tqdm(total=1, desc="Processing Objects") as pbar:
            objects_master_path = processed_dir / "objects_master.json"
            vocab_path = processed_dir / "openimages_v4_vocab.json"
            process_objects(ordered_frames, data_dir, objects_master_path, vocab_path)
        logging.info("Đã xuất JSON Object Master và từ vựng OpenImages V4.")
        
        total_time = time.time() - start_time
        logging.info(f"✅ PIPELINE HOÀN TẤT THÀNH CÔNG (Thời gian: {total_time:.2f}s)")
        
    except FileNotFoundError as fnf_err:
        logging.error(f"❌ THIẾU FILE DỮ LIỆU: {fnf_err}")
    except ValueError as val_err:
        logging.error(f"❌ DỮ LIỆU KHÔNG KHỚP: {val_err}")
    except Exception as e:
        logging.error(f"❌ LỖI HỆ THỐNG KHÔNG XÁC ĐỊNH: {e}")

if __name__ == "__main__":
    main()