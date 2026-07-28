import sys
import time
from pathlib import Path


# =====================================================================
# Cấu hình Đường dẫn Tuyệt đối & System Path (Tránh lỗi Import)
# =====================================================================
CURRENT_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE_PATH.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from sources.database.sqlite_database import SQLiteDatabase
    from sources.database.vector_database import VectorDatabase
    from sources.ingestion.loader import load_all_data
except ImportError as error:
    print(f"❌ Lỗi Import Module: {error}")
    print("💡 Hãy đảm bảo bạn đã đặt đúng tên các file/module theo yêu cầu dự án.")
    sys.exit(1)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FAISS_INDEX_PATH = PROCESSED_DIR / "clip_faiss.index"
FAISS_MAP_PATH = PROCESSED_DIR / "faiss_id_map.json"
SQLITE_DB_PATH = PROCESSED_DIR / "metadata.db"


def verify_file_created(file_path: Path) -> bool:
    """Kiểm tra file đã được tạo thành công và có dung lượng > 0 bytes hay chưa."""
    if file_path.exists() and file_path.stat().st_size > 0:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"   [OK] File tồn tại: {file_path.name} ({size_mb:.2f} MB)")
        return True

    print(f"   [FAIL] File không hợp lệ hoặc dung lượng bằng 0: {file_path}")
    return False


def _load_source_data():
    try:
        metadata_records, (vector_matrix, global_ids) = load_all_data()
    except Exception as error:
        print(f"❌ Lỗi xảy ra trong quá trình gọi load_all_data(): {error}")
        sys.exit(1)

    if not metadata_records or vector_matrix is None or len(vector_matrix) == 0:
        print(
            "❌ ERROR: Dữ liệu đầu vào rỗng (metadata_records hoặc vector_matrix "
            "không có dữ liệu). Dừng chương trình!"
        )
        sys.exit(1)

    return metadata_records, vector_matrix, global_ids


def _build_vector_database(vector_matrix, global_ids, faiss_index_path: Path, faiss_map_path: Path) -> None:
    vector_database = VectorDatabase(dimension=vector_matrix.shape[1])
    vector_database.build_index(vector_matrix=vector_matrix, global_frame_ids=global_ids)
    vector_database.save_index(index_path=str(faiss_index_path), map_path=str(faiss_map_path))


def _build_sqlite_database(metadata_records, sqlite_db_path: Path) -> None:
    sql_database = SQLiteDatabase(db_path=str(sqlite_db_path))
    sql_database.insert_records(metadata_records=metadata_records)


def _verify_output_files() -> bool:
    print("🔍 [BƯỚC 4/4] Đang kiểm tra kết quả đóng gói sản phẩm...")
    index_ok = verify_file_created(FAISS_INDEX_PATH)
    map_ok = verify_file_created(FAISS_MAP_PATH)
    db_ok = verify_file_created(SQLITE_DB_PATH)

    if index_ok and map_ok and db_ok:
        print("🎉 HOÀN THÀNH QUÁ TRÌNH ĐÓNG GÓI CƠ SỞ DỮ LIỆU THÀNH CÔNG!")
        print("---------------------------------------------------------------------")
        print("📊 TỔNG THỐNG KÊ:")
        return True

    print("❌ QUÁ TRÌNH KIỂM TRA THẤT BẠI: Một số file đầu ra chưa được tạo thành công.")
    return False


def main() -> None:
    print("=====================================================================")
    print("🚀 BẮT ĐẦU QUÁ TRÌNH ĐÓNG GÓI CƠ SỞ DỮ LIỆU (FAISS & SQLITE)")
    print("=====================================================================\n")

    start_time = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("📂 [BUỚC 1/4] Đang nạp toàn bộ dữ liệu thô từ thư mục data/...")
    step1_start = time.time()
    metadata_records, vector_matrix, global_ids = _load_source_data()
    step1_time = time.time() - step1_start
    print(
        f"✅ Đã nạp thành công {len(metadata_records)} bản ghi metadata và "
        f"{vector_matrix.shape[0]} vectors ({vector_matrix.shape[1]}D)."
    )
    print(f"⏱️  Thời gian nạp dữ liệu: {step1_time:.2f} giây.\n")

    print("⚡ [BƯỚC 2/4] Đang xây dựng và lưu FAISS Vector Database...")
    step2_start = time.time()
    _build_vector_database(vector_matrix, global_ids, FAISS_INDEX_PATH, FAISS_MAP_PATH)
    step2_time = time.time() - step2_start
    print(f"⏱️  Thời gian xử lý FAISS Index: {step2_time:.2f} giây.\n")

    print("🗄️ [BƯỚC 3/4] Đang khởi tạo và chèn dữ liệu vào SQLite Database...")
    step3_start = time.time()
    _build_sqlite_database(metadata_records, SQLITE_DB_PATH)
    step3_time = time.time() - step3_start
    print(f"⏱️  Thời gian xử lý SQLite Database: {step3_time:.2f} giây.\n")

    if not _verify_output_files():
        print("=====================================================================")
        sys.exit(1)

    total_time = time.time() - start_time
    print(f"  • Tổng số Vector (FAISS Index) : {vector_matrix.shape[0]:,} items")
    print(f"  • Tổng số Metadata (SQLite)     : {len(metadata_records):,} records")
    print(f"  • Vị trí lưu trữ dữ liệu        : {PROCESSED_DIR}")
    print(f"  • Tổng thời gian thực thi       : {total_time:.2f} giây")
    print("=====================================================================")


if __name__ == "__main__":
    main()