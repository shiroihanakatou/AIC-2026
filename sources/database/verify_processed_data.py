import sys
from pathlib import Path

import numpy as np

# =====================================================================
# Cấu hình Đường dẫn Tuyệt đối & System Path (Tránh lỗi Import)
# =====================================================================
CURRENT_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE_PATH.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sources.database.sqlite_database import SQLiteDatabase
from sources.database.vector_database import VectorDatabase

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_PATH = PROCESSED_DIR / "clip_faiss.index"
MAP_PATH = PROCESSED_DIR / "faiss_id_map.json"
DB_PATH = PROCESSED_DIR / "metadata.db"


def _load_vector_database(index_path: Path, map_path: Path) -> VectorDatabase:
    vector_database = VectorDatabase()
    vector_database.load_index(str(index_path), str(map_path))
    return vector_database


def _run_sample_query(vector_database: VectorDatabase) -> list[tuple[str, float]]:
    query_vector = np.random.randn(1, vector_database.dimension).astype(np.float32)
    return vector_database.search(query_vector, top_k=3)


def _load_sqlite_records(global_ids: list[str]) -> list[dict[str, object]]:
    sql_database = SQLiteDatabase(db_path=str(DB_PATH))
    return sql_database.get_records_by_global_ids(global_ids)


def _print_sample_record(record: dict[str, object]) -> None:
    print(f"   • Sample Global ID : {record['global_id']}")
    print(f"   • Sample Title     : {record['video_title']}")
    print(f"   • Sample Frame Path: {record['image_path']}")
    print(f"   • Sample Objects   : {record['detection_class_names']}")


def verify_pipeline() -> None:
    print("--- 🔍 BẮT ĐẦU KIỂM TRẢ DỮ LIỆU ĐÃ ĐÓNG GÓI ---")

    vector_database = _load_vector_database(INDEX_PATH, MAP_PATH)
    print(
        f"1. FAISS Index: Load thành công {vector_database.index.ntotal} vectors "
        f"({vector_database.dimension}D)."
    )

    search_results = _run_sample_query(vector_database)
    print(f"2. Query FAISS Thử nghiệm: Trả về {len(search_results)} kết quả Top-3.")

    top_ids = [global_id for global_id, _score in search_results]
    retrieved_records = _load_sqlite_records(top_ids)

    print(f"3. Truy xuất SQLite: Trả về {len(retrieved_records)} bản ghi tương ứng.")
    if retrieved_records:
        _print_sample_record(retrieved_records[0])

    assert vector_database.index.ntotal == len(vector_database.id_map), (
        "❌ Lỗi: FAISS Index và ID Map không lệch nhau!"
    )
    assert len(retrieved_records) == len(top_ids), (
        "❌ Lỗi: Không tìm thấy bản ghi trong SQLite!"
    )

    print("\n✅ TẤT CẢ FILE TRONG data/processed/ ĐẠT CHUẨN 100%!")


if __name__ == "__main__":
    verify_pipeline()