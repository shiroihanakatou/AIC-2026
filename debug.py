import time
import sys
from pathlib import Path

# Đưa module sources vào sys path để nhận diện các gói bên trong
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from sources.retrieval.query_processor import QueryProcessor
from sources.retrieval.search_engine import SearchEngine

def print_result_table(results: list):
    """Sử dụng format string để render bảng kết quả gọn gàng trực tiếp."""
    if not results:
        print("Không tìm thấy kết quả phù hợp.")
        return
        
    header = f"| {'Rank':<4} | {'Global ID':<9} | {'Score':<7} | {'S_CLIP':<7} | {'S_OBJ':<7} | {'Video ID':<9} | {'PTS Time':<8} | {'Frame Path':<35} |"
    sep = "-" * len(header)
    
    print(sep)
    print(header)
    print(sep)
    
    for i, r in enumerate(results):
        print(f"| {i+1:<4} | {r['global_frame_id']:<9} | {r['score']:<7.4f} | {r['s_clip']:<7.4f} | {r['s_obj']:<7.4f} | {r['video_id']:<9} | {r['pts_time']:<8.2f} | {r['frame_path']:<35} |")
    print(sep)

def main():
    processed_dir = project_root / "data" / "processed"
    vocab_path = processed_dir / "openimages_v4_vocab.json"
    
    print("⏳ Đang nạp Search Engine & SQLite vào RAM (Có thể mất vài giây)...")
    t_boot = time.perf_counter()
    qp = QueryProcessor(vocab_path)
    se = SearchEngine(processed_dir)
    print(f"✅ Hệ thống đã sẵn sàng! (Thời gian Boot: {(time.perf_counter() - t_boot):.2f}s)\n")
    
    while True:
        try:
            query = input("\n🔍 Nhập truy vấn bằng Tiếng Anh (hoặc 'q' để thoát): ").strip()
            if not query:
                continue
            if query.lower() in ['q', 'quit', 'exit']:
                print("Tạm biệt!")
                break
                
            w_clip_str = input("🎛️  Nhập trọng số w_clip (Enter mặc định 0.7): ").strip()
            w_clip = float(w_clip_str) if w_clip_str else 0.7
            
            w_obj_str = input("🎛️  Nhập trọng số w_obj (Enter mặc định 0.3): ").strip()
            w_obj = float(w_obj_str) if w_obj_str else 0.3
            
            top_k_str = input("🔢 Nhập số lượng Top K kết quả (Enter mặc định 10): ").strip()
            top_k = int(top_k_str) if top_k_str else 10
            
            # 1. Query Processing
            t0 = time.perf_counter()
            query_data = qp.process_query(query)
            t1 = time.perf_counter()
            
            # 2. Searching
            results = se.search(query_data, w_clip=w_clip, w_obj=w_obj, top_k=top_k)
            t2 = time.perf_counter()
            
            # 3. Kết quả hiển thị
            print("\n" + "="*80)
            print(f"📄 Original Text : '{query}'")
            print(f"🤖 Extracted E_q : {query_data['entities']} (Trọng lượng Clip/Obj: {w_clip}/{w_obj})")
            print("="*80)
            
            print_result_table(results)
            
            print(f"\n[⏱️ LATENCY]")
            print(f" - NLP & CLIP Encoding : {(t1 - t0)*1000:>6.2f} ms")
            print(f" - Search & Fusion     : {(t2 - t1)*1000:>6.2f} ms")
            print(f" - Total Time          : {(t2 - t0)*1000:>6.2f} ms")
            
        except Exception as e:
            print(f"❌ Lỗi hệ thống: {e}")

if __name__ == "__main__":
    main()