import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_loader import load_pdf
from src.vector_db import init_vector_db, append_documents, load_vector_db
from src.config import VECTOR_DB_PATH, DATA_DIR


def is_db_exist():
    """判断向量库是否已存在"""
    db_flag = os.path.join(VECTOR_DB_PATH, "chroma.sqlite3")
    return os.path.exists(db_flag)


def ingest_single_pdf(pdf_path: str, append: bool = False):
    """导入单个PDF到向量库"""
    if not os.path.exists(pdf_path):
        print(f"[错误] 文件不存在: {pdf_path}")
        return False

    print(f"[1/3] 正在解析PDF: {os.path.basename(pdf_path)}")
    chunks = load_pdf(pdf_path)
    print(f"[2/3] 分块完成，共 {len(chunks)} 个知识片段")

    if append and is_db_exist():
        print("[3/3] 追加模式：写入现有向量库")
        append_documents(chunks)
    else:
        print("[3/3] 新建模式：创建向量数据库")
        init_vector_db(chunks)

    print(f"✅ 导入完成: {os.path.basename(pdf_path)}")
    return True


def ingest_directory(dir_path: str, append: bool = False):
    """批量导入目录下所有PDF"""
    if not os.path.isdir(dir_path):
        print(f"[错误] 目录不存在: {dir_path}")
        return

    pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("[错误] 目录下未找到PDF文件")
        return

    print(f"找到 {len(pdf_files)} 个PDF文件，开始批量导入...\n")

    first_file = True
    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"===== [{idx}/{len(pdf_files)}] {pdf_file} =====")
        pdf_full_path = os.path.join(dir_path, pdf_file)

        if first_file and not is_db_exist():
            # 第一个文件用新建模式
            ingest_single_pdf(pdf_full_path, append=False)
            first_file = False
        else:
            # 后续文件用追加模式
            ingest_single_pdf(pdf_full_path, append=True)
        print()

    print("=" * 50)
    print(f"✅ 批量导入完成，共处理 {len(pdf_files)} 个文件")
    print(f"向量库路径: {VECTOR_DB_PATH}")


def main():
    parser = argparse.ArgumentParser(description="医疗知识库构建脚本")
    parser.add_argument("--pdf", help="单个PDF文件路径")
    parser.add_argument("--dir", help="PDF目录路径，批量导入")
    parser.add_argument("--append", action="store_true", help="追加到现有知识库")
    args = parser.parse_args()

    if not args.pdf and not args.dir:
        # 不传参数默认导入 data/medical_docs 目录
        print("未指定文件，默认导入 data/medical_docs 目录下所有PDF\n")
        ingest_directory(DATA_DIR, append=args.append)
        return

    if args.pdf:
        ingest_single_pdf(args.pdf, append=args.append)
    elif args.dir:
        ingest_directory(args.dir, append=args.append)


if __name__ == "__main__":
    main()