import os
import re
import shutil
import random


def filter_pdfs(source_dir, dest_dir, cutoff_year, n):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    pdfs = [f for f in os.listdir(source_dir) if f.endswith(".pdf")]

    valid_pdfs = []
    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")

    for pdf in pdfs:
        match = date_pattern.search(pdf)
        if match and "MISSING" not in pdf:
            date_str = match.group()
            year = int(date_str.split("-")[0])
            if year > cutoff_year:
                valid_pdfs.append(pdf)

    selected_pdfs = random.sample(valid_pdfs, min(n, len(valid_pdfs)))

    for pdf in selected_pdfs:
        src_path = os.path.join(source_dir, pdf)
        dest_path = os.path.join(dest_dir, pdf)
        shutil.copy2(src_path, dest_path)

    print(f"Copied {len(selected_pdfs)} files to {dest_dir}")


if __name__ == "__main__":
    SOURCE_DIR = "./data/pdfs/"
    DEST_DIR = "./selected_pdfs"
    CUTOFF_YEAR = 2010
    N = 1000

    filter_pdfs(SOURCE_DIR, DEST_DIR, CUTOFF_YEAR, N)
