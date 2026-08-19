import json

filepath = r'e:\NAM_BA\NGHIEN_CUU_KHOA_HOC\phuong_an_1_OT_noise_scheduler\DiffMM_PhuongAn1_OT_Colab.ipynb'
with open(filepath, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Replace GITHUB_REPO_URL
        if 'GITHUB_REPO_URL = ""' in source:
            source = source.replace('GITHUB_REPO_URL = ""', 'GITHUB_REPO_URL = "https://github.com/thyelmot/NGHIEN_CUU_KHOA_HOC.git"')
            cell['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

        # Fix cloning logic
        if 'REPO_DIR = "DiffMM-OT"' in source and '!git clone {GITHUB_REPO_URL} {REPO_DIR}' in source:
            new_source = """# ============================================================
# CELL 3 — CLONE CODE TỪ REPO GITHUB CỦA BẠN
# ============================================================
import os

if not os.path.isdir("NGHIEN_CUU_KHOA_HOC"):
    !git clone {GITHUB_REPO_URL} NGHIEN_CUU_KHOA_HOC
else:
    print(f"Thư mục 'NGHIEN_CUU_KHOA_HOC' đã tồn tại, bỏ qua bước clone (dùng lại code đã có).")

REPO_DIR = "NGHIEN_CUU_KHOA_HOC/phuong_an_1_OT_noise_scheduler/DiffMM-OT"

assert os.path.isdir(REPO_DIR) and os.path.exists(os.path.join(REPO_DIR, "Main.py")), (
    "Clone thất bại hoặc repo không đúng cấu trúc — kiểm tra lại GITHUB_REPO_URL ở Cell 1 "
    "(repo phải công khai hoặc bạn đã đăng nhập git trên Colab)."
)
print(sorted(os.listdir(REPO_DIR)))"""
            cell['source'] = [line + ('\n' if i < len(new_source.split('\n')) - 1 else '') for i, line in enumerate(new_source.split('\n'))]

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
