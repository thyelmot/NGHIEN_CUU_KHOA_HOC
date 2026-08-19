# Phương án 1 — OT-linear noise scheduler cho DiffMM

Triển khai **Phương án 1** trong [`../DiffMM_FlowMatching_Optimization_Plan.md`](../DiffMM_FlowMatching_Optimization_Plan.md) (mục 5): thay lịch trình nhiễu VP-style (eq 3, DiffMM) bằng đường đi tuyến tính kiểu Optimal Transport (eq 20, Flow Matching), giữ nguyên toàn bộ phần còn lại của DiffMM.

## Nội dung folder

- **`DiffMM-OT/`** — bản fork đầy đủ của [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) đã có sẵn patch Phương án 1 áp trực tiếp vào code (`Params.py`, `Model.py`, `Main.py`). Đây là folder bạn cần **đẩy (push) lên GitHub của chính mình** — xem hướng dẫn bên dưới.
- **`DiffMM_PhuongAn1_OT_Colab.ipynb`** — notebook chạy trên Google Colab: clone code từ repo GitHub *của bạn* (không phải clone HKUDS/DiffMM gốc rồi patch lúc chạy nữa — code đã patch sẵn trong `DiffMM-OT/`), tự tải dữ liệu từ Google Drive, chạy huấn luyện, xuất bảng chỉ số tốt nhất (Recall@20 / NDCG@20 / Precision@20).

## Bước 1 — Đẩy `DiffMM-OT/` lên GitHub của bạn

Folder `DiffMM-OT/` đã là một git repo cục bộ (đã `git init` + commit sẵn 1 commit). Bạn chỉ cần tạo
repo trống trên GitHub rồi trỏ remote vào đó:

1. Vào https://github.com/new, tạo một repo mới (ví dụ tên `DiffMM-OT`), **để trống, không tick
   "Initialize with README"**.
2. Chạy các lệnh sau (thay `<your-username>` và `<repo-name>` cho đúng):

```bash
cd phuong_an_1_OT_noise_scheduler/DiffMM-OT
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

3. Copy URL repo (`https://github.com/<your-username>/<repo-name>.git`) — sẽ dùng ở Cell 1 của
   notebook.

*(Nếu dùng GitHub CLI: `gh repo create <repo-name> --private --source=. --push` chạy trong thư mục
`DiffMM-OT/` sẽ làm cả 3 bước trên trong 1 lệnh.)*

⚠️ **Chỉ dùng lệnh `git push` ở trên, KHÔNG kéo-thả folder `DiffMM-OT` lên qua giao diện web GitHub
("Add file > Upload files")** — cách đó sẽ tạo thêm 1 cấp thư mục lồng nhau (`<repo>/DiffMM-OT/Main.py`
thay vì `<repo>/Main.py`), khiến Cell 3 của notebook clone xong nhưng không tìm thấy `Main.py` ở gốc.
(Notebook đã có cơ chế tự dò và tự sửa nếu lỡ bị lồng — xem mục Troubleshooting bên dưới — nhưng push
đúng cách ngay từ đầu vẫn gọn hơn.)

## Bước 2 — Chạy notebook trên Colab

1. Mở Google Colab (colab.research.google.com) → `File > Upload notebook` → chọn `DiffMM_PhuongAn1_OT_Colab.ipynb`.
2. `Runtime > Change runtime type > Hardware accelerator > GPU`.
3. Chỉ chỉnh **Cell 1**: `GITHUB_REPO_URL` (URL repo bạn vừa push ở Bước 1), link Google Drive chứa dữ liệu, tên dataset, số epoch — rồi `Runtime > Run all`.

Dữ liệu trên Google Drive cần chứa (ở đâu đó bên trong file/thư mục chia sẻ, không bắt buộc đúng cấu trúc tuyệt đối — notebook tự dò tìm): `trnMat.pkl`, `tstMat.pkl`, `image_feat.npy`, `text_feat.npy` (và `audio_feat.npy` nếu dùng dataset `tiktok`) — đúng định dạng dữ liệu gốc của DiffMM.

## Troubleshooting

**Cell 3 báo lỗi `AssertionError: ... không đúng cấu trúc DiffMM-OT` dù clone hiện "done"** — nghĩa là
`git clone` chạy thành công (tải được dữ liệu) nhưng `Main.py` không nằm ngay trong thư mục gốc vừa
clone. Nguyên nhân phổ biến nhất: repo trên GitHub bị lồng thêm 1 cấp thư mục (`<repo>/DiffMM-OT/Main.py`
thay vì `<repo>/Main.py`) — thường do push nhầm cả folder cha, hoặc kéo-thả folder qua giao diện web
GitHub thay vì dùng `git push` (xem cảnh báo ở Bước 1). Notebook (bản mới nhất) đã tự dò tìm `Main.py`
trong toàn bộ cây thư mục vừa clone và tự điều chỉnh lại đường dẫn nếu phát hiện bị lồng — nếu vẫn báo
lỗi, kiểm tra lại đúng nội dung đã push lên repo có đúng là *nội dung bên trong* `DiffMM-OT/` hay
không (mở repo trên GitHub, `Main.py` phải nằm ngay trang gốc của repo, không nằm trong 1 thư mục con).

**Cell 7 báo `FileNotFoundError` với `train_log.txt`** — Cell 6 chưa chạy xong trong phiên hiện tại
(hoặc Colab bị ngắt kết nối/reset giữa chừng làm mất file tạm). Chạy lại Cell 6, đợi huấn luyện xong
hẳn (thấy dòng "Huấn luyện xong, đã ghi log..."), rồi chạy lại Cell 7.

## Đã kiểm chứng trước khi bàn giao

- Toàn bộ patch (`GaussianDiffusionOT`: `q_sample`, `p_mean_variance`, `SNR`) đã áp thật lên bản sao mới nhất của `Params.py`/`Model.py`/`Main.py` lấy trực tiếp từ repo gốc — biên dịch (`py_compile`) sạch, không lỗi cú pháp; các file này chính là những gì đang nằm trong `DiffMM-OT/`.
- Kiểm tra số học trên CPU (không cần GPU): điều kiện biên đúng (t=0 gần dữ liệu gốc, t=T−1 gần nhiễu thuần), `p_sample` hội tụ đúng khi denoiser hoàn hảo (sai số ~0), và trọng số loss ở mọi bước t đều hữu hạn, không "nổ" (đã kiểm tra cả trường hợp `SIGMA_MIN` rất nhỏ lẫn an toàn hơn).
- Cell "xác minh code" (Cell 5) trong notebook đã được chạy dry-run thật lên `DiffMM-OT/` cục bộ — pass toàn bộ 4 điều kiện kiểm tra.
- Cell 3 (dò tìm `Main.py`, tự sửa `REPO_DIR` khi repo bị lồng thư mục) và Cell 6/Cell 7 (chạy huấn luyện qua `subprocess`, log đường dẫn tuyệt đối, báo lỗi rõ ràng khi thất bại) đã được dry-run với repo/log giả lập cho cả 2 nhánh: chạy đúng và chạy lỗi — cả 2 đều cho thông báo đúng như thiết kế.

Phần chưa/không thể kiểm chứng ở đây (do môi trường này không có GPU và không có dữ liệu thật): chạy full training thật trên GPU với dữ liệu TikTok/Baby/Sports, và bước push/clone GitHub thật (chưa tự động push giúp bạn vì cần tài khoản GitHub của bạn). Nếu gặp lỗi khi chạy thật trên Colab, gửi lại log để debug tiếp.
