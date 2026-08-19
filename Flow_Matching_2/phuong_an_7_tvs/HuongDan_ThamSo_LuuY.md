# Hướng dẫn Điều chỉnh Siêu tham số & Lưu ý Quan trọng — Phương án 7 (TVS)

Tài liệu này chi tiết hóa toàn bộ các tham số có thể chỉnh sửa tại **Cell 1** của notebook [DiffMM_PhuongAn7_TVS_Colab.ipynb](file:///e:/NAM_BA/NGHIEN_CUU_KHOA_HOC/Flow_Matching_2/phuong_an_7_tvs/DiffMM_PhuongAn7_TVS_Colab.ipynb), ý nghĩa vật lý/toán học của chúng, và các lưu ý vận hành thực tế để tránh lỗi.

---

## 1. Bảng chi tiết các siêu tham số (Cell 1)

| Tên biến trên Colab | Tham số CLI tương ứng | Giá trị mặc định | Khoảng quét khuyến nghị | Ý nghĩa & Tác động thực tế |
| :--- | :--- | :--- | :--- | :--- |
| **`VELOCITY_MODE`** | `--velocity_mode` | `1` | `0` hoặc `1` | **Công tắc chế độ TVS:**<br>• `1`: Kích hoạt hồi quy vận tốc (TVS).<br>• `0`: Tắt TVS, quay về hồi quy dữ liệu gốc (Phương án 6). |
| **`ANCHOR_W`** | `--anchor_w` | `2.0` | `0.0` đến `5.0` | **Cường độ điểm neo thô `\alpha\sb{l}$:**<br>Quyết định độ dịch chuyển của tâm nhiễu về phía gợi ý thô sơ bộ. Bằng `0.0` tương đương tắt điểm neo hoàn toàn. |
| **`LAMBDA_X`** | `--lambda_x` | `1.0` | Cố định `1.0` | **Trọng số loss quỹ đạo chính ($x\sb{t}$):**<br>Quỹ đạo đi từ dữ liệu thực tế đến nhiễu. |
| **`LAMBDA_Y`** | `--lambda_y` | `1.0` | `0.1` đến `1.0` | **Trọng số loss quỹ đạo neo thô ($y\sb{t}$):**<br>Giám sát dòng chảy từ dữ liệu gốc hướng về điểm neo. Thử giảm xuống `0.2`-`0.5` để tăng tính tổng quát hóa. |
| **`LAMBDA_Z`** | `--lambda_z` | `1.0` | `0.1` đến `1.0` | **Trọng số loss quỹ đạo đứng yên ($z\sb{t}$):**<br>Giám sát độ ổn định quanh dữ liệu gốc. Nên quét đồng bộ với `LAMBDA_Y`. |
| **`SIGMA_MIN`** | `--sigma_min` | `1e-3` | `1e-4` đến `5e-3` | **Nhiễu nền tối thiểu `\sigma\sb{min}$:**<br>Tránh chia cho 0 ở biên và làm mịn quỹ đạo phụ. Quá nhỏ dễ gây bất ổn định số, quá lớn gây mờ kết quả. |
| **`W_CLIP`** | `--w_clip` | `50.0` | `10.0` đến `100.0` | **Chặn trên trọng số CFM:**<br>Cắt bớt các giá trị trọng số loss quá lớn ở biên thời gian (tương tự cơ chế Min-SNR). |
| **`NUM_SAMPLE_STEPS`** | `--num_sample_steps`| `0` | `0` hoặc `5` đến `15` | **Số bước suy luận rút gọn:**<br>• `0`: Tự tính bằng $60\%$ của tổng steps (ví dụ `steps=20` $\rightarrow$ dùng `12` bước).<br>• `K > 0`: Cố định đúng `K` bước ODE. |

---

## 2. Chiến lược quét tham số đề xuất (Tối ưu hóa Recall & NDCG)

Không nên thay đổi ngẫu nhiên tất cả các tham số cùng một lúc. Hãy tuân thủ quy trình 3 bước sau:

### Bước 1: Xác lập Baseline của Phương án 6 (Mốc so sánh)
*   **Cấu hình:** Đặt `VELOCITY_MODE = 0`, `ANCHOR_W = 2.0`.
*   **Mục tiêu:** Chạy thử để lấy kết quả Recall/NDCG làm mốc so sánh (Baseline). Điều này đảm bảo hạ tầng chạy đúng và bạn có số liệu đối chiếu xem cơ chế vận tốc (TVS) có thực sự cải thiện mô hình hay không.

### Bước 2: Bật TVS và tối ưu hóa lực kéo điểm neo
*   **Cấu hình:** Đặt `VELOCITY_MODE = 1`, `LAMBDA_X = 1.0`, `LAMBDA_Y = 1.0`, `LAMBDA_Z = 1.0`.
*   **Quét `ANCHOR_W`:** Thử nghiệm lần lượt các giá trị `[0.5, 1.0, 2.0, 3.0, 5.0]`.
*   *Hiện tượng cần quan sát:* Tìm ra giá trị `ANCHOR_W` cho chỉ số cao nhất. Nếu Recall/NDCG bị giảm mạnh so với Baseline, điểm neo đang bị kéo lệch hoặc quá đà.

### Bước 3: Điều hòa lực giám sát phụ (Fine-tuning Lambdas)
*   **Cấu hình:** Giữ `ANCHOR_W` tối ưu ở Bước 2.
*   **Quét cặp (`LAMBDA_Y`, `LAMBDA_Z`):** Thử nghiệm hạ thấp tầm ảnh hưởng của quỹ đạo phụ:
    1.  Cấu hình A: `LAMBDA_Y = 0.5`, `LAMBDA_Z = 0.5`
    2.  Cấu hình B: `LAMBDA_Y = 0.2`, `LAMBDA_Z = 0.2`
    3.  Cấu hình C: `LAMBDA_Y = 0.1`, `LAMBDA_Z = 0.1`
*   *Lý do:* Giảm áp lực bắt ép mạng học thuộc lòng quỹ đạo phụ tuyến tính, giúp tập trung tài nguyên học dòng chảy khuyến nghị thực tế.

---

## 3. Các lưu ý quan trọng để tránh lỗi chạy Colab

### ⚠️ Bẫy giá trị Boolean trong CLI (Đã được vá trong code)
*   **Mô tả:** Trong Python `argparse`, nếu định nghĩa tham số dạng `type=bool`, bất kể bạn truyền gì từ dòng lệnh (kể cả `--velocity_mode False` hay `--velocity_mode 0`), Python vẫn dịch chuỗi đó thành `True` (vì chuỗi không rỗng).
*   **Giải pháp:** Code trong bản fork `DiffMM-TVS` đã chuyển `--velocity_mode` thành `type=int` (nhận giá trị `0` hoặc `1`). Tránh tự sửa kiểu dữ liệu này trong [Params.py](file:///e:/NAM_BA/NGHIEN_CUU_KHOA_HOC/Flow_Matching_2/phuong_an_7_tvs/DiffMM-TVS/Params.py) về lại `type=bool`.

### ⚠️ Lỗi tràn bộ nhớ GPU (OOM - Out of Memory)
*   **Hiện tượng:** Tiến trình huấn luyện bị sụp đổ bất ngờ giữa chừng và báo lỗi *CUDA out of memory*.
*   **Nguyên nhân:** Việc tính toán song song 3 quỹ đạo vận tốc ($x\sb{t}, y\sb{t}, z\sb{t}$) thay vì 1 quỹ đạo dữ liệu đơn lẻ sẽ làm **tăng lượng dữ liệu lưu trữ tạm thời trên GPU** lên gấp khoảng 1.8 làm.
*   **Cách khắc phục:** Nếu bị OOM, hãy mở **Cell 1** giảm kích thước Batch size (`--batch` mặc định là `1024` xuống `512` hoặc `256`).

### ⚠️ Ràng buộc về dữ liệu đầu vào
*   Mô hình luôn yêu cầu cấu trúc thư mục dữ liệu chuẩn đặt trong `Datasets/<tên_dataset>`. Notebook đã tự động làm việc này thông qua liên kết Google Drive bạn cung cấp.
*   Nếu bạn đổi sang chạy dataset khác ngoài `tiktok`, hãy đảm bảo file Drive có đủ: `trnMat.pkl`, `tstMat.pkl`, `image_feat.npy`, `text_feat.npy`.
