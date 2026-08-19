# Hướng dẫn Sử dụng Optuna Tối ưu hóa Siêu tham số — Phương án 7 (TVS)

Tài liệu này hướng dẫn chi tiết quy trình sử dụng **CELL 8 (TÙY CHỌN)** trong notebook [DiffMM_PhuongAn7_TVS_Colab.ipynb](file:///e:/NAM_BA/NGHIEN_CUU_KHOA_HOC/Flow_Matching_2/phuong_an_7_tvs/DiffMM_PhuongAn7_TVS_Colab.ipynb) để tự động dò tìm bộ siêu tham số tốt nhất cho mô hình TVS bằng thuật toán Bayesian Optimization (TPE Sampler).

---

## 1. Tại sao nên dùng Optuna thay vì thử nghiệm thủ công?
*   **Dò tìm thông minh (Bayesian Optimization):** Thay vì thử ngẫu nhiên hoặc quét lưới (Grid Search) mất thời gian, Optuna phân tích lịch sử của các lượt chạy trước để tự động gợi ý bộ tham số tiếp theo có khả năng cải thiện Recall cao hơn.
*   **Quét đồng thời nhiều tham số:** Quét cùng lúc lực kéo neo (`anchor_w`), các trọng số loss quỹ đạo tam giác (`lambda_y`, `lambda_z`) và nhiễu nền (`sigma_min`).

---

## 2. Quy trình 4 bước thực hiện trên Google Colab

### Bước 1: Khởi động phiên làm việc Colab
1.  Tải file notebook mới nhất `DiffMM_PhuongAn7_TVS_Colab.ipynb` từ máy tính lên Colab.
2.  Bật GPU: `Runtime > Change runtime type > Hardware accelerator > GPU`.
3.  Chạy liên tục từ **Cell 1 đến Cell 5** để thiết lập môi trường, tải dữ liệu, và xác minh mã nguồn đã patch thành công.

### Bước 2: Cấu hình tham số quét của Optuna (Tại Cell 8)
Trước khi chạy Cell 8, bạn có thể điều chỉnh 2 biến số quan trọng ở đầu cell:
*   `N_TRIALS = 10`: Tổng số lượt huấn luyện thử nghiệm. (Tăng lên `20` hoặc `30` nếu bạn có nhiều thời gian và muốn tìm kết quả tối ưu sâu hơn).
*   `OPTUNA_EPOCHS = 25`: Số epoch chạy thử cho mỗi lượt. 
    > 📌 **Lưu ý:** Giá trị này đã được đặt mặc định là **`25`** để đảm bảo bao phủ hoàn toàn đỉnh hội tụ thực tế của bạn (thường xuất hiện trong khoảng từ **Epoch 15 đến 25**). Không nên giảm dưới `25` vì mô hình sẽ chưa kịp hội tụ, dẫn đến kết quả đánh giá bị sai lệch.

### Bước 3: Chạy Cell 8 để dò tìm tự động
*   Bấm nút Chạy Cell 8. 
*   Hệ thống sẽ tự động cài đặt thư viện `optuna` (nếu chưa có) và tiến hành huấn luyện liên tục `N_TRIALS` lượt.
*   Mỗi lượt kết thúc, màn hình sẽ in ra thông tin:
    `[Trial X] Đang chạy thử nghiệm với: anchor_w=..., lambda_y=..., lambda_z=..., sigma_min=... -> Recall@20 tốt nhất đạt: ...`

### Bước 4: Áp dụng kết quả tối ưu vào huấn luyện đầy đủ
Khi kết thúc quá trình quét (`TỐI ƯU HÓA HOÀN TẤT!`), Optuna sẽ hiển thị kết quả tốt nhất ở cuối cell dưới dạng:
```python
Bộ tham số tốt nhất: {'anchor_w': 2.45, 'lambda_y': 0.35, 'lambda_z': 0.42, 'sigma_min': 0.0008}
Recall@20 tốt nhất đạt: 0.15420
```
Bạn chỉ cần:
1.  Quay ngược lên **Cell 1**, điền các giá trị tối ưu này vào các biến tương ứng (`ANCHOR_W`, `LAMBDA_Y`, `LAMBDA_Z`, `SIGMA_MIN`).
2.  Đặt lại `NUM_EPOCHS = 50` (hoặc số epoch bạn muốn chạy thực tế).
3.  Chạy **Cell 6 & Cell 7** để huấn luyện và xuất bảng kết quả + biểu đồ PDF cuối cùng với cấu hình tối ưu nhất.

---

## 3. Không gian tìm kiếm mặc định (Có thể tùy biến trong code)

Nếu bạn muốn mở rộng hoặc thu hẹp dải tìm kiếm, bạn có thể chỉnh sửa các dòng lệnh `trial.suggest_...` bên trong hàm `objective(trial)` ở Cell 8:

*   **`anchor_w`** (`trial.suggest_float("anchor_w", 0.5, 5.0)`): Lực kéo neo thô $\alpha\sb{l}$.
*   **`lambda_y`** (`trial.suggest_float("lambda_y", 0.1, 1.0)`): Trọng số loss quỹ đạo neo thô.
*   **`lambda_z`** (`trial.suggest_float("lambda_z", 0.1, 1.0)`): Trọng số loss quỹ đạo đứng yên.
*   **`sigma_min`** (`trial.suggest_float("sigma_min", 1e-4, 5e-3, log=True)`): Nhiễu nền tối thiểu $\sigma\sb{min}$.

---

## 4. Các lưu ý quan trọng để tránh lỗi
*   **Đọc đúng chỉ số mục tiêu:** Hàm `objective` của Optuna phân tích log và trích xuất dòng `Best epoch : ... Recall : ...`. Do đó, Optuna sẽ tối ưu hóa dựa trên chỉ số **Recall@20**.
*   **Quản lý bộ nhớ GPU:** Mỗi lượt thử nghiệm sẽ khởi chạy một tiến trình Python độc lập và tự giải phóng bộ nhớ GPU sau khi kết thúc. Tuy nhiên, nếu bạn nhận được lỗi *CUDA Out of Memory*, hãy giảm `batch` trong Cell 1 xuống `512` hoặc `256` trước khi bắt đầu tối ưu hóa.
*   **Xóa log tự động:** Đoạn code đã được thiết kế tự động xóa file `train_log.txt` cũ trước mỗi lượt chạy thử để đảm bảo Optuna luôn đọc kết quả chính xác của lượt chạy hiện tại.
