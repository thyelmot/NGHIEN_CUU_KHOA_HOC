# Giai đoạn B (Phương án 5) — Patch thật vào fork DiffMM, có công tắc bật/tắt

Đây là kết quả của **Giai đoạn B** đề xuất trong
[`../Phuong_An_5_Modality_Conditioned_OT_KeHoachChiTiet.md`](../Phuong_An_5_Modality_Conditioned_OT_KeHoachChiTiet.md)
(mục 5): viết `GaussianDiffusionModalOT` — patch thật vào bản clone mới nhất của
[HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) (tải lại từ đầu, không tái sử dụng cache cũ) — với công
tắc `kappa` (mặc định `0.0`, tương đương Phương án 1) để có thể merge an toàn.

**Đây chưa phải Giai đoạn C** (chưa đóng gói theo `Folder_Base/`, chưa tạo repo GitHub riêng, chưa có
notebook Colab) — theo đúng lộ trình đã đề ra, Giai đoạn C chỉ bắt đầu sau khi Giai đoạn B cho kết quả
hợp lý và được xác nhận.

## Các file trong folder này

Bản fork đầy đủ (không phải diff) của DiffMM, đã áp patch trực tiếp:

- **`Model.py`** — thêm class `GaussianDiffusionModalOT(GaussianDiffusion)` (cuối file). Không sửa
  `GaussianDiffusion` gốc.
- **`Main.py`** — 1 dòng import, 1 dòng khởi tạo (`GaussianDiffusionModalOT` thay vì `GaussianDiffusion`,
  đọc `args.kappa/g_min/g_max/w_clip/use_msi`), và 3 dòng gọi `p_sample` (thêm tham số `modal_embeds`,
  tái sử dụng biến `image_feats/text_feats/audio_feats` đã có sẵn trong cùng hàm `trainEpoch`).
  `training_losses` **không đổi chữ ký gọi** — `model_feats` được tái dùng làm `modal_embeds`.
- **`Params.py`** — thêm `--sigma_min`, `--w_clip`, `--kappa` (mặc định `0.0`), `--g_min`, `--g_max`,
  `--use_msi`.
- **`DataHandler.py`** — sửa sẵn lỗi môi trường đã biết (`self.trnMat.A` → `.toarray()`, scipy ≥1.14 đã
  bỏ `.A`), phòng lỗi giống các Phương án 1-4 đã gặp.
- **`verify_cpu.py`** — script kiểm chứng số học **trên chính `Model.py` thật** (không phải bản tách rời
  như Giai đoạn A), chạy trên CPU bằng cách monkeypatch `Tensor.cuda()`/`Module.cuda()` thành no-op.

## Lỗi đã phát hiện và sửa trong quá trình Giai đoạn B

Bản nháp đầu tiên của `_per_item_path` tính `mu = tau(t)^g(u,i) * x_start` (gộp luôn hệ số nhân
`x_start` vào "mu"), rồi các hàm `q_sample`/`p_mean_variance` lại nhân thêm `x_start` (hoặc
`model_output`) một lần nữa — dẫn tới 2 lỗi:

1. **Trọng số CFM sai tại mọi toạ độ `x_start[u,i]=0`** (item user chưa tương tác): vì `mu≡0` với mọi
   `t` tại toạ độ đó, sai phân `mu' ≡ 0` → trọng số suy biến về gần 0 thay vì giá trị đúng (một hàm
   thuần của hệ số `tau(t)`, độc lập với `x_start`). Phát hiện được **nhờ bài kiểm hồi quy "κ=0 phải
   trùng khít Phương án 1"** — phép so sánh trực tiếp với công thức đã kiểm chứng trước đó, không phải
   nhờ kiểm tra gradient (Giai đoạn A chỉ kiểm tra tính nhất quán nội tại của công thức tự thiết kế, nên
   **không** bắt được lỗi kiểu "công thức không khớp Theorem 3", chỉ bắt được lỗi "đạo hàm không khớp
   forward" — đây là giới hạn đã nêu rõ trong README Giai đoạn A).
2. **Lỗi tính toán bị che giấu trong `q_sample`** (vì `x_start` nhị phân `{0,1}` nên `x_start² ≡
   x_start`, làm lỗi "vô hình" ở đó) **nhưng KHÔNG bị che trong `p_mean_variance`**, nơi hệ số nhân với
   `model_output` (dự đoán liên tục, không nhị phân) — nếu không sửa, suy luận D4 (rebuild UI matrix) sẽ
   cho kết quả sai một cách âm thầm khi chạy thật trên GPU.

**Cách sửa:** tách `_per_item_path` để trả về `(tau, sigma)` — `tau` là HỆ SỐ thuần (giống hệt vai trò
`mu_coef` của Phương án 1), không nhân với `x_start`. Việc nhân `tau_t * x_start` (trong `q_sample`) hay
`tau_t * model_output` (trong `p_mean_variance`) được thực hiện đúng 1 lần, tại đúng nơi gọi — y hệt
cách `GaussianDiffusionOT` (Phương án 1) đã làm. Sau khi sửa, bài kiểm hồi quy khớp tuyệt đối
(`atol=1e-10`).

Đây là minh chứng cụ thể cho lý do Giai đoạn B (chạy trên patch thật, đối chiếu hồi quy với Phương án
1/2/3) là bước **bắt buộc**, không thể bỏ qua dù Giai đoạn A đã "pass" — hai bài kiểm tra bắt các lớp
lỗi khác nhau.

## Kết quả kiểm chứng (chạy `python verify_cpu.py` để tái tạo, không cần GPU)

| Bài kiểm tra | Kết quả |
|---|---|
| **Hồi quy**: `kappa=0` → `tau, sigma, cfm_weight` so với Phương án 1/2/3 | Trùng khít tuyệt đối (`atol=1e-10` cho tau/sigma, `atol=1e-8` cho cfm_weight) |
| `training_losses`: shape, không NaN/Inf, qua 3 giá trị κ (0, 1, 5) | Đạt; `use_msi=False` → `gc_loss≡0`; `use_msi=True` → `gc_loss≠0` đúng như thiết kế |
| `p_sample` tất định, model hoàn hảo, κ=0, `sampling_steps=0` | Tái tạo chính xác `x_start` (đúng hành vi kỳ vọng của "model hoàn hảo") |
| `p_sample` κ=3 so với κ=0 (cùng seed, cùng model nhiễu) | Không NaN/Inf; **khác rõ** κ=0 (xác nhận đường đi modal thực sự có tác dụng, không phải no-op) |
| Biên: user không tương tác item nào (`x_start` toàn 0 ở 1 hàng) | Không NaN/Inf (centroid rơi về 0 được xử lý an toàn nhờ `clamp(eps)`) |
| Biên: κ rất lớn (bão hoà `clamp(g_min, g_max)`) | Không NaN/Inf |
| Quy mô gần thực tế (batch=1024, num_items=7050, giống Baby dataset) | `training_losses` ~4.4s, `p_sample` (T=5, steps=0) ~1.1s trên CPU/float64 — GPU/float32 thật sẽ nhanh hơn nhiều; không NaN/Inf |

## Kết luận

Patch đã qua được bài kiểm tra **quan trọng nhất của Giai đoạn B**: `kappa=0` (giá trị mặc định trong
`Params.py`) cho kết quả **trùng khít tuyệt đối** với Phương án 1/2/3 đã kiểm chứng trước đó — an toàn để
merge vào codebase mà không phá vỡ hành vi hiện có. Việc bật `kappa>0` cho kết quả khác biệt rõ ràng và
không gây NaN/Inf ở mọi kịch bản biên đã thử.

**Giới hạn của Giai đoạn B (nói rõ để không hiểu nhầm phạm vi đã kiểm chứng):** vẫn dùng `model` giả lập
(hàm biến đổi đơn giản `x_t*scale+offset`), **chưa** chạy qua mạng `Denoise` thật, **chưa** chạy với dữ
liệu thật (TikTok/Baby/Sports) trên GPU, **chưa** có bất kỳ số liệu Recall/NDCG nào, và **chưa** đánh giá
chi phí bộ nhớ GPU thực tế khi `batch=1024` × `num_items` lớn × giữ đồng thời nhiều mảng `(batch, T,
num_items)` (ước tính ~vài trăm MB–1GB cho Baby/Sports, cần theo dõi khi chạy thật, có thể cần giảm
`--batch` nếu tràn bộ nhớ GPU). Bước tiếp theo (nếu người dùng xác nhận) là đóng gói theo
`Folder_Base/` (Giai đoạn C): tạo repo GitHub riêng, notebook Colab, chạy thật với dữ liệu thật trên GPU.
