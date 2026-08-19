# Giai đoạn A (Phương án 5) — Kiểm tra gradient bằng số học trên dữ liệu giả lập

Đây là kết quả của **Giai đoạn A** đề xuất trong
[`../Phuong_An_5_Modality_Conditioned_OT_KeHoachChiTiet.md`](../Phuong_An_5_Modality_Conditioned_OT_KeHoachChiTiet.md)
(mục 5): kiểm tra công thức **tự thiết kế** (CT-5.1 → CT-5.7 — không có sẵn trong 2 bài báo gốc) trên
dữ liệu giả lập, **hoàn toàn bên ngoài codebase DiffMM** (không cần GPU, không cần dataset thật, không
đụng vào repo `HKUDS/DiffMM`) — bước bắt buộc trước khi tin tưởng bất kỳ kết quả huấn luyện nào từ
Phương án 5.

## File

- **`gradcheck_pa5.py`** — cài đặt độc lập (PyTorch thuần, `float64`) của:
  - `modal_affinity` (CT-5.1 + CT-5.2) — độ phù hợp modal giữa item và gu từng user.
  - `per_item_schedule` (CT-5.3 → CT-5.5 + sai phân lùi) — `μₜ, σₜ` theo từng cặp `(user, item)`.
  - `pa5_loss` (CT-5.7) — CFM loss có trọng số theo từng tọa độ, áp dụng trọng số **trước** khi gộp
    (mean) qua chiều item (khác thứ tự với Phương án 2).
  - `finite_diff_grad` — đối chiếu gradient độc lập bằng sai phân hữu hạn trung tâm (không dùng lại bộ
    máy autograd của chính công thức đang kiểm tra, để tránh "tự kiểm tra chính mình").

## Kết quả (chạy `python gradcheck_pa5.py` để tái tạo)

| Kịch bản kiểm tra | Kết quả |
|---|---|
| Trường hợp thường (κ=1.0, batch nhỏ) | Loss hữu hạn; gradient (autograd) hữu hạn cho cả `modal_embeds` lẫn `model_output`; **khớp tuyệt đối** với sai phân hữu hạn độc lập (sai lệch `0.0`) |
| Biên: 1 user không có item nào trong `α₀` (test epsilon trong `normalize`) | Không NaN — `centroid` rơi về vector `0` được xử lý an toàn nhờ `clamp(min=eps)` trước khi chia; gradient vẫn khớp tuyệt đối |
| Biên: κ rất lớn (ép nhiều item vào vùng bị `clamp(g_min, g_max)`) | Không NaN/Inf; gradient vẫn hữu hạn và khớp sai phân hữu hạn (vùng bị clamp cho gradient ≈0 tại đúng những item đó — đúng hành vi mong đợi của `clamp`, không phải lỗi) |
| Quy mô gần thực tế (batch=64, 500 item, dữ liệu thưa ~2% khác 0) | Forward + backward hữu hạn, không NaN/Inf (không chạy sai phân hữu hạn ở quy mô này vì quá chậm — sai phân hữu hạn chỉ dùng để *kiểm chứng công thức* trên dữ liệu nhỏ, không dùng để huấn luyện) |
| κ=0 | `μ, σ, w` **giống hệt tuyệt đối** dù `modal_embeds` hoàn toàn khác nhau (đúng chứng minh đại số: κ=0 ⟹ g=1 mọi nơi) — và **trùng khít** công thức OT-linear thuần của Phương án 1 |

## Kết luận

Công thức tự thiết kế của Phương án 5 (mục 2, bản kế hoạch chi tiết) **đã vượt qua kiểm tra gradient
bằng số học** ở mọi kịch bản đã thử, bao gồm cả 2 trường hợp biên số học quan trọng nhất (user không
tương tác gì, và bão hoà `clamp`). Đây là điều kiện cần (không phải điều kiện đủ) để tiến sang
**Giai đoạn B** (patch có công tắc bật/tắt vào 1 bản fork thật, mặc định `κ=0` để tương thích ngược với
Phương án 1) như đã đề xuất trong bản kế hoạch chi tiết.

**Giới hạn của Giai đoạn A (nói rõ để không hiểu nhầm phạm vi đã kiểm chứng):** đây chỉ là kiểm tra
**công thức toán đúng về mặt đạo hàm**, dùng biến tự do thay cho output mạng thật (`model_output`
không đi qua mạng `Denoise` thật) và dữ liệu hoàn toàn ngẫu nhiên (không phải phân phối tương tác thật
của TikTok/Baby/Sports). **Chưa** kiểm chứng: hành vi khi ghép vào toàn bộ pipeline `GaussianDiffusion`
thật (D1/D4 dùng `q_sample`/`p_mean_variance` theo per-item, top-k rebuild ở D4 có hợp lý không), và
**chưa** có bất kỳ số liệu Recall/NDCG nào — đúng như giới hạn đã nêu ở mục 7 của bản kế hoạch chi tiết.
