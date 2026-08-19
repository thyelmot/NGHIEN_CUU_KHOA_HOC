# DiffMM-CFM — Phương án 2 (CFM loss weighting)

Đây là bản fork của [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) (paper *DiffMM: Multi-Modal
Diffusion Model for Recommendation*, ACM MM 2024) với **Phương án 2** trong kế hoạch chi tiết
[`Phuong_An_2_CFM_Loss_KeHoachChiTiet.md`](../Phuong_An_2_CFM_Loss_KeHoachChiTiet.md) đã được áp dụng
trực tiếp vào code (không phải patch lúc chạy runtime).

## Thay đổi so với bản gốc

Thay **cách tính trọng số (weight) của loss huấn luyện** — từ công thức suy ra bằng ELBO/KL divergence
(giả định đường đi diffusion "variance-preserving", eq 11-13 paper DiffMM) sang công thức suy trực
tiếp từ **CFM loss** của Flow Matching (Lipman et al., ICLR 2023, Theorem 3) — **trong khi giữ nguyên
100% đường đi diffusion VP-style hiện tại (không đổi sang OT như Phương án 1) và giữ nguyên tham số hóa
mạng (vẫn dự đoán α₀ trực tiếp — data-prediction)**.

**Phát hiện toán học cốt lõi** (chứng minh đầy đủ + kiểm chứng số học trong file kế hoạch chi tiết):
nếu mạng vẫn dự đoán α₀ (không đổi kiến trúc/tham số hóa), CFM loss thu gọn đại số chính xác thành:

```
CFM loss = w_CFM(t) · ‖x̂₀ − x₀‖²,   w_CFM(t) = [μₜ' − (σₜ'/σₜ)·μₜ]²
```

— tức **vẫn là đúng MSE hiện có** của DiffMM, chỉ khác **1 hàm trọng số theo bước t**. Do đó patch chỉ
cần sửa đúng 1 hàm.

| File | Thay đổi |
|---|---|
| `Params.py` | Thêm argument `--w_clip` (mặc định `50.0`) — clip trọng số CFM kiểu "Min-SNR weighting" |
| `Model.py` | Thêm class `GaussianDiffusionCFM(GaussianDiffusion)` ở cuối file — class `GaussianDiffusion` gốc **giữ nguyên không đổi**. Chỉ override **đúng 1 hàm**: `training_losses` (thay `weight = SNR(t-1)-SNR(t)` bằng `weight = cfm_weight[t]` đã precompute) |
| `Main.py` | Đổi 1 dòng khởi tạo: `GaussianDiffusion(...)` → `GaussianDiffusionCFM(..., w_clip=args.w_clip)` |
| `DataHandler.py` | Fix tương thích scipy: `self.trnMat.A` → `self.trnMat.toarray()` (`.A` đã bị gỡ ở scipy ≥ 1.14 — bẫy môi trường đã biết, xem `Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md` mục 3) |

Toàn bộ phần còn lại — `q_sample` (forward), `p_mean_variance`/`p_sample` (reverse, D4), `SNR()`,
`get_betas`/`calculate_for_diffusion` (D1), MSI/`gc_loss` (D3, eq 14), Cross-Modal Contrastive
Augmentation (D5), Multi-Modal Graph Aggregation (D6), Multi-Task Training (D7) — **giữ nguyên 100%**
so với bản gốc.

### Đối chiếu công thức gốc ↔ mới

| | DiffMM gốc (D2, eq 11-13) | Phương án 2 |
|---|---|---|
| Target so sánh | `x̂₀` so với `x₀` (MSE) | Y hệt — không đổi |
| Trọng số mỗi bước t | `w_ELBO(t) = SNR(t−1) − SNR(t)`, `SNR(t)=ᾱₜ/(1−ᾱₜ)`, **không clip** | `w_CFM(t) = [μₜ' − (σₜ'/σₜ)·μₜ]²`, **có clip ở `w_clip`** |
| Mạng dự đoán | α₀ (data-prediction) | Y hệt — không đổi |
| D4 (suy luận) | `posterior_mean_coef1/2` (Bayes) | Y hệt — không đổi (path D1 không đổi) |

## Đã kiểm chứng (dry-run trên CPU, không cần GPU — trước khi bàn giao)

- `py_compile` sạch trên `Params.py`, `Model.py`, `Main.py`, `DataHandler.py`, `Utils/*.py`.
- Quét `w_CFM(t)` với `T ∈ {5, 10, 50}` (dùng tham số `noise_scale/noise_min/noise_max` mặc định của
  DiffMM): **không NaN/Inf ở bất kỳ T nào**, `w_CFM(0) = 1.0` chính xác đúng quy ước.
- **Phát hiện thực nghiệm đáng chú ý:** với tham số mặc định, `w_CFM(t)` **giảm dần đều** từ 1.0 (t=0)
  xuống rất nhỏ (~0.0001 ở T=50) — ưu tiên mạnh các bước gần dữ liệu gốc. Trong khi đó `w_ELBO(t)` gốc
  (không clip) có **đỉnh nhọn tại t=1** (ví dụ 8326 với T=5) rồi giảm dần — cả hai đều ưu tiên bước
  nhỏ hơn nhưng ở **thang độ lớn khác hẳn nhau và hình dạng khác nhau**. Đây là dữ liệu tham khảo hữu
  ích khi diễn giải kết quả thực nghiệm sau này (xem file kế hoạch chi tiết, mục 6).
- Test hồi quy: với "denoiser hoàn hảo" (`model_output == x_start`), `diff_loss = 0` với mọi `t` — pass
  (đúng lý thuyết, vì `mse=0` bất kể trọng số nào).
- Test D4 (`p_sample`, kế thừa nguyên vẹn từ lớp cha): với denoiser hoàn hảo, hội tụ đúng về `x_start`
  thật, sai số `0.0` — xác nhận D4 **thực sự không bị ảnh hưởng** bởi patch (đúng như phân tích lý
  thuyết, vì `q_sample`/`p_mean_variance`/`p_sample` không hề bị override).
- `training_losses` chạy 20 vòng dữ liệu ngẫu nhiên khác nhau: luôn hữu hạn, đúng shape.

Phần chưa/không thể kiểm chứng ở đây (do môi trường dev không có GPU/dữ liệu thật): chạy full training
thật trên GPU với dữ liệu TikTok/Baby/Sports.

## Dữ liệu

Không chứa sẵn dữ liệu (giữ repo nhẹ) — tải qua notebook Colab đi kèm (`DiffMM_PhuongAn2_CFM_Colab.ipynb`
ở thư mục cha) từ Google Drive. Cần: `trnMat.pkl`, `tstMat.pkl`, `image_feat.npy`, `text_feat.npy`
(+ `audio_feat.npy` nếu dùng `tiktok`).

## Chạy cục bộ (không qua Colab)

```bash
python Main.py --data tiktok --reg 1e-4 --ssl_reg 1e-2 --epoch 50 --trans 1 --e_loss 0.1 --cl_method 1 --w_clip 50.0
python Main.py --data baby   --reg 1e-5 --ssl_reg 1e-1 --keepRate 1 --e_loss 0.01 --w_clip 50.0
python Main.py --data sports --reg 1e-6 --ssl_reg 1e-2 --temp 0.1 --ris_lambda 0.1 --e_loss 0.5 --keepRate 1 --trans 1 --w_clip 50.0
```

(Yêu cầu GPU — code gốc dùng `.cuda()` trực tiếp, không tự động rơi về CPU.)

## Bản gốc

Xem `Params.py`/`Model.py`/`Main.py` để đối chiếu — mọi phần không liên quan Phương án 2 giữ nguyên y
hệt [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM). Trích dẫn paper gốc:

```
@article{jiang2024diffmm,
  title={DiffMM: Multi-Modal Diffusion Model for Recommendation},
  author={Jiang, Yangqin and Xia, Lianghao and Wei, Wei and Luo, Da and Lin, Kangyi and Huang, Chao},
  journal={arXiv preprint arXiv:2406.11781},
  year={2024}
}
```
