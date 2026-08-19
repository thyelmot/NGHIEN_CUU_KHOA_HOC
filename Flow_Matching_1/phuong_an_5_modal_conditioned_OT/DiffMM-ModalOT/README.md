# DiffMM-ModalOT — Phương án 5 (Modality-conditioned OT path)

Đây là bản fork của [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) (paper *DiffMM: Multi-Modal
Diffusion Model for Recommendation*, ACM MM 2024) với **Phương án 5** trong kế hoạch tối ưu
[`DiffMM_FlowMatching_Optimization_Plan.md`](../../DiffMM_FlowMatching_Optimization_Plan.md) đã được
áp dụng trực tiếp vào code. Chi tiết thiết kế + suy diễn công thức đầy đủ ở
[`Phuong_An_5_Modality_Conditioned_OT_KeHoachChiTiet.md`](../../Phuong_An_5_Modality_Conditioned_OT_KeHoachChiTiet.md);
quá trình kiểm chứng theo 2 giai đoạn (gradient số học ngoài codebase, rồi hồi quy trên patch thật) ở
[`Phuong_An_5_GiaiDoanA_GradCheck/`](../../Phuong_An_5_GiaiDoanA_GradCheck/README.md) và
[`Phuong_An_5_GiaiDoanB_PatchThat/`](../../Phuong_An_5_GiaiDoanB_PatchThat/README.md).

## Thay đổi so với bản gốc

Thay lịch trình OT-linear **dùng chung cho mọi item** (Phương án 1) bằng lịch trình **riêng cho từng
cặp (user, item)**, "co giãn" theo độ phù hợp giữa embedding modal của item và sở thích modal của
user — ý tưởng **tự thiết kế**, không có sẵn trong 2 bài báo gốc:

```
φ(u,i)   = cosine(trọng tâm modal các item user u đã tương tác,  embedding modal của item i)
g(u,i)   = clip(exp(−κ·φ(u,i)), g_min, g_max)      (κ=0 ⟹ g≡1 mọi nơi)
τ(t;u,i) = s(t) ^ g(u,i)                             (s(t) = 1 − t/(T−1), giống Phương án 1)
μₜ(u,i)  = τ(t;u,i) · α₀(u,i)         σₜ(u,i) = 1 − (1−σ_min)·τ(t;u,i)
```

`κ=0` (mặc định trong `Params.py`) làm `g≡1` ở mọi (user, item) ⟹ `τ(t;u,i)=s(t)` không đổi theo
(user,item) ⟹ **công thức trùng khít tuyệt đối Phương án 1** (đã kiểm chứng bằng hồi quy số học, xem
Giai đoạn B) — an toàn để merge vào codebase, rồi tăng dần `κ>0` để bật hiệu ứng modal.

Trọng số loss CFM (Phương án 2, `w_CFM(t) = [τ'(t) − (σ'(t)/σ(t))τ(t)]²`) cũng được tính **riêng cho
từng (user,item)** trên `τ(t;u,i)` thay vì trên 1 đường cong dùng chung — áp dụng **trước** khi gộp
trung bình qua chiều item (khác thứ tự với Phương án 2/3, vì trọng số ở đây khác nhau theo từng toạ độ
trong cùng 1 vector, không còn là 1 số vô hướng dùng chung cho cả vector).

| File | Thay đổi |
|---|---|
| `Params.py` | Thêm `--sigma_min`, `--w_clip`, `--kappa` (mặc định `0.0`), `--g_min` (`0.5`), `--g_max` (`2.0`), `--use_msi` (`0`) |
| `Model.py` | Thêm class `GaussianDiffusionModalOT(GaussianDiffusion)` ở cuối file — class `GaussianDiffusion` gốc giữ nguyên không đổi. Kế thừa trực tiếp `GaussianDiffusion` (không phải `GaussianDiffusionCFM`/Phương án 2 — `__init__` ở đây bỏ qua hoàn toàn khung VP-style của lớp cha, nên kế thừa từ Phương án 2 không tái sử dụng được gì về code; công thức CFM vẫn được tái dùng ở mức đại số). Override `q_sample`, `p_mean_variance`, `p_sample`, `training_losses`; thêm `_modal_affinity`, `_per_item_path`, `_cfm_weight`, `_gather_at_t` |
| `Main.py` | 1 dòng import, 1 dòng khởi tạo (`GaussianDiffusionModalOT(args.sigma_min, args.steps, kappa=args.kappa, g_min=args.g_min, g_max=args.g_max, w_clip=args.w_clip, use_msi=bool(args.use_msi))`), và 3 dòng gọi `p_sample` (thêm tham số `modal_embeds` — tái sử dụng biến `image_feats`/`text_feats`/`audio_feats` đã có sẵn cùng hàm). `training_losses` **không đổi chữ ký gọi** — tham số `model_feats` sẵn có được tái dùng làm `modal_embeds` |
| `DataHandler.py` | `self.trnMat.A` → `self.trnMat.toarray()` (scipy ≥1.14 đã gỡ `.A` — lỗi môi trường đã biết, phòng trước) |

Cờ `--use_msi` (mặc định `0` = tắt): Phương án 5 đề xuất **thay** MSI/`gc_loss` (eq 14 DiffMM) bằng
chính cơ chế điều kiện-modal này, nhưng vẫn giữ `gc_loss` như 1 lựa chọn ablation — đặt `--use_msi 1`
để giữ cả 2 cơ chế song song khi so sánh.

Toàn bộ phần còn lại (kiến trúc `Denoise` MLP, top-k rebuild đồ thị, Cross-Modal Contrastive
Augmentation, Multi-Modal Graph Aggregation, Multi-Task Training) **giữ nguyên 100%** so với bản gốc.

Đã kiểm chứng: patch biên dịch sạch (`py_compile`); kiểm tra số học trên CPU gồm — hồi quy `κ=0` trùng
khít tuyệt đối Phương án 1/2/3, `training_losses`/`p_sample` không NaN/Inf qua nhiều giá trị `κ`, 2
trường hợp biên (user không tương tác, `κ` rất lớn bão hoà clamp), và quy mô gần thực tế (batch=1024,
~7000 item). Xem chi tiết đầy đủ (kể cả 1 lỗi thật đã phát hiện và sửa trong quá trình này) ở
[`Phuong_An_5_GiaiDoanB_PatchThat/README.md`](../../Phuong_An_5_GiaiDoanB_PatchThat/README.md).

**Chưa kiểm chứng** (cần GPU + dữ liệu thật, xem mục "Đã kiểm chứng trước khi bàn giao" ở README thư
mục cha): chạy qua mạng `Denoise` thật, số liệu Recall/NDCG thật, chi phí bộ nhớ GPU thực tế khi giữ
đồng thời nhiều mảng `(batch, T, num_items)` — nếu tràn bộ nhớ GPU, thử giảm `--batch`.

## Dữ liệu

Thư mục `Datasets/` không chứa sẵn dữ liệu (để giữ repo nhẹ) — dữ liệu được tải tự động từ Google Drive
khi chạy notebook Colab đi kèm (xem `DiffMM_PhuongAn5_ModalOT_Colab.ipynb` ở thư mục cha). Nếu chạy cục
bộ, xem `Datasets/README.md` để biết định dạng dữ liệu cần có: `trnMat.pkl`, `tstMat.pkl`,
`image_feat.npy`, `text_feat.npy` (+ `audio_feat.npy` nếu dùng `tiktok`).

## Chạy cục bộ (không qua Colab)

```bash
python Main.py --data tiktok --reg 1e-4 --ssl_reg 1e-2 --epoch 50 --trans 1 --e_loss 0.1 --cl_method 1 --sigma_min 1e-3 --kappa 0.0
python Main.py --data baby   --reg 1e-5 --ssl_reg 1e-1 --keepRate 1 --e_loss 0.01 --sigma_min 1e-3 --kappa 0.0
python Main.py --data sports --reg 1e-6 --ssl_reg 1e-2 --temp 0.1 --ris_lambda 0.1 --e_loss 0.5 --keepRate 1 --trans 1 --sigma_min 1e-3 --kappa 0.0
```

Đổi `--kappa 0.0` thành giá trị `>0` (khuyến nghị bắt đầu từ `0.1`-`0.5`, tăng dần) để bật hiệu ứng
điều kiện-modal — chưa có cơ sở lý thuyết để suy ra giá trị tối ưu, cần quét thực nghiệm.

(Yêu cầu GPU — code gốc dùng `.cuda()` trực tiếp, không tự động rơi về CPU.)

## Bản gốc

Xem `Params.py`/`Model.py`/`Main.py` để đối chiếu — mọi phần không liên quan Phương án 5 giữ nguyên y
hệt [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM). Trích dẫn paper gốc:

```
@article{jiang2024diffmm,
  title={DiffMM: Multi-Modal Diffusion Model for Recommendation},
  author={Jiang, Yangqin and Xia, Lianghao and Wei, Wei and Luo, Da and Lin, Kangyi and Huang, Chao},
  journal={arXiv preprint arXiv:2406.11781},
  year={2024}
}
```
