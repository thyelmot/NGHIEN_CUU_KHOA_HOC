# DiffMM-ResidualOT — Phương án 7 (v2, Residual Head)

Đây là bản fork của [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) (paper *DiffMM: Multi-Modal
Diffusion Model for Recommendation*, ACM MM 2024) với **Phương án 7 (v2) — Residual Head** trong kế
hoạch tối ưu
[`Phuong_An_7_TVS_KeHoachChiTiet_v2.md`](../../Phuong_An_7_TVS_KeHoachChiTiet_v2/Phuong_An_7_TVS_KeHoachChiTiet_v2.md)
(mục 5) đã được áp dụng trực tiếp vào code.

> **Lưu ý phạm vi:** bản kế hoạch v2 có 2 phần — (1) TVS đầy đủ (đổi sang velocity-prediction, mục 4),
> chính tài liệu kết luận **chưa nên triển khai** vì thiếu điều kiện kích hoạt K1-K3; và (2) **Residual
> Head** (mục 5), được khuyến nghị làm ngay vì rẻ và an toàn hơn nhiều. Fork này chỉ triển khai **(2)**.

## Thay đổi so với bản gốc

Ý tưởng: thay vì để mạng `Denoise` dự đoán `α₀` trực tiếp, cho mạng dự đoán **phần dư (residual)** so
với điểm neo thô `α_l` (đã có từ Phương án 6) — nếu `α_l` là 1 ước lượng tốt, phần dư nhỏ hơn và dễ học
hơn toàn bộ `α₀`:

```
α̂₀ = α_l + Denoise(αₜ, t)        (residual_head=True — CT-7.5(v2))
α̂₀ = Denoise(αₜ, t)               (residual_head=False — y hệt Phương án 6)
```

Đây **không phải** 1 công thức xác suất mới — chỉ là cách diễn giải lại đầu ra của mạng (skip-connection
tới điểm neo). `q_sample`, `p_mean_variance`, `cfm_weight` **giữ nguyên 100%** công thức của Phương án 6
— chỉ thêm đúng 1 dòng (`model_output = alpha_l + model_output`) ngay sau mỗi lần gọi mạng `Denoise`,
trước khi đưa vào mọi công thức còn lại.

`residual_head=False` (mặc định trong `Params.py`) ⟹ **trùng khít tuyệt đối Phương án 6** ở **mọi giá
trị `anchor_w`** (đã kiểm chứng bằng hồi quy số học trên chính `Model.py` này — mạnh hơn Phương án 6 chỉ
kiểm chứng hồi quy tại `anchor_w=0`, xem Giai đoạn Kiểm chứng bên dưới) — an toàn để merge.

| File | Thay đổi |
|---|---|
| `Params.py` | Thêm `--sigma_min`, `--w_clip`, `--num_sample_steps`, `--anchor_w` (kế thừa Phương án 3/6) và `--residual_head` (mặc định `0`, mới) |
| `Model.py` | Thêm `GaussianDiffusionOTCFM` (PA3), `GaussianDiffusionAnchorOT` (PA6) — sao chép nguyên vẹn, làm lớp cha — và `GaussianDiffusionResidualOT(GaussianDiffusionAnchorOT)` ở cuối file (mới). Chỉ override `p_mean_variance`, `p_sample`, `training_losses` để thêm đúng 1 dòng diễn giải lại `model_output`; thêm `_apply_residual_head`, `_need_anchor` |
| `Main.py` | 1 dòng import, 1 dòng khởi tạo (đọc thêm `args.residual_head`) — **không đổi thêm gì khác** so với patch Phương án 6 (chữ ký `training_losses`/`p_sample` giữ nguyên) |
| `DataHandler.py` | `self.trnMat.A` → `self.trnMat.toarray()` (scipy ≥1.14 đã gỡ `.A`) |

Toàn bộ phần còn lại (kiến trúc `Denoise` MLP, MSI/`gc_loss`, top-k rebuild đồ thị, Cross-Modal
Contrastive Augmentation, Multi-Modal Graph Aggregation, Multi-Task Training) **giữ nguyên 100%**. MSI
đặc biệt tự động nhất quán với residual head — vì `usr_model_embeds = mm(model_output, model_feats)`
dùng `model_output` **sau khi** đã cộng `α_l`, nên MSI luôn thấy đúng `α̂₀` cuối cùng, không cần sửa gì.

## Đã kiểm chứng

- Patch biên dịch sạch (`py_compile`).
- Hồi quy số học trên CPU: `residual_head=False` trùng khít tuyệt đối Phương án 6 ở **cả `anchor_w=0`
  lẫn `anchor_w>0`** (cả `training_losses` lẫn toàn bộ vòng lặp `p_sample`).
- `residual_head=True` với "denoiser dự đoán đúng phần dư" tái tạo chính xác `α₀` (sai lệch `<10⁻¹⁰`),
  kể cả qua toàn bộ vòng lặp suy luận `p_sample`.
- `_need_anchor()` tính đúng điểm neo ngay cả khi `anchor_w=0` nhưng `residual_head=True` (2 công tắc
  độc lập).
- Không NaN/Inf ở các trường hợp biên (user rỗng, `anchor_w` cực đoan) và ở quy mô gần thực tế
  (batch=1024, ~7000 item).

**Chưa kiểm chứng** (cần GPU + dữ liệu thật): liệu residual head có thực sự giúp mạng hội tụ nhanh
hơn/tốt hơn hay không — đây là câu hỏi thực nghiệm, không thể trả lời ngoài GPU thật.

## Dữ liệu

Thư mục `Datasets/` không chứa sẵn dữ liệu — dữ liệu được tải tự động từ Google Drive khi chạy notebook
Colab đi kèm. Định dạng cần có: `trnMat.pkl`, `tstMat.pkl`, `image_feat.npy`, `text_feat.npy` (+
`audio_feat.npy` nếu `tiktok`).

## Chạy cục bộ (không qua Colab)

```bash
python Main.py --data tiktok --reg 1e-4 --ssl_reg 1e-2 --epoch 50 --trans 1 --e_loss 0.1 --cl_method 1 --sigma_min 1e-3 --anchor_w 0.0 --residual_head 0
python Main.py --data baby   --reg 1e-5 --ssl_reg 1e-1 --keepRate 1 --e_loss 0.01 --sigma_min 1e-3 --anchor_w 0.0 --residual_head 0
python Main.py --data sports --reg 1e-6 --ssl_reg 1e-2 --temp 0.1 --ris_lambda 0.1 --e_loss 0.5 --keepRate 1 --trans 1 --sigma_min 1e-3 --anchor_w 0.0 --residual_head 0
```

Đổi `--residual_head 1` để bật residual head (khuyến nghị thử độc lập với `--anchor_w`, ví dụ giữ
`--anchor_w 0` — chỉ đổi cách diễn giải output — trước khi kết hợp cả 2).

(Yêu cầu GPU — code gốc dùng `.cuda()` trực tiếp, không tự động rơi về CPU.)

## Bản gốc

Xem `Params.py`/`Model.py`/`Main.py` để đối chiếu — mọi phần không liên quan Phương án 7 giữ nguyên y
hệt [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM). Trích dẫn paper gốc:

```
@article{jiang2024diffmm,
  title={DiffMM: Multi-Modal Diffusion Model for Recommendation},
  author={Jiang, Yangqin and Xia, Lianghao and Wei, Wei and Luo, Da and Lin, Kangyi and Huang, Chao},
  journal={arXiv preprint arXiv:2406.11781},
  year={2024}
}
```

Ý tưởng lấy cảm hứng (rồi thiết kế lại theo hướng an toàn hơn) từ Triangle Velocities Synergy trong:
```
@inproceedings{luo2025ofm,
  title={Optical Flow Matching: Reframing Optical Flow as Continuous Transport Dynamics},
  author={Luo, Ao and Li, Xin and Yang, Fan and Li, Yuezun and Yuan, Zhaoquan and Zhao, Shan and Su, Bing and Wu, Xiao},
  booktitle={CVPR},
  year={2025}
}
```
