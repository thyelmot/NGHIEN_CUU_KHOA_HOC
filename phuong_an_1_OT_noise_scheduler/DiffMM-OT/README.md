# DiffMM-OT — Phương án 1 (OT-linear noise scheduler)

Đây là bản fork của [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) (paper *DiffMM: Multi-Modal
Diffusion Model for Recommendation*, ACM MM 2024) với **Phương án 1** trong kế hoạch tối ưu
[`DiffMM_FlowMatching_Optimization_Plan.md`](https://github.com/HKUDS/DiffMM) đã được áp dụng trực
tiếp vào code (không phải patch lúc chạy runtime).

## Thay đổi so với bản gốc

Thay lịch trình nhiễu (noise scheduler) kiểu VP-diffusion (eq 3 trong paper DiffMM) bằng đường đi
tuyến tính kiểu **Optimal Transport** (eq 20 trong paper *Flow Matching for Generative Modeling*,
Lipman et al., ICLR 2023):

```
μₜ = 1 − t/(T−1)              (hệ số nhân với α₀ thật; t=0 gần dữ liệu gốc, t=T−1 gần nhiễu thuần)
σₜ = 1 − (1−σ_min)·μₜ         (hệ số nhân với nhiễu)
```

thay cho công thức gốc (biến thiên theo `noise_min`/`noise_max`, ràng buộc variance-preserving
`mu²+sigma²=1`, quỹ đạo nội suy cong).

| File | Thay đổi |
|---|---|
| `Params.py` | Thêm argument `--sigma_min` (mặc định `1e-3`) |
| `Model.py` | Thêm class `GaussianDiffusionOT(GaussianDiffusion)` ở cuối file — class `GaussianDiffusion` gốc giữ nguyên không đổi. Override 3 hàm: `q_sample` (forward), `p_mean_variance` (reverse, kiểu DDIM tổng quát cho path affine bất kỳ), `SNR` (trọng số loss thật của path OT, có clip ở 50 kiểu "Min-SNR weighting" để tránh trọng số "nổ" gần t=0) |
| `Main.py` | Đổi 1 dòng khởi tạo: `GaussianDiffusion(...)` → `GaussianDiffusionOT(args.sigma_min, args.steps)` |

Toàn bộ phần còn lại (kiến trúc `Denoise` MLP, MSI/`gc_loss`, `training_losses`, top-k rebuild đồ
thị, Cross-Modal Contrastive Augmentation, Multi-Modal Graph Aggregation, Multi-Task Training) **giữ
nguyên 100%** so với bản gốc.

Đã kiểm chứng: patch biên dịch sạch (`py_compile`) và đã kiểm tra số học trên CPU (điều kiện biên
đúng, `p_sample` hội tụ đúng với denoiser hoàn hảo, mọi trọng số loss hữu hạn và ổn định).

## Dữ liệu

Thư mục `Datasets/` không chứa sẵn dữ liệu (để giữ repo nhẹ) — dữ liệu được tải tự động từ Google
Drive khi chạy notebook Colab đi kèm (xem `DiffMM_PhuongAn1_OT_Colab.ipynb` ở thư mục cha). Nếu chạy
cục bộ, xem `Datasets/README.md` để biết định dạng dữ liệu cần có: `trnMat.pkl`, `tstMat.pkl`,
`image_feat.npy`, `text_feat.npy` (+ `audio_feat.npy` nếu dùng `tiktok`).

## Chạy cục bộ (không qua Colab)

```bash
python Main.py --data tiktok --reg 1e-4 --ssl_reg 1e-2 --epoch 50 --trans 1 --e_loss 0.1 --cl_method 1 --sigma_min 1e-3
python Main.py --data baby   --reg 1e-5 --ssl_reg 1e-1 --keepRate 1 --e_loss 0.01 --sigma_min 1e-3
python Main.py --data sports --reg 1e-6 --ssl_reg 1e-2 --temp 0.1 --ris_lambda 0.1 --e_loss 0.5 --keepRate 1 --trans 1 --sigma_min 1e-3
```

(Yêu cầu GPU — code gốc dùng `.cuda()` trực tiếp, không tự động rơi về CPU.)

## Bản gốc

Xem `Params.py`/`Model.py`/`Main.py` để đối chiếu — mọi phần không liên quan Phương án 1 giữ nguyên y
hệt [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM). Trích dẫn paper gốc:

```
@article{jiang2024diffmm,
  title={DiffMM: Multi-Modal Diffusion Model for Recommendation},
  author={Jiang, Yangqin and Xia, Lianghao and Wei, Wei and Luo, Da and Lin, Kangyi and Huang, Chao},
  journal={arXiv preprint arXiv:2406.11781},
  year={2024}
}
```
