# DiffMM-ODE — Phương án 4 (rút gọn D4 bằng ODE solver, không đổi huấn luyện)

Đây là bản fork của [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) (paper *DiffMM: Multi-Modal
Diffusion Model for Recommendation*, ACM MM 2024) với **Phương án 4** trong kế hoạch chi tiết
[`Phuong_An_4_ODE_Solver_KeHoachChiTiet.md`](../Phuong_An_4_ODE_Solver_KeHoachChiTiet.md) đã được áp
dụng trực tiếp vào code — **patch tối giản nhất trong 4 phương án**: chỉ động đến bước suy luận (D4),
**không đổi gì về huấn luyện** (D1 forward, D2 loss ELBO, D3 MSI đều giữ nguyên tuyệt đối).

## Thay đổi so với bản gốc

Thay vòng lặp suy luận `T` bước tuần tự (dùng công thức hậu nghiệm Bayes `posterior_mean_coef1/2` —
**chỉ hợp lệ về mặt toán học cho đúng 1 bước liền kề `t→t−1`**) bằng lịch trình rút gọn `K < T` bước,
dùng công thức DDIM tổng quát hóa (đúng cho **mọi** cặp `(t, t_prev)` trên cùng quỹ đạo). Đây chính là
lời giải đóng, chính xác của "probability flow ODE" ứng với đường VP-style của DiffMM (Song et al.,
2021, *Score-Based Generative Modeling through SDEs*) — tốt hơn một bộ giải Euler/Midpoint số học đơn
thuần (Euler luôn có sai số cắt cụt bậc nhất, công thức đóng này thì không).

**Phát hiện quan trọng đã kiểm chứng trước khi patch:** thử cách "dễ nghĩ tới nhất" — giữ nguyên công
thức `posterior_mean_coef1/2` gốc, chỉ bỏ bớt vài bước trong vòng lặp — cho ra kết quả **SAI** về mặt
lý thuyết khi có denoiser không hoàn hảo (dù test với denoiser hoàn hảo không lộ ra lỗi này, do đẳng
thức đặc biệt `coef1+coef2≡1` khiến `α₀` luôn là điểm bất động bất kể công thức nào). Vì vậy bản patch
**không dùng cách "skip thô"** này.

| File | Thay đổi |
|---|---|
| `Params.py` | Thêm 1 argument: `--num_sample_steps` (số bước suy luận rút gọn D4, `0` = tự động = `round(0.6*steps)`) |
| `Model.py` | Thêm class `GaussianDiffusionFastSample(GaussianDiffusion)` ở cuối file — class gốc **giữ nguyên không đổi**. Override **đúng 2 hàm**: `p_mean_variance`, `p_sample` |
| `Main.py` | Đổi 1 dòng khởi tạo: `GaussianDiffusion(...)` → `GaussianDiffusionFastSample(args.noise_scale, args.noise_min, args.noise_max, args.steps, num_sample_steps=args.num_sample_steps)` |
| `DataHandler.py` | Fix tương thích scipy: `.A` → `.toarray()` (bẫy môi trường đã biết) |

`q_sample`, `training_losses`, `get_betas`, `calculate_for_diffusion`, `SNR` — **kế thừa nguyên vẹn từ
`GaussianDiffusion` qua `super().__init__()`, không tự tính lại bất kỳ hệ số nào** (khác Phương án 1/3
phải tự dựng lại `mu`/`sigma` do đổi đường đi). MSI (D3, `gc_loss`), Cross-Modal Contrastive
Augmentation (D5), Multi-Modal Graph Aggregation (D6), Multi-Task Training (D7), kiến trúc `Denoise`
(MLP) — **giữ nguyên 100%**.

## Đã kiểm chứng (dry-run trên CPU, không cần GPU — trước khi bàn giao)

- `py_compile` sạch trên `Params.py`, `Model.py`, `Main.py`, `DataHandler.py`, `Utils/*.py`.
- **Denoiser hoàn hảo:** `p_sample` hội tụ đúng tuyệt đối (sai số `0.0`) ở mọi `K` đã thử (`K=T`,
  `K=round(0.6T)`, `K=2`, `K=1`), với nhiều `T` (5, 10, 50).
- **Xác nhận D1 thực sự không đổi:** so sánh trực tiếp `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`,
  `posterior_mean_coef1` giữa instance `GaussianDiffusionFastSample` và instance `GaussianDiffusion`
  gốc (cùng tham số) — **giống hệt nhau tuyệt đối** (`torch.allclose` pass).
- Test biên `K=1` (nhảy thẳng 1 bước) với nhiều `T` — không NaN/Inf/crash.
- `training_losses` (ELBO gốc, kế thừa nguyên vẹn) chạy 10 vòng dữ liệu ngẫu nhiên — luôn hữu hạn.
- Dry-run toàn bộ notebook cell-theo-cell (Cell 3/5/6/7) — cả nhánh thành công lẫn nhánh lỗi (repo
  chưa patch, script training crash, thiếu log) đều cho thông báo đúng như thiết kế.

Phần chưa/không thể kiểm chứng ở đây (do môi trường dev không có GPU và không có dữ liệu thật): chạy
full training thật trên GPU với dữ liệu TikTok/Baby/Sports — đặc biệt cần đo thật Recall/NDCG và thời
gian chạy ở các giá trị `NUM_SAMPLE_STEPS` khác nhau với mạng `Denoise` thật đã huấn luyện (mọi kiểm
chứng D4 ở trên dùng denoiser giả lập).

## Dữ liệu

Không chứa sẵn dữ liệu (giữ repo nhẹ) — tải qua notebook Colab đi kèm
(`DiffMM_PhuongAn4_ODE_Colab.ipynb` ở thư mục cha) từ Google Drive. Cần: `trnMat.pkl`, `tstMat.pkl`,
`image_feat.npy`, `text_feat.npy` (+ `audio_feat.npy` nếu dùng `tiktok`).

## Chạy cục bộ (không qua Colab)

```bash
python Main.py --data tiktok --reg 1e-4 --ssl_reg 1e-2 --epoch 50 --trans 1 --e_loss 0.1 --cl_method 1 --num_sample_steps 0
python Main.py --data baby   --reg 1e-5 --ssl_reg 1e-1 --keepRate 1 --e_loss 0.01 --num_sample_steps 0
python Main.py --data sports --reg 1e-6 --ssl_reg 1e-2 --temp 0.1 --ris_lambda 0.1 --e_loss 0.5 --keepRate 1 --trans 1 --num_sample_steps 0
```

(Yêu cầu GPU — code gốc dùng `.cuda()` trực tiếp, không tự động rơi về CPU. Đặt `--num_sample_steps`
khác `0` để thử nghiệm, ví dụ `1` để nhảy thẳng 1 bước.)

## Bản gốc

Xem `Params.py`/`Model.py`/`Main.py` để đối chiếu — mọi phần không liên quan Phương án 4 giữ nguyên y
hệt [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM). Trích dẫn paper gốc:

```
@article{jiang2024diffmm,
  title={DiffMM: Multi-Modal Diffusion Model for Recommendation},
  author={Jiang, Yangqin and Xia, Lianghao and Wei, Wei and Luo, Da and Lin, Kangyi and Huang, Chao},
  journal={arXiv preprint arXiv:2406.11781},
  year={2024}
}
```
