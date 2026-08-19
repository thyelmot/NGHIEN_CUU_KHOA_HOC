# DiffMM-OT-CFM — Phương án 3 (OT path + CFM loss + rút gọn D4)

Đây là bản fork của [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) (paper *DiffMM: Multi-Modal
Diffusion Model for Recommendation*, ACM MM 2024) với **Phương án 3** trong kế hoạch chi tiết
[`Phuong_An_3_OT_CFM_KetHop_KeHoachChiTiet.md`](../Phuong_An_3_OT_CFM_KetHop_KeHoachChiTiet.md) đã
được áp dụng trực tiếp vào code — **kết hợp Phương án 1 + Phương án 2, cộng thêm phần rút gọn số bước
suy luận D4 hoàn toàn mới**.

## Thay đổi so với bản gốc

- **D1 (forward, tái sử dụng nguyên vẹn Phương án 1):** đường đi OT-linear thay cho VP-style —
  `μₜ = 1−t/(T−1)`, `σₜ = 1−(1−σ_min)·μₜ`.
- **D2 (loss huấn luyện, tái sử dụng nguyên vẹn công thức Phương án 2):** trọng số CFM thay cho
  ELBO/SNR — `w_CFM(t) = [μₜ'−(σₜ'/σₜ)·μₜ]²`, áp lên `μₜ, σₜ` của đường OT (công thức này đã chứng
  minh không phụ thuộc đường đi cụ thể, xem file kế hoạch Phương án 2).
- **D4 (suy luận, HOÀN TOÀN MỚI):** thay vòng lặp `T` bước tuần tự bằng lịch trình rút gọn `K < T`
  bước, dùng đúng công thức DDIM tổng quát hóa đã cài ở Phương án 1 (không suy thêm toán học nào) —
  chỉ đổi lịch trình lặp.

| File | Thay đổi |
|---|---|
| `Params.py` | Thêm 3 argument: `--sigma_min` (đường OT), `--w_clip` (clip trọng số CFM), `--num_sample_steps` (số bước suy luận rút gọn D4, `0` = tự động = `round(0.6*steps)`) |
| `Model.py` | Thêm class `GaussianDiffusionOTCFM(GaussianDiffusion)` ở cuối file — class gốc **giữ nguyên không đổi**. Override 4 hàm: `q_sample` (D1), `training_losses` (D2), `p_mean_variance` + `p_sample` (D4) |
| `Main.py` | Đổi 1 dòng khởi tạo: `GaussianDiffusion(...)` → `GaussianDiffusionOTCFM(args.sigma_min, args.steps, w_clip=args.w_clip, num_sample_steps=args.num_sample_steps)` |
| `DataHandler.py` | Fix tương thích scipy: `.A` → `.toarray()` (bẫy môi trường đã biết) |

Toàn bộ phần còn lại — MSI/`gc_loss` (D3), Cross-Modal Contrastive Augmentation (D5), Multi-Modal Graph
Aggregation (D6), Multi-Task Training (D7), kiến trúc mạng `Denoise` (MLP) — **giữ nguyên 100%** (lý do
giống hệt đã chứng minh ở Phương án 1 và 2: mạng vẫn dự đoán α₀ trực tiếp, D3/D5/D6 chỉ tiêu thụ
`model_output`/đồ thị `A^m` mà không quan tâm path/trọng số/số bước suy luận cụ thể).

## Đã kiểm chứng (dry-run trên CPU, không cần GPU — trước khi bàn giao)

- `py_compile` sạch trên `Params.py`, `Model.py`, `Main.py`, `DataHandler.py`, `Utils/*.py`.
- **D1+D2 kết hợp:** quét `w_CFM(t)` trên đường OT với `T ∈ {5,10,50}` × `σ_min ∈ {10⁻⁶,10⁻³,0.1}` —
  không NaN/Inf ở bất kỳ tổ hợp nào, và **max toàn cục luôn ≤ 1.0** (ổn định hơn hẳn Phương án 2 gốc
  trên đường VP, từng "nổ" tới 8326 lần).
- **D4 (phần mới, kiểm chứng kỹ nhất):**
  - Với denoiser hoàn hảo: hội tụ đúng tuyệt đối (`sai số = 0.0`) ở **mọi** lịch trình rút gọn đã thử —
    kể cả nhảy thẳng từ bước cuối về bước 0 chỉ trong 1-2 bước (`K=1`, `K=2`), với nhiều `T` khác nhau
    (5, 10, 50).
  - Với denoiser **không hoàn hảo** (lỗi phụ thuộc cả input `x` và bước `t`): đo được đường cong K vs
    độ chính xác thực tế — sai số tăng nhẹ khi K giảm (ví dụ T=10: sai số trung bình từ ~0.151 ở K=10
    xuống ~0.156 ở K=1) — mức suy giảm khiêm tốn, không phải "vách đá", cho thấy việc rút gọn D4 khả
    thi trong thực nghiệm thật.
  - `K=1` không NaN/Inf/crash với nhiều `T` khác nhau.
- Test hồi quy: denoiser hoàn hảo → `diff_loss = 0` với mọi `t`, ở mọi cấu hình `K`.
- Dry-run toàn bộ notebook cell-theo-cell (Cell 3 clone tự dò, Cell 5 xác minh patch, Cell 6 chạy
  training qua `subprocess`, Cell 7 đọc kết quả) — **cả nhánh thành công lẫn nhánh lỗi** (repo chưa
  patch, script training crash, thiếu log) đều cho thông báo đúng như thiết kế.

Phần chưa/không thể kiểm chứng ở đây (do môi trường dev không có GPU/dữ liệu thật): chạy full training
thật trên GPU với dữ liệu TikTok/Baby/Sports — đặc biệt cần đo thật hiệu năng Recall/NDCG và thời gian
chạy thật ở các giá trị `K` khác nhau, vì test số học ở trên chỉ dùng denoiser giả lập, không phản ánh
đúng độ chính xác của mạng `Denoise` thật đã huấn luyện.

## Dữ liệu

Không chứa sẵn dữ liệu (giữ repo nhẹ) — tải qua notebook Colab đi kèm
(`DiffMM_PhuongAn3_OTCFM_Colab.ipynb` ở thư mục cha) từ Google Drive. Cần: `trnMat.pkl`, `tstMat.pkl`,
`image_feat.npy`, `text_feat.npy` (+ `audio_feat.npy` nếu dùng `tiktok`).

## Chạy cục bộ (không qua Colab)

```bash
python Main.py --data tiktok --reg 1e-4 --ssl_reg 1e-2 --epoch 50 --trans 1 --e_loss 0.1 --cl_method 1 --sigma_min 1e-3 --w_clip 50.0 --num_sample_steps 0
python Main.py --data baby   --reg 1e-5 --ssl_reg 1e-1 --keepRate 1 --e_loss 0.01 --sigma_min 1e-3 --w_clip 50.0 --num_sample_steps 0
python Main.py --data sports --reg 1e-6 --ssl_reg 1e-2 --temp 0.1 --ris_lambda 0.1 --e_loss 0.5 --keepRate 1 --trans 1 --sigma_min 1e-3 --w_clip 50.0 --num_sample_steps 0
```

(Yêu cầu GPU — code gốc dùng `.cuda()` trực tiếp, không tự động rơi về CPU. Đặt `--num_sample_steps`
khác `0` để thử nghiệm rút gọn D4, ví dụ `1` để nhảy thẳng 1 bước.)

## Bản gốc

Xem `Params.py`/`Model.py`/`Main.py` để đối chiếu — mọi phần không liên quan Phương án 3 giữ nguyên y
hệt [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM). Trích dẫn paper gốc:

```
@article{jiang2024diffmm,
  title={DiffMM: Multi-Modal Diffusion Model for Recommendation},
  author={Jiang, Yangqin and Xia, Lianghao and Wei, Wei and Luo, Da and Lin, Kangyi and Huang, Chao},
  journal={arXiv preprint arXiv:2406.11781},
  year={2024}
}
```
