# Kế hoạch chi tiết — Phương án 4: Tăng tốc Inference (D4) bằng ODE solver, không đổi huấn luyện

> Tài liệu này đào sâu **Phương án 4** trong [`DiffMM_FlowMatching_Optimization_Plan.md`](DiffMM_FlowMatching_Optimization_Plan.md)
> (mục 5). Khác với Phương án 1/2/3 (đều sửa D1 và/hoặc D2), **Phương án 4 chỉ động đến D4** — giữ
> nguyên tuyệt đối cách huấn luyện (đường đi VP-style gốc, loss ELBO gốc). Giả định bạn đã đọc
> `DiffMM_Review.md`, `Flow_Matching_1_Review.md`, và nên đọc qua
> [`Phuong_An_3_OT_CFM_KetHop_KeHoachChiTiet.md`](Phuong_An_3_OT_CFM_KetHop_KeHoachChiTiet.md) mục 3 vì
> Phương án 4 tái sử dụng đúng ý tưởng generalized-DDIM đã kiểm chứng ở đó, chỉ áp lên đường đi khác.

---

## 1. Nhắc lại phạm vi Phương án 4

- **Vị trí áp dụng:** DUY NHẤT D4 (thủ tục sinh `α̂₀` — hiện tại: corrupt `T'` bước rồi denoise ngược
  `T` bước tuần tự, sau đó top-k chọn cạnh).
- **Giữ nguyên tuyệt đối:** D1 (Forward Diffusion, VP-style gốc — **không** đổi sang OT như PA1/PA3),
  D2 (ELBO loss gốc — **không** đổi sang CFM như PA2/PA3), D3 (MSI), D5, D6, D7, kiến trúc `Denoise`.
- **Mục tiêu:** trả lời câu hỏi *"chỉ riêng việc giảm số bước suy luận D4 (không đổi gì về huấn luyện)
  mang lại lợi ích gì, và đánh đổi độ chính xác ra sao?"* — đây là phép đo **baseline tốc độ/chất
  lượng** nên làm **trước** khi đầu tư vào Phương án 1/3 (đúng như Bước 1 trong Roadmap của bản kế
  hoạch gốc, mục 6).

---

## 2. Vì sao không thể chỉ đơn giản "bỏ bớt vài bước trong vòng lặp cũ" — cần hiểu rõ trước khi code

### 2.1 Công thức suy luận GỐC của DiffMM là gì

`p_mean_variance` gốc (không đổi trong Phương án 4) dùng:

```
model_mean = posterior_mean_coef1[t]·model_output + posterior_mean_coef2[t]·x_t
```

trong đó `posterior_mean_coef1[t]`, `posterior_mean_coef2[t]` là **công thức hậu nghiệm Bayes** của
riêng phân phối `q(α_{t-1} | α_t, α_0)` — được suy ra **chuyên biệt cho bước chuyển tiếp liền kề
`t → t−1`** (dùng `betas[t]` và `alphas_cumprod[t−1]` cụ thể của đúng 1 bước lùi). Công thức này
**không có ý nghĩa toán học đúng đắn nếu áp dụng cho một bước nhảy `t → t_prev` với `t_prev ≪ t−1`**
(nhảy nhiều bước) — vì nó ngầm giả định `x_t` và bước tiếp theo cách nhau đúng 1 nấc trên chuỗi Markov.

**Kiểm chứng bằng số học (đã chạy trước khi viết kế hoạch):** dựng đúng `betas`/`alphas_cumprod` như
DiffMM (`T=10`, tham số mặc định), test 2 cách suy luận rút gọn:
1. **"Skip thô":** giữ nguyên công thức `posterior_mean_coef1/2`, chỉ đổi tập chỉ số `t` được lặp qua
   (bỏ bớt một số bước) — đây là cách "dễ nghĩ tới nhất" nhưng **sai về mặt lý thuyết** như phân tích ở
   trên.
2. **Reformulate kiểu DDIM:** suy ngược "nhiễu ước lượng" từ dự đoán của mạng rồi tái tạo theo công
   thức tổng quát cho cặp `(t, t_prev)` bất kỳ (giống hệt cách đã làm ở Phương án 1/3, chỉ khác chỗ
   dùng `μₜ, σₜ` của đường VP gốc thay vì đường OT).

Với **denoiser hoàn hảo**, cả 2 cách đều cho sai số `0.0` — nhưng đây là **trường hợp suy biến**: công
thức gốc có đẳng thức `posterior_mean_coef1[t] + posterior_mean_coef2[t] ≡ 1` với mọi `t` (đã kiểm tra
số học: `[1.0000, 0.99999997, ...]`), khiến `x₀` thật luôn là điểm bất động của phép lặp bất kể công
thức nào — **không phải bằng chứng "skip thô" đúng**, chỉ là phép thử chưa đủ khắt khe.

**Thử lại với denoiser KHÔNG hoàn hảo** (lỗi phụ thuộc cả `x_t` lẫn `t`, mô phỏng gần hơn hành vi mạng
thật): 2 cách cho ra kết quả **khác nhau rõ rệt** ở mọi lịch trình rút gọn (dao động ~0.22-0.24 sai số
trung bình tùy schedule, không có cách nào luôn thắng tuyệt đối trong thử nghiệm tổng hợp này) — xác
nhận đây thực sự là **2 thủ tục khác nhau về mặt toán học**, không phải cùng 1 công thức viết lại.

### 2.2 Kết luận thiết kế: dùng cách reformulate kiểu DDIM — đây chính xác là "ODE solver" mà đề bài yêu cầu

Kết quả lý thuyết đã biết (Song et al., 2021, *Score-Based Generative Modeling through SDEs*): mọi quá
trình khuếch tán kiểu VP có một **"probability flow ODE"** xác định — một phương trình vi phân tất định
có cùng phân phối biên `pₜ(x)` với quá trình khuếch tán ngẫu nhiên gốc. **DDIM (công thức reformulate ở
mục 2.1, ý 2) chính là lời giải ĐÓNG (closed-form), CHÍNH XÁC của ODE này khi biết đúng score/nhiễu** —
không phải một xấp xỉ số học như Euler. Nói cách khác: **áp dụng lại đúng công thức đã dùng ở Phương án
1/3 lên `μₜ, σₜ` của đường VP gốc** *chính là* "viết lại vòng lặp suy luận dưới dạng ODE solver" mà bản
kế hoạch gốc yêu cầu — và là lựa chọn **tốt hơn** một bộ giải Euler/Midpoint số học đơn thuần (vì Euler
luôn có sai số cắt cụt bậc nhất `O(Δt)` ngay cả khi mô hình hoàn hảo, còn công thức đóng này thì không).

---

## 3. Thiết kế patch cụ thể — tối giản, chỉ 2 hàm

Vì D1 hoàn toàn không đổi, `__init__` có thể **gọi thẳng `super().__init__(...)`** (khác Phương án 1/3
phải tự dựng lại toàn bộ `mu_coef`/`sigma_coef` do đường đi thay đổi) — đây là điểm khiến Phương án 4
đơn giản hơn hẳn về mặt code so với PA1/PA3, đúng như đánh giá ★★☆☆☆ trong bản kế hoạch gốc.

```python
class GaussianDiffusionFastSample(GaussianDiffusion):
    """
    [Phuong an 4] Chi doi D4 (suy luan) — D1 (forward, VP-style), D2 (ELBO loss), D3 (MSI) GIU
    NGUYEN TUYET DOI (ke thua nguyen ven tu GaussianDiffusion qua super().__init__(), khong tu
    dung lai bat ky mang he so nao — khac PA1/PA3 phai tu tinh lai mu/sigma do doi duong di).

    Thay vong lap T buoc tuan tu (dung cong thuc hau nghiem Bayes posterior_mean_coef1/2, CHI
    dung cho buoc lien ke t->t-1) bang lich trinh rut gon K<T buoc, dung cong thuc DDIM tong
    quat hoa (dung lai y nguyen tu Phuong an 1/3, ap len mu/sigma CO SAN cua duong VP goc:
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod) — day chinh la loi giai dong, chinh xac
    cua "probability flow ODE" cua duong VP (Song et al. 2021), khong phai xap xi Euler.

    Override DUY NHAT 2 ham: p_mean_variance, p_sample.
    """

    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True, num_sample_steps=0):
        super(GaussianDiffusionFastSample, self).__init__(noise_scale, noise_min, noise_max, steps, beta_fixed)
        K = num_sample_steps if num_sample_steps and num_sample_steps > 0 else max(1, round(0.6 * steps))
        self.num_sample_steps = K
        self._build_sample_schedule(K)

    def _build_sample_schedule(self, K):
        idx = torch.linspace(self.steps - 1, 0, steps=K).round().long()
        idx = torch.unique_consecutive(idx)
        self.sample_schedule = idx.tolist()
        self.next_index_map = {}
        for k, t in enumerate(self.sample_schedule):
            self.next_index_map[t] = self.sample_schedule[k + 1] if k + 1 < len(self.sample_schedule) else -1

    def p_mean_variance(self, model, x, t):
        model_output = model(x, t, False)  # du doan alpha_0 — KHONG doi parameterization

        # mu_t, sigma_t cua duong VP GOC — da co san tu __init__ ke thua, KHONG tu tinh lai
        mu_t = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape)
        sigma_t = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        noise_pred = (x - mu_t * model_output) / sigma_t.clamp(min=1e-8)

        t_val = int(t[0].item())
        t_prev_val = self.next_index_map[t_val]
        if t_prev_val == -1:
            mu_prev, sigma_prev = 1.0, 0.0
        else:
            t_prev = torch.full_like(t, t_prev_val)
            mu_prev = self._extract_into_tensor(self.sqrt_alphas_cumprod, t_prev, x.shape)
            sigma_prev = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t_prev, x.shape)

        model_mean = mu_prev * model_output + sigma_prev * noise_pred
        return model_mean, None  # bien the nay chi ho tro suy luan tat dinh (sampling_noise=False)

    def p_sample(self, model, x_start, steps, sampling_noise=False):
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0]).cuda()
            x_t = self.q_sample(x_start, t)  # KHONG doi — dung q_sample cua lop cha (D1 goc)

        for i in self.sample_schedule:  # <-- CHI KHAC lop cha: lap qua lich trinh rut gon
            t = torch.tensor([i] * x_t.shape[0]).cuda()
            model_mean, _ = self.p_mean_variance(model, x_t, t)
            x_t = model_mean
        return x_t
```

`q_sample`, `training_losses`, `get_betas`, `calculate_for_diffusion`, `SNR` — **tất cả kế thừa nguyên
vẹn, không override**. Đây là mức độ thay đổi tối thiểu nhất trong 4 phương án đã lên kế hoạch.

### 3.1 Sửa `Main.py` và `Params.py`

```python
self.diffusion_model = GaussianDiffusionFastSample(
    args.noise_scale, args.noise_min, args.noise_max, args.steps,
    num_sample_steps=args.num_sample_steps,
).cuda()
```

1 argument mới duy nhất trong `Params.py`:

```python
parser.add_argument('--num_sample_steps', type=int, default=0,
                     help='[Phuong an 4] so buoc suy luan rut gon D4 (0 = tu dong = round(0.6*steps))')
```

---

## 4. Kế hoạch kiểm chứng (theo đúng Bước 4, `Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md`)

1. **Biên dịch sạch** `py_compile` trên bản patch, tải mới từ GitHub.
2. **Lặp lại đầy đủ trên code thật** 2 thí nghiệm đã làm sơ bộ ở mục 2.1:
   - Denoiser hoàn hảo: xác nhận sai số `0.0` tuyệt đối với công thức DDIM-reformulated, ở nhiều `K`
     (`K=T`, `K=round(0.6T)`, `K=2`, `K=1`) và nhiều `T` (5, 10, 50).
   - Denoiser không hoàn hảo (phụ thuộc `x`, `t`): đo và **ghi lại rõ ràng trong README** rằng "skip
     thô" và "DDIM-reformulated" cho kết quả khác nhau — không khẳng định bên nào luôn thắng tuyệt đối
     trên dữ liệu tổng hợp, nhưng khuyến nghị dùng DDIM-reformulated vì có cơ sở lý thuyết chặt
     (mục 2.2), và **không dùng "skip thô"** trong bản patch chính thức (chỉ giữ trong test để đối
     chiếu, không đưa vào code sản phẩm).
3. **Test biên:** `K=1` (nhảy thẳng 1 bước) ở nhiều `T` khác nhau — không NaN/Inf/crash.
4. **So sánh trực tiếp với D4 của Phương án 3** (rất đáng làm, vì cả 2 dùng chung công thức DDIM, chỉ
   khác đường đi nền — VP cong ở đây, OT thẳng ở PA3): dùng cùng 1 denoiser giả lập không hoàn hảo, đo
   sai số ở cùng giá trị `K` cho cả 2 biến thể — **kỳ vọng đường VP (cong) mất độ chính xác nhanh hơn
   khi `K` giảm so với đường OT (thẳng)** — đây chính là luận điểm cốt lõi của cả bài Flow Matching áp
   dụng cụ thể vào DiffMM, và là kết quả **thú vị nhất** mà Phương án 4 có thể mang lại khi đặt cạnh
   Phương án 3.
5. **Dry-run toàn bộ notebook** (Cell 3/5/6/7, cả nhánh thành công và lỗi) — Cell 5 kiểm tra sự tồn tại
   của `class GaussianDiffusionFastSample` và `--num_sample_steps`.

---

## 5. Vì sao D3/D5/D6/D7 không cần đổi

Lý do giống hệt đã chứng minh ở Phương án 1/2/3: mạng vẫn dự đoán `α₀` trực tiếp (không đổi tham số
hóa), D3 (MSI) vẫn dùng thẳng `model_output`; D5/D6 chỉ tiêu thụ đồ thị `A^m` (đầu ra D4) dưới dạng đồ
thị nhị phân sau top-k, không quan tâm D4 dùng bao nhiêu bước hay công thức gì để tạo ra nó.

---

## 6. Kế hoạch đóng gói & notebook (dùng `Folder_Base`)

1. Folder mới `phuong_an_4_ODE_solver/` (ngang hàng 3 phương án trước).
2. Bản fork `DiffMM-FastSample/` — copy từ 1 bản `DiffMM` **mới tải lại từ GitHub**, áp patch mục 3,
   kèm fix `DataHandler.py` (`.A → .toarray()`, bẫy môi trường đã biết).
3. Git repo độc lập, đặt **ngoài** mọi repo khác, tự kiểm `git rev-parse --show-toplevel`.
4. Notebook từ `Folder_Base/Colab_Template.ipynb`: Cell 1 chỉ cần thêm `NUM_SAMPLE_STEPS` (không cần
   `SIGMA_MIN`/`W_CLIP` vì D1/D2 không đổi); Cell 5 kiểm tra `class GaussianDiffusionFastSample` và
   `--num_sample_steps`; Cell 6 `CLI_ARGS` thêm `--num_sample_steps`; Cell 7 giữ nguyên `RESULT_REGEX`.
   Cân nhắc thêm 1 cell phụ (hoặc mở rộng Cell 6) đo thời gian chạy thật ở vài giá trị `K` liên tiếp
   trong cùng 1 lần chạy, để có ngay đường cong tốc độ/chất lượng mà không cần chạy lại notebook nhiều
   lần.

---

## 7. Kế hoạch thực nghiệm

Đúng vai trò đã định trong Roadmap gốc (Bước 1 — bước đo baseline sớm, rủi ro thấp):

| Biến thể | Recall@20 | NDCG@20 | NFE lúc sinh A^m | Thời gian train tổng |
|---|---|---|---|---|
| DiffMM gốc (T bước đầy đủ) | (baseline) | (baseline) | T | (baseline) |
| + D4 rút gọn, K=round(0.6T) (Phương án 4) | ? | ? | K < T | ? (kỳ vọng giảm nhẹ) |
| + D4 rút gọn, K nhỏ (K=1,2 — thử biên) | ? | ? | 1-2 | ? |

**Thí nghiệm bổ sung khuyến nghị** (chỉ làm được khi đã có cả Phương án 3): ở **cùng giá trị K**, so
sánh Recall/NDCG giữa Phương án 4 (đường VP cong) và phần D4 của Phương án 3 (đường OT thẳng) — nếu giả
thuyết của bài Flow Matching đúng, Phương án 3 sẽ giữ chất lượng tốt hơn ở cùng K nhỏ.

---

## 8. Rủi ro & lưu ý

- **Lợi ích kỳ vọng khiêm tốn** (đã nêu trong bản kế hoạch gốc): quỹ đạo vẫn cong theo đường VP, nên
  giảm bước dễ mất chính xác hơn so với Phương án 1/3 (đường thẳng). Giá trị chính của Phương án 4 là
  **phép đo baseline nhanh, rủi ro thấp**, không phải mục tiêu tối ưu tốc độ cuối cùng.
- **T mặc định của DiffMM đã rất nhỏ (5)** — cùng lưu ý như Phương án 3, đừng kỳ vọng mức cải thiện ấn
  tượng như trên ảnh.
- **"Skip thô" (mục 2.1) không được đưa vào code sản phẩm** — chỉ dùng làm đối chứng lúc kiểm chứng,
  vì thiếu cơ sở lý thuyết cho việc nhảy nhiều bước.
- **So sánh công bằng:** giữ nguyên mọi hyperparameter khác (kiến trúc `Denoise`, `reg`, `ssl_reg`,
  ...) khi so sánh các dòng trong bảng mục 7 — chỉ đổi `num_sample_steps`.

---

## 9. Tóm tắt vị trí Phương án 4 trong toàn cảnh 4 phương án

| | PA1 | PA2 | PA3 | **PA4** |
|---|---|---|---|---|
| D1 (path) | OT | VP (không đổi) | OT | **VP (không đổi)** |
| D2 (loss) | ELBO (không đổi) | CFM | CFM | **ELBO (không đổi)** |
| D4 (số bước) | T (không đổi) | T (không đổi) | K < T (DDIM trên OT) | **K < T (DDIM trên VP)** |
| Số hàm override | 3 | 1 | 4 | **2** |
| Hyperparameter mới | `--sigma_min` | `--w_clip` | cả 3 | **chỉ `--num_sample_steps`** |
| Độ khó (kế hoạch gốc) | ★☆☆☆☆ | ★★★☆☆ | ★★★★☆ | **★★☆☆☆** |
| Vai trò trong Roadmap | Bước 2 | Bước 3 | Bước 4 | **Bước 1 (đo sớm nhất)** |

Phương án 4 và phần D4 của Phương án 3 dùng **chung một cơ chế** (generalized-DDIM, lịch trình rút
gọn) — chỉ khác đường đi nền. Vì vậy nên triển khai Phương án 4 **trước**, không chỉ vì thứ tự Roadmap
mà còn vì code viết cho Phương án 4 gần như tái sử dụng được 100% khi ráp vào phần D4 của Phương án 3.
