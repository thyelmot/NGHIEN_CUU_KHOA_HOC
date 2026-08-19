# Kế hoạch chi tiết — Phương án 3: Kết hợp OT path + CFM loss, tăng tốc Inference (D1+D2+D4)

> Tài liệu này đào sâu **Phương án 3** trong [`DiffMM_FlowMatching_Optimization_Plan.md`](DiffMM_FlowMatching_Optimization_Plan.md)
> (mục 5). Phương án 3 = **kết hợp** [`Phuong_An_1...`](phuong_an_1_OT_noise_scheduler/README.md) (đổi
> đường đi D1 sang OT-linear) **+** [`Phuong_An_2_CFM_Loss_KeHoachChiTiet.md`](Phuong_An_2_CFM_Loss_KeHoachChiTiet.md)
> (đổi trọng số loss D2 sang CFM) **+ một phần hoàn toàn mới**: tăng tốc D4 (Inference) bằng cách giảm
> số bước suy luận, tận dụng đúng tính chất "quỹ đạo thẳng, tốc độ không đổi" của đường đi OT. Giả định
> bạn đã đọc 2 file trên cùng `DiffMM_Review.md` và `Flow_Matching_1_Review.md`.

---

## 1. Nhắc lại phạm vi Phương án 3

- **Vị trí áp dụng:** D1 + D2 + D4 (toàn bộ "cỗ máy diffusion" của DiffMM).
- **D1 (Forward):** đổi sang đường đi OT-linear — **tái sử dụng nguyên vẹn** thiết kế đã kiểm chứng ở
  Phương án 1 (`μₜ = 1−t/(T−1)`, `σₜ = 1−(1−σ_min)·μₜ`).
- **D2 (Loss huấn luyện):** đổi trọng số sang CFM — **tái sử dụng nguyên vẹn** công thức đã kiểm chứng
  ở Phương án 2 (`w_CFM(t) = [μₜ'−(σₜ'/σₜ)·μₜ]²`), chỉ khác ở chỗ **`μₜ, σₜ` giờ là của đường OT** (Phương
  án 2 dùng cho đường VP-style gốc).
- **D4 (Inference):** đây là phần **thực sự mới**, chưa xuất hiện ở Phương án 1 lẫn 2 — giảm số bước
  suy luận từ `T` (đầy đủ) xuống `K < T` bước, tận dụng đúng tính chất đường thẳng của OT path (đúng
  như mục tiêu ban đầu bài kế hoạch gốc đã đặt ra cho Phương án 3: *"D4 sẽ đổi từ T bước reverse tuần
  tự thành giải ODE bằng solver rẻ với rất ít bước"*).
- **Giữ nguyên:** D3 (MSI), D5, D6, D7, kiến trúc mạng `Denoise` (MLP) — với lý do giống hệt đã chứng
  minh ở Phương án 1 và 2 (xem mục 5 bên dưới).

---

## 2. Phần tái sử dụng từ Phương án 1 + 2 (không cần suy lại)

### 2.1 Từ Phương án 1 — đường đi OT (D1)

```
μₜ = 1 − t/(T−1)              (hệ số nhân α₀; t=0 gần dữ liệu gốc, t=T−1 gần nhiễu thuần)
σₜ = 1 − (1−σ_min)·μₜ         (hệ số nhân nhiễu)
```

`q_sample` giữ nguyên y hệt cách cài đặt ở Phương án 1 (`x_t = μₜ·x₀ + σₜ·ε`).

### 2.2 Từ Phương án 2 — công thức trọng số CFM (D2)

Phát hiện quan trọng nhất của Phương án 2 (đã chứng minh đại số + kiểm chứng số học) là: nếu mạng vẫn
dự đoán α₀ trực tiếp (không đổi tham số hóa), CFM loss thu gọn thành:

```
CFM loss = w_CFM(t) · ‖x̂₀ − x₀‖²,   w_CFM(t) = [μₜ' − (σₜ'/σₜ)·μₜ]²
```

**Điểm mấu chốt cần nhấn mạnh:** công thức này **không hề giả định `μₜ, σₜ` là của đường VP-style** —
phép suy (CT-1)→(CT-4) trong file kế hoạch Phương án 2 chỉ dùng định nghĩa tổng quát của một đường đi
Gauss affine bất kỳ (Theorem 3, Flow Matching). Do đó **công thức `w_CFM(t)` và hàm `training_losses`
dùng nó có thể tái sử dụng 100% không đổi 1 dòng nào** — chỉ cần thay `μₜ, σₜ` đầu vào từ "của đường
VP" (Phương án 2) sang "của đường OT" (mục 2.1). Đây là lý do vì sao việc "kết hợp" 2 phương án không
đòi hỏi suy lại toán học từ đầu cho phần D1+D2.

---

## 3. Phần hoàn toàn mới: tăng tốc D4 bằng generalized-DDIM rút gọn bước

### 3.1 Quan sát chìa khóa

`p_mean_variance` đã cài đặt ở Phương án 1 (áp dụng công thức DDIM tổng quát hóa cho một đường đi Gauss
affine bất kỳ) có dạng:

```
ε̂ = (x_t − μₜ·x̂₀) / σₜ                         (suy ngược "nhiễu ước lượng" từ dự đoán α₀ của mạng)
x_{t_prev} = μ_{t_prev}·x̂₀ + σ_{t_prev}·ε̂       (tái tạo lại điểm trên quỹ đạo tại bước t_prev)
```

Công thức này **đúng với bất kỳ cặp `(t, t_prev)` nào trên cùng 1 quỹ đạo**, không bắt buộc
`t_prev = t−1`. Đây chính là cơ chế đứng sau kỹ thuật "DDIM sampling rút gọn bước" nổi tiếng — hoàn
toàn không cần suy thêm công thức mới, **chỉ cần đổi lịch trình lặp** trong `p_sample`.

### 3.2 Kiểm chứng bằng số học (đã chạy thật trước khi viết kế hoạch này)

Với denoiser giả lập "hoàn hảo" (`x̂₀ = x₀` chính xác) và đường đi OT (`T=10`, `σ_min=10⁻³`), thử 4 lịch
trình suy luận khác nhau — kể cả **nhảy thẳng từ bước T−1 xuống bước 0 chỉ trong 2 bước**:

| Lịch trình (chỉ số bước, giảm dần) | Số bước K | Sai số so với x₀ thật |
|---|---|---|
| `[9,8,7,6,5,4,3,2,1,0]` (đầy đủ, K=T=10) | 10 | `0.000e+00` |
| `[9,6,3,0]` | 4 | `0.000e+00` |
| `[9,4,0]` | 3 | `0.000e+00` |
| `[9,0]` (nhảy thẳng 1 bước) | 2 | `0.000e+00` |

**Kết luận:** với denoiser hoàn hảo, sai số bằng 0 tuyệt đối ở **mọi** lịch trình rút gọn — đúng lý
thuyết ODE tất định (giải bằng công thức đóng thay vì sai phân số học từng bước nhỏ, nên không tích
lũy sai số rời rạc hóa). Trên thực tế (denoiser không hoàn hảo), sai số sẽ tăng dần khi K giảm — đây là
đánh đổi tốc độ/độ chính xác cần đo bằng thực nghiệm thật (mục 6), nhưng khẳng định chắc chắn: **công
thức không hề sai khi giảm số bước**, chỉ có độ chính xác thực nghiệm phụ thuộc chất lượng denoiser.

### 3.3 Thiết kế lịch trình rút gọn (`sample_schedule`)

```
sample_schedule = K chỉ số cách đều nhau trong {0, ..., T−1}, luôn gồm cả 2 đầu mút (T−1 và 0),
                  sắp xếp giảm dần.
next_index_map[chỉ số cuối cùng trong schedule] = None (nghĩa là "sạch hoàn toàn": μ_prev=1, σ_prev=0
                                                          — đúng quy ước biên đã dùng ở Phương án 1)
next_index_map[mỗi chỉ số khác trong schedule] = chỉ số kế tiếp (nhỏ hơn) trong schedule
```

`K` (số bước suy luận, gọi là `num_sample_steps`) là hyperparameter mới — mặc định đề xuất
`K = max(1, round(0.6·T))` (khớp với kỳ vọng "~40% NFE thấp hơn" đã nêu trong bảng ablation của bản kế
hoạch gốc), có thể chỉnh qua CLI để quét thực nghiệm.

### 3.4 ⚠️ Lưu ý quan trọng về kỳ vọng lợi ích thực tế (tránh kỳ vọng quá mức)

Bài Flow Matching đo NFE trên ảnh với `T` lên tới hàng trăm/nghìn bước — giảm 40% ở đó tiết kiệm đáng
kể. **DiffMM mặc định `args.steps = 5`** (`Params.py`) — đã rất nhỏ sẵn. Giảm từ 5 xuống 3 bước
(`K=3`, khớp tỉ lệ 60%) tiết kiệm được **2 lần gọi mạng denoiser mỗi lần rebuild đồ thị `A^m`**, nhân
với 3 modal × mỗi epoch trong suốt D7 (Multi-Task Training) — có tích lũy nhưng **không nên kỳ vọng
mức cải thiện runtime ấn tượng như trên ảnh**. Giá trị thực sự của phần D4 này nằm ở việc **hoàn thiện
đúng khái niệm** "FM w/ OT" mà bài kế hoạch gốc đặt mục tiêu (dùng ODE solver rẻ thay vì T bước cố
định), và mở đường thử nghiệm K rất nhỏ (K=1, K=2) xem chất lượng đồ thị `A^m` sinh ra có còn chấp
nhận được không — đây là câu hỏi thực nghiệm thú vị hơn là mục tiêu tối ưu tốc độ tuyệt đối.

---

## 4. Thiết kế patch cụ thể

### 4.1 Vị trí sửa: 1 class mới, kế thừa `GaussianDiffusion`, override 4 hàm

```python
class GaussianDiffusionOTCFM(GaussianDiffusion):
    """
    [Phuong an 3] Ket hop OT path (Phuong an 1) + CFM loss weighting (Phuong an 2) + rut gon so
    buoc suy luan D4 (generalized-DDIM voi lich trinh cach deu, xem CT o Phuong_An_3...md).
    Override 4 ham: q_sample (OT, tai su dung PA1), training_losses (CFM weight, tai su dung
    cong thuc PA2 nhung ap len mu/sigma cua duong OT), p_mean_variance + p_sample (D4 rut gon,
    MOI — dua tren cung cong thuc DDIM tong quat da dung o PA1, chi doi lich trinh lap).
    """

    def __init__(self, sigma_min, steps, w_clip=50.0, num_sample_steps=None):
        nn.Module.__init__(self)
        self.steps = steps
        self.sigma_min = sigma_min
        self.w_clip = w_clip
        self.noise_scale = 1.0

        # --- D1: duong OT (tai su dung PA1) ---
        t_idx = torch.arange(steps, dtype=torch.float64)
        s = (1.0 - t_idx / (steps - 1)) if steps > 1 else torch.ones_like(t_idx)
        self.mu_coef = s.cuda()
        self.sigma_coef = (1.0 - (1.0 - sigma_min) * s).cuda()

        # --- D2: trong so CFM (tai su dung cong thuc PA2, ap len mu/sigma cua OT) ---
        self._precompute_cfm_weight()   # y het ham cua PA2, khong doi 1 dong

        # --- D4: lich trinh rut gon (MOI) ---
        K = num_sample_steps or max(1, round(0.6 * steps))
        self.num_sample_steps = K
        self._build_sample_schedule(K)

    def _precompute_cfm_weight(self):
        # Y HET ham cua GaussianDiffusionCFM (Phuong an 2) — khong sua gi, chi mu/sigma dau vao khac
        mu, sigma = self.mu_coef, self.sigma_coef
        mu_prev = torch.cat([mu[:1], mu[:-1]])
        sigma_prev = torch.cat([sigma[:1], sigma[:-1]])
        mu_prime = mu_prev - mu
        sigma_prime = sigma_prev - sigma
        w = (mu_prime - (sigma_prime / sigma.clamp(min=1e-8)) * mu) ** 2
        w[0] = 1.0
        self.cfm_weight = w.clamp(max=self.w_clip).cuda()

    def _build_sample_schedule(self, K):
        # K chi so cach deu trong {0,...,T-1}, giam dan, luon co ca 2 dau mut
        idx = torch.linspace(self.steps - 1, 0, steps=K).round().long()
        idx = torch.unique_consecutive(idx)          # tranh trung neu K > T
        self.sample_schedule = idx.tolist()
        # next_index_map[t] = t_prev (hoac -1 = "sach hoan toan") CHI dinh nghia cho cac t trong schedule
        self.next_index_map = {}
        for k, t in enumerate(self.sample_schedule):
            self.next_index_map[t] = self.sample_schedule[k + 1] if k + 1 < len(self.sample_schedule) else -1

    def q_sample(self, x_start, t, noise=None):        # Y HET Phuong an 1
        if noise is None:
            noise = torch.randn_like(x_start)
        mu_t = self._extract_into_tensor(self.mu_coef, t, x_start.shape)
        sigma_t = self._extract_into_tensor(self.sigma_coef, t, x_start.shape)
        return mu_t * x_start + sigma_t * noise

    def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):  # Y HET Phuong an 2
        batch_size = x_start.size(0)
        ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, ts, noise)
        model_output = model(x_t, ts)
        mse = self.mean_flat((x_start - model_output) ** 2)
        weight = self._extract_into_tensor(self.cfm_weight, ts, mse.shape)
        diff_loss = weight * mse
        usr_model_embeds = torch.mm(model_output, model_feats)
        usr_id_embeds = torch.mm(x_start, itmEmbeds)
        gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)
        return diff_loss, gc_loss

    def p_mean_variance(self, model, x, t):             # MOI — nhu PA1 nhung tra ve theo next_index_map
        model_output = model(x, t, False)
        mu_t = self._extract_into_tensor(self.mu_coef, t, x.shape)
        sigma_t = self._extract_into_tensor(self.sigma_coef, t, x.shape)
        noise_pred = (x - mu_t * model_output) / sigma_t.clamp(min=1e-8)

        t_val = int(t[0].item())                        # ca batch dung chung 1 t trong vong lap p_sample
        t_prev_val = self.next_index_map[t_val]
        if t_prev_val == -1:
            mu_prev, sigma_prev = 1.0, 0.0
        else:
            t_prev = torch.full_like(t, t_prev_val)
            mu_prev = self._extract_into_tensor(self.mu_coef, t_prev, x.shape)
            sigma_prev = self._extract_into_tensor(self.sigma_coef, t_prev, x.shape)

        model_mean = mu_prev * model_output + sigma_prev * noise_pred
        return model_mean, None

    def p_sample(self, model, x_start, steps, sampling_noise=False):  # MOI — lap theo sample_schedule
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0]).cuda()
            x_t = self.q_sample(x_start, t)

        for i in self.sample_schedule:                  # <-- CHI KHAC PA1: lap qua schedule rut gon
            t = torch.tensor([i] * x_t.shape[0]).cuda()
            model_mean, _ = self.p_mean_variance(model, x_t, t)
            x_t = model_mean                             # tat dinh (sampling_noise khong ho tro, giong PA1)
        return x_t
```

*(Bản nháp thiết kế — khi triển khai thật, làm lại đúng Bước 1 của `Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md`:
tải lại `Model.py` MỚI NHẤT từ GitHub ngay trước khi patch.)*

### 4.2 Sửa `Main.py` và `Params.py`

```python
self.diffusion_model = GaussianDiffusionOTCFM(
    args.sigma_min, args.steps, w_clip=args.w_clip, num_sample_steps=args.num_sample_steps
).cuda()
```

3 argument mới trong `Params.py` (gộp cả 2 từ Phương án 1+2, cộng 1 cái mới cho D4):

```python
parser.add_argument('--sigma_min', type=float, default=1e-3, help='[PA1/PA3] sigma_min duong OT')
parser.add_argument('--w_clip', type=float, default=50.0, help='[PA2/PA3] clip trong so CFM loss')
parser.add_argument('--num_sample_steps', type=int, default=0,
                     help='[PA3] so buoc suy luan rut gon D4 (0 = tu dong = round(0.6*steps))')
```

*(Quy ước `0` = "tự tính mặc định" để không phải sửa giá trị `None` qua CLI — nếu `args.num_sample_steps
> 0`, dùng đúng giá trị đó; nếu `== 0`, code tự đặt `max(1, round(0.6*args.steps))`.)*

---

## 5. Vì sao D3/D5/D6/D7 vẫn không cần đổi (tổng hợp lý do đã chứng minh ở PA1 + PA2)

| Thành phần | Vì sao không cần đổi khi kết hợp cả OT + CFM |
|---|---|
| D3 (MSI, `gc_loss`) | Vẫn dùng thẳng `model_output` (α̂₀) — mạng vẫn dự đoán α₀ trực tiếp ở Phương án 3, y hệt PA1 và PA2 riêng lẻ. Không phụ thuộc path (OT hay VP) hay cách trọng số hóa loss. |
| D5 (Contrastive), D6 (Graph Aggregation) | Chỉ tiêu thụ đồ thị `A^m` (đầu ra của D4) dưới dạng đồ thị nhị phân sau top-k — không quan tâm D4 sinh ra `A^m` bằng bao nhiêu bước hay theo công thức nào. |
| Kiến trúc `Denoise` (MLP) | Không đổi input/output shape hay ý nghĩa ở bất kỳ phương án nào trong 3 phương án. |

---

## 6. Kế hoạch kiểm chứng (theo đúng Bước 4, `Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md`)

1. **Biên dịch sạch** `py_compile` trên bản patch, tải mới từ GitHub.
2. **Kiểm tra số học D1+D2 (đã làm sơ bộ ở mục 2, cần lặp lại đầy đủ khi code thật):**
   - Quét `T ∈ {5,10,50}` × `σ_min ∈ {10⁻⁶,10⁻⁴,10⁻²}` cho `w_CFM(t)` trên đường OT — xác nhận lại
     **không NaN/Inf**, và xác nhận lại phát hiện đã thấy ở bản nháp: **max toàn cục < 1.0** (khác hẳn
     mức "nổ tới hàng nghìn lần" đã thấy ở PA2 gốc trên đường VP — cần ghi rõ số liệu thật trong README
     bản fork).
   - Test hồi quy: denoiser hoàn hảo → `diff_loss = 0`.
3. **Kiểm tra số học D4 (mới, quan trọng nhất của phương án này):**
   - Lặp lại đúng thí nghiệm ở mục 3.2 (4 lịch trình, denoiser hoàn hảo) **trên code thật** (không chỉ
     script nháp) — xác nhận sai số vẫn `0.0` tuyệt đối ở mọi `K`.
   - Test bổ sung: denoiser **không hoàn hảo** (ví dụ cộng nhiễu Gauss nhỏ vào dự đoán) — đo sai số theo
     từng giá trị `K` (2, 3, 5, 10) để có đường cong "K vs độ chính xác" tham khảo trước khi chạy thật.
   - Test biên: `K=1` (nhảy thẳng 1 bước từ nhiễu về α̂₀) không NaN/Inf/crash.
4. **Dry-run toàn bộ notebook** (Cell 3/5/6/7, cả nhánh thành công và lỗi) — Cell 5 (xác minh patch) cần
   kiểm tra thêm sự tồn tại của `--num_sample_steps` trong `Params.py` và `p_sample`/`p_mean_variance`
   override đúng (ví dụ kiểm tra chuỗi `"sample_schedule"` có trong `Model.py`).

---

## 7. Kế hoạch đóng gói & notebook (dùng `Folder_Base`)

Theo đúng quy trình đã dùng cho Phương án 1 và 2:

1. Folder mới `phuong_an_3_OT_CFM_ketHop/` (ngang hàng `phuong_an_1_OT_noise_scheduler/`,
   `phuong_an_2_CFM_loss/`).
2. Bản fork `DiffMM-OTCFM/` — copy từ 1 bản `DiffMM` **mới tải lại từ GitHub** (không copy từ
   `DiffMM-OT/` hay `DiffMM-CFM/` đã có, để giữ 3 phương án độc lập, dễ so sánh riêng lẻ), áp patch mục
   4, kèm fix `DataHandler.py` (`.A → .toarray()`, bẫy môi trường đã biết).
3. Git repo độc lập, đặt **ngoài** mọi repo khác (ví dụ `E:\NAM_BA\DiffMM-OTCFM`), tự kiểm
   `git rev-parse --show-toplevel`.
4. Notebook từ `Folder_Base/Colab_Template.ipynb` (hoặc nhân bản trực tiếp cấu trúc 7-cell đã dùng ở
   PA1/PA2), điền TODO:
   - Cell 1: thêm cả 3 hyperparameter (`SIGMA_MIN`, `W_CLIP`, `NUM_SAMPLE_STEPS`).
   - Cell 5: kiểm tra `class GaussianDiffusionOTCFM`, cả 3 argument mới trong `Params.py`.
   - Cell 6: `CLI_ARGS` thêm `--sigma_min`, `--w_clip`, `--num_sample_steps`.
   - Cell 7: giữ nguyên `RESULT_REGEX` (định dạng in kết quả của `Main.py` không đổi) — cân nhắc thêm
     cột đo **thời gian chạy Cell 6** (đo bằng `time.time()` trước/sau) vào bảng kết quả, để so sánh
     trực tiếp tốc độ với Phương án 1/2 (đúng tinh thần cột "NFE"/"Thời gian train tổng" trong bảng
     ablation gốc).

---

## 8. Kế hoạch thực nghiệm (mở rộng bảng ablation gốc, mục 6)

| Biến thể | Recall@20 | NDCG@20 | NFE lúc sinh A^m | Thời gian train tổng |
|---|---|---|---|---|
| DiffMM gốc (Diffusion path + ELBO) | (baseline) | (baseline) | T | (baseline) |
| + OT path, giữ ELBO (Phương án 1) | (đã đo) | (đã đo) | T | (đã đo) |
| + CFM loss, giữ Diffusion path (Phương án 2) | (đã đo) | (đã đo) | T | (đã đo) |
| **+ OT path + CFM loss, K=T (Phương án 3, chưa rút gọn D4)** | **?** | **?** | **T** | **?** |
| **+ OT path + CFM loss, K=round(0.6T) (Phương án 3 đầy đủ)** | **?** | **?** | **K < T** | **? (kỳ vọng giảm nhẹ)** |
| **+ OT path + CFM loss, K rất nhỏ (K=1,2 — thử nghiệm biên)** | **?** | **?** | **1-2** | **? (kỳ vọng giảm rõ hơn)** |

Chạy dòng "K=T" **trước** dòng "K=round(0.6T)" để tách bạch: cải thiện Recall/NDCG đến từ **kết hợp
OT+CFM** (không phụ thuộc D4) hay từ **cả D4 rút gọn** — đúng tinh thần ablation nội bộ mà chính bài
Flow Matching đã làm giữa "FM w/ OT" và số NFE khác nhau.

---

## 9. Rủi ro & lưu ý

- **Rủi ro cao nhất trong 3 phương án** (đúng như đánh giá ★★★★☆ trong bản kế hoạch gốc): patch động
  đến 4 hàm thay vì 1-3 như PA1/PA2, cần kiểm chứng D4 kỹ hơn hẳn (mục 6.3) vì đây là phần **hoàn toàn
  mới**, không có tiền lệ trực tiếp từ PA1/PA2 để tái sử dụng.
- **Kỳ vọng lợi ích tốc độ nên thận trọng** (đã nêu ở mục 3.4) — `T=5` mặc định của DiffMM đã rất nhỏ,
  khác hẳn bối cảnh ảnh (`T` hàng trăm/nghìn) mà bài Flow Matching đo NFE.
- **3 phương án dùng chung `w_clip`/`σ_min` mặc định** kế thừa từ PA1/PA2 — nhưng vì kết hợp cả 2 cùng
  lúc, không loại trừ khả năng giá trị tối ưu riêng cho Phương án 3 khác với khi dùng từng phương án
  riêng lẻ; nên quét lại (sweep) độc lập thay vì giả định giá trị cũ vẫn tối ưu.
- **So sánh công bằng:** như PA1/PA2, giữ nguyên mọi hyperparameter khác (kiến trúc `Denoise`, `reg`,
  `ssl_reg`, ...) khi so sánh giữa các dòng trong bảng mục 8.

---

## 10. Tóm tắt 3 phương án (để dễ hình dung toàn cảnh)

| | Phương án 1 | Phương án 2 | Phương án 3 |
|---|---|---|---|
| D1 (path) | OT-linear | VP-style (không đổi) | OT-linear |
| D2 (loss weight) | ELBO/SNR-thật-của-OT (không đổi cách huấn luyện) | CFM | CFM (áp lên OT) |
| D4 (số bước suy luận) | T (không đổi) | T (không đổi) | **K < T (rút gọn, mới)** |
| Số hàm override trong `GaussianDiffusion` | 3 | 1 | **4** |
| Hyperparameter mới | `--sigma_min` | `--w_clip` | `--sigma_min`, `--w_clip`, `--num_sample_steps` |
| Độ khó triển khai (bản kế hoạch gốc) | ★☆☆☆☆ | ★★★☆☆ | ★★★★☆ |
| Đã có sẵn phần nào để tái sử dụng | — | — | **100% D1 từ PA1, 100% công thức D2 từ PA2** — chỉ D4 là mới |

Vì D1 và D2 tái sử dụng gần như nguyên vẹn, phần công sức thật sự của Phương án 3 dồn vào **D4** (mục
3-4) — đây cũng là phần cần kiểm chứng kỹ nhất trước khi bàn giao.
