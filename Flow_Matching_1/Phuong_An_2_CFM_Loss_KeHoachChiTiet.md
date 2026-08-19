# Kế hoạch chi tiết — Phương án 2: Đổi loss huấn luyện từ ELBO sang CFM (giữ nguyên path diffusion cũ)

> Tài liệu này đào sâu **Phương án 2** trong [`DiffMM_FlowMatching_Optimization_Plan.md`](DiffMM_FlowMatching_Optimization_Plan.md)
> (mục 5). Giả định bạn đã đọc file đó cùng [`../DiffMM/DiffMM_Review.md`](../DiffMM/DiffMM_Review.md)
> và [`Flow_Matching_1_Review.md`](Flow_Matching_1_Review.md). Ký hiệu công thức (eq X) tham chiếu lại
> đúng số thứ tự đã dùng trong 3 file đó, trừ các công thức mới được đánh số riêng (CT-1, CT-2, ...)
> trong tài liệu này.

---

## 1. Nhắc lại phạm vi Phương án 2

- **Vị trí áp dụng:** D2 trong DiffMM (khối Reverse Diffusion / Denoising Model Training, eq 4-13).
- **Giữ nguyên:** D1 (Forward Diffusion — vẫn dùng đúng lịch trình nhiễu VP-style hiện tại của DiffMM,
  KHÔNG đổi sang OT như Phương án 1), D3 (MSI), D4 (Inference), D5, D6, D7, và toàn bộ kiến trúc mạng
  `Denoise` (MLP).
- **Thay đổi:** cách tính loss huấn luyện cho mạng denoiser — từ "ELBO rút gọn thành MSE có trọng số
  SNR" (cách DiffMM đang làm) sang "CFM loss" (cách Flow Matching đề xuất).
- **Mục tiêu đo lường:** đây là "vế thứ 2" của thí nghiệm ablation mà bài Flow Matching đã tự làm
  (Flow_Matching_1_Review.md mục 5) — tách riêng đóng góp của "cách huấn luyện" khỏi "hình dạng đường
  đi", **độc lập** với Phương án 1 (vốn chỉ đổi hình dạng đường đi, giữ nguyên cách huấn luyện).

---

## 2. Phân tích toán học cốt lõi (phần quan trọng nhất — quyết định code sẽ trông như thế nào)

### 2.1 Vấn đề cần giải quyết trước tiên: xung đột tham số hóa (parameterization)

CFM loss "chuẩn" (eq 9, Flow_Matching_1_Review.md) yêu cầu mạng nơ-ron **dự đoán trực tiếp một
vận tốc (velocity)**: `L_CFM = E‖v_t(x;θ) − u_t(x|x₁)‖²`, trong đó `v_t(x;θ)` **là chính output của
mạng**.

Nhưng mạng `Denoise` hiện tại của DiffMM **không được thiết kế để xuất ra vận tốc** — nó xuất ra một
ước lượng của **chính α₀ (dữ liệu gốc)**, gọi là **"data-prediction"** hay **"x0-prediction"** (xem
`Model.py`, hàm `p_mean_variance`: `model_output = model(x, t, False)` rồi dùng thẳng
`model_output` như một ước lượng của `x_start` trong công thức hậu nghiệm Bayes).

Nếu đổi mạng sang xuất ra vận tốc trực tiếp, sẽ kéo theo hàng loạt thay đổi dây chuyền:
- `p_mean_variance`/`p_sample` (suy luận, D4) phải viết lại hoàn toàn cách kết hợp output mạng với
  `x_t` để ra bước khử nhiễu tiếp theo.
- **D3 (MSI, eq 14)** hiện dùng thẳng `model_output` (α̂₀) nhân với embedding modal
  (`α̂₀ · eᵢᵐ`) — nếu output mạng không còn mang ý nghĩa "α₀" nữa thì công thức này **sai về ngữ
  nghĩa**, phải viết lại.
- **D5, D6** gián tiếp phụ thuộc vào đồ thị `A^m` được dựng từ `α̂₀` (qua top-k, D4) — nếu α̂₀ không
  còn được tính trực tiếp từ output mạng mà phải suy ngược từ vận tốc, toàn bộ pipeline downstream
  cũng cần soát lại.

→ Đây chính là rủi ro "trung bình" đã nêu trong bản kế hoạch gốc. **Mục 2.2 dưới đây chứng minh rằng
rủi ro này có thể loại bỏ hoàn toàn** bằng một phép biến đổi đại số đơn giản.

### 2.2 Suy ra: giữ nguyên mạng dự đoán α₀, CFM loss thu gọn thành MSE có trọng số mới

**Thiết lập ký hiệu** (dùng lại đúng ký hiệu DiffMM sẵn có trong `Model.py`, không đổi tên):
- `μₜ := sqrt_alphas_cumprod[t]`, `σₜ := sqrt_one_minus_alphas_cumprod[t]` — 2 mảng đã có sẵn, tính từ
  D1 (không đổi, vì Phương án 2 giữ nguyên đường đi diffusion cũ).
- `x_t = μₜ·x₀ + σₜ·ε`, với `ε` là nhiễu Gauss thật đã lấy mẫu lúc tính `q_sample` (đã có sẵn, dùng lại
  y hệt biến `noise` trong `training_losses`).
- Mạng vẫn xuất ra `x̂₀ := model_output = model(x_t, t)` — **giữ nguyên 100% ý nghĩa hiện tại**.

**Bước 1 — Công thức vận tốc mục tiêu thật** (Theorem 3, eq 15 Flow_Matching_1_Review.md, áp dụng tại
điểm `x = x_t`, `x₁ = x₀`):

```
u_t = (σₜ'/σₜ)·(x_t − μₜ·x₀) + μₜ'·x₀                                          (CT-1)
```

**Bước 2 — "Vận tốc suy ra" từ output của mạng.** Vì mạng chỉ xuất ra `x̂₀`, ta suy ngược "nhiễu ước
lượng" theo đúng công thức forward (đảo ngược `x_t = μₜ·x̂₀ + σₜ·ε̂` ⟹ `ε̂ = (x_t − μₜ·x̂₀)/σₜ`), rồi
tính vận tốc ngụ ý bằng công thức tương tự (CT-1) nhưng thay `x₀` bằng `x̂₀`:

```
v̂_t = μₜ'·x̂₀ + σₜ'·ε̂ = μₜ'·x̂₀ + (σₜ'/σₜ)·(x_t − μₜ·x̂₀)                        (CT-2)
```

**Bước 3 — Hiệu số `v̂_t − u_t`.** Khai triển (CT-1) và (CT-2), số hạng `(σₜ'/σₜ)·x_t` xuất hiện ở cả
2 vế và **triệt tiêu**, chỉ còn lại:

```
v̂_t − u_t = (x̂₀ − x₀) · [ μₜ' − (σₜ'/σₜ)·μₜ ]                                  (CT-3)
```

**Bước 4 — Bình phương 2 vế → chính là CFM loss, viết lại hoàn toàn bằng `x̂₀`, `x₀`:**

```
‖v̂_t − u_t‖² = [ μₜ' − (σₜ'/σₜ)·μₜ ]² · ‖x̂₀ − x₀‖²
             =        w_CFM(t)         · ‖x̂₀ − x₀‖²                            (CT-4)
```

### 2.3 Kết luận (đây là điều thực sự cần code, không hơn không kém)

**CFM loss, khi network vẫn giữ nguyên tham số hóa "dự đoán α₀" như DiffMM hiện tại, chính xác bằng
đúng MSE dữ liệu-gốc hiện có (`‖x̂₀ − x₀‖²`, đã có sẵn trong `training_losses` — biến `mse`), chỉ nhân
thêm một hệ số trọng số mới `w_CFM(t)` phụ thuộc duy nhất vào `t`.**

→ Phương án 2 **không cần đổi mạng, không cần đổi `q_sample`, không cần đổi `p_mean_variance`/
`p_sample` (D4), không cần đổi D3/D5/D6**. Toàn bộ khác biệt so với DiffMM gốc nằm gọn trong **đúng 1
chỗ**: công thức tính `weight` bên trong `training_losses`.

| | DiffMM gốc (D2, eq 11-13) | Phương án 2 |
|---|---|---|
| Target so sánh | `x̂₀` so với `x₀` (MSE) | **Y hệt** — không đổi |
| Trọng số mỗi bước t | `w_ELBO(t) = SNR(t−1) − SNR(t)`, với `SNR(t) = ᾱₜ/(1−ᾱₜ)` | `w_CFM(t) = [μₜ' − (σₜ'/σₜ)·μₜ]²` (CT-4) |
| Cơ sở suy ra trọng số | Suy từ KL divergence giữa 2 phân phối Gauss (Bayes, ràng buộc variance-preserving) | Suy trực tiếp từ định nghĩa CFM loss (Theorem 3 FM), tổng quát cho mọi đường đi Gauss affine, không cần ràng buộc variance-preserving |
| Mạng dự đoán | α₀ (data-prediction) | **Y hệt** — không đổi |
| `p_mean_variance`/`p_sample` (D4) | Dùng `posterior_mean_coef1/2` (Bayes) | **Y hệt** — không đổi (vì path D1 không đổi, công thức Bayes vẫn đúng nguyên trạng) |
| MSI (D3, eq 14) | Dùng thẳng `x̂₀` | **Y hệt** — không đổi |

**Đây là phát hiện quan trọng cần nhấn mạnh:** sự khác biệt thật sự giữa "huấn luyện kiểu ELBO" và
"huấn luyện kiểu CFM" — khi cả hai đều áp dụng lên **cùng một đường đi diffusion VP-style và cùng một
tham số hóa dự đoán α₀** — chỉ là **một hàm trọng số khác nhau theo bước thời gian t**, không phải một
kiến trúc mạng khác hay một "loại loss" khác về bản chất. Điều này cũng lý giải đúng phát hiện thực
nghiệm trong bài Flow Matching (Flow_Matching_1_Review.md, Bảng 1): "FM w/ Diffusion" (CFM + path
diffusion cũ) vượt "DDPM/Score Matching" (cùng path, khác cách huấn luyện) mà **không cần đổi kiến
trúc mạng** — đúng như những gì công thức (CT-4) dự đoán.

### 2.4 Rời rạc hóa `μₜ'`, `σₜ'` cho DiffMM (T bước rời rạc, không phải t liên tục)

CFM (Theorem 3) định nghĩa cho `t ∈ [0,1]` liên tục, còn DiffMM dùng chỉ số nguyên `t ∈ {0,...,T−1}`.
Quy ước hướng: chỉ số `t` của DiffMM **tăng theo hướng nhiễu tăng dần** (t=0 gần dữ liệu gốc nhất,
t=T−1 gần nhiễu thuần nhất) — **ngược hướng** với quy ước `t` của Flow Matching (t=0 là nhiễu, t=1 là
dữ liệu). Do đó đạo hàm rời rạc phải lấy theo chiều "t giảm dần = tiến về phía dữ liệu":

```
μₜ' ≈ μ_{t−1} − μₜ        (xấp xỉ sai phân lùi, dùng mảng sqrt_alphas_cumprod có sẵn)
σₜ' ≈ σ_{t−1} − σₜ        (xấp xỉ sai phân lùi, dùng mảng sqrt_one_minus_alphas_cumprod có sẵn)
```

Biên `t=0`: không có `t−1`. Xử lý bằng cách đặt `w_CFM(0) := 1.0` — **giữ đúng quy ước đã có sẵn**
trong code gốc (`weight = torch.where((ts == 0), 1.0, weight)`), không cần thêm logic mới.

### 2.5 ⚠️ Rủi ro số học đã biết trước (học từ Phương án 1 — phải phòng ngừa ngay từ đầu)

Công thức (CT-4) có số hạng `σₜ'/σₜ`. Ở các bước `t` gần 0 (gần dữ liệu gốc), `σₜ` **rất nhỏ** (gần
`sqrt(noise_min)` theo lịch trình nhiễu hiện tại của DiffMM) → chia cho một số rất nhỏ → `w_CFM(t)` có
thể **rất lớn**, tương tự đúng lỗi đã gặp và phải vá bằng clip khi làm Phương án 1 (xem
`phuong_an_1_OT_noise_scheduler/DiffMM-OT/README.md`, mục "Đã kiểm chứng" — SNR gốc từng "nổ" hàng
trăm nghìn lần ở bước gần t=0). **Phải áp dụng lại đúng kỹ thuật đó ở đây**: tính `w_CFM(t)` cho toàn
bộ `t = 0..T−1` ngay lúc khởi tạo (giống cách `posterior_variance` v.v. được tính 1 lần trong
`calculate_for_diffusion`), rồi `clamp(max=w_clip)` với `w_clip` mặc định (ví dụ 50, giữ nguyên giá trị
đã dùng ở Phương án 1 để nhất quán, có thể tinh chỉnh sau qua thực nghiệm).

---

## 3. Thiết kế patch cụ thể

### 3.1 Vị trí sửa trong `Model.py`

Thêm 1 class mới `GaussianDiffusionCFM(GaussianDiffusion)`, **chỉ override đúng 1 hàm**:
`training_losses`. Mọi hàm khác (`get_betas`, `calculate_for_diffusion`, `q_sample`,
`p_mean_variance`, `p_sample`, `SNR`, `mean_flat`, `_extract_into_tensor`) **kế thừa nguyên vẹn từ lớp
cha `GaussianDiffusion`, không đổi 1 dòng nào** — đây là patch **tối thiểu nhất** trong toàn bộ 5
phương án đã liệt kê trong bản kế hoạch gốc (thậm chí gọn hơn cả Phương án 1, vốn cần override 3 hàm).

```python
class GaussianDiffusionCFM(GaussianDiffusion):
    """
    [Phuong an 2 - DiffMM_FlowMatching_Optimization_Plan.md]
    Giu nguyen 100% duong di diffusion VP-style (D1, ke thua nguyen ven get_betas/
    calculate_for_diffusion/q_sample tu lop cha) va tham so hoa du doan alpha_0 (data-prediction,
    ke thua nguyen ven p_mean_variance/p_sample). Chi doi CACH TINH TRONG SO cua loss trong
    training_losses: tu w_ELBO(t) = SNR(t-1)-SNR(t) (suy tu KL divergence, gia dinh variance-
    preserving) sang w_CFM(t) = [mu_t' - (sigma_t'/sigma_t)*mu_t]^2 (suy truc tiep tu CFM loss,
    Theorem 3 Flow Matching — xem CT-4 trong Phuong_An_2_CFM_Loss_KeHoachChiTiet.md).
    w_CFM duoc tinh 1 lan luc khoi tao (giong posterior_variance...) va CLIP o w_clip de tranh
    "no" gan buoc t=0 (cung ly do da gap va vam o Phuong an 1: chia cho sigma_t rat nho).
    """

    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True, w_clip=50.0):
        super().__init__(noise_scale, noise_min, noise_max, steps, beta_fixed)
        self.w_clip = w_clip
        self._precompute_cfm_weight()

    def _precompute_cfm_weight(self):
        mu = self.sqrt_alphas_cumprod                 # mu_t, da co san tu lop cha
        sigma = self.sqrt_one_minus_alphas_cumprod     # sigma_t, da co san tu lop cha

        mu_prev = torch.cat([mu[:1], mu[:-1]])         # mu_{t-1}, voi mu_{-1} := mu_0 (bien)
        sigma_prev = torch.cat([sigma[:1], sigma[:-1]])

        mu_prime = mu_prev - mu
        sigma_prime = sigma_prev - sigma

        w = (mu_prime - (sigma_prime / sigma.clamp(min=1e-8)) * mu) ** 2
        w[0] = 1.0                                     # bien t=0, dung quy uoc co san cua DiffMM
        self.cfm_weight = w.clamp(max=self.w_clip).cuda()

    def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
        batch_size = x_start.size(0)

        ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)     # KHONG doi — dung ham cua lop cha
        else:
            x_t = x_start

        model_output = model(x_t, ts)                   # KHONG doi — van du doan alpha_0

        mse = self.mean_flat((x_start - model_output) ** 2)   # KHONG doi

        weight = self._extract_into_tensor(self.cfm_weight, ts, mse.shape)  # <-- CHI DOI DONG NAY
        diff_loss = weight * mse

        usr_model_embeds = torch.mm(model_output, model_feats)   # KHONG doi — MSI giu nguyen
        usr_id_embeds = torch.mm(x_start, itmEmbeds)
        gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

        return diff_loss, gc_loss
```

*(Đây là bản nháp thiết kế để lên kế hoạch — khi triển khai thật, làm lại đúng quy trình Bước 1 của
`Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md`: tải lại bản `Model.py` MỚI NHẤT từ
`github.com/HKUDS/DiffMM` ngay trước khi patch, để chắc chắn tên hàm/biến chưa đổi.)*

### 3.2 Sửa ở `Main.py`

Y hệt kiểu Phương án 1: đổi 1 dòng khởi tạo

```python
self.diffusion_model = GaussianDiffusionCFM(args.noise_scale, args.noise_min, args.noise_max, args.steps).cuda()
```

**Không cần thêm argument mới trong `Params.py`** (khác Phương án 1, vốn cần thêm `--sigma_min`) — vì
Phương án 2 tái sử dụng nguyên trạng `noise_scale`/`noise_min`/`noise_max`/`steps` đã có sẵn, chỉ thêm
1 hyperparameter tùy chọn `w_clip` (có thể để mặc định trong code, không cần lộ ra CLI, hoặc thêm
`--w_clip` nếu muốn quét thực nghiệm — tùy độ ưu tiên).

---

## 4. Kế hoạch kiểm chứng (theo đúng Bước 4, `Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md`)

1. **Biên dịch sạch:** `py_compile` trên bản `Model.py`/`Main.py` đã patch, tải mới từ GitHub.
2. **Kiểm tra số học độc lập (bắt buộc, vì có công thức toán mới):**
   - Quét toàn bộ `t = 0..T−1` (với `T` mặc định = `args.steps`, và thử thêm 1-2 giá trị `T` khác để
     chắc chắn không phụ thuộc giá trị cụ thể), in ra `w_CFM(t)` — xác nhận **không NaN/Inf**, không có
     giá trị "nổ" bất thường (đặc biệt kiểm tra kỹ vùng `t` nhỏ, nơi `σₜ` nhỏ nhất).
   - So sánh hình dạng `w_CFM(t)` với `w_ELBO(t)` hiện tại (vẽ/in ra cả 2 dãy theo t) — ghi lại nhận
     xét định tính: trọng số mới có ưu tiên các bước khác nhiều so với cũ không? (đây là dữ liệu hữu
     ích để giải thích kết quả thực nghiệm sau này).
   - Test biên: `w_CFM(0) == 1.0` chính xác (đúng quy ước).
   - Test hồi quy (regression check): với `x_start` và `model_output` giống hệt nhau tại mọi phần tử
     (denoiser "hoàn hảo"), xác nhận `diff_loss == 0` với mọi `t` (vì `mse=0` bất kể trọng số nào).
3. **Dry-run `training_losses` với dữ liệu giả lập** (batch ngẫu nhiên nhị phân thưa, giống cách đã
   làm ở Phương án 1) — xác nhận `diff_loss`, `gc_loss` đều hữu hạn, đúng shape `(batch,)`.
4. **Xác nhận suy luận (D4) không bị ảnh hưởng:** vì `p_mean_variance`/`p_sample` kế thừa nguyên vẹn,
   chỉ cần chạy `p_sample` với 1 model giả lập "dự đoán hoàn hảo" (như đã làm ở Phương án 1) và xác
   nhận vẫn hội tụ đúng về `x_start` thật — nếu test này **giống hệt kết quả đã có ở Phương án 1**
   (không cần sửa gì) thì càng củng cố luận điểm ở mục 2.3 rằng D4 thực sự không bị ảnh hưởng.
5. **Dry-run toàn bộ notebook Colab** theo `Folder_Base/Colab_Template.ipynb` (Cell 3/5/6/7).

---

## 5. Kế hoạch đóng gói & notebook (dùng `Folder_Base`)

Theo đúng quy trình đã có:

1. Tạo folder mới `phuong_an_2_CFM_loss/` (ngang hàng với `phuong_an_1_OT_noise_scheduler/`), copy cấu
   trúc từ `Folder_Base/`.
2. Bản fork `DiffMM-CFM/` — copy từ 1 bản `DiffMM` **mới tải lại từ GitHub** (không copy từ
   `DiffMM-OT/` của Phương án 1, để 2 phương án độc lập, dễ so sánh riêng lẻ đúng tinh thần ablation),
   áp patch mục 3, kèm luôn fix tương thích scipy `.A → .toarray()` đã biết là cần thiết cho Colab
   (xem bảng "bẫy môi trường" trong `Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md`).
3. Git repo độc lập, đặt **ngoài** mọi repo khác (bài học từ Phương án 1 — ví dụ
   `E:\NAM_BA\DiffMM-CFM`), tự kiểm bằng `git rev-parse --show-toplevel`.
4. Notebook từ `Folder_Base/Colab_Template.ipynb`, điền TODO:
   - `MAIN_SCRIPT_NAME = "Main.py"` (không đổi).
   - `REQUIRED_DATA_FILES` — y hệt Phương án 1 (`trnMat.pkl`, `tstMat.pkl`, `image_feat.npy`,
     `text_feat.npy`, `+audio_feat.npy` nếu tiktok).
   - Cell 5 (xác minh patch): kiểm tra `"class GaussianDiffusionCFM" in model_src` và
     `"GaussianDiffusionCFM(args.noise_scale" in main_src` — **không còn `--sigma_min`** trong
     `Params.py` (khác Phương án 1), nên bỏ điều kiện đó.
   - Cell 6: `DATASET_HP` — y hệt Phương án 1 (lấy từ README gốc DiffMM), **không thêm** flag
     `--sigma_min` vào `CLI_ARGS` (vì Phương án 2 không cần argument mới).
   - Cell 7: `RESULT_REGEX` — y hệt Phương án 1 (định dạng dòng `"Best epoch : ..."` không đổi vì
     `Main.py` không đổi cấu trúc in kết quả).

---

## 6. Kế hoạch thực nghiệm (đối chiếu Bước 3, mục 6 của bản kế hoạch gốc)

Theo đúng roadmap gốc: chạy **Phương án 2 riêng biệt**, so sánh với:
- DiffMM gốc (baseline).
- Phương án 1 (OT path, giữ ELBO) — đã có sẵn kết quả nếu đã chạy trước đó.

trên cả 3 dataset (TikTok, Amazon-Baby, Amazon-Sports), đo `Recall@20`, `NDCG@20`, `Precision@20`, và
**thời gian huấn luyện tổng** (Phương án 2 không giảm NFE lúc inference như Phương án 1/3, nên cột NFE
trong bảng ablication gốc sẽ giữ nguyên `T` — điều này **dự kiến trước**, không phải lỗi).

Mở rộng bảng ablation gốc (mục 6, file `DiffMM_FlowMatching_Optimization_Plan.md`) thêm 1 dòng:

| Biến thể | Recall@20 | NDCG@20 | NFE lúc sinh A^m | Thời gian train tổng |
|---|---|---|---|---|
| DiffMM gốc (Diffusion path + ELBO) | (baseline) | (baseline) | T | (baseline) |
| + OT path, giữ ELBO (Phương án 1) | (đã đo) | (đã đo) | T | (đã đo) |
| **+ CFM loss, giữ Diffusion path (Phương án 2)** | **?** | **?** | **T (không đổi, như dự kiến)** | **? (kỳ vọng gần bằng baseline — cùng số phép tính mỗi bước, chỉ khác 1 hệ số nhân)** |

Nếu Phương án 2 cải thiện Recall/NDCG mà **không đổi NFE**, đây là bằng chứng cho luận điểm ở mục 2.3:
lợi ích đến từ **cách trọng số hóa loss tốt hơn**, độc lập hoàn toàn với chi phí runtime.

---

## 7. Rủi ro còn lại & điều cần theo dõi khi chạy thật

- **`w_clip` là 1 siêu tham số mới cần quét (sweep)** — giá trị 50 chỉ là điểm khởi đầu hợp lý (kế
  thừa từ Phương án 1), chưa có cơ sở lý thuyết chứng minh là tối ưu cho bài toán này; nên thử thêm
  ít nhất 2-3 giá trị khác (ví dụ 10, 100) nếu kết quả ban đầu không rõ ràng.
- **Xấp xỉ sai phân rời rạc (mục 2.4) chỉ là 1 lựa chọn khả dĩ**, không phải cách duy nhất để rời rạc
  hóa `μₜ'`, `σₜ'` — các lựa chọn khác (ví dụ sai phân trung tâm, hoặc suy đạo hàm giải tích trực tiếp
  từ công thức `get_betas()` thay vì sai phân số) có thể cho kết quả khác biệt; nếu Phương án 2 không
  cho cải thiện rõ, đây là điểm đầu tiên nên xem lại trước khi kết luận "CFM loss không có tác dụng".
- **So sánh công bằng:** giữ nguyên `noise_scale`/`noise_min`/`noise_max`/`steps`/mọi hyperparameter
  khác giống hệt DiffMM gốc khi so sánh — đúng tinh thần "chỉ đổi 1 biến" đã nêu ở mục 7 của bản kế
  hoạch gốc.
- **Không kỳ vọng giảm NFE** như Phương án 1/3 — mục tiêu của Phương án 2 chỉ là "huấn luyện ổn định/
  hiệu quả hơn ở cùng chi phí runtime", không phải "sinh mẫu nhanh hơn". Cần truyền đạt đúng kỳ vọng
  này khi báo cáo kết quả, tránh so sánh nhầm tiêu chí với Phương án 1.

---

## 8. Tóm tắt khác biệt Phương án 1 vs Phương án 2 (để dễ hình dung 2 hướng đi song song)

| | Phương án 1 | Phương án 2 |
|---|---|---|
| Đường đi diffusion (D1) | **Đổi** sang OT-linear | **Giữ nguyên** VP-style |
| Tham số hóa mạng | Giữ nguyên (dự đoán α₀) | Giữ nguyên (dự đoán α₀) |
| Cách tính trọng số loss | Suy lại từ đầu (SNR thật của path OT, `μₜ²/σₜ²` clip) | Suy lại từ đầu (CT-4, `[μₜ'−(σₜ'/σₜ)μₜ]²` clip) |
| Số hàm phải override trong `GaussianDiffusion` | 3 (`q_sample`, `p_mean_variance`, `SNR`) | **1** (`training_losses`) |
| D4 (Inference) có đổi không | Có (do path đổi, công thức Bayes cũ không còn đúng) | **Không** (path không đổi) |
| NFE lúc sinh `A^m` | Không đổi ở Phương án 1 thuần (chỉ giảm nếu kết hợp Phương án 4/3) | Không đổi |
| Argument CLI mới | `--sigma_min` | Không cần (có thể thêm `--w_clip` tùy chọn) |
| Rủi ro chính | Đã biết rõ & đã vá (SNR "nổ" gần t=0) | Tương tự — đã lường trước & thiết kế sẵn cách vá (mục 2.5) trước khi code |

Hai phương án này **độc lập và có thể chạy song song** — đúng đúng tinh thần "ablation tách bạch 2
đóng góp" mà roadmap gốc đã đề ra (Bước 2 và Bước 3). Khi cả hai đều đã có kết quả, Phương án 3 (kết
hợp cả hai) sẽ tận dụng lại gần như nguyên vẹn code của cả 2 phương án này.
