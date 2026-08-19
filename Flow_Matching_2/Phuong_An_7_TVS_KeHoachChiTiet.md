# Phương án 7 — Tam giác hoá vận tốc (Triangle Velocities Synergy - TVS) — Kế hoạch chi tiết

> Dựa trên [`DiffMM_OFM_Optimization_Plan.md`](DiffMM_OFM_Optimization_Plan.md) (mục 5, Phương án 7) và
> kỹ thuật gốc ở Giai đoạn O2 của [`Optical_Flow_Matching_Review.md`](Optical_Flow_Matching_Review.md).
> Tài liệu này phân tích đầy đủ **khi nào** và **vì sao** TVS cần thiết, derive công thức cho
> trường hợp DiffMM muốn chuyển sang velocity-prediction, và xác định rõ điều kiện kích hoạt trước khi
> triển khai bất kỳ thay đổi code thật nào — đúng tinh thần đã áp dụng cho Phương án 5 và Phương án 6.

---

## 1. Nhắc lại bối cảnh: TVS trong OFM giải quyết vấn đề gì?

### 1.1 Vấn đề gốc trong khung Flow Matching chuẩn khi đổi tâm nhiễu

Khung Flow Matching gốc (Lipman et al. 2022) thiết kế vector field mục tiêu (target vector field) cho
trường hợp **tâm nhiễu = 0** (điểm khởi đầu `x_0 ~ N(0, I)`). Trong trường hợp đó, Conditional Flow
Matching (CFM) cho target:

```
u_t(x | x_1) = (x_1 - x_0) / (1 - t)    [CFM chuẩn]
```

trong đó `x_0` là điểm nhiễu được lấy mẫu từ `N(0, I)`.

Khi OFM (Optical Flow Matching) muốn dùng một **điểm neo thô `x_l` học được** (Giai đoạn O1) làm tâm
phân phối khởi tạo — tức là `x_0 ~ N(x_l, I)` thay vì `N(0, I)` — công thức target ở trên **trở nên phụ
thuộc `x_l`**:

```
u_t(x | x_1, x_l) = (x_1 - x_0) / (1 - t)  với x_0 = x_l + ε,  ε ~ N(0, I)
```

Lúc này, mạng học `v_θ(x_t, t)` phải dự đoán **vận tốc CFM** `u_t` — một đại lượng phụ thuộc vào cả
`x_1` (đích) lẫn `x_l` (điểm neo, vốn là ước lượng thô và **có thể sai**). Khi `x_l` sai lệch so với
hướng `x_1`, vận tốc target `u_t` trở thành một **đại lượng trừu tượng, khó dự đoán trực tiếp** —
đặc biệt trong những trường hợp chuyển động lớn hoặc ước lượng `x_l` lệch nhiều.

### 1.2 Giải pháp TVS của OFM — ý tưởng hình học cốt lõi

OFM phát hiện rằng thay vì dạy mạng học trực tiếp `u_t(x | x_1, x_l)` — một đại lượng tổng hợp khó —
có thể **phân rã** nó thành tổng/hiệu của các đại lượng có ý nghĩa vật lý rõ ràng hơn, thông qua một
quan hệ hình học tam giác (triangle velocities).

OFM định nghĩa 3 quỹ đạo phụ, tất cả đều **xuất phát từ cùng một điểm gốc cố định `x_i`** (trong DiffMM
tương đương: từ `α₀`):

- **Quỹ đạo từ `x_i` → `x_l`:** vận tốc `f_xl = x_l - x_i` (chính là ước lượng thô — trong DiffMM:
  `α_l - α₀`)
- **Quỹ đạo từ `x_i` → `x_1`:** vận tốc `f_gt = x_1 - x_i` (ground-truth optical flow — trong DiffMM:
  `α_ref - α₀`, ý nghĩa tương tự "sở thích thật của user")
- **Quỹ đạo từ `x_i` → `x_0`:** vận tốc `f_xn = x_0 - x_i` (trong DiffMM: điểm nhiễu đã dịch chuyển
  về `x_i` làm gốc)

Quan hệ tam giác (eq 9 trong OFM):

```
f_gt = f_xl + (f_gt - f_xl)
     = f_xl + (x_1 - x_l)      [vận tốc "phần còn lại" từ neo → đích]
```

Nhờ đó, mạng học `v_θ(x_t, t)` CHỈ cần dự đoán **`(x_1 - x_l)`** — phần hiệu giữa đích thật và điểm
neo thô — thay vì dự đoán toàn bộ `f_gt = (x_1 - x_i)` từ đầu. Trong những trường hợp `x_l` đã gần
`x_1`, phần hiệu này rất nhỏ, giúp mạng hội tụ dễ hơn nhiều.

---

## 2. Phân tích vì sao TVS KHÔNG cần thiết ở hiện trạng DiffMM

### 2.1 Sự khác biệt cốt lõi: data-prediction vs velocity-prediction

DiffMM (bao gồm toàn bộ PA1-5 và PA6) huấn luyện mạng `Denoise` theo kiểu **data-prediction**:

```python
model_output = Denoise(α_t, t)   →  đây là α̂₀ (dự đoán trực tiếp dữ liệu sạch)
```

Loss trong mọi phiên bản DiffMM (kể cả PA2/PA3 dùng CFM loss):

```
L = Σ_t  w(t) · ‖α̂₀ - α₀‖²
```

**`α̂₀` luôn là dự đoán của `α₀` trực tiếp** — một đại lượng hoàn toàn xác định, không phụ thuộc `α_l`,
không phụ thuộc `t`, không phụ thuộc cách thêm nhiễu (miễn là quỹ đạo forward vẫn đảm bảo điều kiện
biên: `t=0 → α₀`, `t=T-1 → nhiễu`).

Đây là lý do **đã được chứng minh trong CT-6.7 (Phương án 6):** khi chuyển sang tâm nhiễu `w·α_l`, loss
MSE trên `α̂₀` **không thay đổi hình thức** — chỉ cần cập nhật `q_sample` và `p_mean_variance`, mạng
`Denoise` vẫn học cùng đại lượng `α₀` quen thuộc, không bị "trừu tượng hoá".

### 2.2 TVS chỉ cần thiết khi dùng velocity-prediction

OFM cần TVS vì khung Flow Matching gốc dạy mạng dự đoán **vận tốc** `v_t` — một đại lượng phụ thuộc cả
`x_0` lẫn `x_1` lẫn `x_l` theo cách phức tạp. Khi `x_l` thay đổi, target velocity thay đổi theo và trở
nên khó dự đoán.

Trong DiffMM: nếu và chỉ nếu ta muốn chuyển sang cách huấn luyện **velocity-prediction** (tức là
`Denoise(α_t, t)` trả về `v_t = (α̂₀ - α_t) / ...` hoặc một đại lượng phụ thuộc thời gian) — lúc đó
TVS mới trở nên cần thiết.

**Hiện trạng: không ai trong PA1-6 đề xuất chuyển sang velocity-prediction.** Do đó, Phương án 7 **không
có điều kiện kích hoạt** trong bối cảnh PA6 triển khai theo hướng khuyến nghị (data-prediction).

---

## 3. Điều kiện kích hoạt Phương án 7 (khi nào mới nên cân nhắc)

Phương án 7 chỉ cần thiết khi **đồng thời** xảy ra cả 3 điều kiện sau:

| # | Điều kiện | Cách kiểm tra |
|---|---|---|
| **K1** | Phương án 6 đã được chạy thật và điểm neo `α_l` học được cho tín hiệu cải thiện rõ ràng (NDCG/Recall tốt hơn baseline) | Kết quả thực nghiệm từ Giai đoạn D của Phương án 6 |
| **K2** | Có lý do chính đáng để chuyển sang velocity-prediction (ví dụ: muốn tận dụng trực tiếp khung CFM tổng quát chưa rút gọn, hoặc tích hợp PA7 với một mô hình học đặc trưng ngoài cần `v_t` làm tín hiệu trung gian) | Phân tích thiết kế rõ ràng trước khi bắt đầu code |
| **K3** | Data-prediction đã được thử và cho thấy **mất ổn định hoặc không hội tụ tốt** sau khi đổi tâm nhiễu sang `α_l` học được (điều này chưa hề được quan sát) | Thực nghiệm từ PA6 |

> **Nếu chưa có cả 3 điều kiện này, Phương án 7 nên ở trạng thái "ghi nhận, chưa triển khai".** Đây
> không phải sự thiếu sót của kế hoạch — đây là một quyết định thiết kế chủ đích để tránh lãng phí
> công sức vào một phương án không có khoảng trống thực sự cần lấp trong cấu hình hiện tại.

---

## 4. Derive công thức TVS cho DiffMM (giả sử đã kích hoạt K1-K3)

Phần này được chuẩn bị sẵn để tiết kiệm thời gian nếu điều kiện K1-K3 được thoả mãn. **Chưa có kiểm
chứng số học cho phần này** — cần thực hiện trước khi code thật.

Ký hiệu mới (bổ sung trên nền PA6):

- `α₀` — dữ liệu gốc (lịch sử tương tác thật của user u)
- `α_l` — điểm neo thô, tính theo CT-6.1 của PA6: `sigmoid(uEmbeds @ iEmbeds.T)`
- `α_t` — trạng thái nhiễu tại bước t (theo CT-6.2 của PA6):
  `αₜ = s(t)·α₀ + σₜ·w·α_l + σₜ·ε`
- `v_t` — vận tốc tại thời điểm t (target mới khi chuyển sang velocity-prediction)

### CT-7.1 — Ba quỹ đạo phụ (dịch chuyển O2 vào không gian DiffMM)

**Quỹ đạo 1 — từ `α₀` → `α_l` (neo thô):**
```
ỹ_t = t·α_l + (1-t)·α₀ + σ_min·ε_y,   ε_y ~ N(0, I)
```
Vận tốc tương ứng: `v_y = α_l - α₀` (hằng số, không phụ thuộc t)

**Quỹ đạo 2 — từ `α₀` → `α₀` (quỹ đạo "đứng yên" từ gốc về gốc, dùng làm điểm neo):**
```
z̃_t = t·α₀ + (1-t)·α₀ + σ_min·ε_z = α₀ + σ_min·ε_z
```
Vận tốc tương ứng: `v_z = 0` (bằng 0)

> **Lưu ý dịch thuật từ OFM:** trong OFM, 3 quỹ đạo đều xuất phát từ điểm cố định `x_i`. Trong DiffMM,
> điểm tương đương là `α₀` (dữ liệu thô của user). Ta dịch: `x_i → α₀`, `x_l → α_l`, `x_1 → α_ref`
> (ký hiệu `α_ref` thay vì `α₀` để tránh nhầm lẫn với dữ liệu gốc — `α_ref` là "target sở thích thật
> muốn đạt được sau denoising", về ý nghĩa giống `α₀` nhưng phân biệt về ký hiệu trong khung velocity).

**Quỹ đạo chính — từ `α₀` → `α_ref` (sở thích thật):**
```
x̃_t = t·α_ref + (1-t)·α₀ + σ_min·ε_x,   ε_x ~ N(0, I)
```
Vận tốc target thật: `f_gt = α_ref - α₀`

### CT-7.2 — Quan hệ tam giác vận tốc (Triangle Velocity Relation)

Giống hệt eq (9) trong OFM, áp vào không gian DiffMM:

```
f_gt = (α_ref - α₀)
     = (α_l   - α₀) + (α_ref - α_l)
     = v_xl           + v_residual
```

Trong đó:
- `v_xl = α_l - α₀` — vận tốc từ gốc đến điểm neo thô (**đã biết, không cần mạng dự đoán**)
- `v_residual = α_ref - α_l` — phần "dư" từ điểm neo đến đích thật (**cần mạng dự đoán**)

**Insight cốt lõi:** thay vì dạy `Denoise` dự đoán toàn bộ `f_gt = α_ref - α₀` (một vectơ dài, có thể
lớn), ta chỉ cần dự đoán `v_residual = α_ref - α_l` (phần còn lại sau khi đã biết điểm neo). Khi `α_l`
xấp xỉ tốt `α_ref`, `v_residual` rất nhỏ và mạng học dễ hơn.

### CT-7.3 — Loss TVS mới (thay thế MSE trên `α̂₀`)

Thay vì:
```
L_data = ‖α̂₀ - α₀‖²    [data-prediction cũ — PA1 đến PA6]
```

Dùng loss velocity theo TVS:

```
L_TVS = Σ_t  λ_x · ‖v_θ(x̃_t, t) - f_gt‖²
              + λ_y · ‖v_θ(ỹ_t, t) - v_y‖²
              + λ_z · ‖v_θ(z̃_t, t) - v_z‖²
```

Trong đó:
- `v_θ(·, t)` — mạng `Denoise` được đổi output head để trả về **vận tốc** thay vì `α₀`
- `λ_x, λ_y, λ_z` — trọng số điều chỉnh (trong OFM: `λ_x=λ_y=λ_z=1` mặc định, có thể ablate)
- Quỹ đạo phụ `ỹ_t, z̃_t` đóng vai trò "supervisor" bổ sung giúp ổn định gradient

> **Hình dung dễ nhất:** học lái xe bằng 3 bài tập song song — (1) bài chính: lái từ A đến B thật,
> (2) bài phụ 1: lái từ A đến B_ước_tính (điểm tạm), (3) bài phụ 2: đứng yên tại A. Ba bài này
> cho phép giáo viên tách bạch "em lái sai ở đoạn nào" thay vì chỉ biết "em đến B chưa".

### CT-7.4 — Kết nối ngược về `α̂₀` (để dùng được MSI và inference hiện có)

Khi mạng đã học velocity, để lấy lại ước lượng `α̂₀` phục vụ MSI loss (D3) và top-k rebuild đồ thị
(D4) **không cần thay đổi**, ta dùng quan hệ ngược:

```
α̂₀ = x̃_t - t · v_θ(x̃_t, t)     [tái tạo từ velocity, theo ODE ngược]
    = (x̃_t - t · v_θ) / (1 - t)   [viết lại rõ hơn khi 1-t ≠ 0]
```

Biên `t → 1`: `x̃_t → α_ref`, `v_θ → f_gt`, nên `α̂₀ → α_ref - f_gt + α₀ = α₀` — nhất quán.
Biên `t = 0`: `x̃_0 ≈ α₀`, mọi `v_θ` đều có hệ số nhân `t=0` nên `α̂₀ ≈ x̃_0 ≈ α₀` — nhất quán.

> **Chưa kiểm chứng số học cho CT-7.4 — đây là điều kiện bắt buộc của Giai đoạn A TVS** nếu quyết định
> triển khai.

### CT-7.5 — Xác nhận D3 (MSI) và D4 (Inference) không cần đổi giao diện ngoài

Nếu CT-7.4 tái tạo đúng `α̂₀`, thì:
- **D3 (MSI loss):** vẫn nhận `α̂₀` và `α₀` như cũ → không đổi giao diện
- **D4 (Inference, top-k rebuild):** sau mỗi bước ODE, tái tạo `α̂₀` bằng CT-7.4, chọn top-k như cũ
- **D7 (Multi-task training):** chỉ thay `L_elbo` bằng `L_TVS`, tổng loss `L_dm` vẫn cùng cấu trúc

Đây là điều kiện **bắt buộc** để đảm bảo PA7 chỉ là "thay đổi cục bộ ở D2", không kéo theo thay đổi
hàng loạt ở D3-D7 — đúng nguyên tắc "patch an toàn" đã áp dụng cho PA1-6.

---

## 5. Phạm vi áp dụng (tương thích / không tương thích)

Vì PA7 đòi hỏi cả:
1. Framework `μₜ, σₜ` tổng quát + generalized-DDIM (cần thiết để chuyển đổi velocity ↔ `α̂₀`)
2. **Đã có Phương án 6** (điểm neo `α_l` — không có `α_l`, TVS không có ý nghĩa)

Nên phạm vi tương thích **hẹp hơn PA6**:

| Phiên bản | Tương thích PA7 | Lý do |
|---|---|---|
| `GaussianDiffusionAnchorOT` (PA6, kế thừa PA3) | ✅ Nền tảng chính để phát triển | Có sẵn `μₜ/σₜ`, generalized-DDIM, `α_l` |
| `GaussianDiffusionOT` (PA1) | ⚠️ Cần thêm PA6 trước | Chưa có `α_l` |
| `GaussianDiffusionModalOT` (PA5) | ⚠️ Cần thêm PA6 trước | Có `τ(t)` nhưng chưa có `α_l` |
| `GaussianDiffusion` gốc | ❌ Không tương thích | Không có `μₜ/σₜ` tổng quát |
| `GaussianDiffusionCFM` (PA2) | ❌ Không tương thích | Dùng Bayes/ELBO nguyên gốc |

**Kết luận:** PA7 **phải được xây trên nền PA6** — tức là class đề xuất là:

```python
class GaussianDiffusionTVS(GaussianDiffusionAnchorOT):
    ...
```

---

## 6. Thiết kế class (phác thảo — chưa phải code sẵn sàng dán)

```python
class GaussianDiffusionTVS(GaussianDiffusionAnchorOT):
    """
    [Phuong an 7] Triangle Velocities Synergy cho DiffMM.
    Chi kich hoat khi velocity_mode=True; mac dinh False -> trung khit PA6 (data-prediction).
    Yeu cau: da co GaussianDiffusionAnchorOT (PA6) lam nen.
    """
    def __init__(self, sigma_min, steps, w_clip=50.0, num_sample_steps=0,
                 anchor_w=0.0, velocity_mode=False, lambda_x=1.0, lambda_y=1.0, lambda_z=1.0):
        super().__init__(sigma_min, steps, w_clip=w_clip,
                         num_sample_steps=num_sample_steps, anchor_w=anchor_w)
        self.velocity_mode = velocity_mode  # cong tac bat/tat TVS
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.lambda_z = lambda_z

    def _build_auxiliary_trajectories(self, alpha_0, alpha_l, t):
        """Tao 3 quy dao phu (CT-7.1) cho diem thoi gian t."""
        # Quy dao chinh: x_t tren duong tu alpha_0 -> alpha_ref (~ alpha_0 trong khung nay)
        # Quy dao phu 1: y_t tren duong alpha_0 -> alpha_l
        eps_y = torch.randn_like(alpha_0)
        y_t = t * alpha_l + (1 - t) * alpha_0 + self.sigma_min * eps_y
        # Quy dao phu 2: z_t ~ alpha_0 (dung yen)
        eps_z = torch.randn_like(alpha_0)
        z_t = alpha_0 + self.sigma_min * eps_z
        return y_t, z_t

    def training_losses_tvs(self, model, alpha_0, alpha_l, t, noise=None):
        """
        Tinh loss TVS (CT-7.3) thay the L_elbo khi velocity_mode=True.
        model: mang Denoise da duoc doi output head de tra ve velocity.
        """
        if not self.velocity_mode:
            # Fallback ve data-prediction cua PA6
            return super().training_losses(model, alpha_0, alpha_l, t, noise=noise)

        # Quy dao chinh (CT-7.1 x~_t)
        x_t, noise_x = self.q_sample(alpha_0, alpha_l, t, noise=noise)
        v_pred_x = model(x_t, t)
        v_gt_x = alpha_0  # f_gt = alpha_ref - alpha_0; vi alpha_ref ~ alpha_0, f_gt ~ 0
        # NOTE: can dinh nghia lai "alpha_ref" ro rang truoc khi code that

        # Quy dao phu (CT-7.1)
        y_t, z_t = self._build_auxiliary_trajectories(alpha_0, alpha_l, t)
        v_pred_y = model(y_t, t)
        v_pred_z = model(z_t, t)
        v_gt_y = alpha_l - alpha_0   # vat toc neo (CT-7.2)
        v_gt_z = torch.zeros_like(alpha_0)  # vat toc 0

        loss = (self.lambda_x * F.mse_loss(v_pred_x, v_gt_x) +
                self.lambda_y * F.mse_loss(v_pred_y, v_gt_y) +
                self.lambda_z * F.mse_loss(v_pred_z, v_gt_z))
        return loss

    def reconstruct_alpha0_from_velocity(self, v_pred, x_t, t):
        """CT-7.4 — tai tao alpha_0 tu velocity de dung cho D3/D4."""
        t_scalar = t.float().view(-1, 1)
        return (x_t - t_scalar * v_pred) / (1.0 - t_scalar + 1e-8)
```

**Lưu ý về output head của mạng `Denoise`:** chuyển từ data-prediction sang velocity-prediction đòi hỏi
thay đổi **kỳ vọng về giá trị output** của mạng `Denoise` trong `Model.py`. Không nhất thiết phải thay
toàn bộ kiến trúc MLP — có thể chỉ thêm 1 cờ `velocity_mode` vào class `Denoise` để rescale/reinterpret
output. **Đây là thay đổi nhạy cảm nhất trong toàn bộ PA7** và cần được kiểm chứng riêng.

---

## 7. Giai đoạn A — Kiểm chứng công thức (bắt buộc trước khi code thật)

Khác với PA6 (đã hoàn thành Giai đoạn A trong tài liệu đó), PA7 **chưa có kiểm chứng số học nào**. Dưới
đây là danh sách bài test bắt buộc cần chạy bằng script NumPy độc lập trước khi viết code thật:

| Bài test | Mục đích | Điều kiện đạt |
|---|---|---|
| **Test 7.1 — Quan hệ tam giác tự nhất quán** | Xác nhận `f_gt = v_xl + v_residual` với `α_l, α₀, α_ref` ngẫu nhiên | Sai lệch `< 10⁻¹⁰` |
| **Test 7.2 — CT-7.4: Tái tạo `α₀` từ velocity** | Cho denoiser hoàn hảo (`v_pred = f_gt` thật), tái tạo `α̂₀` phải ≈ `α₀` | Sai lệch `< 10⁻⁸` ở mọi `t ∈ (0,1)` |
| **Test 7.3 — `velocity_mode=False` trùng khít PA6** | Khi tắt TVS, loss và output phải giống hệt PA6 | Sai lệch tuyệt đối = `0` |
| **Test 7.4 — Biên `t→0` và `t→1` ổn định số học** | CT-7.4 có `(1-t)` ở mẫu số — kiểm tra không NaN/Inf ở `t=0.99` và `t=0.001` | Không NaN/Inf với `1e-8` epsilon |
| **Test 7.5 — Gradient ổn định qua 3 loss nhánh** | Dùng PyTorch autograd trên batch nhỏ, kiểm tra không NaN gradient ở `λ_x=λ_y=λ_z=1` | Gradient norm `< 100x` so với PA6 |

> Nếu Test 7.2 thất bại (CT-7.4 không tái tạo đúng `α₀`), **dừng lại, không triển khai** — điều kiện
> bắt buộc để D3/D4 không bị phá vỡ chưa được đảm bảo.

---

## 8. Rủi ro và biện pháp giảm thiểu

| Rủi ro | Mức độ | Biện pháp |
|---|---|---|
| **CT-7.4 không ổn định số học** (chia `1-t` gần 0) | Cao | Thêm epsilon `1e-8` ở mẫu số; chỉ dùng CT-7.4 trong inference, không trong loss | 
| **Gradient "nổ"** từ 3 loss nhánh cùng lúc | Cao | Clip gradient, điều chỉnh `λ_x, λ_y, λ_z`; bắt đầu với `λ_y=λ_z=0.1` thay vì `=1` |
| **Output head `Denoise` không học được velocity tốt** | Rất cao | Thử thêm 1 linear layer riêng cho velocity head; giữ data-prediction head để so sánh |
| **Không cải thiện so với PA6** | Rất cao (khả năng cao nhất) | Đây là rủi ro đã được nêu ngay từ bản kế hoạch tổng — TVS không có đảm bảo lý thuyết cải thiện DiffMM trong hiện trạng |
| **Phát sinh bug trong tương tác PA6 × PA7** | Trung bình | Kiểm tra hồi quy `velocity_mode=False` phải trùng khít PA6 (Test 7.3) |

---

## 9. Lộ trình đề xuất (theo đúng khuôn 3 giai đoạn đã dùng cho PA5, PA6)

- **Giai đoạn A (kiểm chứng công thức, CHƯA làm) — Ưu tiên thấp, chỉ bắt đầu khi K1-K3 được thoả:**
  - Viết script NumPy/PyTorch độc lập, không cần GPU, không cần codebase DiffMM thật
  - Chạy đủ 5 bài test ở mục 7, đặc biệt Test 7.2 (CT-7.4) và Test 7.4 (ổn định số học biên)
  - Nếu tất cả đạt: ghi kết quả vào file này (mục 10 dự phòng bên dưới), chuyển Giai đoạn B

- **Giai đoạn B (patch thật, chỉ sau Giai đoạn A đạt hoàn toàn):**
  - Viết `GaussianDiffusionTVS(GaussianDiffusionAnchorOT)` với cờ `velocity_mode=False` mặc định
  - Kiểm chứng hồi quy: `velocity_mode=False` → trùng khít PA6 (Test 7.3 trên code thật)
  - Kiểm chứng `velocity_mode=True` không NaN/Inf trên batch tensor ngẫu nhiên (không cần data thật)

- **Giai đoạn C (đóng gói Folder_Base và Colab notebook, chỉ sau Giai đoạn B không lỗi):**
  - Fork `DiffMM-AnchorOT` → `DiffMM-TVS` (git repo độc lập)
  - Viết `DiffMM_PhuongAn7_TVS_Colab.ipynb` theo template PA6
  - Thêm tham số dòng lệnh `--velocity_mode` vào `Params.py`

---

## 10. Kết quả kiểm chứng số học (để trống — điền vào khi Giai đoạn A được thực hiện)

| Bài kiểm tra | Kết quả | Ngày thực hiện |
|---|---|---|
| Test 7.1 — Quan hệ tam giác tự nhất quán | *Chưa chạy* | — |
| Test 7.2 — CT-7.4: Tái tạo `α₀` từ velocity | *Chưa chạy* | — |
| Test 7.3 — `velocity_mode=False` trùng khít PA6 | *Chưa chạy* | — |
| Test 7.4 — Biên `t→0` và `t→1` ổn định số học | *Chưa chạy* | — |
| Test 7.5 — Gradient ổn định qua 3 loss nhánh | *Chưa chạy* | — |

---

## 11. Tóm tắt để ra quyết định nhanh

```
Câu hỏi 1: PA6 (Phương án 6) đã chạy thật chưa và kết quả có tích cực không?
  → Chưa / Không tích cực:  DỪNG LẠI — PA7 không có điều kiện kích hoạt (K1 chưa thoả)

Câu hỏi 2 (chỉ nếu PA6 tích cực): Data-prediction của PA6 có bất ổn định không?
  → Không bất ổn định:  DỪNG LẠI — PA7 không cần thiết (K3 chưa thoả); ưu tiên PA8 hoặc PA5+PA6

Câu hỏi 3 (chỉ nếu Q2 = có bất ổn): Có lý do rõ ràng để chuyển sang velocity-prediction không?
  → Không rõ ràng:  DỪNG LẠI — thử các điều chỉnh khác trước (điều chỉnh λ, scheduler, v.v.)
  → Có rõ ràng:     BẮT ĐẦU GIAI ĐOẠN A — chạy 5 bài test số học ở mục 7 trước khi code bất cứ gì
```

**Mức độ ưu tiên hiện tại:** 🔴 Thấp nhất trong PA6-PA9. Ưu tiên theo thứ tự:
PA8 (★☆☆☆☆) → PA6 Giai đoạn D (★★☆☆☆) → PA9 (★★★☆☆) → **PA7 (★★★★★, chỉ khi có đủ K1-K3)**
