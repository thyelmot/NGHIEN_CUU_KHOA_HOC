# Phương án 6 — Điểm neo thô học được (Learnable Coarse Anchor) — Kế hoạch chi tiết

> Dựa trên [`DiffMM_OFM_Optimization_Plan.md`](DiffMM_OFM_Optimization_Plan.md) (mục 5, Phương án 6) và
> ý tưởng gốc ở Giai đoạn O1 của [`Optical_Flow_Matching_Review.md`](Optical_Flow_Matching_Review.md).
> Tài liệu này derive công thức đầy đủ, chứng minh điều kiện biên, và **đã kiểm chứng số học** (script
> Python độc lập, xem mục 4) trước khi đề xuất bất kỳ thay đổi code thật nào — đúng tinh thần đã áp dụng
> cho Phương án 5.

---

## 1. Tóm tắt ý tưởng (nhắc lại ngắn gọn)

DiffMM hiện tại (D1, mọi biến thể PA1-5) luôn khởi tạo/thêm nhiễu với **tâm cố định = 0** (không phụ
thuộc user/item) — nghĩa là ở bước nhiễu nhất (`t=T-1`), mọi user đều xuất phát từ **cùng 1 phân phối
nhiễu chung** `N(0, I)`, bất kể user đó có sở thích rõ ràng/lệch nhiều so với xu hướng chung hay không.

OFM (Giai đoạn O1) giải quyết vấn đề tương tự cho optical flow bằng cách **dự đoán trước 1 điểm neo thô**
`x_l` (ước lượng sơ bộ vị trí đích), rồi khởi tạo nhiễu **quanh điểm neo đó** thay vì quanh 1 tâm cố
định — với lý do: sai số ước lượng tăng theo "khoảng cách" cần đi, nên xuất phát gần đích hơn giúp giảm
sai số.

**Phương án 6** chuyển ý tưởng này sang D1 của DiffMM: dự đoán 1 **α_l — ước lượng thô, phụ thuộc từng
user**, về việc user đó có khả năng thích những item nào, rồi dùng α_l làm tâm dịch chuyển của nhiễu
trong forward process.

---

## 2. Phạm vi áp dụng (quan trọng — đọc trước khi triển khai)

Phương án 6 **chỉ tương thích trực tiếp** khi xây trên nền đã dùng framework `μₜ, σₜ` tổng quát + reverse
step kiểu **generalized-DDIM** — tức là:

- ✅ `GaussianDiffusionOT` (Phương án 1)
- ✅ `GaussianDiffusionOTCFM` (Phương án 3)
- ✅ `GaussianDiffusionModalOT` (Phương án 5, κ=0 hoặc κ>0 đều được — xem mục 6)
- ❌ `GaussianDiffusion` gốc và `GaussianDiffusionCFM` (Phương án 2) — **KHÔNG tương thích trực tiếp**,
  vì 2 lớp này vẫn dùng bước reverse suy từ Bayes/ELBO (`posterior_mean_coef1/2`, kế thừa nguyên vẹn từ
  lớp gốc), không phải công thức "suy noise_pred rồi tái tạo x ở t bất kỳ" mà Phương án 6 cần. Muốn dùng
  Phương án 6 với Phương án 2 sẽ cần suy lại toàn bộ khối Bayes/KL divergence — độ khó tương đương làm
  lại từ đầu, **không khuyến nghị**.

Đây là lý do mục "Vị trí áp dụng" trong bản kế hoạch tổng cần làm rõ thêm: **Phương án 6 nên được viết
thành `GaussianDiffusionAnchorOT(GaussianDiffusionOTCFM)`** (kế thừa Phương án 3, để có sẵn cả OT path
lẫn CFM loss đã tổng quát hoá), với công tắc `w=0` mặc định để tương thích ngược.

---

## 3. Derive công thức (CT-6.1 → CT-6.7)

Ký hiệu kế thừa từ Phương án 1/3: `s(t) = μₜ` (hệ số nhân α₀, `s(0)=1` gần dữ liệu, `s(T-1)=0` gần
nhiễu), `σₜ = 1-(1-σ_min)·s(t)`. Thêm 2 đại lượng mới:

- `α_l` — điểm neo thô, tính theo CT-6.1 bên dưới.
- `w ≥ 0` — hệ số cường độ neo (hyperparameter mới, **mặc định `w=0`**).

### CT-6.1 — Cách tính điểm neo `α_l` (không thêm tham số học mới)

```
α_l[u, :] = sigmoid( uEmbeds[u].detach() @ iEmbeds.detach()ᵀ )
```

**Diễn giải:** `uEmbeds`, `iEmbeds` là embedding user/item **đã có sẵn** trong DiffMM (từ nhánh GNN
chính D6, `self.model.getUserEmbeds()`/`getItemEmbeds()`), và **đã được `.detach()`** khi truyền vào
module diffusion trong code gốc (xem `Main.py`: `iEmbeds = self.model.getItemEmbeds().detach()`) — nên
dùng chúng để tính `α_l` **không tạo thêm phụ thuộc gradient/vòng lặp phản hồi nào mới**, hoàn toàn nhất
quán với cách DiffMM đã xử lý các embedding này ở mọi nơi khác.

`sigmoid(dot product)` cho ra 1 điểm số trong khoảng (0,1) cho mỗi cặp (user, item) — cùng thang giá trị
với α₀ (nhị phân 0/1) — đóng vai trò "ước lượng độ tin cậy thô rằng user u thích item i", đúng như OFM
dùng khớp toàn cục (global matching) để ước lượng thô hướng chuyển động.

**Điểm quan trọng — thiết kế "không tham số" (parameter-free), giống nguyên gốc OFM:** OFM tự nêu rõ họ
**chỉ dùng khớp toàn cục không-tham-số** (parameter-free global matching) riêng cho việc suy `x_l`, để
"giảm thiểu chi phí tính toán phát sinh thêm" (`Optical_Flow_Matching_Review.md`, mục 3.3). Thiết kế
CT-6.1 giữ đúng tinh thần này: `α_l` là 1 hàm số đóng (không có trọng số học riêng) của embedding đã có
sẵn — **không thêm 1 tham số học mới nào**, giảm đáng kể rủi ro so với ước tính ban đầu trong bản kế
hoạch tổng (vốn giả định cần "1 nhánh dự đoán" phải huấn luyện riêng).

**Chi phí tính toán:** với batch=1024, số item ~7000 (Sports, trường hợp lớn nhất), `latdim=64`, phép
nhân ma trận `uEmbeds @ iEmbedsᵀ` tốn ~1024×7000×64×2 ≈ 0.9 tỷ FLOPs/batch — **rẻ hơn nhiều** so với 1
lớp lan truyền GNN đầy đủ trên toàn đồ thị (D6) vốn đã chạy mỗi epoch. Rủi ro "chi phí tính toán" nêu ở
bản kế hoạch tổng (mục 7) được **hạ xuống thấp** sau khi tính toán cụ thể này.

### CT-6.2 — Forward process (q_sample) mới

```
αₜ = s(t)·α₀ + σₜ·w·α_l + σₜ·ε,        ε ~ N(0, I)
```

**So sánh với Phương án 1/3 gốc** (`αₜ = s(t)·α₀ + σₜ·ε`): chỉ thêm đúng 1 số hạng `σₜ·w·α_l` — hệ số
nhân với `α_l` dùng **chung** `σₜ` đã có sẵn (không cần hệ số mới), chỉ nhân thêm hằng số vô hướng `w`.

### CT-6.3 — Chứng minh điều kiện biên

- **`t=0`** (gần dữ liệu sạch): `s(0)=1`, `σ(0)=σ_min≈0` (nhỏ, chọn giống PA1) →
  `α₀' = α₀ + σ_min·w·α_l + σ_min·ε ≈ α₀` — vẫn xấp xỉ dữ liệu sạch, sai số cỡ `σ_min` như PA1, **không
  phụ thuộc `w`**.
- **`t=T-1`** (nhiễu tối đa): `s(T-1)=0`, `σ(T-1)=1` → `α_{T-1} = w·α_l + ε ~ N(w·α_l, I)` — đúng như
  thiết kế: phân phối "nhiễu thuần" giờ có **tâm dịch chuyển tới `w·α_l`** thay vì tâm 0.

*(Đã kiểm chứng số học — xem mục 4, Test 1.)*

### CT-6.4 — Reverse process (p_mean_variance) mới — generalized-DDIM

Suy `noise_pred` từ CT-6.2 (thay `α₀` bằng dự đoán của mạng `model_output`):

```
noise_pred = (x − s(t)·model_output − σₜ·w·α_l) / σₜ
```

Tái tạo `x` tại bước `t_prev` bất kỳ trên cùng quỹ đạo (đúng nguyên lý "generalized-DDIM" đã dùng ở
PA1/PA3/PA4/PA5 — công thức đúng cho MỌI cặp `(t, t_prev)`, không bắt buộc liền kề):

```
x_prev = s(t_prev)·model_output + σ_{t_prev}·w·α_l + σ_{t_prev}·noise_pred
```

Biên `t=0` (bước cuối cùng): `s(0)=1, σ(0)=0` → `x_prev = model_output` — số hạng `α_l` **tự triệt
tiêu**, đúng y hệt quy ước đã dùng ở PA1/PA3/PA5 (không cần viết thêm 1 trường hợp đặc biệt nào).

### CT-6.5 — `w=0` phải trùng khít tuyệt đối với Phương án 1/3 gốc

Khi `w=0`, CT-6.2 và CT-6.4 rút gọn đúng về công thức gốc của PA1/PA3 (số hạng `σₜ·w·α_l` biến mất hoàn
toàn, độc lập với giá trị `α_l`) — đây là **công tắc an toàn** để merge, đúng tinh thần đã áp dụng cho
`κ=0` ở Phương án 5.

*(Đã kiểm chứng số học — xem mục 4, Test 2: khớp tuyệt đối, sai lệch `0`.)*

### CT-6.6 — Round-trip với "denoiser hoàn hảo" phải tái tạo đúng `α₀`

Với `model_output ≡ α₀` thật (denoiser hoàn hảo, phép thử tiêu chuẩn đã dùng cho PA1/PA3/PA4/PA5), lặp
CT-6.4 từ `t=T-1` xuống `t=0` phải cho lại đúng `α₀` — **với mọi tổ hợp `(w, α_l)`**, không chỉ khi
`w=0`. Đây là điều kiện cần để xác nhận công thức reverse tự nhất quán với forward.

*(Đã kiểm chứng số học — xem mục 4, Test 3: khớp `α₀` với sai số `<10⁻⁹`, mọi tổ hợp `(w, α_l)` đã thử.)*

### CT-6.7 — Trọng số CFM (Phương án 2/3) BẤT BIẾN với điểm neo — kết quả quan trọng nhất

Suy `μₜ_hiệu_dụng(α₀) := s(t)·α₀ + σₜ·w·α_l` — đây vẫn là 1 hàm **tuyến tính theo `α₀`** (vì `α_l` không
phụ thuộc `α₀`, chỉ là 1 hằng số cộng thêm ứng với mỗi ví dụ huấn luyện). Áp lại đúng phép suy đại số đã
dùng ở Phương án 2 (`Phuong_An_2_CFM_Loss_KeHoachChiTiet.md`, CT-1→CT-4), thay mạng dự đoán `α̂₀` vào vị
trí `α₀` trong công thức `uₜ` (Theorem 3), rồi tính hiệu `v_θ − uₜ`:

```
v_θ − uₜ = [s'(t)·α̂₀ + σₜ'·w·α_l + (σₜ'/σₜ)(x − s(t)·α̂₀ − σₜ·w·α_l)]
         − [s'(t)·α₀  + σₜ'·w·α_l + (σₜ'/σₜ)(x − s(t)·α₀  − σₜ·w·α_l)]
```

Mọi số hạng chứa `σₜ'·w·α_l` xuất hiện **giống hệt nhau ở cả 2 vế** (không phụ thuộc `α̂₀` hay `α₀`) nên
**triệt tiêu hoàn toàn** khi trừ — kết quả rút gọn còn lại:

```
v_θ − uₜ = [s'(t) − (σₜ'/σₜ)·s(t)] · (α̂₀ − α₀)
```

**Đây chính xác là công thức `w_CFM(t)` gốc của Phương án 2/3, không đổi 1 ký tự nào.** Nói cách khác:
**việc thêm điểm neo `α_l` không đòi hỏi tính lại trọng số CFM** — có thể tái sử dụng nguyên vẹn hàm
`_cfm_weight()`/`_precompute_cfm_weight()` đã có sẵn từ Phương án 2/3/5, chỉ cần sửa `q_sample` và
`p_mean_variance`.

*(Đã kiểm chứng số học độc lập — xem mục 4, Test 4: trọng số CFM tính ra **giống hệt tuyệt đối**
(`atol=10⁻¹⁰`) giữa có/không có điểm neo, ở mọi tổ hợp `α_l ∈ {0, 0.715, −3.7, 5.2}` × `w ∈ {0,1,2,10}`
đã thử.)*

**Ý nghĩa thực tiễn:** kết quả này **hạ đáng kể độ khó/rủi ro** so với ước tính ban đầu (★★★★☆ trong bản
kế hoạch tổng) — phần rủi ro lớn nhất được lo ngại trước đó ("cần suy lại công thức trọng số, kiểm tra
tương thích") **đã được giải quyết bằng chứng minh đóng + xác nhận số học**, không còn là ẩn số.

---

## 4. Kết quả kiểm chứng số học (script Python độc lập, không cần GPU/codebase thật)

| Bài kiểm tra | Kết quả |
|---|---|
| **Test 1 — Điều kiện biên** (`t=0` gần α₀, `t=T-1` ~ `N(w·α_l, I)`) | Đạt — `t=0` lệch `α₀` đúng cỡ `σ_min` (không phụ thuộc `w`), `t=T-1` khớp chính xác `w·α_l + ε` |
| **Test 2 — `w=0` trùng khít PA1/PA3 gốc** | Khớp **tuyệt đối** (sai lệch `0`) ở mọi bước `t=0..T-1` |
| **Test 3 — Round-trip với denoiser hoàn hảo** | Tái tạo đúng `α₀` (sai số `<10⁻⁹`) với **mọi** tổ hợp `w∈{0,1,5}` × `α_l∈{0,0.6,−1.2}` đã thử |
| **Test 4 — Trọng số CFM bất biến với điểm neo** | Khớp **tuyệt đối** (`atol=10⁻¹⁰`) giữa có/không điểm neo, mọi tổ hợp `α_l×w` đã thử — xác nhận CT-6.7 |

**Giới hạn của kiểm chứng này (như Giai đoạn A của Phương án 5):** đây là kiểm tra **công thức toán học
tự nhất quán**, dùng số vô hướng/mảng NumPy đơn giản thay cho tensor batch thật và chưa dùng `α_l` được
tính từ embedding thật (CT-6.1 chưa được test số — công thức `sigmoid(dot product)` là hàm chuẩn, rủi ro
sai sót thấp, nhưng vẫn nên xác nhận shape/broadcast đúng khi viết code thật). **Chưa** kiểm chứng: hiệu
năng thực tế (Recall/NDCG) khi `α_l` là ước lượng thô có thể sai (rủi ro đã nêu ở mục 5 bản kế hoạch
tổng vẫn còn nguyên — chỉ có rủi ro *toán học/kỹ thuật* được giải quyết ở đây, không phải rủi ro *thực
nghiệm*).

---

## 5. Thiết kế class (phác thảo, chưa phải code sẵn sàng dán — theo đúng mức độ của mục 6 trong
`Phuong_An_5_Modality_Conditioned_OT_KeHoachChiTiet.md` trước khi có Giai đoạn B thật)

```python
class GaussianDiffusionAnchorOT(GaussianDiffusionOTCFM):
    """
    [Phuong an 6] Ke thua GaussianDiffusionOTCFM (Phuong an 3) de tai su dung nguyen ven ca duong OT
    (D1) lan trong so CFM (D2, da chung minh BAT BIEN voi diem neo - xem CT-6.7). Chi override q_sample
    va p_mean_variance de them so hang diem neo w*alpha_l (CT-6.2, CT-6.4).
    """
    def __init__(self, sigma_min, steps, w_clip=50.0, num_sample_steps=0, anchor_w=0.0):
        super().__init__(sigma_min, steps, w_clip=w_clip, num_sample_steps=num_sample_steps)
        self.anchor_w = anchor_w  # mac dinh 0.0 -> trung khit Phuong an 3 (CT-6.5)

    def _compute_anchor(self, uEmbeds_batch, iEmbeds):
        # CT-6.1 - khong tham so hoc moi, dung embedding da detach() san co
        return torch.sigmoid(uEmbeds_batch @ iEmbeds.t())

    def q_sample(self, x_start, alpha_l, t, noise=None):
        ...  # CT-6.2: cong them self._extract_into_tensor(self.sigma_coef, t, ...) * self.anchor_w * alpha_l

    def p_mean_variance(self, model, x, alpha_l, t):
        ...  # CT-6.4
```

**Lưu ý về chữ ký hàm:** `training_losses`/`p_sample` cần thêm tham số `alpha_l` (hoặc `uEmbeds_batch`
để tự tính bên trong) — tương tự cách Phương án 5's Giai đoạn B đã phải mở rộng chữ ký `p_sample` để
nhận `modal_embeds`, kéo theo cần sửa các điểm gọi trong `Main.py`. Mức độ thay đổi `Main.py` dự kiến
**tương đương Phương án 5** (thêm 1 tham số vào các lệnh gọi `training_losses`/`p_sample` đã có).

---

## 6. Khả năng kết hợp với Phương án 5 (đã được hé lộ ở bản kế hoạch tổng)

Vì Phương án 5 chỉ đổi **tốc độ** quỹ đạo (`τ(t)^g(u,i)` thay cho `s(t)`) còn Phương án 6 chỉ đổi **tâm**
nhiễu (`w·α_l`), 2 thay đổi này **độc lập về mặt công thức** — có thể ghép trực tiếp bằng cách thay
`s(t)` trong CT-6.2/6.4 bằng `τ(t;u,i)` của Phương án 5:

```
αₜ[u,i] = τ(t;u,i)·α₀[u,i] + σₜ[u,i]·w·α_l[u,i] + σₜ[u,i]·ε
```

và lặp lại đúng phép chứng minh CT-6.7 (thay `s(t)` bằng `τ(t;u,i)`) — về nguyên tắc kết quả "bất biến
trọng số CFM với điểm neo" vẫn giữ nguyên vì phép chứng minh chỉ dùng tính chất "α_l không phụ thuộc
α₀", không phụ thuộc dạng cụ thể của hệ số `s(t)`/`τ(t;u,i)`. **Chưa kiểm chứng số học riêng cho tổ hợp
này** — nếu quyết định theo hướng Bước 4 của lộ trình (mục 6, `DiffMM_OFM_Optimization_Plan.md`), cần
làm lại Test 1-4 ở mục 4 với `τ(t;u,i)` thay vì `s(t)` trước khi tin tưởng.

---

## 7. Lộ trình đề xuất (theo đúng khuôn 3 giai đoạn đã dùng cho Phương án 5)

- **Giai đoạn A (kiểm chứng công thức ngoài codebase) — ✅ ĐÃ HOÀN THÀNH trong tài liệu này** (mục 3-4):
  derive CT-6.1→CT-6.7, kiểm chứng số học 4 bài test, xác nhận điều kiện biên + tính bất biến của trọng
  số CFM.
- **Giai đoạn B (patch thật, công tắc `anchor_w=0` mặc định, hồi quy với PA3) — ✅ ĐÃ HOÀN THÀNH, xem
  [`Phuong_An_6_GiaiDoanB_PatchThat/`](Phuong_An_6_GiaiDoanB_PatchThat/README.md):** đã viết
  `GaussianDiffusionAnchorOT(GaussianDiffusionOTCFM)` patch thẳng vào bản clone mới nhất của DiffMM.
  **Đã kiểm chứng hồi quy trên `Model.py` thật**: `anchor_w=0` cho `q_sample`, `p_mean_variance`, và cả
  vòng lặp `p_sample` đầy đủ **trùng khít tuyệt đối** với Phương án 3; `cfm_weight` xác nhận không đổi ở
  mọi `anchor_w` (CT-6.7 trên code thật); `anchor_w>0` không NaN/Inf ở mọi kịch bản biên đã thử. Không
  phát hiện lỗi nào trong quá trình patch (khác Phương án 5) — nhờ đã derive kỹ ở Giai đoạn A trước khi
  viết code. **Chưa** chạy với dữ liệu thật/mạng `Denoise` thật/GPU — đó là phạm vi của Giai đoạn C.
- **Giai đoạn C (đóng gói `Folder_Base`) — ✅ ĐÃ HOÀN THÀNH, xem
  [`phuong_an_6_learnable_anchor/`](phuong_an_6_learnable_anchor/README.md):** đã áp dụng đầy đủ quy
  trình 8 bước như PA1-5 — fork `DiffMM-AnchorOT` (git repo độc lập, đã commit, nằm ngoài mọi repo khác
  tại `E:\NAM_BA\DiffMM-AnchorOT`, **chưa push** vì chưa có link GitHub repo trống được cung cấp),
  notebook `DiffMM_PhuongAn6_AnchorOT_Colab.ipynb` (build từ `Folder_Base/Colab_Template.ipynb`, có sẵn
  xuất PDF tự động), đã dry-run cell-theo-cell cả nhánh thành công lẫn nhánh lỗi. **Chưa** chạy thật
  trên GPU với dữ liệu thật — cần người dùng cung cấp link GitHub repo trống + link Google Drive dữ liệu
  để hoàn tất, rồi tự chạy `Runtime > Run all` trên Colab.

**Đề xuất chưa nên làm ngay:** như đã nêu ở mục 6 bản kế hoạch tổng, nên chạy Phương án 8 (quét
`num_sample_steps`, gần như miễn phí) trước, rồi mới đầu tư vào Giai đoạn B của Phương án 6.
