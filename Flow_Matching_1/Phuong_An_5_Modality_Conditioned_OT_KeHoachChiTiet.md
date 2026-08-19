# Kế hoạch chi tiết — Phương án 5: Modality-conditioned OT path (tích hợp MSI vào đường đi)

> Tài liệu này đào sâu **Phương án 5** trong [`DiffMM_FlowMatching_Optimization_Plan.md`](DiffMM_FlowMatching_Optimization_Plan.md)
> (mục 5) — phương án được chính bản kế hoạch gốc xếp hạng **★★★★★ (khó nhất)** và gọi là **"hướng
> nghiên cứu dài hạn... chưa có tiền lệ trực tiếp trong cả 2 bài báo gốc"**. Khác với Phương án 1-4 (đều
> **tái sử dụng** công thức đã có sẵn từ FM hoặc từ chính các phương án trước), Phương án 5 đòi hỏi
> **tự thiết kế công thức toán mới từ đầu** — tài liệu này thực hiện đúng yêu cầu đó: thiết kế, **chứng
> minh điều kiện biên**, và **kiểm chứng số học** trước khi đưa ra bất kỳ khuyến nghị triển khai nào.
> Giả định bạn đã đọc `DiffMM_Review.md`, `Flow_Matching_1_Review.md`, và nên đọc qua
> [`Phuong_An_2_CFM_Loss_KeHoachChiTiet.md`](Phuong_An_2_CFM_Loss_KeHoachChiTiet.md) vì thiết kế loss ở
> đây xây trên nền CFM đã kiểm chứng ở đó.

---

## 1. Nhắc lại phạm vi & đặt lại vấn đề cho chính xác

Bản kế hoạch gốc mô tả ý tưởng ở mức khái niệm: *"định nghĩa μₜ, σₜ phụ thuộc luôn vào đặc trưng modal
eᵢᵐ... để bản thân quỹ đạo sinh ra đã thiên vị đúng theo modal, không cần loss phụ trợ nữa"*. Trước khi
thiết kế công thức, cần làm rõ **3 câu hỏi mà bản gốc chưa trả lời** (đây chính là phần "tự thiết kế"):

1. `α₀` là một **vector nguyên cả tập item** (`ℝ^|I|`), còn `eᵢᵐ` là **embedding của từng item riêng
   lẻ** — vậy "μₜ phụ thuộc vào modal" nghĩa là mỗi **tọa độ item** của `μₜ` có một đường đi riêng
   (không đồng nhất cho cả vector như Phương án 1), hay có cách nào khác?
2. Modal "phù hợp" ở đây là phù hợp **với cái gì** — với chính item đó, hay với **gu của user cụ thể**
   đang được sinh dữ liệu (đúng tinh thần MSI gốc — MSI vốn so sánh theo **từng user** qua
   `usr_model_embeds`/`usr_id_embeds`, eq 14 DiffMM)?
3. Nếu mỗi tọa độ có tốc độ riêng, làm sao đảm bảo **vẫn thỏa điều kiện biên bắt buộc** của Flow
   Matching (`μ₀=0, σ₀=1` độc lập hoàn toàn với `x₁`/modal; `μ₁=x₁, σ₁=σ_min`) — đây là điều kiện Flow
   Matching **yêu cầu tuyệt đối** (Theorem 1, Flow_Matching_1_Review.md) để đường đi biên hội tụ đúng
   về phân phối dữ liệu thật ở `t=1`.

Câu trả lời cho cả 3 câu hỏi này quyết định toàn bộ thiết kế bên dưới.

---

## 2. Thiết kế công thức (phần cốt lõi, tự xây dựng — không có sẵn trong 2 bài báo gốc)

### 2.1 Định nghĩa "độ phù hợp modal" theo từng user (trả lời câu hỏi 2)

Với mỗi user `u` (1 hàng trong batch, có vector tương tác thật `α₀[u,:] ∈ {0,1}^|I|`), định nghĩa
**tâm modal của user** — trung bình embedding modal của các item user đã thật sự thích:

```
c_u^m = normalize( Σ_{j : α₀[u,j]=1} eⱼᵐ )                                        (CT-5.1)
```

Rồi định nghĩa **độ phù hợp modal của item i đối với user u**:

```
φ_{u,i} = cos_sim( eᵢᵐ, c_u^m ) ∈ [−1, 1]                                          (CT-5.2)
```

*Diễn giải:* `φ_{u,i}` dương lớn nghĩa là "item i có nội dung modal (ảnh/chữ/âm thanh) gần với gu của
user u", âm nghĩa là "lệch gu". Đây chính là phần **thay thế vai trò của MSI** — thay vì tính
`||α̂₀·eᵢᵐ − α₀·eᵢ||²` như một loss riêng (eq 14 DiffMM), thông tin "item nào hợp gu user nào" giờ được
đưa thẳng vào **hình dạng đường đi** thông qua `φ_{u,i}`.

**Lưu ý quan trọng về chi phí tính toán:** `φ_{u,i}` phụ thuộc cả `u` lẫn `i` → là **1 ma trận
`(batch, |I|)` tính lại mỗi batch** (không phải 1 mảng cố định tính 1 lần lúc khởi tạo như Phương án
1-4). Đây là khác biệt kiến trúc quan trọng nhất của Phương án 5, sẽ nhắc lại ở mục 4.

### 2.2 Uốn thời gian theo từng tọa độ (trả lời câu hỏi 1 + 3)

Ý tưởng: thay vì mọi tọa độ item dùng chung 1 "đồng hồ" `t` như Phương án 1
(`μₜ = t·α₀`, `σₜ = 1−(1−σ_min)·t`), cho mỗi cặp `(u,i)` một đồng hồ riêng `τ_{u,i}(t)` — **nhưng bắt
buộc `τ_{u,i}(0)=0` và `τ_{u,i}(1)=1` với MỌI `(u,i)`** (để giữ đúng điều kiện biên).

**Lựa chọn công thức (đã kiểm chứng số học ở mục 3):**

```
g_{u,i} = clip( exp(−κ·φ_{u,i}) , g_min, g_max )        (κ ≥ 0: siêu tham số độ mạnh hiệu ứng modal)
τ_{u,i}(t) = t ^ g_{u,i}                                                            (CT-5.3)
```

```
μₜ[u,i] = τ_{u,i}(t) · α₀[u,i]                                                     (CT-5.4)
σₜ[u,i] = 1 − (1−σ_min) · τ_{u,i}(t)                                               (CT-5.5)
```

**Vì sao chọn đúng dạng lũy thừa `t^g` (không phải cộng/trừ tuyến tính):** với **mọi** giá trị `g>0`,
`t^g` luôn thỏa `0^g=0` và `1^g=1` — **tự động đúng điều kiện biên mà không cần ràng buộc thêm gì**,
đây là lý do chọn dạng này thay vì ví dụ `τ=t+κφ` (dạng cộng tuyến tính, sẽ **phá vỡ** điều kiện biên
tại `t=1` trừ khi thêm ràng buộc phức tạp hơn).

**Trực giác:** `φ_{u,i}>0` (item hợp gu) → `g_{u,i}<1` → `τ_{u,i}(t) > t` với `t∈(0,1)` → tọa độ này
"chạy nhanh hơn đồng hồ chung", tức là quỹ đạo **cam kết vào giá trị thật sớm hơn** (mô hình "biết"
sớm hơn rằng item này nên được đề xuất). `φ_{u,i}<0` (lệch gu) → `g_{u,i}>1` → tọa độ "chạy chậm hơn",
giữ trạng thái mơ hồ/nhiễu lâu hơn.

**Trường hợp suy biến (đã kiểm chứng ở mục 3):** khi `κ=0` (hoặc `φ_{u,i}=0` với mọi cặp), `g_{u,i}=1`
với mọi `(u,i)` → `τ_{u,i}(t)=t` → **công thức thu gọn về đúng đường OT-linear của Phương án 1**. Đây
là tính chất quan trọng: Phương án 5 là **một họ tổng quát hóa thực sự** của Phương án 1 (không phải
một cơ chế hoàn toàn tách biệt), giúp việc so sánh/ablation rất tự nhiên (`κ=0` chính là baseline PA1).

### 2.3 Trường véc-tơ mục tiêu — tổng quát hóa Theorem 3 theo từng tọa độ (Hệ quả mới, chưa có sẵn)

Theorem 3 (Flow_Matching_1_Review.md) được phát biểu cho 1 cặp `(μₜ,σₜ)` dùng chung cho cả vector. Với
hiệp phương sai **chéo (diagonal)** — đúng trường hợp ở đây, vì mỗi tọa độ có `μₜ,σₜ` riêng — định lý
áp dụng **độc lập theo từng tọa độ** (chứng minh: flow chính tắc `ψₜ,i(x) = σₜ,i·x_i + μₜ,i` tách biệt
hoàn toàn theo `i`, nên đạo hàm `d/dt ψₜ,i` cũng tách biệt theo `i`, dẫn thẳng đến công thức Theorem 3
áp cho từng `i`):

```
uₜ,i(x|x₁) = (σₜ,i'/σₜ,i)·(x_i − μₜ,i) + μₜ,i'                                     (CT-5.6, "Theorem 3 theo tọa độ")
```

### 2.4 Trọng số CFM theo từng tọa độ — tổng quát hóa CT-4 của Phương án 2

Áp lại đúng phép suy CT-1→CT-4 (Phương án 2) **cho từng tọa độ** (mạng vẫn dự đoán `α₀` trực tiếp,
không đổi tham số hóa):

```
CFM loss = Σᵢ w_{u,i}(t) · (x̂₀[u,i] − x₀[u,i])² ,   w_{u,i}(t) = [μₜ,i' − (σₜ,i'/σₜ,i)·μₜ,i]²      (CT-5.7)
```

**Khác biệt quan trọng so với Phương án 2:** ở Phương án 2, trọng số `w_CFM(t)` là **1 số vô hướng cho
cả batch/cả vector** (chỉ phụ thuộc `t`), nhân vào **sau khi** đã lấy trung bình MSE qua các item
(`mean_flat`). Ở đây, `w_{u,i}(t)` là **1 ma trận `(batch, |I|)`**, phải nhân vào **trước khi** lấy
trung bình (nếu nhân sau thì vô nghĩa vì mỗi tọa độ đã bị gộp mất). Đây là thay đổi cấu trúc bắt buộc
đối với `training_losses`, không thể chỉ đổi 1 dòng như Phương án 2.

### 2.5 Vai trò của MSI (D3) sau khi áp dụng Phương án 5

**Loại bỏ hoàn toàn `L_msi` (eq 14) và siêu tham số `λ₀`** — đúng lợi ích đã hứa trong bản kế hoạch
gốc ("giảm số hyperparameter cần tune"). Thông tin modal giờ nằm trong `φ_{u,i}` → `g_{u,i}` → hình
dạng đường đi, không cần ép bằng loss riêng nữa. **Khuyến nghị giữ lại 1 công tắc bật/tắt** (`use_msi`)
để có thể chạy ablation "PA5 có/không có MSI song song" — vì đây là thay đổi lớn, cần bằng chứng thực
nghiệm trước khi khẳng định bỏ MSI hoàn toàn là đúng đắn.

---

## 3. Kiểm chứng bằng số học (đã chạy thật trước khi viết khuyến nghị — không chỉ suy luận trên giấy)

### 3.1 Điều kiện biên — kiểm chứng bắt buộc trước tiên

Quét `κ ∈ {0, 1, 3, 8}` với 8 item có `φ̃ᵢ` trải đều từ `−1` đến `1`:

| κ | mu(s=0)=0? | mu(s=1)=1? | sigma(s=0)=1? | sigma(s=1)=σ_min? |
|---|---|---|---|---|
| 0.0 | ✓ | ✓ | ✓ | ✓ |
| 1.0 | ✓ | ✓ | ✓ | ✓ |
| 3.0 | ✓ | ✓ | ✓ | ✓ |
| 8.0 | ✓ | ✓ | ✓ | ✓ |

**Đúng như chứng minh ở mục 2.2** — điều kiện biên thỏa mãn **chính xác tuyệt đối** ở mọi `κ`, không
phải xấp xỉ. Đồng thời xác nhận `κ=0` cho `g_{u,i}=1` với mọi item, quỹ đạo trùng khít đường OT-linear
gốc (sai số `0.0` tuyệt đối khi so `τ(s)` với `s`).

### 3.2 ⚠️ Phát hiện rủi ro số học (giống nhóm lỗi đã gặp ở PA1/PA2 — cùng cách vá)

Quét trọng số `w_{u,i}(t)` (CT-5.7, chưa nhân `α₀`) trên lưới rời rạc `T=5` (mặc định DiffMM) với các
`g_i` cực biên:

```
g_i=0.2 : w_CFM_item = [1.0, 9.73, 0.75, 1.67, 3126.29]   <- no gan s=1 rat lon
g_i=1.0 : w_CFM_item = [1.0, 0.11, 0.25, 0.99, 62501.62]
g_i=5.0 : w_CFM_item = [1.0, 0.00, 0.00, 0.07, 581719.13]
```

**Nguyên nhân:** giống hệt lý do đã gặp ở Phương án 1/2 — tại bước gần `s=1` (gần dữ liệu),
`σₜ,i → σ_min` (rất nhỏ), chia cho số rất nhỏ → trọng số "nổ". Đây **không phải lỗi thiết kế mới**, mà
là **cùng 1 loại rủi ro đã biết**, và **cách vá cũng giống hệt**: `clip(w, max=w_clip)`. Sau khi clip ở
`w_clip=50` (kế thừa từ PA1-3): mọi giá trị hữu hạn, hợp lý (`[1.0, 9.73, 0.75, 1.67, 50.0]` cho
`g_i=0.2`, tương tự cho các `g_i` khác).

### 3.3 ⚠️ Phát hiện rủi ro MỚI (chỉ có ở Phương án 5, cần lưu ý riêng): quỹ đạo quá lệch khi `T` nhỏ

Với `T=5` (mặc định DiffMM — **rất thô**), quỹ đạo `μᵢ(s)` ở các `g_i` cực biên bị **méo mạnh**:

```
g_i=5.0 : mu = [0.0, 0.001, 0.031, 0.237, 1.0]   <- gan nhu khong doi cho toi buoc CUOI CUNG moi nhay vot
g_i=0.2 : mu = [0.0, 0.758, 0.871, 0.944, 1.0]   <- gan nhu "lo" gia tri that ngay tu buoc 2
```

Với `g_i=5.0`, tín hiệu học gần như **không đổi** qua 3/5 bước đầu rồi mới nhảy vọt ở bước cuối — dễ
gây khó học (gradient signal quá tập trung vào 1 bước). Với `g_i=0.2`, mô hình gần như "lộ đáp án" chỉ
sau 1 bước, giảm ý nghĩa của các bước trung gian. **Khuyến nghị:** giới hạn dải `g_i` hẹp hơn nhiều so
với mặc định lý thuyết (`[0.2, 5.0]`) — quét thêm cho thấy dải `[0.5, 2.0]` giữ quỹ đạo trải đều hợp lý
qua cả 5 bước ở `T=5`:

```
g_i=0.5 : mu = [0.0, 0.500, 0.707, 0.866, 1.0]
g_i=2.0 : mu = [0.0, 0.062, 0.250, 0.562, 1.0]
```

**Khuyến nghị cụ thể:** đặt `g_min=0.5, g_max=2.0` làm mặc định (thay vì `[0.2,5.0]` chỉ mang tính lý
thuyết), và **ưu tiên thử nghiệm Phương án 5 sau khi đã có kết quả của Phương án 4** (rút gọn D4) — vì
nếu dự án sau này tăng `T` (số bước diffusion) để cải thiện chất lượng, dải `g_i` hợp lý cũng sẽ rộng
hơn, giảm rủi ro méo quỹ đạo này.

---

## 4. Khác biệt kiến trúc cốt lõi so với Phương án 1-4 (đọc kỹ trước khi ước lượng công sức triển khai)

**Đây là điểm quan trọng nhất cần hiểu trước khi quyết định có triển khai Phương án 5 hay không.**

| | Phương án 1-4 | Phương án 5 |
|---|---|---|
| `μₜ, σₜ` được tính | **1 lần duy nhất** lúc khởi tạo model (`__init__`), lưu thành mảng `(T,)` cố định | **Tính lại MỖI BATCH** trong lúc huấn luyện (vì phụ thuộc `α₀` và `eᵐ` của batch đó) — mảng `(batch, T, |I|)` |
| Vị trí đặt logic | `__init__`/`_precompute_...` | Phải chuyển vào `q_sample`/`training_losses`/`p_mean_variance` — nhận thêm tham số `itmEmbeds`/`model_feats`/`α₀` mà trước đây các hàm này **không cần** |
| Chi phí bộ nhớ/tính toán | Không đáng kể (mảng `(T,)`, vài chục phần tử) | Đáng kể hơn — ma trận `(batch, |I|)` cho `φ`, và tương tự cho `μₜ,σₜ,w` tại mỗi bước `t` được lấy mẫu (vẫn cùng bậc độ lớn với các tensor `(batch,|I|)` đã có sẵn trong DiffMM — không phải vấn đề chặn cứng, nhưng là **thay đổi luồng dữ liệu**, không chỉ thêm 1 class) |
| Số hàm cần override | 1-4 (kế thừa phần lớn từ `GaussianDiffusion`) | **Toàn bộ chữ ký hàm cần đổi** (`q_sample`, `training_losses`, `p_mean_variance`, `p_sample` đều cần nhận thêm `α₀`/`eᵐ` làm tham số) — gần như viết lại `GaussianDiffusion` thành 1 lớp khác hẳn, khó giữ nguyên "chỉ thêm không sửa đè" như các phương án trước |
| Tương thích top-k rebuild (D4) | Không đổi | Cần xem lại: top-k hiện chọn theo `α̂₀` — với path bất đối xứng theo modal, cần xác nhận việc "chọn top-k" vẫn cho ra đồ thị `A^m` hợp lý (không thiên vị quá mức về phía item có `g_i` nhỏ, vốn "lộ đáp án" sớm hơn) |

**Kết luận thực dụng:** Phương án 5 **không thể** làm theo đúng khuôn "thêm 1 class, override vài
hàm, giữ `GaussianDiffusion` gốc nguyên vẹn" như PA1-4 — nó đòi hỏi **thay đổi giao diện (interface)**
của cả pipeline diffusion (các hàm cần biết `α₀` thật và `eᵐ` ngay trong lúc suy luận, không chỉ lúc
huấn luyện). Đây chính là lý do bản kế hoạch gốc xếp ★★★★★ và gọi là "hướng nghiên cứu" — **không phải
một phép phóng đại**, mà là đặc điểm kỹ thuật thật sự.

---

## 5. Lộ trình triển khai đề xuất (chia giai đoạn — KHÔNG làm 1 lần như PA1-4)

Không khuyến nghị áp dụng ngay quy trình `Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md` "làm 1 lần xong"
như PA1-4. Đề xuất chia nhỏ:

**Giai đoạn A — Kiểm chứng ý tưởng ngoài codebase — ✅ ĐÃ HOÀN THÀNH, xem
[`Phuong_An_5_GiaiDoanA_GradCheck/`](Phuong_An_5_GiaiDoanA_GradCheck/README.md):**
Đã làm: chứng minh điều kiện biên, kiểm tra quỹ đạo/trọng số bằng script độc lập (mục 3), **và** cài
đặt độc lập `modal_affinity`/`per_item_schedule`/`pa5_loss` (CT-5.1→CT-5.7) trên dữ liệu giả lập, đối
chiếu gradient autograd với sai phân hữu hạn độc lập — **khớp tuyệt đối** (sai lệch `0.0`) ở mọi kịch
bản đã thử, gồm cả 2 trường hợp biên số học quan trọng nhất (user không có tương tác nào; κ rất lớn
gây bão hoà `clamp`), và kiểm tra ở quy mô gần thực tế (batch=64, 500 item). Xem chi tiết đầy đủ và
giới hạn của những gì ĐÃ/CHƯA kiểm chứng ở README của folder đó.

**Giai đoạn B — Patch tối thiểu, có công tắc bật/tắt (`κ=0` mặc định) — ✅ ĐÃ HOÀN THÀNH, xem
[`Phuong_An_5_GiaiDoanB_PatchThat/`](Phuong_An_5_GiaiDoanB_PatchThat/README.md):**
Đã viết `GaussianDiffusionModalOT` patch thẳng vào bản clone mới nhất của DiffMM (kế thừa trực tiếp
`GaussianDiffusion` gốc thay vì `GaussianDiffusionCFM` — lý do: `__init__` ở đây bỏ qua hoàn toàn khung
VP-style của lớp cha giống PA1/3/4, nên kế thừa từ `GaussianDiffusionCFM` không tái sử dụng được gì về
code, công thức CFM vẫn được tái dùng ở mức đại số). Mặc định `κ=0` trong `Params.py` (tương đương PA1)
để merge an toàn. **Đã kiểm chứng bằng hồi quy trực tiếp trên `Model.py` thật** (không phải bản tách rời
như Giai đoạn A): `κ=0` cho `tau, sigma, cfm_weight` trùng khít tuyệt đối Phương án 1/2/3; `κ>0` không
NaN/Inf ở mọi kịch bản biên đã thử (user rỗng, κ rất lớn, quy mô gần thực tế). Quá trình này phát hiện
và sửa 1 lỗi thật (gộp nhầm hệ số `tau` với `tau*x_start`, làm sai trọng số CFM tại toạ độ `x_start=0`
và làm sai `p_mean_variance` một cách âm thầm) — xem chi tiết ở README của folder đó. **Chưa** chạy với
dữ liệu thật/mạng `Denoise` thật/GPU — đó là phạm vi của Giai đoạn C.

**Giai đoạn C — Đóng gói theo `Folder_Base` (chỉ làm sau khi Giai đoạn A+B đã có kết quả sơ bộ hợp
lý):** lúc này mới áp dụng đầy đủ quy trình 8 bước như PA1-4.

---

## 6. Thiết kế patch (phác thảo kiến trúc — CHƯA phải code sẵn sàng dán, khác hẳn PA1-4)

```python
class GaussianDiffusionModalOT(GaussianDiffusionCFM):
    """
    [Phuong an 5 - nghien cuu, chua co tien le truc tiep]
    Ke thua tu GaussianDiffusionCFM (Phuong an 2) vi can khung CFM tong quat hoa duoc cho path
    bat ky (khong phai ELBO/SNR chi hop voi path VP). OVERRIDE TOAN BO q_sample, training_losses,
    p_mean_variance, p_sample de nhan them alpha0_true va modal_embeds lam tham so — day la thay
    doi GIAO DIEN, khong the "chi them, khong sua" nhu PA1-4.
    """

    def __init__(self, sigma_min, steps, kappa=0.0, g_min=0.5, g_max=2.0, w_clip=50.0, use_msi=False):
        # KHONG the goi super().__init__() theo kieu cu, vi mu_coef/sigma_coef khong con la
        # mang (T,) co dinh nua - can khoi tao lai tu dau, chi giu lai sigma_min/steps/w_clip.
        ...

    def _modal_affinity(self, alpha0_true, modal_embeds):
        # CT-5.1 + CT-5.2 — tra ve ma tran (batch, |I|)
        ...

    def _per_item_schedule(self, alpha0_true, modal_embeds, t):
        # CT-5.3 + CT-5.4 + CT-5.5, roi sai phan lui (giong PA1-3) de ra mu_prime/sigma_prime — tra
        # ve mu_t, sigma_t, mu_prime, sigma_prime deu shape (batch, |I|)
        ...

    def q_sample(self, alpha0_true, modal_embeds, t, noise=None):
        # can them alpha0_true, modal_embeds — KHAC CHU KY ham q_sample cua lop cha
        ...

    def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
        # x_start CHINH LA alpha0_true o day (da co san trong chu ky ham goc — thuan loi), nhung
        # PHAI nhan w_{u,i}(t) TRUOC KHI mean_flat, khac hoan toan thu tu cua PA1-4
        ...
```

*(Cố tình để dạng phác thảo, không viết đủ thân hàm — vì mục tiêu tài liệu này là xác nhận **thiết kế
đúng** và **rủi ro đã biết trước**, không phải đưa ra code "chắc chắn chạy đúng ngay" như PA1-4. Việc
viết đủ thân hàm nên làm ở Giai đoạn A/B của mục 5, có kiểm tra gradient bằng số học đi kèm.)*

---

## 7. Rủi ro & câu hỏi mở (trung thực về những gì CHƯA có câu trả lời)

- **Lựa chọn `cos_sim` cho `φ_{u,i}` (CT-5.2) là 1 lựa chọn, không phải lựa chọn DUY NHẤT đúng** — có
  thể thử tích vô hướng thô, hoặc học 1 hàm affinity nhỏ (thay vì công thức đóng) — đây là không gian
  thiết kế mở, cần thực nghiệm so sánh.
- **Tương tác với D5 (Cross-Modal Contrastive Augmentation):** D5 cũng dùng thông tin modal (qua đồ
  thị `A^m`) để tạo tín hiệu tương phản — cần kiểm tra Phương án 5 có "trùng lặp vai trò" với D5 không
  (cả 2 đều cố gắng đưa thông tin modal vào biểu diễn cuối cùng, chỉ khác cơ chế) — nếu trùng lặp quá
  nhiều, hiệu quả tăng thêm của Phương án 5 có thể nhỏ hơn kỳ vọng.
- **`κ` (độ mạnh hiệu ứng modal) cần quét (sweep) thực nghiệm** — chưa có cách suy ra giá trị tối ưu
  từ lý thuyết, chỉ có thể xác định qua thử nghiệm (khuyến nghị bắt đầu từ `κ` rất nhỏ, ví dụ `0.1-0.5`,
  tăng dần).
- **Chi phí runtime tăng thêm** (tính `φ`, `μ`, `σ`, `w` theo batch thay vì tra bảng cố định) cần đo
  thực tế — có thể làm chậm đáng kể vòng lặp D7 (Multi-Task Training) so với PA1-4, ngược lại hoàn
  toàn với mục tiêu tăng tốc của PA3/PA4.
- ~~Chưa kiểm chứng gradient bằng số học~~ — **✅ đã xong**, xem
  [`Phuong_An_5_GiaiDoanA_GradCheck/`](Phuong_An_5_GiaiDoanA_GradCheck/README.md).
- ~~Chưa patch vào codebase thật, chưa hồi quy với PA1~~ — **✅ đã xong**, xem
  [`Phuong_An_5_GiaiDoanB_PatchThat/`](Phuong_An_5_GiaiDoanB_PatchThat/README.md). Quá trình này phát
  hiện 1 lỗi thật (gộp nhầm hệ số `tau` với `tau*x_start`) mà Giai đoạn A không bắt được — minh chứng cụ
  thể rằng "gradient đúng về công thức" (Giai đoạn A) và "công thức khớp với Theorem 3/PA1 khi κ=0"
  (Giai đoạn B) là 2 lớp kiểm chứng khác nhau, cả hai đều cần thiết. Câu hỏi mở còn lại: liệu tín hiệu
  học có thực sự **hữu ích** (Recall/NDCG) khi chạy với dữ liệu thật trên GPU hay không — đó là phạm vi
  của Giai đoạn C.

---

## 8. Vị trí Phương án 5 trong toàn cảnh 5 phương án

| | PA1 | PA2 | PA3 | PA4 | **PA5** |
|---|---|---|---|---|---|
| Nền tảng | OT path mới | CFM loss mới | PA1+PA2 | Suy luận rút gọn | **Mở rộng PA1+PA2 theo modal** |
| `μₜ,σₜ` tính khi nào | 1 lần lúc init | (không đổi path) | 1 lần lúc init | (không đổi path) | **Mỗi batch, phụ thuộc dữ liệu** |
| Cần đổi chữ ký hàm | Không | Không | Không | Không | **Có — thay đổi kiến trúc** |
| Vai trò MSI (D3) | Giữ nguyên | Giữ nguyên | Giữ nguyên | Giữ nguyên | **Loại bỏ, thay bằng path** |
| Kiểm chứng gradient số học | Không cần (dùng lại công thức đã biết đúng) | Không cần | Không cần | Không cần | **Bắt buộc — ✅ đã làm xong (Giai đoạn A)** |
| Hồi quy trên patch thật (κ=0 ≡ PA1) | — | — | — | — | **✅ đã làm xong (Giai đoạn B)** |
| Có thể dùng `Folder_Base` ngay | Có | Có | Có | Có | **Chưa — cần Giai đoạn C (mục 5)** |
| Độ khó (kế hoạch gốc) | ★☆☆☆☆ | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ | ★★★★★ |

Phương án 5 là hướng đi **hợp lý về mặt lý thuyết** (đã chứng minh điều kiện biên đúng, đã kiểm chứng
gradient bằng số học ở Giai đoạn A, và đã kiểm chứng hồi quy trên patch thật ở Giai đoạn B — `κ=0` trùng
khít tuyệt đối Phương án 1) nhưng **chưa sẵn sàng để đóng gói thành folder chạy Colab ngay** như PA1-4 —
chưa chạy với dữ liệu thật/mạng thật/GPU, chưa có số liệu Recall/NDCG. Giai đoạn C (đóng gói
`Folder_Base`, chạy thật) là bước hợp lý tiếp theo nếu muốn theo đuổi Phương án 5 nghiêm túc.
