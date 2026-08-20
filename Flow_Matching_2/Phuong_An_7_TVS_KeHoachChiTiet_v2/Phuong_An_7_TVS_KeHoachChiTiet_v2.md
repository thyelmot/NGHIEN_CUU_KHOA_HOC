# Phương án 7 (v2) — Tam giác hoá vận tốc (TVS) — Kế hoạch chi tiết, đã sửa lỗi dịch + bổ sung phương án thay thế

> **Đây là bản v2, KHÔNG thay thế** [`../Phuong_An_7_TVS_KeHoachChiTiet.md`](../Phuong_An_7_TVS_KeHoachChiTiet.md)
> (bản v1 giữ nguyên, không sửa). Bản này sửa 1 lỗi dịch thuật quan trọng phát hiện được khi đối chiếu
> lại `Algorithm 1` thật của OFM (mục 1-2), và bổ sung 1 phương án thay thế rẻ hơn nhiều (mục 5 —
> **Residual Head**) nếu vẫn muốn giữ tinh thần của TVS mà không cần đổi sang velocity-prediction.

---

## 0. Tóm tắt khác biệt so với v1

| | v1 | v2 |
|---|---|---|
| Cách hiểu TVS | Huấn luyện qua **3 nhánh loss riêng** (`x_t`, `y_t`, `z_t`, mỗi nhánh 1 lần forward qua mạng) | **Sửa lại đúng Algorithm 1 của OFM**: mạng chỉ forward qua `x_t`, **1 loss duy nhất** — `y_t`/`z_t` chỉ là công cụ chứng minh trên giấy, không bao giờ đi qua mạng |
| Hệ quả | PA7 là 1 thiết kế huấn luyện phức tạp, nhiều rủi ro (gradient 3 nhánh) | PA7 dịch đúng thì **gần như không có gì để code thêm** — cấu trúc trùng với data-prediction đã có sẵn |
| Đề xuất khi K1-K3 thoả | Triển khai `GaussianDiffusionTVS` với `velocity_mode` | Ưu tiên thử **Residual Head** trước (mục 5) — rẻ hơn, an toàn hơn, capture đúng động lực gốc của TVS mà không cần đổi tham số hoá |

---

## 1. Sửa lỗi dịch thuật: TVS thật sự huấn luyện như thế nào?

### 1.1 Đọc lại đúng Algorithm 1 của OFM (không suy diễn)

Trích nguyên văn từ `Optical_Flow_Matching_Review.md` (Giai đoạn 3 — Huấn luyện):

```
2: x_l ← V_θ^OF(I1,I2)
3: p_l(x) ← N(x | x_l, I)
4: Sample x_0 ~ p_l(x), t~U(0,1)
5: x_1 ← x_i + f_gt
6: x_t ← t*x_1 + (1-t)*x_0
7: // Triangle Velocities Synergy objective adaptation
8: v̂_t ← V_θ^OF(x_t, t | I1, I2)
9: L(θ) ← ||v̂_t − f_gt||²_2
```

**Mạng chỉ nhận `x_t` (điểm trên quỹ đạo chính) làm đầu vào — không có bước nào đưa `y_t` hay `z_t` qua
mạng.** Chỉ có **1 giá trị `v̂_t` duy nhất**, được giám sát trực tiếp bằng `f_gt` — đúng 1 phép trừ bình
phương, không phải tổng của 3 loss.

### 1.2 Vậy `y_t`, `z_t` (công thức 8 OFM) dùng để làm gì?

Chúng là **công cụ chứng minh**, dùng **1 lần duy nhất, trên giấy**, để trả lời câu hỏi: "Tại sao giám
sát `v̂_t` (tính từ `x_t`) bằng `f_gt` lại là hợp lệ, trong khi đích lý thuyết thật của Flow Matching tại
`x_t` là `v_t(x_t|x_1) = x_1 − x_0` (không phải `f_gt`)?"

Câu trả lời (công thức 9 OFM): vì 3 vận tốc `v_t(x_t|x_1)`, `v_t(y_t|x_1)=f_gt`, `v_t(z_t|x_0)` tạo
thành 1 tam giác, nên `v_t(x_t|x_1) = f_gt − v_t(z_t|x_0)`. Tại **thời điểm suy luận** (Algorithm 2),
`v_t(z_t|x_0)` được **tính trực tiếp bằng công thức đóng** (`x_0 − x_i`, không cần mạng dự đoán) rồi
trừ đi để suy ra vận tốc quỹ đạo thật dùng cho bước Euler. Ở **thời điểm huấn luyện**, phép trừ này
không cần thực hiện tường minh — vì đã chứng minh xong tính hợp lệ, nên huấn luyện có thể giám sát thẳng
`v̂_t` bằng `f_gt`, y hệt như dòng 8-9 của Algorithm 1.

**Kết luận sửa lỗi:** v1 (mục 4, CT-7.3, và class `GaussianDiffusionTVS` ở mục 6) đã hiểu nhầm `y_t`,
`z_t` là 2 nhánh huấn luyện bổ sung, cần forward qua mạng và tính loss riêng. **Đây là hiểu sai** — OFM
không làm vậy. TVS, dịch đúng, có cấu trúc huấn luyện: `input=(x_t, t)`, `output=v̂_t`, `loss=MSE(v̂_t,
f_gt)` — **1 forward pass, 1 loss**, giống hệt cấu trúc data-prediction `input=(αₜ,t)`, `output=α̂₀`,
`loss=w(t)·MSE(α̂₀,α₀)` mà DiffMM đã dùng từ PA1 đến PA6.

---

## 2. Hệ quả: PA7 dịch đúng thì còn "trống" hơn v1 đã kết luận

v1 (mục 2) đã lập luận đúng rằng DiffMM sidestep được vấn đề TVS giải quyết nhờ data-prediction. Sau khi
sửa mục 1, có thể phát biểu mạnh hơn:

> **DiffMM (PA1-6) không chỉ "không cần" TVS — nó ĐÃ CÓ cấu trúc huấn luyện tương đương với cái mà TVS
> cố đạt được cho OFM** (1 forward pass, 1 loss, target bất biến-theo-`t`, không phụ thuộc cách thêm
> nhiễu). OFM cần "phát minh" ra TVS vì khung Flow Matching gốc ép buộc tham số hoá velocity-prediction;
> DiffMM chưa từng bị ép buộc như vậy vì đã chọn data-prediction từ đầu (kế thừa từ paper DiffMM gốc,
> không phải một lựa chọn của các Phương án 1-6).

Hệ quả thực tế: **nếu dịch đúng TVS và áp cho DiffMM, sẽ không có dòng code nào mới cần viết** — không
có class `GaussianDiffusionTVS`, không có `velocity_mode`, không có CT-7.4 (nghịch đảo ODE). Điều kiện
kích hoạt K1-K3 ở mục 3 của bản v1 vẫn đúng và vẫn nên giữ, nhưng nay có thêm lý do: **ngay cả khi K1-K3
thoả mãn, hướng đi đúng đắn theo tinh thần TVS-thật cũng không tạo ra 1 Phương án 7 khác biệt** — chỉ có
2 lựa chọn thực chất còn lại nếu vẫn muốn "học" gì đó từ TVS: (a) đổi hẳn sang velocity-prediction vì 1
lý do khác (không liên quan ổn định huấn luyện — ví dụ muốn sinh mẫu 1 bước kiểu MeanFlow, xem mục 6),
hoặc (b) áp dụng ý tưởng **residual learning** (mục 5) mà không cần đổi tham số hoá.

---

## 3. Điều kiện kích hoạt (giữ nguyên từ v1, vẫn đúng)

| # | Điều kiện | Cách kiểm tra |
|---|---|---|
| **K1** | Phương án 6 đã chạy thật, `α_l` cho tín hiệu cải thiện rõ ràng | Kết quả thực nghiệm PA6 |
| **K2** | Có lý do **khác** (không phải ổn định huấn luyện) để chuyển sang velocity-prediction — ví dụ muốn sinh mẫu 1 bước (xem mục 6) | Phân tích thiết kế trước khi code |
| **K3** | Data-prediction (PA6) cho thấy mất ổn định — **về mặt lý thuyết khó xảy ra hơn cả v1 dự đoán**, vì đã xác nhận DiffMM cấu trúc tương đương "hậu-TVS" ngay từ đầu | Thực nghiệm PA6 |

Không đổi so với v1: nếu chưa có cả 3, PA7 (bản đổi tham số hoá) ở trạng thái "ghi nhận, chưa triển
khai".

---

## 4. Công thức TVS đã sửa (chỉ 1 nhánh, không phải 3)

Nếu K1-K3 thoả và vẫn muốn đổi sang velocity-prediction (lý do K2, không phải vì ổn định), thiết kế
**đúng** (đã sửa) như sau — thay thế hoàn toàn CT-7.1/7.3/7.5 và class `GaussianDiffusionTVS` của bản v1:

### CT-7.1(v2) — Chỉ 1 quỹ đạo chính, không có nhánh phụ trong vòng lặp huấn luyện

```
x̃_t = t·α_ref + (1-t)·α₀ + σ_min·ε,   ε ~ N(0, I)    (quỹ đạo chính, y hệt trước)
```

### CT-7.2(v2) — Loss huấn luyện (thay thế CT-7.3 của v1)

```
L_TVS = Σ_t  w(t) · ‖v_θ(x̃_t, t) − f_gt‖²
```

**Đúng 1 số hạng — không có `λ_x, λ_y, λ_z`, không có forward pass trên `y_t`/`z_t`.** `w(t)` có thể tái
sử dụng nguyên vẹn công thức trọng số CFM đã kiểm chứng ở Phương án 2/3/6 (cùng lý do CT-6.7: trọng số
này chỉ phụ thuộc `μₜ, σₜ`, không phụ thuộc việc mạng dự đoán `α₀` hay vận tốc).

### CT-7.3(v2) — Tái tạo `α̂₀` (giữ nguyên CT-7.4 của v1, đây là phần v1 làm đúng)

```
α̂₀ = (x̃_t − t·v_θ(x̃_t,t)) / (1 − t + ε)
```

Vẫn cần kiểm chứng số học điều kiện biên (`t→0`, `t→1`) trước khi dùng — xem Test 7.2/7.4 ở mục 7, giữ
nguyên yêu cầu từ v1.

### CT-7.4(v2) — `y_t`/`z_t` chỉ dùng khi SUY LUẬN (Algorithm 2), không dùng khi HUẤN LUYỆN

```
v̄_t = x_0 − x_i    (= α_l·w − α₀, dùng CT-6.2 của Phương án 6 — tính trực tiếp, KHÔNG qua mạng)
v_t  = v_θ(x̃_t, t) − v̄_t    (áp dụng CHỈ ở bước Euler khi suy luận)
```

Đây là điểm khác biệt quan trọng nhất so với v1: **v1 đưa phép trừ tam giác vào loss huấn luyện (sai),
v2 chỉ đưa vào bước suy luận (đúng theo Algorithm 2 của OFM)** — điều này loại bỏ hoàn toàn rủi ro
"gradient nổ từ 3 nhánh" mà v1 lo ngại ở mục 8, vì huấn luyện giờ chỉ còn 1 nhánh.

---

## 5. Phương án thay thế: Residual Head — capture động lực TVS mà KHÔNG đổi tham số hoá

### 5.1 Ý tưởng

Động lực gốc của TVS là: *đừng bắt mạng dự đoán 1 đại lượng lớn khi đã có sẵn 1 ước lượng thô gần đó —
chỉ cần dự đoán phần dư (residual)*. Có thể đạt đúng lợi ích này **mà không rời khỏi data-prediction**,
bằng cách thêm 1 phép cộng đơn giản vào đầu ra của `Denoise`:

### CT-7.5(v2) — Residual head

```
α̂₀ = α_l + Denoise_residual(αₜ, t)
L  = w(t) · ‖α̂₀ − α₀‖²     (Y HỆT loss hiện tại của PA1-6, không đổi công thức)
```

`Denoise_residual` là **cùng 1 mạng MLP hiện có**, chỉ khác ở việc kết quả cuối cùng được cộng thêm
`α_l` (phép "skip-connection" tới điểm neo, giống ý tưởng residual learning tiêu chuẩn trong deep
learning — không phải phát minh riêng của OFM, nhưng đúng tinh thần "học phần dư thay vì học toàn bộ"
mà TVS theo đuổi).

### 5.2 Vì sao đây là lựa chọn tốt hơn PA7(v1)/PA7(v2 — bản đổi velocity) nếu muốn thử điều gì đó ngay

| Tiêu chí | TVS (đổi velocity-prediction) | Residual Head |
|---|---|---|
| Đổi `q_sample`/`p_mean_variance`/`cfm_weight` | Không (nếu dùng CT-7.1-7.4 đúng) | **Không đổi gì cả** |
| Cần chứng minh điều kiện biên mới | Có (CT-7.3(v2), chia `1-t`) | **Không** — D1/D4 nguyên vẹn từ PA1/PA3/PA5/PA6 |
| Rủi ro số học (chia gần 0, NaN) | Có (`t→1`) | Không có phép chia nào mới |
| Tương thích ngược (công tắc tắt) | Cần cờ `velocity_mode=False` | Cần cờ `residual_head=False` — **đơn giản hơn**: chỉ là 1 điều kiện cộng/không cộng `α_l` vào output |
| Điều kiện kích hoạt | K1+K2+K3 (hiếm khi cả 3 cùng đúng) | **Chỉ cần đã có PA6** (đã xong) — có thể thử ngay khi chạy thật, không cần chờ K1-K3 |
| Độ khó ước tính | ★★★★★ | ★★☆☆☆ |

### 5.3 Thiết kế class (phác thảo)

```python
class GaussianDiffusionAnchorOT(GaussianDiffusionOTCFM):
    # ... (giu nguyen moi thu cua Phuong an 6) ...

    def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats, uEmbeds_batch,
                         residual_head=False):
        ...
        alpha_l = self._compute_anchor(uEmbeds_batch, itmEmbeds) if self.anchor_w != 0 else torch.zeros_like(x_start)
        ...
        model_output = model(x_t, ts)
        if residual_head:
            model_output = alpha_l + model_output   # CT-7.5(v2) - chi 1 dong moi
        mse = self.mean_flat((x_start - model_output) ** 2)
        ...
```

**Lưu ý quan trọng:** `residual_head` có thể bật độc lập với `anchor_w` — thậm chí hợp lý hơn khi dùng
**2 giá trị `α_l` khác nhau cho 2 vai trò khác nhau** (1 làm tâm nhiễu forward — CT-6.2, 1 làm điểm cộng
ở output — CT-7.5) nếu muốn tách bạch 2 cơ chế khi ablation. Ở phiên bản đơn giản nhất, dùng chung 1
`α_l` cho cả 2 vai trò (như code trên) là hợp lý để bắt đầu.

**Không cần kiểm chứng điều kiện biên toán học mới** — đây là 1 phép cộng skip-connection ở output, một
kỹ thuật kiến trúc tiêu chuẩn, không phải 1 công thức xác suất mới. Chỉ cần kiểm chứng thực nghiệm (so
sánh `residual_head=True` vs `False` trên cùng dữ liệu thật) khi chạy GPU thật — không cần Giai đoạn A
kiểu derive+numeric-check như các phương án khác.

---

## 6. Khi nào K2 (mục 3) thực sự có cơ sở — ghi chú cho tương lai

v1 để K2 khá mơ hồ ("có lý do chính đáng"). Sau khi đọc lại `Optical_Flow_Matching_Review.md` mục 6.3
(Hướng phát triển), có 1 lý do K2 cụ thể, xác đáng: chính OFM cũng tự nhận TVS + Euler ODE solver "là
giải pháp cổ điển, không phải tối ưu", và chỉ ra hướng đi thật sự hiệu quả là **MeanFlow**/**Shortcut
Models** — các phương pháp sinh mẫu **1 bước** (one-step), không cần vòng lặp Euler nhiều bước. Nếu
tương lai DiffMM muốn giảm số bước suy luận D4 xuống còn 1 (thay vì K=3 của PA3, hay quét K của PA8),
đó mới là lúc velocity-prediction (và các kỹ thuật liên quan) có lý do độc lập để cân nhắc — không phải
vì TVS, mà vì bản thân MeanFlow/Shortcut Models đòi hỏi tham số hoá đó. Đây nên là 1 "Phương án 10"
riêng trong tương lai (chưa viết kế hoạch), không nên gộp vào lý do kích hoạt PA7.

---

## 7. Giai đoạn A — Danh sách test cập nhật

Nếu theo đuổi TVS(v2) — bản đổi velocity-prediction đã sửa (mục 4):

| Bài test | Mục đích | Điều kiện đạt |
|---|---|---|
| Test 7.1 — Quan hệ tam giác tự nhất quán (suy luận, CT-7.4(v2)) | `f_gt = v̄_t + (v_θ − v̄_t)` tự nhất quán | Sai lệch `< 10⁻¹⁰` |
| Test 7.2 — CT-7.3(v2): Tái tạo `α₀` từ velocity | Denoiser hoàn hảo → `α̂₀ ≈ α₀` | Sai lệch `< 10⁻⁸` ở mọi `t∈(0,1)` |
| Test 7.3 — `velocity_mode=False` trùng khít PA6 | Hồi quy | Sai lệch `= 0` |
| Test 7.4 — Biên `t→0`, `t→1` ổn định số học (CT-7.3(v2)) | Không NaN/Inf | Epsilon `1e-8` đủ |
| ~~Test 7.5 — Gradient qua 3 nhánh~~ | **Bỏ** — không còn 3 nhánh trong huấn luyện (mục 1-4) | — |

Nếu theo đuổi Residual Head (mục 5) — không cần Giai đoạn A kiểu derive, chỉ cần:

| Bài test | Mục đích |
|---|---|
| Test R.1 — `residual_head=False` trùng khít PA6 tuyệt đối | Hồi quy, tương tự cách đã làm cho `anchor_w=0` |
| Test R.2 — `residual_head=True` không NaN/Inf ở batch/scale thực tế | Sanity check, không cần chứng minh gì thêm |

---

## 8. Rủi ro (cập nhật)

| Rủi ro | Mức độ | Ghi chú so với v1 |
|---|---|---|
| CT-7.3(v2) không ổn định số học (chia `1-t`) | Cao | Giữ nguyên từ v1, vẫn cần epsilon |
| ~~Gradient "nổ" từ 3 loss nhánh~~ | ~~Cao~~ | **Không còn tồn tại** — huấn luyện chỉ còn 1 nhánh (mục 1-4) |
| Output head `Denoise` không học tốt velocity | Rất cao | Giữ nguyên từ v1 |
| Không cải thiện so với PA6 (bản đổi velocity) | Rất cao | Giữ nguyên từ v1 |
| Residual Head không cải thiện so với PA6 thường | Trung bình | **Rủi ro thấp hơn hẳn** — dù không cải thiện cũng không có gì để mất (không đổi q_sample/p_mean_variance, dễ tắt) |

---

## 9. Lộ trình đề xuất (cập nhật thứ tự ưu tiên)

```
Bước 1: Phương án 8 (quét num_sample_steps) — chưa đổi, vẫn rẻ nhất
Bước 2: Chạy Phương án 6 thật trên GPU (Giai đoạn D chưa có trong PA6) — xác nhận K1
Bước 3 (mới, thay cho "chờ K1-K3 rồi làm TVS đầy đủ"):
        Thử Residual Head (mục 5) — độc lập với K1-K3, độ khó ★★☆☆☆, có thể làm ngay sau Bước 2
        nếu muốn thử thêm 1 biến thể rẻ của PA6
Bước 4: CHỈ nếu K1+K3 đều xác nhận VÀ có lý do K2 kiểu mục 6 (hướng one-step sampling) —
        mới cân nhắc TVS(v2) đầy đủ (mục 4), bắt đầu bằng Test 7.1-7.4 ở mục 7
```

---

## 10. Kết quả kiểm chứng số học

| Bài kiểm tra | Kết quả | Ghi chú |
|---|---|---|
| Test 7.1-7.4 (TVS(v2), bản đổi velocity) | *Chưa chạy* | Vẫn ở trạng thái "ghi nhận, chưa triển khai" — đúng như mục 3/11 |
| Test R.1-R.2 (Residual Head) | **✅ Đã chạy, PASS** (6 phép kiểm tra trên `Model.py` thật, mạnh hơn dự kiến ban đầu) | Xem [`phuong_an_7_residual_head/README.md`](phuong_an_7_residual_head/README.md) — hồi quy `residual_head=False` xác nhận trùng khít Phương án 6 ở **cả `anchor_w=0` lẫn `anchor_w>0`** (không chỉ 1 trường hợp), `residual_head=True` với denoiser dự đoán đúng phần dư tái tạo chính xác `α₀` (sai lệch `<10⁻¹⁰`), không NaN/Inf ở mọi trường hợp biên đã thử. Đã đóng gói đầy đủ theo `Folder_Base` (fork `DiffMM-ResidualOT`, notebook Colab, đã push lên [thyelmot/DiffMM_7_v2](https://github.com/thyelmot/DiffMM_7_v2)) |

---

## 11. Tóm tắt quyết định nhanh (cập nhật)

```
Muốn thử điều gì đó rẻ, an toàn, capture tinh thần TVS ngay bây giờ?
  → Làm Residual Head (mục 5) — không cần chờ K1-K3, độ khó ★★☆☆☆

Muốn TVS "đúng nghĩa" (đổi hẳn velocity-prediction)?
  → Vẫn phải chờ đủ K1+K2+K3 như v1 đã nêu — VÀ giờ biết thêm: dịch đúng thì TVS
    không mang lại kiến trúc mới so với DiffMM hiện tại (mục 1-2), trừ khi K2 đến từ
    nhu cầu one-step sampling thật sự (mục 6), không phải từ nhu cầu ổn định huấn luyện.
```

**Mức độ ưu tiên:** Residual Head 🟢 **✅ đã đóng gói xong (Giai đoạn A→C), sẵn sàng chạy trên Colab** —
xem [`phuong_an_7_residual_head/`](phuong_an_7_residual_head/README.md). TVS(v2) đầy đủ vẫn 🔴 thấp
nhất, giữ nguyên ở trạng thái "ghi nhận, không triển khai".
