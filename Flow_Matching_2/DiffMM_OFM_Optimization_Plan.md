# Kế hoạch tối ưu DiffMM bằng Optical Flow Matching (OFM)

### Mục tiêu: dùng các phát hiện của bài OFM để cải tiến khối Diffusion Model bên trong DiffMM

> Tài liệu này giả định bạn đã đọc [DiffMM_Review.md](../DiffMM/DiffMM_Review.md) và
> [Optical_Flow_Matching_Review.md](Optical_Flow_Matching_Review.md). Các ký hiệu công thức (eq X)
> tham chiếu lại đúng số thứ tự đã dùng trong 2 file đó. Tài liệu này **bổ sung** cho
> [Flow_Matching_1/DiffMM_FlowMatching_Optimization_Plan.md](../Flow_Matching_1/DiffMM_FlowMatching_Optimization_Plan.md)
> (đã đề xuất 5 phương án dựa trên bài Flow Matching gốc, PA1-4 đã triển khai thật, PA5 đang ở Giai đoạn
> C) — các phương án ở đây được đánh số **tiếp theo (6, 7...)**, không trùng với PA1-5.

---

## 1. Vì sao 2 bài báo này "ghép" được với nhau — và vì sao KHÔNG phải mọi thứ đều ghép được

OFM về bản chất là **Flow Matching áp dụng cho toạ độ điểm ảnh** — cùng 1 lý thuyết nền (Flow Matching)
mà [`Flow_Matching_1_Review.md`](../Flow_Matching_1/Flow_Matching_1_Review.md) đã trình bày, nên điều
kiện tiên quyết ("dữ liệu phải biểu diễn được dưới dạng liên tục trong ℝᵈ") **đã được xác nhận thoả
mãn** từ kế hoạch PA1-5 trước đó (α₀ là vector liên tục hoá được, xem mục 1 của
`DiffMM_FlowMatching_Optimization_Plan.md`) — không cần chứng minh lại.

Điểm mới mà OFM mang lại **không phải** ở lý thuyết Flow Matching cốt lõi (đã khai thác hết ở PA1-5), mà
ở **2 kỹ thuật thực dụng cụ thể** mà OFM tự thiết kế thêm để áp dụng Flow Matching vào 1 bài toán "đã có
sẵn điểm khởi đầu vật lý" (toạ độ gốc `x_i`) thay vì sinh từ hư vô — **đúng hoàn cảnh của DiffMM** (đã
có sẵn α₀ thật, không sinh α₀ từ hư vô):

1. **Điểm neo thô học được (`x_l`, Giai đoạn O1)** — thay vì khởi tạo nhiễu tại 1 tâm cố định, OFM dự
   đoán trước 1 vị trí gần đích hơn để giảm "quãng đường" quỹ đạo cần đi. **Đây là ý tưởng CHƯA từng
   được khai thác trong PA1-5** — cả 5 phương án cũ đều dùng schedule/tâm nhiễu **cố định, không phụ
   thuộc dữ liệu** (kể cả PA5 chỉ đổi *tốc độ* quỹ đạo theo modal, không đổi *tâm xuất phát*).

2. **Triangle Velocities Synergy — TVS (Giai đoạn O2)** — một mẹo hình học giúp mạng học 1 đại lượng có
   ý nghĩa vật lý (chính là optical flow thật) thay vì phải học trực tiếp 1 đại lượng trừu tượng
   ("flow trừ nhiễu"). **Đây là điểm cần đọc kỹ trước khi vội áp dụng:** mục 4 bên dưới sẽ chỉ ra rằng
   DiffMM (và toàn bộ PA1-5) **đã tránh được đúng vấn đề mà TVS được sinh ra để giải quyết**, nhờ một
   lựa chọn thiết kế đã có sẵn từ đầu (tham số hoá kiểu **data-prediction** — mạng luôn dự đoán α₀ trực
   tiếp, không dự đoán vận tốc/độ dịch chuyển). Vì vậy TVS **không mang lại lợi ích ngay lập tức** cho
   cấu hình hiện tại — nó chỉ trở nên cần thiết **nếu** Phương án 6 (điểm neo học được) được triển khai
   theo đúng kiểu tham số hoá của OFM. Đây là ví dụ cụ thể cho nguyên tắc "không cố ghép cho bằng được"
   đã nêu trong prompt mẫu.

---

## 2. Liệt kê tách rời các bước của DiffMM (giữ nguyên từ kế hoạch PA1-5, để tiện đối chiếu)

| # | Giai đoạn (theo DiffMM_Review.md) | Input → Output | Cơ chế cốt lõi | Công thức |
|---|---|---|---|---|
| D0 | Chuẩn bị dữ liệu | Đồ thị G, đặc trưng modal F̂ᵢᵐ | Biểu diễn tương tác user u thành vector nhị phân α₀ = aᵤ | — |
| D1 | Forward Diffusion Process | α₀ → αₜ | Thêm nhiễu Gauss dần theo T bước, **tâm nhiễu cố định tại α₀**, lịch trình nhiễu tuyến tính | eq (1)-(3) |
| D2 | Reverse Diffusion Process (Denoising Model Training) | αₜ, t → α̂₀ | MLP dự đoán lại **α₀ trực tiếp** (data-prediction) từ bản nhiễu; loss = ELBO rút gọn thành MSE có trọng số | eq (4)-(13) |
| D3 | Modality-aware Signal Injection (MSI) | α̂₀, eᵢᵐ, α₀, eᵢ → L_msi | Loss phụ ép embedding theo modal-view khớp với embedding theo collaborative-view thật | eq (14), L_dm = L_elbo + λ₀L_msi |
| D4 | Inference của Diffusion Model | α₀ → α_T' (forward T' bước) → α̂₀ (reverse T bước, deterministic) → A^m | Corrupt một phần rồi denoise ngược lại **T bước lặp tuần tự**, sau đó top-k chọn cạnh mới | — |
| D5 | Cross-Modal Contrastive Augmentation | A^m → z^m, ẑ^m → L_cl | GNN aggregation trên A^m + đồ thị gốc A, 2 kiểu InfoNCE | eq (15)-(19) |
| D6 | Multi-Modal Graph Aggregation (nhánh chính) | F̂^m, A, A^m → H | GNN gộp đa modal (trọng số κ_m học được) + lan truyền L lớp trên A | eq (20)-(23) |
| D7 | Multi-Task Training | — | L_dm^m = L_elbo + λ₀L_msi^m (diffusion) ; L_rec = L_bpr + λ₁L_cl + λ₂‖Θ‖² (recommendation) | eq (24)-(26) |

---

## 3. Liệt kê tách rời các bước của Optical Flow Matching (OFM)

| # | Giai đoạn (theo Optical_Flow_Matching_Review.md) | Input → Output | Cơ chế cốt lõi | Công thức |
|---|---|---|---|---|
| O0 | Trích đặc trưng | Cặp ảnh (I₁,I₂) → f_c (đặc trưng ngữ cảnh), f_cv (cost volume) | Feature/Context Encoder kiểu RAFT + Correlation Layer | Hình 2 |
| O1 | Điểm neo thô + khởi tạo quỹ đạo | f_cv → x_l (điểm neo, **học được, phụ thuộc dữ liệu**) → x_0 ~ N(x_l, I) | Khớp toàn cục (global matching, kiểu GMFlow-softmax) dự đoán flow thô, dùng làm tâm phân phối khởi tạo | mục 3.2.1 |
| O2 | Triangle Velocities Synergy (TVS) | x_0, x_1, x_i → 3 quỹ đạo phụ (x_t, y_t, z_t) | Thêm 2 quỹ đạo phụ đều xuất phát từ x_i (điểm gốc thật); 3 vận tốc hằng số tạo thành 1 tam giác, cho phép quy target khó (`v_t(x_t\|x_1)`) về hiệu của 2 target dễ (1 trong đó chính là flow thật) | eq (7)-(9) |
| O3 | Huấn luyện (Algorithm 1) | (I₁,I₂), f_gt → V_θ^OF | Nhờ TVS, mạng chỉ cần được giám sát bằng `L(θ)=‖v̂_t − f_gt‖²` — **đúng loss chuẩn của optical flow**, không cần học đại lượng trừu tượng | Alg. 1 |
| O4 | Suy luận (Algorithm 2) | x_i → x_pred | Tích phân ODE bằng Euler qua K bước (NFE=K), mỗi bước áp lại quan hệ tam giác để suy vận tốc thật từ vận tốc dự đoán | Alg. 2 |
| O5 | Kiến trúc cụ thể hoá | f_cv, f_c, x_t, t → v_t | Decoder hồi quy kiểu RAFT (TwinsSVT encoder), nhận thêm (x_t, t) làm đầu vào trạng thái, giữ cơ chế tinh chỉnh lặp | mục 3.3 |

---

## 4. Bảng đối chiếu tương đồng (mapping) DiffMM ↔ OFM

| Thành phần DiffMM | Vai trò tương đương trong OFM | Khoảng cách/khác biệt cốt lõi |
|---|---|---|
| D1 (Forward Diffusion, tâm nhiễu **cố định** tại α₀) | O1 (điểm neo `x_l` **học được**, phụ thuộc dữ liệu) | **Khoảng trống thực sự** — DiffMM (kể cả PA5) chưa từng thử "dự đoán trước 1 điểm gần đích rồi khởi tạo nhiễu quanh đó". Đây là nội dung chính của Phương án 6 bên dưới. |
| D2 (Reverse Diffusion, **data-prediction** — dự đoán α₀ trực tiếp) | O2+O3 (TVS — cơ chế cho phép mạng học optical flow thật thay vì "flow trừ nhiễu") | **KHÔNG có khoảng trống cần lấp** — DiffMM đã dùng data-prediction từ đầu (mạng luôn dự đoán α₀ trực tiếp, mọi PA1-5 đều giữ nguyên lựa chọn này), nên **đã tránh được** đúng vấn đề "target trừu tượng, khó hội tụ" mà TVS được sinh ra để giải quyết ở O2/O3. Xem phân tích chi tiết ở mục 5, Phương án 7. |
| D4 (Inference, T bước reverse **tuần tự cố định**) | O4 (Euler ODE, K bước = NFE, **linh hoạt**) | Đã được khai thác ở PA3 (rút gọn lịch trình) và PA4 (viết lại thành ODE solver tường minh). OFM bổ sung 1 insight thực nghiệm cụ thể (K=1 đã cạnh tranh tốt, K=3 là điểm cân bằng) — xem Phương án 8 (tinh chỉnh nhỏ, không phải hướng mới). |
| *(không có D tương đương)* | O5 (decoder kiểu RAFT, cơ chế tinh chỉnh lặp nhiều vòng bên trong 1 bước) | DiffMM dùng 1 MLP đơn giản cho denoiser — không có khái niệm "lặp tinh chỉnh nhiều vòng trong 1 bước thời gian" hay "cost volume". Đây là ý tưởng kiến trúc thuần tuý, không bắt nguồn từ lý thuyết Flow Matching — xem Phương án 9 (suy đoán, ưu tiên thấp). |
| D3 (MSI, loss phụ) | *(không có tương đương trực tiếp)* | OFM không có khái niệm "modal" — bảng này để trống có chủ đích, tránh gượng ép liên hệ không có cơ sở. |

---

## 5. Các phương án tối ưu khả thi (đánh số tiếp theo PA1-5, xếp theo độ ưu tiên)

### 🟢 Phương án 6 — Điểm neo thô học được (Learnable Coarse Anchor) cho tâm khởi tạo nhiễu — ✅ Giai
đoạn A + B (kiểm chứng công thức + patch thật, hồi quy trùng khít PA3) đã hoàn thành, xem
[`Phuong_An_6_Learnable_Anchor_KeHoachChiTiet.md`](Phuong_An_6_Learnable_Anchor_KeHoachChiTiet.md)

- **Vị trí áp dụng:** D1 (Forward Diffusion) — cách xây dựng αₜ từ α₀.
- **Thay đổi:** thêm 1 nhánh dự đoán nhẹ (ví dụ: tích vô hướng — dot product — giữa `uEmbeds` (user
  embedding đã có sẵn trong DiffMM, từ nhánh GNN chính D6) và `iEmbeds`, qua 1 phép chuẩn hoá
  softmax/top-k nhẹ) để suy ra **α_l — 1 ước lượng thô, phụ thuộc từng user**, về "user này có khả năng
  thích những item nào" — đúng vai trò của bước khớp toàn cục (global matching) O1 trong OFM. Dùng α_l
  này làm **tâm của nhiễu** trong forward process, thay vì tâm cố định như hiện tại: ví dụ
  `αₜ = μₜ·α₀ + σₜ·(α_l + ε)` hoặc một biến thể tương đương, **cần tự suy lại và kiểm chứng số học** để
  đảm bảo vẫn thoả 2 điều kiện biên bắt buộc: `t=0 → α₀` (sạch hoàn toàn) và `t=T-1 → phân phối nhiễu
  hợp lý` (giống hệt yêu cầu đã áp dụng khi thiết kế PA1/PA5).
- **Cơ sở lý thuyết:** O1 (OFM) — quan sát thực nghiệm của OFM rằng sai số ước lượng tăng theo độ lớn
  "quãng đường" cần đi; đặt điểm khởi đầu gần đích hơn giúp giảm sai số, đặc biệt với các trường hợp
  "lệch xa trung bình" (ở OFM là chuyển động lớn; ở DiffMM có thể tương ứng với user có sở thích khác
  biệt mạnh so với phần đông).
- **Lợi ích kỳ vọng:** cải thiện chất lượng α̂₀ cho nhóm user có tương tác lệch khỏi xu hướng chung; có
  thể cho phép giảm số bước cần thiết ở D4 (quỹ đạo "ngắn" hơn từ đầu).
- **Rủi ro:** đã **hạ xuống trung bình** sau khi derive + kiểm chứng số học đầy đủ (xem file kế hoạch
  chi tiết): (1) công thức D1/D2 đã suy xong, chứng minh điều kiện biên + tính bất biến của trọng số CFM
  (Phương án 2/3) với điểm neo — không cần suy lại từ đầu; (2) `α_l` dùng thiết kế **không thêm tham số
  học mới** (chỉ là hàm đóng của embedding đã có sẵn), giảm rủi ro so với ước tính ban đầu; (3) D3 (MSI)
  **không cần đổi gì** vì chỉ dùng `α̂₀`/`α₀` cuối cùng, không phụ thuộc cơ chế nhiễu nội bộ. Rủi ro còn
  lại chủ yếu là **thực nghiệm** (α_l dự đoán thô có thể không đủ tốt để cải thiện — chỉ kiểm chứng được
  khi chạy thật, xem rủi ro OFM tự nêu ở `Optical_Flow_Matching_Review.md` phần 6.2), không còn là rủi
  ro *toán học/kỹ thuật*.
- **Độ khó triển khai:** ★★☆☆☆ (hạ tiếp từ ★★★☆☆ sau khi hoàn thành Giai đoạn B — patch thật không phát
  sinh lỗi nào, hồi quy trùng khít tuyệt đối với PA3)

### 🔴 Phương án 7 — Tam giác hoá vận tốc (TVS), CHỈ dùng làm công cụ dự phòng cho Phương án 6

- **Vị trí áp dụng:** D2 (Reverse Diffusion / Denoising Model Training) — **chỉ khi và chỉ khi** Phương
  án 6 được triển khai theo đúng kiểu tham số hoá "dự đoán đại lượng phụ thuộc-anchor" thay vì giữ
  nguyên data-prediction thuần.
- **Phân tích vì sao KHÔNG cần thiết ở hiện trạng:** như đã nêu ở mục 4, DiffMM's D2 luôn cho mạng dự
  đoán **α₀ trực tiếp** (`α̂₀ = Denoise(αₜ, t)`), bất kể tâm nhiễu của forward process là gì — nghĩa là
  **loss `MSE(α̂₀, α₀)` luôn có ý nghĩa vật lý rõ ràng**, không phụ thuộc vào việc tâm nhiễu đặt ở đâu.
  Đây khác hẳn OFM: OFM (do bám sát khung Flow Matching gốc, vốn được thiết kế để học **vận tốc** —
  velocity — chứ không phải trực tiếp học dữ liệu sạch) buộc phải đối mặt với vấn đề "target trở thành
  trừu tượng khi đổi tâm nhiễu" — chính vì vậy **cần** TVS. **Nếu Phương án 6 chỉ đổi tâm nhiễu nhưng
  vẫn giữ nguyên cách huấn luyện data-prediction hiện tại của DiffMM (khuyến nghị), thì KHÔNG cần
  Phương án 7 — Phương án 6 tự nó vẫn ổn định.**
- **Khi nào mới cần:** chỉ nếu có lý do khác (ví dụ nghiên cứu mở rộng dùng velocity-prediction để tận
  dụng trực tiếp công thức CFM tổng quát của PA2/PA3 theo đúng khung Flow Matching gốc, thay vì phiên
  bản "đã rút gọn về MSE trên α₀" mà PA2/PA3 hiện dùng) — lúc đó áp dụng đúng quan hệ tam giác (eq 9
  OFM): định nghĩa 2 quỹ đạo phụ xuất phát từ 1 điểm neo cố định (ví dụ chính `x_i`/α_gốc của DiffMM),
  suy ra target khả thi bằng hiệu 2 vận tốc phụ.
- **Cơ sở lý thuyết:** O2 (OFM) — công thức (9), tái sử dụng nguyên vẹn nếu cần.
- **Rủi ro:** **cao nhất trong toàn kế hoạch** — chưa có tiền lệ trực tiếp, đòi hỏi tự suy công thức,
  chứng minh điều kiện biên, và **hiện tại chưa có động lực rõ ràng để triển khai** (vì Phương án 6 đã
  đủ mà không cần nó). Ghi nhận ở đây chủ yếu để **không bỏ sót** ý tưởng cốt lõi của OFM, không phải
  vì có bằng chứng nó sẽ cải thiện DiffMM.
- **Độ khó triển khai:** ★★★★★ (và **chỉ nên cân nhắc sau khi** Phương án 6 đã chạy thật và cho thấy
  cần đổi hướng tham số hoá)

### 🟢 Phương án 8 — Tinh chỉnh phương pháp quét NFE ở D4 theo insight thực nghiệm của OFM (bổ sung cho PA3/PA4, không phải hướng mới)

- **Vị trí áp dụng:** D4 — đã được PA3 (lịch trình rút gọn cố định `0.6×T`) và PA4 (ODE solver tường
  minh) khai thác.
- **Thay đổi:** OFM cho thấy 1 phát hiện thực nghiệm cụ thể đáng tham khảo (Bảng 3, mục 5.4
  `Optical_Flow_Matching_Review.md`): chỉ với **K=1 bước** đã vượt phần lớn baseline, và lợi ích tăng
  thêm từ K=1→K=4 có xu hướng **giảm dần** (diminishing returns) — gợi ý nên **quét có hệ thống**
  `num_sample_steps ∈ {1,2,3,4,...}` khi đánh giá PA3/PA4 (thay vì chỉ cố định 1 giá trị suy từ tỉ lệ
  `0.6×T` như hiện tại), để tìm điểm cân bằng tốc độ/chất lượng thực sự tối ưu cho DiffMM thay vì mượn
  nguyên tỉ lệ chưa được kiểm chứng riêng.
- **Cơ sở lý thuyết:** O4 (OFM) — kết quả ablation "Sampling Steps" của chính OFM.
- **Lợi ích kỳ vọng:** không tốn thêm code mới — chỉ là mở rộng cách đánh giá `num_sample_steps` đã có
  sẵn tham số ở PA3/PA4 (`args.num_sample_steps`), có thể chạy ngay trên các repo `DiffMM-OT-CFM`/
  `DiffMM-ODE` đã đóng gói.
- **Rủi ro:** rất thấp — thuần tuý là mở rộng thực nghiệm.
- **Độ khó triển khai:** ★☆☆☆☆

### ⚪ Phương án 9 (suy đoán, ưu tiên thấp, không bắt nguồn từ lý thuyết Flow Matching) — Cơ chế tinh chỉnh lặp nhiều vòng trong 1 bước thời gian, phỏng theo decoder kiểu RAFT của O5

- **Vị trí áp dụng:** D2 — kiến trúc mạng `Denoise` (hiện là 1 MLP đơn giản, dự đoán α̂₀ trong 1 lần
  forward duy nhất cho mỗi bước `t`).
- **Thay đổi (suy đoán, chưa có cơ sở lý thuyết trực tiếp từ Flow Matching):** OFM dùng 1 decoder kiểu
  RAFT có cơ chế **lặp tinh chỉnh nhiều vòng ngay bên trong 1 lần đánh giá** (không phải lặp qua các
  bước thời gian `t`, mà lặp *bên trong* 1 bước, dựa trên cost volume). Có thể thử thêm 1 vài vòng tinh
  chỉnh nội bộ tương tự cho `Denoise` (ví dụ: dự đoán α̂₀ sơ bộ, dùng nó tính lại 1 "tín hiệu tương quan"
  thô với `iEmbeds`, rồi tinh chỉnh lại α̂₀ 1-2 lần trước khi trả về kết quả cuối của bước `t`).
- **Cơ sở lý thuyết:** **yếu** — đây là ý tưởng kiến trúc mượn từ O5, **không** xuất phát từ lý thuyết
  Flow Matching (không có công thức toán đảm bảo cải thiện), giá trị của nó hoàn toàn phụ thuộc thực
  nghiệm.
- **Lợi ích kỳ vọng:** không chắc chắn — liệt kê để đầy đủ bức tranh, không khuyến nghị ưu tiên.
- **Rủi ro:** tăng chi phí tính toán ở D2 (vốn đã được gọi rất nhiều lần trong D4/D7) mà lợi ích chưa
  được chứng minh; có thể không đáng đánh đổi so với Phương án 6/8.
- **Độ khó triển khai:** ★★★☆☆ (dễ code, nhưng khó biện minh ưu tiên)

---

## 6. Lộ trình thực nghiệm đề xuất (Roadmap)

```
Bước 1 (vài giờ, rủi ro rất thấp)
  └─ Phương án 8: quét num_sample_steps ∈ {1,2,3,4,6} trên DiffMM-OT-CFM / DiffMM-ODE đã có sẵn
       │            → xác nhận DiffMM có cùng xu hướng "diminishing returns" như OFM hay không,
       │              trước khi đầu tư công sức vào bất kỳ phương án mới nào
       │
Bước 2 (1 tuần, phân tích/không code) 
  └─ Xác nhận lại bằng thực nghiệm nhỏ (không cần sửa DiffMM): kiểm tra giả thuyết nền của Phương án 7
       │            → nếu vẫn xác nhận D2 hiện tại (data-prediction) ổn định tốt, CHÍNH THỨC gác lại
       │              Phương án 7, tập trung nguồn lực vào Phương án 6
       │
Bước 3 (2-3 tuần, rủi ro trung bình-cao)
  └─ Phương án 6: cài đặt nhánh dự đoán α_l (global-matching-nhẹ), thử nghiệm 2 biến thể:
       │            (a) α_l cố định = 0 (baseline, tương đương hiện tại — kiểm tra hồi quy)
       │            (b) α_l học được — so sánh Recall@20/NDCG@20 + số NFE cần dùng ở D4
       │
Bước 4 (chỉ nếu Bước 3 có tín hiệu tích cực)
  └─ Kết hợp Phương án 6 + Phương án 5 (modal-conditioned OT path, đã có ở Flow_Matching_1):
       │            điểm neo α_l học được + tốc độ quỹ đạo điều kiện-modal cùng lúc
       │
Bước 5 (chỉ nếu phát sinh nhu cầu đổi tham số hoá)
  └─ Phương án 7: TVS — chỉ triển khai nếu Bước 3/4 cho thấy cần chuyển sang velocity-prediction
```

**Bảng ablation đề xuất** (mở rộng bảng đã có ở `DiffMM_FlowMatching_Optimization_Plan.md`, thêm dòng
cho các phương án mới):

| Biến thể | Recall@20 | NDCG@20 | NFE lúc sinh A^m | Ghi chú |
|---|---|---|---|---|
| DiffMM gốc / PA1-5 (đã có) | (tham chiếu) | (tham chiếu) | — | xem bảng gốc |
| + quét num_sample_steps hệ thống (PA8) | = PA3/PA4 | = PA3/PA4 | 1,2,3,4,6 | chỉ đổi cách đánh giá, không đổi mô hình |
| + Điểm neo α_l cố định=0 (PA6, kiểm tra hồi quy) | phải ≈ PA1 gốc | phải ≈ PA1 gốc | như PA1 | bắt buộc trùng khít trước khi thử (b) |
| + Điểm neo α_l học được (PA6b) | ? | ? | ? (kỳ vọng ít hơn) | so với PA1 và PA5 |
| + PA6 + PA5 kết hợp | ? | ? | ? | chỉ chạy nếu PA6b có tín hiệu tốt |

Chạy trên cả 3 dataset gốc (TikTok, Amazon-Baby, Amazon-Sports), giữ nguyên nguyên tắc "đo lường công
bằng" đã nêu ở mục 7 của kế hoạch PA1-5.

---

## 7. Rủi ro & lưu ý chung khi áp dụng

- **Không lặp lại sai lầm "ghép cho bằng được":** như đã phân tích ở mục 4-5, TVS (Phương án 7) là ví
  dụ rõ ràng của 1 kỹ thuật hay nhưng **không có khoảng trống thực sự** để lấp trong cấu hình hiện tại
  của DiffMM — luôn kiểm tra lại xem "vấn đề mà kỹ thuật nguồn giải quyết" có **thực sự tồn tại** trong
  hệ thống đích hay không, trước khi đầu tư công sức triển khai.
- **Phương án 6 cần kiểm tra hồi quy nghiêm ngặt:** vì đổi tâm nhiễu ảnh hưởng trực tiếp tới D1/D2 (nền
  tảng của mọi PA1-5), bắt buộc phải xác nhận cấu hình "α_l cố định = 0" cho kết quả **trùng khít**
  cấu hình gốc trước khi thử "α_l học được" — đúng tinh thần "công tắc bật/tắt an toàn" đã áp dụng nhất
  quán cho PA1-5 (đặc biệt là PA5, Giai đoạn B).
- **Chi phí tính toán của nhánh dự đoán α_l:** nếu dùng cơ chế khớp toàn cục kiểu OFM (so khớp toàn bộ
  user với toàn bộ item), cần đặc biệt lưu ý độ phức tạp tính toán trên tập item lớn (Amazon-Sports có
  hàng nghìn item) — có thể cần giới hạn (ví dụ top-k item gần nhất) thay vì so khớp toàn cục đầy đủ như
  OFM làm với ảnh (không gian toạ độ ảnh nhỏ hơn nhiều so với không gian item).
- **Đo lường công bằng:** giữ nguyên kiến trúc `Denoise` MLP và mọi hyperparameter khác khi so sánh
  Phương án 6/8, đúng nguyên tắc đã áp dụng xuyên suốt PA1-5.
