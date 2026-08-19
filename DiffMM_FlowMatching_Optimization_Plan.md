# Kế hoạch tối ưu DiffMM bằng Flow Matching

### Mục tiêu: dùng các phát hiện của bài Flow Matching (FM) để cải tiến khối Diffusion Model bên trong DiffMM

> Tài liệu này giả định bạn đã đọc [DiffMM_Review.md](DiffMM_Review.md) và [Flow_Matching_1_Review.md](Flow_Matching_1_Review.md). Các ký hiệu công thức (eq X) tham chiếu lại đúng số thứ tự đã dùng trong 2 file đó.

---

## 1. Vì sao 2 bài báo này "ghép" được với nhau?

Khối lõi của **DiffMM** (Giai đoạn 1-2-4 trong DiffMM_Review.md — Forward/Reverse Diffusion + Inference) chính là một **mô hình khuếch tán kiểu DDPM cổ điển** áp dụng lên vector tương tác nhị phân α₀ (thay vì lên ảnh). Bài **Flow Matching** đã chứng minh (Bảng 1 + mục Ablation, Flow_Matching_1_Review.md phần 4-5) rằng:

1. Mô hình khuếch tán kiểu DDPM chỉ là **một trường hợp riêng** (Diffusion conditional VF) trong họ đường đi tổng quát mà Flow Matching hỗ trợ.
2. Nếu **giữ nguyên đường đi khuếch tán cũ nhưng đổi cách huấn luyện** sang CFM (Conditional Flow Matching) → đã cải thiện (ổn định hơn, kết quả tốt hơn score matching gốc).
3. Nếu **đổi luôn sang đường đi Optimal Transport (OT — đường thẳng)** → cải thiện thêm: huấn luyện hội tụ nhanh hơn, sinh mẫu cần ít bước hơn (NFE thấp hơn ~40%), tổng quát hóa tốt hơn.

→ Vì khối diffusion của DiffMM về bản chất toán học **tương đương với "FM w/ Diffusion path"** trong bài FM, nên **mọi cải tiến mà FM đã chứng minh có tác dụng đều là ứng viên khả thi để cấy vào DiffMM**, nhắm đúng vào điểm mà chính ablation của DiffMM (mục "w/o DM" trong DiffMM_Review.md) đã chỉ ra là **thành phần quan trọng nhất** của toàn hệ thống.

**Điều kiện khả thi về mặt biểu diễn dữ liệu:** DiffMM đã sẵn coi α₀ (vector tương tác 0/1) như một biến **liên tục** trong không gian ℝ^|I| khi thêm nhiễu Gauss vào nó (eq 1 DiffMM) — đúng khuôn khổ mà Flow Matching yêu cầu (dữ liệu liên tục trong ℝᵈ). Việc rời rạc hóa chỉ xảy ra ở bước cuối (top-k threshold, Giai đoạn 4 DiffMM). Do đó **không cần thay đổi cách biểu diễn dữ liệu** để áp dụng FM — đây là điều kiện tiên quyết thuận lợi.

---

## 2. Liệt kê tách rời các bước của DiffMM

| # | Giai đoạn (theo DiffMM_Review.md) | Input → Output | Cơ chế cốt lõi | Công thức |
|---|---|---|---|---|
| D0 | Chuẩn bị dữ liệu | Đồ thị G, đặc trưng modal F̂ᵢᵐ | Biểu diễn tương tác user u thành vector nhị phân α₀ = aᵤ | — |
| D1 | Forward Diffusion Process | α₀ → αₜ | Thêm nhiễu Gauss dần theo T bước, lịch trình nhiễu tuyến tính (linear noise scheduler) | eq (1)-(3) |
| D2 | Reverse Diffusion Process (Denoising Model Training) | αₜ, t → α̂₀ | MLP dự đoán lại α₀ từ bản nhiễu; loss = ELBO rút gọn thành MSE có trọng số | eq (4)-(13) |
| D3 | Modality-aware Signal Injection (MSI) | α̂₀, eᵢᵐ, α₀, eᵢ → L_msi | Loss phụ ép embedding theo modal-view khớp với embedding theo collaborative-view thật | eq (14), L_dm = L_elbo + λ₀L_msi |
| D4 | Inference của Diffusion Model | α₀ → α_T' (forward T' bước) → α̂₀ (reverse T bước, deterministic) → A^m | Corrupt một phần rồi denoise ngược lại **T bước lặp tuần tự**, sau đó top-k chọn cạnh mới | — |
| D5 | Cross-Modal Contrastive Augmentation | A^m → z^m, ẑ^m → L_cl | GNN aggregation trên A^m + đồ thị gốc A, 2 kiểu InfoNCE (modality-view / main-view làm anchor) | eq (15)-(19) |
| D6 | Multi-Modal Graph Aggregation (nhánh chính) | F̂^m, A, A^m → H | GNN gộp đa modal (trọng số κ_m học được) + lan truyền L lớp trên A, chống over-smoothing (ω) | eq (20)-(23) |
| D7 | Multi-Task Training | — | L_dm^m = L_elbo + λ₀L_msi^m (diffusion) ; L_rec = L_bpr + λ₁L_cl + λ₂‖Θ‖² (recommendation) | eq (24)-(26) |

**Nhận xét quan trọng:** D1+D2+D4 chính là "cỗ máy diffusion" tốn kém nhất — D4 đặc biệt đáng chú ý vì **mỗi lần cần tạo lại đồ thị A^m** (lặp lại nhiều lần trong suốt quá trình huấn luyện đa nhiệm D7), DiffMM phải chạy **T bước tuần tự** của mạng denoiser (giống hệt vấn đề "NFE cao" mà bài FM chỉ ra ở diffusion path).

---

## 3. Liệt kê tách rời các bước của Flow Matching

| # | Giai đoạn (theo Flow_Matching_1_Review.md) | Input → Output | Cơ chế cốt lõi | Công thức |
|---|---|---|---|---|
| F0 | Nền tảng CNF | x, t → φₜ(x) | Flow xác định bởi ODE dφₜ/dt = vₜ(φₜ) | eq (1)-(4) |
| F1 | Mục tiêu FM (chưa khả thi) | pₜ, uₜ | Muốn regress vₜ vào uₜ nhưng cả 2 đều không biết công thức | eq (5) |
| F2 | Xây pₜ, uₜ từ conditional path | x₁ → pₜ(x\|x₁), uₜ(x\|x₁) | Marginalize (lấy biên) qua từng mẫu dữ liệu x₁ → Theorem 1 | eq (6)-(8) |
| F3 | Conditional Flow Matching (CFM) | x₁, t, x → loss | Loss tính được trực tiếp, gradient giống hệt FM gốc (Theorem 2) | eq (9) |
| F4 | Thiết kế họ đường đi Gauss có điều kiện | μₜ(x₁), σₜ(x₁) | Chọn tự do 2 hàm số μₜ, σₜ, suy ra uₜ(x\|x₁) bằng công thức đóng (Theorem 3) | eq (10)-(15) |
| F5a | Instance: Diffusion path | β(t) → μₜ, σₜ | Khớp lại VE/VP diffusion cũ — quỹ đạo **cong** | eq (16)-(19) |
| F5b | Instance: Optimal Transport (OT) path | — | μₜ, σₜ tuyến tính theo t — quỹ đạo **thẳng**, tốc độ không đổi | eq (20)-(24) |
| F6 | Thuật toán huấn luyện | x₁, x₀, t → gradient step | Lấy mẫu + tính target đóng + MSE — **không cần giải ODE lúc train** | — |
| F7 | Suy luận/Sinh mẫu | x₀ → x₁ | Giải ODE bằng solver (Euler/Midpoint/RK4/dopri5), OT path cần **ít NFE hơn hẳn** | — |

---

## 4. Bảng đối chiếu tương đồng (mapping) DiffMM ↔ Flow Matching

| Thành phần DiffMM | Vai trò tương đương trong FM | Khoảng cách/khác biệt cốt lõi |
|---|---|---|
| D1 (Forward Diffusion, eq 1-3) | F5a (Diffusion path — khởi tạo pₜ(x\|x₁)) | DiffMM dùng **đúng** công thức diffusion path (VP-style) mà FM chứng minh là **quỹ đạo cong**, kém hơn OT path |
| D2 (Reverse Diffusion / ELBO, eq 4-13) | F1→F3 (suy ra CFM loss qua Theorem 1+2) | DiffMM tự suy ELBO/KL divergence phức tạp để ra target α₀; FM cho một con đường **ngắn hơn, tổng quát hơn** để ra target (velocity uₜ(x\|x₁)) qua công thức đóng (Theorem 3), không cần suy Bayes riêng cho từng loại path |
| D4 (Inference, T bước reverse tuần tự) | F7 (giải ODE bằng solver) | D4 dùng **T bước cố định, tuần tự** (giống fixed-step solver rất nhiều bước trên path cong) → NFE cao; F7 + OT path chỉ cần **vài bước** (path thẳng, tốc độ hằng) |
| D3 (MSI — loss phụ ép theo modal) | F4 (tự do thiết kế μₜ(x₁), σₜ(x₁)) | DiffMM "ép" modal thông qua **loss cộng thêm**; FM cho phép **thiết kế thẳng vào đường đi** (μₜ, σₜ là hàm số tùy chọn) — MSI có thể trở thành một phần *kiến trúc* thay vì một phần *loss* |
| D7 λ₀ (cân bằng L_elbo/L_msi) | Không có tương đương trực tiếp — đây là hyperparameter đặc thù của DiffMM | Nếu MSI được tích hợp vào path (theo hướng trên), có thể **giảm bớt 1 hyperparameter cần tune** |

---

## 5. Các phương án tối ưu khả thi (xếp theo độ ưu tiên / độ rủi ro tăng dần)

### 🟢 Phương án 1 — "Quick win": chỉ đổi lịch trình nhiễu (noise scheduler) từ kiểu VP sang kiểu OT, giữ nguyên toàn bộ ELBO/kiến trúc

- **Vị trí áp dụng:** D1 — công thức (3) trong DiffMM (`1 − γ̄ₜ = s·[γ_min + (t−1)/(T−1)·(γ_max−γ_min)]`)
- **Thay đổi:** thay công thức γ̄ₜ hiện tại (tăng theo kiểu phương sai cộng dồn) bằng phép nội suy **tuyến tính kiểu OT** (F5b, eq 20 FM): μₜ = t·α₀, σₜ = 1−(1−σ_min)·t. Mọi phần còn lại của DiffMM (ELBO, MSI, top-k...) **giữ nguyên 100%**.
- **Cơ sở lý thuyết:** đúng thí nghiệm "ablation nội bộ" mà chính bài FM đã làm (Flow_Matching_1_Review.md, mục 5) — tách riêng ảnh hưởng của "hình dạng đường đi" khỏi "cách huấn luyện".
- **Lợi ích kỳ vọng:** đo trực tiếp được liệu quỹ đạo thẳng hơn có cải thiện Recall@20/NDCG@20 hay không, **mà không cần viết lại code huấn luyện**.
- **Rủi ro:** thấp — chỉ đổi 1 công thức, dễ rollback.
- **Độ khó triển khai:** ★☆☆☆☆ (rất dễ — sửa 1 hàm tính lịch trình nhiễu)

### 🟢 Phương án 2 — Đổi target dự đoán + loss huấn luyện từ ELBO sang CFM, giữ nguyên path diffusion cũ

- **Vị trí áp dụng:** D2 — toàn bộ khối suy diễn Bayes (eq 6-13 DiffMM)
- **Thay đổi:** bỏ quy trình suy ELBO → KL divergence → công thức đóng q(αₜ₋₁|αₜ,α₀) phức tạp; thay bằng CFM loss trực tiếp kiểu F3 (eq 9 FM): với path diffusion hiện tại, tính target uₜ(αₜ|α₀) qua Theorem 3 (eq 15 FM: `uₜ = σₜ'/σₜ·(x−μₜ) + μₜ'`), rồi regress mạng denoiser thẳng vào target này bằng MSE — **không cần bước trung gian tính μθ, σ²(t) qua Bayes**.
- **Cơ sở lý thuyết:** finding chính của FM (Theorem 2) — CFM cho gradient **giống hệt** FM gốc nhưng **huấn luyện ổn định và mạnh mẽ hơn hẳn** score-matching/ELBO truyền thống, ngay cả khi giữ nguyên path.
- **Lợi ích kỳ vọng:** huấn luyện diffusion module (D2) ổn định hơn, code đơn giản hơn (bỏ được cả khối suy diễn Bayes phức tạp trong DiffMM).
- **Rủi ro:** trung bình — DiffMM hiện tại tham số hóa theo kiểu "dự đoán α₀ trực tiếp" (data-prediction), còn CFM chuẩn tham số hóa theo "dự đoán vận tốc" (velocity-prediction) → cần suy lại công thức quy đổi qua lại giữa 2 cách tham số hóa này, và kiểm tra MSI (D3, vốn đang dùng α̂₀ trực tiếp trong eq 14) có cần viết lại tương ứng không.
- **Độ khó triển khai:** ★★★☆☆

### 🟡 Phương án 3 — Kết hợp Phương án 1 + 2: thay trọn khối Diffusion Model của DiffMM bằng "Graph Conditional Flow Matching" dùng OT path

- **Vị trí áp dụng:** D1 + D2 + D4 (gộp toàn bộ 3 giai đoạn)
- **Thay đổi:** dùng path OT (Phương án 1) **và** loss CFM (Phương án 2) cùng lúc — đây chính là cấu hình "FM w/ OT" đã thắng tuyệt đối trong Bảng 1 của bài FM. D4 (Inference) sẽ đổi từ "T bước reverse tuần tự" thành "giải ODE bằng solver rẻ (Euler/Midpoint) với rất ít bước" vì quỹ đạo OT là đường thẳng, tốc độ không đổi.
- **Cơ sở lý thuyết:** kết quả thực nghiệm mạnh nhất trong bài FM (FM w/ OT > FM w/ Diffusion > DDPM/SM/ScoreFlow trên mọi chỉ số: NLL, FID, NFE).
- **Lợi ích kỳ vọng:** đòn bẩy lớn nhất trong toàn bộ kế hoạch — vì chính ablation "w/o DM" trong DiffMM_Review.md đã cho thấy **chất lượng của khối Diffusion Model ảnh hưởng mạnh nhất** đến hiệu năng cuối cùng (Recall/NDCG) trong toàn hệ thống DiffMM. Ngoài ra, việc D4 chỉ cần vài NFE thay vì T bước sẽ giảm mạnh chi phí runtime của cả vòng lặp Multi-Task Training (D7), vì đồ thị A^m phải được tái tạo nhiều lần trong quá trình huấn luyện.
- **Rủi ro:** cao nhất trong nhóm — cần viết lại phần lớn module diffusion, tự chọn σ_min phù hợp với thang giá trị của α (vốn là xác suất/nhị phân, khác pixel ảnh), và kiểm định lại toàn bộ pipeline (D3 MSI, D5 contrastive, D6 aggregation) vẫn tương thích với target/dạng đầu ra mới.
- **Độ khó triển khai:** ★★★★☆

### 🟡 Phương án 4 — Chỉ tăng tốc bước Inference (D4) bằng ODE solver, không đổi cách huấn luyện

- **Vị trí áp dụng:** D4 — thủ tục sinh α̂₀ hiện tại (forward T' bước + reverse T bước deterministic)
- **Thay đổi:** giữ nguyên toàn bộ D1-D2-D3 (không đổi gì về huấn luyện); chỉ viết lại vòng lặp suy luận ở D4 dưới dạng một **ODE solver tường minh** (Euler/Midpoint) với số bước ít hơn T hiện tại — vì bản chất bước "reverse deterministic dùng μθ, bỏ variance" (eq 4 DiffMM) vốn đã tương đương một Euler-step đơn giản của một ODE ẩn.
- **Cơ sở lý thuyết:** F7 (Flow_Matching_1_Review.md) — off-the-shelf ODE solver có thể thay thế thủ tục sampling thủ công của diffusion, với khả năng đánh đổi NFE/độ chính xác linh hoạt hơn.
- **Lợi ích kỳ vọng:** tăng tốc runtime **ngay lập tức**, không cần huấn luyện lại mô hình đã có.
- **Rủi ro:** thấp về mặt code, nhưng lợi ích **giới hạn hơn nhiều** so với Phương án 1/3 — vì quỹ đạo vẫn "cong" theo path diffusion cũ, dùng ít bước trên đường cong dễ mất độ chính xác hơn so với dùng ít bước trên đường OT thẳng.
- **Độ khó triển khai:** ★★☆☆☆ — phù hợp làm bước đo baseline nhanh trước khi đầu tư Phương án 1/3.

### 🔴 Phương án 5 (nâng cao, hướng nghiên cứu dài hạn) — Modality-conditioned OT path: tích hợp MSI vào định nghĩa đường đi thay vì làm loss phụ

- **Vị trí áp dụng:** D3 (MSI) tái cấu trúc thành một phần của F4 (thiết kế μₜ, σₜ)
- **Thay đổi:** thay vì ép α̂₀ khớp với embedding modal qua loss L_msi (eq 14 DiffMM, cần hyperparameter λ₀ để cân bằng), hãy thử định nghĩa μₜ, σₜ **phụ thuộc luôn vào đặc trưng modal eᵢᵐ** — ví dụ μₜ(α₀, eᵐ) là một hàm trộn giữa vị trí đích α₀ và "lực hút" về phía các item có eᵐ tương đồng — để bản thân quỹ đạo sinh ra đã thiên vị đúng theo modal, không cần loss phụ trợ nữa.
- **Cơ sở lý thuyết:** F4 (Flow_Matching_1_Review.md) — Flow Matching cho phép μₜ(x₁), σₜ(x₁) là **hàm số tùy ý** miễn thỏa điều kiện biên, mở khả năng "nhúng" thêm điều kiện phụ (ở đây là modal) trực tiếp vào định nghĩa path.
- **Lợi ích kỳ vọng:** giảm số hyperparameter cần tune (bớt λ₀), có thể cho kết quả nhất quán hơn giữa các dataset (hiện tại DiffMM_Review.md ghi nhận λ₀ tối ưu khác nhau tùy dataset).
- **Rủi ro:** đây là ý tưởng **chưa có tiền lệ trực tiếp trong cả 2 bài báo gốc** — cần tự thiết kế công thức, chứng minh lại điều kiện biên vẫn thỏa mãn, và validate thực nghiệm từ đầu. Phù hợp là **hướng đóng góp mới** (đủ để viết thành phần chính của một bài báo/luận văn) hơn là một tinh chỉnh kỹ thuật đơn thuần.
- **Độ khó triển khai:** ★★★★★

---

## 6. Lộ trình thực nghiệm đề xuất (Roadmap)

```
Bước 1 (1 tuần, rủi ro thấp)
  └─ Phương án 4: đổi solver suy luận D4 → đo tốc độ/chất lượng cơ sở
       │
Bước 2 (1-2 tuần, rủi ro thấp)
  └─ Phương án 1: đổi noise scheduler D1 sang OT-linear, giữ nguyên D2
       │            → so sánh Recall@20/NDCG@20 với DiffMM gốc
       │
Bước 3 (2-3 tuần, rủi ro trung bình)
  └─ Phương án 2: đổi loss D2 từ ELBO → CFM, giữ path cũ
       │            → so sánh riêng với Bước 2 để tách bạch 2 đóng góp
       │              (đúng tinh thần "ablation" mà FM paper tự làm)
       │
Bước 4 (kết hợp, rủi ro trung bình-cao)
  └─ Phương án 3: OT path + CFM loss cùng lúc (Bước 2 + Bước 3)
       │            → đây là ứng viên chính cho phiên bản "DiffMM-FM"
       │
Bước 5 (nếu Bước 4 thành công, nghiên cứu mở rộng)
  └─ Phương án 5: modality-conditioned OT path
```

**Bảng ablation đề xuất** (mở rộng theo đúng khuôn Table 2 của DiffMM_Review.md, thêm cột NFE để đo chi phí sinh A^m):

| Biến thể | Recall@20 | NDCG@20 | NFE lúc sinh A^m | Thời gian train tổng |
|---|---|---|---|---|
| DiffMM gốc (Diffusion path + ELBO) | (baseline) | (baseline) | T (đầy đủ) | (baseline) |
| + OT path, giữ ELBO (Phương án 1) | ? | ? | T | ? |
| + CFM loss, giữ Diffusion path (Phương án 2) | ? | ? | T | ? |
| + OT path + CFM loss (Phương án 3) | ? | ? | ≪T (kỳ vọng ~40-60%) | ? (kỳ vọng giảm) |
| + Modality-conditioned OT (Phương án 5) | ? | ? | ≪T | ? |

Chạy trên cả 3 dataset gốc của DiffMM (TikTok, Amazon-Baby, Amazon-Sports) để đảm bảo kết luận không chỉ đúng cục bộ trên 1 dataset — giống cách chính DiffMM_Review.md đã trình bày kết quả.

---

## 7. Rủi ro & lưu ý chung khi áp dụng

- **Khác biệt về loại dữ liệu:** FM được kiểm chứng trên ảnh (dữ liệu liên tục, mật độ cao); DiffMM vận hành trên vector tương tác **rất thưa** (sparse — phần lớn giá trị 0). Cần kiểm tra: đường thẳng OT nội suy giữa nhiễu Gauss và vector 0/1 thưa có tạo ra "đường đi" hợp lý về mặt xác suất tương tác hay không (ví dụ có cần chặn giá trị trong [0,1] không, hay để mô hình tự học qua top-k threshold như DiffMM đang làm).
- **σ_min cần chọn lại:** giá trị σ_min trong bài FM được thiết kế cho ảnh (pixel chuẩn hóa); với đồ thị tương tác cần thử nghiệm quét (sweep) giá trị riêng.
- **Tương thích với MSI (D3) và Contrastive (D5):** cả 2 khối này hiện dùng trực tiếp α̂₀ (đầu ra dự đoán α₀) — nếu chuyển sang tham số hóa velocity-prediction (Phương án 2/3), cần suy ra công thức chuyển đổi ngược từ vận tốc dự đoán về α̂₀ tương ứng trước khi đưa vào D3/D5, để không phải viết lại toàn bộ D3, D5, D6.
- **Đo lường công bằng:** luôn giữ nguyên kiến trúc mạng denoiser (MLP) và mọi hyperparameter khác khi so sánh — đúng tinh thần "cùng kiến trúc, chỉ đổi loss/path" mà bài FM đã làm ở Bảng 1, để kết luận rút ra thực sự đến từ Flow Matching chứ không phải từ thay đổi ngẫu nhiên khác.
