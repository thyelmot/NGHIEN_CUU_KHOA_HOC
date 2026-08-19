# DiffMM: Multi-Modal Diffusion Model for Recommendation
### (Mô hình Khuếch tán Đa phương thức cho Hệ thống Gợi ý)

**Nguồn:** Yangqin Jiang, Lianghao Xia, Wei Wei, Da Luo, Kangyi Lin, Chao Huang — ACM Multimedia 2024 (HKU + Tencent).
Code: github.com/HKUDS/DiffMM

> Bản này viết lại **dễ hiểu hơn**: mọi thuật ngữ tiếng Anh được giải thích ngay bằng tiếng Việt trong ngoặc **ngay tại chỗ xuất hiện** (lặp lại xuyên suốt bài, không bắt bạn phải nhớ hay tra lại), và mọi ký hiệu/công thức toán đều được diễn giải bằng lời trước khi dùng.

---

## 0. Bảng thuật ngữ nền (đọc trước để dễ theo dõi)

| Thuật ngữ tiếng Anh | Giải thích tiếng Việt |
|---|---|
| **Recommendation / Recommender system** | Hệ thống gợi ý — dự đoán user (người dùng) sẽ thích item (sản phẩm/video/bài hát) nào |
| **User / Item** | User = người dùng; Item = sản phẩm/nội dung được gợi ý (ví dụ: video TikTok, sản phẩm Amazon) |
| **Interaction** | Tương tác — hành động user đã từng bấm/xem/mua item nào đó |
| **Modality (số nhiều: modalities)** | Phương thức/loại dữ liệu mô tả item — ví dụ: hình ảnh (visual), văn bản (textual), âm thanh (acoustic) |
| **Multi-modal** | Đa phương thức — dùng nhiều loại dữ liệu (ảnh + chữ + âm thanh) cùng lúc |
| **Embedding** | Một vector số (dãy số thực) đại diện cho một user/item, giúp máy tính "hiểu" và so sánh được sự giống/khác nhau |
| **Graph** | Đồ thị — ở đây là mạng lưới các đường nối giữa user và item mà user đã tương tác (giống một tấm lưới các cặp "user–item") |
| **Sparse / Sparsity** | Thưa/độ thưa — tình trạng phần lớn user chỉ tương tác với rất ít item, thiếu dữ liệu để học |
| **Self-supervised learning (SSL)** | Học tự giám sát — máy tự tạo ra "bài tập" và "đáp án" từ chính dữ liệu (không cần người gán nhãn) để học thêm |
| **Data augmentation** | Tăng cường dữ liệu — tạo thêm dữ liệu giả lập/biến thể từ dữ liệu gốc để mô hình học tốt hơn |
| **Noise / Noisy** | Nhiễu — thông tin sai lệch, không mong muốn, làm giảm chất lượng dữ liệu |
| **Diffusion model** | Mô hình khuếch tán — kỹ thuật sinh dữ liệu (nổi tiếng trong tạo ảnh AI) hoạt động theo 2 chiều: (1) từ từ "làm hỏng/làm nhiễu" dữ liệu gốc, (2) học cách "phục hồi" lại dữ liệu gốc từ bản bị nhiễu |
| **Forward process** | Quá trình thuận — bước làm nhiễu dữ liệu dần dần |
| **Reverse process** | Quá trình nghịch — bước khử nhiễu, phục hồi dữ liệu dần dần |
| **Denoising** | Khử nhiễu — loại bỏ nhiễu để lấy lại dữ liệu sạch |
| **Gaussian noise / Gaussian distribution** | Nhiễu Gauss / phân phối Gauss (còn gọi phân phối chuẩn) — một dạng nhiễu ngẫu nhiên rất phổ biến trong toán, có hình chuông, đặc trưng bởi giá trị trung bình (mean) và độ phân tán (variance/phương sai) |
| **Mean** | Giá trị trung bình — điểm "trung tâm" mà nhiễu Gauss dao động quanh nó |
| **Variance** | Phương sai — mức độ "dao động rộng hay hẹp" quanh giá trị trung bình |
| **Markov chain** | Chuỗi Markov — một chuỗi các bước mà bước sau chỉ phụ thuộc vào bước ngay trước nó (không cần nhớ toàn bộ lịch sử) |
| **Reparameterization trick** | Mẹo "đổi biến" — một kỹ thuật toán học giúp tính nhanh trực tiếp từ bước đầu đến bước bất kỳ mà không cần lặp từng bước một |
| **Neural network / MLP (Multi-Layer Perceptron)** | Mạng nơ-ron / Mạng nơ-ron nhiều lớp — một mô hình máy học gồm nhiều lớp phép tính nối tiếp, dùng để học các quy luật phức tạp từ dữ liệu |
| **GNN (Graph Neural Network)** | Mạng nơ-ron đồ thị — loại mạng nơ-ron chuyên xử lý dữ liệu dạng đồ thị, hoạt động bằng cách cho các node (user/item) "truyền thông tin" qua lại với hàng xóm của mình trên đồ thị |
| **Message passing** | Truyền thông điệp — cơ chế trong GNN: mỗi node gửi/nhận thông tin (embedding) từ các node láng giềng để cập nhật hiểu biết của mình |
| **Aggregation** | Tổng hợp/gộp — bước cộng dồn/kết hợp nhiều thông tin (từ nhiều node, nhiều modality...) lại thành một |
| **Contrastive learning** | Học đối chiếu — kỹ thuật huấn luyện: kéo các cặp dữ liệu "giống nhau" (positive pairs) lại gần nhau và đẩy các cặp "khác nhau" (negative pairs) ra xa nhau trong không gian embedding |
| **Anchor** | Điểm neo — trong contrastive learning, là embedding được chọn làm "điểm chuẩn" để so sánh với các embedding khác |
| **Positive pair / Negative pair** | Cặp dương/cặp âm — cặp dương là 2 thứ nên giống nhau (ví dụ: cùng 1 user nhìn từ 2 modal khác nhau); cặp âm là 2 thứ nên khác nhau (ví dụ: 2 user khác nhau) |
| **InfoNCE loss** | Hàm mất mát InfoNCE — công thức toán cụ thể để thực hiện contrastive learning (kéo gần cặp dương, đẩy xa cặp âm) |
| **Loss / Loss function** | Hàm mất mát — con số đo "mô hình đang sai bao nhiêu"; huấn luyện = tìm cách giảm số này càng nhỏ càng tốt |
| **Training** | Huấn luyện — quá trình cho mô hình học từ dữ liệu bằng cách chỉnh tham số để giảm loss |
| **Inference** | Suy luận — giai đoạn dùng mô hình đã huấn luyện xong để đưa ra dự đoán thực tế (không học thêm nữa) |
| **BPR (Bayesian Personalized Ranking)** | Xếp hạng cá nhân hóa kiểu Bayes — một hàm loss chuyên dùng cho gợi ý: buộc item mà user đã thích phải được chấm điểm cao hơn item ngẫu nhiên mà user chưa từng tương tác |
| **Ranking** | Xếp hạng — sắp xếp danh sách item theo điểm số dự đoán, cao nhất lên đầu |
| **Top-K** | Top-K — chọn ra K item có điểm cao nhất để gợi ý |
| **Hyperparameter** | Siêu tham số — các "nút vặn" của mô hình do người thiết kế chọn trước khi huấn luyện (không tự học được), ví dụ: số bước khuếch tán T, trọng số λ |
| **Regularization / L2 regularization** | Điều chuẩn / điều chuẩn L2 — kỹ thuật phạt các tham số mô hình quá lớn, giúp tránh học vẹt (overfitting) |
| **Overfitting** | Học vẹt/quá khớp — mô hình học thuộc lòng dữ liệu huấn luyện nhưng dự đoán kém trên dữ liệu mới |
| **Over-smoothing** | Làm mượt quá mức — vấn đề trong GNN khi truyền quá nhiều lớp khiến các embedding của mọi node trở nên giống hệt nhau, mất khả năng phân biệt |
| **Ablation (study)** | Nghiên cứu cắt bỏ — thử nghiệm bỏ từng thành phần của mô hình ra để xem thành phần đó quan trọng tới đâu |
| **Baseline** | Mô hình nền/mô hình đối chứng — các mô hình có sẵn (của người khác) dùng để so sánh với mô hình mới đề xuất |
| **Benchmark / Dataset** | Bộ dữ liệu chuẩn — tập dữ liệu công khai dùng để đánh giá, so sánh các mô hình |
| **Recall@K, Precision@K, NDCG@K** | Ba chỉ số đánh giá độ chính xác gợi ý (giải thích chi tiết ở phần Kết quả bên dưới) |
| **VAE (Variational Autoencoder)** | Bộ tự mã hóa biến phân — một loại mô hình sinh dữ liệu khác (không phải diffusion), dùng để so sánh/thay thế trong ablation |
| **VGAE (Variational Graph Autoencoder)** | VAE áp dụng cho dữ liệu đồ thị — dùng làm đối chứng thay cho diffusion model trong ablation |
| **KL divergence (Kullback–Leibler divergence)** | Độ phân kỳ KL — một cách đo "hai phân phối xác suất khác nhau bao nhiêu"; dùng trong công thức toán của diffusion model |
| **ELBO (Evidence Lower Bound)** | Cận dưới bằng chứng — một công thức toán trong huấn luyện diffusion model, dùng làm mục tiêu tối ưu (thay vì tính trực tiếp công thức gốc quá phức tạp) |
| **MSE (Mean Squared Error)** | Sai số bình phương trung bình — cách đo khoảng cách giữa 2 giá trị/vector bằng cách lấy trung bình bình phương hiệu số của chúng |

---

## 1. Bài toán & động lực nghiên cứu

Các nền tảng như TikTok, YouTube gợi ý nội dung dựa trên nhiều **modality** (loại dữ liệu) của item: hình ảnh, chữ mô tả, âm thanh. Có 2 vấn đề paper muốn giải quyết:

1. **Dữ liệu tương tác thưa (sparse interactions):** phần lớn user chỉ tương tác (interaction) với rất ít item trong tổng số hàng triệu item → mô hình học có giám sát (supervised — học từ nhãn có sẵn, ở đây là các tương tác đã ghi nhận) không đủ dữ liệu để học chính xác.
2. **Tăng cường dữ liệu (data augmentation) ngẫu nhiên gây nhiễu:** các phương pháp SSL (học tự giám sát) trước đó như SGL, NCL, HCCF tạo dữ liệu giả lập bằng cách xóa ngẫu nhiên vài node/edge (đỉnh/cạnh của đồ thị) — cách này **không quan tâm đến nội dung modal thực sự**, nên dễ vô tình tạo ra tín hiệu sai (noise).

**Ý tưởng cốt lõi của DiffMM:** thay vì tạo dữ liệu giả bằng cách xóa ngẫu nhiên, hãy dùng **diffusion model** (mô hình khuếch tán — công nghệ đứng sau các AI vẽ ảnh như Stable Diffusion) để "tưởng tượng/sinh ra" một đồ thị user–item mới **có điều kiện theo từng modal** — nghĩa là đồ thị sinh ra phản ánh đúng "nếu chỉ nhìn theo ảnh/chữ/âm thanh thì user sẽ thích item nào". Đồ thị này sau đó được dùng làm dữ liệu tăng cường (augmentation) chất lượng cao cho contrastive learning (học đối chiếu).

---

## 2. Kiến trúc tổng thể (4 khối chính — theo Hình 1 trong paper)

```
Đặc trưng thô của các modality (visual/textual/acoustic — ảnh/chữ/âm thanh)
        │
        ▼
┌───────────────────────────────────┐
│ (A) Multi-Modal Graph Diffusion    │  Forward process: làm nhiễu đồ thị tương tác gốc
│     Model (Mô hình khuếch tán      │  Reverse process: khử nhiễu, có "định hướng" theo modal
│     đồ thị đa phương thức)         │
└───────────────────────────────────┘
        │  sinh ra A^m = đồ thị "modality-aware" (đồ thị nhận biết modal) — mỗi modal 1 đồ thị riêng
        ▼
┌───────────────────────────────┐        ┌──────────────────────────────────┐
│ (B) Multi-Modal Graph          │◄──────►│ (C) Cross-Modal Contrastive        │
│     Aggregation (Tổng hợp đồ   │        │     Augmentation (Tăng cường đối   │
│     thị đa phương thức — nhánh │        │     chiếu xuyên-modal, dùng A^m)   │
│     dự đoán chính, dùng đồ thị │        │                                    │
│     gốc A)                     │        │                                    │
└───────────────────────────────┘        └──────────────────────────────────┘
        │
        ▼
   Embedding user/item cuối cùng H → dự đoán điểm số ŷ_ui = h_u · h_i
```

**Tóm tắt bằng lời:** Khối (A) học cách "vẽ lại" đồ thị tương tác dựa trên đặc trưng modal → tạo ra khối (C) các cặp dữ liệu để dạy mô hình rằng "nhìn từ modal nào thì user/item cũng phải có ý nghĩa nhất quán" → khối (B) là nhánh chính dùng GNN để tính embedding cuối cùng và đưa ra gợi ý.

---

## 3. Quy trình chi tiết — hình dung từng bước

### Giai đoạn 0 — Chuẩn bị dữ liệu đầu vào

- Đồ thị tương tác gốc **G = {(u,i)}**: mỗi cặp (u,i) nghĩa là user u đã tương tác với item i.
- Với mỗi item i và mỗi modal m (visual/textual/acoustic), có một vector đặc trưng thô **F̂ᵢᵐ** — đây là một dãy số (embedding) được trích sẵn từ các mô hình chuyên dụng có sẵn (ví dụ mạng nơ-ron xử lý ảnh cho visual, mạng xử lý văn bản cho textual).
- Với mỗi user u, biểu diễn toàn bộ tương tác của họ dưới dạng một vector nhị phân (chỉ gồm số 0 và 1):
  **α₀ = aᵤ = [a¹ᵤ, a²ᵤ, ..., a^|I|ᵤ]**
  Ý nghĩa: vector này dài bằng tổng số item; ở vị trí item i, giá trị = 1 nếu user u đã tương tác với item i, = 0 nếu chưa. Đây chính là "dữ liệu gốc" mà mô hình khuếch tán sẽ học cách làm nhiễu rồi phục hồi.

### Giai đoạn 1 — Forward Process (Quá trình thuận: làm nhiễu đồ thị)

Đây là bước "phá hủy" dữ liệu một cách có kiểm soát, mô phỏng theo kỹ thuật DDPM (Denoising Diffusion Probabilistic Model — Mô hình khuếch tán xác suất khử nhiễu):

**Công thức (1):** q(αₜ | αₜ₋₁) = N( √(1−βₜ) · αₜ₋₁ ,  βₜ · I )

*Giải nghĩa từng ký hiệu:*
- **αₜ**: phiên bản của vector tương tác sau khi đã bị làm nhiễu t lần (t = 0, 1, 2, ..., T).
- **N(mean, variance)**: ký hiệu phân phối Gauss (phân phối chuẩn) — nghĩa là "giá trị αₜ được lấy ngẫu nhiên quanh một giá trị trung bình (mean), với độ dao động (variance) cho trước".
- **βₜ**: một con số nhỏ (nằm trong khoảng 0 đến 1) quyết định "bơm bao nhiêu nhiễu" ở bước t — βₜ càng lớn thì nhiễu thêm vào càng nhiều.
- **I**: ma trận đơn vị (identity matrix) — chỉ mang ý nghĩa kỹ thuật là nhiễu được thêm độc lập vào từng chiều của vector, không cần hiểu sâu.
- **Ý nghĩa cả công thức:** để có αₜ, ta lấy αₜ₋₁ (bước ngay trước), thu nhỏ nó lại một chút (nhân với √(1−βₜ)), rồi cộng thêm một chút nhiễu ngẫu nhiên có độ lớn βₜ. Lặp lại T lần liên tiếp → dữ liệu ban đầu α₀ dần dần "chìm" trong nhiễu.
- Đây là một **Markov chain** (chuỗi Markov): bước αₜ chỉ phụ thuộc vào αₜ₋₁ ngay trước nó, không cần nhớ toàn bộ các bước trước đó.

**Công thức (2):** q(αₜ | α₀) = N( √γₜ · α₀ ,  (1−γₜ) · I ), với ε ~ N(0, I)

*Giải nghĩa:* nhờ một mẹo toán học gọi là **reparameterization trick** (mẹo đổi biến), ta không cần lặp t bước một cách tuần tự mà có thể **nhảy thẳng** từ α₀ đến αₜ ở bất kỳ bước t nào chỉ bằng một phép tính — tiết kiệm rất nhiều thời gian tính toán khi huấn luyện.
- **γₜ**: một hệ số tổng hợp cho biết "còn giữ lại bao nhiêu % thông tin gốc" sau t bước làm nhiễu (γₜ càng nhỏ nghĩa là càng nhiều nhiễu đã được thêm vào, thông tin gốc còn lại càng ít).
- **ε ~ N(0, I)**: một nhiễu ngẫu nhiên chuẩn (lấy mẫu ngẫu nhiên từ phân phối Gauss có trung bình 0).

**Công thức (3):** 1 − γ̄ₜ = s · [ γ_min + (t−1)/(T−1) · (γ_max − γ_min) ]

*Giải nghĩa:* đây là **lịch trình nhiễu tuyến tính (linear noise scheduler)** — công thức quyết định trước "ở bước t thì lượng nhiễu là bao nhiêu", tăng dần đều từ γ_min đến γ_max qua T bước; s là hệ số chỉnh tổng thể mức nhiễu.

**Hình dung dễ nhất:** giống như lấy một bức ảnh rõ nét (đồ thị tương tác gốc), rồi rắc "hạt nhiễu" (giống hạt bụi/tuyết trên tivi cũ) lên ảnh đó từng chút một qua nhiều bước, đến bước cuối cùng T thì bức ảnh chỉ còn toàn nhiễu trắng, không nhìn ra gì nữa.

### Giai đoạn 2 — Reverse Process (Quá trình nghịch: khử nhiễu, phục hồi lại đồ thị)

Đây là phần mô hình phải **học** — làm sao đi ngược lại từ nhiễu để phục hồi đồ thị gốc.

**Công thức (4):** p_θ(αₜ₋₁ | αₜ) = N( μ_θ(αₜ,t) ,  Σ_θ(αₜ,t) )

*Giải nghĩa:*
- **p_θ**: ký hiệu phân phối mà mạng nơ-ron (neural network, với tham số học được θ — "theta") dự đoán ra, nhằm mục đích khử một phần nhiễu, đi từ bước t về bước t−1.
- **μ_θ(αₜ,t)**: giá trị trung bình (mean) do mạng nơ-ron dự đoán — tức là "phiên bản đỡ nhiễu hơn" của αₜ.
- **Σ_θ(αₜ,t)**: phương sai (variance) — độ chắc chắn của dự đoán; trong paper này được đơn giản hóa thành một hằng số cố định để dễ tính toán.
- **Cách hoạt động:** mạng nơ-ron được huấn luyện (training) sao cho, khi đưa vào một bản đồ thị đã bị nhiễu ở bước t (αₜ), nó sẽ dự đoán ra bản đỡ nhiễu hơn ở bước t−1. Lặp lại quá trình này T lần liên tiếp (từ bước T về bước 0) sẽ phục hồi lại được đồ thị gần giống với đồ thị gốc α₀.
- Mô hình thực hiện việc dự đoán này thông qua một **MLP (Multi-Layer Perceptron — mạng nơ-ron nhiều lớp)**: nó nhận đầu vào là αₜ và một "embedding của bước t" (giúp mạng biết mình đang ở bước nhiễu nào), rồi xuất ra dự đoán của α₀ (ký hiệu **α̂_θ** — "alpha mũ theta", nghĩa là "α₀ mà mạng θ dự đoán").

**Công thức (5) — ELBO (Evidence Lower Bound — Cận dưới bằng chứng):**

L_elbo = tổng các L_t từ t = 0 đến T

*Giải nghĩa:* việc huấn luyện lý tưởng là tối ưu trực tiếp "xác suất mô hình tái tạo đúng dữ liệu gốc", nhưng công thức đó quá phức tạp để tính trực tiếp. Toán học chứng minh rằng ta có thể tối ưu một **cận dưới** của nó (gọi là ELBO) thay thế — tối ưu ELBO cũng gián tiếp làm tăng khả năng tái tạo đúng. L_elbo là tổng của nhiều thành phần nhỏ L_t (mỗi t một thành phần).

**Công thức (6):** L_t được chia làm 3 trường hợp tùy giá trị t — nhưng bản chất, mỗi L_t đo "mạng nơ-ron dự đoán sai bao nhiêu" tại bước t, dùng **KL divergence** (độ phân kỳ KL — thước đo "2 phân phối xác suất khác nhau bao nhiêu") giữa phân phối "đúng" (tính được từ dữ liệu thật, ký hiệu q) và phân phối mạng dự đoán (ký hiệu p_θ).

**Công thức (7)-(11):** Đây là các bước biến đổi toán học (dùng quy tắc Bayes) để rút gọn công thức KL divergence phức tạp ở trên thành một dạng đơn giản hơn nhiều: chỉ còn là khoảng cách **MSE (Mean Squared Error — sai số bình phương trung bình)** giữa giá trị α₀ thật và giá trị α̂_θ mà mạng dự đoán, có nhân thêm một hệ số trọng số phụ thuộc bước t. Bạn không cần nhớ từng bước biến đổi — điều quan trọng cần nắm là:

> **Kết luận thực dụng:** toàn bộ quá trình huấn luyện khử nhiễu, sau khi rút gọn toán học, chỉ còn lại một việc rất trực quan: **cho mạng nơ-ron nhìn đồ thị đã bị nhiễu ở bước t ngẫu nhiên, bắt nó đoán lại đồ thị gốc α₀, rồi so sánh (bằng MSE) giữa dự đoán và giá trị thật để chỉnh sửa tham số mạng.**

**Công thức (12):** L₀ = ‖α̂_θ(α₁,1) − α₀‖² — trường hợp riêng khi t=0 (bước nhiễu nhẹ nhất), cũng chỉ là MSE.

**Công thức (13):** L_elbo (dạng thực dụng để code) = kỳ vọng (lấy trung bình khi t được chọn ngẫu nhiên đều trong khoảng 1 đến T, và α₀ được lấy ngẫu nhiên từ dữ liệu) của ‖α̂_θ(αₜ,t) − α₀‖² — tức là công thức (11) nhưng bỏ hệ số trọng số đi cho đơn giản, chỉ giữ lại phần MSE, và **mỗi lần huấn luyện chỉ cần lấy mẫu (sample) một bước t ngẫu nhiên** thay vì tính hết cả T bước — giúp tiết kiệm tính toán rất nhiều.

**Hình dung dễ nhất:** mạng khử nhiễu giống một "họa sĩ phục chế tranh cổ" — được đưa cho một bức tranh bị mờ/nhiễu ở một mức độ ngẫu nhiên nào đó, và bài tập của họa sĩ là đoán lại bức tranh gốc trông như thế nào; càng luyện tập nhiều, họa sĩ càng đoán chuẩn.

### Giai đoạn 3 — Modality-aware Signal Injection — MSI (Cơ chế "tiêm" tín hiệu nhận biết modal)

Đây là điểm **đặc trưng và quan trọng nhất** của DiffMM: nếu chỉ huấn luyện diffusion model như Giai đoạn 2, mạng chỉ học tái tạo lại đúng đồ thị tương tác — chưa hề "biết" gì về nội dung ảnh/chữ/âm thanh của item. MSI là cơ chế để "tiêm" thông tin modal vào, buộc mô hình sinh ra đồ thị phản ánh đúng đặc trưng modal.

**Công thức (14):** L_msi^m = ‖ α̂₀ · eᵢᵐ  −  α₀ · eᵢ ‖²

*Giải nghĩa từng phần:*
- **eᵢᵐ**: embedding (vector số) đặc trưng modal m của item i, sau khi đã được căn chỉnh về cùng kích thước với các embedding khác (chi tiết ở công thức 15, Giai đoạn 5).
- **eᵢ**: embedding "id" gốc của item i (embedding học từ chính hệ thống gợi ý, không liên quan modal, giống như một "mã định danh học được" của item).
- **α̂₀ · eᵢᵐ**: lấy đồ thị mà mạng khử nhiễu vừa dự đoán ra (α̂₀), nhân với embedding modal của item → cho ra một embedding "user nhìn theo góc độ modal" (nếu đồ thị dự đoán đúng, user sẽ có embedding gần với các item có đặc trưng modal tương tự).
- **α₀ · eᵢ**: lấy đồ thị tương tác **thật** (quan sát được), nhân với embedding id gốc của item → embedding "user nhìn theo góc độ tương tác thực tế đã biết" (collaborative — tương tác cộng đồng).
- **‖ ... ‖²**: chuẩn bình phương (bình phương độ dài vector hiệu số) — chính là công thức MSE, đo khoảng cách giữa 2 embedding trên.
- **Ý nghĩa tổng thể:** ép buộc "embedding user tính theo đồ thị mà mô hình mới sinh ra + đặc trưng modal" phải giống với "embedding user tính theo đồ thị tương tác thật + đặc trưng id gốc". Nhờ vậy, đồ thị mà mô hình sinh ra (α̂₀) không chỉ đúng về mặt thống kê tương tác mà còn phải "hợp lý" về mặt nội dung modal.

**Công thức (24):** L_dm^m (Loss của Diffusion Model cho modal m) = L_elbo + λ₀ · L_msi^m

*Giải nghĩa:* tổng 2 phần loss lại, có một **hyperparameter (siêu tham số)** λ₀ (lambda-không) để cân bằng: λ₀ càng lớn thì mô hình càng ưu tiên khớp theo modal, càng nhỏ thì ưu tiên khớp theo đúng công thức khử nhiễu chuẩn.

**Hình dung dễ nhất:** MSI giống như việc "ghim" quá trình phục hồi tranh vào một bộ ảnh tham chiếu về màu sắc/chủ đề (đặc trưng modal), để bức tranh phục hồi ra không chỉ đúng bố cục mà còn đúng "màu sắc/phong cách" đặc trưng — ở đây "phong cách" chính là thông tin ảnh/chữ/âm thanh của sản phẩm.

### Giai đoạn 4 — Inference (Suy luận) của Diffusion Model: tạo ra đồ thị A^m

**Inference** (giai đoạn dùng mô hình đã huấn luyện xong để tạo ra kết quả thực tế, không học thêm) ở đây cần một mẹo riêng, vì lúc này ta không có sẵn nhiễu Gauss "đúng chuẩn" để bắt đầu như lúc huấn luyện:

1. Lấy đồ thị tương tác thật α₀, cho nó đi qua **forward process** (Giai đoạn 1) nhưng chỉ T' bước (T' nhỏ hơn T tổng, tức là chỉ làm nhiễu **một phần**, không làm nhiễu hoàn toàn) → được αT'.
2. Từ αT', chạy **reverse process** (Giai đoạn 2) nhưng theo kiểu **deterministic** (tất định — nghĩa là bỏ phần ngẫu nhiên đi, chỉ dùng giá trị trung bình μ_θ dự đoán được, không cộng thêm nhiễu ngẫu nhiên nữa) qua T bước để tính ra bản phục hồi cuối cùng **α̂₀**.
3. Với mỗi user u, trong vector α̂₀ vừa tính được (là một dãy số thể hiện "xác suất/độ phù hợp" giữa user u và từng item), chọn ra **top-k giá trị lớn nhất** (k số lớn nhất) → xem đó là k liên kết (edge) mới giữa user u và các item tương ứng.
4. Gộp toàn bộ các liên kết mới của tất cả user lại → được đồ thị **A^m** — đồ thị "nhận biết modal" (**modality-aware**) dành riêng cho modal m.

Vì có 3 modal (visual, textual, acoustic), bước này được lặp lại 3 lần (mỗi modal huấn luyện một mô hình khuếch tán riêng), cho ra 3 đồ thị: A^visual, A^textual, A^acoustic.

**Hình dung dễ nhất:** giống như hỏi "nếu chỉ nhìn vào ảnh sản phẩm (bỏ qua mọi thứ khác), thì user này có khả năng thích những item nào nhất?" — câu trả lời chính là đồ thị A^visual.

### Giai đoạn 5 — Cross-Modal Contrastive Augmentation (Tăng cường đối chiếu xuyên-modal)

Dùng các đồ thị A^m vừa sinh ra ở Giai đoạn 4 để tạo tín hiệu học tự giám sát (self-supervised).

**Công thức (15):** eᵐ = Norm( Trans(F̂ᵐ) )

*Giải nghĩa:* **F̂ᵐ** là đặc trưng modal thô ban đầu (có thể có kích thước khác nhau tùy modal). **Trans(·)** là một phép biến đổi (thực hiện bởi MLP) để đưa mọi đặc trưng modal về **cùng một kích thước d** (dễ so sánh, cộng, nhân với nhau). **Norm(·)** là chuẩn hóa (normalization) — đưa giá trị về một thang đo chuẩn, tránh modal này có giá trị quá lớn/nhỏ so với modal khác.

**Công thức (16):** zᵐᵤ = Ã^m_{u,*} · Eᵘ  ;   zᵐᵢ = Ã^m_{*,i} · Eᵐ ,  Ã^m_{u,i} = A^m_{u,i} / √(|N^m_u| · |N^m_i|)

*Giải nghĩa:*
- **Ã^m**: là **Ã** — phiên bản "chuẩn hóa" của đồ thị A^m (chuẩn hóa theo số lượng hàng xóm — **N^m_u**, **N^m_i** là tập hàng xóm (neighbor) của user u / item i trong đồ thị A^m — để các node có nhiều kết nối không bị "lấn át" các node có ít kết nối).
- **Eᵘ, Eᵐ**: ma trận embedding user và embedding modal của item.
- **Ý nghĩa:** đây chính là một bước **GNN aggregation (tổng hợp GNN)** — mỗi user/item lấy trung bình có trọng số embedding của các "hàng xóm" trong đồ thị A^m để tạo ra embedding mới zᵐ (embedding "theo góc nhìn modal m").

**Công thức (17):** lan truyền tiếp qua đồ thị gốc Ã (đồ thị tương tác thật đã chuẩn hóa), qua nhiều lớp (layer), rồi cộng dồn (sum-pooling) các lớp lại → cho ra **ẑᵐ** (embedding modal cuối cùng, đã "ngấm" thêm thông tin tương tác thật bậc cao).

**Công thức (18) — InfoNCE loss, kiểu "modality view làm anchor":**

L_cl^user = tổng qua các cặp modal (m1, m2) và các user u của:  −log [ exp(sim(ẑᵐ¹ᵤ, ẑᵐ²ᵤ)/τ) / tổng theo mọi v exp(sim(ẑᵐ¹ᵤ, ẑᵐ²ᵥ)/τ) ]

*Giải nghĩa:*
- **sim(·,·)**: hàm đo độ giống nhau (similarity — thường là tích vô hướng hoặc cosine) giữa 2 embedding.
- **τ (tau)**: hệ số nhiệt độ (temperature) — một siêu tham số điều chỉnh "độ gắt" của phân biệt giống/khác (τ nhỏ → phân biệt gắt hơn).
- **exp(·)**: hàm mũ e — dùng để biến điểm số similarity thành giá trị dương, thuận tiện tính tỉ lệ xác suất.
- **Tử số** (phần trên): độ giống nhau giữa embedding của **cùng một user u** nhưng nhìn từ 2 modal khác nhau (m1 và m2) — đây là **positive pair (cặp dương)**, ta muốn 2 giá trị này càng giống nhau càng tốt.
- **Mẫu số** (phần dưới): tổng độ giống nhau giữa u (theo modal m1) với **tất cả user v khác** (theo modal m2) — bao gồm cả positive lẫn **negative pairs (cặp âm)**.
- **Toàn bộ công thức** (dạng −log của một phân số): đây là công thức InfoNCE loss chuẩn — càng làm cho tử số lớn hơn (so với mẫu số) thì loss càng nhỏ, nghĩa là mô hình được thưởng khi kéo cặp dương lại gần và phạt khi cặp âm ở gần.
- **Hình dung:** giống bài kiểm tra trắc nghiệm — cho embedding của user u theo modal "ảnh", mô hình phải "nhận ra" đúng embedding của chính user u đó theo modal "chữ" trong một rổ gồm nhiều user khác — càng nhận đúng, loss càng thấp.

**Công thức (19) — InfoNCE loss, kiểu "main view làm anchor":**

L_cl^user (bản 2) = tổng qua modal m và user u của: −log [ exp(sim(H̃ᵤ, ẑᵐᵤ)/τ) / tổng theo mọi v exp(sim(H̃ᵤ, ẑᵐᵥ)/τ) ]

*Giải nghĩa:* tương tự công thức (18), nhưng lần này **anchor (điểm neo)** là embedding H̃ᵤ từ **nhánh dự đoán chính** (Giai đoạn 6) thay vì một modal khác — nghĩa là buộc nhánh chính phải "học hỏi" và nhất quán với từng view modal riêng lẻ.

- Tính tương tự cho phía item, ra **L_cl^item**.
- Tổng hợp: **L_cl = L_cl^user + L_cl^item**

**Hình dung dễ nhất:** tưởng tượng mỗi user có nhiều "chân dung" khác nhau — chân dung theo sở thích hình ảnh, chân dung theo sở thích văn bản, chân dung theo sở thích âm thanh. Contrastive learning dạy mô hình rằng dù nhìn qua "chân dung" nào, đó vẫn phải là cùng một người — tức các embedding phải nhất quán với nhau.

### Giai đoạn 6 — Multi-Modal Graph Aggregation (nhánh dự đoán chính)

**Công thức (20):** ẑᵐᵤ = Ã_{u,*}·Eᵘ + Ã_{u,*}·(Ã_{u,*}·Eᵘ) + Ãᵐ_{u,*}·Eᵘ  (và tương tự cho item)

*Giải nghĩa:* đây là bước **GNN aggregation** trên **đồ thị gốc Ã** (không phải A^m sinh ra từ diffusion), kết hợp thêm một chút thông tin từ Ãᵐ. Về bản chất: mỗi user/item lấy trung bình embedding của hàng xóm (bậc 1) và hàng xóm của hàng xóm (bậc 2, thể hiện qua việc nhân Ã hai lần liên tiếp) — giống cách hoạt động chuẩn của LightGCN (một mô hình GNN nổi tiếng cho gợi ý), cộng thêm ảnh hưởng nhẹ của đồ thị modal.

**Công thức (21):** hᵤ = Σ_m κ_m · ẑᵐᵤ  (và tương tự hᵢ)

*Giải nghĩa:* **κ_m (kappa-m)** là trọng số **học được** (learnable) riêng cho mỗi modal — cho phép mô hình tự quyết định "modal nào quan trọng hơn" khi gộp (aggregate) embedding của 3 modal lại thành một embedding tổng hᵤ, hᵢ.

**Công thức (22):** H_{l+1} = Ã · H_l ,  H₀ = hᵤ hoặc hᵢ

*Giải nghĩa:* tiếp tục lan truyền qua **L lớp (layer) GNN** trên đồ thị tương tác gốc Ã — mỗi lớp l+1 được tính bằng cách nhân đồ thị Ã với embedding của lớp l ngay trước đó, giúp mô hình "nhìn xa hơn" (thu thập thông tin từ hàng xóm bậc cao, tức là hàng xóm của hàng xóm của hàng xóm...).

**Công thức (23):** H̃ = Σ (từ lớp 0 đến L) H_l  +  ω · Norm(H₀)

*Giải nghĩa:* cộng dồn (**sum-pooling**) embedding của tất cả các lớp GNN lại, cộng thêm một phần embedding gốc H₀ đã chuẩn hóa, với trọng số **ω (omega)** — kỹ thuật này giúp chống lại **over-smoothing (làm mượt quá mức)**, tức là hiện tượng khi truyền qua quá nhiều lớp GNN, mọi embedding trở nên gần giống hệt nhau, làm mất khả năng phân biệt user/item. **H̃** chính là embedding cuối cùng, dùng để dự đoán.

**Công thức không đánh số — dự đoán cuối cùng:** ŷ_ui = h̃ᵤᵀ · h̃ᵢ

*Giải nghĩa:* điểm số dự đoán mức độ user u sẽ thích item i, tính bằng **tích vô hướng (dot product)** giữa 2 vector embedding cuối cùng của u và i — điểm càng cao thì mô hình càng "tự tin" là user u sẽ thích item i. Danh sách gợi ý cuối cùng = xếp hạng (ranking) các item theo điểm số này, lấy **top-K** (K item cao điểm nhất).

### Giai đoạn 7 — Huấn luyện đa nhiệm (Multi-Task Model Training)

Có 2 nhóm loss được huấn luyện song song, bổ trợ nhau:

| Thành phần | Công thức | Ý nghĩa |
|---|---|---|
| Diffusion module (mỗi modal) | L_dm^m = L_elbo + λ₀·L_msi^m | Dạy mô hình khuếch tán sinh đúng đồ thị, có "ghim" theo modal |
| Recommendation task (tác vụ gợi ý chính) | L_rec = L_bpr + λ₁·L_cl + λ₂·‖Θ‖² | Dạy nhánh chính dự đoán đúng, có thêm hỗ trợ từ contrastive learning |

**Công thức (25) — L_bpr (BPR — Bayesian Personalized Ranking):**

L_bpr = tổng qua các bộ ba (u,i,j) trong tập O của: −log σ(ŷ_ui − ŷ_uj)

*Giải nghĩa:*
- **O = {(u,i,j) | (u,i) ∈ O⁺, (u,j) ∈ O⁻}**: **O⁺** là tập các cặp user-item **đã** tương tác thật (positive); **O⁻** là tập cặp **chưa** tương tác (negative, được lấy mẫu ngẫu nhiên từ những item user chưa từng chạm tới).
- **σ(·)**: hàm sigmoid — biến một số thực bất kỳ thành giá trị trong khoảng (0,1), giống như "xác suất".
- **ŷ_ui − ŷ_uj**: hiệu số điểm dự đoán giữa item i (mà user thực sự thích) và item j (item ngẫu nhiên, chưa tương tác).
- **Ý nghĩa toàn bộ:** loss này buộc mô hình phải cho điểm ŷ_ui (item đã thích) **cao hơn** điểm ŷ_uj (item chưa từng chạm tới) — nếu đúng như vậy, hiệu số dương lớn, σ tiến gần 1, −log(gần 1) tiến gần 0 (loss nhỏ, tốt); nếu sai (điểm ngược lại), loss sẽ lớn (bị phạt).

**Công thức (26):** L_rec = L_bpr + λ₁·L_cl + λ₂·‖Θ‖²₂

*Giải nghĩa:* cộng thêm 2 phần: **λ₁·L_cl** (phần contrastive learning ở Giai đoạn 5, với trọng số λ₁ điều chỉnh mức độ quan trọng) và **λ₂·‖Θ‖²₂** — đây là **L2 regularization (điều chuẩn L2)**: phạt các tham số mô hình (**Θ — theta hoa**, đại diện cho toàn bộ tham số học được) nếu chúng có giá trị quá lớn, giúp tránh **overfitting (học vẹt)**.

**Quy trình huấn luyện tổng thể (hình dung theo vòng lặp):**
1. Huấn luyện diffusion model (Giai đoạn 1-3) cho mỗi modal → càng huấn luyện, mô hình càng sinh đồ thị A^m sát thực tế hơn.
2. Dùng diffusion model đã huấn luyện để suy luận (Giai đoạn 4) ra đồ thị A^m mới nhất.
3. Dùng A^m để tính contrastive loss (Giai đoạn 5) và huấn luyện nhánh dự đoán chính (Giai đoạn 6) bằng L_rec.
4. Lặp lại các bước trên nhiều vòng (epoch — một lượt duyệt qua toàn bộ dữ liệu huấn luyện) cho đến khi mô hình hội tụ (hiệu năng không cải thiện thêm).

---

## 4. Kết quả thực nghiệm (Evaluation)

### Bộ dữ liệu & cách đánh giá
- **Dataset (bộ dữ liệu):** TikTok, Amazon-Baby, Amazon-Sports — đều là dữ liệu thật, công khai, có sẵn đặc trưng modal (ảnh/chữ/[âm thanh riêng TikTok]).
- **Recall@K:** trong K item được gợi ý, có bao nhiêu % item mà user **thực sự thích** (trong tập kiểm tra) đã được "tìm thấy" — đo khả năng "không bỏ sót" item đúng.
- **Precision@K:** trong K item được gợi ý, có bao nhiêu % là item user thực sự thích — đo độ "chính xác" của danh sách gợi ý.
- **NDCG@K (Normalized Discounted Cumulative Gain):** giống Recall/Precision nhưng có tính thêm **thứ tự** — item đúng nằm ở vị trí càng cao (gần đầu danh sách) thì điểm càng cao, phạt nhẹ nếu item đúng nằm ở cuối danh sách.
- Cả 3 chỉ số đều: **càng cao càng tốt**.

### Kết quả chính
- DiffMM vượt qua toàn bộ **baseline (mô hình đối chứng)** — gồm cả mô hình cổ điển (MF-BPR, NGCF, LightGCN), mô hình SSL (SGL, NCL, HCCF), và mô hình đa phương thức hiện đại (VBPR, MMGCN, GRCN, LATTICE, CLCRec, MMGCL, SLMRec, LightGT, BM3) — trên cả 3 dataset.
- Ví dụ trên TikTok: Recall@20 của DiffMM = 0.1129, trong khi baseline tốt nhất (SLMRec) chỉ đạt 0.0957.

### Ablation study (nghiên cứu cắt bỏ) — kiểm tra từng thành phần có thực sự cần thiết không
Thử bỏ lần lượt 3 thành phần và đo hiệu năng giảm bao nhiêu:
- **w/o CL** (without Contrastive Learning — bỏ phần học đối chiếu, Giai đoạn 5): hiệu năng giảm rõ rệt → chứng minh contrastive learning giúp ích thật sự.
- **w/o DM** (without Diffusion Model — thay mô hình khuếch tán bằng **VGAE**, một mô hình sinh dữ liệu khác cũ hơn): hiệu năng giảm **mạnh nhất** trong 3 thử nghiệm → đây là bằng chứng rõ nhất rằng **diffusion model là "linh hồn"** của phương pháp, vượt trội hơn hẳn so với các kỹ thuật sinh dữ liệu khác.
- **w/o MSI** (without Modality-aware Signal Injection — bỏ Giai đoạn 3): hiệu năng cũng giảm → xác nhận MSI thực sự giúp "ghim" đúng thông tin modal vào quá trình sinh đồ thị.

### Khả năng xử lý dữ liệu thưa (RQ3 — Research Question 3)
Chia user thành các nhóm theo số lượng tương tác (từ rất ít đến nhiều), so sánh DiffMM với 4 baseline mạnh (VBPR, LATTICE, SLMRec, BM3). Kết quả: DiffMM vượt trội **rõ rệt nhất** ở nhóm user có **ít tương tác nhất** — đúng trọng tâm mà bài báo muốn giải quyết (vấn đề sparsity/dữ liệu thưa).

### Phân tích hyperparameter (siêu tham số — RQ4)
- **λ₀** (mức độ MSI): mỗi dataset có một giá trị tối ưu riêng, không phải "càng lớn càng tốt".
- **ω** (trọng số chống over-smoothing): giá trị quá nhỏ → mô hình dựa quá nhiều vào thông tin bậc cao (dễ bị mượt hóa quá mức); giá trị quá lớn → bỏ qua thông tin bậc cao, cũng làm giảm hiệu năng. Cần chọn cân bằng.
- **τ, λ₁** (nhiệt độ và trọng số contrastive learning): ảnh hưởng đáng kể, cần tinh chỉnh theo từng dataset.
- **Chọn anchor (điểm neo):** dùng "modality view làm anchor" tốt hơn trên Amazon-Baby/Sports, nhưng "main view làm anchor" lại tốt hơn trên TikTok — không có lựa chọn nào thắng tuyệt đối ở mọi trường hợp.

### So sánh với tăng cường ngẫu nhiên (RQ5)
Trộn đồ thị do diffusion sinh ra với đồ thị tăng cường theo kiểu ngẫu nhiên cũ (random augmentation — ví dụ xóa cạnh ngẫu nhiên) theo các tỉ lệ khác nhau. Kết quả: **càng dùng nhiều phần ngẫu nhiên, hiệu năng càng giảm** → khẳng định chắc chắn đồ thị do diffusion model sinh ra "có ý nghĩa/thông minh" hơn hẳn so với việc tạo dữ liệu ngẫu nhiên đơn thuần.

---

## 5. Điểm mạnh, hạn chế và hướng phát triển

**Điểm mạnh:**
- Thay kỹ thuật tăng cường dữ liệu (data augmentation) ngẫu nhiên, dễ gây nhiễu, bằng một quá trình **sinh dữ liệu có điều kiện theo modal** — vừa giảm nhiễu vừa tăng liên kết có ý nghĩa ngữ nghĩa.
- Cơ chế **MSI** đơn giản (chỉ thêm 1 loss MSE) nhưng hiệu quả rõ rệt (theo ablation) để gắn thông tin modal vào quá trình khuếch tán.
- Kết quả nhất quán vượt trội trên nhiều dataset, đặc biệt hiệu quả với user có ít dữ liệu (giải quyết đúng vấn đề sparsity).

**Hạn chế (suy ra từ thiết kế và ablation):**
- Cần huấn luyện **riêng một mô hình khuếch tán cho mỗi modal** → tốn thêm chi phí tính toán so với các mô hình GNN thông thường.
- Có khá nhiều **hyperparameter** cần tinh chỉnh (τ, λ₀, λ₁, ω, loại anchor, loại alignment/căn chỉnh modal) và giá trị tối ưu **khác nhau tùy dataset** — nghĩa là cần thử nghiệm nhiều để áp dụng vào dữ liệu mới.

**Hướng phát triển (paper đề xuất):** tích hợp **LLM (Large Language Model — mô hình ngôn ngữ lớn)** để "dẫn dắt" quá trình khuếch tán bằng khả năng hiểu ngữ nghĩa mạnh hơn, kỳ vọng tạo ra tăng cường dữ liệu thông minh hơn nữa.
