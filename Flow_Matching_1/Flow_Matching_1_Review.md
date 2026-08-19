# Flow Matching for Generative Modeling
### (Đối khớp Dòng chảy cho Mô hình sinh dữ liệu)

**Nguồn:** Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le — Meta AI (FAIR) & Weizmann Institute of Science. Preprint arXiv:2210.02747 (bài này sau đó được chấp nhận tại ICLR 2023 — một hội nghị lớn về học máy — dưới dạng spotlight paper, tức là bài được đánh giá rất cao).

> Toàn bộ thuật ngữ tiếng Anh được giải thích ngay bằng tiếng Việt trong ngoặc **tại chỗ xuất hiện**, lặp lại xuyên suốt bài. Mọi công thức toán đều được diễn giải ý nghĩa từng ký hiệu bằng lời, kèm ví von thực tế để dễ hình dung.

---

## 0. Bảng thuật ngữ nền (đọc trước để dễ theo dõi)

| Thuật ngữ tiếng Anh | Giải thích tiếng Việt |
|---|---|
| **Generative model / Generative modeling** | Mô hình sinh dữ liệu — mô hình học cách "vẽ ra" dữ liệu mới (ảnh, âm thanh...) trông giống dữ liệu thật |
| **Data distribution** | Phân phối dữ liệu — quy luật thống kê mô tả dữ liệu thật trông như thế nào (ví dụ: phân phối của "mọi bức ảnh mèo có thể có") |
| **Data space** | Không gian dữ liệu — tập hợp tất cả các điểm dữ liệu có thể có, ký hiệu ℝᵈ (không gian d chiều số thực) |
| **Probability density function** | Hàm mật độ xác suất — hàm số cho biết "khả năng" một điểm dữ liệu xuất hiện cao hay thấp |
| **Probability (density) path** | Đường đi xác suất — một chuỗi các phân phối xác suất thay đổi dần theo thời gian t (từ t=0 đến t=1), giống một "đoạn phim" biến hình từ phân phối này sang phân phối khác |
| **Vector field** | Trường véc-tơ — một hàm số gán cho mỗi điểm trong không gian một mũi tên (véc-tơ) chỉ hướng và tốc độ di chuyển tại điểm đó |
| **Time-dependent** | Phụ thuộc thời gian — giá trị hàm số thay đổi theo biến thời gian t |
| **ODE (Ordinary Differential Equation)** | Phương trình vi phân thường — phương trình toán mô tả "vật thể di chuyển như thế nào" dựa trên vận tốc tức thời của nó tại từng thời điểm |
| **Flow** | Dòng chảy — quỹ đạo chuyển động của một điểm dữ liệu theo thời gian, được xác định bằng cách "đi theo" trường véc-tơ (giống một chiếc lá trôi theo dòng nước) |
| **Diffeomorphic map** | Ánh xạ vi phôi — một phép biến đổi "mượt" và có thể đảo ngược được (không xé rách, không gấp không gian) |
| **CNF (Continuous Normalizing Flow)** | Dòng chuẩn hóa liên tục — mô hình dùng một trường véc-tơ do mạng nơ-ron học được để biến đổi liên tục một phân phối đơn giản thành phân phối phức tạp |
| **Neural network** | Mạng nơ-ron — mô hình máy học nhiều lớp, ở đây dùng để biểu diễn trường véc-tơ |
| **Learnable parameters (θ)** | Tham số học được — các con số bên trong mạng nơ-ron được điều chỉnh dần trong quá trình huấn luyện |
| **Prior density (p₀)** | Phân phối "tiền nghiệm"/phân phối khởi đầu — phân phối đơn giản ban đầu, thường là nhiễu ngẫu nhiên thuần túy (ví dụ phân phối Gauss chuẩn) |
| **Push-forward** | Phép đẩy tới — cách một phép biến đổi (ánh xạ) làm thay đổi một phân phối xác suất thành một phân phối khác |
| **Change of variables** | Đổi biến — kỹ thuật toán học tính lại mật độ xác suất khi ta biến đổi biến số qua một hàm số khác |
| **Continuity equation** | Phương trình liên tục — phương trình toán (thường dùng trong vật lý) mô tả mối liên hệ bắt buộc giữa một trường véc-tơ và đường đi xác suất mà nó tạo ra |
| **Diffusion model** | Mô hình khuếch tán — họ mô hình sinh dữ liệu hoạt động bằng cách thêm nhiễu dần vào dữ liệu rồi học cách khử nhiễu ngược lại (xem thêm ở bài DiffMM_Review.md) |
| **Denoising score matching** | Đối khớp điểm số kiểu khử nhiễu — kỹ thuật huấn luyện phổ biến của mô hình khuếch tán, học một hàm "điểm số" (score — hướng đi để tăng khả năng xảy ra của dữ liệu) |
| **Score function** | Hàm điểm số — građient (đạo hàm) của log mật độ xác suất, chỉ ra "hướng nào làm dữ liệu trông thật hơn" |
| **Simulation-free (training)** | Huấn luyện không cần mô phỏng — kiểu huấn luyện KHÔNG cần giải phương trình ODE nhiều bước trong lúc học (khác với training kiểu cũ phải mô phỏng cả quá trình mới tính được loss) — nhanh và rẻ hơn nhiều |
| **Maximum likelihood (training)** | Huấn luyện theo hợp lý cực đại — cách huấn luyện cổ điển của CNF, đòi hỏi mô phỏng ODE tốn kém |
| **Intractable** | Không khả thi để tính toán trực tiếp — một biểu thức toán học đúng về mặt lý thuyết nhưng quá phức tạp/tốn kém để tính ra con số cụ thể |
| **Biased gradient** | Građient thiên lệch — khi ước lượng đạo hàm dùng để cập nhật mô hình bị sai lệch có hệ thống, khiến việc huấn luyện không hội tụ đúng |
| **Unbiased estimator** | Ước lượng không thiên lệch — một cách tính gần đúng nhưng đúng "về mặt trung bình", không bị lệch theo hướng nào |
| **Flow Matching (FM)** | Đối khớp dòng chảy — phương pháp chính của bài báo: huấn luyện trường véc-tơ của CNF bằng cách "đối khớp" (matching = so khớp, hồi quy) nó với một trường véc-tơ mục tiêu đã biết trước |
| **Regress / Regression** | Hồi quy — kỹ thuật huấn luyện mô hình để dự đoán ra một giá trị số càng gần giá trị mục tiêu càng tốt (ở đây mục tiêu là một véc-tơ) |
| **Target vector field** | Trường véc-tơ mục tiêu — trường véc-tơ "đúng" mà ta muốn mạng nơ-ron học theo |
| **Conditional (probability path/vector field)** | Có điều kiện — được định nghĩa "ứng với riêng từng mẫu dữ liệu x₁ cụ thể", chứ không phải chung cho toàn bộ tập dữ liệu |
| **Marginal (probability path/vector field)** | Biên (marginal) — kết quả tổng hợp/trung bình hóa qua toàn bộ các mẫu dữ liệu x₁, thu được bằng phép "lấy biên" (marginalizing = cộng dồn/tích phân qua một biến để loại bỏ nó) |
| **Marginalizing** | Lấy biên/biên hóa — thao tác tích phân (hoặc lấy tổng) một biến ra khỏi công thức, để chỉ còn lại các biến quan tâm |
| **Per-sample / Per-example** | Theo từng mẫu — được thiết kế/tính riêng cho mỗi điểm dữ liệu, thay vì tính chung |
| **Mixture (distribution)** | Phân phối hỗn hợp — một phân phối được tạo bằng cách trộn (pha) nhiều phân phối con lại với nhau |
| **CFM (Conditional Flow Matching)** | Đối khớp dòng chảy có điều kiện — phiên bản "thực dụng" của FM, tính loss dựa trên từng mẫu dữ liệu riêng lẻ, khả thi để tính toán (không giống FM gốc) |
| **Theorem** | Định lý — một phát biểu toán học đã được chứng minh chặt chẽ là đúng |
| **Gradient** | Građient — đạo hàm, cho biết hướng và độ lớn cần chỉnh tham số mô hình để giảm loss |
| **Gaussian distribution / Gaussian noise** | Phân phối Gauss / nhiễu Gauss (còn gọi phân phối chuẩn) — dạng phân phối hình chuông rất phổ biến, đặc trưng bởi giá trị trung bình (mean) và độ lệch chuẩn (standard deviation/std) |
| **Mean (μ)** | Giá trị trung bình — điểm "trung tâm" của phân phối Gauss |
| **Standard deviation (std, σ)** | Độ lệch chuẩn — mức độ "phân tán rộng hay hẹp" quanh giá trị trung bình |
| **Affine transformation** | Phép biến đổi affine — phép biến đổi đơn giản dạng "nhân với một số rồi cộng thêm một số" (không bẻ cong không gian) |
| **Boundary condition** | Điều kiện biên — ràng buộc bắt buộc phải thỏa mãn ở 2 đầu mút (ở đây là tại t=0 và t=1) |
| **Divergence-free** | Không phân kỳ — tính chất của một thành phần véc-tơ không làm thay đổi mật độ xác suất tổng thể (chỉ "xoay" chứ không "nén/giãn" xác suất), dùng để loại bỏ các lựa chọn dư thừa khi chọn trường véc-tơ |
| **Canonical (vector field)** | Trường véc-tơ chính tắc — lựa chọn "đơn giản nhất, tự nhiên nhất" trong vô số lựa chọn khả dĩ |
| **VE (Variance Exploding) diffusion path** | Đường khuếch tán kiểu "phương sai bùng nổ" — một kiểu quá trình khuếch tán trong đó độ nhiễu (variance) tăng dần không giới hạn |
| **VP (Variance Preserving) diffusion path** | Đường khuếch tán kiểu "giữ phương sai" — kiểu quá trình khuếch tán mà tổng "năng lượng" (variance) được giữ ổn định trong khi trộn dần dữ liệu gốc với nhiễu |
| **Noise schedule (β)** | Lịch trình nhiễu — hàm số quy định "bơm bao nhiêu nhiễu ở mỗi thời điểm t" trong quá trình khuếch tán |
| **Optimal Transport (OT)** | Vận chuyển tối ưu — lý thuyết toán học tìm cách "di chuyển" một phân phối xác suất sang phân phối khác với "chi phí" (quãng đường di chuyển) nhỏ nhất có thể |
| **Displacement interpolation / OT displacement map** | Phép nội suy dịch chuyển / ánh xạ dịch chuyển OT — cách di chuyển từng điểm dữ liệu theo đường thẳng, tốc độ đều, từ phân phối đầu đến phân phối cuối, sao cho tổng quãng đường di chuyển là nhỏ nhất |
| **Straight-line trajectory** | Quỹ đạo đường thẳng — đường đi của một điểm dữ liệu là một đường thẳng (không cong, không lượn) |
| **Overshoot** | Đi quá đà — quỹ đạo đi vượt qua điểm đích rồi phải quay lại (do đường đi bị cong), gây lãng phí thời gian/công tính toán |
| **NLL (Negative Log-Likelihood)** | Hợp lý âm log — chỉ số đo mức độ mô hình "giải thích tốt" dữ liệu thật; càng thấp càng tốt |
| **BPD (Bits Per Dimension)** | Số bit trên mỗi chiều dữ liệu — đơn vị đo NLL, thường dùng cho ảnh; càng thấp càng tốt |
| **FID (Fréchet Inception Distance)** | Khoảng cách Fréchet-Inception — chỉ số đo "ảnh do mô hình tạo ra trông giống ảnh thật đến mức nào" (so sánh 2 phân phối đặc trưng ảnh); càng thấp càng tốt |
| **IS (Inception Score)** | Điểm số Inception — chỉ số khác đo chất lượng và độ đa dạng của ảnh sinh ra; càng cao càng tốt |
| **PSNR (Peak Signal-to-Noise Ratio)** | Tỉ lệ tín hiệu đỉnh trên nhiễu — chỉ số đo độ giống nhau giữa ảnh tạo ra và ảnh gốc theo từng điểm ảnh; càng cao càng tốt |
| **SSIM (Structural Similarity Index)** | Chỉ số tương đồng cấu trúc — chỉ số đo độ giống nhau về cấu trúc/hình dạng giữa 2 ảnh; càng cao càng tốt |
| **NFE (Number of Function Evaluations)** | Số lần gọi hàm — số lần bộ giải ODE phải "hỏi" mạng nơ-ron trong lúc tạo một mẫu; NFE càng thấp thì sinh mẫu càng nhanh/rẻ |
| **ODE solver** | Bộ giải phương trình vi phân — thuật toán số học để "đi theo" trường véc-tơ và tính ra điểm đích cuối cùng, ví dụ: Euler, Midpoint, RK4 (Runge-Kutta bậc 4), dopri5 (bộ giải thích ứng — tự động điều chỉnh độ chính xác) |
| **Adaptive solver** | Bộ giải thích ứng — tự động điều chỉnh số bước tính toán để đạt độ chính xác mong muốn |
| **Fixed-step solver** | Bộ giải bước cố định — dùng số bước tính toán cố định, đơn giản và nhanh hơn nhưng kém chính xác hơn bộ giải thích ứng |
| **U-Net** | Kiến trúc mạng nơ-ron hình chữ U — kiến trúc phổ biến dùng làm "bộ não" xử lý ảnh trong các mô hình sinh ảnh (diffusion, FM...) |
| **Ablation** | Cắt bỏ/so sánh thành phần — thử thay đổi từng phần của phương pháp (ví dụ đổi loss, đổi đường đi xác suất) để xem phần nào thực sự đóng góp vào kết quả |
| **Unconditional generation** | Sinh không điều kiện — mô hình tự tạo ra mẫu mới mà không dựa vào bất kỳ thông tin đầu vào nào khác (ví dụ không cho biết trước nhãn/ảnh gốc) |
| **Conditional generation** | Sinh có điều kiện — mô hình tạo ra mẫu mới dựa trên một thông tin đầu vào cho trước (ví dụ ảnh độ phân giải thấp, để "vẽ" ra ảnh độ phân giải cao) |
| **Super-resolution** | Siêu phân giải — tác vụ "phóng to" ảnh có độ phân giải thấp thành ảnh có độ phân giải cao và rõ nét hơn |
| **Reference (baseline trong bảng kết quả)** | Ảnh tham chiếu/dữ liệu gốc — dùng làm chuẩn so sánh |

---

## 1. Bối cảnh & động lực nghiên cứu

**Bức tranh chung:** Các mô hình sinh dữ liệu (**generative models**) hiện đại nhất cho ảnh (như DALL-E, Stable Diffusion) chủ yếu dựa trên **diffusion model** (mô hình khuếch tán). Diffusion model huấn luyện tương đối ổn định và có thể mở rộng (scale) lên dữ liệu lớn, nhưng nó bị **giới hạn trong một không gian hẹp các "đường đi xác suất" (probability paths)** — vì cách xây dựng quá trình khuếch tán (thêm nhiễu Gauss dần dần theo một công thức cố định) chỉ cho ra một số dạng đường đi nhất định. Hệ quả: thời gian huấn luyện lâu, và lúc sinh mẫu (sampling) cần rất nhiều bước tính toán, đòi hỏi các phương pháp giải đặc biệt (specialized methods) để tăng tốc.

**Continuous Normalizing Flows (CNF — Dòng chuẩn hóa liên tục)** là một khung lý thuyết **tổng quát hơn nhiều**: nó có thể biểu diễn *bất kỳ* đường đi xác suất nào (kể cả các đường đi mà diffusion model tạo ra, coi như một trường hợp riêng). Vấn đề là: trước bài báo này, **chưa có cách huấn luyện CNF nào vừa hiệu quả vừa mở rộng được quy mô lớn**:
- Cách huấn luyện cổ điển (**maximum likelihood** — hợp lý cực đại) đòi hỏi **mô phỏng (simulate)** cả quá trình giải ODE (phương trình vi phân) nhiều bước ngay trong lúc huấn luyện → rất tốn kém, chậm.
- Một số cách huấn luyện "không cần mô phỏng" (**simulation-free**) từng được đề xuất, nhưng hoặc dính phải các phép tích phân **không khả thi để tính** (**intractable**), hoặc cho ra **građient thiên lệch** (**biased gradient**) khiến việc học không chính xác.

**Câu hỏi bài báo đặt ra:** Làm sao huấn luyện CNF theo kiểu **simulation-free** (không cần mô phỏng ODE lúc huấn luyện, giống diffusion model) nhưng vẫn giữ được **sự tổng quát** của CNF (không bị bó buộc phải là một quá trình khuếch tán)?

**Câu trả lời — Flow Matching (FM):** một mục tiêu huấn luyện (training objective) đơn giản, chỉ là một phép **hồi quy (regress)** trường véc-tơ của mạng nơ-ron về một trường véc-tơ mục tiêu đã biết trước — không cần giải ODE, không cần biết hàm mật độ xác suất tường minh. Điều đặc biệt: bài báo chứng minh có thể xây dựng mục tiêu này **theo từng mẫu dữ liệu riêng lẻ (per-sample/conditional)**, rồi việc tổng hợp (marginalize) các mục tiêu riêng lẻ đó lại **cho ra đúng kết quả tổng thể mong muốn** — đây là phát hiện cốt lõi giúp bài toán từ "không tính được" trở thành "tính được dễ dàng".

Kết quả phụ đặc biệt thú vị: framework FM cho phép dùng đường đi xác suất kiểu **Optimal Transport (OT — vận chuyển tối ưu)** — đường đi **thẳng** (thay vì cong như diffusion) → huấn luyện nhanh hơn, sinh mẫu nhanh hơn, và tổng quát hóa (generalization) tốt hơn.

---

## 2. Kiến trúc / quy trình tổng thể

Khác với các mô hình như DiffMM (có nhiều khối/module), Flow Matching về bản chất là **một mục tiêu huấn luyện (training objective)** cho một mạng nơ-ron duy nhất đóng vai trò trường véc-tơ. Có thể hình dung quy trình theo 2 giai đoạn lớn: **Huấn luyện (Training)** và **Suy luận/Sinh mẫu (Inference/Sampling)**.

```
                         ==== GIAI ĐOẠN HUẤN LUYỆN (Training) ====

  Dữ liệu thật x₁ ~ q(x₁)          Nhiễu chuẩn x₀ ~ N(0,I)
          │                                  │
          └───────────────┬──────────────────┘
                           ▼
        Xây "đường đi có điều kiện" (conditional probability path)
        pₜ(x|x₁): với mỗi mẫu x₁, vẽ một đường co dần từ nhiễu (t=0)
        về gần x₁ (t=1)  — ví dụ: đường thẳng OT, hoặc đường cong diffusion
                           │
                           ▼
        Lấy 1 điểm x trên đường đi đó tại thời điểm t ngẫu nhiên
        + tính trường véc-tơ mục tiêu uₜ(x|x₁) (công thức đóng, biết trước)
                           │
                           ▼
        Mạng nơ-ron vₜ(x;θ) dự đoán trường véc-tơ tại điểm (x,t)
                           │
                           ▼
        So sánh vₜ(x;θ) với mục tiêu uₜ(x|x₁) bằng MSE (CFM loss)
        → cập nhật tham số θ để 2 giá trị này càng gần nhau càng tốt
                           │
                           ▼
                    Lặp lại hàng triệu lần


                      ==== GIAI ĐOẠN SUY LUẬN (Sampling) ====

     Lấy mẫu nhiễu ngẫu nhiên x₀ ~ N(0,I)  ("hạt bụi" khởi đầu)
                           │
                           ▼
     Dùng bộ giải ODE (Euler/Midpoint/RK4/dopri5) "đi theo" trường
     véc-tơ vₜ(x;θ) đã học, từ t=0 đến t=1
                           │
                           ▼
     Kết quả cuối cùng x₁ = φ₁(x₀)  →  đây chính là dữ liệu (ảnh) mới được sinh ra
```

---

## 3. Quy trình chi tiết — hình dung từng bước

### Giai đoạn 0 — Kiến thức nền: Continuous Normalizing Flow (CNF)

- Không gian dữ liệu (**data space**) ký hiệu ℝᵈ (mỗi điểm dữ liệu x là một vector d chiều — ví dụ ảnh 32×32×3 pixel thì d ≈ 3072).
- **Probability density path** (đường đi xác suất) pₜ: [0,1] × ℝᵈ → ℝ>0 — nghĩa là tại mỗi thời điểm t (từ 0 đến 1), ta có một phân phối xác suất pₜ khác nhau trên không gian dữ liệu; yêu cầu tích phân của pₜ trên toàn không gian luôn bằng 1 (∫pₜ(x)dx = 1, đúng định nghĩa một phân phối xác suất hợp lệ).
- **Vector field** (trường véc-tơ) vₜ: [0,1] × ℝᵈ → ℝᵈ — tại mỗi điểm x và thời điểm t, hàm này trả về một mũi tên (vận tốc) cho biết điểm đó nên di chuyển theo hướng nào, nhanh hay chậm.

**Công thức (1)-(2):** d/dt φₜ(x) = vₜ(φₜ(x)) ,  φ₀(x) = x

*Giải nghĩa:* đây là một **ODE (phương trình vi phân thường)**. **φₜ(x)** (gọi là **flow** — dòng chảy) là vị trí của điểm x sau khi đã "trôi" theo trường véc-tơ vₜ được một khoảng thời gian t. Công thức nói rằng: "vận tốc thay đổi vị trí tại thời điểm t (đạo hàm theo t) chính bằng giá trị trường véc-tơ tại vị trí hiện tại". Điều kiện ban đầu φ₀(x)=x nghĩa là tại t=0, điểm chưa di chuyển gì cả (vị trí đúng bằng điểm gốc).

*Hình dung dễ nhất:* giống thả một chiếc lá (điểm dữ liệu x) xuống một dòng sông có dòng chảy thay đổi theo thời gian và vị trí (trường véc-tơ vₜ) — chiếc lá sẽ trôi theo một quỹ đạo (flow φₜ) nhất định.

- **CNF (Continuous Normalizing Flow):** nếu ta dùng một **mạng nơ-ron** vₜ(x;θ) (với tham số học được θ) để biểu diễn trường véc-tơ, thì flow φₜ tạo ra được gọi là CNF. CNF dùng để biến đổi (reshape) một phân phối đơn giản ban đầu p₀ (ví dụ: nhiễu ngẫu nhiên thuần túy) thành một phân phối phức tạp hơn p₁ (ví dụ: phân phối của ảnh thật).

**Công thức (3)-(4):** pₜ = [φₜ]* p₀ ,  với  [φₜ]* p₀(x) = p₀(φₜ⁻¹(x)) · det[∂φₜ⁻¹(x)/∂x]

*Giải nghĩa:* **push-forward** (phép đẩy tới, ký hiệu dấu *) là công thức toán học chuẩn cho biết: nếu ta di chuyển toàn bộ các điểm của phân phối p₀ theo ánh xạ φₜ, thì phân phối kết quả pₜ được tính như thế nào — cần nhân thêm một hệ số **định thức Jacobian (Jacobian determinant, ký hiệu det[...])** để "bù trừ" việc không gian bị co giãn khi biến đổi (giống như khi kéo giãn một tấm bản đồ, mật độ các điểm trên đó cũng phải thay đổi theo tỷ lệ tương ứng). Bạn không cần nhớ công thức chi tiết — chỉ cần hiểu: **mỗi phép di chuyển điểm dữ liệu đều kéo theo một phép "cập nhật lại" phân phối xác suất tương ứng**, và công thức (4) là cách tính chính xác việc cập nhật đó.

### Giai đoạn 1 — Mục tiêu Flow Matching (FM) — vì sao chưa dùng trực tiếp được

Gọi **x₁** là biến ngẫu nhiên đại diện cho dữ liệu thật, tuân theo phân phối dữ liệu chưa biết q(x₁) (ta chỉ có các mẫu dữ liệu thật, không biết công thức của q). Gọi pₜ là một đường đi xác suất mục tiêu mà ta muốn CNF học theo, sao cho p₀ là phân phối đơn giản (ví dụ nhiễu Gauss chuẩn N(x|0,I)) và p₁ xấp xỉ phân phối dữ liệu thật q.

**Công thức (5) — FM loss:** L_FM(θ) = 𝔼_{t, pₜ(x)} ‖vₜ(x) − uₜ(x)‖²

*Giải nghĩa:*
- **𝔼[...]**: ký hiệu kỳ vọng — tức "giá trị trung bình" khi lấy mẫu ngẫu nhiên t (theo phân phối đều U[0,1] — mọi giá trị t từ 0 đến 1 có khả năng được chọn như nhau) và x (theo phân phối pₜ(x) tại thời điểm t đó).
- **uₜ(x)**: trường véc-tơ mục tiêu — trường véc-tơ "đúng" mà nếu mạng nơ-ron học được chính xác thì sẽ sinh ra đúng đường đi pₜ mong muốn.
- **‖vₜ(x) − uₜ(x)‖²**: bình phương độ dài của hiệu 2 véc-tơ — chính là công thức MSE, đo "mạng dự đoán sai lệch bao nhiêu so với mục tiêu".
- **Ý nghĩa toàn bộ:** huấn luyện mạng vₜ(x;θ) sao cho nó "bắt chước" đúng trường véc-tơ mục tiêu uₜ(x) tại mọi điểm x, mọi thời điểm t. Nếu loss = 0 hoàn toàn, mạng học được chính xác đường đi mong muốn.

**Vấn đề:** công thức (5) tuy đơn giản và hấp dẫn, nhưng **không khả thi để dùng trực tiếp (intractable)** vì 2 lý do: (a) ta không biết trước công thức của pₜ hợp lý (chỉ biết p₀ và mong muốn p₁≈q); (b) ngay cả khi chọn được pₜ, nói chung **không có công thức đóng (closed form)** cho uₜ tương ứng.

### Giai đoạn 2 — Xây dựng pₜ, uₜ từ các "đường đi có điều kiện" (conditional probability paths)

Đây là **ý tưởng đột phá chính** của bài báo: thay vì cố định nghĩa pₜ và uₜ một cách trực tiếp (khó), hãy xây dựng chúng **gián tiếp thông qua từng mẫu dữ liệu riêng lẻ**.

**Bước xây dựng:**
- Với **mỗi mẫu dữ liệu cụ thể x₁** (ví dụ: một bức ảnh mèo cụ thể trong tập huấn luyện), định nghĩa một **conditional probability path** (đường đi xác suất có điều kiện) pₜ(x|x₁) — nghĩa là: "nếu ta biết trước đích đến là x₁, thì đường đi xác suất riêng cho mẫu này trông như thế nào". Yêu cầu: tại t=0, pₜ(x|x₁) = p(x) (phân phối nhiễu đơn giản, giống nhau cho mọi mẫu); tại t=1, p₁(x|x₁) là một phân phối **co cụm rất sát quanh x₁** (ví dụ N(x|x₁, σ²I) với σ rất nhỏ — gần như "chắc chắn" bằng x₁).

**Công thức (6) — Marginal probability path (đường đi xác suất biên):**

pₜ(x) = ∫ pₜ(x|x₁) · q(x₁) dx₁

*Giải nghĩa:* đây là phép **lấy biên (marginalizing)** — lấy trung bình có trọng số của tất cả các đường đi có điều kiện pₜ(x|x₁), với trọng số chính là q(x₁) (xác suất mẫu x₁ xuất hiện trong dữ liệu thật). Có thể hiểu: pₜ(x) là một **phân phối hỗn hợp (mixture)** của vô số "đường đi riêng lẻ", mỗi đường đi hướng về một mẫu dữ liệu cụ thể.

**Công thức (7):** p₁(x) = ∫ p₁(x|x₁) q(x₁) dx₁ ≈ q(x)

*Giải nghĩa:* tại t=1, vì mỗi p₁(x|x₁) đã co cụm rất sát quanh x₁, nên phân phối hỗn hợp tổng thể p₁ sẽ xấp xỉ đúng bằng phân phối dữ liệu thật q — đúng như mục tiêu ban đầu đặt ra.

*Hình dung dễ nhất:* tưởng tượng bạn có hàng triệu "cái phễu" nhỏ, mỗi cái phễu hướng về đúng một bức ảnh thật trong tập dữ liệu. Đổ nhiễu ngẫu nhiên vào chung tất cả các phễu (t=0, mọi phễu trông giống hệt nhau — nhiễu thuần túy), rồi để nó chảy dần xuống (t tăng dần) — mỗi hạt nhiễu sẽ "chảy" theo phễu mà nó rơi vào và tụ dần về đúng bức ảnh mà phễu đó hướng tới (t=1). Tổng thể toàn bộ các hạt ở đáy phễu, gộp lại, sẽ tạo thành đúng hình dạng phân phối của toàn bộ tập ảnh thật.

**Công thức (8) — Marginal vector field (trường véc-tơ biên):**

uₜ(x) = ∫ uₜ(x|x₁) · [pₜ(x|x₁)·q(x₁) / pₜ(x)] dx₁

*Giải nghĩa:*
- **uₜ(x|x₁)**: trường véc-tơ có điều kiện — trường véc-tơ riêng cho mẫu x₁, biết trước công thức (vì ta tự thiết kế đường đi pₜ(x|x₁) nên có thể suy ra uₜ(x|x₁) tương ứng — sẽ nói rõ ở Giai đoạn 5).
- **[pₜ(x|x₁)·q(x₁) / pₜ(x)]**: đây chính là công thức Bayes — tính "xác suất hậu nghiệm" rằng, nếu đang đứng tại điểm x ở thời điểm t, thì khả năng điểm này thuộc về "phễu" hướng tới x₁ là bao nhiêu.
- **Ý nghĩa toàn bộ:** trường véc-tơ biên uₜ(x) tại một điểm x bất kỳ = **trung bình có trọng số** của tất cả các trường véc-tơ có điều kiện uₜ(x|x₁), với trọng số là "khả năng điểm x này thuộc về mẫu x₁ nào".

**Nhận định mấu chốt (Theorem 1 — Định lý 1):** *"Marginal vector field (trường véc-tơ biên, công thức 8) sinh ra đúng marginal probability path (đường đi xác suất biên, công thức 6)."*

*Giải nghĩa & vì sao quan trọng:* nghe có vẻ hiển nhiên nhưng thực ra **không tầm thường** — nó có nghĩa là: nếu ta chỉ cần biết cách thiết kế các đường đi/trường véc-tơ **riêng lẻ cho từng mẫu** (rất dễ, vì mỗi mẫu chỉ là một điểm cụ thể), thì phép "trộn" chúng lại theo công thức Bayes ở trên sẽ **tự động** cho ra đúng trường véc-tơ tổng thể ta mong muốn — dù trường véc-tơ tổng thể này (uₜ(x)) vẫn là một đại lượng **không thể tính trực tiếp được** (vì phải tích phân qua toàn bộ dữ liệu).

### Giai đoạn 3 — Conditional Flow Matching (CFM): biến cái "không tính được" thành cái "tính được"

Vấn đề còn lại: công thức (5) (FM loss) cần biết uₜ(x) — nhưng theo công thức (8), uₜ(x) là một tích phân **không khả thi để tính (intractable)** (phải tích phân qua toàn bộ tập dữ liệu x₁). Bài báo đưa ra giải pháp cực kỳ thanh lịch:

**Công thức (9) — CFM loss:** L_CFM(θ) = 𝔼_{t, q(x₁), pₜ(x|x₁)} ‖vₜ(x) − uₜ(x|x₁)‖²

*Giải nghĩa:* thay vì lấy mẫu x theo pₜ(x) (đường đi biên, khó) và so sánh với uₜ(x) (khó tính), ta:
1. Lấy mẫu **một mẫu dữ liệu thật x₁** ngẫu nhiên từ tập huấn luyện (theo q(x₁) — điều này rất dễ, chỉ cần lấy ngẫu nhiên 1 ảnh trong tập dữ liệu).
2. Lấy mẫu **t** ngẫu nhiên trong [0,1].
3. Lấy mẫu **x** từ đường đi có điều kiện pₜ(x|x₁) (dễ, vì ta tự thiết kế nó, thường chỉ là một phân phối Gauss đơn giản).
4. Tính **uₜ(x|x₁)** theo công thức đóng đã biết trước (dễ, xem Giai đoạn 5).
5. So sánh với dự đoán của mạng vₜ(x;θ) bằng MSE.

**Nhận định mấu chốt thứ 2 (Theorem 2 — Định lý 2):** *"FM (công thức 5) và CFM (công thức 9) có građient giống hệt nhau theo θ (sai khác một hằng số không phụ thuộc θ)."*

*Giải nghĩa & vì sao quan trọng:* điều này nghĩa là **tối ưu CFM loss (dễ, tính toán được) tương đương hoàn toàn với việc tối ưu FM loss (khó, không tính toán được)** — cùng một điểm tối ưu, cùng hướng cập nhật tham số. Nhờ định lý này, ta có thể huấn luyện CNF hoàn toàn theo kiểu **simulation-free** (không cần mô phỏng/giải ODE trong lúc huấn luyện — chỉ cần lấy mẫu và tính MSE trực tiếp), mà vẫn đảm bảo học đúng đường đi xác suất mong muốn pₜ, tức cuối cùng p₁ sẽ xấp xỉ đúng phân phối dữ liệu thật q.

*Hình dung dễ nhất:* giống việc bạn muốn học "quy luật chung của cả một khu rừng" (uₜ(x) — khó, vì phải khảo sát toàn bộ khu rừng cùng lúc), nhưng thay vào đó bạn chỉ cần học "cách một cái cây cụ thể mọc" (uₜ(x|x₁) — dễ, chỉ quan sát 1 cây), rồi lặp lại việc học đó với rất nhiều cây khác nhau được chọn ngẫu nhiên — về mặt toán học, kết quả trung bình sẽ giống hệt như việc học quy luật chung của cả khu rừng.

### Giai đoạn 4 — Thiết kế họ đường đi Gauss có điều kiện (Conditional Gaussian probability paths)

Bây giờ cần chọn cụ thể pₜ(x|x₁) là gì. Bài báo chọn họ **phân phối Gauss** vì đơn giản, dễ lấy mẫu, dễ tính công thức đóng.

**Công thức (10):** pₜ(x|x₁) = N( x | μₜ(x₁), σₜ(x₁)² · I )

*Giải nghĩa:*
- **μₜ(x₁)** (viết là "muy"): giá trị trung bình (mean) của phân phối Gauss tại thời điểm t — một hàm số phụ thuộc cả t lẫn x₁, cho biết "tâm điểm" của phân phối đang ở đâu.
- **σₜ(x₁)** (viết là "sigma"): độ lệch chuẩn (standard deviation/std) tại thời điểm t — cho biết phân phối đang "co cụm chặt" hay "trải rộng".
- **Điều kiện biên (boundary condition):** μ₀(x₁)=0, σ₀(x₁)=1 → tại t=0, mọi mẫu x₁ đều cho ra **cùng một phân phối** N(x|0,I) (nhiễu Gauss chuẩn, không phân biệt được mẫu nào) — đúng như ta cần: điểm khởi đầu là nhiễu thuần túy. Còn μ₁(x₁)=x₁, σ₁(x₁)=σ_min (một số rất nhỏ) → tại t=1, phân phối co cụm rất sát quanh chính mẫu dữ liệu x₁.

**Công thức (11)-(13):** ψₜ(x) = σₜ(x₁)·x + μₜ(x₁)

*Giải nghĩa:* trong vô số cách chọn trường véc-tơ có thể sinh ra cùng một đường đi pₜ(x|x₁) (vì có thể cộng thêm các thành phần **divergence-free** — không phân kỳ, chỉ "xoay" mà không ảnh hưởng đến phân phối xác suất), bài báo chọn lựa chọn **đơn giản nhất (canonical)**: một **phép biến đổi affine** (nhân rồi cộng) ψₜ(x) = σₜ(x₁)·x + μₜ(x₁). Nếu x là một mẫu từ phân phối Gauss chuẩn N(0,I), thì sau khi áp dụng ψₜ, kết quả chính xác là một mẫu từ phân phối Gauss N(μₜ(x₁), σₜ(x₁)²I) mong muốn — đây là tính chất toán học cơ bản của phân phối Gauss (nhân với hằng số và cộng thêm hằng số vẫn cho ra phân phối Gauss, chỉ đổi mean/std). Công thức (13): d/dt ψₜ(x) = uₜ(ψₜ(x)|x₁) — nghĩa là đạo hàm theo thời gian của phép biến đổi ψₜ chính là giá trị trường véc-tơ tại điểm đó.

**Công thức (14):** viết lại CFM loss theo x₀ (thay vì x) — chỉ là một bước biến đổi kỹ thuật để thuận tiện lấy mẫu (lấy x₀ từ Gauss chuẩn rồi biến đổi qua ψₜ để ra x, thay vì lấy x trực tiếp).

**Theorem 3 (Định lý 3) — Công thức (15):**

uₜ(x|x₁) = [σₜ'(x₁) / σₜ(x₁)] · (x − μₜ(x₁)) + μₜ'(x₁)

*Giải nghĩa:* đây là **công thức đóng (closed form)** — tính được trực tiếp bằng một phép toán đơn giản — cho trường véc-tơ có điều kiện, suy ra từ phép biến đổi affine ở trên. **σₜ'** và **μₜ'** (dấu phẩy trên đầu, ký hiệu đạo hàm theo thời gian, "f' = df/dt") là tốc độ thay đổi của độ lệch chuẩn và giá trị trung bình theo thời gian. Vì ta **tự chọn** μₜ(x₁) và σₜ(x₁) là hàm số gì (miễn thỏa điều kiện biên), nên ta luôn tính được đạo hàm của chúng, từ đó tính trực tiếp ra uₜ(x|x₁) mà **không cần biết** phương trình vi phân phức tạp nào cả — đây chính là điều làm cho CFM loss (công thức 9) hoàn toàn khả thi tính toán trong thực tế.

*Hình dung dễ nhất:* việc chọn μₜ, σₜ giống như bạn tự vẽ trước "quỹ đạo mong muốn" của một hạt bụi (đi từ đâu, co cụm dần về đâu, theo tốc độ nào) — một khi đã vẽ trước quỹ đạo, tốc độ tức thời tại từng điểm trên quỹ đạo (chính là uₜ(x|x₁)) chỉ đơn giản là "đạo hàm" của quỹ đạo đó, tính được ngay bằng công thức.

### Giai đoạn 5 — Hai trường hợp đặc biệt: Diffusion path và Optimal Transport (OT) path

Cùng một công thức (15), chỉ cần **chọn μₜ và σₜ khác nhau** sẽ cho ra các họ đường đi khác nhau. Bài báo trình bày 2 lựa chọn tiêu biểu:

**(a) Diffusion conditional VFs (Trường véc-tơ có điều kiện kiểu khuếch tán) — công thức (16)-(19):**

- Đây là cách chọn μₜ, σₜ sao cho **khớp lại đúng** những gì các mô hình diffusion truyền thống (VE — variance exploding/phương sai bùng nổ, và VP — variance preserving/giữ phương sai) đã và đang dùng.
- Ví dụ đường VE: pₜ(x) = N(x|x₁, σ²₁₋ₜ·I) với σₜ là hàm **tăng dần**, σ₀=0, σ₁≫1 (rất lớn) — nghĩa là càng lùi về t=0 thì nhiễu càng ít, càng tiến gần t=1 thì nhiễu càng "bùng nổ".
- Đường VP dùng một hàm tỉ lệ αₜ = e^(−½T(t)) với T(t) = ∫β(s)ds — **β** là **noise schedule** (lịch trình nhiễu, hàm quyết định tốc độ bơm nhiễu theo thời gian), giống hệt cách các mô hình diffusion cổ điển (DDPM) định nghĩa.
- **Điểm mấu chốt của phần này:** bài báo chứng minh rằng diffusion model **chỉ là một trường hợp đặc biệt** của khung Flow Matching tổng quát — nếu chọn μₜ, σₜ đúng theo công thức khuếch tán cũ, ta thu được cùng một trường véc-tơ mà các phương pháp score matching (đối khớp điểm số) cũ từng dùng, nhưng giờ **huấn luyện bằng CFM lại ổn định và mạnh mẽ hơn (more robust and stable)** so với huấn luyện bằng score matching truyền thống — đây là một phát hiện thực nghiệm quan trọng của bài báo (dùng framework mới để huấn luyện lại đường đi cũ, vẫn thắng).
- Nhược điểm của đường diffusion: **quỹ đạo bị cong (curved)**.

**(b) Optimal Transport (OT) conditional VFs — công thức (20)-(24):**

**Công thức (20):** μₜ(x) = t·x₁ ,  σₜ(x) = 1 − (1−σ_min)·t

*Giải nghĩa:* đây là lựa chọn **đơn giản và "tự nhiên" nhất có thể nghĩ ra**: cho mean và std thay đổi **tuyến tính (linear)** theo thời gian t — mean đi thẳng từ 0 (t=0) đến x₁ (t=1), std đi thẳng từ 1 (t=0) xuống σ_min (t=1).

**Công thức (21):** uₜ(x|x₁) = [x₁ − (1−σ_min)·x] / [1 − (1−σ_min)·t]

*Giải nghĩa:* áp công thức (15) (Theorem 3) vào lựa chọn tuyến tính ở trên, ra được trường véc-tơ có điều kiện dạng đơn giản này — **xác định cho mọi t ∈ [0,1]** (khác với công thức diffusion (19) chỉ xấp xỉ được, không có công thức đóng gọn cho mọi t).

**Công thức (22):** ψₜ(x) = (1 − (1−σ_min)t)·x + t·x₁ — chính là quỹ đạo (flow) tương ứng, một phép nội suy tuyến tính giữa điểm nhiễu x₀ và điểm dữ liệu x₁.

**Công thức (23):** L_CFM(θ) = 𝔼 ‖vₜ(ψₜ(x₀)) − (x₁ − (1−σ_min)x₀)‖² — đây là dạng cụ thể, dễ code, của CFM loss khi dùng đường đi OT: chỉ cần lấy 1 điểm nhiễu x₀, 1 điểm dữ liệu x₁, nội suy tuyến tính ra điểm x tại thời điểm t, rồi bắt mạng dự đoán đúng hướng "từ x₀ đi thẳng tới x₁".

**Công thức (24) — liên hệ với lý thuyết Optimal Transport:**

pₜ = [(1−t)·id + t·ψ]* p₀ ,  với id(x)=x (ánh xạ đồng nhất — giữ nguyên, không đổi gì)

*Giải nghĩa:* đây chính là định nghĩa toán học chuẩn của **OT displacement interpolation** (phép nội suy dịch chuyển kiểu vận chuyển tối ưu — theo lý thuyết của McCann, 1997): cách "di chuyển tối ưu" (ít tốn công sức di chuyển nhất) một phân phối xác suất p₀ thành phân phối p₁. Bài báo chứng minh: khi p₀ là phân phối Gauss chuẩn, lựa chọn tuyến tính đơn giản ở công thức (20) **chính xác trùng khớp** với lời giải toán học tối ưu này — tức là **quỹ đạo tuyến tính (đường thẳng) không chỉ đơn giản mà còn là lựa chọn tối ưu về mặt lý thuyết vận chuyển**.

**So sánh trực quan (Figure 2, 3 trong paper):**
- Đường **diffusion**: quỹ đạo các điểm dữ liệu di chuyển theo đường **cong**, hướng đi (được thể hiện qua hàm điểm số — score function) **thay đổi liên tục theo thời gian**, đôi khi "đi quá đà" rồi phải quay lại (**overshoot**) — gây lãng phí bước tính toán.
- Đường **OT**: quỹ đạo là **đường thẳng tuyệt đối**, hướng đi và tốc độ **không đổi theo thời gian** — dễ học hơn cho mạng nơ-ron (bài toán hồi quy đơn giản hơn nhiều), và khi sinh mẫu, bộ giải ODE cần **ít bước hơn hẳn** để đi theo đúng đường thẳng này.

*Hình dung dễ nhất:* nếu diffusion path giống việc lái xe đi từ điểm A đến điểm B qua một con đường **đèo núi quanh co** (phải đánh lái liên tục, đi chậm, dễ đi lố khúc cua), thì OT path giống việc đi trên một **đường cao tốc thẳng tắp** từ A đến B — vừa nhanh vừa đơn giản để "học" cách lái đúng.

### Giai đoạn 6 — Tóm tắt thuật toán huấn luyện (Training algorithm)

Gộp toàn bộ Giai đoạn 1-5 lại thành các bước thực thi cụ thể mỗi vòng lặp huấn luyện (dùng đường đi OT làm ví dụ, vì đơn giản nhất):

1. Lấy ngẫu nhiên 1 mẫu dữ liệu thật **x₁** từ tập huấn luyện.
2. Lấy ngẫu nhiên 1 điểm nhiễu **x₀ ~ N(0,I)**.
3. Lấy ngẫu nhiên thời điểm **t ~ U[0,1]** (phân phối đều trong [0,1]).
4. Tính điểm nội suy **x = ψₜ(x₀) = (1−(1−σ_min)t)·x₀ + t·x₁** (một điểm nằm trên đường thẳng nối x₀ và x₁, ở vị trí tương ứng thời điểm t).
5. Tính trường véc-tơ mục tiêu **uₜ(x|x₁) = x₁ − (1−σ_min)x₀** (công thức đóng, không cần mạng nơ-ron nào tính cả).
6. Cho mạng nơ-ron dự đoán **vₜ(x; θ)**.
7. Tính loss = MSE giữa vₜ(x;θ) và uₜ(x|x₁), rồi cập nhật θ bằng gradient descent (giảm dần theo građient — thuật toán tối ưu chuẩn của deep learning).
8. Lặp lại hàng triệu lần với các mẫu x₁, x₀, t khác nhau.

**Điểm mấu chốt cần nhớ:** toàn bộ quá trình huấn luyện **không hề cần giải ODE** — mọi bước đều là các phép tính đại số đơn giản, có thể tính song song hàng loạt (batch) rất nhanh, giống hệt cách huấn luyện các mạng nơ-ron thông thường khác.

### Giai đoạn 7 — Suy luận / Sinh mẫu (Inference/Sampling)

Sau khi huấn luyện xong, để tạo ra một mẫu dữ liệu mới (ví dụ một bức ảnh mới):

1. Lấy 1 điểm nhiễu ngẫu nhiên **x₀ ~ N(0,I)**.
2. Dùng một **bộ giải ODE** (ODE solver — có thể là loại đơn giản như Euler/Midpoint/RK4 với số bước cố định — **fixed-step solver**, hoặc loại thông minh tự điều chỉnh độ chính xác như **dopri5** — **adaptive solver**) để "đi theo" trường véc-tơ vₜ(x;θ) mà mạng đã học được, xuất phát từ x₀ tại t=0, tích lũy dần đến t=1 theo đúng công thức ODE (1)-(2) ở Giai đoạn 0.
3. Kết quả cuối cùng **x₁ = φ₁(x₀)** chính là mẫu dữ liệu (ảnh) mới được sinh ra.

**Lợi ích của đường đi OT ở bước suy luận:** vì quỹ đạo là đường thẳng, tốc độ không đổi, nên bộ giải ODE **cần ít bước (ít NFE — số lần gọi hàm) hơn hẳn** để đi theo đúng đường mà vẫn chính xác — thực nghiệm cho thấy chỉ cần khoảng **60% số NFE** so với mô hình huấn luyện theo đường diffusion để đạt cùng độ chính xác.

---

## 4. Kết quả thực nghiệm (Experiments)

### Cách đánh giá — giải thích các chỉ số trước khi xem số liệu
- **NLL (Negative Log-Likelihood — hợp lý âm log)**, đo bằng đơn vị **BPD (Bits Per Dimension — số bit trên mỗi chiều dữ liệu)**: đo mức độ mô hình "giải thích" tốt xác suất của dữ liệu thật — **càng thấp càng tốt**.
- **FID (Fréchet Inception Distance)**: so sánh phân phối đặc trưng (rút trích bởi một mạng nơ-ron có sẵn tên Inception) giữa ảnh thật và ảnh do mô hình sinh ra — **càng thấp càng tốt** (ảnh sinh ra càng giống ảnh thật).
- **NFE (Number of Function Evaluations — số lần gọi hàm)**: số lần bộ giải ODE phải gọi mạng nơ-ron để sinh ra 1 mẫu — **càng thấp thì sinh mẫu càng nhanh/rẻ**.
- **IS (Inception Score)**: đo chất lượng + độ đa dạng ảnh sinh ra — **càng cao càng tốt**.
- **PSNR, SSIM**: đo độ giống nhau về mặt điểm ảnh/cấu trúc giữa ảnh sinh ra và ảnh gốc (dùng trong tác vụ siêu phân giải — super-resolution) — **càng cao càng tốt**.

### Kết quả chính (Bảng 1 — CIFAR-10, ImageNet 32×32, 64×64, 128×128)

So sánh **cùng một kiến trúc mạng (U-Net)**, chỉ đổi cách huấn luyện/đường đi xác suất — đây chính là một dạng **ablation** (so sánh cắt bỏ/thay thế thành phần) rất rõ ràng:

| Phương pháp | Ý nghĩa |
|---|---|
| DDPM | Mô hình khuếch tán chuẩn (huấn luyện qua score matching kiểu DDPM) |
| Score Matching (SM) | Huấn luyện bằng đối khớp điểm số, dùng đường đi khuếch tán VP |
| ScoreFlow (SF) | Một biến thể cải tiến của score matching |
| **FM w/ Diffusion** | Flow Matching, nhưng dùng **cùng đường đi khuếch tán** như các phương pháp trên (chỉ đổi cách huấn luyện) |
| **FM w/ OT** | Flow Matching, dùng đường đi **Optimal Transport** (đường thẳng) |

**Kết quả:** trên cả CIFAR-10, ImageNet 32×32, 64×64:
- **FM w/ Diffusion đã vượt qua DDPM/SM/ScoreFlow** dù dùng cùng một đường đi xác suất → chứng minh **cách huấn luyện CFM tốt hơn hẳn score matching truyền thống** (ổn định hơn, kết quả NLL/FID tốt hơn).
- **FM w/ OT vượt qua tất cả**, kể cả FM w/ Diffusion → chứng minh **đổi sang đường đi OT mang lại lợi ích thêm nữa** (cả về NLL, FID, lẫn NFE cần thiết để sinh mẫu).
- Trên ImageNet 128×128, FM w/ OT đạt FID tốt nhất trong số các phương pháp so sánh (trừ một phương pháp GAN đặc biệt dùng thêm điều kiện phụ trợ, không so sánh công bằng trực tiếp được).

### Huấn luyện nhanh hơn (Faster training)
Đo FID theo số epoch (epoch — một lượt duyệt qua toàn bộ dữ liệu huấn luyện) trong quá trình huấn luyện: FM w/ OT **hội tụ nhanh hơn hẳn** — hạ FID xuống thấp chỉ sau ít epoch hơn nhiều so với các baseline (mô hình đối chứng) khác, dù dùng ít tài nguyên tính toán hơn (ví dụ so với VDM cần 10 triệu iteration — vòng lặp cập nhật, FM chỉ cần 500 nghìn iteration với batch — lô dữ liệu — nhỏ hơn).

### Hiệu quả khi sinh mẫu (Sampling efficiency)
- Đo sai số của lời giải ODE (Error of ODE solution) khi dùng số bước NFE thấp: FM w/ OT đạt sai số nhỏ nhất với **cùng số NFE**, tức là "đi đúng đường" hơn với ít bước tính toán hơn — chỉ cần khoảng **60% NFE** so với mô hình dùng đường diffusion để đạt cùng ngưỡng sai số.
- Với các bộ giải bước cố định (fixed-step, chi phí rẻ — Euler, Midpoint, RK4) ở mức NFE thấp (≤100), FM w/ OT vẫn giữ FID tốt nhất, cho thấy khả năng đánh đổi (trade-off) tốt giữa chất lượng ảnh và chi phí tính toán.
- Quan sát trực quan (Figure 6): với mô hình đường OT, nhiễu giảm dần **đều đặn/tuyến tính** theo suốt quá trình sinh ảnh; còn với đường diffusion, nhiễu **chỉ giảm rõ rệt vào gần cuối** quá trình — phản ánh đúng bản chất "đường cong" đã phân tích ở Giai đoạn 5.

### Sinh có điều kiện: siêu phân giải ảnh (Conditional generation — Super-resolution, Bảng 2)
Thử nghiệm phóng to ảnh từ 64×64 lên 256×256, so sánh với:
- **Reference**: chính ảnh gốc độ phân giải cao thật (dùng làm chuẩn đối chiếu, không phải một phương pháp).
- **Regression**: phương pháp hồi quy đơn giản trực tiếp dự đoán ảnh độ phân giải cao.
- **SR3**: một phương pháp diffusion nổi tiếng khác (Saharia et al., 2022).
- **FM w/ OT**: phương pháp của bài báo này.

**Kết quả:** FM w/ OT đạt **FID và IS tốt nhất** (tức ảnh trông thật và đa dạng hơn), trong khi **PSNR/SSIM tương đương** SR3 (độ khớp điểm ảnh gần bằng nhau) — điều này phù hợp với lập luận rằng FID/IS phản ánh chất lượng cảm quan tốt hơn PSNR/SSIM (theo lập luận của chính nhóm tác giả SR3).

---

## 5. "Ablation" — điều gì thực sự tạo nên hiệu quả của Flow Matching?

Từ Bảng 1, có thể tách bạch **2 đóng góp độc lập** của bài báo:

1. **Đổi cách huấn luyện** (từ score matching → CFM), **giữ nguyên đường đi khuếch tán cũ**: đã mang lại cải thiện (FM w/ Diffusion > DDPM/SM/ScoreFlow) → cho thấy bản thân công thức CFM **ổn định và hiệu quả hơn** để huấn luyện, độc lập với việc chọn đường đi nào.
2. **Đổi đường đi** (từ diffusion path cong → OT path thẳng), **giữ nguyên cách huấn luyện CFM**: mang lại cải thiện thêm nữa (FM w/ OT > FM w/ Diffusion) → cho thấy **hình dạng đường đi xác suất** (thứ mà trước đây bị khóa cứng bởi công thức khuếch tán) cũng là một "nút vặn" quan trọng có thể tối ưu riêng, và đường thẳng OT là lựa chọn tốt hơn đường cong diffusion.

Hai phát hiện này tách rời khỏi nhau đúng nhờ tính **tổng quát** của khung Flow Matching — nếu dùng framework cũ (chỉ có score matching gắn liền với diffusion process), sẽ không thể "mổ xẻ" độc lập hai yếu tố này.

---

## 6. Điểm mạnh / hạn chế / hướng phát triển

**Điểm mạnh:**
- **Tổng quát hóa** vượt xa diffusion model: diffusion model chỉ là **một trường hợp riêng** trong họ các đường đi Gaussian mà Flow Matching hỗ trợ.
- **Huấn luyện simulation-free** (không cần giải ODE lúc train), **unbiased** (không thiên lệch) — kết hợp được sự đơn giản/ổn định của diffusion training với sự tổng quát của CNF.
- Đường đi **OT** vừa đơn giản về mặt lý thuyết (có nền tảng toán học chặt — Optimal Transport) vừa mang lại lợi ích thực dụng rõ rệt: huấn luyện nhanh hơn, sinh mẫu cần ít bước hơn (NFE thấp hơn), tổng quát hóa tốt hơn.
- Cho phép dùng **off-the-shelf ODE solver** (bộ giải ODE có sẵn, không cần thiết kế riêng) — khác với nhiều phương pháp diffusion cần bộ giải chuyên biệt để sinh mẫu nhanh.

**Hạn chế (suy luận từ thiết kế và thực nghiệm):**
- Vẫn cần **giải ODE ở bước suy luận (sampling)** — dù ít bước hơn diffusion, vẫn tốn nhiều thời gian hơn so với các mô hình sinh một bước (one-step) như GAN.
- Vẫn giới hạn trong họ **phân phối Gauss có điều kiện** (dù rất tổng quát, chưa chắc là lựa chọn tối ưu cho mọi loại dữ liệu — ví dụ dữ liệu có cấu trúc đặc biệt như đồ thị, dữ liệu rời rạc).
- Cần chọn trước **σ_min** (siêu tham số — hyperparameter) và dạng hàm μₜ, σₜ — vẫn đòi hỏi một mức độ am hiểu/thử nghiệm để chọn phù hợp cho từng bài toán.
- Bản thân bài báo cũng thừa nhận: dù đường đi có điều kiện (per-sample) là tối ưu theo nghĩa OT, **không có gì đảm bảo đường đi biên (marginal) tổng thể cũng tối ưu theo nghĩa OT** — đây là một khoảng cách lý thuyết còn bỏ ngỏ.

**Hướng phát triển (theo Kết luận của bài báo):** framework Flow Matching mở ra khả năng thiết kế **nhiều loại đường đi xác suất khác nữa**, ví dụ dùng **kernel không đẳng hướng (non-isotropic Gaussian)** — nhiễu Gauss không đối xứng đều theo mọi hướng, hoặc các dạng kernel tổng quát hơn nữa — thay vì chỉ giới hạn ở Gauss đẳng hướng đơn giản như trong bài báo này.

**Cân nhắc trách nhiệm xã hội (Social responsibility — mục 8 của paper):** mô hình sinh ảnh có thể bị lạm dụng (ví dụ tạo ảnh giả mạo); nhóm tác giả khuyến nghị dùng tập dữ liệu huấn luyện được kiểm soát nội dung và các công cụ kiểm định/phân loại ảnh để giảm thiểu rủi ro. Ngoài ra, nhu cầu năng lượng để huấn luyện các mô hình sâu đang tăng nhanh — việc Flow Matching giúp huấn luyện hội tụ nhanh hơn (ít iteration hơn) cũng gián tiếp góp phần tiết kiệm năng lượng.
