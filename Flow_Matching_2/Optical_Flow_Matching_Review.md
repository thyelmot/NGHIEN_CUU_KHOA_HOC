# Optical Flow Matching: Reframing Optical Flow as Continuous Transport Dynamics (OFM)

> **Nguồn:** Ao Luo, Xin Li, Fan Yang, Yuezun Li, Zhaoquan Yuan, Shan Zhao, Bing Su, Xiao Wu — CVPR
> 2025 (bản Open Access do Computer Vision Foundation cung cấp). Code: https://github.com/LA30/OFM
>
> Bài này (gọi tắt là **OFM**) làm cùng một việc mà [`Flow_Matching_1_Review.md`](../Flow_Matching_1/Flow_Matching_1_Review.md)
> đã trình bày về mặt lý thuyết (Flow Matching — "so khớp dòng chảy") nhưng áp dụng cụ thể vào bài toán
> **optical flow** (luồng quang học — ước lượng mỗi điểm ảnh ở khung hình 1 di chuyển tới đâu ở khung
> hình 2). Nên đọc file kia trước nếu chưa quen với Flow Matching/ODE — file này sẽ nhắc lại phần cần
> thiết nhưng không giảng lại từ đầu.

---

## 0. Bảng thuật ngữ nền (tra nhanh — nhưng mọi thuật ngữ vẫn được giải thích lại tại chỗ xuất hiện)

| Thuật ngữ tiếng Anh | Giải thích tiếng Việt |
|---|---|
| **Optical Flow** (luồng quang học) | Với 2 khung hình liên tiếp của 1 video, optical flow là 1 "bản đồ" gán cho **mỗi điểm ảnh** ở khung 1 một vector 2 chiều (dx, dy) cho biết điểm ảnh đó đã di chuyển tới đâu ở khung 2. |
| **Displacement** (độ dịch chuyển) | Khoảng cách + hướng mà 1 điểm ảnh di chuyển giữa 2 khung hình — chính là giá trị mà optical flow cần dự đoán. |
| **Discrete correspondence paradigm** (mô hình tương ứng rời rạc) | Cách làm optical flow truyền thống: with mỗi điểm ảnh, đi "tìm" điểm ảnh tương ứng ở khung sau (dựa vào độ giống nhau về màu sắc/đặc trưng), rồi suy ra độ dịch chuyển — 1 bước "khớp" duy nhất, không mô hình hoá *quá trình* di chuyển. |
| **Continuous transport dynamics** (động lực học vận chuyển liên tục) | Cách làm mới của OFM: thay vì "khớp 1 phát", mô hình hoá việc điểm ảnh di chuyển như 1 **quỹ đạo liên tục theo thời gian**, giống như 1 hạt vật lý chuyển động dưới tác dụng của 1 trường vận tốc (velocity field), thay vì "dịch chuyển tức thời". |
| **Velocity field** (trường vận tốc) | Một hàm số cho biết: tại vị trí này, thời điểm này, vật đang di chuyển theo hướng/tốc độ nào. Trong OFM, mạng nơ-ron học ra chính hàm này. |
| **ODE — Ordinary Differential Equation** (phương trình vi phân thường) | Phương trình mô tả **tốc độ thay đổi tức thời** của 1 đại lượng theo thời gian (ví dụ: vận tốc = đạo hàm của vị trí theo thời gian). "Giải" 1 ODE nghĩa là tìm quỹ đạo đầy đủ, biết trước điểm xuất phát và trường vận tốc. |
| **Flow Matching (FM)** | Một họ mô hình sinh dữ liệu (generative model) học cách "vận chuyển" (transport) 1 điểm nhiễu ngẫu nhiên thành 1 điểm dữ liệu thật, bằng cách học 1 trường vận tốc rồi tích phân theo ODE. Xem chi tiết ở `Flow_Matching_1_Review.md`. |
| **Conditional velocity** (vận tốc có điều kiện) | Vận tốc tính riêng cho **1 cặp** (điểm xuất phát, điểm đích) cụ thể — dễ tính nhưng không phải là vận tốc "thật" cần học. |
| **Marginal velocity** (vận tốc biên) | Vận tốc "thật" cần học — là **trung bình có trọng số** của mọi vận tốc có điều kiện có thể dẫn qua đúng vị trí đang xét, tại đúng thời điểm đang xét. |
| **Conditional Flow Matching loss (CFM loss)** | Hàm mất mát dùng vận tốc có điều kiện (dễ tính) để huấn luyện, nhưng về mặt toán học đã được chứng minh tương đương với việc tối ưu vận tốc biên (khó tính trực tiếp). |
| **Conditional Optimal Transport (OT) path** (đường vận chuyển tối ưu có điều kiện) | Cách nối 2 điểm bằng đường thẳng — "quỹ đạo ngắn nhất, đơn giản nhất" giữa điểm nhiễu và điểm dữ liệu, theo lý thuyết Vận chuyển Tối ưu (Optimal Transport). |
| **Euler method / Euler-based ODE solver** (bộ giải ODE bằng phương pháp Euler) | Cách đơn giản nhất để "đi" theo 1 ODE bằng số: chia thời gian thành từng bước nhỏ, ở mỗi bước cứ đi thẳng theo hướng vận tốc hiện tại rồi cập nhật vị trí — càng chia nhỏ bước càng chính xác. |
| **Triangle Velocities Synergy (TVS)** (hiệp lực 3 vận tốc dạng tam giác) | **Đóng góp cốt lõi** của paper: một mẹo hình học, biến 1 vận tốc "khó học" (không có ý nghĩa vật lý rõ ràng) thành **hiệu của 2 vận tốc khác dễ học hơn** (1 trong 2 cái đó chính là optical flow thật), nhờ 3 vận tốc này tạo thành 3 cạnh của 1 tam giác. |
| **OFM-Naive** | Phiên bản "ngây thơ" — áp thẳng công thức Flow Matching gốc vào optical flow mà chưa qua TVS — paper chứng minh phiên bản này **huấn luyện không hội tụ** (không học được). |
| **OFM-TVS** | Phiên bản đầy đủ, có áp dụng TVS — đây là mô hình chính thức của paper (gọi tắt "OFM" trong phần thực nghiệm). |
| **RAFT (Recurrent All-Pairs Field Transforms)** | Một kiến trúc optical flow rất phổ biến (ECCV 2020): trích đặc trưng, tính "quan hệ tương quan" giữa mọi cặp điểm ảnh, rồi dùng 1 mạng hồi quy (recurrent) để tinh chỉnh dần dự đoán qua nhiều vòng lặp. OFM xây trên nền kiến trúc kiểu RAFT. |
| **Cost volume** (khối chi phí/khối tương quan) | Một khối dữ liệu 4 chiều lưu độ giống nhau (tương quan) giữa **mỗi** điểm ảnh ở khung 1 với **mọi** điểm ảnh (hoặc vùng lân cận) ở khung 2 — "bảng tra cứu" để mạng biết điểm nào ở khung 2 giống điểm đang xét ở khung 1. |
| **Correlation layer / correlation volume** | Lớp mạng tính ra cost volume nói trên. |
| **Context encoder / Feature encoder** (bộ mã hoá ngữ cảnh / đặc trưng) | 2 mạng CNN trích xuất đặc trưng: 1 cái lấy đặc trưng "ngữ cảnh" từ khung hình 1 (bối cảnh chung), 1 cái lấy đặc trưng dùng để tính cost volume. |
| **Global matching** (khớp toàn cục) | Kỹ thuật so khớp đặc trưng giữa 2 khung hình dựa trên **toàn bộ ảnh** (dùng softmax) thay vì chỉ trong 1 vùng lân cận nhỏ — giúp bắt được cả những chuyển động lớn/xa. |
| **TwinsSVT** | Một kiến trúc Transformer (mạng chú ý — attention) dùng làm bộ mã hoá đặc trưng trong OFM. |
| **Recurrent decoder** (bộ giải mã hồi quy) | Phần mạng lặp đi lặp lại nhiều vòng, mỗi vòng tinh chỉnh dần dự đoán vị trí điểm ảnh (giống cách RAFT hoạt động). |
| **NFE — Number of Function Evaluations** (số lần gọi mạng) | Số bước rời rạc dùng khi tích phân ODE bằng Euler — mỗi bước cần gọi mạng nơ-ron 1 lần để lấy vận tốc, nên số bước = số lần "gọi hàm". NFE càng lớn, quỹ đạo mô phỏng càng chính xác nhưng càng chậm. |
| **EPE — End-Point Error** (sai số điểm cuối) | Chỉ số đánh giá optical flow phổ biến nhất: khoảng cách Euclid (theo đường chim bay) trung bình giữa điểm cuối dự đoán và điểm cuối thật. |
| **Fl-all / Fl-epe** (tỉ lệ điểm sai nặng) | Phần trăm điểm ảnh có sai số **vượt ngưỡng lớn** (ví dụ > 3 pixel **và** > 5% độ dài flow thật) — đo "số điểm bị sai nghiêm trọng", khác EPE là đo *trung bình*. |
| **1px** | Phần trăm điểm ảnh có sai số lớn hơn 1 pixel — chỉ số đặc trưng của benchmark Spring (đòi hỏi độ chính xác dưới-pixel). |
| **WAUC — Weighted Area Under Curve** | Chỉ số tổng hợp đo "diện tích dưới đường cong" tỉ lệ điểm đúng theo ngưỡng sai số, có trọng số — càng cao càng tốt (ngược với EPE/Fl càng thấp càng tốt). |
| **Zero-shot evaluation** (đánh giá không tinh chỉnh) | Đánh giá mô hình trên 1 bộ dữ liệu mà nó **chưa từng được huấn luyện/tinh chỉnh riêng** — đo khả năng tổng quát hoá (generalization). |
| **Ablation study** (nghiên cứu cắt bỏ thành phần) | Thí nghiệm lần lượt bỏ/đổi từng thành phần của mô hình để xem thành phần nào thực sự đóng góp vào hiệu năng. |
| **Occlusion** (che khuất) | Vùng ảnh ở khung 1 bị vật khác che mất tại khung 2 (hoặc ngược lại) — optical flow ở vùng này về bản chất "vô nghĩa"/rất khó đoán vì không có điểm tương ứng thật. |
| **Rigid-flow pretraining** (tiền huấn luyện luồng cứng) | Huấn luyện trước trên dữ liệu có chuyển động "cứng" (rigid — như camera di chuyển trong cảnh tĩnh, ví dụ bộ dữ liệu TartanAir) trước khi huấn luyện chính, giúp mô hình học được cấu trúc hình học cơ bản. |
| **Dirac delta function** (hàm delta Dirac) | Một phân phối xác suất "suy biến" — toàn bộ xác suất dồn vào **đúng 1 điểm duy nhất** (không có độ ngẫu nhiên nào cả). Trong bài, dùng để mô tả trường hợp α=0 (không còn nhiễu ngẫu nhiên). |
| **Gaussian mixture distribution** (phân phối hỗn hợp Gauss) | Một phân phối xác suất được ghép từ nhiều "cụm" hình chuông Gauss, mỗi cụm có tâm khác nhau — ở đây tâm mỗi cụm phụ thuộc vào ảnh đầu vào (do mạng dự đoán). |

---

## 1. Bối cảnh & động lực nghiên cứu

### 1.1 Optical flow là gì, và tại sao nó quan trọng?

Cho 2 khung hình liên tiếp của 1 đoạn video, **optical flow** (luồng quang học) là một bản đồ 2 chiều
gán cho từng điểm ảnh (pixel) ở khung hình thứ nhất một vector `(dx, dy)` cho biết điểm ảnh đó đã "trôi"
tới đâu ở khung hình thứ hai. Đây là một trong những biểu diễn nền tảng nhất của thị giác máy tính,
đứng sau rất nhiều tác vụ khác: phân tích video, tái tạo cảnh 3D/scene-flow, phân đoạn chuyển động
(motion segmentation), nội suy khung hình video (video frame interpolation), sinh video, và cả việc
"vá" khung hình bị thiếu (video inpainting).

### 1.2 Vấn đề của các phương pháp cũ: chỉ đoán "kết quả cuối", không mô hình hoá "quá trình"

Kể từ khi ra đời, cách tiếp cận optical flow gần như không đổi về mặt bản chất: xem đây là 1 **bài toán
tương ứng rời rạc** (discrete correspondence problem) — với mỗi điểm ảnh ở khung 1, đi tìm điểm ảnh
tương ứng ở khung 2 (dựa trên giả định độ sáng không đổi + tính mượt trong không gian), rồi lấy hiệu số
vị trí làm ra độ dịch chuyển (displacement).

Các kiến trúc mạng sâu hiện đại — **PWC-Net** (dùng kim tự tháp đặc trưng + warping + cost volume),
**RAFT** (tương quan toàn-cặp dày đặc + tinh chỉnh lặp lại nhiều vòng), **GMFlow** (khớp toàn cục) —
đã cải thiện độ chính xác rất nhiều, nhưng **vẫn bị trói buộc vào cùng 1 khuôn khổ**: xem chuyển động
là 1 phép **so khớp đặc trưng theo cặp** (pairwise feature alignment) giữa 2 khung hình liên tiếp. Nói
cách khác, các mô hình này rất giỏi **ước lượng vị trí cuối cùng** nhưng **không hề học "quá trình"**
sinh ra chuyển động đó — chúng chỉ khôi phục lại *kết quả* của chuyển động (tức là độ dịch chuyển), mà
tách rời hoàn toàn khỏi động lực học (dynamics) vật lý đã tạo ra chuyển động ấy.

Hệ quả cụ thể: các mô hình này thường mất tính nhất quán theo thời gian (temporal consistency) khi gặp
vùng bị che khuất (occlusion — điểm ảnh biến mất ở khung sau do vật khác che), chuyển động lớn, hoặc
thay đổi độ sáng.

**Hình dung dễ nhất:** giống như việc bạn chỉ được xem 2 tấm ảnh chụp trước và sau khi 1 quả bóng lăn
qua sân, rồi được yêu cầu đoán "bóng đã di chuyển bao xa" — bạn có thể đoán đúng điểm đến, nhưng bạn
hoàn toàn không biết quả bóng đã lăn theo đường nào, nhanh hay chậm ở đoạn nào, có bị vật cản làm đổi
hướng hay không. Optical flow kiểu cũ chính là kiểu "đoán điểm đến" đó — không có khái niệm về *đường
đi liên tục*.

### 1.3 Ý tưởng của OFM: quay lại cách vật lý mô tả chuyển động

Trong đời thực, chuyển động vật lý (được mô tả bởi cơ học chất lỏng — fluid mechanics — và lý thuyết
vận chuyển — transport theory) luôn tuân theo động lực học **mượt và liên tục**, chi phối bởi 1 **trường
vận tốc** (velocity field) bên dưới. Bài báo này đặt câu hỏi: *tại sao không dạy cho optical flow cách
"suy luận" theo đúng bản chất vật lý đó?*

Ý tưởng cốt lõi: thay vì dự đoán trực tiếp độ dịch chuyển (1 con số/vector tĩnh), **học 1 trường vận
tốc phụ thuộc thời gian** để "vận chuyển" toạ độ điểm ảnh đi theo 1 quỹ đạo liên tục, mượt mà, nhất
quán với phân phối chuyển động thật — rồi dùng 1 bộ giải phương trình vi phân (ODE solver) để "đi theo"
quỹ đạo đó từ điểm xuất phát tới điểm đích. Cách tiếp cận này lấy cảm hứng trực tiếp từ **Flow Matching**
— một khuôn khổ lý thuyết vốn được dùng để **sinh dữ liệu mới** (generative modeling, ví dụ sinh ảnh)
bằng cách học cách "vận chuyển" nhiễu ngẫu nhiên thành dữ liệu thật.

Nhóm tác giả gọi phương pháp của mình là **Optical Flow Matching (OFM)** — theo hiểu biết của họ, đây
là cách tiếp cận **đầu tiên** kết nối lý thuyết vận chuyển sinh dữ liệu (generative transport theory)
với bài toán ước lượng chuyển động thị giác.

Tuy nhiên, việc áp thẳng công thức Flow Matching gốc vào optical flow gặp 1 trở ngại lớn (sẽ giải thích
chi tiết ở mục 4): mục tiêu huấn luyện lý thuyết của Flow Matching **không khớp** với mục tiêu vật lý
rõ ràng mà các kiến trúc optical flow (như RAFT) vốn được thiết kế để tối ưu. Đóng góp kỹ thuật chính
của bài báo — **Triangle Velocities Synergy (TVS — hiệp lực 3 vận tốc dạng tam giác)** — chính là lời
giải cho mâu thuẫn này.

### 1.4 3 đóng góp chính (theo lời tác giả)

1. **Định nghĩa lại bài toán (Redefining the Paradigm):** chuyển optical flow từ "suy luận tương ứng
   rời rạc" sang "động lực học vận chuyển liên tục" — coi chuyển động là 1 trường vận tốc tiến hoá theo
   thời gian, có nền tảng vật lý rõ ràng.
2. **Đổi mới phương pháp (Methodological Innovation):** đề xuất OFM-TVS — xây trên vận chuyển tối ưu có
   điều kiện cổ điển (conditional optimal transport) + bộ giải ODE kiểu Euler, cộng thêm 1 phép biến đổi
   hình học khéo léo (TVS) dung hoà giữa đảm bảo lý thuyết của Flow Matching và mục tiêu huấn luyện thực
   dụng, đã được kiểm chứng tốt của các mô hình optical flow.
3. **Hiệu năng SOTA (State-Of-The-Art — dẫn đầu):** OFM có thể tích hợp liền mạch vào nhiều kiến trúc
   optical flow khác nhau, cải thiện nhất quán độ chính xác và độ ổn định theo thời gian trên Sintel,
   KITTI, Spring — đưa optical flow từ "ánh xạ tĩnh" (static mapping) sang "suy luận động" (dynamical
   reasoning).

---

## 2. Kiến trúc tổng thể (sơ đồ luồng dữ liệu)

```
                              ẢNH ĐẦU VÀO: cặp khung hình liên tiếp (I₁, I₂)
                                          │
                 ┌────────────────────────┼────────────────────────┐
                 │                        │                        │
                 ▼                        ▼                        ▼
        ┌─────────────────┐      ┌─────────────────┐               │
        │ Context Encoder  │      │  Feature Encoder │◄── I₁ và I₂  │
        │ (bộ mã hoá ngữ  │      │  (bộ mã hoá đặc  │    đều đi qua│
        │  cảnh — chỉ I₁) │      │  trưng — cả 2)    │    Feature   │
        └────────┬─────────┘      └────────┬─────────┘    Encoder  │
                 │                          │                       │
                 │ f_c (đặc trưng           ▼                       │
                 │  ngữ cảnh)      ┌─────────────────┐              │
                 │                 │ Correlation Layer│              │
                 │                 │ (lớp tương quan — │              │
                 │                 │ so khớp mọi cặp   │              │
                 │                 │ điểm I₁ ↔ I₂)     │              │
                 │                 └────────┬─────────┘              │
                 │                          │ f_cv (cost volume —    │
                 │                          │  khối chi phí/tương    │
                 │                          │  quan 4 chiều)          │
                 │                          ▼                        │
                 │                 ┌──────────────────┐              │
                 │                 │  x_l  (dự đoán    │◄─────────────
                 │                 │  luồng thô toàn   │  global
                 │                 │  cục — global flow)│  matching
                 │                 └────────┬──────────┘
                 │                          │ dùng làm TÂM của
                 │                          │ phân phối khởi tạo
                 │                          ▼
                 │                 ┌──────────────────┐
                 │                 │ x₀ ~ N(x | x_l, I)│  (lấy mẫu:
                 │                 │ (nhiễu Gauss quanh│   toạ độ
                 │                 │        x_l)       │   ban đầu)
                 │                 └────────┬──────────┘
                 │                          │
                 ▼                          ▼
        ┌───────────────────────────────────────────────────┐
        │           Optical Flow Matching (V_θ^OF)            │
        │  Bộ giải mã hồi quy (recurrent decoder) — nhận vào:  │
        │    x_t (toạ độ hiện tại trên quỹ đạo), t (thời gian),│
        │    f_cv (cost volume), f_c (đặc trưng ngữ cảnh)      │
        │  Ở mỗi bước t, dự đoán vận tốc tức thời:              │
        │       v_t = v_θ(x_t, t | f_cv, f_c)                  │
        │  Rồi cập nhật (Euler ODE solver):                    │
        │       x_{t+Δ} = x_t + v_t · Δt                       │
        │  Lặp lại K bước (K = NFE, mặc định K=3)              │
        └───────────────────────────┬───────────────────────┘
                                     │
                                     ▼
                             x_pred (toạ độ điểm
                             ảnh cuối cùng, sau khi
                             "đi hết" quỹ đạo)
                                     │
                                     ▼
                     f_pred = x_pred − coord₀ (toạ độ gốc)
                                     │
                                     ▼
                          OPTICAL FLOW DỰ ĐOÁN (đầu ra)
```

**Đọc sơ đồ như thế nào:** Nhánh trái/giữa (Context Encoder + Feature Encoder + Correlation Layer) y
hệt 1 kiến trúc kiểu RAFT thông thường — trích đặc trưng và tính độ tương quan giữa 2 khung hình. Điểm
khác biệt cốt lõi nằm ở khối lớn ở giữa/dưới: thay vì cho bộ giải mã hồi quy tinh chỉnh trực tiếp toạ độ
đích `coord_1` như RAFT gốc, OFM cho nó nhận thêm 2 đầu vào **hoàn toàn mới**: `x_t` (vị trí hiện tại
trên 1 quỹ đạo liên tục) và `t` (thời gian trên quỹ đạo đó, chạy từ 0 → 1), rồi output ra **vận tốc**
thay vì output ra **vị trí**. Optical flow cuối cùng chỉ xuất hiện **sau khi** đã "đi hết" toàn bộ quỹ
đạo bằng cách tích phân vận tốc theo thời gian.

---

## 3. Quy trình chi tiết theo từng giai đoạn (pipeline)

Trước khi đi vào từng bước, cần nắm 4 đại lượng xuất hiện xuyên suốt bài báo — **ký hiệu toán học** dùng
nhất quán ở mọi công thức phía dưới:

- `x_i`: toạ độ **gốc** của 1 điểm ảnh ở khung hình I₁ (ví dụ điểm ảnh tại hàng 10, cột 20).
- `f_gt` / `f`: **optical flow thật** (ground-truth) — độ dịch chuyển thật của điểm ảnh đó.
- `x_1`: toạ độ **đích** — vị trí thật của điểm ảnh đó ở khung hình I₂, tức `x_1 = x_i + f_gt`.
- `x_0`: toạ độ **khởi tạo** của quỹ đạo — tương tự "điểm nhiễu ban đầu" trong Flow Matching gốc, nhưng
  ở đây được đặt gần `x_i` (không phải nhiễu Gauss thuần tuý ngẫu nhiên như khi sinh ảnh).
- `x_t`: toạ độ **tại thời điểm `t`** trên quỹ đạo nối `x_0` → `x_1`, với `t` chạy liên tục từ 0 đến 1.

### Giai đoạn 0 — Trích đặc trưng (giống RAFT)

1. Cặp ảnh `(I₁, I₂)` đi qua **Feature Encoder** (bộ mã hoá đặc trưng — 1 mạng CNN/Transformer, ở đây
   dùng kiến trúc **TwinsSVT**, 1 loại Transformer Vision Transformer) để lấy ra đặc trưng của cả 2
   ảnh.
2. `I₁` (riêng khung 1) còn đi qua **Context Encoder** (bộ mã hoá ngữ cảnh) để lấy đặc trưng ngữ cảnh
   `f_c` — thông tin "bối cảnh chung" của cảnh, dùng để hỗ trợ bộ giải mã sau này.
3. Đặc trưng của cả 2 ảnh đi qua **Correlation Layer** (lớp tương quan) để tính ra **cost volume**
   `f_cv` — 1 khối 4 chiều lưu độ giống nhau giữa từng điểm ảnh ở I₁ với mọi điểm (hoặc vùng lân cận)
   ở I₂. Đây chính là "bảng tra cứu" cho bộ giải mã biết điểm nào ở I₂ *có khả năng* tương ứng với điểm
   đang xét ở I₁.

Đến đây, mọi thứ đều giống hệt optical flow kiểu RAFT truyền thống — sự khác biệt bắt đầu từ bước tiếp
theo.

### Giai đoạn 1 — Xác định điểm neo `x_l` và điểm khởi tạo `x_0` (thay cho nhiễu Gauss thuần)

Trong Flow Matching gốc (dùng để **sinh ảnh**), điểm khởi tạo luôn là 1 mẫu nhiễu Gauss thuần tuý
`ε ~ N(0, I)` — vì mục tiêu là biến "hư vô ngẫu nhiên" thành ảnh thật. Nhưng optical flow **không phải**
bài toán sinh từ hư vô — ta **đã có sẵn** vị trí gốc `x_i` của điểm ảnh, nên khởi tạo hoàn toàn ngẫu
nhiên là lãng phí và vô nghĩa về mặt vật lý.

Bài báo giải quyết việc này bằng cách **học 1 điểm neo** `x_l` (constant reference — điểm tham chiếu
cố định cho từng ảnh): mô hình dùng cơ chế **khớp toàn cục** (global matching, kiểu GMFlow — so khớp
đặc trưng dựa trên softmax trên toàn ảnh) để dự đoán **1 ước lượng flow thô, toàn cục** trước, từ đó suy
ra `x_l`. Sau đó, điểm khởi tạo của quỹ đạo được lấy mẫu quanh `x_l`:

```
p_l(x) = N(x | x_l, I)        # phân phối Gauss có TÂM tại x_l
x_0 ~ p_l(x)                   # lấy 1 mẫu ngẫu nhiên từ phân phối đó
```

**Diễn giải:** `x_l` không phải là 1 hằng số cố định cho mọi ảnh — nó **do mạng dự đoán riêng cho từng
cặp ảnh đầu vào**, dựa trên ước lượng chuyển động thô ban đầu. Vì có nhiều điểm ảnh khác nhau trong 1
ảnh (mỗi điểm có `x_l` riêng suy từ flow thô tại vị trí đó), nên xét trên toàn ảnh, `x_0` tuân theo 1
**phân phối hỗn hợp Gauss** (Gaussian mixture) — mỗi "cụm" trong hỗn hợp có tâm là `x_l` của 1 điểm ảnh.

**Hình dung dễ nhất:** giống như trước khi ném phi tiêu vào bia, bạn không nhắm đại vào 1 điểm ngẫu
nhiên trên tường — bạn **liếc nhanh** qua để đoán "khu vực" mà đích có thể nằm ở đó (`x_l` = ước lượng
thô), rồi mới bắt đầu ném thử xung quanh khu vực đó (`x_0` = điểm khởi tạo, có nhiễu ngẫu nhiên nhỏ
quanh khu vực đã đoán) thay vì ném hú hoạ khắp cả bức tường.

*(Tại sao cần bước "liếc nhanh" này thay vì đặt `x_l` cố định bằng `x_i` luôn? Xem giải thích ở Giai
đoạn 2 — đây chính là phần khó nhất và cũng là đóng góp cốt lõi của bài báo.)*

### Giai đoạn 2 — Xây dựng quỹ đạo và áp dụng "Triangle Velocities Synergy" (TVS)

Đây là bước quan trọng nhất, cần giải thích kỹ **tại sao** nó tồn tại.

**2a. Vấn đề của cách làm "ngây thơ" (OFM-Naive):**

Nếu áp thẳng công thức đường thẳng tối ưu (linear conditional optimal-transport path — đường vận
chuyển tối ưu dạng đường thẳng) của Flow Matching gốc:

```
x_t = t·x_1 + (1−t)·x_0        (công thức 7)
```

thì vận tốc lý thuyết cần học là:

```
v_t(x_t | x_1) = x_1 − x_0
```

**Diễn giải công thức:** `x_t` là điểm nằm trên đoạn thẳng nối `x_0` và `x_1`, tại vị trí tỉ lệ `t` (khi
`t=0`, `x_t=x_0`; khi `t=1`, `x_t=x_1`). Vận tốc đi theo đường thẳng này là **hằng số** theo thời gian
— đúng bằng hiệu số điểm đầu và điểm cuối `x_1 − x_0`.

**Vấn đề nằm ở đây:** vì `x_0` không còn là `x_i` (điểm gốc) mà là 1 điểm bị nhiễu (`x_0 = x_i + αε`,
với `α` là hệ số tỉ lệ và `ε` là nhiễu Gauss chuẩn), nên:

```
x_1 − x_0 = (x_i + f_gt) − (x_i + αε) = f_gt − αε
```

**Diễn giải:** vận tốc mà lý thuyết Flow Matching yêu cầu mạng học không phải là optical flow thật
`f_gt`, mà là "optical flow thật **trừ đi** 1 lượng nhiễu ngẫu nhiên `αε`" — 1 đại lượng **không có ý
nghĩa vật lý rõ ràng nào cả**. Trong khi đó, các kiến trúc optical flow (RAFT, cost volume, attention...)
vốn được **thiết kế chuyên biệt** để tìm tương ứng điểm-ảnh-với-điểm-ảnh có ý nghĩa vật lý cụ thể. Khi
bị ép học 1 đại lượng trừu tượng như "flow trừ nhiễu Gauss", việc hội tụ trở nên **rất bất ổn định**,
hiệu năng huấn luyện giảm mạnh (paper xác nhận bằng thực nghiệm ở Bảng 3 — mục 5 bên dưới: OFM-Naive
gần như không học được gì, EPE tệ hơn 15-19 lần so với baseline).

Ngược lại, nếu "chiều lòng" kiến trúc mạng bằng cách đặt `α=0` (loại bỏ nhiễu, `x_0` trở thành 1 hằng
số cố định — về mặt toán gọi là **hàm delta Dirac**, tức phân phối "suy biến" dồn hết xác suất vào
đúng 1 điểm), thì lại nảy sinh vấn đề khác: **mọi quỹ đạo huấn luyện đều xuất phát từ đúng 1 điểm** →
mất hết sự đa dạng cần có của các quỹ đạo điều kiện (conditional trajectories) → bài toán trở thành
"suy biến" (degenerate) → mạng buộc phải học cách vận chuyển 1 điểm khởi đầu duy nhất tới **toàn bộ**
phân phối đích phức tạp — 1 bài toán khó hơn nhiều so với cần thiết.

**Hình dung dễ nhất:** giống như bạn dạy 1 người mới học lái xe bằng cách yêu cầu họ tính "vị trí xe
trừ đi 1 số ngẫu nhiên nào đó" thay vì "vị trí xe thật" — dù về mặt toán 2 cách đều "khớp" theo 1 công
thức nào đó, nhưng cách đầu chẳng liên quan gì tới trực giác lái xe thật của người học, nên họ sẽ học
rất chậm và hay nhầm lẫn.

**2b. Giải pháp: Triangle Velocities Synergy (TVS)**

Bài báo đề xuất **giữ lại nhiễu ngẫu nhiên** (để có đủ đa dạng quỹ đạo) **nhưng đổi tâm của nhiễu** từ
`x_i` (điểm gốc) sang `x_l` (điểm neo học được ở Giai đoạn 1):

```
x_0 = x_l + αε
```

Sau đó, thay vì chỉ có 1 quỹ đạo `x_0 → x_1`, bài báo **thêm 2 quỹ đạo phụ** (auxiliary trajectories),
cả 2 đều xuất phát/kết thúc tại `x_i` (điểm gốc — vị trí thật, không nhiễu) và dùng chung 1 thời gian
`t`:

```
y_t = t·x_1 + (1−t)·x_i      (công thức 8, quỹ đạo phụ 1: từ x_i tới x_1)
z_t = t·x_0 + (1−t)·x_i      (công thức 8, quỹ đạo phụ 2: từ x_i tới x_0)
```

Từ đó suy ra 2 vận tốc phụ (cũng là hằng số theo thời gian, giống lý luận ở trên):

```
v_t(y_t | x_1) = x_1 − x_i        # ĐÂY CHÍNH LÀ optical flow thật f_gt !
v_t(z_t | x_0) = x_0 − x_i        # đây là 1 lượng chỉ liên quan tới điểm neo + nhiễu
```

**Điểm mấu chốt (insight) của toàn bộ bài báo:** dù trong Flow Matching, vận tốc thường được xem là 1
đại lượng **tức thời** (thay đổi theo từng thời điểm), nhưng vì công thức cụ thể ở đây dùng đường thẳng
tối ưu (conditional optimal-transport), nên với 1 mẫu dữ liệu `x_1` cho trước, **cả 3 vận tốc**
`v_t(x_t|x_1)`, `v_t(y_t|x_1)`, `v_t(z_t|x_0)` đều là **hằng số** (không đổi theo `t`). Về mặt hình học,
3 vector hằng số này tạo thành **3 cạnh của 1 hình tam giác** (xem Hình 3 trong bài báo: đỉnh tam giác
là `x_0`, `x_1`, `x_i`), thoả mãn quan hệ cộng vector đơn giản:

```
v_t(x_t | x_1) = v_t(y_t | x_1) − v_t(z_t | x_0)        (công thức 9)
```

**Diễn giải:** đây chỉ là **quy tắc cộng/trừ vector hình tam giác** thông thường — nếu 3 đỉnh tam giác
là A, B, C thì cạnh AB = AC − BC (hay tương đương). Áp dụng vào đây: "vận tốc lý thuyết khó học"
(`v_t(x_t|x_1)`) được viết lại thành **hiệu của 2 vận tốc dễ học hơn**: một cái (`v_t(y_t|x_1)`) chính
là optical flow thật, cái còn lại (`v_t(z_t|x_0)`) chỉ phụ thuộc vào điểm neo và nhiễu — **không phụ
thuộc vào flow thật** nên có thể tính trực tiếp mà không cần học.

**Ý nghĩa thực tiễn:** thay vì bắt mạng học trực tiếp đại lượng khó hiểu `v_t(x_t|x_1) = f_gt − αε` (như
OFM-Naive), TVS cho phép mạng chỉ cần học `v̂_t = v_t(y_t|x_1)` — mà đại lượng này **chính là optical
flow thật** `f_gt`! Vậy là mạng có thể được huấn luyện bằng đúng nhãn optical flow chuẩn đã có sẵn (như
mọi mô hình optical flow khác từ trước tới nay) — hoàn toàn tương thích với cách huấn luyện có giám sát
truyền thống, mà vẫn giữ được toàn bộ đảm bảo lý thuyết của Flow Matching.

**Hình dung dễ nhất:** giống như bạn muốn biết "từ nhà bạn A đến nhà bạn B đi hướng nào, bao xa"
(`v_t(x_t|x_1)`), nhưng đường đó khó đo trực tiếp (địa hình phức tạp). Thay vào đó, bạn đo 2 đoạn dễ đo
hơn: "từ nhà bạn C (mốc chung, dễ xác định — ở đây là `x_i`) đến nhà bạn B" và "từ nhà bạn C đến nhà bạn
A" — rồi **lấy hiệu 2 vector đó** để suy ra chính xác đoạn đường A→B mà không cần đo trực tiếp. Về mặt
hình học, đây là 1 mẹo hoàn toàn "miễn phí" (không mất thông tin) — chỉ là đổi góc nhìn để đo đạc dễ hơn.

**2c. Vì sao cần `x_l` học được (thay vì đặt cố định)?**

Sai số ước lượng optical flow thường **lớn hơn** với những chuyển động lớn (dịch chuyển xa) so với
chuyển động nhỏ. Lý tưởng nhất, điểm neo `x_l` nên nằm **gần đích thật** `x_1` để giảm thiểu quãng
đường mà quỹ đạo cần "đi" (giảm sai số). Nhưng vì hướng của đích thật là chưa biết trước khi suy luận,
1 hằng số cố định không thể phù hợp cho mọi trường hợp — nên bài báo chọn cách **học `x_l`** thông qua
dự đoán luồng thô toàn cục (coarse global flow), như đã mô tả ở Giai đoạn 1.

### Giai đoạn 3 — Huấn luyện (Thuật toán 1, tóm tắt lại bằng lời)

Với mỗi cặp ảnh `(I₁, I₂)` có nhãn optical flow thật `f_gt`, và toạ độ gốc `x_i`:

1. Mô hình dự đoán điểm neo `x_l` từ cặp ảnh (qua nhánh khớp toàn cục).
2. Xây phân phối `p_l(x) = N(x | x_l, I)` và lấy mẫu 1 điểm khởi tạo `x_0` từ phân phối đó; lấy mẫu
   ngẫu nhiên 1 thời điểm `t ~ U(0,1)` (phân phối đều trong khoảng 0 đến 1).
3. Tính điểm đích thật `x_1 = x_i + f_gt`.
4. Tính điểm trên quỹ đạo tại thời điểm `t`: `x_t = t·x_1 + (1−t)·x_0`.
5. Cho `x_t`, `t`, và cặp ảnh `(I₁, I₂)` đi qua mô hình `V_θ^OF` (bộ giải mã hồi quy kiểu RAFT đã mô tả
   ở mục 2) để lấy dự đoán `v̂_t`.
6. **Nhờ TVS**, dự đoán `v̂_t` này được **giám sát trực tiếp bằng optical flow thật**:
   `L(θ) = ||v̂_t − f_gt||²` — tức là hàm mất mát bình phương sai số (giống hệt cách huấn luyện optical
   flow truyền thống, chỉ khác là mạng còn nhận thêm 2 đầu vào `x_t, t`).
7. Cập nhật trọng số mô hình bằng cách tối thiểu hoá hàm mất mát này (qua lan truyền ngược —
   backpropagation, như mọi mạng nơ-ron khác).

**Điều quan trọng cần nhấn mạnh:** dù công thức lý thuyết đầy đủ có `v_t(x_t|x_1) = v̂_t − v̄_t` (trừ đi
1 vận tốc phụ `v̄_t`), nhưng ở bước **huấn luyện**, nhờ TVS, ta chỉ cần trực tiếp giám sát `v̂_t` bằng
`f_gt` — không cần tính `v̄_t` ở giai đoạn này. `v̄_t` chỉ cần dùng ở giai đoạn **suy luận** (sinh mẫu),
xem Giai đoạn 4 ngay dưới đây.

### Giai đoạn 4 — Suy luận / sinh mẫu (Thuật toán 2, tóm tắt lại bằng lời)

Ở giai đoạn suy luận (chưa biết `f_gt`), cần **tích phân** quỹ đạo từ điểm khởi tạo tới điểm đích bằng
cách đi qua nhiều bước rời rạc (Euler ODE solver):

1. Chia thời gian thành `K` bước bằng nhau (mặc định `K=3` — xem lý do chọn `K=3` ở mục 5.4), mỗi bước
   dài `Δt = 1/K`.
2. Dự đoán điểm neo `x_l` từ cặp ảnh (giống hệt bước huấn luyện).
3. Lấy mẫu điểm khởi tạo `x_0 ~ N(x | x_l, I)`. Khởi tạo `x_t ← x_0`.
4. Lặp `K` lần, mỗi lần (bước thứ `i`, tương ứng thời điểm `t = i/K`):
   a. Cho `x_t`, `t`, và đặc trưng ảnh đi qua mô hình để lấy `v̂_t = V_θ^OF(x_t, t | I₁, I₂)` — **vận
      tốc tức thời** dự đoán tại vị trí/thời điểm hiện tại.
   b. Tính vận tốc phụ (không cần học, tính trực tiếp bằng công thức): `v̄_t = x_0 − x_i`.
   c. Áp dụng quan hệ tam giác (công thức 9) để suy ra vận tốc "thật" cần dùng để cập nhật:
      `v_t = v̂_t − v̄_t`.
   d. Cập nhật vị trí theo phương pháp Euler: `x_t ← x_t + v_t · Δt`.
5. Sau `K` bước, thu được `x_pred` — vị trí cuối cùng trên quỹ đạo.
6. Optical flow dự đoán cuối cùng: `f_pred = x_pred − coord_0` (trừ lại toạ độ gốc để ra độ dịch
   chuyển, đúng quy ước như các kiến trúc optical flow khác — giữ tính tương thích).

**Hình dung dễ nhất cho cả quy trình huấn luyện + suy luận:** Huấn luyện giống như dạy 1 người đi bộ
"nhìn bản đồ, đoán hướng cần đi ngay bây giờ để tới đích" — càng đoán đúng hướng ở mọi thời điểm trên
đường, càng học tốt. Suy luận giống như người đó **thực sự đi bộ theo từng bước nhỏ** (K bước), mỗi
bước nhìn lại bản đồ 1 lần để chỉnh hướng — đi càng nhiều bước nhỏ thì đường đi càng mượt và chính xác
(giống dùng nhiều đoạn thẳng ngắn để xấp xỉ 1 đường cong), nhưng cũng tốn thời gian hơn (nhiều lần "nhìn
bản đồ" hơn = nhiều lần gọi mạng hơn = NFE lớn hơn).

### Giai đoạn 5 — Kiến trúc cụ thể hoá (mục 3.3 của paper)

Khung lý thuyết trên (OFM-TVS) hoạt động hoàn toàn ở **mức thuật toán**, không phụ thuộc vào 1 kiến trúc
mạng cụ thể nào — có thể "cắm" vào bất kỳ kiến trúc kiểu RAFT nào (dòng kiến trúc lặp-tinh-chỉnh chiếm
ưu thế trong optical flow hiện đại). Bài báo hiện thực hoá cụ thể như sau:

- **Encoder** (bộ mã hoá): dùng **TwinsSVT** (kiến trúc Transformer), theo đúng các công trình trước
  (FlowFormer, FlowDiffuser, FlowFormer++).
- **Global flow (luồng toàn cục — để suy ra `x_l`)**: dùng khớp toàn cục kiểu **softmax** (như GMFlow)
  — nhưng **chỉ dùng riêng cho việc dự đoán `x_l`**, **bỏ qua** các mô-đun lan truyền/tinh chỉnh flow
  khác của GMFlow để giảm chi phí tính toán thêm.
- **Decoder** (bộ giải mã): phát triển theo FlowDiffuser, có thêm thành phần liên quan thời gian
  (time-relevant components) và chiến lược huấn luyện trạng thái ẩn (hidden-state training strategy).
- **Khác biệt so với decoder truyền thống:** decoder của OFM nhận **`x_t`** (vị trí trên quỹ đạo) làm
  đầu vào trạng thái, thay vì nhận `coord_1` (toạ độ đích ước lượng) như các decoder kiểu RAFT gốc — vẫn
  giữ nguyên cơ chế tinh chỉnh lặp lại (iterative refinement) đặc trưng của RAFT.
- Ở bước giải mã, vận tốc quỹ đạo xác suất tại thời điểm `t` được ước lượng: `v_t = v_θ(x_t, t | f_cv,
  f_c)` (chính là công thức đã mô tả trong sơ đồ ở mục 2).
- Cấu hình mặc định là **OFM-TVS đầy đủ**, nhưng trong phần thực nghiệm, bài báo gọi tắt là **"OFM"**
  cho gọn (bỏ chữ "TVS").

---

## 4. Giải thích các công thức cốt lõi (tổng hợp lại, kèm "hình dung dễ nhất")

*(Phần này gom lại các công thức đã xuất hiện rải rác ở mục 3, để tiện tra cứu nhanh mà không cần đọc
lại toàn bộ pipeline.)*

**Công thức (1) — Đường quỹ đạo tổng quát trong Flow Matching gốc:**
```
x_t = a_t·x + b_t·ε
```
`a_t`, `b_t` là 2 "lịch trình" (schedule) định trước theo thời gian `t`, quy định điểm `x_t` là 1 tổ hợp
có trọng số giữa dữ liệu thật `x` và nhiễu `ε`. **Hình dung:** giống pha trộn 2 màu sơn theo tỉ lệ thay
đổi dần theo thời gian — lúc `t` nhỏ thì gần màu nhiễu, `t` lớn thì gần màu dữ liệu thật.

**Công thức (2) — Vận tốc biên (marginal velocity):**
```
v(x_t, t) := E_{p_t(v_t|x_t)}[v_t]
```
Vì có **nhiều** cặp (dữ liệu, nhiễu) khác nhau có thể cùng cho ra đúng 1 điểm `x_t` (nhiều "quỹ đạo" đi
qua cùng 1 điểm), vận tốc "đúng" cần học là **trung bình có trọng số xác suất** của mọi vận tốc điều
kiện khả dĩ tại điểm đó. **Hình dung:** giống như tại 1 ngã tư đông người, "hướng đi trung bình của đám
đông" tại đúng vị trí đó là trung bình cộng có trọng số của hướng đi của từng người đang đứng gần đúng
vị trí ấy.

**Công thức (3) — Hàm mất mát Flow Matching gốc:**
```
L_FM(θ) = E_{t, p_t(x_t)} ||v_θ(x_t,t) − v(x_t,t)||²
```
Đo sai số bình phương giữa vận tốc mạng dự đoán và vận tốc biên "đúng", lấy trung bình trên mọi thời
điểm `t` và mọi điểm `x_t` có thể gặp.

**Công thức (4) — ODE để sinh mẫu:**
```
dx_t/dt = v(x_t, t)
```
Đây là định nghĩa của vận tốc: "vận tốc = đạo hàm vị trí theo thời gian". Biết vận tốc ở mọi nơi/mọi
lúc, ta có thể "đi" từ điểm khởi tạo tới điểm đích bằng cách tích phân phương trình này.

**Công thức (5) — Chuyển sang toạ độ ảnh (áp dụng cho optical flow):**
```
x_t = a_t·x_1 + b_t·x_0
```
Giống hệt công thức (1) nhưng đổi ký hiệu: `x_1` (đích, suy từ flow thật) đóng vai "dữ liệu thật",
`x_0` (khởi tạo) đóng vai "nhiễu ban đầu" — nhưng khác Flow Matching sinh ảnh, ở đây `x_0` **không phải**
nhiễu Gauss thuần, mà là toạ độ bị nhiễu nhẹ quanh 1 vị trí có ý nghĩa vật lý.

**Công thức (6) — Hàm mất mát CFM (Conditional Flow Matching) áp cho optical flow:**
```
L_CFM(θ) = E_{t,x_1,x_0} ||V_θ^OF(x_t,t|I₁,I₂) − v_t(x_t|x_1)||²
```
Thay vì minh hoạ trực tiếp vận tốc biên (khó tính vì cần lấy trung bình trên vô số quỹ đạo — công thức
2), ta dùng vận tốc **có điều kiện** (dễ tính hơn nhiều — chỉ ứng với 1 cặp cụ thể) để huấn luyện. Bài
báo (dựa trên lý thuyết Flow Matching gốc) khẳng định: tối thiểu hoá `L_CFM` **tương đương về mặt toán
học** với tối thiểu hoá `L_FM` — nên dùng công thức dễ tính này không làm mất tính đúng đắn lý thuyết.
**Hình dung:** giống như thay vì tính "vận tốc trung bình của cả đám đông" (khó, cần biết hết mọi
người), ta chỉ cần dạy mạng bắt chước "vận tốc của từng người cụ thể" trong nhiều tình huống khác nhau
— về mặt thống kê, học đủ nhiều tình huống cá nhân sẽ tự động hội tụ về đúng hành vi trung bình.

**Công thức (7) — Đường vận chuyển tối ưu tuyến tính (OT-linear path):**
```
x_t = t·x_1 + (1−t)·x_0   ⟹   v_t(x_t|x_1) = x_1 − x_0
```
Trường hợp riêng đơn giản nhất của công thức (5): đi thẳng theo 1 đường thẳng nối `x_0` và `x_1` với
vận tốc không đổi. **Hình dung:** như đi bộ thẳng từ nhà tới trường với tốc độ đều, không tăng tốc/giảm
tốc dọc đường.

**Công thức (8) và (9)** — đã giải thích chi tiết ở Giai đoạn 2 (mục 3) — chính là "quả tam giác" TVS,
phần đóng góp cốt lõi của bài báo.

---

## 5. Kết quả thực nghiệm

### 5.1 Các chỉ số đánh giá (metric) — giải thích trước khi đọc số liệu

- **EPE (End-Point Error — sai số điểm cuối):** khoảng cách Euclid trung bình (đường chim bay, tính
  bằng pixel) giữa vị trí điểm cuối **dự đoán** và vị trí điểm cuối **thật**. Ví dụ EPE = 1.0 nghĩa là
  trung bình, mỗi điểm ảnh bị lệch khoảng 1 pixel so với đáp án đúng. **Càng thấp càng tốt.**
- **Fl-epe / Fl-all (tỉ lệ điểm sai nặng, tính theo %):** phần trăm số điểm ảnh có sai số **vượt
  ngưỡng lớn** (thường là > 3 pixel **và** đồng thời > 5% độ dài vector flow thật tại điểm đó) — đo
  "bao nhiêu điểm bị sai nghiêm trọng", khác với EPE là đo sai số *trung bình* trên toàn bộ điểm ảnh
  (kể cả những điểm sai nhẹ). **Càng thấp càng tốt.**
- **1px:** riêng ở benchmark Spring — phần trăm điểm ảnh có sai số lớn hơn 1 pixel (Spring đòi hỏi độ
  chính xác **dưới mức 1 pixel**, khắt khe hơn Sintel/KITTI). **Càng thấp càng tốt.**
- **WAUC (Weighted Area Under Curve — diện tích dưới đường cong có trọng số):** 1 chỉ số tổng hợp, đo
  diện tích dưới đường cong biểu diễn "tỉ lệ điểm đúng" theo các mức ngưỡng sai số khác nhau, có trọng
  số. **Càng cao càng tốt** (khác hướng so với EPE/Fl/1px).
- **Đánh giá "train" vs "test" vs "zero-shot":** đánh giá trên tập **train** của Sintel/KITTI thường
  dùng để đo khả năng **tổng quát hoá** (mô hình chưa từng thấy chính xác các ảnh này khi chỉ huấn
  luyện trên FlyingChairs+FlyingThings3D — gọi là protocol "C+T"), trong khi đánh giá trên tập **test**
  (nộp lên server chính thức, ẩn nhãn) đo hiệu năng sau khi đã huấn luyện/tinh chỉnh đầy đủ. **Zero-shot**
  (không tinh chỉnh) nghĩa là mô hình được đánh giá trên 1 bộ dữ liệu mà nó **hoàn toàn chưa được huấn
  luyện/tinh chỉnh riêng** trên đó — đo khả năng tổng quát hoá sang phân phối dữ liệu mới.

### 5.2 Các bộ dữ liệu (dataset) dùng để huấn luyện/đánh giá

FlyingChairs, FlyingThings3D (dữ liệu tổng hợp — synthetic), Sintel, KITTI 2015, HD1K (dữ liệu thực tế
— real-world), TartanAir (dùng để tiền huấn luyện thêm — rigid-flow pretraining), và **Spring** (bộ dữ
liệu benchmark mới, đòi hỏi độ chính xác dưới-pixel).

Lịch trình huấn luyện gồm 4 giai đoạn chuẩn (theo thông lệ chung của lĩnh vực): Giai đoạn 1 trên
FlyingChairs (120k vòng lặp — iterations), Giai đoạn 2 thêm FlyingThings3D (150k vòng lặp, tạo thành
protocol "C+T" — chữ viết tắt tên 2 bộ dữ liệu), Giai đoạn 3 huấn luyện tiếp trên hỗn hợp
"C+T+S+K+H" (Chairs+Things+Sintel+KITTI+HD1K, 150k vòng lặp — mô hình này dùng để đánh giá online trên
Sintel/Spring), Giai đoạn 4 tinh chỉnh thêm trên tập huấn luyện KITTI (5k vòng lặp — để nộp kết quả lên
KITTI online). Ngoài ra còn có "Giai đoạn 0" (tuỳ chọn): tiền huấn luyện trên TartanAir.

Toàn bộ đánh giá/nộp kết quả dùng batch size = 1, chạy trên 1 GPU NVIDIA RTX 4090.

### 5.3 Kết quả chính

**Trên Sintel và KITTI (Bảng 1, huấn luyện theo protocol "C+T", đo khả năng tổng quát hoá):**
- Sintel (train): EPE = **0.81** (clean pass — ít nhiễu) và **2.16** (final pass — có thêm hiệu ứng mờ
  chuyển động, sương mù...) — mạnh nhất trong số các phương pháp 2-khung-hình (two-frame) đã công bố.
- KITTI (train): EPE = **3.32**, Fl-all = **10.9%** — vượt qua mọi phương pháp gần đây.
- Sintel (test, online): EPE = **0.94** và **1.85** — vượt SEA-RAFT (L) và DPFlow lần lượt **28.5%** và
  **8.4%**.
- KITTI (test, online): Fl-all = **3.78%** — xếp thứ 2, chỉ thua DPFlow 1 chút.

**So sánh định tính (Hình 4):** OFM cho biên (boundary) sắc nét hơn, cấu trúc chuyển động mạch lạc hơn ở
vùng khó (chuyển động lớn, hoạ tiết lặp lại), trong khi các mô hình cạnh tranh (DPFlow, SEA-RAFT) hay bị
"nhoè"/làm mượt quá mức (bleeding/over-smoothing). Định dạng phân phối chuyển động tường minh (explicit
motion-distribution formulation) của OFM cũng cho kết quả ổn định hơn ở vùng bị che khuất (occlusion).

**Trên Spring (Bảng 2, huấn luyện theo protocol "C+T+S+K+H", đánh giá **zero-shot** — không tinh chỉnh
riêng cho Spring):**
- Spring (train): 1px = **3.496%**, EPE = **0.361** — vượt phương pháp gần đây FlowSeek (L) lần lượt
  **8.9%** và **10.2%**.
- Spring (test, online): đạt kỷ lục mới trên cả 4 chỉ số chuẩn: 1px = **3.660**, EPE = **0.468**, Fl =
  **1.477**, WAUC = **94.462**.
- So với các phương pháp **đa khung hình** (multi-frame — dùng nhiều hơn 2 khung hình liền kề, vốn có
  lợi thế thông tin hơn), OFM (dù chỉ dùng 2 khung hình) vẫn cạnh tranh tốt hoặc vượt trội: vượt
  StreamFlow và MemFlow trung bình lần lượt **18.6%** và **23.6%** trên mọi chỉ số.

### 5.4 Ablation study (nghiên cứu cắt bỏ thành phần) — thành phần nào thực sự có tác dụng?

**Các biến thể công thức OFM (Bảng 3):**

| Biến thể | Mô tả | Kết quả (tóm tắt) |
|---|---|---|
| **OFM-baseline** | Kiến trúc optical flow nền, **không** có mô hình hoá thời gian hay thành phần Flow Matching nào | Điểm chuẩn để so sánh |
| **OFM-Naive** | Áp thẳng công thức lý thuyết gốc (không qua TVS) | **Không hội tụ** — Sintel EPE nhảy vọt lên 15.7-15.73 (so với baseline ~0.96-2.45) — xác nhận đúng vấn đề đã phân tích ở mục 3 (Giai đoạn 2a): mục tiêu huấn luyện thô không có ý nghĩa vật lý khiến huấn luyện đầu-cuối (end-to-end) cực kỳ bất ổn |
| **OFM-Model** | Thêm nhúng thời gian (temporal embeddings) nhưng vẫn theo kiểu phân biệt thuần tuý (purely discriminative — không dùng framework Flow Matching), | Hiệu năng **giảm xuống dưới cả baseline** — chứng tỏ chỉ thêm "tín hiệu thời gian" mà không có cả khung lý thuyết Flow Matching + TVS thì **không đủ**, thậm chí phản tác dụng |
| **OFM-TVS (đầy đủ)** | Kết hợp cả kiến trúc + khung Flow Matching + chiến lược TVS | Cải thiện **nhất quán trên mọi chỉ số** |

**Kết luận từ ablation này:** OFM chỉ thực sự hiệu quả khi **cả 3 yếu tố** (thiết kế kiến trúc, cách
thiết lập theo Flow Matching, và chiến lược TVS) được dùng **cùng nhau** — thiếu bất kỳ yếu tố nào,
hiệu năng không những không tăng mà còn có thể tệ hơn baseline.

**Dữ liệu bổ sung (Extra Data — tiền huấn luyện TartanAir):** OFM đã cho kết quả tốt ngay cả khi không
có TartanAir; tiền huấn luyện thêm bằng luồng cứng (rigid-flow) từ TartanAir cho thêm 1 chút cải thiện,
đặc biệt rõ trên KITTI.

**Hệ số tỉ lệ `α` (Scale Factor):** OFM khá **bền vững** (robust) với nhiều giá trị `α` khác nhau — hiệu
quả của phương pháp không phụ thuộc quá nhiều vào việc tinh chỉnh chính xác `α`. Cấu hình cuối chọn
`α = 10`, cho lợi thế nhỏ nhưng nhất quán so với các giá trị khác đã thử (5, 15).

**Số bước lấy mẫu (Sampling Steps / NFE — Number of Function Evaluations):** giống các mô hình sinh dựa
trên Flow Matching khác, OFM hưởng lợi khi **tăng số bước** trong lúc suy luận. Chỉ với **1 bước** (K=1,
tức 1 lần gọi mạng), OFM đã vượt qua phần lớn phương pháp 2-khung-hình đã công bố. Tăng từ K=1 lên K=4
cho thêm cải thiện, đạt kỷ lục mới. Để cân bằng giữa độ chính xác và chi phí tính toán, bài báo chọn
**K=3** làm mặc định.

**Khả năng tương thích (Compatibility):** thử "cắm" OFM vào 2 kiến trúc RAFT-style khác (RAFT gốc và
SKFlow) mà **không** tinh chỉnh kỹ chi tiết implementation — chỉ áp thuật toán + các mô-đun nhúng thời
gian cần thiết:

| Mô hình | Sintel Clean | Sintel Final | KITTI EPE | KITTI Fl-all |
|---|---|---|---|---|
| RAFT (gốc) | 1.44 | 2.71 | 5.04 | 17.4 |
| **RAFT + OFM** | **1.23** | **2.59** | **4.43** | **15.8** |
| SKFlow (gốc) | 1.22 | 2.46 | 4.27 | 15.5 |
| **SKFlow + OFM** | **1.16** | **2.35** | **3.93** | **14.6** |

→ OFM cải thiện nhất quán cả 2 kiến trúc nền trên mọi chỉ số — cho thấy phương pháp **không** chỉ hiệu
quả với 1 kiến trúc cụ thể được tinh chỉnh riêng, mà có tính tổng quát cao.

**Số tham số & thời gian chạy (Bảng 4):** so với FlowFormer++ (16.2 triệu tham số, 375ms/ảnh) và
FlowDiffuser (16.3 triệu tham số, 260ms/ảnh), OFM (3-NFE, tức K=3) có **15.6 triệu tham số, 270ms/ảnh**
— tương đương hoặc nhẹ hơn 1 chút, dù phải "lặp" K=3 lần trong lúc suy luận. Lý do: các bước NFE chỉ
tương ứng với **1 phần** của quá trình giải mã (decoding) chứ không phải toàn bộ pipeline, nên chi phí
tăng thêm là **cộng dồn từng phần** (partial) chứ không phải **nhân lên toàn bộ** (multiplicative) theo
số bước K.

---

## 6. Điểm mạnh / hạn chế / hướng phát triển (nhận xét khách quan)

### 6.1 Điểm mạnh

- **Ý tưởng có nền tảng lý thuyết chặt chẽ:** không chỉ là 1 mẹo kỹ thuật hời hợt — bài báo chứng minh
  rõ ràng bằng đại số (công thức 5-9) và xác nhận bằng thực nghiệm (ablation Bảng 3) rằng cách làm
  "ngây thơ" (OFM-Naive) thực sự thất bại, và TVS thực sự là lời giải cần thiết chứ không phải tuỳ chọn.
- **Tương thích ngược tốt:** vì bản chất TVS biến vận tốc cần học thành optical flow thật, mô hình có
  thể được huấn luyện bằng đúng dữ liệu/hàm mất mát chuẩn đã dùng cho optical flow từ trước tới nay —
  không cần thay đổi quy trình thu thập/gán nhãn dữ liệu.
- **Có thể "cắm" vào nhiều kiến trúc khác:** kết quả trên RAFT và SKFlow (mục 5.4) cho thấy đây không
  chỉ là 1 mô hình đơn lẻ mà là 1 **khung phương pháp** (framework) có thể áp dụng rộng.
- **Đánh đổi tốc độ/độ chính xác linh hoạt:** nhờ cơ chế NFE, người dùng có thể chọn chạy nhanh (K=1,
  vẫn cạnh tranh tốt) hoặc chính xác hơn (K=4) tuỳ nhu cầu, mà không cần huấn luyện lại mô hình.
- **Vượt trội rõ rệt về tổng quát hoá cross-dataset** (huấn luyện 1 nơi, test nơi khác) — đây thường là
  điểm yếu của các mô hình optical flow học sâu, và là 1 trong những đóng góp được nêu bật nhất.

### 6.2 Hạn chế (bao gồm cả những điều tác giả tự thừa nhận và những điều cần lưu ý thêm)

- **Tự thừa nhận trong bài (mục Limitation):** OFM-TVS dùa trên vận chuyển tối ưu có điều kiện
  (conditional optimal transport) + bộ giải ODE kiểu Euler — đây là **1 giải pháp cổ điển, không phải
  giải pháp tối ưu**. Trong khi đó, lĩnh vực Flow Matching đã tiến bộ với các phương pháp hiệu quả hơn
  như **MeanFlow** và **Shortcut Models** (cho phép sinh mẫu chỉ với 1 bước duy nhất, hiệu quả hơn nhiều
  so với việc phải chạy nhiều bước Euler). Bài báo để ngỏ việc tận dụng những tiến bộ này cho tương lai.
- **Chi phí tính toán tăng theo NFE:** dù đã lý giải mức tăng là "cộng dồn từng phần" chứ không nhân
  toàn bộ, việc cần chạy K lần forward-pass (dù chỉ phần decoder) vẫn tăng độ trễ (latency) so với các
  phương pháp chỉ cần 1 lần chạy thẳng — với các ứng dụng thời gian thực nghiêm ngặt, đây vẫn là 1 chi
  phí cần cân nhắc.
- **Phụ thuộc vào chất lượng của bước "khớp toàn cục" (global matching) để suy `x_l`:** nếu ước lượng
  flow thô ban đầu (dùng để đặt điểm neo `x_l`) sai lệch nhiều — ví dụ với ảnh có hoạ tiết lặp lại
  nghiêm trọng (dễ gây nhầm lẫn khi khớp toàn cục), điểm neo có thể bị đặt sai chỗ, kéo theo toàn bộ quỹ
  đạo xuất phát từ vị trí không phù hợp. Bài báo không phân tích sâu độ nhạy của kết quả cuối với chất
  lượng của bước dự đoán `x_l` này (không có ablation riêng cho phần global matching).
  Trên KITTI online, OFM chỉ xếp thứ 2, kém DPFlow — dù các chỉ số khác OFM đều nhất quán dẫn đầu. Không
  có phân tích lý do vì sao KITTI (dữ liệu thực tế, tình huống lái xe) lại là trường hợp duy nhất OFM
  chưa đứng đầu.
- **Chưa kiểm chứng ở các bài toán motion khác:** dù mở đầu bài báo có nhắc tới nhiều ứng dụng downstream
  của optical flow (video generation, frame interpolation, scene-flow...), thực nghiệm chỉ dừng ở việc
  đo optical flow "thuần" trên benchmark chuẩn — chưa có đánh giá xem cách tiếp cận liên tục theo thời
  gian này có mang lại lợi ích thực sự cho các tác vụ downstream đó hay không (mới chỉ là động lực/lý
  do mở đầu, chưa phải thực nghiệm).
- **Cần hạ tầng huấn luyện phức tạp hơn:** so với 1 mô hình regression thuần tuý (chỉ cần 1 lần
  forward), OFM cần thêm bước lấy mẫu ngẫu nhiên `t`, `x_0`, tính điểm trên quỹ đạo — tăng thêm độ phức
  tạp trong code/hạ tầng huấn luyện, dù không tăng nhiều về chi phí tính toán.

### 6.3 Hướng phát triển (tác giả đề xuất + nhận xét thêm)

- **Do tác giả đề xuất:** tận dụng các phương pháp Flow Matching hiệu quả hơn gần đây (MeanFlow,
  Shortcut Models) để giảm số bước cần thiết trong suy luận, tiến gần tới "sinh 1 bước" (one-step
  generation) mà vẫn giữ chất lượng.
- **Nhận xét thêm (không có trong bài):** vì bài báo tự nhận đây là "cách tiếp cận đầu tiên" kết nối lý
  thuyết vận chuyển sinh dữ liệu với ước lượng chuyển động thị giác, một hướng tự nhiên là mở rộng sang
  **scene flow 3D** hoặc **motion trong video đa khung hình** (dùng nhiều hơn 2 khung liền kề) — nơi bản
  chất "quỹ đạo liên tục theo thời gian" của OFM có thể phát huy lợi thế rõ hơn nữa (vì có nhiều hơn 2
  điểm neo thời gian để ràng buộc quỹ đạo), thay vì chỉ dùng 2 khung hình như hiện tại.
- Việc phân tích độ nhạy với chất lượng bước "khớp toàn cục" (dùng để suy `x_l`), và thử nghiệm các cách
  khác để chọn/khởi tạo điểm neo (thay vì chỉ dùng global matching kiểu GMFlow), có thể là 1 hướng cải
  tiến tiếp theo hợp lý.
