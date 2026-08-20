# Tổng quan Phương án 7 (TVS) — Tam giác hoá vận tốc (Triangle Velocities Synergy)

## 1. Giới thiệu chung
Phương án 7 (TVS) là một kỹ thuật cải tiến dành cho mô hình DiffMM, được xây dựng trực tiếp trên nền tảng của Phương án 6 (Sử dụng Điểm neo thô - Learnable Coarse Anchor). Mục tiêu của phương pháp này là giải quyết sự thiếu ổn định và khó hội tụ của mô hình khi tâm của phân phối nhiễu bị dịch chuyển khỏi gốc toạ độ $0$.

Bằng cách áp dụng các nguyên lý hình học từ Optical Flow Matching (OFM), TVS chuyển đổi bài toán từ việc "dự đoán trực tiếp dữ liệu" sang "dự đoán vận tốc", và chia nhỏ quá trình dự đoán này thành các thành phần dễ học hơn thông qua một quan hệ tam giác.

## 2. Sự khác biệt cốt lõi so với DiffMM (Gốc)

| Đặc điểm | DiffMM (Gốc) | Phương án 7 (TVS) |
| :--- | :--- | :--- |
| **Mục tiêu học (Target)** | **Data-prediction:** Mạng Denoise dự đoán trực tiếp dữ liệu sạch (sở thích thật của user - $\alpha_0$). | **Velocity-prediction:** Mạng Denoise dự đoán *vận tốc* ($v_t$), tức là hướng và tốc độ để đi từ trạng thái nhiễu về đích. |
| **Tâm nhiễu (Noise Center)** | Điểm $0$ (theo phân phối Gaussian chuẩn). | **Điểm neo học được ($\alpha_l$):** Một ước lượng thô về sở thích user được học qua các epoch. |
| **Hàm mất mát (Loss Function)** | $L_{data} = \|\hat{\alpha}_0 - \alpha_0\|^2$ (Chỉ một hàm Loss duy nhất). | $L_{TVS}$: Hàm loss tổng hợp từ 3 nhánh (Tương ứng với 3 quỹ đạo vận tốc). |

## 3. Cơ chế hoạt động của "Tam giác vận tốc" (TVS)

Khi đổi tâm nhiễu từ $0$ sang điểm neo $\alpha_l$, việc dự đoán vận tốc đi thẳng về đích thật ($\alpha_0$) trở nên trừu tượng và khó đoán, vì đích và tâm đều thay đổi. 

TVS giải quyết bằng cách định nghĩa **3 quỹ đạo phụ**, tạo thành một tam giác hình học:
1.  **Quỹ đạo chính (Từ tâm đến Đích thật):** Vận tốc hướng về dữ liệu thật của user.
2.  **Quỹ đạo phụ 1 (Từ tâm đến Điểm neo thô):** Vận tốc hướng về ước lượng ban đầu $\alpha_l$.
3.  **Quỹ đạo phụ 2 (Đứng yên):** Vận tốc bằng 0.

**Tại sao cách này hiệu quả?**
Dựa vào tính chất bắc cầu của vectơ trong tam giác, vận tốc đi về Đích thật sẽ bằng tổng của: (Vận tốc đi đến Điểm neo thô) + (Phần chênh lệch giữa Điểm neo và Đích thật). 
Vì (Vận tốc đi đến Điểm neo thô) là một giá trị ta đã biết trước, mạng Denoise giờ đây **chỉ cần phải dự đoán phần "chênh lệch" (residual)**. Nếu điểm neo ban đầu khá sát với đích, phần chênh lệch này rất nhỏ, giúp giảm đáng kể gánh nặng tính toán và giúp mô hình hội tụ ổn định hơn.

## 4. Tóm tắt luồng xử lý
1.  **Forward:** Thêm nhiễu vào dữ liệu, hướng tâm nhiễu về phía điểm neo $\alpha_l$.
2.  **Denoise (TVS):** Mô hình dự đoán vận tốc dựa trên 3 quỹ đạo (Loss TVS) để tìm ra vận tốc chênh lệch.
3.  **Tái tạo:** Sử dụng phương trình ODE ngược để chuyển đổi vận tốc dự đoán được trở lại thành dữ liệu dự đoán $\hat{\alpha}_0$.
4.  **Inference:** Kết quả $\hat{\alpha}_0$ tiếp tục được đưa vào quy trình đánh giá Top-K và hàm loss MSI giống hệt như kiến trúc DiffMM ban đầu.
