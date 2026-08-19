# Tổng quan về Phương pháp và Thuật toán DiffMM (Mô hình Khuếch tán Đa phương thức cho Hệ thống Gợi ý)

Tài liệu này tổng hợp và mô tả thật chi tiết phương pháp đề xuất của mô hình DiffMM (Multi-Modal Diffusion Model for Recommendation - Mô hình Khuếch tán Đa phương thức dành cho Hệ thống Gợi ý). Quy trình hoạt động của thuật toán (Algorithm - một tập hợp các bước hoặc chỉ dẫn để giải quyết một vấn đề) được phân tích từng bước một cách trực quan để bạn dễ dàng nắm bắt và hình dung.

---

## 1. Đặt vấn đề & Động lực (Motivation - Lý do hoặc nguồn cảm hứng dẫn đến việc nghiên cứu)

Trong các hệ thống gợi ý đa phương thức (Multi-modal Recommendation Systems - các hệ thống gợi ý cho người dùng dựa trên nhiều loại dữ liệu khác nhau như hình ảnh, văn bản, âm thanh; ví dụ như TikTok, YouTube, Amazon), dữ liệu tương tác (Interaction - hành vi người dùng click, xem, mua hàng) giữa Người dùng (User) và Mục tiêu/Sản phẩm (Item) thường rất loãng (Sparse - rời rạc, có rất ít dữ liệu so với tổng số lượng kết hợp có thể xảy ra).

* **Hạn chế của các phương pháp tự giám sát (Self-Supervised Learning - SSL - Phương pháp học máy tự động tạo ra nhãn từ chính dữ liệu đầu vào mà không cần con người gán nhãn trước):**
  Các phương pháp này thường sử dụng kỹ thuật tăng cường ngẫu nhiên (Random Augmentation - làm phong phú dữ liệu một cách ngẫu nhiên) như: Dropout nút/cạnh đồ thị (Graph Node/Edge Dropout - ngẫu nhiên loại bỏ một số điểm hoặc đoạn nối trong cấu trúc dữ liệu mạng lưới) hoặc căn chỉnh xuyên chế độ (Cross-view Alignment - cố gắng làm cho các biểu diễn từ các loại dữ liệu khác nhau trở nên giống nhau). 
  Việc làm ngẫu nhiên này vô tình đưa thêm nhiễu (Noise - những thông tin sai lệch, không có ích) vào dữ liệu, ví dụ như những lần người dùng nhấp chuột nhầm hoặc độ thiên lệch phổ biến (Popularity Bias - xu hướng gợi ý các sản phẩm đang hot thay vì sản phẩm thực sự phù hợp với cá nhân). Điều này làm cho mô hình không liên kết tốt thông tin đa phương thức (Multi-modal information - thông tin từ nhiều nguồn: hình ảnh, chữ, tiếng) với hành vi thực tế của người dùng.

* **Giải pháp của DiffMM:** 
  Sử dụng mô hình sinh khuếch tán (Diffusion Model - một loại mô hình trí tuệ nhân tạo có khả năng học cách thêm nhiễu vào dữ liệu rồi sau đó học cách khử nhiễu để sinh ra dữ liệu mới hoàn chỉnh) để tự động sinh ra các đồ thị tương tác (Interaction Graphs - các mạng lưới thể hiện mối liên hệ) giữa Người dùng (User) và Mục tiêu (Item). Quá trình này có khả năng nhận biết đặc trưng đa phương thức (Modality-aware - hiểu được đặc điểm riêng của hình ảnh, văn bản, âm thanh), từ đó dẫn dắt việc học biểu diễn (Representation Learning - quá trình chuyển đổi dữ liệu thô thành các vector số học mà máy tính có thể hiểu được) với chất lượng cao hơn, giúp giảm thiểu nhiễu (Noise) và giải quyết triệt để vấn đề dữ liệu bị loãng (Sparse).

---

## 2. Kiến trúc tổng quan của DiffMM (Architecture - Cấu trúc tổng thể của hệ thống)

Kiến trúc (Architecture) của DiffMM gồm 3 phần chính, được huấn luyện đa nhiệm (Multi-task Training - quá trình học máy dạy cho mô hình làm nhiều việc cùng một lúc):

1. **Multi-Modal Graph Diffusion Model (MMGDM - Mô hình Khuếch tán Đồ thị Đa phương thức):** 
   Sử dụng cơ chế khuếch tán (Diffusion Mechanism - quá trình thêm và xóa nhiễu) để tạo ra các đồ thị tương tác (Interaction Graphs) cụ thể cho từng loại dữ liệu (Modality - phương thức/loại dữ liệu, ví dụ: Modality hình ảnh, Modality văn bản, Modality âm thanh).

2. **Cross-Modal Contrastive Augmentation (Tăng cường Tương phản Xuyên phương thức):** 
   Giúp tối đa hóa sự đồng nhất (Alignment - sự khớp nhau) về thông tin sở thích của người dùng giữa các loại dữ liệu khác nhau. Quá trình này sử dụng hàm mất mát tương phản (Contrastive Loss - một công thức toán học giúp đẩy các dữ liệu giống nhau lại gần nhau và kéo các dữ liệu khác nhau ra xa nhau trong không gian biểu diễn).

3. **Multi-Modal Graph Aggregation (Tổng hợp Đồ thị Đa phương thức):** 
   Gom nhóm và kết hợp các biểu diễn (Representations - các vector số học) từ các đồ thị phương thức khác nhau vào mạng GCN chính (Graph Convolutional Network - Mạng nơ-ron tích chập đồ thị, dùng để xử lý dữ liệu dạng mạng lưới) để đưa ra dự đoán tương tác cuối cùng (Prediction - kết quả dự đoán xem người dùng có thích sản phẩm hay không).

---

## 3. Quy trình Thuật toán Chi tiết (Detailed Algorithm Process)

### Bước 1: Quá trình Khuếch tán Đồ thị Thuận (Forward Graph Diffusion - Quá trình phá hủy dữ liệu bằng cách thêm nhiễu)

**Mô tả:** Quá trình này lấy dữ liệu gốc của người dùng và từ từ làm hỏng nó bằng cách thêm vào những tín hiệu nhiễu ngẫu nhiên qua nhiều bước.

**Giải thích Ký hiệu và Công thức:**
* $u$: Ký hiệu đại diện cho một Người dùng (User) cụ thể.
* $I$: Tập hợp chứa toàn bộ các Mục tiêu/Sản phẩm (Items) có trong hệ thống. $|I|$ là tổng số lượng các sản phẩm đó.
* $\mathbf{a}\sb u = [a\sb {u,0}, a\sb {u,1}, \dots, a\sb {u,|I|-1}]$: Đây là một vector (một mảng các con số) biểu diễn lịch sử tương tác của người dùng $u$. Nếu $a\sb {u,i} = 1$, nghĩa là người dùng $u$ đã từng tương tác với sản phẩm $i$. Nếu bằng $0$, nghĩa là chưa tương tác.
* $\alpha\sb 0 = \mathbf{a}\sb u$: $\alpha$ (đọc là alpha) biểu diễn trạng thái của vector tại một thời điểm. $\alpha\sb 0$ là trạng thái ở thời điểm ban đầu (bước 0), tức là dữ liệu gốc nguyên bản, chưa có nhiễu.
* $T$: Tổng số bước thời gian (Time steps) trong quá trình thêm nhiễu.
* $\alpha\sb t$: Trạng thái của vector tương tác tại bước thời gian thứ $t$ (đã bị thêm một lượng nhiễu nhất định).

**Công thức thêm nhiễu từng bước:**
$$q(\alpha\sb t | \alpha\sb {t-1}) = \mathcal{N}(\alpha\sb t; \sqrt{1-\beta\sb t}\alpha\sb {t-1}, \beta\sb t \mathbf{I})$$
* **Ý nghĩa công thức:** Xác suất (ký hiệu $q$) để đạt được trạng thái $\alpha\sb t$ nếu ta đã biết trạng thái trước đó $\alpha\sb {t-1}$ tuân theo một Phân phối Chuẩn (Gauss) ký hiệu là $\mathcal{N}$ (Normal Distribution - Phân phối hình chuông ngẫu nhiên).
* $\beta\sb t$ (đọc là beta): Mức độ nhiễu (Variance - phương sai) được thêm vào tại bước $t$.
* $\sqrt{1-\beta\sb t}\alpha\sb {t-1}$: Đây là giá trị trung bình (Mean) của phân phối, cho thấy trạng thái hiện tại vẫn giữ lại một phần thông tin từ trạng thái trước đó, nhưng bị thu hẹp lại một chút (nhân với phần căn bậc hai).
* $\mathbf{I}$: Ma trận đơn vị (Identity Matrix), kết hợp với $\beta\sb t$ đại diện cho mức độ nhiễu ngẫu nhiên được bơm vào.

**Công thức tính gộp nhiễu trực tiếp từ ban đầu:**
Nhờ tính chất toán học của phân phối chuẩn, thay vì tính từng bước, ta có thể nhảy cóc từ $\alpha\sb 0$ lên $\alpha\sb t$:
$$\alpha\sb t = \sqrt{\bar{\gamma}\sb t}\alpha\sb 0 + \sqrt{1 - \bar{\gamma}\sb t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
* $\gamma\sb t = 1 - \beta\sb t$ (đọc là gamma): Lượng thông tin gốc còn được giữ lại ở mỗi bước.
* $\bar{\gamma}\sb t = \prod\sb {i=1}^t \gamma\sb i$: Ký hiệu $\prod$ (Pi) nghĩa là tích (nhân tất cả lại với nhau). $\bar{\gamma}\sb t$ là tổng lượng thông tin gốc còn sót lại sau $t$ bước.
* $\epsilon$ (đọc là epsilon): Biến ngẫu nhiên đại diện cho tín hiệu nhiễu thuần túy (Noise), được lấy mẫu từ phân phối chuẩn $\mathcal{N}(0, \mathbf{I})$ (trung bình bằng 0, phương sai bằng 1).
* **Ý nghĩa công thức:** Vector ở bước $t$ ($\alpha\sb t$) đơn giản là sự pha trộn giữa dữ liệu gốc ($\alpha\sb 0$) và một lượng nhiễu ngẫu nhiên ($\epsilon$). Càng về sau (khi $t$ lớn), phần gốc càng mờ nhạt và phần nhiễu càng lớn.

---

### Bước 2: Quá trình Khôi phục Đồ thị Ngược (Reverse Graph Denoising - Quá trình khử nhiễu để tái tạo dữ liệu)

**Mô tả:** Mô hình học cách đi lùi, tức là từ trạng thái nhiễu $\alpha\sb t$, nó cố gắng đoán xem dữ liệu gốc $\alpha\sb 0$ trông như thế nào.

**Giải thích Ký hiệu và Công thức:**
* $\hat{\alpha}\sb \theta(\alpha\sb t, t)$: Một hàm toán học được biểu diễn bởi một mạng nơ-ron nhân tạo (MLP - Multi-Layer Perceptron), với các tham số (thông số cấu hình) là $\theta$ (theta). Ký hiệu mũ $\hat{}$ (hat) thể hiện đây là một giá trị *được dự đoán*, không phải giá trị thực tế. Mạng này nhận đầu vào là trạng thái bị nhiễu $\alpha\sb t$ và thời điểm $t$, để dự đoán ra dữ liệu gốc $\alpha\sb 0$.

**Hàm mất mát chính (ELBO Loss):**
$$\mathcal{L}\sb {elbo} = \mathbb{E}\sb {t \sim U(1, T), q(\alpha\sb 0)} \left[ \| \hat{\alpha}\sb \theta(\alpha\sb t, t) - \alpha\sb 0 \|\sb 2^2 \right]$$
* $\mathcal{L}$ (Loss): Hàm mất mát (thước đo mức độ sai sót của mô hình, càng nhỏ càng tốt).
* $\mathbb{E}$ (Expectation): Kỳ vọng (giá trị trung bình toán học) trên tất cả các bước thời gian $t$ và các dữ liệu gốc $\alpha\sb 0$.
* $\| x \|\sb 2^2$: Ký hiệu chuẩn bậc 2 bình phương (L2 Norm squared). Nó chỉ đơn giản là phép tính tổng bình phương khoảng cách giữa hai giá trị.
* **Ý nghĩa công thức:** Mô hình cố gắng điều chỉnh để khoảng cách (sự khác biệt) giữa kết quả nó dự đoán ($\hat{\alpha}\sb \theta(\alpha\sb t, t)$) và dữ liệu gốc thực tế ($\alpha\sb 0$) là nhỏ nhất có thể. Càng giống gốc càng tốt.

**Cơ chế Tiêm Tín Hiệu Nhận Biết Phương Thức (Modality-aware Signal Injection - MSI):**
Khác với các hệ thống thường, mô hình này muốn dữ liệu khôi phục không chỉ đúng mà còn phải mang dấu ấn của loại phương tiện (Modality - ví dụ: âm thanh, hình ảnh).

$$\mathcal{L}\sb {msi}^m = \| \hat{\alpha}\sb 0 \cdot \mathbf{e}\sb i^m - \alpha\sb 0 \cdot \mathbf{e}\sb i \|\sb 2^2$$
* $m$: Ký hiệu cho loại phương thức (Modality), ví dụ $m$ có thể là chữ, $m$ có thể là hình.
* $\mathbf{e}\sb i$: Vector biểu diễn danh tính (ID Embedding - một chuỗi số đại diện duy nhất cho mục tiêu $i$).
* $\mathbf{e}\sb i^m$: Vector biểu diễn đặc trưng đa phương thức (Modality Feature Embedding - ví dụ: một chuỗi số miêu tả đặc điểm hình ảnh của mục tiêu $i$).
* $\hat{\alpha}\sb 0$: Tương tác dự đoán (dữ liệu mô hình sinh ra).
* $\alpha\sb 0$: Tương tác thực tế.
* $\cdot$: Phép nhân.
* **Ý nghĩa công thức:** Ta muốn sản phẩm (phép nhân) giữa tương tác dự đoán ($\hat{\alpha}\sb 0$) kết hợp với đặc trưng phương thức ($\mathbf{e}\sb i^m$) phải giống nhất có thể với sản phẩm của tương tác thực tế ($\alpha\sb 0$) kết hợp với danh tính người dùng gốc ($\mathbf{e}\sb i$). Điều này "bắt ép" mô hình khi sinh ra dữ liệu phải lồng ghép kiến thức về hình ảnh/âm thanh vào kết quả.

**Hàm mất mát tổng hợp:**
$$\mathcal{L}\sb {dm}^m = \mathcal{L}\sb {elbo} + \lambda\sb 0 \mathcal{L}\sb {msi}^m$$
* $\lambda\sb 0$ (đọc là lambda không): Một hằng số (con số cố định) đóng vai trò như núm vặn âm lượng, quyết định xem thành phần MSI ($\mathcal{L}\sb {msi}^m$) quan trọng đến mức nào so với tổng thể.
* **Ý nghĩa công thức:** Tổng sai sót ($\mathcal{L}\sb {dm}^m$) bằng sai sót trong quá trình khôi phục gốc cộng với sai sót trong việc ghép thông tin đa phương thức.

---

### Bước 3: Học Tương phản Đa Phương thức (Cross-Modal Contrastive Learning)

**Mô tả:** Ở bước này, mô hình đối chiếu các kết quả từ các nguồn khác nhau để đảm bảo sự thống nhất (ví dụ: sở thích của người dùng trên phương diện hình ảnh cũng phải có sự tương đồng với sở thích trên phương diện văn bản).

**Giải thích Ký hiệu và Công thức:**
* $\mathcal{A}^m$: Đồ thị tương tác (Interaction Graph - mạng lưới kết nối giữa người dùng và sản phẩm) sau khi đã được làm sạch nhiễu cho loại phương thức $m$.
* $\mathcal{N}\sb u^m$: Tập hợp những người hàng xóm (Neighbor) của người dùng $u$ trên đồ thị $\mathcal{A}^m$.
* $\mathbf{z}\sb u^m$: Vector đại diện cho sở thích của người dùng $u$ sau khi gom thông tin từ các hàng xóm.

**Lan truyền tin nhắn (Message Passing):**
$$\mathbf{z}\sb u^m = \sum\sb {i \in \mathcal{N}\sb u^m} \frac{1}{\sqrt{|\mathcal{N}\sb u^m| |\mathcal{N}\sb i^m|}} \mathbf{e}\sb i^m$$
* $\sum$ (Sigma): Phép tính tổng. Ta cộng tất cả các thông tin từ các mục tiêu $i$ mà người dùng $u$ kết nối.
* $\frac{1}{\sqrt{|\mathcal{N}\sb u^m| |\mathcal{N}\sb i^m|}}$: Một hệ số chuẩn hóa (Normalization factor - giúp các con số không bị quá lớn), tính dựa trên số lượng hàng xóm của $u$ và $i$.
* **Ý nghĩa công thức:** Sở thích của người dùng được xác định bằng cách cộng gộp trung bình các đặc điểm ($\mathbf{e}\sb i^m$) của những sản phẩm mà họ (hoặc những người giống họ) đã tương tác trên đồ thị mới sinh ra. Vector sau đó được truyền tiếp qua mạng để thành kết quả cuối cùng $\mathbf{\bar{z}}\sb u^m$ (ký hiệu có thanh ngang trên đầu biểu thị giá trị cuối cùng).

**Hàm mất mát tương phản (Cross-Modal Contrastive Loss - InfoNCE Loss):**
$$\mathcal{L}\sb {cl}^{user} = \sum\sb {m\sb 1 \in \mathcal{M}} \sum\sb {m\sb 2 \in \mathcal{M}, m\sb 2 \neq m\sb 1} \sum\sb {u \in \mathcal{U}} -\log \frac{\exp(s(\mathbf{\bar{z}}\sb u^{m\sb 1}, \mathbf{\bar{z}}\sb u^{m\sb 2})/\tau)}{\sum\sb {v \in \mathcal{U}} \exp(s(\mathbf{\bar{z}}\sb u^{m\sb 1}, \mathbf{\bar{z}}\sb v^{m\sb 2})/\tau)}$$
* $m\sb 1, m\sb 2$: Hai loại phương thức khác nhau (ví dụ $m\sb 1$ là hình ảnh, $m\sb 2$ là âm thanh).
* $\mathcal{U}$: Tập hợp tất cả người dùng (Users), $u$ là người dùng đang xét, $v$ là một người dùng bất kỳ khác để so sánh.
* $s(a, b)$: Hàm tính độ tương đồng (Similarity score - đo xem hai vector $a$ và $b$ giống nhau mức nào).
* $\exp()$: Hàm mũ (Exponential function $e^x$, giúp khuếch đại sự khác biệt).
* $\tau$ (Tau): Tham số nhiệt độ (Temperature parameter - một con số nhỏ giúp tinh chỉnh độ nhạy của phép chia).
* $\log$: Hàm logarit.
* **Ý nghĩa công thức:** Tử số tính độ giống nhau về sở thích của cùng **một người dùng $u$** nhưng trên 2 phương diện khác nhau ($m\sb 1$ và $m\sb 2$). Mẫu số tính độ giống nhau giữa người dùng $u$ trên phương diện $m\sb 1$ với **tất cả người dùng khác $v$** trên phương diện $m\sb 2$. Dấu trừ $-\log$ phía trước biến tỷ lệ này thành sai sót (Loss). Mô hình sẽ phải giảm thiểu Loss này, tức là ép buộc: Tỷ lệ (Sự giống nhau của chính mình / Sự giống nhau với người khác) phải càng lớn càng tốt. Tóm lại, biểu diễn của một người trên góc nhìn hình ảnh phải cực kỳ giống biểu diễn của chính người đó trên góc nhìn âm thanh, và khác biệt với người khác.

---

### Bước 4: Tích hợp Đồ thị Đa Phương thức & Dự đoán (Aggregation & Prediction)

**Mô tả:** Gom tất cả kiến thức học được từ các loại phương thức (hình, chữ, âm thanh) lại làm một để đưa ra kết luận người dùng có mua/click vào sản phẩm không.

**Công thức tích hợp (Aggregation):**
$$\mathbf{h}\sb u = \sum\sb {m \in \mathcal{M}} \kappa\sb m \mathbf{\hat{z}}\sb u^m, \quad \mathbf{h}\sb i = \sum\sb {m \in \mathcal{M}} \kappa\sb m \mathbf{\hat{z}}\sb i^m$$
* $\mathbf{h}\sb u$, $\mathbf{h}\sb i$: Biểu diễn tích hợp tổng thể (Tổng hợp tất cả phương thức) của User $u$ và Item $i$.
* $\kappa\sb m$ (Kappa): Một trọng số (Weight) thể hiện mức độ quan trọng của phương thức $m$ (ví dụ: với sản phẩm này, hình ảnh quan trọng hơn chữ thì $\kappa\sb {hinhanh} > \kappa\sb {chu}$). Máy tính sẽ tự động học (Learning) giá trị này.
* **Ý nghĩa công thức:** Biểu diễn cuối cùng của User/Item bằng tổng hợp các biểu diễn từ mỗi phương thức, nhân với độ quan trọng của phương thức đó. Sau đó, nó đi qua mạng lưới (GCN) một lần nữa để ra bản chốt $\mathbf{\bar{h}}\sb u$ và $\mathbf{\bar{h}}\sb i$.

**Dự đoán (Prediction):**
$$\hat{y}\sb {u,i} = \mathbf{\bar{h}}\sb u^T \cdot \mathbf{\bar{h}}\sb i$$
* $\hat{y}\sb {u,i}$ (y mũ): Điểm số dự đoán cuối cùng (Predictive score) xem người dùng $u$ có khả năng tương tác với item $i$ cao hay thấp.
* $\mathbf{\bar{h}}\sb u^T$: Ký hiệu $T$ ở trên cùng (Transpose - chuyển vị) dùng để đổi vector thành hàng ngang để có thể nhân ma trận.
* $\cdot$: Phép nhân vô hướng (Dot product).
* **Ý nghĩa công thức:** Điểm dự đoán bằng sự tương đồng (tích vô hướng) giữa vector sở thích tổng hợp của người dùng ($\mathbf{\bar{h}}\sb u$) và vector đặc điểm tổng hợp của sản phẩm ($\mathbf{\bar{h}}\sb i$). Điểm số càng cao, khả năng gợi ý sản phẩm đó thành công càng lớn.

---

## 4. Giải mã quy trình chạy thuật toán (Pseudo-Code - Mã giả / Cách viết mô phỏng lại code lập trình)

### Thuật toán 1: Quá trình Huấn luyện Bộ Khuếch tán (MMGDM Training - Dạy cho mô hình cách sinh dữ liệu)
```python
# Đầu vào (Input - những gì ta cung cấp cho mô hình): 
# - Đồ thị tương tác gốc α0 (alpha 0 - dữ liệu hành vi chưa bị nhiễu)
# - Đặc trưng đa phương thức của các sản phẩm Eᵐ (E m - ví dụ: các vector hình ảnh, âm thanh)
# Đầu ra (Output - cái ta thu được): Trọng lượng mạng MLP θ (theta) tối ưu (Tức là "bộ não" của mô hình đã được tinh chỉnh tốt nhất)

Lặp lại (Repeat) các bước sau cho đến khi mô hình học xong (Hội tụ - Converge, không thể tốt hơn được nữa):
    1. Lấy ra một nhóm (Batch) dữ liệu tương tác của người dùng α0.
    2. Chọn ngẫu nhiên một thời điểm t từ khoảng {1, 2, ..., T} (T là tổng số bước).
    3. Tạo ra tín hiệu nhiễu ngẫu nhiên ε (epsilon) tuân theo phân phối chuẩn N(0, I).
    4. Tính toán trạng thái dữ liệu tại thời điểm t (khi đã bị nhiễu) gọi là αt:
       αt = căn_bậc_hai(γ̄t) * α0 + căn_bậc_hai(1 - γ̄t) * ε
       (Trộn một phần dữ liệu gốc và một phần nhiễu)
    5. Yêu cầu mô hình (Mạng nơ-ron MLP) đoán thử xem dữ liệu gốc là gì dựa trên αt:
       α̂0 (alpha 0 mũ - dự đoán) = MLP(αt, thời_điểm_t)
    6. Tính toán Loss gốc (L_elbo - Sai số khôi phục):
       L_elbo = Trung bình bình phương sự khác biệt giữa α̂0 (dự đoán) và α0 (thực tế).
    7. Tính toán Loss đa phương thức (L_msi - Sai số tiêm thông tin) để nhồi kiến thức vào:
       L_msi = Bình phương sự khác biệt giữa (α̂0 nhân Eᵐ) và (α0 nhân vector danh tính).
    8. Tính Loss tổng (Tổng hợp các sai số): 
       L_dm = L_elbo + (Hằng số λ0 * L_msi)
    9. Thực hiện Thuật toán Giảm dần Độ dốc (Gradient Descent - một thuật toán toán học giúp cập nhật lại các tham số θ của mô hình sao cho Loss tổng giảm đi ở vòng lặp sau).
```

### Thuật toán 2: Sinh Đồ thị Đa Phương thức trong Pha Suy diễn (MMGDM Inference - Đem mô hình ra sử dụng thực tế)
```python
# Đầu vào (Input): 
# - Tương tác gốc α0.
# - Mô hình MLP đã học xong ở Thuật toán 1 (có tham số θ).
# - T' (T phẩy) là số bước nhiễu phụ (ít hơn quá trình học T).
# Đầu ra (Output): Đồ thị tương tác mới hoàn toàn Aᵐ (A m - đã được lồng ghép nhận biết hình ảnh/âm thanh).

1. Sinh ra một lượng nhiễu ngẫu nhiên ε (epsilon) ~ N(0, I).
2. Chạy quá trình thuận (Forward) đập vỡ dữ liệu gốc ngay lập tức đến bước T':
   α_T' = căn_bậc_hai(γ̄_T') * α0 + căn_bậc_hai(1 - γ̄_T') * ε
3. Đặt trạng thái ban đầu để bắt đầu khôi phục là α̂_T = α_T'.
4. Bắt đầu vòng lặp đi lùi (khử nhiễu ngược - Reverse denoising) từ t = T' lùi dần về 1:
   a. Dùng mô hình MLP dự đoán dữ liệu gốc sạch nhiễu:
      α̂0 (dự đoán) = MLP(α̂_t, t)
   b. Dùng các công thức toán học xác suất để tính ngược lại trạng thái trước đó (t-1):
      α̂_{t-1} = Hằng_Số_1 * α̂_t + Hằng_Số_2 * α̂0  (Quy trình khử nhiễu để tiến sát về gốc)
5. Xây dựng lại (Rebuild) đồ thị mới:
   Với mỗi Người dùng (User u):
      - Sắp xếp các giá trị trong vector kết quả α̂0 cuối cùng từ lớn đến bé.
      - Chọn ra Top-k (k sản phẩm đứng đầu danh sách) mà người dùng chưa từng tương tác.
      - Coi đây là các mối quan hệ (Cạnh đồ thị) mới và vẽ chúng vào mạng lưới.
   Trả về đồ thị mới tinh Aᵐ dành riêng cho loại phương thức m.
```

---

## 5. Kết luận (Conclusion)
DiffMM kết hợp khéo léo sức mạnh của **Mô hình Sinh (Generative Model - Cụ thể là Diffusion - Khuếch tán, khả năng học cách phục hồi từ đống đổ nát ngẫu nhiên để sinh ra dữ liệu mới phong phú)** nhằm giải quyết tình trạng thiếu hụt dữ liệu (Sparsity) trong gợi ý. Đồng thời nó cũng khai thác khả năng liên kết mạnh mẽ của **Học tương phản (Contrastive Learning - ép các góc nhìn khác nhau của cùng một sự vật phải đồng nhất)**. Bằng cách sinh ra các đồ thị tương tác riêng biệt nhưng liên kết chặt chẽ cho từng phương thức như hình ảnh, văn bản, âm thanh và kết nối chúng lại với nhau, DiffMM trở nên vượt trội và thông minh hơn nhiều so với các mô hình GNN (Graph Neural Networks - Mạng nơ ron đồ thị) truyền thống, giúp mang lại trải nghiệm cá nhân hóa sâu sắc và chính xác nhất cho người dùng.
