# Hướng dẫn xây dựng 1 folder "Phương án" mới chạy tốt trên Colab (không cần sửa)

> Đây là quy trình chuẩn, đúc kết từ toàn bộ quá trình xây dựng và **gỡ lỗi thật** cho
> `phuong_an_1_OT_noise_scheduler/` (DiffMM + Flow Matching). Mỗi bước dưới đây tồn tại vì đã có
> **ít nhất 1 lỗi thật** xảy ra khi bỏ qua bước đó — xem mục 4 để đối chiếu từng bài học với lỗi thực
> tế đã gặp. Làm đúng và đủ các bước này thì folder mới sẽ chạy được trên Colab **ngay lần đầu**, thay
> vì phải sửa qua lại nhiều vòng như lần đầu làm `phuong_an_1_OT_noise_scheduler/`.

**Input cần có trước khi bắt đầu:**
1. Một file kế hoạch tối ưu (dạng `*_Optimization_Plan.md`) mô tả rõ: thuật toán/paper gốc, phương án
   cụ thể cần triển khai, vị trí áp dụng, công thức thay đổi, cơ sở lý thuyết.
2. (Tuỳ chọn) Các file review paper gốc (`*_Review.md`) nếu phương án dựa trên ý tưởng từ paper khác.
3. Folder này (`Folder_Base/`) làm khuôn mẫu cấu trúc + quy trình.

---

## 0. Mục đích của `Folder_Base`

Cho phép tạo một folder "Phương án X" mới — hoàn chỉnh, tự chạy được trên Google Colab bằng cách chỉ
dán 2-3 link (repo GitHub + Google Drive) — **mà không cần vòng lặp debug qua lại với người dùng**.
Điều này chỉ khả thi nếu quy trình bên dưới được làm đầy đủ, đặc biệt là mục 2 (Bước 4: Kiểm chứng)
— đây là phần hay bị bỏ qua nhất nhưng lại quan trọng nhất.

---

## 1. Cấu trúc chuẩn

```
<ten_folder_phuong_an>/
├── README.md                              # Hướng dẫn dùng folder này (mẫu: xem Folder_Base/README.md)
├── <TenRepo>-<TenPhuongAn>/                # Bản fork ĐẦY ĐỦ của codebase gốc, patch áp THẲNG vào code
│   ├── .gitignore                          # Bỏ qua data nặng, __pycache__, *.pyc
│   ├── README.md                           # Giải thích patch: file/hàm nào đổi, đối chiếu công thức
│   ├── <toàn bộ source code gốc>           # KHÔNG đoán — tải thật từ repo GitHub gốc (xem Bước 1)
│   └── ... (chỉ sửa đúng phần patch chỉ định, giữ nguyên mọi thứ khác)
└── <Ten>_Colab.ipynb                       # Notebook chạy trên Colab (7 cell chuẩn, xem Bước 6)
```

**Quy tắc đặt vị trí `<TenRepo>-<TenPhuongAn>/` — RẤT QUAN TRỌNG:**
Luôn tạo git repo độc lập cho `<TenRepo>-<TenPhuongAn>/` **ở một vị trí NẰM NGOÀI mọi git repo khác
đang tồn tại** trên máy người dùng (kiểm tra bằng `git rev-parse --show-toplevel` trước khi `git init`).
Nếu tạo `.git` bên trong một thư mục đã bị một git repo cha theo dõi, sẽ xảy ra tình trạng "repo lồng
nhau" — khi push lên GitHub, Main script sẽ không nằm ở gốc repo mà bị lồng thêm 1-2 cấp thư mục, gây
lỗi clone không tìm thấy entry point (đã xảy ra thật ở `phuong_an_1_OT_noise_scheduler`). Vị trí an
toàn: một thư mục tạm (scratchpad) rồi copy ra một đường dẫn **ngoài** thư mục dự án chính đang được
git theo dõi (ví dụ thư mục cha của thư mục dự án).

---

## 2. Quy trình từng bước

### Bước 0 — Đọc kế hoạch

Đọc kỹ file `*_Optimization_Plan.md` được đưa vào. Xác định chính xác:
- Tên + link GitHub repo chính thức của thuật toán/paper gốc.
- Phương án cụ thể cần triển khai (số công thức, file/hàm/class dự kiến bị ảnh hưởng).
- Cơ sở lý thuyết (paper nào, công thức nào) — để viết lại đối chiếu công thức gốc ↔ công thức mới
  trong README của bản fork.

### Bước 1 — Tải source code gốc THẬT (không đoán)

**Không bao giờ tự đoán tên biến/hàm/class dựa trên trí nhớ hay suy luận từ paper.** Luôn:

```bash
curl -s https://api.github.com/repos/<owner>/<repo>/contents/ | python -c "import json,sys; [print(x['type'], x['name']) for x in json.load(sys.stdin)]"
curl -s https://raw.githubusercontent.com/<owner>/<repo>/main/<file>.py
```

Đọc toàn bộ các file có khả năng liên quan đến phương án (không chỉ file "chính" — kiểm tra cả
`DataHandler`/`utils`/`Params`/entry point chính, vì patch thường cần sửa nhiều hơn 1 file: file định
nghĩa tham số dòng lệnh, file chứa class/hàm cần patch, và file khởi tạo/gọi class đó).

Đọc luôn `README.md` gốc của repo để biết: lệnh chạy mẫu cho từng dataset/cấu hình, cấu trúc thư mục
dữ liệu mong đợi, và file entry point chính (không phải lúc nào cũng tên `Main.py`).

### Bước 2 — Thiết kế patch tối thiểu

Nguyên tắc: **thêm, không sửa đè** khi có thể (ví dụ thêm 1 class mới kế thừa từ class gốc, override
đúng số hàm cần thiết, thay vì viết lại toàn bộ class gốc). Điều này giúp:
- Người đọc dễ đối chiếu code cũ/mới.
- Giảm rủi ro phá vỡ các phần không liên quan.

Với mỗi hàm định override, tự hỏi: "Có ràng buộc toán học nào của code gốc mà công thức mới không còn
thoả mãn không?" (bài học từ Phương án 1: code DDPM gốc ràng buộc `mu² + sigma² = 1`, đường OT không
còn ràng buộc này, nên không thể chỉ đổi 1 hàm — phải đổi cả `q_sample`, `p_mean_variance`, và `SNR`).

### Bước 3 — Áp patch thật + viết code

Sửa trực tiếp vào bản sao đã tải ở Bước 1 (không phải string-replace lúc chạy runtime trong notebook
— bài học: cách đó khó debug, khó review, và không tận dụng được git history). Viết comment ngắn gọn
tại mọi chỗ sửa, giải thích *vì sao* (không phải *cái gì* — code đã tự nói cái gì).

### Bước 4 — Kiểm chứng (BẮT BUỘC, không được bỏ qua bước nào)

Đây là bước quan trọng nhất để đảm bảo "chạy tốt trên Colab mà không phải sửa". Thực hiện **trước khi
đóng gói**, dùng Bash cục bộ (không cần GPU):

1. **Biên dịch sạch:** `python -m py_compile <mọi file .py bị sửa>` — bắt lỗi cú pháp ngay.
2. **Dry-run áp patch thật:** tải lại bản MỚI NHẤT của các file gốc từ GitHub (không dùng bản đã tải ở
   Bước 1 nếu đã qua nhiều giờ — repo gốc có thể đã đổi), áp patch lên bản mới này, biên dịch lại. Đảm
   bảo patch không phụ thuộc vào giả định sai về cấu trúc file.
3. **Kiểm tra số học độc lập (nếu patch có công thức toán mới):** viết 1 script Python độc lập, monkey-
   patch `.cuda()` thành no-op (vì máy dev không có GPU), test:
   - Điều kiện biên đúng như thiết kế (ví dụ t=0 và t=T-1 cho ra đúng giá trị mong đợi).
   - Chạy thử với 1 "denoiser hoàn hảo" giả lập (hoặc tương đương tuỳ bài toán) để xác nhận thuật toán
     suy luận hội tụ đúng về đáp án đã biết trước.
   - **Quét toàn bộ dải giá trị tham số hợp lệ** (không chỉ 1-2 giá trị ngẫu nhiên) để chắc chắn không
     có NaN/Inf/giá trị bất thường ở bất kỳ điểm biên nào — bài học thật: công thức SNR gốc gây chia
     cho 0 khi áp trực tiếp cho đường OT, chỉ phát hiện được nhờ quét toàn bộ dải t chứ không phải chỉ
     test 1 giá trị t ngẫu nhiên.
4. **Dry-run toàn bộ notebook, cell theo cell**, bằng cách extract source từng code cell ra file `.py`
   riêng, thay các dòng `!shell-command`/`%magic` bằng `pass` (để `ast.parse` qua được) hoặc bằng
   `subprocess.run(...)` thật (để test hành vi thật), rồi `exec()` từng cell theo đúng thứ tự, dùng
   biến giả lập (fake repo, fake log file, fake script in ra đúng định dạng output mà cell sau cần
   parse) để mô phỏng **cả nhánh thành công lẫn nhánh lỗi**. Với nhánh lỗi, xác nhận thông báo lỗi in
   ra đủ rõ ràng để người dùng tự sửa được mà không cần hỏi lại.

Chỉ báo "đã kiểm chứng" khi cả 4 mục trên đều pass thật (không chỉ đọc code rồi tự tin là đúng).

### Bước 5 — Đóng gói thành git repo độc lập

```bash
# Tạo ở vị trí NGOÀI mọi git repo hiện có (xem quy tắc ở mục 1)
cp -r <ban_da_patch> <vi_tri_ngoai_repo>/<TenRepo>-<TenPhuongAn>
cd <vi_tri_ngoai_repo>/<TenRepo>-<TenPhuongAn>
git rev-parse --show-toplevel   # PHẢI in ra đúng thư mục này -- nếu không, dừng lại, chọn vị trí khác
rm -rf __pycache__ **/__pycache__
git init -q
git add -A
git status --short              # soát lại danh sách file trước khi commit
git commit -q -m "<mo ta patch>"
```

Viết README riêng cho bản fork này (`<TenRepo>-<TenPhuongAn>/README.md`): nêu rõ file/hàm bị sửa, đối
chiếu công thức gốc ↔ mới, cách chạy cục bộ, và mục "Đã kiểm chứng" tóm tắt kết quả Bước 4.

**Không tự động `git push`** trừ khi người dùng đã xác nhận rõ ràng cho phép (đây là hành động ảnh
hưởng tài khoản/repo GitHub của người dùng). Nếu người dùng đã đưa link 1 repo GitHub trống mới tạo và
yêu cầu commit/push, thì được phép `git remote add` + `git push` trực tiếp.

### Bước 6 — Tạo notebook Colab từ template

Dùng `Folder_Base/Colab_Template.ipynb` làm khuôn — copy sang `<ten_folder>/<Ten>_Colab.ipynb` rồi
điền các chỗ đánh dấu `# TODO(...)`. **Không viết lại từ đầu** — mọi phần hạ tầng (infra) trong
template đã được kiểm chứng kỹ và xử lý sẵn các lỗi môi trường hay gặp (xem mục 3). Chỉ cần điền:

- Tên/đường dẫn entry point script thật (không phải lúc nào cũng là `Main.py`).
- Danh sách file dữ liệu bắt buộc phải có (`REQUIRED_DATA_FILES`).
- Bộ hyperparameter khuyến nghị theo từng dataset/cấu hình (`DATASET_HP`, lấy từ README gốc — Bước 1).
- `DATASET_NAME` ở Cell 1 (tên dataset dùng để chạy — nếu thuật toán chỉ hỗ trợ 1 dataset thì để cố
  định giá trị đó).
- Định dạng dòng log cuối cùng chứa kết quả tốt nhất, để viết đúng regex parse ở Cell 7 (chạy thử
  script thật hoặc đọc kỹ code phần in kết quả để biết chính xác định dạng, không đoán) — và map đúng
  các cột của `result_df` (bao gồm cột `"Dataset"` và cột `"Phương án"`, vì phần xuất PDF ở cuối Cell 7
  tự động lấy tên phương án/dataset từ 2 cột này, không cần sửa gì thêm ở phần xuất PDF).
- Các thư viện phụ trợ cần `pip install` thêm ở Cell 2 (kiểm tra `import` ở đầu mỗi file gốc —
  `matplotlib` dùng để xuất PDF ở Cell 7 đã có sẵn trên Colab, không cần thêm vào danh sách cài đặt).

**Tính năng có sẵn, không cần code lại:** cuối Cell 7, notebook tự vẽ `result_df` (bảng chỉ số cao
nhất) kèm tên dataset + tên phương án thành 1 trang PDF (`matplotlib`, không cần cài thêm thư viện),
lưu file `KetQua_<dataset>.pdf`, và tự động tải file đó về máy qua trình duyệt (`google.colab.files.download`)
ngay khi Cell 7 chạy xong. Chỉ hoạt động đúng khi Cell 5 (map kết quả) đã điền đủ 2 cột `"Dataset"` và
`"Phương án"` trong `result_df` — nếu thiếu, PDF vẫn xuất được nhưng thiếu thông tin trong tiêu đề.

Sau khi điền xong, **chạy lại toàn bộ Bước 4 mục 4 (dry-run cell theo cell)** trên bản notebook cuối
cùng trước khi bàn giao — với Cell 7, dry-run luôn cả phần xuất PDF (kiểm tra file `.pdf` thật sự được
tạo ra, không rỗng) chứ không chỉ phần in bảng.

### Bước 7 — Điền sẵn Cell 1 nếu người dùng đã có link

Nếu người dùng đã cung cấp link GitHub repo + link Google Drive, điền thẳng vào Cell 1 của notebook
(không để trống bắt người dùng tự dán) — mục tiêu là "chạy được ngay, không cần sửa gì".

### Bước 8 — Checklist bàn giao cuối cùng

Trước khi báo "xong" với người dùng, xác nhận đã làm **toàn bộ** các mục sau (không chỉ một phần):

- [ ] Đã tải source gốc thật từ GitHub, không đoán tên biến/hàm.
- [ ] Patch biên dịch sạch (`py_compile`) trên bản mới nhất tải lại từ GitHub.
- [ ] Nếu có công thức toán: đã dry-run số học, quét hết dải tham số, xác nhận không NaN/Inf.
- [ ] Đã dry-run toàn bộ notebook cell-theo-cell, cả nhánh thành công lẫn nhánh lỗi.
- [ ] Cell 7 đã dry-run cả phần xuất PDF — xác nhận file `.pdf` được tạo ra thật, không rỗng, tiêu đề
      có đúng tên phương án + tên dataset (không phải giá trị mặc định/rỗng).
- [ ] Git repo độc lập được tạo ở vị trí **ngoài** mọi repo khác (đã tự kiểm bằng
      `git rev-parse --show-toplevel`).
- [ ] README của bản fork có đối chiếu công thức gốc ↔ mới, rõ ràng file/hàm bị sửa.
- [ ] Notebook Cell 1 đã điền sẵn mọi link người dùng cung cấp.
- [ ] Đã liệt kê rõ với người dùng: phần nào đã kiểm chứng, phần nào **chưa/không thể** kiểm chứng ở
      máy dev (thường là: chạy thật trên GPU với dữ liệu thật, và bước push GitHub thật nếu người dùng
      cần tự làm).

---

## 3. Danh sách "bẫy môi trường" đã biết (kiểm tra chủ động, đừng đợi người dùng báo lỗi)

Trước khi bàn giao, `grep` qua toàn bộ code đã patch (và cả phần code gốc không đổi, vì lỗi có thể nằm
ở đó) để tìm các pattern sau — đây đều là lỗi **đã xảy ra thật** khi chạy code cũ (viết cho môi trường
2020-2023) trên Colab hiện tại (2025-2026, scipy/numpy/torch mới hơn):

| Pattern cần tìm | Vấn đề | Cách sửa |
|---|---|---|
| `.A` sau 1 biến scipy sparse matrix (ví dụ `mat.A`) | scipy ≥ 1.14 đã gỡ bỏ thuộc tính `.A` | Đổi thành `.toarray()` |
| `np.float`, `np.int`, `np.bool`, `np.object` (không có hậu tố số, ví dụ không phải `np.float64`) | numpy ≥ 1.24 đã gỡ các alias này | Dùng kiểu Python gốc (`float`, `int`, `bool`) hoặc `np.float64`/`np.int64` |
| `torch.sparse.FloatTensor(...)` | Deprecated (hiện tại vẫn chạy được, chỉ cảnh báo — nhưng có thể bị gỡ ở bản torch tương lai) | Nếu thấy warning nhưng vẫn chạy được thì để nguyên (đừng sửa cái không hỏng); nếu thấy lỗi thật thì đổi sang `torch.sparse_coo_tensor(...)` |
| `!cd X && Y \| tee Z` trong 1 cell notebook (chuỗi lệnh shell phức tạp qua `!`) | Dễ lỗi mơ hồ về đường dẫn tương đối giữa các cell, khó biết chính xác lệnh có chạy xong hay không | Dùng `subprocess.Popen(cmd, cwd=..., ...)` với đường dẫn log **tuyệt đối**, kiểm tra `returncode` tường minh |
| Cell clone code chỉ chạy `git clone` khi thư mục CHƯA tồn tại | Nếu người dùng sửa code trên GitHub rồi chạy lại notebook trong CÙNG phiên Colab (không Restart runtime), cell sẽ bỏ qua clone và dùng code CŨ, gây hiểu lầm "đã sửa mà vẫn lỗi y hệt" | Luôn xoá thư mục cũ (nếu có) rồi clone lại từ đầu mỗi lần chạy cell |
| Giả định entry point script luôn nằm ngay gốc thư mục vừa clone | Repo trên GitHub có thể bị lồng thêm 1-2 cấp thư mục (do người dùng push nhầm cả folder cha, hoặc kéo-thả qua giao diện web GitHub thay vì `git push`) | Dò tìm entry point script bằng `glob.glob(os.path.join(REPO_DIR, "**", "<ten_script>.py"), recursive=True)`, tự động điều chỉnh `REPO_DIR` nếu tìm thấy ở cấp sâu hơn |
| Cell đọc file log/kết quả không kiểm tra file có tồn tại trước | Nếu cell chạy training chưa chạy xong (hoặc phiên Colab bị ngắt kết nối/reset giữa chừng), cell đọc kết quả sẽ báo `FileNotFoundError` khó hiểu | Luôn `assert os.path.exists(LOG_PATH)` với thông báo lỗi giải thích rõ nguyên nhân thường gặp trước khi mở file |

Mỗi khi phát hiện thêm 1 "bẫy môi trường" mới ở phương án sau này, **bổ sung ngay vào bảng này** để các
phương án sau nữa không giẫm lại vết xe đổ.

---

## 4. Đối chiếu với ví dụ thực tế đã làm (`phuong_an_1_OT_noise_scheduler`)

| Bước trong quy trình | Đã làm gì thực tế cho Phương án 1 (DiffMM + OT-linear) |
|---|---|
| Bước 1 (tải source thật) | `curl` trực tiếp `Params.py`, `Model.py`, `Main.py`, `DataHandler.py`, `README.md` từ `github.com/HKUDS/DiffMM` |
| Bước 2 (thiết kế patch tối thiểu) | Thêm class mới `GaussianDiffusionOT(GaussianDiffusion)`, chỉ override `q_sample`, `p_mean_variance`, `SNR` — không đụng `training_losses`, MSI, contrastive learning, graph aggregation |
| Bước 4.3 (quét dải tham số) | Phát hiện `SNR(0)` chia cho 0 (rồi lớn bất thường) khi quét t=0..T-1 với `SIGMA_MIN` nhỏ — phải thêm clip `max=50` vào `SNR()` |
| Bước 4.4 (dry-run notebook) | Giả lập `Main.py` fake in ra đúng định dạng `"Best epoch : ..."`, test cả nhánh thành công và nhánh `RuntimeError` giả lập |
| Bước 5 (vị trí git độc lập) | Lần đầu tạo `.git` ngay trong `phuong_an_1_OT_noise_scheduler/DiffMM-OT/` (bên trong project đang được git theo dõi ở gốc) → sau đó bị lồng thêm 1 cấp khi push → phải tạo lại ở `E:\NAM_BA\DiffMM-OT` (ngoài project) mới hết lỗi |
| Mục 3 (bẫy môi trường) | Gặp đúng cả 3 bẫy: `coo_matrix.A` bị gỡ (scipy mới), clone-skip-khi-đã-tồn-tại gây dùng nhầm code cũ, và entry point bị lồng thư mục |

---

## 5. Checklist nhanh (dán vào cuối tin nhắn bàn giao cho người dùng)

```
[ ] Source gốc lấy từ GitHub thật (link: ...)
[ ] Patch biên dịch sạch (py_compile)
[ ] Đã dry-run số học (nếu có công thức toán mới) — không NaN/Inf ở mọi điểm biên
[ ] Đã dry-run notebook cell-theo-cell (thành công + lỗi)
[ ] Cell 7 đã dry-run phần xuất PDF — file .pdf tạo ra thật, đúng tên phương án + tên dataset
[ ] Git repo độc lập, nằm NGOÀI mọi repo khác
[ ] README bản fork có đối chiếu công thức gốc/mới
[ ] Cell 1 notebook đã điền sẵn link người dùng cung cấp
[ ] Đã liệt kê rõ phần CHƯA kiểm chứng được (thường: chạy GPU thật + dữ liệu thật)
```
