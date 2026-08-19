"""
Giai doan A (Phuong_An_5_Modality_Conditioned_OT_KeHoachChiTiet.md, muc 5):
Kiem tra gradient bang so hoc cho cong thuc TU THIET KE cua Phuong an 5 (CT-5.1 .. CT-5.7),
tren du lieu gia lap, KHONG dung GPU, KHONG dung dataset that, KHONG dong vao codebase DiffMM.

Muc tieu: xac nhan
  1) Shape dung, khong NaN/Inf (ke ca truong hop bien: user khong co item nao trong alpha0).
  2) Gradient chay dung qua ca 2 duong: model_output (nhu binh thuong) VA modal_embeds (duong
     MOI ma Phuong an 5 them vao, thay cho vai tro cua MSI) -- doi chieu gradient tu autograd
     voi gradient tinh bang sai phan huu han (finite-difference) DOC LAP, khong dung lai may
     autograd cua chinh cong thuc dang kiem tra.
  3) kappa=0 quy ve dung Phuong an 1 (g=1 moi noi, khong phu thuoc modal_embeds).
"""
import torch

torch.manual_seed(0)
DTYPE = torch.float64  # can float64 cho kiem tra gradient bang sai phan huu han duoc chinh xac


def modal_affinity(alpha0, modal_embeds, eps=1e-8):
    """CT-5.1 + CT-5.2. alpha0: (batch, I). modal_embeds: (I, d). Tra ve phi: (batch, I)."""
    centroid = alpha0 @ modal_embeds  # (batch, d) = Sum_j alpha0[u,j] * e_j
    centroid_norm = centroid.norm(dim=-1, keepdim=True).clamp(min=eps)  # phong NaN khi user khong co item nao
    centroid_unit = centroid / centroid_norm

    embeds_norm = modal_embeds.norm(dim=-1, keepdim=True).clamp(min=eps)
    embeds_unit = modal_embeds / embeds_norm

    phi = centroid_unit @ embeds_unit.T  # (batch, I)
    return phi


def per_item_schedule(alpha0, modal_embeds, kappa, g_min, g_max, steps, sigma_min, w_clip):
    """CT-5.3 .. CT-5.7 (phan tinh mang he so). Tra ve mu, sigma, w deu shape (batch, steps, I)."""
    batch, num_items = alpha0.shape
    phi = modal_affinity(alpha0, modal_embeds)  # (batch, I)
    g = torch.exp(-kappa * phi).clamp(min=g_min, max=g_max)  # (batch, I)

    s = torch.linspace(0.0, 1.0, steps, dtype=DTYPE)  # (T,)
    # tau[b,t,i] = s[t] ** g[b,i]
    tau = s.view(1, steps, 1) ** g.view(batch, 1, num_items)  # (batch, T, I)

    mu = tau * alpha0.view(batch, 1, num_items)
    sigma = 1.0 - (1.0 - sigma_min) * tau

    mu_prev = torch.cat([mu[:, :1, :], mu[:, :-1, :]], dim=1)
    sigma_prev = torch.cat([sigma[:, :1, :], sigma[:, :-1, :]], dim=1)
    mu_prime = mu_prev - mu
    sigma_prime = sigma_prev - sigma

    w = (mu_prime - (sigma_prime / sigma.clamp(min=1e-8)) * mu) ** 2
    w = torch.cat([torch.ones_like(w[:, :1, :]), w[:, 1:, :]], dim=1)  # bien s=0: dung quy uoc w=1
    w = w.clamp(max=w_clip)
    return mu, sigma, w


def gather_at_t(arr, t_idx):
    """arr: (batch, T, I). t_idx: (batch,) long. Tra ve (batch, I) — gia tri tai dung buoc t cua tung vi du."""
    batch, T, I = arr.shape
    idx = t_idx.view(batch, 1, 1).expand(batch, 1, I)
    return torch.gather(arr, dim=1, index=idx).squeeze(1)


def pa5_loss(modal_embeds, model_output, alpha0, noise, t_idx, kappa, g_min, g_max, steps, sigma_min, w_clip):
    """
    Loss CFM theo tung toa do (CT-5.7), ap dung dung thu tu: nhan trong so TRUOC khi gop item
    (khac PA2, noi w la vo huong nhan SAU khi da mean_flat).
    model_output dong vai tro du doan alpha0 cua mang denoiser (o day de la bien tu do, khong
    qua 1 mang that, vi muc tieu Giai doan A la kiem tra CONG THUC, khong phai kien truc mang).
    """
    mu, sigma, w = per_item_schedule(alpha0, modal_embeds, kappa, g_min, g_max, steps, sigma_min, w_clip)
    w_t = gather_at_t(w, t_idx)  # (batch, I)
    sq_err = (model_output - alpha0) ** 2  # (batch, I)  -- x_start = alpha0 (khong dung x_t vi day la kiem tra CT-5.7 doc lap voi q_sample)
    per_example = (w_t * sq_err).sum(dim=-1)  # (batch,)
    return per_example.mean()


def finite_diff_grad(f, x, eps=1e-6):
    """Sai phan huu han TRUNG TAM, doc lap voi autograd, dung de doi chieu.
    Thao tac tren x.data (bo qua theo doi autograd) de duoc phep sua tai cho leaf tensor."""
    grad = torch.zeros_like(x)
    flat_x = x.data.flatten()
    flat_grad = grad.flatten()
    for i in range(flat_x.numel()):
        orig = flat_x[i].item()
        flat_x[i] = orig + eps
        f_plus = f().item()
        flat_x[i] = orig - eps
        f_minus = f().item()
        flat_x[i] = orig
        flat_grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def run_case(name, batch, num_items, embed_dim, kappa, seed, extra_note="", zero_out_first_user=False):
    print(f"\n=== {name} (seed={seed}) {extra_note}===")
    g = torch.Generator().manual_seed(seed)

    alpha0 = (torch.rand(batch, num_items, generator=g) > 0.6).to(DTYPE)
    if zero_out_first_user:
        alpha0[0, :] = 0.0  # co tinh tao truong hop bien: user dau tien khong tuong tac item nao

    modal_embeds = torch.randn(num_items, embed_dim, generator=g, dtype=DTYPE, requires_grad=True)
    model_output = torch.rand(batch, num_items, generator=g, dtype=DTYPE, requires_grad=True)
    noise = torch.randn(batch, num_items, generator=g, dtype=DTYPE)
    t_idx = torch.randint(0, 5, (batch,), generator=g)

    kwargs = dict(alpha0=alpha0, noise=noise, t_idx=t_idx, kappa=kappa, g_min=0.5, g_max=2.0,
                  steps=5, sigma_min=1e-3, w_clip=50.0)

    loss = pa5_loss(modal_embeds, model_output, **kwargs)
    assert torch.isfinite(loss), f"Loss KHONG huu han! loss={loss}"
    print(f"loss = {loss.item():.6f}  (huu han: OK)")

    loss.backward()
    assert modal_embeds.grad is not None, "modal_embeds KHONG nhan duoc gradient!"
    assert torch.isfinite(modal_embeds.grad).all(), "Gradient cua modal_embeds co NaN/Inf!"
    assert torch.isfinite(model_output.grad).all(), "Gradient cua model_output co NaN/Inf!"
    print("Gradient (autograd) cho modal_embeds va model_output: huu han, khong NaN/Inf -- OK")

    # Doi chieu gradient autograd vs sai phan huu han DOC LAP, cho modal_embeds (duong MOI, quan trong nhat)
    def f_wrt_modal():
        with torch.no_grad():
            return pa5_loss(modal_embeds, model_output.detach(), **kwargs)

    fd_grad = finite_diff_grad(f_wrt_modal, modal_embeds)
    ad_grad = modal_embeds.grad
    max_abs_diff = (fd_grad - ad_grad).abs().max().item()
    denom = ad_grad.abs().max().item() + 1e-8
    rel_diff = max_abs_diff / denom
    print(f"Doi chieu gradient (modal_embeds): sai lech tuyet doi lon nhat = {max_abs_diff:.3e}, "
          f"sai lech tuong doi = {rel_diff:.3e}")
    assert max_abs_diff < 1e-5, f"Gradient autograd va sai phan huu han LECH NHAU qua nhieu! {max_abs_diff}"
    print("=> Gradient autograd KHOP voi sai phan huu han doc lap -- cong thuc dao ham dung.")

    return alpha0, modal_embeds, model_output, kwargs


# ------------------------------------------------------------------
run_case("Truong hop thuong: kappa=1.0, batch nho", batch=3, num_items=6, embed_dim=4, kappa=1.0, seed=0)

run_case("Bien: 1 user khong co item nao (test epsilon trong normalize)",
         batch=3, num_items=6, embed_dim=4, kappa=1.0, seed=1,
         extra_note="-- de xac nhan khong NaN khi centroid = 0 ",
         zero_out_first_user=True)

run_case("Bien: kappa lon (ep nhieu item vao vung bi clamp g_min/g_max)",
         batch=4, num_items=8, embed_dim=4, kappa=20.0, seed=7,
         extra_note="-- kiem tra gradient van huu han khi nhieu item bi clamp ")

print("\n=== Quy mo gan thuc te hon (batch=64, item=500) — chi forward+backward, KHONG finite-diff (qua cham) ===")
_batch, _num_items, _embed_dim, _T = 64, 500, 32, 5
_g = torch.Generator().manual_seed(3)
_alpha0 = (torch.rand(_batch, _num_items, generator=_g, dtype=DTYPE) > 0.98).to(DTYPE)  # thua, giong du lieu that
_modal_embeds = torch.randn(_num_items, _embed_dim, generator=_g, dtype=DTYPE, requires_grad=True)
_model_output = torch.rand(_batch, _num_items, generator=_g, dtype=DTYPE, requires_grad=True)
_noise = torch.randn(_batch, _num_items, generator=_g, dtype=DTYPE)
_t_idx = torch.randint(0, _T, (_batch,), generator=_g)

_loss = pa5_loss(_modal_embeds, _model_output, _alpha0, _noise, _t_idx, kappa=1.0, g_min=0.5, g_max=2.0,
                  steps=_T, sigma_min=1e-3, w_clip=50.0)
assert torch.isfinite(_loss)
_loss.backward()
assert torch.isfinite(_modal_embeds.grad).all()
assert torch.isfinite(_model_output.grad).all()
print(f"loss={_loss.item():.4f}, batch={_batch}, num_items={_num_items} — huu han, khong NaN/Inf. OK")

print("\n=== Kiem tra kappa=0 quy ve dung Phuong an 1 (g=1 moi noi, KHONG phu thuoc modal_embeds) ===")
batch, num_items, embed_dim = 4, 7, 5
g = torch.Generator().manual_seed(42)
alpha0 = (torch.rand(batch, num_items, generator=g) > 0.5).to(DTYPE)
modal_embeds_a = torch.randn(num_items, embed_dim, generator=g, dtype=DTYPE)
modal_embeds_b = torch.randn(num_items, embed_dim, generator=g, dtype=DTYPE)  # KHAC HAN modal_embeds_a
mu_a, sigma_a, w_a = per_item_schedule(alpha0, modal_embeds_a, kappa=0.0, g_min=0.5, g_max=2.0,
                                        steps=5, sigma_min=1e-3, w_clip=50.0)
mu_b, sigma_b, w_b = per_item_schedule(alpha0, modal_embeds_b, kappa=0.0, g_min=0.5, g_max=2.0,
                                        steps=5, sigma_min=1e-3, w_clip=50.0)
assert torch.allclose(mu_a, mu_b) and torch.allclose(sigma_a, sigma_b) and torch.allclose(w_a, w_b)
print("OK - voi kappa=0, mu/sigma/w GIONG HET NHAU du modal_embeds khac nhau hoan toan")
print("      (dung nhu chung minh dai so: kappa=0 => g=1 moi noi => khong con phu thuoc modal_embeds)")

# so sanh voi cong thuc OT-linear thuan cua Phuong an 1 (mu=s*alpha0, sigma=1-(1-sigma_min)*s)
s_ref = torch.linspace(0, 1, 5, dtype=DTYPE)
mu_pa1 = s_ref.view(1, 5, 1) * alpha0.view(batch, 1, num_items)
sigma_pa1 = 1.0 - (1.0 - 1e-3) * s_ref.view(1, 5, 1)
assert torch.allclose(mu_a, mu_pa1, atol=1e-9)
assert torch.allclose(sigma_a, sigma_pa1.expand_as(sigma_a), atol=1e-9)
print("OK - va giong het cong thuc OT-linear thuan cua Phuong an 1 (sai so ~0)")

print("\n=== TAT CA KIEM TRA GIAI DOAN A (gradient so hoc, tren du lieu gia lap) DEU PASS ===")
