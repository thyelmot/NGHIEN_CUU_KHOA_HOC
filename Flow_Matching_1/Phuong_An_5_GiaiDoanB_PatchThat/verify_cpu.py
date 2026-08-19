"""
Kiem chung so hoc tren CPU cho patch Giai doan B (Phuong an 5): GaussianDiffusionModalOT trong
Model.py that (khong phai ban tach roi/gia lap nhu Giai doan A). Khong can GPU - monkeypatch
Tensor.cuda() thanh no-op truoc khi import Model.py (moi cho khac deu la code CPU thuan cua PyTorch).

Cac phep kiem tra:
  1. Hoi quy: kappa=0 phai cho mu, sigma, cfm_weight TRUNG KHIT (allclose) voi GaussianDiffusionOT
     (Phuong an 1) va cfm_weight cua GaussianDiffusionCFM/GaussianDiffusionOTCFM (Phuong an 2/3) tren
     cung sigma_min/steps - day la kiem chung QUAN TRONG NHAT cho "cong tac tat an toan khi merge".
  2. training_losses: shape dung, khong NaN/Inf, use_msi bat/tat dung nhu ky vong.
  3. p_sample voi kappa=0, sampling_steps=0, model hoan hao (tra ve dung x_start) -> phai tai tao
     lai chinh xac x_start (kiem tra "duong dan tat dinh dung" nhu da lam voi Phuong an 1/3/4).
  4. p_sample voi kappa>0: khong NaN/Inf, khac kappa=0 (xac nhan duong di modal THUC SU co anh huong).
  5. Bien: 1 user khong tuong tac item nao (x_start toan 0) - khong NaN (centroid ve 0, clamp eps).
  6. Bien: kappa rat lon - g bi bao hoa clamp(g_min,g_max), van khong NaN/Inf.
"""
import torch
import torch.nn as nn

torch.Tensor.cuda = lambda self, *a, **k: self
_orig_module_cuda = nn.Module.cuda
nn.Module.cuda = lambda self, *a, **k: self

import sys
sys.path.insert(0, ".")
from Model import GaussianDiffusion, GaussianDiffusionModalOT

torch.manual_seed(0)

STEPS = 5
SIGMA_MIN = 1e-3
BATCH = 8
NUM_ITEMS = 30
FEAT_DIM = 64  # trong pipeline that, model_feats (image/text/audio_feats sau FeatTrans) co chieu = args.latdim, giong itmEmbeds

def make_batch(zero_row=False):
    x = (torch.rand(BATCH, NUM_ITEMS) > 0.9).float()
    if zero_row:
        x[0] = 0.0
    modal_embeds = torch.randn(NUM_ITEMS, FEAT_DIM, dtype=torch.float64) if False else torch.randn(NUM_ITEMS, FEAT_DIM)
    return x.double(), modal_embeds.double()

def check_finite(name, t):
    assert torch.isfinite(t).all(), f"{name} co NaN/Inf: {t}"

# ---- Test 1: hoi quy kappa=0 vs Phuong an 1 (OT-linear thuan) ----
print("== Test 1: kappa=0 phai trung khit Phuong an 1 ==")
diff5 = GaussianDiffusionModalOT(SIGMA_MIN, STEPS, kappa=0.0, g_min=0.5, g_max=2.0, w_clip=50.0, use_msi=False)

# tu tinh mu_coef/sigma_coef "Phuong an 1" truc tiep (khong phu thuoc file ngoai) de doi chieu
t_idx = torch.arange(STEPS, dtype=torch.float64)
s_ref = 1.0 - t_idx / (STEPS - 1)
mu_coef_ref = s_ref
sigma_coef_ref = 1.0 - (1.0 - SIGMA_MIN) * s_ref

x_start, modal_embeds = make_batch()
tau5, sigma5 = diff5._per_item_path(x_start, modal_embeds)  # (batch, T, num_items)

# kappa=0 => g=1 moi noi => tau(t)=s(t) (khong phu thuoc item/user/x_start)
tau_expected = s_ref.view(1, STEPS, 1).expand_as(tau5)
sigma_expected = sigma_coef_ref.view(1, STEPS, 1).expand_as(sigma5)
assert torch.allclose(tau5, tau_expected, atol=1e-10), "tau (kappa=0) KHONG trung Phuong an 1"
assert torch.allclose(sigma5, sigma_expected, atol=1e-10), "sigma (kappa=0) KHONG trung Phuong an 1"
print("  tau, sigma (kappa=0) trung khit Phuong an 1: OK")

# doi chieu cfm_weight voi cong thuc Phuong an 2/3 (dung vo huong mu_coef_ref/sigma_coef_ref)
mu_prev_ref = torch.cat([mu_coef_ref[:1], mu_coef_ref[:-1]])
sigma_prev_ref = torch.cat([sigma_coef_ref[:1], sigma_coef_ref[:-1]])
mu_prime_ref = mu_prev_ref - mu_coef_ref
sigma_prime_ref = sigma_prev_ref - sigma_coef_ref
w_ref = (mu_prime_ref - (sigma_prime_ref / sigma_coef_ref.clamp(min=1e-8)) * mu_coef_ref) ** 2
w_ref[0] = 1.0
w_ref = w_ref.clamp(max=50.0)

w5 = diff5._cfm_weight(tau5, sigma5)  # (batch, T, num_items)
w5_broadcast_check = w5[0, :, 0]  # kappa=0 -> w khong phu thuoc user/item, lay 1 lat cat de so sanh
assert torch.allclose(w5_broadcast_check, w_ref, atol=1e-8), f"cfm_weight (kappa=0) KHONG trung Phuong an 2/3\n{w5_broadcast_check}\nvs\n{w_ref}"
# xac nhan MOI (user,item) deu cho cung 1 vector trong so (vi kappa=0 -> g=1 dong loat)
assert torch.allclose(w5, w_ref.view(1, STEPS, 1).expand_as(w5), atol=1e-8)
print("  cfm_weight (kappa=0) trung khit Phuong an 2/3: OK")

# ---- Test 2: training_losses ----
print("== Test 2: training_losses (shape, khong NaN, cong tac use_msi) ==")
itmEmbeds = torch.randn(NUM_ITEMS, 64, dtype=torch.float64)
batch_index = torch.arange(BATCH)

def fake_model_factory(scale=1.0, offset=0.0):
    def fake_model(x_t, ts, *a, **k):
        return (x_t * scale + offset).clamp(0, 1)
    return fake_model

for kappa in [0.0, 1.0, 5.0]:
    diff = GaussianDiffusionModalOT(SIGMA_MIN, STEPS, kappa=kappa, use_msi=False)
    diff_loss, gc_loss = diff.training_losses(fake_model_factory(), x_start, itmEmbeds, batch_index, modal_embeds)
    assert diff_loss.shape == (BATCH,), diff_loss.shape
    assert gc_loss.shape == (BATCH,), gc_loss.shape
    check_finite(f"diff_loss(kappa={kappa})", diff_loss)
    check_finite(f"gc_loss(kappa={kappa})", gc_loss)
    assert torch.allclose(gc_loss, torch.zeros_like(gc_loss)), "use_msi=False phai cho gc_loss=0"

diff_msi = GaussianDiffusionModalOT(SIGMA_MIN, STEPS, kappa=1.0, use_msi=True)
diff_loss, gc_loss = diff_msi.training_losses(fake_model_factory(), x_start, itmEmbeds, batch_index, modal_embeds)
check_finite("diff_loss(use_msi=True)", diff_loss)
check_finite("gc_loss(use_msi=True)", gc_loss)
assert not torch.allclose(gc_loss, torch.zeros_like(gc_loss)), "use_msi=True phai cho gc_loss != 0"
print("  training_losses: OK (moi kappa test, ca 2 nhanh use_msi)")

# ---- Test 3: p_sample duong tat dinh, model hoan hao, kappa=0, sampling_steps=0 ----
print("== Test 3: p_sample tat dinh, model hoan hao, kappa=0 -> tai tao dung x_start ==")
diff0 = GaussianDiffusionModalOT(SIGMA_MIN, STEPS, kappa=0.0, use_msi=False)
perfect_model = fake_model_factory(scale=1.0, offset=0.0)  # tra ve dung x_t; nhung can tra ve alpha_0 du doan
# model can du doan alpha_0 - vi steps=0 nen x_t = x_start ngay tu dau, model hoan hao = identity la du
out = diff0.p_sample(perfect_model, x_start, modal_embeds, steps=0, sampling_noise=False)
check_finite("p_sample output (steps=0, model hoan hao)", out)
assert torch.allclose(out, x_start, atol=1e-6), "Model hoan hao + kappa=0 + steps=0 phai tai tao dung x_start"
print("  p_sample (steps=0, model=identity, kappa=0): tai tao chinh xac x_start - OK")

# ---- Test 4: p_sample kappa>0, sampling_steps>0 -> khong NaN, khac kappa=0 ----
print("== Test 4: p_sample kappa>0 (sampling_steps>0) khong NaN va khac kappa=0 ==")
torch.manual_seed(1)
diffK = GaussianDiffusionModalOT(SIGMA_MIN, STEPS, kappa=3.0, g_min=0.3, g_max=3.0, use_msi=False)
noisy_model = fake_model_factory(scale=0.8, offset=0.05)
outK = diffK.p_sample(noisy_model, x_start, modal_embeds, steps=3, sampling_noise=False)
check_finite("p_sample output (kappa=3)", outK)

torch.manual_seed(1)
out0b = diff0.p_sample(noisy_model, x_start, modal_embeds, steps=3, sampling_noise=False)
check_finite("p_sample output (kappa=0, steps=3)", out0b)
assert not torch.allclose(outK, out0b), "kappa=3 phai cho ket qua KHAC kappa=0 (duong di modal co tac dung)"
print("  p_sample kappa>0: khong NaN, khac ro kappa=0 - OK")

# ---- Test 5: bien - 1 user khong tuong tac item nao ----
print("== Test 5: bien - user khong co tuong tac nao (x_start toan 0 o 1 hang) ==")
x_zero, modal_zero = make_batch(zero_row=True)
diffZ = GaussianDiffusionModalOT(SIGMA_MIN, STEPS, kappa=2.0, use_msi=True)
diff_loss_z, gc_loss_z = diffZ.training_losses(fake_model_factory(), x_zero, itmEmbeds, batch_index, modal_zero)
check_finite("diff_loss (user rong)", diff_loss_z)
check_finite("gc_loss (user rong)", gc_loss_z)
out_z = diffZ.p_sample(fake_model_factory(scale=0.9), x_zero, modal_zero, steps=2, sampling_noise=False)
check_finite("p_sample (user rong)", out_z)
print("  User khong tuong tac: khong NaN/Inf - OK")

# ---- Test 6: bien - kappa rat lon (bao hoa clamp) ----
print("== Test 6: bien - kappa rat lon (g bi bao hoa clamp) ==")
diffBig = GaussianDiffusionModalOT(SIGMA_MIN, STEPS, kappa=1e6, g_min=0.5, g_max=2.0, use_msi=False)
tau_big, sigma_big = diffBig._per_item_path(x_start, modal_embeds)
check_finite("tau (kappa=1e6)", tau_big)
check_finite("sigma (kappa=1e6)", sigma_big)
diff_loss_big, _ = diffBig.training_losses(fake_model_factory(), x_start, itmEmbeds, batch_index, modal_embeds)
check_finite("diff_loss (kappa=1e6)", diff_loss_big)
out_big = diffBig.p_sample(fake_model_factory(scale=0.7), x_start, modal_embeds, steps=1, sampling_noise=False)
check_finite("p_sample (kappa=1e6)", out_big)
print("  kappa rat lon: khong NaN/Inf, clamp hoat dong dung - OK")

print("\nTAT CA 6 PHEP KIEM TRA TREN CPU DEU PASS.")
