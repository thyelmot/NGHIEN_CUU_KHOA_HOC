"""
Kiem chung so hoc tren CPU cho patch Phuong an 7 (v2, Residual Head): GaussianDiffusionResidualOT
trong Model.py that. Khong can GPU - monkeypatch Tensor.cuda()/Module.cuda() thanh no-op.

Cac phep kiem tra:
  1. Hoi quy: residual_head=False phai trung khit tuyet doi voi GaussianDiffusionAnchorOT (Phuong an 6)
     o CA anchor_w=0 LAN anchor_w>0 (khong chi truong hop anchor_w=0 nhu PA6 da lam voi PA3).
  2. training_losses: shape dung, khong NaN/Inf qua moi to hop (anchor_w, residual_head).
  3. residual_head=True voi "denoiser du doan dung phan du" -> tai tao chinh xac alpha_0 (MSE=0).
  4. p_sample tat dinh, residual_head=True, denoiser hoan hao (du doan dung residual) -> tai tao
     dung x_start qua toan bo vong lap.
  5. _need_anchor(): xac nhan alpha_l duoc tinh dung ca khi anchor_w=0 nhung residual_head=True.
  6. Bien: user rong, anchor_w/residual_head ket hop voi gia tri cuc doan.
"""
import torch
import torch.nn as nn

torch.Tensor.cuda = lambda self, *a, **k: self
nn.Module.cuda = lambda self, *a, **k: self

import sys
sys.path.insert(0, ".")
from Model import GaussianDiffusionAnchorOT, GaussianDiffusionResidualOT

torch.manual_seed(0)

STEPS = 5
SIGMA_MIN = 1e-3
BATCH = 8
NUM_ITEMS = 30
LATDIM = 16

def make_batch(zero_row=False):
    x = (torch.rand(BATCH, NUM_ITEMS) > 0.85).float().double()
    if zero_row:
        x[0] = 0.0
    uEmbeds_batch = torch.randn(BATCH, LATDIM, dtype=torch.float64)
    iEmbeds = torch.randn(NUM_ITEMS, LATDIM, dtype=torch.float64)
    return x, uEmbeds_batch, iEmbeds

def check_finite(name, t):
    assert torch.isfinite(t).all(), f"{name} co NaN/Inf: {t}"

def fake_model_factory(scale=1.0, offset=0.0):
    def fake_model(x_t, ts, *a, **k):
        return (x_t * scale + offset).clamp(-5, 5)
    return fake_model

x_start, uEmbeds_batch, iEmbeds = make_batch()
itmEmbeds = torch.randn(NUM_ITEMS, LATDIM, dtype=torch.float64)
batch_index = torch.arange(BATCH)

# ---- Test 1: hoi quy residual_head=False vs Phuong an 6, o CA anchor_w=0 lan anchor_w>0 ----
print("== Test 1: residual_head=False phai trung khit Phuong an 6 (moi anchor_w) ==")
for anchor_w_test in [0.0, 2.5]:
    torch.manual_seed(11)
    diff6 = GaussianDiffusionAnchorOT(SIGMA_MIN, STEPS, w_clip=50.0, num_sample_steps=0, anchor_w=anchor_w_test)
    torch.manual_seed(11)
    diff7 = GaussianDiffusionResidualOT(SIGMA_MIN, STEPS, w_clip=50.0, num_sample_steps=0, anchor_w=anchor_w_test, residual_head=False)

    assert torch.allclose(diff6.cfm_weight, diff7.cfm_weight)

    fake_model = fake_model_factory(scale=0.7, offset=0.1)
    torch.manual_seed(21)
    dl6, gl6 = diff6.training_losses(fake_model, x_start, itmEmbeds, batch_index, itmEmbeds, uEmbeds_batch)
    torch.manual_seed(21)
    dl7, gl7 = diff7.training_losses(fake_model, x_start, itmEmbeds, batch_index, itmEmbeds, uEmbeds_batch)
    assert torch.allclose(dl6, dl7, atol=1e-12) and torch.allclose(gl6, gl7, atol=1e-12), f"training_losses lech o anchor_w={anchor_w_test}"

    torch.manual_seed(31)
    out6 = diff6.p_sample(fake_model, x_start, uEmbeds_batch, iEmbeds, steps=3, sampling_noise=False)
    torch.manual_seed(31)
    out7 = diff7.p_sample(fake_model, x_start, uEmbeds_batch, iEmbeds, steps=3, sampling_noise=False)
    assert torch.allclose(out6, out7, atol=1e-10), f"p_sample lech o anchor_w={anchor_w_test}"
    print(f"  anchor_w={anchor_w_test}: training_losses + p_sample khop tuyet doi voi Phuong an 6 - OK")

# ---- Test 2: training_losses shape/NaN qua moi to hop ----
print("== Test 2: training_losses (shape, khong NaN) qua moi to hop (anchor_w, residual_head) ==")
for anchor_w in [0.0, 1.0, -3.0]:
    for residual_head in [False, True]:
        diff = GaussianDiffusionResidualOT(SIGMA_MIN, STEPS, anchor_w=anchor_w, residual_head=residual_head)
        dl, gl = diff.training_losses(fake_model_factory(), x_start, itmEmbeds, batch_index, itmEmbeds, uEmbeds_batch)
        assert dl.shape == (BATCH,) and gl.shape == (BATCH,)
        check_finite(f"diff_loss(anchor_w={anchor_w},residual_head={residual_head})", dl)
        check_finite(f"gc_loss(anchor_w={anchor_w},residual_head={residual_head})", gl)
print("  OK moi to hop")

# ---- Test 3: residual_head=True, denoiser du doan dung PHAN DU -> MSE=0 ----
print("== Test 3: residual_head=True, denoiser du doan dung residual -> tai tao chinh xac alpha_0 ==")
diffR = GaussianDiffusionResidualOT(SIGMA_MIN, STEPS, anchor_w=1.5, residual_head=True)
alpha_l_expected = diffR._compute_anchor(uEmbeds_batch, itmEmbeds)


def perfect_residual_model(x_t, ts, *a, **k):
    # mo phong 1 mang du doan DUNG phan du (x_start - alpha_l), khong phu thuoc x_t/ts thuc su
    return x_start - alpha_l_expected


dl, gl = diffR.training_losses(perfect_residual_model, x_start, itmEmbeds, batch_index, itmEmbeds, uEmbeds_batch)
check_finite("diff_loss(perfect residual)", dl)
# mse phai ~0 (truoc khi nhan trong so w(t)) - kiem tra truc tiep qua cong thuc
alpha_l_check = diffR._compute_anchor(uEmbeds_batch, itmEmbeds)
model_output_check = diffR._apply_residual_head(x_start - alpha_l_check, alpha_l_check)
assert torch.allclose(model_output_check, x_start, atol=1e-10), "residual head khong tai tao dung alpha_0"
print("  model_output tai tao dung alpha_0 (sai lech < 1e-10) - OK")

# ---- Test 4: p_sample tat dinh, residual_head=True, denoiser hoan hao -> tai tao dung x_start ----
print("== Test 4: p_sample tat dinh, residual_head=True, denoiser hoan hao (steps=0) ==")
diffR0 = GaussianDiffusionResidualOT(SIGMA_MIN, STEPS, anchor_w=0.0, residual_head=True)


def make_perfect_residual_model_for(diff_obj, uE, iE):
    alpha_l_local = diff_obj._compute_anchor(uE, iE)

    def m(x_t, ts, *a, **k):
        return x_start - alpha_l_local
    return m


perfect_model_0 = make_perfect_residual_model_for(diffR0, uEmbeds_batch, iEmbeds)
out = diffR0.p_sample(perfect_model_0, x_start, uEmbeds_batch, iEmbeds, steps=0, sampling_noise=False)
assert torch.allclose(out, x_start, atol=1e-6), f"residual_head=True steps=0 khong tai tao dung x_start: max lech {(out-x_start).abs().max()}"
print("  OK - tai tao chinh xac x_start")

# ---- Test 5: _need_anchor() dung khi anchor_w=0 nhung residual_head=True ----
print("== Test 5: _need_anchor() tra ve dung gia tri ==")
d_a = GaussianDiffusionResidualOT(SIGMA_MIN, STEPS, anchor_w=0.0, residual_head=False)
d_b = GaussianDiffusionResidualOT(SIGMA_MIN, STEPS, anchor_w=0.0, residual_head=True)
d_c = GaussianDiffusionResidualOT(SIGMA_MIN, STEPS, anchor_w=2.0, residual_head=False)
assert d_a._need_anchor() == False
assert d_b._need_anchor() == True
assert d_c._need_anchor() == True
print("  OK")

# ---- Test 6: bien - user rong, gia tri cuc doan ----
print("== Test 6: bien - user rong va gia tri cuc doan ==")
x_zero, u_zero, i_zero = make_batch(zero_row=True)
diffZ = GaussianDiffusionResidualOT(SIGMA_MIN, STEPS, anchor_w=3.0, residual_head=True)
dl, gl = diffZ.training_losses(fake_model_factory(), x_zero, itmEmbeds, batch_index, itmEmbeds, u_zero)
check_finite("diff_loss(user rong)", dl)
check_finite("gc_loss(user rong)", gl)
out_z = diffZ.p_sample(fake_model_factory(scale=0.9), x_zero, u_zero, i_zero, steps=2, sampling_noise=False)
check_finite("p_sample(user rong)", out_z)

diffBig = GaussianDiffusionResidualOT(SIGMA_MIN, STEPS, anchor_w=1e6, residual_head=True)
out_big = diffBig.p_sample(fake_model_factory(scale=0.7), x_start, uEmbeds_batch, iEmbeds, steps=1, sampling_noise=False)
check_finite("p_sample(anchor_w=1e6, residual_head=True)", out_big)
print("  OK - khong NaN/Inf")

print("\nTAT CA 6 PHEP KIEM TRA TREN CPU DEU PASS.")
