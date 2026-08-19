"""
Kiem chung so hoc tren CPU cho patch Giai doan B (Phuong an 6): GaussianDiffusionAnchorOT trong
Model.py that. Khong can GPU - monkeypatch Tensor.cuda()/Module.cuda() thanh no-op.

Cac phep kiem tra:
  1. Hoi quy: anchor_w=0 phai cho q_sample/p_sample TRUNG KHIT tuyet doi voi GaussianDiffusionOTCFM
     (Phuong an 3) tren cung sigma_min/steps/seed - kiem chung QUAN TRONG NHAT cho cong tac an toan.
  2. training_losses: shape dung, khong NaN/Inf, cfm_weight KHONG doi khi bat/tat anchor (CT-6.7).
  3. p_sample tat dinh, model hoan hao, anchor_w=0 -> tai tao dung x_start.
  4. p_sample anchor_w>0 -> khong NaN/Inf, khac ro anchor_w=0.
  5. Round-trip voi denoiser hoan hao, moi to hop (anchor_w, alpha_l) -> tai tao dung x_start (CT-6.6).
  6. Bien: user rong (x_start toan 0), anchor_w rat lon.
"""
import torch
import torch.nn as nn

torch.Tensor.cuda = lambda self, *a, **k: self
nn.Module.cuda = lambda self, *a, **k: self

import sys
sys.path.insert(0, ".")
from Model import GaussianDiffusionOTCFM, GaussianDiffusionAnchorOT

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
        return (x_t * scale + offset).clamp(0, 1)
    return fake_model

x_start, uEmbeds_batch, iEmbeds = make_batch()
itmEmbeds = torch.randn(NUM_ITEMS, LATDIM, dtype=torch.float64)
batch_index = torch.arange(BATCH)

# ---- Test 1: hoi quy anchor_w=0 vs Phuong an 3 (OT+CFM thuan) ----
print("== Test 1: anchor_w=0 phai trung khit Phuong an 3 ==")
torch.manual_seed(42)
diff3 = GaussianDiffusionOTCFM(SIGMA_MIN, STEPS, w_clip=50.0, num_sample_steps=0)
torch.manual_seed(42)
diff6 = GaussianDiffusionAnchorOT(SIGMA_MIN, STEPS, w_clip=50.0, num_sample_steps=0, anchor_w=0.0)

assert torch.allclose(diff3.mu_coef, diff6.mu_coef)
assert torch.allclose(diff3.sigma_coef, diff6.sigma_coef)
assert torch.allclose(diff3.cfm_weight, diff6.cfm_weight), "cfm_weight PHAI giong het (CT-6.7)"
print("  mu_coef, sigma_coef, cfm_weight: khop tuyet doi - OK")

t_fixed = torch.tensor([2] * BATCH)
noise_fixed = torch.randn(BATCH, NUM_ITEMS, dtype=torch.float64)
x_t_3 = diff3.q_sample(x_start, t_fixed, noise_fixed.clone())
alpha_l_zero = torch.zeros_like(x_start)
x_t_6 = diff6.q_sample(x_start, alpha_l_zero, t_fixed, noise_fixed.clone())
assert torch.allclose(x_t_3, x_t_6, atol=1e-12), "q_sample (anchor_w=0) KHONG trung khit PA3"
print("  q_sample (anchor_w=0): khop tuyet doi - OK")

fake_model = fake_model_factory(scale=0.7, offset=0.1)
mean3, _ = diff3.p_mean_variance(fake_model, x_t_3, t_fixed)
mean6, _ = diff6.p_mean_variance(fake_model, x_t_6, alpha_l_zero, t_fixed)
assert torch.allclose(mean3, mean6, atol=1e-12), "p_mean_variance (anchor_w=0) KHONG trung khit PA3"
print("  p_mean_variance (anchor_w=0): khop tuyet doi - OK")

torch.manual_seed(7)
out3 = diff3.p_sample(fake_model, x_start, steps=3, sampling_noise=False)
torch.manual_seed(7)  # dong bo RNG - p_sample voi steps>0 tu ve 1 mau nhieu ngau nhien ben trong q_sample
out6 = diff6.p_sample(fake_model, x_start, uEmbeds_batch, iEmbeds, steps=3, sampling_noise=False)
assert torch.allclose(out3, out6, atol=1e-10), "p_sample (anchor_w=0) KHONG trung khit PA3"
print("  p_sample (anchor_w=0) toan bo vong lap: khop tuyet doi - OK")

# ---- Test 2: training_losses ----
print("== Test 2: training_losses (shape, khong NaN, cfm_weight bat bien) ==")
for anchor_w in [0.0, 1.0, 5.0, -2.0]:
    diff = GaussianDiffusionAnchorOT(SIGMA_MIN, STEPS, anchor_w=anchor_w)
    diff_loss, gc_loss = diff.training_losses(fake_model_factory(), x_start, itmEmbeds, batch_index, itmEmbeds, uEmbeds_batch)
    assert diff_loss.shape == (BATCH,) and gc_loss.shape == (BATCH,)
    check_finite(f"diff_loss(anchor_w={anchor_w})", diff_loss)
    check_finite(f"gc_loss(anchor_w={anchor_w})", gc_loss)
    assert torch.allclose(diff.cfm_weight, diff3.cfm_weight), f"cfm_weight bi doi voi anchor_w={anchor_w}!"
print("  training_losses: OK moi anchor_w, cfm_weight luon bat bien")

# ---- Test 3: p_sample tat dinh, model hoan hao, anchor_w=0 ----
print("== Test 3: p_sample tat dinh, model hoan hao, anchor_w=0, steps=0 -> tai tao dung x_start ==")
diff0 = GaussianDiffusionAnchorOT(SIGMA_MIN, STEPS, anchor_w=0.0)
perfect_model = fake_model_factory(scale=1.0, offset=0.0)
out = diff0.p_sample(perfect_model, x_start, uEmbeds_batch, iEmbeds, steps=0, sampling_noise=False)
assert torch.allclose(out, x_start, atol=1e-6)
print("  OK")

# ---- Test 4: anchor_w>0, khong NaN, khac ro anchor_w=0 ----
print("== Test 4: p_sample anchor_w>0 khong NaN va khac anchor_w=0 ==")
diffA = GaussianDiffusionAnchorOT(SIGMA_MIN, STEPS, anchor_w=3.0)
noisy_model = fake_model_factory(scale=0.8, offset=0.05)
outA = diffA.p_sample(noisy_model, x_start, uEmbeds_batch, iEmbeds, steps=3, sampling_noise=False)
check_finite("p_sample(anchor_w=3)", outA)
out0b = diff0.p_sample(noisy_model, x_start, uEmbeds_batch, iEmbeds, steps=3, sampling_noise=False)
assert not torch.allclose(outA, out0b), "anchor_w=3 phai cho ket qua khac anchor_w=0"
print("  OK - khong NaN, khac ro anchor_w=0")

# ---- Test 5: round-trip voi denoiser hoan hao, moi to hop (anchor_w, alpha_l) ----
print("== Test 5: round-trip denoiser hoan hao (CT-6.6) ==")
for anchor_w in [0.0, 1.0, 5.0]:
    for seed in range(3):
        torch.manual_seed(100 + seed)
        diff = GaussianDiffusionAnchorOT(SIGMA_MIN, STEPS, anchor_w=anchor_w)
        u_b = torch.randn(BATCH, LATDIM, dtype=torch.float64)
        i_e = torch.randn(NUM_ITEMS, LATDIM, dtype=torch.float64)
        alpha_l = diff._compute_anchor(u_b, i_e) if anchor_w != 0 else torch.zeros_like(x_start)
        t_last = torch.tensor([STEPS - 1] * BATCH)
        x_t = diff.q_sample(x_start, alpha_l, t_last, noise=torch.randn(BATCH, NUM_ITEMS, dtype=torch.float64))
        x_cur = x_t
        for t_i in range(STEPS - 1, -1, -1):
            diff.next_index_map = {t_i: (t_i - 1 if t_i > 0 else -1)}
            t_tensor = torch.tensor([t_i] * BATCH)
            model_mean, _ = diff.p_mean_variance(lambda x, t, *a, **k: x_start, x_cur, alpha_l, t_tensor)
            x_cur = model_mean
        assert torch.allclose(x_cur, x_start, atol=1e-8), f"anchor_w={anchor_w} seed={seed}: khong tai tao dung x_start"
print("  OK - moi to hop (anchor_w, alpha_l, seed) deu tai tao chinh xac x_start")

# ---- Test 6: bien ----
print("== Test 6: bien - user rong va anchor_w rat lon ==")
x_zero, u_zero, i_zero = make_batch(zero_row=True)
diffZ = GaussianDiffusionAnchorOT(SIGMA_MIN, STEPS, anchor_w=2.0)
dl, gl = diffZ.training_losses(fake_model_factory(), x_zero, itmEmbeds, batch_index, itmEmbeds, u_zero)
check_finite("diff_loss(user rong)", dl)
check_finite("gc_loss(user rong)", gl)
out_z = diffZ.p_sample(fake_model_factory(scale=0.9), x_zero, u_zero, i_zero, steps=2, sampling_noise=False)
check_finite("p_sample(user rong)", out_z)

diffBig = GaussianDiffusionAnchorOT(SIGMA_MIN, STEPS, anchor_w=1e6)
out_big = diffBig.p_sample(fake_model_factory(scale=0.7), x_start, uEmbeds_batch, iEmbeds, steps=1, sampling_noise=False)
check_finite("p_sample(anchor_w=1e6)", out_big)
print("  OK - khong NaN/Inf")

print("\nTAT CA 6 PHEP KIEM TRA TREN CPU DEU PASS.")
