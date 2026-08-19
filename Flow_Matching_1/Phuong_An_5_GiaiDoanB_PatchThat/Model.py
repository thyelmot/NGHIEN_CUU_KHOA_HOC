import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
import math
from Utils.Utils import *

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
	def __init__(self, image_embedding, text_embedding, audio_embedding=None):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

		self.edgeDropper = SpAdjDropEdge(args.keepRate)

		if args.trans == 1:
			self.image_trans = nn.Linear(args.image_feat_dim, args.latdim)
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
		elif args.trans == 0:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Parameter(init(torch.empty(size=(args.text_feat_dim, args.latdim))))
		else:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
		if audio_embedding != None:
			if args.trans == 1:
				self.audio_trans = nn.Linear(args.audio_feat_dim, args.latdim)
			else:
				self.audio_trans = nn.Parameter(init(torch.empty(size=(args.audio_feat_dim, args.latdim))))

		self.image_embedding = image_embedding
		self.text_embedding = text_embedding
		if audio_embedding != None:
			self.audio_embedding = audio_embedding
		else:
			self.audio_embedding = None

		if audio_embedding != None:
			self.modal_weight = nn.Parameter(torch.Tensor([0.3333, 0.3333, 0.3333]))
		else:
			self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
		self.softmax = nn.Softmax(dim=0)

		self.dropout = nn.Dropout(p=0.1)

		self.leakyrelu = nn.LeakyReLU(0.2)
				
	def getItemEmbeds(self):
		return self.iEmbeds
	
	def getUserEmbeds(self):
		return self.uEmbeds
	
	def getImageFeats(self):
		if args.trans == 0 or args.trans == 2:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			return image_feats
		else:
			return self.image_trans(self.image_embedding)
	
	def getTextFeats(self):
		if args.trans == 0:
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
			return text_feats
		else:
			return self.text_trans(self.text_embedding)

	def getAudioFeats(self):
		if self.audio_embedding == None:
			return None
		else:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)
		return audio_feats

	def forward_MM(self, adj, image_adj, text_adj, audio_adj=None):
		if args.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		elif args.trans == 1:
			image_feats = self.image_trans(self.image_embedding)
			text_feats = self.text_trans(self.text_embedding)
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.text_trans(self.text_embedding)

		if audio_adj != None:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)

		weight = self.softmax(self.modal_weight)

		embedsImageAdj = torch.concat([self.uEmbeds, self.iEmbeds])
		embedsImageAdj = torch.spmm(image_adj, embedsImageAdj)

		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
		embedsImage = torch.spmm(adj, embedsImage)

		embedsImage_ = torch.concat([embedsImage[:args.user], self.iEmbeds])
		embedsImage_ = torch.spmm(adj, embedsImage_)
		embedsImage += embedsImage_
		
		embedsTextAdj = torch.concat([self.uEmbeds, self.iEmbeds])
		embedsTextAdj = torch.spmm(text_adj, embedsTextAdj)

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		embedsText = torch.spmm(adj, embedsText)

		embedsText_ = torch.concat([embedsText[:args.user], self.iEmbeds])
		embedsText_ = torch.spmm(adj, embedsText_)
		embedsText += embedsText_

		if audio_adj != None:
			embedsAudioAdj = torch.concat([self.uEmbeds, self.iEmbeds])
			embedsAudioAdj = torch.spmm(audio_adj, embedsAudioAdj)

			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(adj, embedsAudio)

			embedsAudio_ = torch.concat([embedsAudio[:args.user], self.iEmbeds])
			embedsAudio_ = torch.spmm(adj, embedsAudio_)
			embedsAudio += embedsAudio_

		embedsImage += args.ris_adj_lambda * embedsImageAdj
		embedsText += args.ris_adj_lambda * embedsTextAdj
		if audio_adj != None:
			embedsAudio += args.ris_adj_lambda * embedsAudioAdj
		if audio_adj == None:
			embedsModal = weight[0] * embedsImage + weight[1] * embedsText
		else:
			embedsModal = weight[0] * embedsImage + weight[1] * embedsText + weight[2] * embedsAudio

		embeds = embedsModal
		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst)

		embeds = embeds + args.ris_lambda * F.normalize(embedsModal)

		return embeds[:args.user], embeds[args.user:]

	def forward_cl_MM(self, adj, image_adj, text_adj, audio_adj=None):
		if args.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		elif args.trans == 1:
			image_feats = self.image_trans(self.image_embedding)
			text_feats = self.text_trans(self.text_embedding)
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.text_trans(self.text_embedding)

		if audio_adj != None:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)

		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
		embedsImage = torch.spmm(image_adj, embedsImage)

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		embedsText = torch.spmm(text_adj, embedsText)

		if audio_adj != None:
			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(audio_adj, embedsAudio)

		embeds1 = embedsImage
		embedsLst1 = [embeds1]
		for gcn in self.gcnLayers:
			embeds1 = gcn(adj, embedsLst1[-1])
			embedsLst1.append(embeds1)
		embeds1 = sum(embedsLst1)

		embeds2 = embedsText
		embedsLst2 = [embeds2]
		for gcn in self.gcnLayers:
			embeds2 = gcn(adj, embedsLst2[-1])
			embedsLst2.append(embeds2)
		embeds2 = sum(embedsLst2)

		if audio_adj != None:
			embeds3 = embedsAudio
			embedsLst3 = [embeds3]
			for gcn in self.gcnLayers:
				embeds3 = gcn(adj, embedsLst3[-1])
				embedsLst3.append(embeds3)
			embeds3 = sum(embedsLst3)

		if audio_adj == None:
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:]
		else:
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:], embeds3[:args.user], embeds3[args.user:]

	def reg_loss(self):
		ret = 0
		ret += self.uEmbeds.norm(2).square()
		ret += self.iEmbeds.norm(2).square()
		return ret

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return torch.spmm(adj, embeds)

class SpAdjDropEdge(nn.Module):
	def __init__(self, keepRate):
		super(SpAdjDropEdge, self).__init__()
		self.keepRate = keepRate

	def forward(self, adj):
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

		newVals = vals[mask] / self.keepRate
		newIdxs = idxs[:, mask]

		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
		
class Denoise(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
		super(Denoise, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.time_emb_dim = emb_size
		self.norm = norm

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

		in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

		out_dims_temp = self.out_dims

		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

		self.drop = nn.Dropout(dropout)
		self.init_weights()

	def init_weights(self):
		for layer in self.in_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)
		
		for layer in self.out_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		size = self.emb_layer.weight.size()
		std = np.sqrt(2.0 / (size[0] + size[1]))
		self.emb_layer.weight.data.normal_(0.0, std)
		self.emb_layer.bias.data.normal_(0.0, 0.001)

	def forward(self, x, timesteps, mess_dropout=True):
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		emb = self.emb_layer(time_emb)
		if self.norm:
			x = F.normalize(x)
		if mess_dropout:
			x = self.drop(x)
		h = torch.cat([x, emb], dim=-1)
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				h = torch.tanh(h)

		return h

class GaussianDiffusion(nn.Module):
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()

		self.noise_scale = noise_scale
		self.noise_min = noise_min
		self.noise_max = noise_max
		self.steps = steps

		if noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			if beta_fixed:
				self.betas[0] = 0.0001

			self.calculate_for_diffusion()

	def get_betas(self):
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas) 

	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	def p_sample(self, model, x_start, steps, sampling_noise=False):
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t)
		
		indices = list(range(self.steps))[::-1]

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
			if sampling_noise:
				noise = torch.randn_like(x_t)
				nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
				x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
			else:
				x_t = model_mean
		return x_t

	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)

	def p_mean_variance(self, model, x, t):
		model_output = model(x, t, False)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		
		return model_mean, model_log_variance

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
		batch_size = x_start.size(0)

		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
		noise = torch.randn_like(x_start)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start

		model_output = model(x_t, ts)

		mse = self.mean_flat((x_start - model_output) ** 2)

		weight = self.SNR(ts - 1) - self.SNR(ts)
		weight = torch.where((ts == 0), 1.0, weight)

		diff_loss = weight * mse

		usr_model_embeds = torch.mm(model_output, model_feats)
		usr_id_embeds = torch.mm(x_start, itmEmbeds)

		gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

		return diff_loss, gc_loss
		
	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))

	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

class GaussianDiffusionModalOT(GaussianDiffusion):
	"""
	[Phuong an 5 - Phuong_An_5_Modality_Conditioned_OT_KeHoachChiTiet.md - Giai doan B]
	Ke thua truc tiep GaussianDiffusion (khong tu GaussianDiffusionCFM cua Phuong an 2): __init__ o day
	bo qua hoan toan khung VP-style cua lop cha (giong Phuong an 1/3/4), nen ke thua tu
	GaussianDiffusionCFM se khong tai su dung duoc gi ve code, chi them 1 phu thuoc khong can thiet.
	Cong thuc duoc TAI SU DUNG o muc dai so tu Phuong an 1 (duong OT-linear) + Phuong an 2 (trong so CFM
	tong quat, khong phu thuoc duong di cu the) - xem class GaussianDiffusionOT / GaussianDiffusionCFM.

	Y tuong MOI (khong co san trong 2 bai bao goc, tu thiet ke - xem CT-5.1..CT-5.7 trong ban ke hoach,
	da kiem chung gradient bang sai phan huu han o Giai doan A, thu muc Phuong_An_5_GiaiDoanA_GradCheck/):
	so mu (T,) DUNG CHUNG cho moi item cua Phuong an 1, thay bang mu[u,t,i] RIENG cho tung cap
	(user, item) trong batch, qua 1 "so mu" tau(t)^g(u,i), trong do:
		g(u,i)   = clip(exp(-kappa * phi(u,i)), g_min, g_max)
		phi(u,i) = cosine(centroid_modal(u), modal_embed(i))   (centroid_modal(u) = trung binh embedding
		           modal cua nhung item user u da tuong tac, suy tu x_start - CHINH LA du lieu dung de
		           tinh MSI/gc_loss trong code goc, nay dung de DIEU HUONG duong di thay vi lam 1 loss phu)
	kappa=0 => g=1 moi noi => tau(t)^1 = s(t) => CONG THUC TRUNG KHIT Phuong an 1 (kiem chung hoi quy
	o cuoi file nay, ham _pa5_regression_check_vs_pa1, va trong README.md cua Giai doan B).

	Cong tac use_msi (mac dinh False, tuong ung "enable_modal_path" mac dinh kappa=0.0 trong Params.py -
	an toan khi merge vi khong doi hanh vi mac dinh so voi 1 ban GaussianDiffusionOT thuan): Phuong an 5
	de xuat thay MSI (eq 14 DiffMM) bang chinh co che dieu kien-modal nay, nhung van giu gc_loss (MSI)
	lam 1 lua chon ablation - dat use_msi=True de giu ca 2 co che song song khi so sanh.
	"""

	def __init__(self, sigma_min, steps, kappa=0.0, g_min=0.5, g_max=2.0, w_clip=50.0, use_msi=False, affinity_eps=1e-8):
		nn.Module.__init__(self)  # bo qua __init__ cua GaussianDiffusion (khong can beta kieu VP)
		self.steps = steps
		self.sigma_min = sigma_min
		self.kappa = kappa
		self.g_min = g_min
		self.g_max = g_max
		self.w_clip = w_clip
		self.use_msi = use_msi
		self.affinity_eps = affinity_eps
		self.noise_scale = 1.0  # de tuong thich dieu kien "if self.noise_scale != 0" o cac lop khac

		t_idx = torch.arange(steps, dtype=torch.float64)
		s = (1.0 - t_idx / (steps - 1)) if steps > 1 else torch.ones_like(t_idx)
		self.s = s.cuda()  # (T,) - s(t)=1 gan du lieu (t=0), s(t)=0 gan nhieu (t=T-1), giong Phuong an 1

	def _modal_affinity(self, x_start, modal_embeds):
		# phi(u,i) (CT-5.1 + CT-5.2): cosine similarity giua "trong tam modal" cua cac item user u da
		# tuong tac (suy tu x_start, giong cach MSI/gc_loss goc dung x_start) va embedding modal cua item i
		centroid = torch.mm(x_start, modal_embeds)  # (batch, feat_dim)
		centroid = centroid / centroid.norm(dim=-1, keepdim=True).clamp(min=self.affinity_eps)
		embeds_unit = modal_embeds / modal_embeds.norm(dim=-1, keepdim=True).clamp(min=self.affinity_eps)
		return torch.mm(centroid, embeds_unit.t())  # (batch, num_items)

	def _per_item_path(self, x_start, modal_embeds):
		# CT-5.3 -> CT-5.5: tra ve (tau, sigma) dang (batch, T, num_items).
		# QUAN TRONG: tau la HE SO (giong mu_coef cua Phuong an 1/2/3 - KHONG nhan voi x_start o day).
		# x_start chi duoc dung de tinh phi (do phu hop modal), KHONG nhan vao tau - viec nhan x_start
		# (hoac model_output, tuy ngu canh) phai lam RIENG o noi goi (q_sample/p_mean_variance), giong
		# het cach Phuong an 1 dung mu_coef[t]*x_start / mu_coef[t]*model_output. Gop x_start vao tau
		# ngay tai day la 1 loi da phat hien trong ban dau (lam trong so CFM sai tai moi toa do x_start=0,
		# va nhan x_start 2 LAN trong q_sample) - xem README.md muc "Loi da phat hien va sua" cua Giai
		# doan B.
		phi = self._modal_affinity(x_start, modal_embeds)  # (batch, num_items)
		g = torch.exp(-self.kappa * phi).clamp(min=self.g_min, max=self.g_max)  # (batch, num_items)
		s = self.s.to(x_start.dtype)  # (T,)
		tau = s.view(1, self.steps, 1) ** g.unsqueeze(1)  # (batch, T, num_items) - HE SO, chua nhan x_start
		sigma = 1.0 - (1.0 - self.sigma_min) * tau
		return tau, sigma

	def _cfm_weight(self, tau, sigma):
		# CT-5.6: y het sai phan lui cua Phuong an 2 (_precompute_cfm_weight), ap len HE SO tau (khong
		# phai mu=tau*x_start) - dung nhu suy dien: w_CFM(t) = [c'(t) - (sigma'(t)/sigma(t))c(t)]^2 la
		# ham THUAN CUA HE SO c(t)=tau(t), KHONG phu thuoc gia tri du lieu x1 (xem lai suy dien CT-2..CT-4
		# cua Phuong an 2: x1 tu trieu tieu khoi trong so, chi con lai trong (x_hat1-x1)^2).
		tau_prev = torch.cat([tau[:, :1, :], tau[:, :-1, :]], dim=1)
		sigma_prev = torch.cat([sigma[:, :1, :], sigma[:, :-1, :]], dim=1)
		tau_prime = tau_prev - tau
		sigma_prime = sigma_prev - sigma
		w = (tau_prime - (sigma_prime / sigma.clamp(min=1e-8)) * tau) ** 2
		w = torch.cat([torch.ones_like(w[:, :1, :]), w[:, 1:, :]], dim=1)  # bien t=0: quy uoc w=1
		return w.clamp(max=self.w_clip)

	@staticmethod
	def _gather_at_t(arr, t):
		# arr: (batch, T, num_items), t: (batch,) long -> tra ve (batch, num_items) tai chi so t rieng
		# cho tung phan tu batch (dung trong training_losses, moi vi du lay 1 t ngau nhien khac nhau)
		idx = t.view(-1, 1, 1).expand(-1, 1, arr.shape[-1])
		return arr.gather(1, idx).squeeze(1)

	def q_sample(self, x_start, modal_embeds, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		tau, sigma = self._per_item_path(x_start, modal_embeds)
		tau_t = self._gather_at_t(tau, t)
		sigma_t = self._gather_at_t(sigma, t)
		return tau_t * x_start + sigma_t * noise

	def p_mean_variance(self, model, x, t, x_start, modal_embeds, tau=None, sigma=None):
		model_output = model(x, t, False)  # du doan alpha_0 (giong het parameterization cua code goc)

		if tau is None or sigma is None:
			tau, sigma = self._per_item_path(x_start, modal_embeds)
		tau_t = self._gather_at_t(tau, t)
		sigma_t = self._gather_at_t(sigma, t)
		noise_pred = (x - tau_t * model_output) / sigma_t.clamp(min=1e-8)

		t_prev = torch.clamp(t - 1, min=0)
		tau_prev_raw = self._gather_at_t(tau, t_prev)
		sigma_prev_raw = self._gather_at_t(sigma, t_prev)

		is_last_step = (t == 0).float().view(-1, *([1] * (len(x.shape) - 1)))
		# t==0 la buoc khu nhieu cuoi cung -> coi "buoc truoc" la du lieu sach hoan toan (he so=1, sigma=0),
		# y het quy uoc cua Phuong an 1/3
		tau_prev = is_last_step * 1.0 + (1 - is_last_step) * tau_prev_raw
		sigma_prev = is_last_step * 0.0 + (1 - is_last_step) * sigma_prev_raw

		model_mean = tau_prev * model_output + sigma_prev * noise_pred
		model_log_variance = None  # bien the ModalOT chi ho tro suy luan tat dinh (sampling_noise=False)
		return model_mean, model_log_variance

	def p_sample(self, model, x_start, modal_embeds, steps, sampling_noise=False):
		# tinh tau, sigma (batch, T, num_items) 1 LAN duy nhat cho ca vong lap (khong tinh lai moi buoc t)
		tau, sigma = self._per_item_path(x_start, modal_embeds)

		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps - 1] * x_start.shape[0]).cuda()
			tau_t = self._gather_at_t(tau, t)
			sigma_t = self._gather_at_t(sigma, t)
			x_t = tau_t * x_start + sigma_t * torch.randn_like(x_start)

		indices = list(range(self.steps))[::-1]
		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, _ = self.p_mean_variance(model, x_t, t, x_start, modal_embeds, tau=tau, sigma=sigma)
			x_t = model_mean
		return x_t

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
		# model_feats duoc TAI SU DUNG lam modal_embeds (CT-5.1) - khong doi chu ky goi ham o Main.py
		modal_embeds = model_feats
		batch_size = x_start.size(0)

		tau, sigma = self._per_item_path(x_start, modal_embeds)
		w = self._cfm_weight(tau, sigma)  # (batch, T, num_items)

		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
		noise = torch.randn_like(x_start)
		tau_t = self._gather_at_t(tau, ts)
		sigma_t = self._gather_at_t(sigma, ts)
		x_t = tau_t * x_start + sigma_t * noise

		model_output = model(x_t, ts)

		w_t = self._gather_at_t(w, ts)  # (batch, num_items)
		# CT-5.7: trong so ap TRUOC khi gop mean qua chieu item (khac thu tu voi Phuong an 2, vi o day
		# trong so w_t khac nhau THEO TUNG ITEM trong cung 1 vi du, khong con la 1 he so vo huong dung
		# chung cho ca vector nhu Phuong an 1/2/3 - xem CT-5.7 trong ban ke hoach chi tiet)
		diff_loss = self.mean_flat(w_t * (x_start - model_output) ** 2)

		if self.use_msi:
			usr_model_embeds = torch.mm(model_output, model_feats)
			usr_id_embeds = torch.mm(x_start, itmEmbeds)
			gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)
		else:
			gc_loss = torch.zeros(batch_size, device=x_start.device, dtype=x_start.dtype)

		return diff_loss, gc_loss