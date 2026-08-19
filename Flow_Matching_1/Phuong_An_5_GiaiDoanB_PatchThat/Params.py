import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=1024, type=int, help='batch size')
	parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
	parser.add_argument('--reg', default=1e-5, type=float, help='weight decay regularizer')
	parser.add_argument('--epoch', default=50, type=int, help='number of epochs')
	parser.add_argument('--latdim', default=64, type=int, help='embedding size')
	parser.add_argument('--gnn_layer', default=1, type=int, help='number of gnn layers')
	parser.add_argument('--topk', default=20, type=int, help='K of top K')
	parser.add_argument('--data', default='allrecipes', type=str, help='name of dataset')
	parser.add_argument('--ssl_reg', default=1e-2, type=float, help='weight for contrative learning')
	parser.add_argument('--temp', default=0.5, type=float, help='temperature in contrastive learning')
	parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
	parser.add_argument('--gpu', default='0', type=str, help='indicates which gpu to use')
	parser.add_argument("--seed", type=int, default=421, help="random seed")

	parser.add_argument('--keepRate', default=0.5, type=float, help='ratio of edges to keep')
	
	parser.add_argument('--dims', type=str, default='[1000]')
	parser.add_argument('--d_emb_size', type=int, default=10)
	parser.add_argument('--norm', type=bool, default=False)
	parser.add_argument('--steps', type=int, default=5)
	parser.add_argument('--noise_scale', type=float, default=0.1)
	parser.add_argument('--noise_min', type=float, default=0.0001)
	parser.add_argument('--noise_max', type=float, default=0.02)
	parser.add_argument('--sampling_noise', type=bool, default=False)
	parser.add_argument('--sampling_steps', type=int, default=0)

	# [Phuong an 5 - Giai doan B] duong OT dieu kien-modal (Phuong_An_5_..._KeHoachChiTiet.md).
	# sigma_min/w_clip: y het Phuong an 1/2/3. kappa=0.0 la MAC DINH AN TOAN (tuong duong Phuong an 1,
	# da kiem chung hoi quy trong Model.py) - chi tang kappa khi da xac nhan kappa=0 chay dung.
	parser.add_argument('--sigma_min', type=float, default=0.001)
	parser.add_argument('--w_clip', type=float, default=50.0)
	parser.add_argument('--kappa', type=float, default=0.0, help='cuong do dieu kien-modal; 0 = tat, trung khit Phuong an 1')
	parser.add_argument('--g_min', type=float, default=0.5, help='chan duoi cua so mu g(u,i)')
	parser.add_argument('--g_max', type=float, default=2.0, help='chan tren cua so mu g(u,i)')
	parser.add_argument('--use_msi', type=int, default=0, help='1 = giu them gc_loss (MSI) song song voi duong di dieu kien-modal, 0 = tat (mac dinh)')

	parser.add_argument('--rebuild_k', type=int, default=1)
	parser.add_argument('--e_loss', type=float, default=0.1)
	parser.add_argument('--ris_lambda', type=float, default=0.5)
	parser.add_argument('--ris_adj_lambda', type=float, default=0.2)
	parser.add_argument('--trans', type=int, default=0, help='0: R*R, 1: Linear, 2: allrecipes')
	parser.add_argument('--cl_method', type=int, default=0, help='0:m vs m ; 1:m vs main')
	return parser.parse_args()
args = ParseArgs()
