import os
import torch
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_args():
    parser = argparse.ArgumentParser(description="'Model Params'")
    parser.add_argument('--bpr_batch', type=int, default=2048)
    parser.add_argument('--recdim', type=int, default=64)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=int, default=1)
    parser.add_argument('--keepprob', type=float, default=0.7)
    parser.add_argument('--a_fold', type=int, default=100)
    parser.add_argument('--testbatch', type=int, default=4096)
    parser.add_argument('--dataset', type=str, default='Tmall')
    parser.add_argument('--topks', nargs='?', default="[20]")
    parser.add_argument('--tensorboard', type=int, default=0)
    parser.add_argument('--comment', type=str, default="lgn")
    parser.add_argument('--multicore', type=int, default=0)
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--test_file', type=str, default='test.txt')

    parser.add_argument('--noise_eps', default=0.1, type=float)
    parser.add_argument('--tail_k_threshold', default=20, type=float, help='deg')
    parser.add_argument('--agg_function', default=0, type=int)
    parser.add_argument('--use_relation_rf', default=True, type=str)
    parser.add_argument('--agg_w_exp_scale', default=20, type=int)
    parser.add_argument('--l2_reg', default=0.0001, type=float)
    parser.add_argument('--layer_gnn', default=3, type=int)
    parser.add_argument('--m_h_eps', default=0.001, type=float)
    parser.add_argument('--kl_eps', default=10, type=float)
    parser.add_argument('--neighbor_norm_type', default="left", type=str)
    parser.add_argument('--annealing_type', default=1, type=int)
    parser.add_argument('--lambda_gp', default=1, type=float)

    return parser.parse_args()


args = parse_args()

config = {}
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers'] = args.layer
config['dropout'] = args.dropout
config['keep_prob'] = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
entity_num_per_item = 10
kgc_temp = 0.2
kg_p_drop = 0.5
dataset = args.dataset
test_file = "/" + args.test_file
