import paddle
import os
import argparse
from solver_encoder import Solver
from data_loader import get_loader


def str2bool(v):
    return v.lower() in 'true'


def main(config):
    vcc_loader = get_loader(config.data_dir, config.batch_size, config.len_crop
        )
    solver = Solver(vcc_loader, config)
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_cd', type=float, default=1, help=
        'weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=16)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=16)
    parser.add_argument('--data_dir', type=str, default='./spmel')
    parser.add_argument('--batch_size', type=int, default=2, help=
        'mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help=
        'number of total iterations')
    parser.add_argument('--len_crop', type=int, default=128, help=
        'dataloader output sequence length')
    parser.add_argument('--log_step', type=int, default=10)
    config = parser.parse_args()
    print(config)
    main(config)
