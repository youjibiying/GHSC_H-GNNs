import os, sys
import math, time, random
import pickle
import argparse, configargparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric

from tqdm import tqdm

from models import SetGNN, HCHA, HNHN, HyperGCN, HyperSAGE, \
    LEGCN, UniGCNII, HyperND, EquivSetGNN
from models.get_norm_from_H import _generate_G_from_H, _generate_G_from_H_with_dist

import datasets
import utils


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    torch.cuda.current_device()
    torch.cuda._initialized = True

@torch.no_grad()
def evaluate(model, data, split_idx, evaluator, loss_fn=None, return_out=False):
    model.eval()
    out = model(data)
    out = F.log_softmax(out, dim=1)

    train_acc = evaluator.eval(data.y[split_idx['train']], out[split_idx['train']])['acc']
    valid_acc = evaluator.eval(data.y[split_idx['valid']], out[split_idx['valid']])['acc']
    test_acc = evaluator.eval(data.y[split_idx['test']], out[split_idx['test']])['acc']

    ret_list = [train_acc, valid_acc, test_acc]

    # Also keep track of losses
    if loss_fn is not None:
        train_loss = loss_fn(out[split_idx['train']], data.y[split_idx['train']])
        valid_loss = loss_fn(out[split_idx['valid']], data.y[split_idx['valid']])
        test_loss = loss_fn(out[split_idx['test']], data.y[split_idx['test']])
        ret_list += [train_loss, valid_loss, test_loss]

    if return_out:
        ret_list.append(out)

    return ret_list
import csv
def data_write_csv(filename, data):
    print(data)
    with open(filename, "a", encoding="utf-8", newline='') as w:
        writer = csv.writer(w)
        writer.writerow(data)

def main(args):

    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print(args)
    if args.method not in ['HyperGCN', 'HyperSAGE']:
        transform = torch_geometric.transforms.Compose([datasets.AddHypergraphSelfLoops()])
    else:
        transform = None

    if args.dname in datasets.HypergraphDataset.Visual_list:
        os.system(f"rm -r {args.data_dir}")
    data = datasets.HypergraphDataset(root=args.data_dir, name=args.dname, path_to_download=args.raw_data_dir,
        feature_noise=args.feature_noise, transform=transform, args=args).data

    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data = SetGNN.norm_contruction(data, option=args.normtype)
    elif args.method == 'HNHN':

        data = HNHN.generate_norm(data, args)
    elif args.method == 'HyperSAGE':
        data = HyperSAGE.generate_hyperedge_dict(data)
    elif args.method == 'LEGCN':
        data = LEGCN.line_expansion(data)
    elif args.method.startswith("H_"):

        data = _generate_G_from_H(data, args=args)

    data = data.to(device)

    # Get splits
    split_idx_lst = []
    # no_random_split = True
    if args.no_random_split:
        if os.path.exists(os.path.join(args.data_dir, "splits")):
            for run in range(10):
                split_idx = utils.get_split_from_path(file=os.path.join(args.data_dir, "splits", str(run+1) + ".pickle"))
                split_idx_lst.append(split_idx)
        elif "train_idx" in data:
            split_idx_lst = 10* [{"train": data.train_idx, "valid": data.test_idx, "test": data.test_idx}]


    else:
        for run in range(args.runs):
            split_idx = utils.rand_train_test_idx(
                data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
            split_idx_lst.append(split_idx)

    if args.method == 'AllSetTransformer':
        if args.AllSet_LearnMask:
            model = SetGNN(data.num_features, data.num_classes, args, data.norm)
        else:
            model = SetGNN(data.num_features, data.num_classes, args)
    elif args.method == 'AllDeepSets':
        args.AllSet_PMA = False
        args.aggregate = 'add'
        if args.AllSet_LearnMask:
            model = SetGNN(data.num_features, data.num_classes, args, data.norm)
        else:
            model = SetGNN(data.num_features, data.num_classes, args)
    elif args.method in ['HGNN', 'HCHA']: # Hypergraph Convolution and Hypergraph Attention
        model = HCHA(data.num_features, data.num_classes, args)
    elif args.method in 'HNHN':
        model = HNHN(data.num_features, data.num_classes, args)
    elif args.method in 'HyperGCN':
        model = HyperGCN(data.num_features, data.num_classes, args)
    elif args.method == 'HyperSAGE':
        model = HyperSAGE(data.num_features, data.num_classes, args)
    elif args.method == 'LEGCN':
        model = LEGCN(data.num_features, data.num_classes, args)
    elif args.method == 'UniGCNII':
        model = UniGCNII(data.num_features, data.num_classes, args)
    elif args.method == 'HyperND':
        model = HyperND(data.num_features, data.num_classes, args)
    elif args.method == 'EDGNN':
        model = EquivSetGNN(data.num_features, data.num_classes, args)
    elif args.method.upper()== "H_GCNII":
        from models.gcnii import H_GCNII
        model = H_GCNII(nfeat = data.num_features,nlayers=args.degree, nclass=data.num_classes, nhidden =args.MLP_hidden,
                        dropout=args.dropout, lamda=args.H_GNN_lamda, alpha=args.H_GNN_alpha,args=args)
    elif args.method.upper()=='H_REVGNN':
        from models.revgnn import H_RevGNN
        model = H_RevGNN(
            in_channels=data.num_features,
            hidden_channels=args.MLP_hidden,
            out_channels=data.num_classes,
            num_layers=args.degree,  # You can try 1000 layers for fun
            dropout=0.2,
            num_groups=2,
        )
    else:
        raise ValueError(f'Undefined model name: {args.method}')
    model = model.to(device)
    print("# Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(model)
    logger = utils.Logger(args.runs, args)
    
    loss_fn = nn.NLLLoss()
    evaluator = utils.NodeClsEvaluator()

    runtime_list = []
    for run in range(args.runs):
        if "train_idx" in data: setup_seed(run)
        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        model.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        best_val = float('-inf')
        for epoch in range(args.epochs):
            # Training loop
            model.train()
            optimizer.zero_grad()
            out = model(data)
            out = F.log_softmax(out, dim=1)
            loss = loss_fn(out[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()

            # Evaluation and logging
            result = evaluate(model, data, split_idx, evaluator, loss_fn)
            logger.add_result(run, *result[:3])
            if epoch % args.display_step == 0 and args.display_step > 0:
                print(f'Run: {run:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss:.4f}, '
                      f'Valid Loss: {result[4]:.4f}, '
                      f'Test Loss: {result[5]:.4f}, '
                      f'Train Acc: {100 * result[0]:.2f}%, '
                      f'Valid Acc: {100 * result[1]:.2f}%, '
                      f'Test Acc: {100 * result[2]:.2f}%')

        end_time = time.time()
        runtime_list.append(end_time - start_time)
    print(args.data_dir,args.method)
    result = logger.print_statistics()
    data_write_csv("results.csv", [ f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", args.data_dir,args.method,f"Final Test: {result['test_mean']:.2f} $\pm$ {result['test_std']:.2f}",
                                      f"Highest Train: {result['max_train_mean']:.2f} $\pm$ {result['max_train_std']:.2f}  "
                    f"Highest Valid: {result['valid_mean']:.2f} $\pm$ {result['valid_std']:.2f}  "
                    f"Final Train: {result['train_mean']:.2f} $\pm$ {result['train_std']:.2f}  ",f"avg_time: {np.mean(runtime_list)} $\pm$ {np.std(runtime_list)} ",str(args)
                   ])

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--config', is_config_file=True)

    # Dataset specific arguments
    parser.add_argument('--dname', default='walmart-trips-100')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--raw_data_dir', type=str, required=True)
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--feature_noise', default='1', type=str, help='std for synthetic feature noise')
    parser.add_argument('--normtype', default='all_one', choices=['all_one','deg_half_sym'])
    parser.add_argument('--add_self_loop', action='store_false')
    parser.add_argument('--exclude_self', action='store_true', help='whether the he contain self node or not')

    # Training specific hyperparameters
    parser.add_argument('--epochs', default=500, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--runs', default=10, type=int)
    parser.add_argument('--cuda', default=0, type=int)

    parser.add_argument('--input_dropout', default=0.2, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    parser.add_argument('--display_step', type=int, default=50)

    # Model common hyperparameters
    parser.add_argument('--method', default='EDGNN', help='model type')
    parser.add_argument('--All_num_layers', default=2, type=int, help='number of basic blocks')
    parser.add_argument('--MLP_num_layers', default=2, type=int, help='layer number of mlps')
    parser.add_argument('--MLP_hidden', default=64, type=int, help='hidden dimension of mlps')
    parser.add_argument('--Classifier_num_layers', default=2,
                        type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=64,
                        type=int)  # Decoder hidden units
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    parser.add_argument('--normalization', default='ln', choices=['bn','ln','None'])
    parser.add_argument('--activation', default='relu', choices=['Id','relu', 'prelu'])
    
    # Args for EDGNN
    parser.add_argument('--MLP2_num_layers', default=-1, type=int, help='layer number of mlp2')
    parser.add_argument('--MLP3_num_layers', default=-1, type=int, help='layer number of mlp3')
    parser.add_argument('--edconv_type', default='EquivSet', type=str, choices=['EquivSet', 'JumpLink', 'MeanDeg', 'Attn', 'TwoSets'])
    parser.add_argument('--restart_alpha', default=0.5, type=float)

    # Args for AllSet
    parser.add_argument('--AllSet_input_norm', default=True)
    parser.add_argument('--AllSet_GPR', action='store_false')  # skip all but last dec
    parser.add_argument('--AllSet_LearnMask', action='store_false')
    parser.add_argument('--AllSet_PMA', action='store_true')
    parser.add_argument('--AllSet_num_heads', default=1, type=int)
    # Args for CEGAT
    parser.add_argument('--output_heads', default=1, type=int)  # Placeholder
    # Args for HyperGCN
    parser.add_argument('--HyperGCN_mediators', action='store_true')
    parser.add_argument('--HyperGCN_fast', action='store_true')
    # Args for HyperSAGE
    parser.add_argument('--HyperSAGE_power', default=1., type=float)
    parser.add_argument('--HyperSAGE_num_sample', default=100, type=int)


    # Args for HNHN
    parser.add_argument('--HNHN_alpha', default=-1.5, type=float)
    parser.add_argument('--HNHN_beta', default=-0.5, type=float)
    parser.add_argument('--HNHN_nonlinear_inbetween', default=True, type=bool)
    # Args for HCHA
    parser.add_argument('--HCHA_symdegnorm', action='store_true')
    # Args for UniGNN
    parser.add_argument('--UniGNN_use_norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--UniGNN_degV', default = 0)
    parser.add_argument('--UniGNN_degE', default = 0)
    # Args for HyperND
    parser.add_argument('--HyperND_ord', default = 1., type=float)
    parser.add_argument('--HyperND_tol', default = 1e-4, type=float)
    parser.add_argument('--HyperND_steps', default = 100, type=int)

    # ARGS  FOR H-GNNS
    parser.add_argument('--sigma', default=-1, type=float)
    parser.add_argument('--no_random_split', default=False, action='store_true')
    parser.add_argument('--degree', type=int, default=32,
                        help='degree of the approximation.')
    parser.add_argument('--dropout', default=0.5, type=float)  # H-GCNII default0.5
    parser.add_argument('--H_GNN_lamda', default=0.5, type=float)  # H-GCNII default0.5
    parser.add_argument('--H_GNN_alpha', default=0.1, type=float)  # H-GCNII default0.5

    parser.add_argument('--no_mvcnn_feature_structure', action='store_true', default=False,
                        help='no_use_mvcnn_feature_for_structure')
    parser.add_argument('--no_gvcnn_feature_structure', action='store_true', default=False,
                        help='use_gvcnn_feature_for_structure')
    parser.add_argument('--no_use_gvcnn_feature', action='store_true', default=False,
                        help='use_gvcnn_feature_add to features X')
    parser.add_argument('--no_use_mvcnn_feature', action='store_true', default=False,
                        help='use_gvcnn_feature_add  to features X')

    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(AllSet_GPR=False)
    parser.set_defaults(AllSet_LearnMask=False)
    parser.set_defaults(AllSet_PMA=True)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HyperGCN_fast=True)
    parser.set_defaults(HCHA_symdegnorm=False)
    
    #     Use the line below for .py file
    args = parser.parse_args()
    #     Use the line below for notebook
    # args = parser.parse_args([])
    # args, _ = parser.parse_known_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)
