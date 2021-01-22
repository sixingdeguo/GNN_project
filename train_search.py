from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import *


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return format(acc_test.item())


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args(args=[])
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 模型搜索列表
# model_list = ['GCN', 'GCN3', 'GCN4', 'GCN5', 'GCN10', 'GCN15', 'GCN20', 'GCN25']
model_list = ['ResGCN', 'ResGCN3', 'ResGCN4', 'ResGCN5', 'ResGCN10', 'ResGCN15', 'ResGCN20', 'ResGCN25']
# model_list = ['DenseGCN','DenseGCN3','DenseGCN4','DenseGCN5','DenseGCN10','DenseGCN15','DenseGCN20','DenseGCN25']
data_list = ['cora', 'citeseer', 'pubmed']

# 参数搜索列表
epoch_list = [100, 150, 200, 250, 300, 400]
drop_list = [0.7, 0.8, 0.9]
hidden_list = [24, 32, 40, 48, 56, 64]
res = []
best = []
i = 0
model = GCN(nfeat=1, nhid=1, nclass=1, dropout=0)
for model_name in model_list:
    for data_name in data_list:
        best_acc = 0
        best_data = ''
        best_epoch = 0
        best_hid = 0
        best_drop = 0
        # Load data
        adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=data_name)
        for hidden in hidden_list:
            for drop in drop_list:
                # Model and optimizer
                exec(
                    'model =' + model_name + '(nfeat=features.shape[1], nhid=hidden,nclass=labels.max().item() + 1,dropout=drop)')
                optimizer = optim.Adam(model.parameters(),
                                       lr=args.lr, weight_decay=args.weight_decay)

                if args.cuda:
                    model.cuda()
                    features = features.cuda()
                    adj = adj.cuda()
                    labels = labels.cuda()
                    idx_train = idx_train.cuda()
                    idx_val = idx_val.cuda()
                    idx_test = idx_test.cuda()
                for epoch_num in epoch_list:
                    # Train model
                    t_total = time.time()
                    for epoch in range(epoch_num):
                        train(epoch)
                    print("Optimization Finished!")
                    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

                    # Testing
                    acc = float(test())
                    if acc > best_acc:
                        best_data, best_hid, best_drop, best_epoch, best_acc = data_name, hidden, drop, epoch_num, acc
                    res.append([model_name, data_name, hidden, drop, epoch_num, acc])

        best.append([model_name, best_data, best_hid, best_drop, best_epoch, best_acc])

res = np.array(res)
import pandas as pd

# 储存最优参数结果
res = pd.DataFrame(res, columns=['model', 'dataset', 'hidden', 'drop', 'epoch', 'acc'])
best = pd.DataFrame(best, columns=['model', 'dataset', 'hidden', 'drop', 'epoch', 'acc'])
res.to_csv(model_list[0] + '.csv', index=False)
best.to_csv(model_list[0] + 'best.csv', index=False)
