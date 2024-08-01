import copy

from sklearn.metrics import f1_score

from DataLoader import *
from utils import *
import time
import argparse
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import GCN, MvGCN, MLP, MvGCN_noMLP, MvGCN_noCon
from loss import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=44, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.03,  # 0.03
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default="Minesweeper",
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--lamda', type=float, default=0.04,
                    help='parameter of contrast loss.')
parser.add_argument('--mu', type=float, default=1,
                    help='parameter of train loss.')
parser.add_argument('--num_train_per_class', type=int, default=20,
                    help='num_train_per_class')
# dataset
parser.add_argument('--num_val', type=int, default=500,
                    help='num_val')
parser.add_argument('--num_test', type=int, default=1500,
                    help='num_test')
parser.add_argument('--train_ratio', type=float, default=0.6,
                    help='train_ratio')
parser.add_argument('--valid_ratio', type=float, default=0.2,
                    help='valid_ratio')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='test_ratio')
parser.add_argument('--data_split_mode', type=str, default='Ratio',
                    help='data_split_mode')  #Num or Ratio
args = parser.parse_args()

# data_name = "Film"

args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# load data
features, labels, adj = loadMatData(dataset=args.dataset)
adj = torch.tensor(adj)
tmp_labels = copy.deepcopy(labels)
tmp_labels = tmp_labels.cuda()

# 划分数据
idx_train, idx_val, idx_test, train_mask = generate_permutation(labels, args)

# 用特征构造邻接矩阵
adj_k = np.matmul(features, features.T)
adj_k = torch.sigmoid_(adj_k)
adj_k = torch.tensor(adj_k)
sp_adj = scipy.sparse.coo_matrix(adj_k)
sp_adj = aug_normalized_adjacency(sp_adj)
adj_k = sparse_mx_to_torch_sparse_tensor(sp_adj, device='cuda')

# Model and optimizer
model = MvGCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
            lr=args.lr, weight_decay=args.weight_decay)

model_MLP = MLP(nfeat=features.shape[1],
            nhid=256,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer_mlp = optim.Adam(model_MLP.parameters(), lr=args.lr, weight_decay=0.02)

if args.cuda:
    model.cuda()
    model_MLP.cuda()
    features = features.cuda()
    adj = adj.cuda()
    adj_k = adj_k.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

 ## MLP pre-train
for i in range(20):
    model_MLP.train()
    optimizer_mlp.zero_grad()
    output = model_MLP(features)
    # loss = F.nll_loss(output[idx_train], labels[idx_train])
    loss = F.cross_entropy(output[idx_train], labels[idx_train])
    acc = accuracy(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer_mlp.step()
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model_MLP(features)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print('epoch:{}'.format(i+1),
            'loss: {:.4f}'.format(loss.item()),
            'acc: {:.4f}'.format(acc.item()),
            'val: {:.4f}'.format(acc_val.item()),
            'test: {:.4f}'.format(acc_test.item()))


def train(con_epoch):
    t = time.time()
    # model.train()
    # model_MLP.train()
    best_test_acc = 0
    best_f1 = 0
    acc_val_t = []

    for epoch in range(con_epoch):
        model.train()
        model_MLP.train()
        # caculate L
        # L = generate_similarity_matrix(labels, train_mask)

        optimizer.zero_grad()
        optimizer_mlp.zero_grad()
        # MLP
        output_mlp = model_MLP(features)

        loss_mlp = F.cross_entropy(output_mlp[idx_train], labels[idx_train])
        # loss_mlp = F.nll_loss(output_mlp[idx_train], labels[idx_train])
        output_mlp = torch.tanh(output_mlp)
        # MyGCN
        output, x, k = model(features, adj, adj_k, output_mlp)


        loss1 = ucloss(x, k)  # contrastive loss

        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss =loss_mlp + args.lamda * loss1 + args.mu * loss_train  # total_loss

        loss.backward()
        optimizer.step()
        optimizer_mlp.step()

        # if i % 100 == 0:
        #     print("Epoch: ", epoch + 1, " Iteration: ", i + 1, " Loss: ", loss.item())
        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            model_MLP.eval()

            output_mlp = model_MLP(features)
            output, x, k = model(features, adj, adj_k, output_mlp)



        loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_val_t.append(acc_val)


        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        if acc_test > best_test_acc:
            best_test_acc = acc_test

        label_max = []
        for idx in idx_test:
            label_max.append(torch.argmax(output[idx]).item())
        labelcpu = labels[idx_test].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        if macro_f1 > best_f1:
            best_f1 = macro_f1
        if epoch + 1 == con_epoch:
            # index = torch.cat([idx_val, idx_train], dim=0)
            # draw_plt(output, labels.cpu(), index.cpu(), args.dataset)
            f = open("./result.txt", 'a')
            f.write(f"dataset: {args.dataset}, lamda: {args.lamda}, mu: {args.mu}, , acc_test: {acc_test.item()}, "
                    f"best_acc_test: {best_test_acc}, macro_f1: {macro_f1.item()}, best_f1: {best_f1}\n")
            f.close()


        print('Epoch: {:04d}'.format(epoch + 1),
              'ce_loss: {:.4f}'.format(loss_train.item()),
              'uc_loss: {:.4f}'.format(loss1.item()),
              'loss: {:.4f}'.format(loss.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'loss_test: {:.4f}'.format(loss_test.item()),
              'acc_test: {:.4f}'.format(acc_test.item()),
              'best_acc_test: {:.4f}'.format(best_test_acc),
              'macro_f1: {:.4f}'.format(macro_f1.item()),
              'best_f1: {:.4f}'.format(best_f1))


def test():
    model.eval()
    model_MLP.eval()
    output_mlp = model_MLP(features)
    output, x, k= model(features, adj, adj_k, output_mlp)
    # loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    # loss_mlp = F.cross_entropy(output_mlp[idx_train], labels[idx_train])
    loss_mlp = F.nll_loss(output_mlp[idx_train], labels[idx_train])
    loss1_test = ucloss(x, k)
    loss2_test = ucloss(k, output_mlp)

    loss_test = F.cross_entropy(output[idx_train], labels[idx_train])
    loss_test = loss_mlp + args.mu * loss1_test + loss_test
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()

# training
train(args.epochs)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing

# test()