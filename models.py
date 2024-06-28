import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
import numpy as np
from numpy import random


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        output = F.relu(self.gc1(x, adj))
        output = F.dropout(output, self.dropout, training=self.training)
        output = self.gc2(output, adj)
        # output = F.softmax(output, dim=1)

        return output


class GCN_multi(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nview):
        super(GCN_multi, self).__init__()
        self.GCNlist = torch.nn.ModuleList()
        self.dropout = dropout
        for dv in range(nview):
            self.GCNlist.append(GCN(nfeat, nhid, nclass, dropout))

    def forward(self, x, adj):
        GCN_outputs = []
        for idx, model in enumerate(self.GCNlist):
            tmp_output = model(x, adj[idx].cuda())
            GCN_outputs.append(tmp_output)
        output = torch.stack(GCN_outputs, dim=1)
        output = output.sum(1)
        output = F.dropout(output, self.dropout, training=self.training)
        # output = torch.sigmoid(output)
        # output = F.softmax(output, dim=1)

        return output


class GCN_1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_1, self).__init__()

        self.gc1 = myGraphConvolution(nfeat, nhid)
        self.gc2 = myGraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, adj_homo):
        output = F.relu(self.gc1(x, adj, adj_homo))
        output = F.dropout(output, self.dropout, training=self.training)
        output = self.gc2(output, adj, adj_homo)
        # output = F.softmax(output, dim=1)

        return output


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(nfeat, nhid),
            #nn.ReLU(),
            nn.Linear(nhid, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.mlp(x)

    def get_emb(self, x):
        return self.mlp[0](x).detach()


class MvGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MvGCN, self).__init__()
        self.gc1 = GCN_1(nfeat, nhid, nclass, dropout)
        self.gc2 = GCN_1(nfeat, nhid, nclass, dropout)
        self.fusion = nn.Parameter(torch.FloatTensor(2, 1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fusion.size(1))
        self.fusion.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, adj_k, output_mlp):
        output_m = output_mlp.exp()
        homo_matrix = torch.matmul(output_m, output_m.t())  # mlp的表示相乘，即S
        # 将矩阵保存为文本文件
        # torch.save(homo_matrix.to(torch.device('cpu')), "myTensor.pth")
        adj = adj.cuda()
        adj_homo = torch.mul(adj, homo_matrix)
        # adj_homo = adj @ homo_matrix  # multi
        # adj_k_homo = torch.mul(adj_k, homo_matrix)

        x1 = self.gc1(x, adj, adj_homo)
        x2 = self.gc2(x, adj_k, adj_k)
        Mv_outputs = [x1, x2]
        output = torch.stack(Mv_outputs, dim=1)

        output = self.fusion * output
        output = output.sum(1)
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        # output = F.softmax(output, dim=1)
        output = torch.sigmoid(output)
        return output, x1, x2


class MvGCN_multi(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nview):
        super(MvGCN_multi, self).__init__()
        self.dropout = dropout
        self.MvGCNlist = torch.nn.ModuleList()
        for dv in range(nview):
            self.MvGCNlist.append(MvGCN(nfeat, nhid, nclass, dropout))
        # self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.IMNNs)):
            self.MvGCNlist[i].reset_parameters()

    def forward(self, X, adj, adj_k, output_mlp):
        MvGCN_outputs = []
        x1 = []
        x2 = []
        for idx, model in enumerate(self.MvGCNlist):
            tmp_output, tmp_x1, tmp_x2 = model(X, adj[idx], adj_k, output_mlp)
            MvGCN_outputs.append(tmp_output)
            x1.append(tmp_x1)
            x2.append(tmp_x2)
        output = torch.stack(MvGCN_outputs, dim=1)
        output = output.sum(1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.sigmoid(output)
        return output, x1, x2


class MvGCN_noMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MvGCN_noMLP, self).__init__()
        self.gc1 = GCN_1(nfeat, nhid, nclass, dropout)
        self.gc2 = GCN_1(nfeat, nhid, nclass, dropout)
        self.fusion = nn.Parameter(torch.FloatTensor(2, 1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fusion.size(1))
        self.fusion.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, adj_k):
        x1 = self.gc1(x, adj, adj)
        x2 = self.gc2(x, adj_k, adj_k)
        Mv_outputs = [x1, x2]
        output = torch.stack(Mv_outputs, dim=1)

        output = self.fusion * output
        output = output.sum(1)
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        # output = F.softmax(output, dim=1)
        return output, x1, x2

class MvGCN_noCon(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MvGCN_noCon, self).__init__()
        self.gc1 = GCN_1(nfeat, nhid, nclass, dropout)
        self.gc2 = GCN_1(nfeat, nhid, nclass, dropout)
        self.fusion = nn.Parameter(torch.FloatTensor(2, 1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fusion.size(1))
        self.fusion.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, output_mlp):
        output_m = output_mlp.exp()
        homo_matrix = torch.matmul(output_m, output_m.t())  # mlp的表示相乘，即S
        # 将矩阵保存为文本文件
        # torch.save(homo_matrix.to(torch.device('cpu')), "myTensor.pth")

        adj_homo = torch.mul(adj, homo_matrix)

        x1 = self.gc1(x, adj, adj_homo)
        # x2 = self.gc2(x, adj_k, adj_k)
        # Mv_outputs = [x1, x2]
        # output = torch.stack(Mv_outputs, dim=1)

        # output = self.fusion * output
        # output = output.sum(1)
        x1 = F.softmax(x1, dim=1)
        # x2 = F.softmax(x2, dim=1)
        # output = F.softmax(output, dim=1)
        return x1