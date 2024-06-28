import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = input.float()
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj.float(), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class myGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(myGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # neighbor
        self.weight_self = Parameter(torch.FloatTensor(in_features, out_features))  # self
        # self.w = Parameter(torch.FloatTensor(1))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        stdv_self = 1. / math.sqrt(self.weight_self.size(1))
        self.weight_self.data.uniform_(-stdv_self, stdv_self)
        # self.w.data.uniform_(0.5, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, adj_homo):
        adj_self = torch.eye(adj.shape[0]).cuda()
        input = input.float()
        support = torch.mm(input, self.weight)
        # adj_homo = adj_homo - adj_self
        adj_homo = adj_homo.type(torch.FloatTensor).cuda()
        output = torch.spmm(adj_homo, support)
        # self
        support_self = torch.mm(input, self.weight_self)
        output_self = torch.spmm(adj_self, support_self)
        # total_output
        output = output + output_self
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
