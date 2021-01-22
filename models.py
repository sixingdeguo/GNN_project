import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import math
import numpy as np
from torch.nn.parameter import Parameter
import torch
# 在基础GCN网络的基础上建立了ResGCN、DenseGCN模型，模型名称如下所示


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GCN3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN3, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)


class GCN4(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN4, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        return F.log_softmax(x, dim=1)


class GCN5(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN5, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc5(x, adj)
        return F.log_softmax(x, dim=1)


class GCN10(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN10, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = F.relu(self.gc8(x, adj))
        x = F.relu(self.gc9(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc10(x, adj)
        return F.log_softmax(x, dim=1)


class GCN15(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN15, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid, nhid)
        self.gc11 = GraphConvolution(nhid, nhid)
        self.gc12 = GraphConvolution(nhid, nhid)
        self.gc13 = GraphConvolution(nhid, nhid)
        self.gc14 = GraphConvolution(nhid, nhid)
        self.gc15 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = F.relu(self.gc8(x, adj))
        x = F.relu(self.gc9(x, adj))
        x = F.relu(self.gc10(x, adj))
        x = F.relu(self.gc11(x, adj))
        x = F.relu(self.gc12(x, adj))
        x = F.relu(self.gc13(x, adj))
        x = F.relu(self.gc14(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc15(x, adj)
        return F.log_softmax(x, dim=1)


class GCN20(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN20, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid, nhid)
        self.gc11 = GraphConvolution(nhid, nhid)
        self.gc12 = GraphConvolution(nhid, nhid)
        self.gc13 = GraphConvolution(nhid, nhid)
        self.gc14 = GraphConvolution(nhid, nhid)
        self.gc15 = GraphConvolution(nhid, nhid)
        self.gc16 = GraphConvolution(nhid, nhid)
        self.gc17 = GraphConvolution(nhid, nhid)
        self.gc18 = GraphConvolution(nhid, nhid)
        self.gc19 = GraphConvolution(nhid, nhid)
        self.gc20 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = F.relu(self.gc8(x, adj))
        x = F.relu(self.gc9(x, adj))
        x = F.relu(self.gc10(x, adj))
        x = F.relu(self.gc11(x, adj))
        x = F.relu(self.gc12(x, adj))
        x = F.relu(self.gc13(x, adj))
        x = F.relu(self.gc14(x, adj))
        x = F.relu(self.gc15(x, adj))
        x = F.relu(self.gc16(x, adj))
        x = F.relu(self.gc17(x, adj))
        x = F.relu(self.gc18(x, adj))
        x = F.relu(self.gc19(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc20(x, adj)
        return F.log_softmax(x, dim=1)


class GCN25(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN25, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid, nhid)
        self.gc11 = GraphConvolution(nhid, nhid)
        self.gc12 = GraphConvolution(nhid, nhid)
        self.gc13 = GraphConvolution(nhid, nhid)
        self.gc14 = GraphConvolution(nhid, nhid)
        self.gc15 = GraphConvolution(nhid, nhid)
        self.gc16 = GraphConvolution(nhid, nhid)
        self.gc17 = GraphConvolution(nhid, nhid)
        self.gc18 = GraphConvolution(nhid, nhid)
        self.gc19 = GraphConvolution(nhid, nhid)
        self.gc20 = GraphConvolution(nhid, nhid)
        self.gc21 = GraphConvolution(nhid, nhid)
        self.gc22 = GraphConvolution(nhid, nhid)
        self.gc23 = GraphConvolution(nhid, nhid)
        self.gc24 = GraphConvolution(nhid, nhid)
        self.gc25 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = F.relu(self.gc8(x, adj))
        x = F.relu(self.gc9(x, adj))
        x = F.relu(self.gc10(x, adj))
        x = F.relu(self.gc11(x, adj))
        x = F.relu(self.gc12(x, adj))
        x = F.relu(self.gc13(x, adj))
        x = F.relu(self.gc14(x, adj))
        x = F.relu(self.gc15(x, adj))
        x = F.relu(self.gc16(x, adj))
        x = F.relu(self.gc17(x, adj))
        x = F.relu(self.gc18(x, adj))
        x = F.relu(self.gc19(x, adj))
        x = F.relu(self.gc20(x, adj))
        x = F.relu(self.gc21(x, adj))
        x = F.relu(self.gc22(x, adj))
        x = F.relu(self.gc23(x, adj))
        x = F.relu(self.gc24(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc25(x, adj)
        return F.log_softmax(x, dim=1)


class DenseGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(DenseGCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid + nfeat, nclass)
        self.dropout1 = dropout
        self.dropout2 = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x11 = F.dropout(x1, self.dropout1, training=self.training)

        x2 = torch.cat((x11, x), 1)
        x2 = F.dropout(x2, self.dropout2, training=self.training)

        x22 = F.relu(self.gc2(x2, adj))

        return F.log_softmax(x22, dim=1)


class DenseGCN3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(DenseGCN3, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid + nfeat, nhid)
        self.gc3 = GraphConvolution(nhid + 1 * nhid + nfeat, nclass)
        self.dropout1 = dropout
        self.dropout2 = dropout
        self.dropout3 = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x11 = F.dropout(x1, self.dropout1, training=self.training)

        x2 = torch.cat((x11, x), 1)
        x2 = F.dropout(x2, self.dropout2, training=self.training)

        x22 = F.relu(self.gc2(x2, adj))

        x3 = torch.cat((x22, x11, x), 1)
        x3 = F.dropout(x3, self.dropout3, training=self.training)
        x33 = F.relu(self.gc3(x3, adj))

        return F.log_softmax(x33, dim=1)


class DenseGCN4(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(DenseGCN4, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid + nfeat, nhid)
        self.gc3 = GraphConvolution(nhid + 1 * nhid + nfeat, nhid)
        self.gc4 = GraphConvolution(nhid + 2 * nhid + nfeat, nclass)
        self.dropout1 = dropout
        self.dropout2 = dropout
        self.dropout3 = dropout
        self.dropout4 = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x11 = F.dropout(x1, self.dropout1, training=self.training)

        x2 = torch.cat((x11, x), 1)
        x2 = F.dropout(x2, self.dropout2, training=self.training)

        x22 = F.relu(self.gc2(x2, adj))

        x3 = torch.cat((x22, x11, x), 1)
        x3 = F.dropout(x3, self.dropout3, training=self.training)
        x33 = F.relu(self.gc3(x3, adj))

        x4 = torch.cat((x33, x22, x11, x), 1)
        x4 = F.dropout(x4, self.dropout4, training=self.training)

        x44 = F.relu(self.gc4(x4, adj))
        return F.log_softmax(x44, dim=1)


class DenseGCN5(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(DenseGCN5, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid + nfeat, nhid)
        self.gc3 = GraphConvolution(nhid + 1 * nhid + nfeat, nhid)
        self.gc4 = GraphConvolution(nhid + 2 * nhid + nfeat, nhid)
        self.gc5 = GraphConvolution(nhid + 3 * nhid + nfeat, nclass)
        self.dropout1 = dropout
        self.dropout2 = dropout
        self.dropout3 = dropout
        self.dropout4 = dropout
        self.dropout5 = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x11 = F.dropout(x1, self.dropout1, training=self.training)
        x2 = torch.cat((x11, x), 1)
        x2 = F.dropout(x2, self.dropout2, training=self.training)
        x22 = F.relu(self.gc2(x2, adj))
        x3 = torch.cat((x22, x11, x), 1)
        x3 = F.dropout(x3, self.dropout3, training=self.training)
        x33 = F.relu(self.gc3(x3, adj))
        x4 = torch.cat((x33, x22, x11, x), 1)
        x4 = F.dropout(x4, self.dropout4, training=self.training)
        x44 = F.relu(self.gc4(x4, adj))
        x5 = torch.cat((x44, x33, x22, x11, x), 1)
        x5 = F.dropout(x5, self.dropout5, training=self.training)
        x55 = F.relu(self.gc5(x5, adj))
        return F.log_softmax(x55, dim=1)


class DenseGCN10(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(DenseGCN10, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid + nfeat, nhid)
        self.gc3 = GraphConvolution(nhid + 1 * nhid + nfeat, nhid)
        self.gc4 = GraphConvolution(nhid + 2 * nhid + nfeat, nhid)
        self.gc5 = GraphConvolution(nhid + 3 * nhid + nfeat, nhid)
        self.gc6 = GraphConvolution(nhid + 4 * nhid + nfeat, nhid)
        self.gc7 = GraphConvolution(nhid + 5 * nhid + nfeat, nhid)
        self.gc8 = GraphConvolution(nhid + 6 * nhid + nfeat, nhid)
        self.gc9 = GraphConvolution(nhid + 7 * nhid + nfeat, nhid)
        self.gc10 = GraphConvolution(nhid + 8 * nhid + nfeat, nclass)
        self.dropout1 = dropout
        self.dropout2 = dropout
        self.dropout3 = dropout
        self.dropout4 = dropout
        self.dropout5 = dropout
        self.dropout6 = dropout
        self.dropout7 = dropout
        self.dropout8 = dropout
        self.dropout9 = dropout
        self.dropout10 = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x11 = F.dropout(x1, self.dropout1, training=self.training)

        x2 = torch.cat((x11, x), 1)
        x2 = F.dropout(x2, self.dropout2, training=self.training)

        x22 = F.relu(self.gc2(x2, adj))

        x3 = torch.cat((x22, x11, x), 1)
        x3 = F.dropout(x3, self.dropout3, training=self.training)
        x33 = F.relu(self.gc3(x3, adj))

        x4 = torch.cat((x33, x22, x11, x), 1)
        x4 = F.dropout(x4, self.dropout4, training=self.training)

        x44 = F.relu(self.gc4(x4, adj))

        x5 = torch.cat((x44, x33, x22, x11, x), 1)
        x5 = F.dropout(x5, self.dropout5, training=self.training)

        x55 = F.relu(self.gc5(x5, adj))

        x6 = torch.cat((x55, x44, x33, x22, x11, x), 1)
        x6 = F.dropout(x6, self.dropout6, training=self.training)

        x66 = self.gc6(x6, adj)

        x7 = torch.cat((x66, x55, x44, x33, x22, x11, x), 1)
        x7 = F.dropout(x7, self.dropout7, training=self.training)

        x77 = self.gc7(x7, adj)

        x8 = torch.cat((x77, x66, x55, x44, x33, x22, x11, x), 1)
        x8 = F.dropout(x8, self.dropout8, training=self.training)

        x88 = self.gc8(x8, adj)

        x9 = torch.cat((x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x9 = F.dropout(x9, self.dropout9, training=self.training)

        x99 = self.gc9(x9, adj)

        x10 = torch.cat((x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x10 = F.dropout(x10, self.dropout10, training=self.training)

        x1010 = self.gc10(x10, adj)
        return F.log_softmax(x1010, dim=1)


class DenseGCN15(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(DenseGCN15, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid + nfeat, nhid)
        self.gc3 = GraphConvolution(nhid + 1 * nhid + nfeat, nhid)
        self.gc4 = GraphConvolution(nhid + 2 * nhid + nfeat, nhid)
        self.gc5 = GraphConvolution(nhid + 3 * nhid + nfeat, nhid)
        self.gc6 = GraphConvolution(nhid + 4 * nhid + nfeat, nhid)
        self.gc7 = GraphConvolution(nhid + 5 * nhid + nfeat, nhid)
        self.gc8 = GraphConvolution(nhid + 6 * nhid + nfeat, nhid)
        self.gc9 = GraphConvolution(nhid + 7 * nhid + nfeat, nhid)
        self.gc10 = GraphConvolution(nhid + 8 * nhid + nfeat, nhid)
        self.gc11 = GraphConvolution(nhid + 9 * nhid + nfeat, nhid)
        self.gc12 = GraphConvolution(nhid + 10 * nhid + nfeat, nhid)
        self.gc13 = GraphConvolution(nhid + 11 * nhid + nfeat, nhid)
        self.gc14 = GraphConvolution(nhid + 12 * nhid + nfeat, nhid)
        self.gc15 = GraphConvolution(nhid + 13 * nhid + nfeat, nclass)
        self.dropout1 = dropout
        self.dropout2 = dropout
        self.dropout3 = dropout
        self.dropout4 = dropout
        self.dropout5 = dropout
        self.dropout6 = dropout
        self.dropout7 = dropout
        self.dropout8 = dropout
        self.dropout9 = dropout
        self.dropout10 = dropout
        self.dropout11 = dropout
        self.dropout12 = dropout
        self.dropout13 = dropout
        self.dropout14 = dropout
        self.dropout15 = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x11 = F.dropout(x1, self.dropout1, training=self.training)

        x2 = torch.cat((x11, x), 1)
        x2 = F.dropout(x2, self.dropout2, training=self.training)

        x22 = F.relu(self.gc2(x2, adj))

        x3 = torch.cat((x22, x11, x), 1)
        x3 = F.dropout(x3, self.dropout3, training=self.training)
        x33 = F.relu(self.gc3(x3, adj))

        x4 = torch.cat((x33, x22, x11, x), 1)
        x4 = F.dropout(x4, self.dropout4, training=self.training)

        x44 = F.relu(self.gc4(x4, adj))

        x5 = torch.cat((x44, x33, x22, x11, x), 1)
        x5 = F.dropout(x5, self.dropout5, training=self.training)

        x55 = F.relu(self.gc5(x5, adj))

        x6 = torch.cat((x55, x44, x33, x22, x11, x), 1)
        x6 = F.dropout(x6, self.dropout6, training=self.training)

        x66 = self.gc6(x6, adj)

        x7 = torch.cat((x66, x55, x44, x33, x22, x11, x), 1)
        x7 = F.dropout(x7, self.dropout7, training=self.training)

        x77 = self.gc7(x7, adj)

        x8 = torch.cat((x77, x66, x55, x44, x33, x22, x11, x), 1)
        x8 = F.dropout(x8, self.dropout8, training=self.training)

        x88 = self.gc8(x8, adj)

        x9 = torch.cat((x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x9 = F.dropout(x9, self.dropout9, training=self.training)

        x99 = self.gc9(x9, adj)

        x10 = torch.cat((x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x10 = F.dropout(x10, self.dropout10, training=self.training)

        x1010 = self.gc10(x10, adj)

        x_11 = torch.cat((x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x_11 = F.dropout(x_11, self.dropout11, training=self.training)

        x1111 = self.gc11(x_11, adj)

        x12 = torch.cat((x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x12 = F.dropout(x12, self.dropout12, training=self.training)

        x1212 = self.gc12(x12, adj)

        x13 = torch.cat((x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x13 = F.dropout(x13, self.dropout13, training=self.training)

        x1313 = self.gc13(x13, adj)

        x14 = torch.cat((x1313, x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x14 = F.dropout(x14, self.dropout14, training=self.training)

        x1414 = self.gc14(x14, adj)

        x15 = torch.cat((x1414, x1313, x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x15 = F.dropout(x15, self.dropout15, training=self.training)

        x1515 = self.gc15(x15, adj)
        return F.log_softmax(x1515, dim=1)


class DenseGCN20(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(DenseGCN20, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid + nfeat, nhid)
        self.gc3 = GraphConvolution(nhid + 1 * nhid + nfeat, nhid)
        self.gc4 = GraphConvolution(nhid + 2 * nhid + nfeat, nhid)
        self.gc5 = GraphConvolution(nhid + 3 * nhid + nfeat, nhid)
        self.gc6 = GraphConvolution(nhid + 4 * nhid + nfeat, nhid)
        self.gc7 = GraphConvolution(nhid + 5 * nhid + nfeat, nhid)
        self.gc8 = GraphConvolution(nhid + 6 * nhid + nfeat, nhid)
        self.gc9 = GraphConvolution(nhid + 7 * nhid + nfeat, nhid)
        self.gc10 = GraphConvolution(nhid + 8 * nhid + nfeat, nhid)
        self.gc11 = GraphConvolution(nhid + 9 * nhid + nfeat, nhid)
        self.gc12 = GraphConvolution(nhid + 10 * nhid + nfeat, nhid)
        self.gc13 = GraphConvolution(nhid + 11 * nhid + nfeat, nhid)
        self.gc14 = GraphConvolution(nhid + 12 * nhid + nfeat, nhid)
        self.gc15 = GraphConvolution(nhid + 13 * nhid + nfeat, nhid)
        self.gc16 = GraphConvolution(nhid + 14 * nhid + nfeat, nhid)
        self.gc17 = GraphConvolution(nhid + 15 * nhid + nfeat, nhid)
        self.gc18 = GraphConvolution(nhid + 16 * nhid + nfeat, nhid)
        self.gc19 = GraphConvolution(nhid + 17 * nhid + nfeat, nhid)
        self.gc20 = GraphConvolution(nhid + 18 * nhid + nfeat, nclass)
        self.dropout1 = dropout
        self.dropout2 = dropout
        self.dropout3 = dropout
        self.dropout4 = dropout
        self.dropout5 = dropout
        self.dropout6 = dropout
        self.dropout7 = dropout
        self.dropout8 = dropout
        self.dropout9 = dropout
        self.dropout10 = dropout
        self.dropout11 = dropout
        self.dropout12 = dropout
        self.dropout13 = dropout
        self.dropout14 = dropout
        self.dropout15 = dropout
        self.dropout16 = dropout
        self.dropout17 = dropout
        self.dropout18 = dropout
        self.dropout19 = dropout
        self.dropout20 = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x11 = F.dropout(x1, self.dropout1, training=self.training)

        x2 = torch.cat((x11, x), 1)
        x2 = F.dropout(x2, self.dropout2, training=self.training)

        x22 = F.relu(self.gc2(x2, adj))

        x3 = torch.cat((x22, x11, x), 1)
        x3 = F.dropout(x3, self.dropout3, training=self.training)
        x33 = F.relu(self.gc3(x3, adj))

        x4 = torch.cat((x33, x22, x11, x), 1)
        x4 = F.dropout(x4, self.dropout4, training=self.training)

        x44 = F.relu(self.gc4(x4, adj))

        x5 = torch.cat((x44, x33, x22, x11, x), 1)
        x5 = F.dropout(x5, self.dropout5, training=self.training)

        x55 = F.relu(self.gc5(x5, adj))

        x6 = torch.cat((x55, x44, x33, x22, x11, x), 1)
        x6 = F.dropout(x6, self.dropout6, training=self.training)

        x66 = self.gc6(x6, adj)

        x7 = torch.cat((x66, x55, x44, x33, x22, x11, x), 1)
        x7 = F.dropout(x7, self.dropout7, training=self.training)

        x77 = self.gc7(x7, adj)

        x8 = torch.cat((x77, x66, x55, x44, x33, x22, x11, x), 1)
        x8 = F.dropout(x8, self.dropout8, training=self.training)

        x88 = self.gc8(x8, adj)

        x9 = torch.cat((x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x9 = F.dropout(x9, self.dropout9, training=self.training)

        x99 = self.gc9(x9, adj)

        x10 = torch.cat((x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x10 = F.dropout(x10, self.dropout10, training=self.training)

        x1010 = self.gc10(x10, adj)

        x_11 = torch.cat((x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x_11 = F.dropout(x_11, self.dropout11, training=self.training)

        x1111 = self.gc11(x_11, adj)

        x12 = torch.cat((x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x12 = F.dropout(x12, self.dropout12, training=self.training)

        x1212 = self.gc12(x12, adj)

        x13 = torch.cat((x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x13 = F.dropout(x13, self.dropout13, training=self.training)

        x1313 = self.gc13(x13, adj)

        x14 = torch.cat((x1313, x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x14 = F.dropout(x14, self.dropout14, training=self.training)

        x1414 = self.gc14(x14, adj)

        x15 = torch.cat((x1414, x1313, x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x15 = F.dropout(x15, self.dropout15, training=self.training)

        x1515 = self.gc15(x15, adj)

        x16 = torch.cat((x1515, x1414, x1313, x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x16 = F.dropout(x16, self.dropout16, training=self.training)

        x1616 = self.gc16(x16, adj)

        x17 = torch.cat(
            (x1616, x1515, x1414, x1313, x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x17 = F.dropout(x17, self.dropout17, training=self.training)

        x1717 = self.gc17(x17, adj)

        x18 = torch.cat(
            (x1717, x1616, x1515, x1414, x1313, x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x18 = F.dropout(x18, self.dropout18, training=self.training)

        x1818 = self.gc18(x18, adj)

        x19 = torch.cat(
            (x1818, x1717, x1616, x1515, x1414, x1313, x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11,
             x), 1)
        x19 = F.dropout(x19, self.dropout19, training=self.training)

        x1919 = self.gc19(x19, adj)

        x20 = torch.cat(
            (x1919, x1818, x1717, x1616, x1515, x1414, x1313, x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33,
             x22, x11,
             x), 1)
        x20 = F.dropout(x20, self.dropout20, training=self.training)

        x2020 = self.gc20(x20, adj)
        return F.log_softmax(x2020, dim=1)


class DenseGCN25(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(DenseGCN25, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid + nfeat, nhid)
        self.gc3 = GraphConvolution(nhid + 1 * nhid + nfeat, nhid)
        self.gc4 = GraphConvolution(nhid + 2 * nhid + nfeat, nhid)
        self.gc5 = GraphConvolution(nhid + 3 * nhid + nfeat, nhid)
        self.gc6 = GraphConvolution(nhid + 4 * nhid + nfeat, nhid)
        self.gc7 = GraphConvolution(nhid + 5 * nhid + nfeat, nhid)
        self.gc8 = GraphConvolution(nhid + 6 * nhid + nfeat, nhid)
        self.gc9 = GraphConvolution(nhid + 7 * nhid + nfeat, nhid)
        self.gc10 = GraphConvolution(nhid + 8 * nhid + nfeat, nhid)
        self.gc11 = GraphConvolution(nhid + 9 * nhid + nfeat, nhid)
        self.gc12 = GraphConvolution(nhid + 10 * nhid + nfeat, nhid)
        self.gc13 = GraphConvolution(nhid + 11 * nhid + nfeat, nhid)
        self.gc14 = GraphConvolution(nhid + 12 * nhid + nfeat, nhid)
        self.gc15 = GraphConvolution(nhid + 13 * nhid + nfeat, nhid)
        self.gc16 = GraphConvolution(nhid + 14 * nhid + nfeat, nhid)
        self.gc17 = GraphConvolution(nhid + 15 * nhid + nfeat, nhid)
        self.gc18 = GraphConvolution(nhid + 16 * nhid + nfeat, nhid)
        self.gc19 = GraphConvolution(nhid + 17 * nhid + nfeat, nhid)
        self.gc20 = GraphConvolution(nhid + 18 * nhid + nfeat, nhid)
        self.gc21 = GraphConvolution(nhid + 19 * nhid + nfeat, nhid)
        self.gc22 = GraphConvolution(nhid + 20 * nhid + nfeat, nhid)
        self.gc23 = GraphConvolution(nhid + 21 * nhid + nfeat, nhid)
        self.gc24 = GraphConvolution(nhid + 22 * nhid + nfeat, nhid)
        self.gc25 = GraphConvolution(nhid + 23 * nhid + nfeat, nclass)
        self.dropout1 = dropout
        self.dropout2 = dropout
        self.dropout3 = dropout
        self.dropout4 = dropout
        self.dropout5 = dropout
        self.dropout6 = dropout
        self.dropout7 = dropout
        self.dropout8 = dropout
        self.dropout9 = dropout
        self.dropout10 = dropout
        self.dropout11 = dropout
        self.dropout12 = dropout
        self.dropout13 = dropout
        self.dropout14 = dropout
        self.dropout15 = dropout
        self.dropout16 = dropout
        self.dropout17 = dropout
        self.dropout18 = dropout
        self.dropout19 = dropout
        self.dropout20 = dropout
        self.dropout21 = dropout
        self.dropout22 = dropout
        self.dropout23 = dropout
        self.dropout24 = dropout
        self.dropout25 = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x11 = F.dropout(x1, self.dropout1, training=self.training)

        x2 = torch.cat((x11, x), 1)
        x2 = F.dropout(x2, self.dropout2, training=self.training)

        x22 = F.relu(self.gc2(x2, adj))

        x3 = torch.cat((x22, x11, x), 1)
        x3 = F.dropout(x3, self.dropout3, training=self.training)
        x33 = F.relu(self.gc3(x3, adj))

        x4 = torch.cat((x33, x22, x11, x), 1)
        x4 = F.dropout(x4, self.dropout4, training=self.training)

        x44 = F.relu(self.gc4(x4, adj))

        x5 = torch.cat((x44, x33, x22, x11, x), 1)
        x5 = F.dropout(x5, self.dropout5, training=self.training)

        x55 = F.relu(self.gc5(x5, adj))

        x6 = torch.cat((x55, x44, x33, x22, x11, x), 1)
        x6 = F.dropout(x6, self.dropout6, training=self.training)

        x66 = self.gc6(x6, adj)

        x7 = torch.cat((x66, x55, x44, x33, x22, x11, x), 1)
        x7 = F.dropout(x7, self.dropout7, training=self.training)

        x77 = self.gc7(x7, adj)

        x8 = torch.cat((x77, x66, x55, x44, x33, x22, x11, x), 1)
        x8 = F.dropout(x8, self.dropout8, training=self.training)

        x88 = self.gc8(x8, adj)

        x9 = torch.cat((x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x9 = F.dropout(x9, self.dropout9, training=self.training)

        x99 = self.gc9(x9, adj)

        x10 = torch.cat((x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x10 = F.dropout(x10, self.dropout10, training=self.training)

        x1010 = self.gc10(x10, adj)

        x_11 = torch.cat((x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x_11 = F.dropout(x_11, self.dropout11, training=self.training)

        x1111 = self.gc11(x_11, adj)

        x12 = torch.cat((x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x12 = F.dropout(x12, self.dropout12, training=self.training)

        x1212 = self.gc12(x12, adj)

        x13 = torch.cat((x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x13 = F.dropout(x13, self.dropout13, training=self.training)

        x1313 = self.gc13(x13, adj)

        x14 = torch.cat((x1313, x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x14 = F.dropout(x14, self.dropout14, training=self.training)

        x1414 = self.gc14(x14, adj)

        x15 = torch.cat((x1414, x1313, x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x15 = F.dropout(x15, self.dropout15, training=self.training)

        x1515 = self.gc15(x15, adj)

        x16 = torch.cat((x1515, x1414, x1313, x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x16 = F.dropout(x16, self.dropout16, training=self.training)

        x1616 = self.gc16(x16, adj)

        x17 = torch.cat(
            (x1616, x1515, x1414, x1313, x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x17 = F.dropout(x17, self.dropout17, training=self.training)

        x1717 = self.gc17(x17, adj)

        x18 = torch.cat(
            (x1717, x1616, x1515, x1414, x1313, x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11, x), 1)
        x18 = F.dropout(x18, self.dropout18, training=self.training)

        x1818 = self.gc18(x18, adj)

        x19 = torch.cat(
            (x1818, x1717, x1616, x1515, x1414, x1313, x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33, x22, x11,
             x), 1)
        x19 = F.dropout(x19, self.dropout19, training=self.training)

        x1919 = self.gc19(x19, adj)

        x20 = torch.cat(
            (x1919, x1818, x1717, x1616, x1515, x1414, x1313, x1212, x1111, x1010, x99, x88, x77, x66, x55, x44, x33,
             x22, x11, x), 1)
        x20 = F.dropout(x20, self.dropout20, training=self.training)

        x2020 = self.gc20(x20, adj)

        x21 = torch.cat((
            x2020, x1919, x1818, x1717, x1616, x1515, x1414, x1313, x1212, x1111, x1010, x99, x88, x77, x66,
            x55, x44, x33, x22, x11, x), 1)
        x21 = F.dropout(x21, self.dropout21, training=self.training)

        x2121 = self.gc21(x21, adj)

        x_22 = torch.cat(
            (
                x2121, x2020, x1919, x1818, x1717, x1616, x1515, x1414, x1313, x1212, x1111, x1010, x99, x88, x77, x66,
                x55,
                x44, x33, x22, x11, x), 1)
        x_22 = F.dropout(x_22, self.dropout22, training=self.training)

        x2222 = self.gc22(x_22, adj)

        x23 = torch.cat(
            (x2222, x2121, x2020, x1919, x1818, x1717, x1616, x1515, x1414, x1313, x1212, x1111, x1010, x99, x88, x77,
             x66,
             x55,
             x44, x33, x22, x11, x), 1)
        x23 = F.dropout(x23, self.dropout23, training=self.training)

        x2323 = self.gc23(x23, adj)

        x24 = torch.cat(
            (x2323, x2222, x2121, x2020, x1919, x1818, x1717, x1616, x1515, x1414, x1313, x1212, x1111, x1010, x99, x88,
             x77,
             x66, x55, x44, x33, x22, x11, x), 1)
        x24 = F.dropout(x24, self.dropout24, training=self.training)

        x2424 = self.gc24(x24, adj)

        x25 = torch.cat(
            (x2424, x2323, x2222, x2121, x2020, x1919, x1818, x1717, x1616, x1515, x1414, x1313, x1212, x1111, x1010,
             x99, x88, x77,
             x66, x55, x44, x33, x22, x11, x), 1)
        x25 = F.dropout(x25, self.dropout25, training=self.training)

        x2525 = self.gc25(x25, adj)

        return F.log_softmax(x2525, dim=1)


class ResGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResGCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

        self.weight = Parameter(torch.FloatTensor(nfeat, nhid))
        self.bias = Parameter(torch.FloatTensor(nhid))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        z = torch.mm(x, self.weight) + self.bias
        x1 = F.relu(self.gc1(x, adj)) + z
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.gc2(x1, adj)
        return F.log_softmax(x2, dim=1)


class ResGCN3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResGCN3, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid * 2, nclass)
        self.dropout = dropout

        self.weight = Parameter(torch.FloatTensor(nfeat, nhid))
        self.bias = Parameter(torch.FloatTensor(nhid))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        z = torch.mm(x, self.weight) + self.bias
        x1 = F.relu(self.gc1(x, adj)) + z
        x2 = F.relu(self.gc2(x1, adj)) + x1
        x_ = torch.cat((x2, x1), 1)
        x_ = F.dropout(x_, self.dropout, training=self.training)
        x3 = self.gc3(x_, adj)
        return F.log_softmax(x3, dim=1)


class ResGCN4(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResGCN4, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid * 3, nclass)
        self.dropout = dropout

        self.weight = Parameter(torch.FloatTensor(nfeat, nhid))
        self.bias = Parameter(torch.FloatTensor(nhid))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        z = torch.mm(x, self.weight) + self.bias
        x1 = F.relu(self.gc1(x, adj)) + z
        x2 = F.relu(self.gc2(x1, adj)) + x1
        x3 = F.relu(self.gc3(x2, adj)) + x2
        x_ = torch.cat((x3, x2, x1), 1)
        x_ = F.dropout(x_, self.dropout, training=self.training)
        x4 = self.gc4(x_, adj)
        return F.log_softmax(x4, dim=1)


class ResGCN5(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResGCN5, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid * 4, nclass)
        self.dropout = dropout

        self.weight = Parameter(torch.FloatTensor(nfeat, nhid))
        self.bias = Parameter(torch.FloatTensor(nhid))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        z = torch.mm(x, self.weight) + self.bias
        x1 = F.relu(self.gc1(x, adj)) + z
        x2 = F.relu(self.gc2(x1, adj)) + x1
        x3 = F.relu(self.gc2(x2, adj)) + x2
        x4 = F.relu(self.gc3(x3, adj)) + x3
        x_ = torch.cat((x4, x3, x2, x1), 1)
        x_ = F.dropout(x_, self.dropout, training=self.training)
        x5 = self.gc5(x_, adj)
        return F.log_softmax(x5, dim=1)


class ResGCN10(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResGCN10, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid * 9, nclass)
        self.dropout = dropout

        self.weight = Parameter(torch.FloatTensor(nfeat, nhid))
        self.bias = Parameter(torch.FloatTensor(nhid))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        z = torch.mm(x, self.weight) + self.bias
        x1 = F.relu(self.gc1(x, adj)) + z
        x2 = F.relu(self.gc2(x1, adj)) + x1
        x3 = F.relu(self.gc3(x2, adj)) + x2
        x4 = F.relu(self.gc4(x3, adj)) + x3
        x5 = F.relu(self.gc5(x4, adj)) + x4
        x6 = F.relu(self.gc6(x5, adj)) + x5
        x7 = F.relu(self.gc7(x6, adj)) + x6
        x8 = F.relu(self.gc8(x7, adj)) + x7
        x9 = F.relu(self.gc9(x8, adj)) + x8
        x_ = torch.cat((x9, x8, x7, x6, x5, x4, x3, x2, x1), 1)
        x_ = F.dropout(x_, self.dropout, training=self.training)
        x10 = self.gc10(x_, adj)
        return F.log_softmax(x10, dim=1)


class ResGCN15(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResGCN15, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid, nhid)
        self.gc11 = GraphConvolution(nhid, nhid)
        self.gc12 = GraphConvolution(nhid, nhid)
        self.gc13 = GraphConvolution(nhid, nhid)
        self.gc14 = GraphConvolution(nhid, nhid)
        self.gc15 = GraphConvolution(nhid * 14, nclass)
        self.dropout = dropout

        self.weight = Parameter(torch.FloatTensor(nfeat, nhid))
        self.bias = Parameter(torch.FloatTensor(nhid))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        z = torch.mm(x, self.weight) + self.bias
        x1 = F.relu(self.gc1(x, adj)) + z
        x2 = F.relu(self.gc2(x1, adj)) + x1
        x3 = F.relu(self.gc3(x2, adj)) + x2
        x4 = F.relu(self.gc4(x3, adj)) + x3
        x5 = F.relu(self.gc5(x4, adj)) + x4
        x6 = F.relu(self.gc6(x5, adj)) + x5
        x7 = F.relu(self.gc7(x6, adj)) + x6
        x8 = F.relu(self.gc8(x7, adj)) + x7
        x9 = F.relu(self.gc9(x8, adj)) + x8
        x10 = F.relu(self.gc10(x9, adj)) + x9
        x11 = F.relu(self.gc11(x10, adj)) + x10
        x12 = F.relu(self.gc12(x11, adj)) + x11
        x13 = F.relu(self.gc13(x12, adj)) + x12
        x14 = F.relu(self.gc14(x13, adj)) + x13
        x_ = torch.cat((x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1), 1)
        x_ = F.dropout(x_, self.dropout, training=self.training)
        x15 = self.gc15(x_, adj)
        return F.log_softmax(x15, dim=1)


class ResGCN20(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResGCN20, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid, nhid)
        self.gc11 = GraphConvolution(nhid, nhid)
        self.gc12 = GraphConvolution(nhid, nhid)
        self.gc13 = GraphConvolution(nhid, nhid)
        self.gc14 = GraphConvolution(nhid, nhid)
        self.gc15 = GraphConvolution(nhid, nhid)
        self.gc16 = GraphConvolution(nhid, nhid)
        self.gc17 = GraphConvolution(nhid, nhid)
        self.gc18 = GraphConvolution(nhid, nhid)
        self.gc19 = GraphConvolution(nhid, nhid)
        self.gc20 = GraphConvolution(nhid * 19, nclass)
        self.dropout = dropout

        self.weight = Parameter(torch.FloatTensor(nfeat, nhid))
        self.bias = Parameter(torch.FloatTensor(nhid))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        z = torch.mm(x, self.weight) + self.bias
        x1 = F.relu(self.gc1(x, adj)) + z
        x2 = F.relu(self.gc2(x1, adj)) + x1
        x3 = F.relu(self.gc3(x2, adj)) + x2
        x4 = F.relu(self.gc4(x3, adj)) + x3
        x5 = F.relu(self.gc5(x4, adj)) + x4
        x6 = F.relu(self.gc6(x5, adj)) + x5
        x7 = F.relu(self.gc7(x6, adj)) + x6
        x8 = F.relu(self.gc8(x7, adj)) + x7
        x9 = F.relu(self.gc9(x8, adj)) + x8
        x10 = F.relu(self.gc10(x9, adj)) + x9
        x11 = F.relu(self.gc11(x10, adj)) + x10
        x12 = F.relu(self.gc12(x11, adj)) + x11
        x13 = F.relu(self.gc13(x12, adj)) + x12
        x14 = F.relu(self.gc14(x13, adj)) + x13
        x15 = F.relu(self.gc15(x14, adj)) + x14
        x16 = F.relu(self.gc16(x15, adj)) + x15
        x17 = F.relu(self.gc17(x16, adj)) + x16
        x18 = F.relu(self.gc18(x17, adj)) + x17
        x19 = F.relu(self.gc19(x18, adj)) + x18
        x_ = torch.cat((x19, x18, x17, x16, x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1), 1)
        x_ = F.dropout(x_, self.dropout, training=self.training)
        x20 = self.gc20(x_, adj)
        return F.log_softmax(x20, dim=1)


class ResGCN25(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResGCN25, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid, nhid)
        self.gc11 = GraphConvolution(nhid, nhid)
        self.gc12 = GraphConvolution(nhid, nhid)
        self.gc13 = GraphConvolution(nhid, nhid)
        self.gc14 = GraphConvolution(nhid, nhid)
        self.gc15 = GraphConvolution(nhid, nhid)
        self.gc16 = GraphConvolution(nhid, nhid)
        self.gc17 = GraphConvolution(nhid, nhid)
        self.gc18 = GraphConvolution(nhid, nhid)
        self.gc19 = GraphConvolution(nhid, nhid)
        self.gc20 = GraphConvolution(nhid, nhid)
        self.gc21 = GraphConvolution(nhid, nhid)
        self.gc22 = GraphConvolution(nhid, nhid)
        self.gc23 = GraphConvolution(nhid, nhid)
        self.gc24 = GraphConvolution(nhid, nhid)
        self.gc25 = GraphConvolution(nhid * 24, nclass)
        self.dropout = dropout

        self.weight = Parameter(torch.FloatTensor(nfeat, nhid))
        self.bias = Parameter(torch.FloatTensor(nhid))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        z = torch.mm(x, self.weight) + self.bias
        x1 = F.relu(self.gc1(x, adj)) + z
        x2 = F.relu(self.gc2(x1, adj)) + x1
        x3 = F.relu(self.gc3(x2, adj)) + x2
        x4 = F.relu(self.gc4(x3, adj)) + x3
        x5 = F.relu(self.gc5(x4, adj)) + x4
        x6 = F.relu(self.gc6(x5, adj)) + x5
        x7 = F.relu(self.gc7(x6, adj)) + x6
        x8 = F.relu(self.gc8(x7, adj)) + x7
        x9 = F.relu(self.gc9(x8, adj)) + x8
        x10 = F.relu(self.gc10(x9, adj)) + x9
        x11 = F.relu(self.gc11(x10, adj)) + x10
        x12 = F.relu(self.gc12(x11, adj)) + x11
        x13 = F.relu(self.gc13(x12, adj)) + x12
        x14 = F.relu(self.gc14(x13, adj)) + x13
        x15 = F.relu(self.gc15(x14, adj)) + x14
        x16 = F.relu(self.gc16(x15, adj)) + x15
        x17 = F.relu(self.gc17(x16, adj)) + x16
        x18 = F.relu(self.gc18(x17, adj)) + x17
        x19 = F.relu(self.gc19(x18, adj)) + x18
        x20 = F.relu(self.gc20(x19, adj)) + x19
        x21 = F.relu(self.gc21(x20, adj)) + x20
        x22 = F.relu(self.gc22(x21, adj)) + x21
        x23 = F.relu(self.gc23(x22, adj)) + x22
        x24 = F.relu(self.gc24(x23, adj)) + x23
        x_ = torch.cat((x24, x23, x22, x21, x20, x19, x18, x17, x16, x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5,
                        x4, x3, x2, x1), 1)
        x_ = F.dropout(x_, self.dropout, training=self.training)
        x25 = self.gc25(x_, adj)
        return F.log_softmax(x25, dim=1)
