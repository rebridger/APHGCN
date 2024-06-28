import numpy as np
import scipy.sparse as sp
import torch
import math
import matplotlib.pyplot as plt
from sklearn import manifold


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def con_loss(x1, x2):
    # positive = np.array(x1.shape[0], 1)
    loss = 0.0
    x1 = x1.cpu().detach().numpy()
    x2 = x2.cpu().detach().numpy()
    x = x1 * x2
    positive = np.sum(x, 1)
    # positive = positive.cpu().detach().numpy()
    # negative = np.array(x1.shape[0], 1)
    for i in range(x1.shape[0]):
        # positive[i] = np.dot(x1[i], x2[i])
        negative = np.longdouble(0)
        for j in range(x1.shape[0]):
            b = np.dot(x1[i], x2[j])
            b = np.exp(b)
            negative = negative + np.exp(b)
        loss = loss - math.log(np.exp(positive[i])/negative)
    return loss


def generate_similarity_matrix(labels, mask):
    n = len(labels)
    similarity_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j] and mask[i] == True:
                similarity_matrix[i, j] = 1

    return torch.Tensor(similarity_matrix)


# def constrastive_loss(x_list, L):
#     loss = 0
#     for i in range(len(x_list)):
#         for j in range(i + 1, len(x_list)):
#             loss += (uc_loss(x_list[i], x_list[j], L) + uc_loss(x_list[j], x_list[i], L))
#             # loss += ucloss(x_list[i], x_list[j])
#     return loss / len(x_list)


# def uc_loss(h1,h2,L):
#     for i in range(len(h1)):



def ucloss(h1, h2):
    h_dim = h1.shape[0]
    h1_h2 = torch.matmul(h1, h2.T)
    h1_h2_diag = torch.diag(h1_h2)
    temp1 = torch.max(h1_h2)
    h1_h2_down = torch.log(torch.sum(torch.exp(h1_h2), dim=1) + 1)
    Luc1_loss = -torch.div((h1_h2_diag - h1_h2_down), 2 * h_dim)
    temp1 = torch.max(Luc1_loss)

    h2_h1 = torch.matmul(h2, h1.T)
    h1_h2_diag = torch.diag(h2_h1)
    h2_h1_down = torch.log(torch.sum(torch.exp(h2_h1), dim=1) + 1)
    Luc2_loss = -torch.div((h1_h2_diag - h2_h1_down), 2 * h_dim)
    temp2 = torch.max(Luc2_loss)
    Luc_loss = torch.sum(Luc1_loss) + torch.sum(Luc2_loss)

    return Luc_loss


def draw_plt(output_, labels, index, dataset):
    output_ = output_.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    X_tsne = manifold.TSNE(n_components=2, learning_rate=100, random_state=42).fit_transform(output_)
    plt.figure(figsize=(8, 6))
    # plt.title('Dataset : ' + dataset_name + '   (Label rate : 20 nodes per class)')

    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=8, cmap='rainbow')
    # scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=8)
    handles, _ = scatter.legend_elements(prop='colors')
    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    # plt.legend(handles, labels, loc='upper right')
    plt.axis('off')
    # plt.colorbar(ticks=range(5))
    plt.savefig('./' + dataset + 'com_GCN.svg')
    plt.show()