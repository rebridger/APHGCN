import scipy
import torch
import scipy.sparse as sp
import numpy as np
import scipy.io as scio
import scipy.io as sio
from sklearn.decomposition import PCA

"""
Here we don't do any normalization for input node features
    graph.adj: SparseTensor (use matmul) 
"""


def count_each_class_num(gnd):
    '''
    Count the number of samples in each class
    '''
    count_dict = {}
    for label in gnd:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict


def generate_permutation(gnd, args):
    '''
    Generate permutation for training, validating and testing data.
    '''
    gnd = np.array(gnd)
    N = gnd.shape[0]
    each_class_num = count_each_class_num(gnd)
    training_each_class_num = {}  ## number of labeled samples for each class
    # valid_each_class_num = {}
    # test_each_class_num = {}
    for label in each_class_num.keys():
        if args.data_split_mode == "Ratio":
            train_ratio = args.train_ratio
            valid_ratio = args.valid_ratio
            test_ratio = args.test_ratio
            training_each_class_num[label] = max(round(each_class_num[label] * train_ratio), 1)  # min is 1
            valid_num = max(round(N * valid_ratio), 1)  # min is 1
            test_num = max(round(N * test_ratio), 1)  # min is 1
        else:
            training_each_class_num[label] = args.num_train_per_class
            valid_num = args.num_val
            test_num = args.num_test

    # index of labeled and unlabeled samples
    train_mask = torch.from_numpy(np.full((N), False))
    valid_mask = torch.from_numpy(np.full((N), False))
    test_mask = torch.from_numpy(np.full((N), False))

    train_idx = []
    valid_idx = []
    test_idx = []

    # shuffle the data
    data_idx = np.random.permutation(range(N))

    # Get training data
    for idx in data_idx:
        label = gnd[idx]
        if (training_each_class_num[label] > 0):
            training_each_class_num[label] -= 1
            train_mask[idx] = True
            train_idx.append(idx)
    for idx in data_idx:
        if train_mask[idx] == True:
            continue
        if (valid_num > 0):
            valid_num -= 1
            valid_mask[idx] = True
            valid_idx.append(idx)
        elif (test_num > 0):
            test_num -= 1
            test_mask[idx] = True
            test_idx.append(idx)
    return torch.from_numpy(np.array(train_idx)).long(), torch.from_numpy(np.array(valid_idx)).long(), torch.from_numpy(
        np.array(test_idx)).long(), train_mask


def loadMatData(dataset="Cora"):
    '''
    return features: Tensor
    edges: Sparse Tensor
    edge_weights: Sparse Tensor
    gnd: Tensor
    '''
    print('Loading {} dataset...'.format(dataset))
    data = scio.loadmat("/data/huangy/dataset/{}.mat".format(dataset))

    features = data["X"]
    labels = data["Y"].flatten()
    labels = encode_onehot(labels)
    adj = data['adj']
    # adj = sp.coo_matrix(adj)
    sp_adj = scipy.sparse.coo_matrix(adj)
    sp_adj = aug_normalized_adjacency(sp_adj)
    # adj = sparse_mx_to_torch_sparse_tensor(sp_adj, device='cuda')

    features = normalize(features)
    # features = torch.FloatTensor(np.array(features))

    features = torch.from_numpy(features).float()
    labels = torch.LongTensor(np.where(labels)[1])

    num_of_data = features.shape[0]

    return features, labels, adj



def sparse_mx_to_torch_sparse_tensor(sparse_mx, device=None):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    tensor = torch.sparse.FloatTensor(indices, values, shape)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def aug_normalized_adjacency(adj, need_orig=False):
    if not need_orig:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


'''
Load data from torch_geometric api
'''


def normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mx = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    if isinstance(mx, torch.Tensor):
        return mx
    else:
        mx = np.array(mx.toarray())
        return torch.from_numpy(mx).float()


# def normalize(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     if isinstance(mx, np.ndarray):
#         return torch.from_numpy(mx)
#     else:
#         return mx

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot
