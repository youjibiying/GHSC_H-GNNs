import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import torch
import torch_scatter


def get_G_from_H(data, sigma, args):
    edge_index = data.edge_index
    ones = torch.ones(data.edge_index.shape[1], device=edge_index.device)

    # alpha = args.HNHN_alpha
    # beta = args.HNHN_beta

    # the degree of the node
    DV = torch_scatter.scatter_add(ones, edge_index[0], dim=0)
    # the degree of the hyperedge
    DE = torch_scatter.scatter_add(ones, edge_index[1], dim=0)
    invDE = DE ** sigma

def edge_index_to_coo_matrix(edge_index,num_nodes, num_edges, data=None):
    # num_edges = int(edge_index[1].max()) + 1
    # num_nodes = edge_index[0].max().item() + 1

    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    if data is None: data = [1] * edge_index.shape[1]

    # 创建 COO 矩阵
    H_coo = coo_matrix((data, (row, col)), shape=(num_nodes, num_edges))

    return H_coo

def _generate_G_from_H(data, add_self_loop=True, sigma=None, args=None):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    # add_self_loop = False
    if args is not None:
        sigma = args.sigma
    else:
        sigma = -1
    edge_weight=None
    if "edge_weight" in data:
        edge_weight=data.edge_weight
    H = edge_index_to_coo_matrix(data.edge_index, data.num_nodes, data.num_edges,data=edge_weight)
    assert np.all(H.row == data.edge_index[0].numpy()), np.all(H.col == data.edge_index[1].numpy())
    n_edge = H.shape[1]  # 4024
    # the weight of the hyperedge
    W = np.ones(n_edge)

    # the degree of the hyperedge
    DE = np.sum(H, axis=0)  # [4024]


    DE = DE.tolist()[0]
    invDE = np.power(np.array(DE).astype(float), sigma)
    invDE[np.isinf(invDE)] = 0
    invDE = coo_matrix((invDE, (range(n_edge),range(n_edge))),shape=(n_edge, n_edge))
    K = H * invDE * H.T


    #
    if add_self_loop:
        print('renormalization!!')
        K += sp.eye(H.shape[0])

    DV = np.sum(K, 0).tolist()[0]
    invDV = np.power(np.array(DV).astype(float), -0.5)
    invDV[np.isinf(invDV)] = 0
    DV2 = coo_matrix((invDV, (range(H.shape[0]), range(H.shape[0]))), shape=(H.shape[0], H.shape[0]))

    G = DV2 * K * DV2
    G = G.tocoo()
    data.edge_index_graph = torch.LongTensor(np.array([G.row, G.col]))
    data.edge_weight_graph = torch.tensor(G.data).float()


    return data



def _generate_G_from_H_with_dist(data, add_self_loop=True, sigma=None, args=None):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    # add_self_loop = False
    if args is not None:
        sigma = args.sigma
    else:
        sigma = -1
    dist_map = Eu_dis(data.x)
    H = edge_index_to_coo_matrix(data.edge_index, data.num_nodes, data.num_edges, data=dist_map)
    assert np.all(H.row == data.edge_index[0].numpy()), np.all(H.col == data.edge_index[1].numpy())
    n_edge = H.shape[1]  # 4024
    # the weight of the hyperedge
    W = np.ones(n_edge)

    # the degree of the hyperedge
    DE = np.sum(H, axis=0)  # [4024]


    DE = DE.tolist()[0]
    invDE = np.power(np.array(DE).astype(float), sigma)
    invDE[np.isinf(invDE)] = 0
    invDE = coo_matrix((invDE, (range(n_edge),range(n_edge))),shape=(n_edge, n_edge))
    K = H * invDE * H.T


    #
    if add_self_loop:
        print('renormalization!!')
        K += sp.eye(H.shape[0])

    DV = np.sum(K, 0).tolist()[0]
    invDV = np.power(np.array(DV).astype(float), -0.5)
    invDV[np.isinf(invDV)] = 0
    DV2 = coo_matrix((invDV, (range(H.shape[0]), range(H.shape[0]))), shape=(H.shape[0], H.shape[0]))

    G = DV2 * K * DV2
    G = G.tocoo()
    data.edge_index_graph = torch.LongTensor(np.array([G.row, G.col]))
    data.edge_weight_graph = torch.tensor(G.data).float()


    return data


def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    if len(x.shape) != 2:
        X = x.reshape(-1, x.shape[-1])

    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return -dist_mat