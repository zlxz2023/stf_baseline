import os
import math
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from fastdtw import fastdtw


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


# load实体矩阵
def load_adj(pkl_filename, adjtype):
    graph = load_pickle(pkl_filename)
    if isinstance(graph, (tuple, list)) and len(graph) == 3:
        sensor_ids, sensor_id_to_ind, adj_mx = graph
    elif isinstance(graph, np.ndarray):
        sensor_ids, sensor_id_to_ind, adj_mx = [], {}, graph
    else:
        raise ValueError('Expected an adjacency matrix or (sensor_ids, mapping, adjacency)')
    
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def laplacian(W):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    d = 1 / np.sqrt(d)
    D = sp.diags(d, 0)
    I = sp.identity(d.size, dtype=W.dtype)
    L = I - D * W * D
    return L

def largest_k_lamb(L, k):
    lamb, U = sp.linalg.eigsh(L, k=k, which='LM')
    return (lamb, U)

def get_eigv(adj,k):
    L = laplacian(adj)
    eig = largest_k_lamb(L,k)
    return eig

def construct_tem_adj(data, num_node, steps_per_day):
    # data_mean = np.mean([data[24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
    data_mean = np.mean([data[steps_per_day*i: steps_per_day*(i+1)] for i in range(data.shape[0]//steps_per_day)], axis=0)
    data_mean = data_mean.squeeze().T
    dtw_distance = np.zeros((num_node, num_node))
    for i in range(num_node):
        for j in range(i, num_node):
            dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
    for i in range(num_node):
        for j in range(i):
            dtw_distance[i][j] = dtw_distance[j][i]

    nth = np.sort(dtw_distance.reshape(-1))[
        int(np.log2(dtw_distance.shape[0])*dtw_distance.shape[0]):
        int(np.log2(dtw_distance.shape[0])*dtw_distance.shape[0])+1] # NlogN edges
    tem_matrix = np.zeros_like(dtw_distance)
    tem_matrix[dtw_distance <= nth] = 1
    tem_matrix = np.logical_or(tem_matrix, tem_matrix.T).astype(int)
    return tem_matrix

# load频域矩阵
def load_Wavelets_Graph(spatial_graph, temporal_graph, dims, data, logger, steps_per_day):
    # calculate spatial and temporal graph wavelets
    graph = load_pickle(spatial_graph)
    if isinstance(graph, (tuple, list)) and len(graph) == 3:
        sensor_ids, sensor_id_to_ind, adj_mx = graph
    elif isinstance(graph, np.ndarray):
        sensor_ids, sensor_id_to_ind, adj_mx = [], {}, graph
    else:
        raise ValueError('Expected an adjacency matrix or (sensor_ids, mapping, adjacency)')
    # 变为对称矩阵
    adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    # 01化
    adj = (adj_mx != 0).astype(int)
    # 添加对角线自环
    if adj[0, 0] == 0:
        adj = adj + np.eye(adj.shape[0])
    if os.path.exists(temporal_graph):
        tem_adj = np.load(temporal_graph)
    else:
        tem_adj = construct_tem_adj(data, adj.shape[0], steps_per_day)
        np.save(temporal_graph, tem_adj)
    spawave = get_eigv(adj, dims)
    temwave = get_eigv(tem_adj, dims)
    logger.info(f'Shape of graphwave eigenvalue and eigenvector: {spawave[0].shape}, {spawave[1].shape}')

    # derive neighbors
    sampled_nodes_number = int(math.log(adj.shape[0], 2))
    graph = csr_matrix(adj)
    dist_matrix = dijkstra(csgraph=graph)
    dist_matrix[dist_matrix==0] = dist_matrix.max() + 10
    localadj = np.argpartition(dist_matrix, sampled_nodes_number, -1)[:, :sampled_nodes_number]

    logger.info(f'Shape of localadj: {localadj.shape}')
    return localadj, spawave, temwave
