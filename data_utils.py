

import torch
import pickle
import os
import ipdb
import numpy as np
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import scipy.sparse as sp
from torch_sparse import coalesce
from sklearn.feature_extraction.text import CountVectorizer
from visual_data import load_feature_construct_H, generate_H_from_dist


def load_NTU_ModelNet(path, dataset, args=None):
    features, labels, idx_train, idx_test, mvcnn_dist, gvcnn_dist = \
        load_feature_construct_H(os.path.join(path, dataset+ "_mvcnn_gvcnn.mat"),
                                 gamma=0.5,
                                 K_neigs=6,
                                 is_probH=True,
                                 use_mvcnn_feature=not args.no_use_mvcnn_feature,
                                 use_gvcnn_feature=not args.no_use_gvcnn_feature,
                                 use_mvcnn_feature_for_structure=not args.no_mvcnn_feature_structure,
                                 use_gvcnn_feature_for_structure=not args.no_gvcnn_feature_structure)
    H = generate_H_from_dist(mvcnn_dist=mvcnn_dist,
                             gvcnn_dist=gvcnn_dist,
                             split_diff_scale=False,
                             gamma=0.5,
                             K_neigs=6,
                             is_probH=True,
                             use_mvcnn_feature_for_structure=not args.no_mvcnn_feature_structure,
                             use_gvcnn_feature_for_structure=not args.no_gvcnn_feature_structure)
    num_hyperedges = H.shape[-1]
    H_sparse = sp.coo_matrix(H)
    edge_index = np.array([H_sparse.row, H_sparse.col])
    edge_attr = np.array(H_sparse.data)
    num_nodes, feature_dim = features.shape
    edge_list = edge_index[1] +  num_nodes
    node_list = edge_index[0]

    edge_index = np.array([np.hstack([node_list, edge_list]),
              np.hstack([edge_list, node_list])], dtype=np.int32)
    edge_attr = np.hstack([edge_attr,edge_attr[:len(edge_list)]])
    data = Data(
            x = torch.FloatTensor(np.array(features[:num_nodes])),
            edge_index = torch.LongTensor(edge_index),
            edge_weight = torch.FloatTensor(edge_attr),
            y = torch.LongTensor(labels[:num_nodes]).squeeze())

    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_weight = coalesce(data.edge_index,
            data.edge_weight,
            total_num_node_id_he_id,
            total_num_node_id_he_id)
    data.num_features = feature_dim
    data.num_classes = len(np.unique(labels))
    data.num_nodes = num_nodes
    data.num_hyperedges = num_hyperedges
    data.train_idx = torch.LongTensor(idx_train)
    data.test_idx = torch.LongTensor(idx_test)

    return data

def load_LE_dataset(path, dataset):

    # load edges, features, and labels.
    print('Loading {} dataset...'.format(dataset))
    file_name = f'{dataset}.content'
    p2idx_features_labels = os.path.join(path, file_name)
    idx_features_labels = np.genfromtxt(p2idx_features_labels,
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))


    print ('load features')

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    file_name = f'{dataset}.edges'
    p2edges_unordered = os.path.join(path, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered,
                                    dtype=np.int32)
    
    
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    print ('load edges')


    projected_features = torch.FloatTensor(np.array(features.todense()))

    
    # From adjacency matrix to edge_list
    edge_index = edges.T 
    assert edge_index[0].max() == edge_index[1].min() - 1

    # check if values in edge_index is consecutive. i.e. no missing value for node_id/he_id.
    assert len(np.unique(edge_index)) == edge_index.max() + 1
    
    num_nodes = edge_index[0].max() + 1
    num_he = edge_index[1].max() - num_nodes + 1
    
    edge_index = np.hstack((edge_index, edge_index[::-1, :]))
    
    # build torch data class
    data = Data(
            x = torch.FloatTensor(np.array(features[:num_nodes].todense())), 
            edge_index = torch.LongTensor(edge_index),
            y = labels[:num_nodes])

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = len(np.unique(edge_index))
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            
    
    data.num_features = data.x.shape[-1]
    data.num_classes = len(np.unique(labels[:num_nodes].numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = num_he
    
    return data

def load_citation_dataset(path, dataset):
    '''
    this will read the citation dataset from HyperGCN, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]
    '''
    print(f'Loading hypergraph dataset from hyperGCN: {dataset}')

    # first load node features:
    with open(os.path.join(path, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        features = features.todense()

    # then load node labels:
    with open(os.path.join(path, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    with open(os.path.join(path, 'hypergraph.pickle'), 'rb') as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f'number of hyperedges: {len(hypergraph)}')

    edge_idx = num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    edge_index = np.array([ node_list + edge_list,
                            edge_list + node_list], dtype = np.int32)
    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)

    data.num_features = features.shape[-1]
    data.num_classes = len(np.unique(labels.numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = len(hypergraph)

    return data

def load_yelp_dataset(path, dataset, name_dictionary_size = 1000):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - latitude, longitude
        - state, in one-hot coding. 
        - city, in one-hot coding. 
        - name, in bag-of-words

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    print(f'Loading hypergraph dataset from {dataset}')

    # first load node features:
    # load longtitude and latitude of restaurant.
    latlong = pd.read_csv(os.path.join(path, 'yelp_restaurant_latlong.csv')).values

    # city - zipcode - state integer indicator dataframe.
    loc = pd.read_csv(os.path.join(path, 'yelp_restaurant_locations.csv'))
    state_int = loc.state_int.values
    city_int = loc.city_int.values

    num_nodes = loc.shape[0]
    state_1hot = np.zeros((num_nodes, state_int.max()))
    state_1hot[np.arange(num_nodes), state_int - 1] = 1

    city_1hot = np.zeros((num_nodes, city_int.max()))
    city_1hot[np.arange(num_nodes), city_int - 1] = 1

    # convert restaurant name into bag-of-words feature.
    vectorizer = CountVectorizer(max_features = name_dictionary_size, stop_words = 'english', strip_accents = 'ascii')
    res_name = pd.read_csv(os.path.join(path, 'yelp_restaurant_name.csv')).values.flatten()
    name_bow = vectorizer.fit_transform(res_name).todense()

    features = np.hstack([latlong, state_1hot, city_1hot, name_bow])

    # then load node labels:
    df_labels = pd.read_csv(os.path.join(path, 'yelp_restaurant_business_stars.csv'))
    labels = df_labels.values.flatten()

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    # Yelp restaurant review hypergraph is store in a incidence matrix.
    H = pd.read_csv(os.path.join(path, 'yelp_restaurant_incidence_H.csv'))
    node_list = H.node.values - 1
    edge_list = H.he.values - 1 + num_nodes

    edge_index = np.vstack([node_list, edge_list])
    edge_index = np.hstack([edge_index, edge_index[::-1, :]])

    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)

    data.y = data.y - data.y.min()
    assert data.y.min().item() == 0

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)

    data.num_features = features.shape[-1]
    data.num_classes = len(np.unique(labels.numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = H.he.values.max()

    return data

def load_cornell_dataset(path, dataset, feature_noise = 0.1, feature_dim = None):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - add gaussian noise with sigma = nosie, mean = one hot coded label.

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    print(f'Loading hypergraph dataset from cornell: {dataset}')

    # first load node labels
    df_labels = pd.read_csv(os.path.join(path, f'node-labels-{dataset}.txt'), names = ['node_label'])
    num_nodes = df_labels.shape[0]
    labels = df_labels.values.flatten()

    # then create node features.
    num_classes = df_labels.values.max()
    features = np.zeros((num_nodes, num_classes))

    features[np.arange(num_nodes), labels - 1] = 1
    if feature_dim is not None:
        num_row, num_col = features.shape
        zero_col = np.zeros((num_row, feature_dim - num_col), dtype = features.dtype)
        features = np.hstack((features, zero_col))

    features = np.random.normal(features, feature_noise, features.shape)
    print(f'number of nodes:{num_nodes}, feature dimension: {features.shape[1]}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    labels = labels - labels.min() # shift label to 0

    # The last, load hypergraph.
    # Corenll datasets are stored in lines of hyperedges. Each line is the set of nodes for that edge.
    p2hyperedge_list = os.path.join(path, f'hyperedges-{dataset}.txt')
    node_list = []
    he_list = []
    he_id = num_nodes

    with open(p2hyperedge_list, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            cur_set = line.split(',')
            cur_set = [int(x) for x in cur_set]

            node_list += cur_set
            he_list += [he_id] * len(cur_set)
            he_id += 1
    # shift node_idx to start with 0.
    node_idx_min = np.min(node_list)
    node_list = [x - node_idx_min for x in node_list]

    edge_index = [node_list + he_list, 
                  he_list + node_list]

    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)
    assert data.y.min().item() == 0

    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)

    data.num_features = features.shape[-1]
    data.num_classes = len(np.unique(labels.numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = he_id - num_nodes
    
    return data
