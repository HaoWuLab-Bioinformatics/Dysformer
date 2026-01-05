import json
import os
import pickle as pkl
import random
import shutil
from os import path
import urllib.request        # ★ 没有这行就会 NameError
import urllib.error
from urllib.request import urlretrieve

import networkx as nx
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from data_utils import normalize_feat, rand_train_test_idx, split_data
from sklearn.preprocessing import label_binarize
from torch_geometric.datasets import Planetoid, Coauthor
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
DATAPATH = '/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/'


class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(
            self,
            train_prop: float = 0.7,
            valid_prop: float = 0.15,
            split_type: str = "random",
            seed: int = 42):
        """
        若已有 self.split_idx 直接返回；
        否则按比例随机切分 (train_prop, valid_prop, 1-两者之和)。
        """
        # ---------- 兼容旧错误调用 ----------
        if isinstance(train_prop, str):
            # 老代码传的是 (split_type, train_prop, valid_prop)
            split_type, train_prop, valid_prop = train_prop, valid_prop, split_type

        # ---------- 已缓存 ----------
        if getattr(self, "split_idx", None) is not None:
            return self.split_idx

        # ---------- random split ----------
        if split_type == "random":
            ignore_negative = (self.name != "ogbn-proteins")
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label,
                train_prop=train_prop,
                valid_prop=valid_prop,
                ignore_negative=ignore_negative,

            )
            split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}

        else:
            raise ValueError(f"未知 split_type: {split_type}")

        self.split_idx = split_idx
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_nc_dataset(args):
    """ Loader for NCDataset
        Returns NCDataset
    """
    global DATAPATH

    #DATAPATH = args.data_dir
    dataname = args.dataset
    print('>> Loading dataset: {}'.format(dataname))

    if dataname == 'deezer-europe':
        dataset = load_deezer_dataset()

    elif dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(dataname, args.no_feat_norm)

    elif dataname in ('film'):
        dataset = load_geom_gcn_dataset(dataname)

    elif dataname in ('chameleon', 'squirrel'):
        dataset = load_wiki_new(dataname, args.no_feat_norm)

    elif dataname == 'airport':
        dataset = load_airport_dataset()

    elif dataname == 'disease':
        dataset = load_disease_dataset()

    elif dataname == '20news':
        dataset = load_20news()

    elif dataname == 'mini':
        dataset = load_mini_imagenet()

    # Adding new datasets
    elif dataname == 'PPI':
        dataset = load_ppi_dataset()

    elif dataname == 'node2vec_PPI':
        dataset = load_node2vec_PPI_dataset()

    elif dataname == 'Mashup_PPI':
        dataset = load_Mashup_PPI_dataset()

    elif dataname == 'alzheimers':
        dataset = load_alzheimers_dataset()

    elif dataname == 'Clin_Term_COOC':
        dataset = load_clin_term_cooc_dataset()

    elif dataname == 'diabet':
        dataset = load_diabet_dataset()

    elif dataname == 'diabetuci':
        dataset = load_diabetuci_dataset()
    elif dataname == 'jiaolv':
        dataset = load_jiaolv_dataset()
    elif dataname == 'yiyu':
        dataset = load_yiyu_dataset()
    elif dataname == 'jiaolv_missing':
        dataset = load_jiaolv_dataset_missing()
    elif dataname == 'yiyu_missing':
        dataset = load_yiyu_dataset_missing()
    elif dataname == 'PROTEINS':
        dataset = load_PROTEINS_data()
    elif dataname == 'CreditFraud':#金融欺诈
        dataset = load_credit_fraud_data()
    elif dataname == 'METR_LA':#交通流量预测
        dataset = load_traffic_metr_data()
    elif dataname == 'HeartDisease':#医疗预测
        dataset = load_heart_disease_data()
    elif dataname == 'NSL_KDD':#网络安全
        dataset = load_nsl_kdd_data()
    elif dataname == 'GermanCredit':  # 信用评分
        dataset = load_german_credit_data()
    elif dataname == 'CCLE':  # 细胞系分类，通过基因表达数据预测癌症类型
        dataset = load_ccle_data()
    elif dataname == 'BreastCancer':  # 乳腺癌的分类，基于肿瘤细胞的特征，图数据集
        dataset = load_breast_cancer_data()
    elif dataname == 'Coauthor_physics':  # 乳腺癌的分类，基于肿瘤细胞的特征，图数据集
        dataset = load_coauthor_physics_dataset()
    elif dataname == 'Coauthor_CS':  # 乳腺癌的分类，基于肿瘤细胞的特征，图数据集
        dataset = load_coauthor_CS_dataset()
    elif dataname == 'Gene_Disease_Association_Prediction_with_GAT':  # 乳腺癌的分类，基于肿瘤细胞的特征，图数据集
        dataset = load_gene_disease_association_prediction_with_gat()#用于 疾病预测 的图数据集

    else:
        raise ValueError('Invalid dataname')
    return dataset



def load_deezer_dataset():
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{DATAPATH}/deezer/deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_airport_dataset():
    filename = 'airport'
    dataset = NCDataset(filename)
    graph = pkl.load(open(os.path.join(DATAPATH, 'hgcn_data', 'airport', 'airport.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph._node[u]['feat'] for u in graph.nodes()])
    label_idx = 4
    labels = features[:, label_idx]
    features = features[:, :label_idx]
    labels = bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])
    num_nodes = adj.shape[0]
    features = torch.tensor(features, dtype=torch.float)
    val_prop, test_prop = 0.15, 0.15
    idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop)
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = labels
    dataset.train_idx = idx_train
    dataset.valid_idx = idx_val
    dataset.test_idx = idx_test
    return dataset


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


def load_planetoid_dataset(name, no_feat_norm=False):
    # import pdb
    # pdb.set_trace()
    if not no_feat_norm:
        transform = T.NormalizeFeatures()
        torch_dataset = Planetoid(root=f'{DATAPATH}/Planetoid',
                                  name=name, transform=transform)
    else:
        torch_dataset = Planetoid(root=f'{DATAPATH}/Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset


def load_geom_gcn_dataset(name):
    # graph_adjacency_list_file_path = '../../data/geom-gcn/{}/out1_graph_edges.txt'.format(
    #     name)
    # graph_node_features_and_labels_file_path = '../../data/geom-gcn/{}/out1_node_feature_label.txt'.format(
    #     name)
    graph_adjacency_list_file_path = os.path.join(DATAPATH, 'geom-gcn/{}/out1_graph_edges.txt'.format(name))
    graph_node_features_and_labels_file_path = os.path.join(DATAPATH,
                                                            'geom-gcn/{}/out1_node_feature_label.txt'.format(name))

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if name == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(
                    line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(
                    line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(
                    line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(
                    line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.tocoo().astype(np.float32)
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    print(features.shape)

    def preprocess_features(feat):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(feat.sum(1))
        rowsum = (rowsum == 0) * 1 + rowsum
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        feat = r_mat_inv.dot(feat)
        return feat

    features = preprocess_features(features)

    edge_index = torch.from_numpy(
        np.vstack((adj.row, adj.col)).astype(np.int64))
    node_feat = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    num_nodes = node_feat.shape[0]
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = labels

    return dataset


def load_wiki_new(name, no_feat_norm=False):
    path = os.path.join(DATAPATH, f'wiki_new/{name}/{name}_filtered.npz')
    data = np.load(path)
    # lst=data.files
    # for item in lst:
    #     print(item)
    node_feat = data['node_features']  # unnormalized
    labels = data['node_labels']
    edges = data['edges']  # (E, 2)
    edge_index = edges.T

    if not no_feat_norm:
        node_feat = normalize_feat(node_feat)

    dataset = NCDataset(name)

    edge_index = torch.as_tensor(edge_index)
    node_feat = torch.as_tensor(node_feat)
    labels = torch.as_tensor(labels)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': node_feat.shape[0]}
    dataset.label = labels

    return dataset


def load_disease_dataset():
    object_to_idx = {}
    idx_counter = 0
    edges = []
    name = "disease_nc"
    dataset = NCDataset(name)

    with open(os.path.join(DATAPATH, 'hgcn_data', f'{name}', f"{name}.edges.csv"), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    features = sp.load_npz(os.path.join(DATAPATH, 'hgcn_data', f'{name}', "{}.feats.npz".format("disease_nc")))
    if sp.issparse(features):
        features = features.toarray()
    features = normalize_feat(features)
    features = torch.tensor(features, dtype=torch.float)

    labels = np.load(os.path.join(DATAPATH, 'hgcn_data', f'{name}', "{}.labels.npy".format("disease_nc")))
    val_prop, test_prop = 0.10, 0.60
    idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop)
    num_nodes = adj.shape[0]
    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    labels = torch.LongTensor(labels)
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = labels
    dataset.train_idx = idx_train
    dataset.valid_idx = idx_val
    dataset.test_idx = idx_test

    return dataset


def load_20news(n_remove=0):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    import pickle as pkl

    if path.exists(DATAPATH + '20news/20news.pkl'):
        data = pkl.load(open(DATAPATH + '20news/20news.pkl', 'rb'))
    else:
        categories = ['alt.atheism',
                      'comp.sys.ibm.pc.hardware',
                      'misc.forsale',
                      'rec.autos',
                      'rec.sport.hockey',
                      'sci.crypt',
                      'sci.electronics',
                      'sci.med',
                      'sci.space',
                      'talk.politics.guns']
        data = fetch_20newsgroups(subset='all', categories=categories)
        # with open(data_dir + '20news/20news.pkl', 'wb') as f:
        #     pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)

    vectorizer = CountVectorizer(stop_words='english', min_df=0.05)
    X_counts = vectorizer.fit_transform(data.data).toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    features = transformer.fit_transform(X_counts).todense()
    features = torch.Tensor(features)
    y = data.target
    y = torch.LongTensor(y)

    num_nodes = features.shape[0]

    if n_remove > 0:
        num_nodes -= n_remove
        features = features[:num_nodes, :]
        y = y[:num_nodes]

    dataset = NCDataset('20news')
    dataset.graph = {'edge_index': None,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = torch.LongTensor(y)

    return dataset


def load_mini_imagenet():
    import pickle as pkl

    dataset = NCDataset('mini_imagenet')
    DATAPATH="/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/"
    data = pkl.load(open(os.path.join(DATAPATH, 'mini_imagenet/mini_imagenet.pkl'), 'rb'))
    x_train = data['x_train']
    x_val = data['x_val']
    x_test = data['x_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']

    features = torch.cat((x_train, x_val, x_test), dim=0)
    labels = np.concatenate((y_train, y_val, y_test))
    num_nodes = features.shape[0]

    dataset.graph = {'edge_index': None,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = torch.LongTensor(labels)
    return dataset


def load_ppi_dataset():
    folder_path = "/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian/PPI/raw"

    # Helper function to load numpy arrays
    def load_npy(file):
        return np.load(os.path.join(folder_path, file))

    # Helper function to load and process graph files
    def load_graph(file):
        with open(os.path.join(folder_path, file), 'r') as f:
            graph = json.load(f)
        nodes = [node['id'] for node in graph['nodes']]
        edge_list = [(edge['source'], edge['target']) for edge in graph['links']]
        row = [edge[0] for edge in edge_list]
        col = [edge[1] for edge in edge_list]
        data = np.ones(len(edge_list))
        adj = coo_matrix((data, (row, col)), shape=(len(nodes), len(nodes)))
        return adj, len(nodes)

    # Load the data
    train_adj, num_nodes = load_graph('train_graph.json')
    val_adj, _ = load_graph('valid_graph.json')
    test_adj, _ = load_graph('test_graph.json')

    train_feats = load_npy('train_feats.npy')
    val_feats = load_npy('valid_feats.npy')
    test_feats = load_npy('test_feats.npy')

    train_labels = load_npy('train_labels.npy')
    val_labels = load_npy('valid_labels.npy')
    test_labels = load_npy('test_labels.npy')

    # Concatenate the data
    adj = coo_matrix((np.ones(train_adj.nnz + val_adj.nnz + test_adj.nnz),
                      (np.hstack([train_adj.row, val_adj.row, test_adj.row]),
                       np.hstack([train_adj.col, val_adj.col, test_adj.col]))),
                     shape=(num_nodes, num_nodes))

    features = np.vstack((train_feats, val_feats, test_feats))
    labels = np.concatenate((train_labels, val_labels, test_labels))

    # Create index arrays
    idx_train = np.arange(train_feats.shape[0])
    idx_val = np.arange(train_feats.shape[0], train_feats.shape[0] + val_feats.shape[0])
    idx_test = np.arange(train_feats.shape[0] + val_feats.shape[0],
                         train_feats.shape[0] + val_feats.shape[0] + test_feats.shape[0])

    # Convert adjacency matrix to edge_index (format for PyTorch Geometric)
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    # Create the dataset object
    dataset = NCDataset('ppi')
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,  # No edge features in this case
        'node_feat': torch.tensor(features, dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)
    dataset.train_idx = torch.tensor(idx_train, dtype=torch.long)
    dataset.valid_idx = torch.tensor(idx_val, dtype=torch.long)
    dataset.test_idx = torch.tensor(idx_test, dtype=torch.long)

    return dataset



def load_node2vec_PPI_dataset():
    dataset_str = 'node2vec_PPI'
    data_path    = '/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian/node2vec_PPI'
    use_feats    = True

    edge_file  = os.path.join(data_path, f'{dataset_str}.edgelist')
    label_file = os.path.join(data_path, f'{dataset_str}_labels.txt')

    # ----------- 读取边 -------------
    # 只要 src、dst 两列
    edges = np.loadtxt(edge_file, dtype=np.int64, usecols=(0, 1))
    num_nodes = edges.max() + 1

    row, col = edges[:, 0], edges[:, 1]
    data = np.ones(len(edges), dtype=np.float32)
    adj  = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj  = adj + adj.T         # 无向图
    adj  = adj.tocoo()

    # ----------- 节点特征 -----------
    if use_feats:
        features = sp.identity(num_nodes, format='csr', dtype=np.float32)
    else:
        features = np.ones((num_nodes, 1), dtype=np.float32)

    # ----------- 读取多标签 ----------
    # 先扫描文件确定类别数
    max_label = -1
    with open(label_file) as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if len(parts) > 1:
                max_label = max(max_label, max(parts[1:]))

    num_classes = max_label + 1              # e.g. 153
    labels = -np.ones((num_nodes, num_classes), dtype=np.float32)

    # 第二遍真正填值
    with open(label_file) as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            node_id, labs = parts[0], parts[1:]
            labels[node_id] = 0               # 先清 0
            if labs:                          # 可能有节点无标签
                labels[node_id, labs] = 1

    # ----------- 打包成 Dataset -------
    edge_index = torch.from_numpy(np.vstack((adj.row, adj.col))).long()
    node_feat  = torch.tensor(features.todense(), dtype=torch.float32)
    label_t    = torch.from_numpy(labels)     # shape (N, num_classes)

    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat' : None,
        'node_feat' : node_feat,
        'num_nodes' : num_nodes,
    }
    dataset.label = label_t                  # (N, C)  multi‑label

    # ----------- 简单检查 -------------
    print(f'节点数: {num_nodes}')
    print(f'标签张量: {dataset.label.shape}, dtype={dataset.label.dtype}')
    print('标签示例 (前 3 行):\n', dataset.label[:3])

    return dataset

def load_Mashup_PPI_dataset():
    dataset_str = 'Mashup_PPI'
    data_path   = '/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian/Mashup_PPI'
    use_feats   = True

    edge_file  = os.path.join(data_path, f'{dataset_str}.edgelist')
    label_file = os.path.join(data_path, f'{dataset_str}_labels.txt')

    # ----------- 读取边 -------------
    edges = np.loadtxt(edge_file, dtype=np.int64, usecols=(0, 1))
    num_nodes = edges.max() + 1

    row, col = edges[:, 0], edges[:, 1]
    data = np.ones(len(edges), dtype=np.float32)
    adj  = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj  = adj + adj.T
    adj  = adj.tocoo()

    # ----------- 节点特征 -----------
    if use_feats:
        features = sp.identity(num_nodes, format='csr', dtype=np.float32)
    else:
        features = np.ones((num_nodes, 1), dtype=np.float32)

    # ----------- 读取多标签 ----------
    # 扫描最大标签值确定类别数
    max_label = -1
    with open(label_file) as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if len(parts) > 1:
                max_label = max(max_label, max(parts[1:]))
    num_classes = max_label + 1

    # 初始化标签为全 0（表示未标注）
    labels = np.zeros((num_nodes, num_classes), dtype=np.float32)

    # 读取真实标签（置为 1）
    with open(label_file) as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            node_id, labs = parts[0], parts[1:]
            if labs:
                labels[node_id, labs] = 1

    # ----------- 打包成 Dataset -------
    edge_index = torch.from_numpy(np.vstack((adj.row, adj.col))).long()
    node_feat  = torch.tensor(features.todense(), dtype=torch.float32)
    label_t    = torch.from_numpy(labels)  # shape (N, num_classes)

    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat' : None,
        'node_feat' : node_feat,
        'num_nodes' : num_nodes,
    }
    dataset.label = label_t  # 多标签 (N, C)

    # ----------- 简单检查 -------------
    num_labeled = (labels.sum(axis=1) > 0).sum()
    print(f'节点数: {num_nodes}')
    print(f'有标签节点数: {num_labeled}')
    print(f'标签张量: {dataset.label.shape}, dtype={dataset.label.dtype}')
    print('标签示例 (前 3 行):\n', dataset.label[:3])

    return dataset




def load_alzheimers_dataset():
    dataset_str = 'alzheimers'
    data_path = '/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian/alzheimers'
    use_feats = True
    label_col = 'Group'
    knn_k = 5  # KNN 邻居数

    # 读取 Excel 数据
    data_file_path = os.path.join(data_path, f'{dataset_str}.xlsx')
    data = pd.read_excel(data_file_path)

    # 提取标签
    labels_raw = data[label_col].values
    classes, labels = np.unique(labels_raw, return_inverse=True)  # 转成整数编码标签

    # 特征处理
    if use_feats:
        feature_values = data.drop(columns=[label_col]).values
        features = sp.csr_matrix(feature_values)
    else:
        features = np.ones((data.shape[0], 1))

    num_nodes = data.shape[0]

    # 构建邻接矩阵（KNN）
    knn = NearestNeighbors(n_neighbors=knn_k)
    knn.fit(features.toarray())
    knn_indices = knn.kneighbors(return_distance=False)

    adj = sp.lil_matrix((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in knn_indices[i]:
            adj[i, j] = 1
            adj[j, i] = 1  # 保证对称性
    adj = adj.tocoo()

    # 构建 edge_index
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    # 构造 NCDataset 返回对象
    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray(), dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    # 默认划分
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']

    return dataset




def load_clin_term_cooc_dataset():
    dataset_str = 'Clin_Term_COOC'
    data_path = '/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian/Clin_Term_COOC'
    use_feats = True

    # === 1. 读取 edgelist 构建邻接矩阵 ===
    edgelist_path = os.path.join(data_path, f'{dataset_str}.edgelist')
    edges = np.loadtxt(edgelist_path, dtype=np.float32)
    num_nodes = int(np.max(edges[:, :2])) + 1
    adj = sp.coo_matrix((edges[:, 2], (edges[:, 0].astype(int), edges[:, 1].astype(int))),
                        shape=(num_nodes, num_nodes))

    # === 2. 读取 node_list.txt 判断节点总数（非必要，仅验证） ===
    node_list_path = os.path.join(data_path, 'node_list.txt')
    node_ids = np.loadtxt(node_list_path, dtype=int, skiprows=1, usecols=(0,))
    assert num_nodes == int(np.max(node_ids)) + 1, "节点数量与node_list.txt不一致！"

    # === 3. 构建特征 ===
    if use_feats:
        features = sp.identity(num_nodes, dtype=np.float32)
    else:
        features = np.ones((num_nodes, 1), dtype=np.float32)

    # === 4. 读取标签 ===
    def read_node_labels(label_file):
        node_list = []
        label_list = []
        with open(label_file, 'r') as f:
            next(f)  # 跳过标题行
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    node_list.append(int(parts[0]))
                    label_list.append(int(parts[1]))
        return node_list, label_list

    labels_path = os.path.join(data_path, f'{dataset_str}_labels.txt')
    node_list, label_values = read_node_labels(labels_path)
    labels = np.zeros((num_nodes,), dtype=np.int64)
    labels[node_list] = label_values
    labels = torch.tensor(labels, dtype=torch.long)

    # === 5. 构建 edge_index ===
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    # === 6. 构建 NCDataset 对象 ===
    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray() if sp.issparse(features) else features,
                                  dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = labels

    # === 7. 默认划分 train / valid / test ===
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']

    return dataset

def load_diabet_dataset():
    dataset_name = 'diabet'
    data_path = '/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian/diabet'
    use_feats = True

    # 1. 读取CSV数据
    csv_path = os.path.join(data_path, f'{dataset_name}.csv')
    data = pd.read_csv(csv_path)

    # 2. 分离特征和标签
    if use_feats:
        feature_values = data.iloc[:, :-1].values
        features = sp.csr_matrix(feature_values)
    else:
        features = np.ones((data.shape[0], 1), dtype=np.float32)

    labels = data.iloc[:, -1].values
    labels = torch.tensor(labels, dtype=torch.long)

    # 3. 构建邻接矩阵（KNN）
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(feature_values)
    knn_distances, knn_indices = knn.kneighbors(feature_values)

    num_nodes = feature_values.shape[0]
    adj = sp.lil_matrix((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in knn_indices[i]:
            adj[i, j] = 1
            adj[j, i] = 1

    adj = adj.tocoo()

    # 4. 构建 edge_index
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    # 5. 构建 NCDataset 对象
    dataset = NCDataset(dataset_name)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray() if sp.issparse(features) else features,
                                  dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = labels

    # 6. 划分训练/验证/测试集索引
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']

    return dataset



def load_diabetuci_dataset():
    dataset_name = 'diabetuci'
    data_path = '/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian/diabetuci'
    use_feats = True

    # 1. 读取CSV数据
    csv_path = os.path.join(data_path, f'{dataset_name}.csv')
    data = pd.read_csv(csv_path)

    # 2. 分离特征和标签
    if use_feats:
        feature_values = data.iloc[:, :-1].values
        features = sp.csr_matrix(feature_values)
    else:
        features = np.ones((data.shape[0], 1), dtype=np.float32)

    labels = data.iloc[:, -1].values
    labels = torch.tensor(labels, dtype=torch.long)

    # 3. 构建邻接矩阵（KNN）
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(feature_values)
    knn_distances, knn_indices = knn.kneighbors(feature_values)

    num_nodes = feature_values.shape[0]
    adj = sp.lil_matrix((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in knn_indices[i]:
            adj[i, j] = 1
            adj[j, i] = 1

    adj = adj.tocoo()

    # 4. 构建 edge_index
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    # 5. 构建 NCDataset 对象
    dataset = NCDataset(dataset_name)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray() if sp.issparse(features) else features,
                                  dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = labels

    # 6. 划分训练/验证/测试集索引
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']

    return dataset
import os
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from dataset import NCDataset  # 假设你有 NCDataset 类
import os, numpy as np, pandas as pd, torch
import scipy.sparse as sp  # ★ 新增
from sklearn.neighbors import NearestNeighbors
from types import SimpleNamespace  # 若用自定义 NCDataset 可删

'''
def load_jiaolv_dataset():
    dataset_str = 'jiaolv'
    data_path = '/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian/jiaolvyiyu'
    file_path = os.path.join(data_path, '2020-2023年数据汇总_等级划分_最终.xlsx')

    # 1. 加载数据
    data = pd.read_excel(file_path)
    data = data.dropna(subset=['焦虑症状等级']).reset_index(drop=True)

    # 2. 简单的特征工程
    data['运动协调性'] = (data['50米跑得分'] * data['立定跳远得分']) ** 0.5
    data['心肺耐力'] = (data['肺活量得分'] + data['耐力跑得分']) / 2

    # 3. 确定分类特征 & 数值特征
    cat_features = ['性别', '年级', '居住地', '家庭经济水平', '父亲教育水平', '母亲教育水平']
    num_features = ['BMI得分', '肺活量得分', '50米跑得分', '坐位体前屈得分',
                    '立定跳远得分', '运动协调性', '心肺耐力', '精神病性']

    # 4. 构造预处理管道
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', QuantileTransformer(n_quantiles=1000, output_distribution='normal'))
        ]), num_features)
    ])

    # 在训练集上 fit
    preprocessor.fit(data)

    # 转换数据
    X_data = preprocessor.transform(data)

    # 处理标签
    label_map = {'焦虑症状不明显': 0, '中度焦虑症状': 1, '焦虑症状较明显': 2}
    labels = data['焦虑症状等级'].map(label_map).values
    labels = torch.tensor(labels, dtype=torch.long)

    # 5. 使用 KNN 构建邻接矩阵
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(X_data)
    knn_distances, knn_indices = knn.kneighbors(X_data)

    num_nodes = X_data.shape[0]
    adj = sp.lil_matrix((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in knn_indices[i]:
            adj[i, j] = 1
            adj[j, i] = 1  # 保证邻接矩阵是对称的

    adj = adj.tocoo()

    # 6. 构建 edge_index
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    # 7. 构建 NCDataset 对象
    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(X_data, dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = labels

    # 8. 默认划分 train / valid / test（此部分已经由 get_idx_split 处理）
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']

    return dataset
'''
def load_jiaolv_dataset():
    dataset_str = 'jiaolv'
    data_path = '/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian/jiaolvyiyu'
    file_path = os.path.join(data_path, '2020-2023年数据汇总_等级划分_最终.xlsx')

    # 1. 加载数据
    data = pd.read_excel(file_path)
    data = data.dropna(subset=['焦虑症状等级']).reset_index(drop=True)

    # 2. 简单的特征工程
    data['运动协调性'] = (data['50米跑得分'] * data['立定跳远得分']) ** 0.5
    data['心肺耐力'] = (data['肺活量得分'] + data['耐力跑得分']) / 2

    # 3. 确定分类特征 & 数值特征
    cat_features = ['性别', '年级', '居住地', '家庭经济水平', '父亲教育水平', '母亲教育水平']
    num_features = ['BMI得分', '肺活量得分', '50米跑得分', '坐位体前屈得分',
                    '立定跳远得分', '运动协调性', '心肺耐力', '精神病性']

    # 4. 预处理
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))

            #('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', QuantileTransformer(n_quantiles=1000, output_distribution='normal'))
        ]), num_features)
    ])

    preprocessor.fit(data)
    X_data = preprocessor.transform(data)

    # 收集特征名（用于画图 x 轴）
    #cat_names = preprocessor.named_transformers_['cat'][1].get_feature_names_out(cat_features)
    cat_encoder = preprocessor.named_transformers_['cat'][1]
    cat_names = cat_encoder.get_feature_names(cat_features)

    feat_names = list(cat_names) + num_features

    # 5. 标签
    label_map = {'焦虑症状不明显': 0, '中度焦虑症状': 1, '焦虑症状较明显': 2}
    labels = torch.tensor(data['焦虑症状等级'].map(label_map).values, dtype=torch.long)

    # 6. KNN 构图
    knn = NearestNeighbors(n_neighbors=5).fit(X_data)
    _, knn_indices = knn.kneighbors(X_data)

    num_nodes = X_data.shape[0]
    adj = sp.lil_matrix((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in knn_indices[i]:
            adj[i, j] = adj[j, i] = 1

    adj = adj.tocoo()                               # ★ 关键：转成 COO 才有 row/col

    edge_index = torch.tensor(
        np.vstack((adj.row, adj.col)), dtype=torch.long
    )

    # 7. 组装数据集（示例：SimpleNamespace；你也可以替换为自定义 NCDataset）
    dataset = SimpleNamespace()
    dataset.name = dataset_str
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(X_data, dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = labels
    dataset.feat_names = feat_names                 # ★ 供后续画图使用

    # 8. 默认划分
    # 8. 默认划分 —— 先得到三组索引
    from sklearn.model_selection import train_test_split
    idx_all = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(
        idx_all, test_size=0.2, stratify=labels, random_state=42
    )
    train_idx, valid_idx = train_test_split(
        train_idx, test_size=0.2, stratify=labels[train_idx], random_state=42
    )

    # 转为张量
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    valid_idx = torch.tensor(valid_idx, dtype=torch.long)
    test_idx  = torch.tensor(test_idx,  dtype=torch.long)

    # 保存到 dataset
    dataset.train_idx = train_idx
    dataset.valid_idx = valid_idx
    dataset.test_idx  = test_idx

    split_dict = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    # ---- 关键修改：让 get_idx_split 能接收任意参数 ----
    dataset.get_idx_split = lambda *args, **kwargs: split_dict
    # ------------------------------------------------

    return dataset


def load_yiyu_dataset():

    dataset_str = 'yiyu'
    data_path = '/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian/jiaolvyiyu'
    file_path = os.path.join(data_path, '2020-2023年数据汇总_等级划分_最终.xlsx')

    # 1. 加载数据
    data = pd.read_excel(file_path)
    data = data.dropna(subset=['抑郁症状等级']).reset_index(drop=True)

    # 2. 简单的特征工程
    data['运动协调性'] = (data['50米跑得分'] * data['立定跳远得分']) ** 0.5
    data['心肺耐力'] = (data['肺活量得分'] + data['耐力跑得分']) / 2

    # 3. 确定分类特征 & 数值特征
    cat_features = ['性别', '年级', '居住地', '家庭经济水平', '父亲教育水平', '母亲教育水平']
    num_features = ['BMI得分', '肺活量得分', '50米跑得分', '坐位体前屈得分',
                    '立定跳远得分', '运动协调性', '心肺耐力', '精神病性']

    # 4. 预处理
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))

            #('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', QuantileTransformer(n_quantiles=1000, output_distribution='normal'))
        ]), num_features)
    ])

    preprocessor.fit(data)
    X_data = preprocessor.transform(data)

    # 收集特征名（用于画图 x 轴）
    #cat_names = preprocessor.named_transformers_['cat'][1].get_feature_names_out(cat_features)
    cat_encoder = preprocessor.named_transformers_['cat'][1]
    cat_names = cat_encoder.get_feature_names(cat_features)

    feat_names = list(cat_names) + num_features

    # 5. 标签
    label_map = {'抑郁症状不明显': 0, '中度抑郁症状': 1, '抑郁症状较明显': 2}
    labels = torch.tensor(data['抑郁症状等级'].map(label_map).values, dtype=torch.long)

    # 6. KNN 构图
    knn = NearestNeighbors(n_neighbors=5).fit(X_data)
    _, knn_indices = knn.kneighbors(X_data)

    num_nodes = X_data.shape[0]
    adj = sp.lil_matrix((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in knn_indices[i]:
            adj[i, j] = adj[j, i] = 1

    adj = adj.tocoo()                               # ★ 关键：转成 COO 才有 row/col

    edge_index = torch.tensor(
        np.vstack((adj.row, adj.col)), dtype=torch.long
    )

    # 7. 组装数据集（示例：SimpleNamespace；你也可以替换为自定义 NCDataset）
    dataset = SimpleNamespace()
    dataset.name = dataset_str
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(X_data, dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = labels
    dataset.feat_names = feat_names                 # ★ 供后续画图使用

    # 8. 默认划分
    # 8. 默认划分 —— 先得到三组索引
    from sklearn.model_selection import train_test_split
    idx_all = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(
        idx_all, test_size=0.2, stratify=labels, random_state=42
    )
    train_idx, valid_idx = train_test_split(
        train_idx, test_size=0.2, stratify=labels[train_idx], random_state=42
    )

    # 转为张量
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    valid_idx = torch.tensor(valid_idx, dtype=torch.long)
    test_idx  = torch.tensor(test_idx,  dtype=torch.long)

    # 保存到 dataset
    dataset.train_idx = train_idx
    dataset.valid_idx = valid_idx
    dataset.test_idx  = test_idx

    split_dict = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    # ---- 关键修改：让 get_idx_split 能接收任意参数 ----
    dataset.get_idx_split = lambda *args, **kwargs: split_dict
    # ------------------------------------------------

    return dataset

def load_jiaolv_dataset_missing():
    dataset_str = 'jiaolv'
    data_path = '/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian/jiaolvyiyu'
    file_path = os.path.join(data_path, '2020-2023年体质+问卷数据汇总处理后_等级划分(带缺失).xlsx')

    # 1. 加载数据
    data = pd.read_excel(file_path)
    data = data.dropna(subset=['焦虑等级']).reset_index(drop=True)

    # 2. 简单的特征工程
    data['运动协调性'] = (data['50米跑得分'] * data['立定跳远得分']) ** 0.5
    data['心肺耐力'] = (data['肺活量得分'] + data['耐力跑得分']) / 2

    # 3. 确定分类特征 & 数值特征
    cat_features = ['性别', '年级', '居住地', '家庭经济水平', '父亲教育水平', '母亲教育水平']
    num_features = ['BMI得分', '肺活量得分', '50米跑得分', '坐位体前屈得分',
                    '立定跳远得分', '运动协调性', '心肺耐力', '精神病性']

    # 4. 预处理
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))

            #('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', QuantileTransformer(n_quantiles=1000, output_distribution='normal'))
        ]), num_features)
    ])

    preprocessor.fit(data)
    X_data = preprocessor.transform(data)

    # 收集特征名（用于画图 x 轴）
    #cat_names = preprocessor.named_transformers_['cat'][1].get_feature_names_out(cat_features)
    cat_encoder = preprocessor.named_transformers_['cat'][1]
    cat_names = cat_encoder.get_feature_names(cat_features)

    feat_names = list(cat_names) + num_features

    # 5. 标签
    label_map = {'焦虑症状不明显': 0, '中度焦虑症状': 1, '焦虑症状较明显': 2}
    labels = torch.tensor(data['焦虑等级'].map(label_map).values, dtype=torch.long)

    # 6. KNN 构图
    knn = NearestNeighbors(n_neighbors=5).fit(X_data)
    _, knn_indices = knn.kneighbors(X_data)

    num_nodes = X_data.shape[0]
    adj = sp.lil_matrix((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in knn_indices[i]:
            adj[i, j] = adj[j, i] = 1

    adj = adj.tocoo()                               # ★ 关键：转成 COO 才有 row/col

    edge_index = torch.tensor(
        np.vstack((adj.row, adj.col)), dtype=torch.long
    )

    # 7. 组装数据集（示例：SimpleNamespace；你也可以替换为自定义 NCDataset）
    dataset = SimpleNamespace()
    dataset.name = dataset_str
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(X_data, dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = labels
    dataset.feat_names = feat_names                 # ★ 供后续画图使用

    # 8. 默认划分
    # 8. 默认划分 —— 先得到三组索引
    from sklearn.model_selection import train_test_split
    idx_all = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(
        idx_all, test_size=0.2, stratify=labels, random_state=42
    )
    train_idx, valid_idx = train_test_split(
        train_idx, test_size=0.2, stratify=labels[train_idx], random_state=42
    )

    # 转为张量
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    valid_idx = torch.tensor(valid_idx, dtype=torch.long)
    test_idx  = torch.tensor(test_idx,  dtype=torch.long)

    # 保存到 dataset
    dataset.train_idx = train_idx
    dataset.valid_idx = valid_idx
    dataset.test_idx  = test_idx

    split_dict = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    # ---- 关键修改：让 get_idx_split 能接收任意参数 ----
    dataset.get_idx_split = lambda *args, **kwargs: split_dict
    # ------------------------------------------------

    return dataset


def load_yiyu_dataset_missing():

    dataset_str = 'yiyu'
    data_path = '/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian/jiaolvyiyu'
    file_path = os.path.join(data_path, '2020-2023年体质+问卷数据汇总处理后_等级划分(带缺失).xlsx')

    # 1. 加载数据
    data = pd.read_excel(file_path)
    data = data.dropna(subset=['抑郁等级']).reset_index(drop=True)

    # 2. 简单的特征工程
    data['运动协调性'] = (data['50米跑得分'] * data['立定跳远得分']) ** 0.5
    data['心肺耐力'] = (data['肺活量得分'] + data['耐力跑得分']) / 2

    # 3. 确定分类特征 & 数值特征
    cat_features = ['性别', '年级', '居住地', '家庭经济水平', '父亲教育水平', '母亲教育水平']
    num_features = ['BMI得分', '肺活量得分', '50米跑得分', '坐位体前屈得分',
                    '立定跳远得分', '运动协调性', '心肺耐力', '精神病性']

    # 4. 预处理
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))

            #('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', QuantileTransformer(n_quantiles=1000, output_distribution='normal'))
        ]), num_features)
    ])

    preprocessor.fit(data)
    X_data = preprocessor.transform(data)

    # 收集特征名（用于画图 x 轴）
    #cat_names = preprocessor.named_transformers_['cat'][1].get_feature_names_out(cat_features)
    cat_encoder = preprocessor.named_transformers_['cat'][1]
    cat_names = cat_encoder.get_feature_names(cat_features)

    feat_names = list(cat_names) + num_features

    # 5. 标签
    label_map = {'抑郁症状不明显': 0, '中度抑郁症状': 1, '抑郁症状较明显': 2}
    labels = torch.tensor(data['抑郁等级'].map(label_map).values, dtype=torch.long)

    # 6. KNN 构图
    knn = NearestNeighbors(n_neighbors=5).fit(X_data)
    _, knn_indices = knn.kneighbors(X_data)

    num_nodes = X_data.shape[0]
    adj = sp.lil_matrix((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in knn_indices[i]:
            adj[i, j] = adj[j, i] = 1

    adj = adj.tocoo()                               # ★ 关键：转成 COO 才有 row/col

    edge_index = torch.tensor(
        np.vstack((adj.row, adj.col)), dtype=torch.long
    )

    # 7. 组装数据集（示例：SimpleNamespace；你也可以替换为自定义 NCDataset）
    dataset = SimpleNamespace()
    dataset.name = dataset_str
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(X_data, dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = labels
    dataset.feat_names = feat_names                 # ★ 供后续画图使用

    # 8. 默认划分
    # 8. 默认划分 —— 先得到三组索引
    from sklearn.model_selection import train_test_split
    idx_all = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(
        idx_all, test_size=0.2, stratify=labels, random_state=42
    )
    train_idx, valid_idx = train_test_split(
        train_idx, test_size=0.2, stratify=labels[train_idx], random_state=42
    )

    # 转为张量
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    valid_idx = torch.tensor(valid_idx, dtype=torch.long)
    test_idx  = torch.tensor(test_idx,  dtype=torch.long)

    # 保存到 dataset
    dataset.train_idx = train_idx
    dataset.valid_idx = valid_idx
    dataset.test_idx  = test_idx

    split_dict = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    # ---- 关键修改：让 get_idx_split 能接收任意参数 ----
    dataset.get_idx_split = lambda *args, **kwargs: split_dict
    # ------------------------------------------------

    return dataset

def load_PROTEINS_data():
    dataset_str = 'PROTEINS'
    data_path = '/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian/PROTEINS'

    # 加载 PROTEINS 数据（自动下载至指定路径）
    pyg_data = TUDataset(root=data_path, name=dataset_str)

    # 合并所有图为一个大图（node classification 任务）
    all_graphs = []
    all_labels = []
    offset = 0
    edge_list = []
    for data in pyg_data:
        G = to_networkx(data, to_undirected=True)
        n_nodes = G.number_of_nodes()
        mapping = {i: i + offset for i in range(n_nodes)}
        G = nx.relabel_nodes(G, mapping)
        edge_list += list(G.edges())
        all_graphs.append(G)
        all_labels.append(int(data.y.item()))
        offset += n_nodes

    num_nodes = offset
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # 简化节点特征：使用单位特征（可以根据需要替换为实际 data.x 拼接）
    features = sp.identity(num_nodes, format='csr', dtype=np.float32)

    # 简化标签处理：将每个图的 label 复制给它的所有节点
    labels = np.zeros(num_nodes, dtype=np.int64)
    offset = 0
    for i, G in enumerate(all_graphs):
        n = G.number_of_nodes()
        labels[offset:offset + n] = all_labels[i]
        offset += n

    # 封装为 NCDataset 格式
    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray(), dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    # 默认划分
    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx  = split['test']

    return dataset






def build_knn_graph(features, k=5):
    adj = kneighbors_graph(features, n_neighbors=k, mode='connectivity', include_self=False)
    adj = adj + adj.T  # 对称化
    adj.setdiag(0)
    edge_index = torch.tensor(np.vstack(adj.nonzero()), dtype=torch.long)
    return edge_index


DATA_ROOT = "/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian"


def load_credit_fraud_data():
    dataset_str = 'CreditFraud'
    local_path = os.path.join(DATA_ROOT, 'creditcard.csv')
    if not os.path.exists(local_path):
        url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
        df = pd.read_csv(url)
        df.to_csv(local_path, index=False)
    else:
        df = pd.read_csv(local_path)

    features = df.drop(columns=['Class']).values
    labels = df['Class'].values
    features = StandardScaler().fit_transform(features)

    edge_index = build_knn_graph(features)
    features = sp.csr_matrix(features)
    num_nodes = features.shape[0]

    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray(), dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']
    return dataset


def load_heart_disease_data():
    dataset_str = 'HeartDisease'
    local_path = os.path.join(DATA_ROOT, 'heart.csv')
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

    if not os.path.exists(local_path):
        df = pd.read_csv(url, names=columns)
        df.to_csv(local_path, index=False)
    else:
        df = pd.read_csv(local_path)

    df.replace('?', np.nan, inplace=True)
    df = df.dropna().astype(float)
    df['target'] = (df['target'] > 0).astype(int)

    features = StandardScaler().fit_transform(df.drop(columns=['target']))
    labels = df['target'].values

    edge_index = build_knn_graph(features)
    features = sp.csr_matrix(features)
    num_nodes = features.shape[0]

    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray(), dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']
    return dataset


def load_nsl_kdd_data():
    dataset_str = 'NSL_KDD'
    local_path = os.path.join(DATA_ROOT, 'KDDTrain+.txt')
    url = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt'

    if not os.path.exists(local_path):
        df = pd.read_csv(url, header=None)
        df.to_csv(local_path, index=False, header=False)
    else:
        df = pd.read_csv(local_path, header=None)

    labels = df.iloc[:, -2]
    features = df.iloc[:, :-2]

    for col in [1, 2, 3]:
        features[col] = LabelEncoder().fit_transform(features[col])

    labels = (labels != 'normal').astype(int)
    features = StandardScaler().fit_transform(features)

    edge_index = build_knn_graph(features)
    features = sp.csr_matrix(features)
    num_nodes = features.shape[0]

    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray(), dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']
    return dataset


def load_german_credit_data():
    dataset_str = 'GermanCredit'
    local_path = os.path.join(DATA_ROOT, 'german.data')
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'

    if not os.path.exists(local_path):
        df = pd.read_csv(url, delimiter=' ', header=None)
        df.to_csv(local_path, index=False, header=False, sep=' ')
    else:
        df = pd.read_csv(local_path, delimiter=' ', header=None)

    labels = (df.iloc[:, -1].values == 2).astype(int)
    df = df.iloc[:, :-1]
    df = pd.get_dummies(df)
    features = StandardScaler().fit_transform(df.values)

    edge_index = build_knn_graph(features)
    features = sp.csr_matrix(features)
    num_nodes = features.shape[0]

    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray(), dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']
    return dataset


def load_traffic_metr_data():
    dataset_str = 'METR_LA'
    local_path = os.path.join(DATA_ROOT, 'metr_dummy.csv')

    # 模拟数据保存（真实 METR-LA 复杂，简化为分类任务）
    if not os.path.exists(local_path):
        dummy = pd.DataFrame(np.random.rand(207, 10))
        dummy.to_csv(local_path, index=False)
    else:
        dummy = pd.read_csv(local_path)

    features = dummy.values
    labels = (features[:, -1] > 0.5).astype(int)

    edge_index = build_knn_graph(features)
    features = sp.csr_matrix(features)
    num_nodes = features.shape[0]

    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray(), dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']
    return dataset


def load_gene_disease_association_prediction_with_gat():
    dataset_str = 'Gene_Disease_Association_Prediction_with_GAT'
    save_dir = '/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian'
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'GDA_associations.csv')

    # ---------- 1. 下载数据 ----------
    if not os.path.exists(csv_path):
        print('未找到 GDA_associations.csv，正在下载 …')
        url = ('https://raw.githubusercontent.com/'
               'RudraxDave/Gene_Disease_Association_Prediction_with_GAT/main/'
               'GDA_associations.csv')
        urlretrieve(url, csv_path)
        print('✅ 数据下载完成')

    # ---------- 2. 读取 + 清洗 ----------
    df = pd.read_csv(csv_path).dropna(subset=['score'])
    gene_cols     = ['gene_dsi', 'gene_dpi', 'gene_pli', 'protein_class']
    disease_cols  = ['disease_type', 'disease_semantic_type']
    disease_class = [
        'C01','C04','C05','C06','C07','C08','C09','C10','C11','C12','C13',
        'C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24',
        'C25','C26','F01','F02','F03'
    ]

    # 2-1 基因节点特征
    gene_df = df[['geneid'] + gene_cols].drop_duplicates()
    gene_df['protein_class'] = pd.factorize(gene_df['protein_class'])[0]
    gene_df[gene_cols] = gene_df[gene_cols].apply(pd.to_numeric, errors='coerce')
    gene_df = gene_df.dropna(subset=gene_cols)

    # 2-2 疾病节点特征
    dis_df = df[['diseaseid', 'disease_class'] + disease_cols].drop_duplicates()
    for c in disease_cols:
        dis_df[c] = pd.factorize(dis_df[c])[0]
    for c in disease_class:
        dis_df[c] = 0
    for i, row in dis_df.iterrows():
        for c in str(row['disease_class']).split(';'):
            if c in disease_class:
                dis_df.at[i, c] = 1
    dis_df = dis_df.drop(columns=['disease_class'])

    # ---------- 3. 建立节点索引 ----------
    genes      = gene_df['geneid'].unique()
    diseases   = dis_df['diseaseid'].unique()
    gene2idx   = {g:i for i, g in enumerate(genes)}
    dis2idx    = {d:i+len(genes) for i, d in enumerate(diseases)}
    num_nodes  = len(genes) + len(diseases)

    # ---------- 4. 生成节点特征矩阵 ----------
    gene_feat = gene_df.set_index('geneid')[gene_cols].loc[genes].values
    dis_feat  = dis_df.set_index('diseaseid')[disease_cols + disease_class].loc[diseases].values
    total_dim = gene_feat.shape[1] + dis_feat.shape[1]
    # 对齐维度
    gene_pad  = np.zeros((len(genes), dis_feat.shape[1]))
    dis_pad   = np.zeros((len(diseases), gene_feat.shape[1]))
    X = np.vstack([np.hstack([gene_feat, gene_pad]),
                   np.hstack([dis_pad,  dis_feat])])
    X = StandardScaler().fit_transform(X)

    # ---------- 5. 正/负边 ----------
    pos_df = df[['geneid', 'diseaseid']].drop_duplicates()
    pos_src = pos_df['geneid'].map(gene2idx).to_numpy()
    pos_dst = pos_df['diseaseid'].map(dis2idx).to_numpy()

    # 负边随机采样（与正边同数量）
    neg_src, neg_dst = [], []
    gene_arr, dis_arr = list(genes), list(diseases)
    pos_set = set(zip(pos_src, pos_dst))
    while len(neg_src) < len(pos_src):
        g = gene2idx[random.choice(gene_arr)]
        d = dis2idx[random.choice(dis_arr)]
        if (g, d) not in pos_set:
            neg_src.append(g)
            neg_dst.append(d)

    # 无向图：边双向添加
    src = np.concatenate([pos_src, neg_src, pos_dst, neg_dst])
    dst = np.concatenate([pos_dst, neg_dst, pos_src, neg_src])
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    # edge_label 仅对“原向”边赋值
    edge_label_index = torch.tensor(
        np.vstack([np.concatenate([pos_src, neg_src]),
                   np.concatenate([pos_dst, neg_dst])]),
        dtype=torch.long
    )
    edge_label = torch.tensor(
        np.concatenate([np.ones(len(pos_src)), np.zeros(len(neg_src))]),
        dtype=torch.float
    )

    # ---------- 6. 打包 NCDataset ----------
    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(X, dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.edge_label_index = edge_label_index   # 👉 训练时用
    dataset.edge_label       = edge_label
    # 若框架仍要求 dataset.label，可随意占位
    dataset.label = edge_label

    # 划分索引（可以用框架自带，也可自行分）
    split = dataset.get_idx_split()     # 若此方法只管节点，可忽略
    dataset.train_idx = torch.arange(edge_label.size(0))   # 让你自己再切分
    dataset.valid_idx = torch.tensor([], dtype=torch.long)
    dataset.test_idx  = torch.tensor([], dtype=torch.long)

    return dataset






def load_ccle_data():


    # 路径设置
    root = "/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian/ccle"
    expr_path = os.path.join(root, "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    info_path = os.path.join(root, "Model.csv")

    # 加载数据
    df_expr = pd.read_csv(expr_path, index_col=0)
    df_info = pd.read_csv(info_path)

    print("表达矩阵 shape:", df_expr.shape)
    print("样本信息 shape:", df_info.shape)
    print("表达矩阵索引（前5行）:", df_expr.index[:5])
    print("样本信息列名：", df_info.columns.tolist())

    # 设置索引名为 ModelID，准备按 index 合并
    df_expr.index.name = "ModelID"

    # merge on ModelID（正确字段）
    df = df_info.merge(df_expr, on="ModelID")

    # 过滤无标签数据
    df = df[~df["OncotreePrimaryDisease"].isna()]
    print("合并后样本数:", len(df))
    print("标签列频数:\n", df["OncotreePrimaryDisease"].value_counts())

    # 提取表达特征列（即 df_expr 的列）
    expr_cols = df_expr.columns
    X = df[expr_cols].fillna(0.0).astype(np.float32).values

    X = StandardScaler().fit_transform(X)

    # 标签编码
    labels, _ = pd.factorize(df["OncotreePrimaryDisease"])

    # 构建 k-NN 图
    adj = kneighbors_graph(X, n_neighbors=5, include_self=False)
    edge_index = torch.tensor(np.vstack(adj.nonzero()), dtype=torch.long)

    # 构造 NCDataset 对象
    dataset = NCDataset("CCLE")
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(X, dtype=torch.float32),
        'num_nodes': X.shape[0]
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    # 默认划分
    split = dataset.get_idx_split()
    dataset.train_idx = split["train"]
    dataset.valid_idx = split["valid"]
    dataset.test_idx  = split["test"]

    return dataset







def load_breast_cancer_data():
    dataset_str = 'BreastCancer'
    save_path = '/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian/breast_cancer.csv'

    # 指定真实数据 URL
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
    column_names = ['Sample_code_number', 'Clump_Thickness', 'Uniformity_Cell_Size',
                    'Uniformity_Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Cell_Size',
                    'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']

    if not os.path.exists(save_path):
        df = pd.read_csv(url, header=None, names=column_names)
        df.to_csv(save_path, index=False)
    else:
        df = pd.read_csv(save_path)

    df = df.drop(columns=['Sample_code_number'])
    df = df.replace('?', np.nan).dropna()
    df = df.astype(float)

    features = df.drop(columns=['Class']).values
    labels = df['Class'].values
    features = StandardScaler().fit_transform(features)

    edge_index = kneighbors_graph(features, n_neighbors=5, mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.vstack(edge_index.nonzero()), dtype=torch.long)

    features = sp.csr_matrix(features)
    num_nodes = features.shape[0]

    dataset = NCDataset(dataset_str)
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': torch.tensor(features.toarray(), dtype=torch.float),
        'num_nodes': num_nodes
    }
    dataset.label = torch.tensor(labels, dtype=torch.long)

    split = dataset.get_idx_split()
    dataset.train_idx = split['train']
    dataset.valid_idx = split['valid']
    dataset.test_idx = split['test']
    return dataset
import os, shutil, pickle
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Coauthor

# from your_lib import NCDataset  # 保持你的 NCDataset 导入

DATA_DIR = "/home/user012/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/data/zijian/coauthor/"  # 注意：这里以 coauthor 为结尾

def _load_splits(npz_path: str, num_nodes: int):
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)

    def _get_first(*cands):
        for k in cands:
            if k in keys:
                return z[k]
        return None

    # 优先掩码
    tm = _get_first("train_mask")
    vm = _get_first("val_mask", "valid_mask")
    sm = _get_first("test_mask")
    if tm is not None and vm is not None and sm is not None:
        tm = torch.as_tensor(tm).view(-1).bool()
        vm = torch.as_tensor(vm).view(-1).bool()
        sm = torch.as_tensor(sm).view(-1).bool()
        return tm, vm, sm, torch.where(tm)[0], torch.where(vm)[0], torch.where(sm)[0]

    # 其次索引
    ti = _get_first("train", "train_idx", "idx_train", "train_indices")
    vi = _get_first("val", "valid", "val_idx", "idx_val", "validation")
    si = _get_first("test", "test_idx", "idx_test")
    if ti is None or vi is None or si is None:
        raise KeyError(f"{npz_path} 中未找到可识别的划分键名。")

    ti = torch.as_tensor(ti, dtype=torch.long).view(-1)
    vi = torch.as_tensor(vi, dtype=torch.long).view(-1)
    si = torch.as_tensor(si, dtype=torch.long).view(-1)

    def to_mask(idx):
        m = torch.zeros(num_nodes, dtype=torch.bool)
        m[idx] = True
        return m

    return to_mask(ti), to_mask(vi), to_mask(si), ti, vi, si


def _safe_load_coauthor(name: str):
    """
    可靠地加载 PyG Coauthor 数据（root=DATA_DIR）。
    如遇到损坏的 raw/processed 缓存，会清理后重试一次。
    """
    transform = T.NormalizeFeatures()
    try:
        return Coauthor(root=DATA_DIR, name=name, transform=transform)
    except (pickle.UnpicklingError, EOFError, ValueError) as e:
        # 清理损坏缓存并重试
        base = os.path.join(DATA_DIR, name)
        for sub in ["raw", "processed"]:
            shutil.rmtree(os.path.join(base, sub), ignore_errors=True)
        return Coauthor(root=DATA_DIR, name=name, transform=transform)


def _build_nc_dataset(name: str, torch_data, split_npz_path: str):
    edge_index = torch_data.edge_index
    node_feat = torch_data.x
    label = torch_data.y
    num_nodes = int(torch_data.num_nodes)

    tr_mask, va_mask, te_mask, tr_idx, va_idx, te_idx = _load_splits(split_npz_path, num_nodes)

    dataset = NCDataset(name)
    dataset.graph = {
        "edge_index": edge_index,
        "node_feat": node_feat,
        "edge_feat": None,
        "num_nodes": num_nodes,
    }
    dataset.label = label
    dataset.train_mask, dataset.val_mask, dataset.test_mask = tr_mask, va_mask, te_mask
    dataset.train_idx,  dataset.val_idx,  dataset.test_idx  = tr_idx,  va_idx,  te_idx
    try:
        dataset.num_classes = int(label.max().item() + 1)
    except Exception:
        pass
    return dataset


def load_coauthor_CS_dataset():
    pyg_ds = _safe_load_coauthor("CS")  # ✅ root=DATA_DIR，由 PyG 自动拼 /CS/...
    data = pyg_ds[0]
    split_path = os.path.join(DATA_DIR, "coauthor-cs_split.npz")  # 你的 split 路径
    return _build_nc_dataset("coauthor-cs", data, split_path)


def load_coauthor_physics_dataset():
    pyg_ds = _safe_load_coauthor("Physics")  # ✅ root=DATA_DIR，由 PyG 自动拼 /Physics/...
    data = pyg_ds[0]
    split_path = os.path.join(DATA_DIR, "coauthor-physics_split.npz")  # 你的 split 路径
    return _build_nc_dataset("coauthor-physics", data, split_path)

if __name__ == '__main__':
    # load_airport()
    # load_wikipedia('squirrel')
    # load_wiki_new('chameleon')
    pass
