from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T

from data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, dataset_drive_url, class_rand_splits

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, TUDataset
from torch_geometric.transforms import NormalizeFeatures
from os import path

from torch_sparse import SparseTensor
from google_drive_downloader import GoogleDriveDownloader as gdd

import networkx as nx
import scipy.sparse as sp
import torch

# 保存原始的 torch.load
_orig_torch_load = torch.load

def _torch_load_unsafe(*args, **kwargs):
    # 如果外部没传 weights_only，就默认设为 False
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)

# 全局替换 torch.load
torch.load = _torch_load_unsafe

from ogb.nodeproppred import NodePropPredDataset, PygNodePropPredDataset
import os

from torch_geometric.utils import subgraph, k_hop_subgraph, to_undirected
import pickle as pkl

class NCDataset(object):
    def __init__(self, name):
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

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """
        
        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):  
        return '{}({})'.format(self.__class__.__name__, len(self))

import gdown
def load_dataset(data_dir, dataname, sub_dataname=''):
    """ Loader for NCDataset 
        Returns NCDataset 
    """
    # print(dataname)
    if  dataname == 'pokec':
        dataset = load_pokec_mat(data_dir)

    elif dataname in ('ogbn-arxiv', 'ogbn-products'):
        dataset = load_ogb_dataset(data_dir, dataname)

    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_ogb_dataset(data_dir, name):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name, root=f'{data_dir}/ogb')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    def ogb_idx_to_tensor():
        split_idx = ogb_dataset.get_idx_split()
        tensor_split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
        return tensor_split_idx
    dataset.load_fixed_splits = ogb_idx_to_tensor  # ogb_dataset.get_idx_split
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    return dataset


def load_pokec_mat(data_dir):
    """ requires pokec.mat """
    if not path.exists(f'{data_dir}/pokec/pokec.mat'):
        gdd.download_file_from_google_drive(
            file_id= dataset_drive_url['pokec'], \
            dest_path=f'{data_dir}/pokec/pokec.mat', showsize=True)

    try:
        fulldata = scipy.io.loadmat(f'{data_dir}/pokec/pokec.mat')
        edge_index = fulldata['edge_index']
        node_feat = fulldata['node_feat']
        label = fulldata['label']
    except:
        edge_index = np.load(f'{data_dir}/pokec/edge_index.npy')
        node_feat = np.load(f'{data_dir}/pokec/node_feat.npy')
        label = np.load(f'{data_dir}/pokec/label.npy')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat).float()
    num_nodes = int(node_feat.shape[0])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = torch.tensor(label).flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)

    def load_fixed_splits(train_prop=0.5, val_prop=0.25):
        dir = f'{data_dir}pokec/split_0.5_0.25'
        tensor_split_idx = {}
        if os.path.exists(dir):
            tensor_split_idx['train'] = torch.as_tensor(np.loadtxt(dir + '/pokec_train.txt'), dtype=torch.long)
            tensor_split_idx['valid'] = torch.as_tensor(np.loadtxt(dir + '/pokec_valid.txt'), dtype=torch.long)
            tensor_split_idx['test'] = torch.as_tensor(np.loadtxt(dir + '/pokec_test.txt'), dtype=torch.long)
        else:
            os.makedirs(dir)
            tensor_split_idx['train'], tensor_split_idx['valid'], tensor_split_idx['test'] \
                = rand_train_test_idx(dataset.label, train_prop=train_prop, valid_prop=val_prop)
            np.savetxt(dir + '/pokec_train.txt', tensor_split_idx['train'], fmt='%d')
            np.savetxt(dir + '/pokec_valid.txt', tensor_split_idx['valid'], fmt='%d')
            np.savetxt(dir + '/pokec_test.txt', tensor_split_idx['test'], fmt='%d')
        return tensor_split_idx

    dataset.load_fixed_splits = load_fixed_splits
    return dataset

def load_pokec_mat(data_dir):
    """ requires pokec.mat """
    if not path.exists(f'{data_dir}/pokec/pokec.mat'):
        drive_id = '1575QYJwJlj7AWuOKMlwVmMz8FcslUncu'
        gdown.download(id=drive_id, output="data/pokec/")
        #import sys; sys.exit()
        #gdd.download_file_from_google_drive(
        #    file_id= drive_id, \
        #    dest_path=f'{data_dir}/pokec/pokec.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{data_dir}/pokec/pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)
    return dataset

