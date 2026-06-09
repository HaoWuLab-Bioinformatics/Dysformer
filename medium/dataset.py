
import os, numpy as np, pandas as pd, torch
import scipy.sparse as sp  # ★ 新增
from types import SimpleNamespace  # 若用自定义 NCDataset 可删
from data_utils import normalize_feat, rand_train_test_idx, split_data
import pickle as pkl
from torch_geometric.datasets import Planetoid, Coauthor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
import torch
import torch_geometric.transforms as T
DATAPATH = '../data/'
DATA_ROOT = "../data/zijian"
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

    if  dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(dataname, args.no_feat_norm)
    elif dataname == 'mini':
        dataset = load_mini_imagenet()
    elif dataname == 'Anxiety':
        dataset = load_jiaolv_dataset()
    elif dataname == 'Depression':
        dataset = load_yiyu_dataset()
    elif dataname == 'Anxiety_raw':
        dataset = load_jiaolv_dataset_missing()
    elif dataname == 'Depression_raw':
        dataset = load_yiyu_dataset_missing()
    elif dataname == 'GermanCredit':  # 信用评分
        dataset = load_german_credit_data()
    else:
        raise ValueError('Invalid dataname')
    return dataset
from sklearn.neighbors import NearestNeighbors
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
def load_mini_imagenet():
    dataset = NCDataset('mini_imagenet')
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
def load_jiaolv_dataset():
    dataset_str = 'jiaolv'
    data_path = '../data/zijian/jiaolvyiyu'
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
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

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
    cat_names = cat_encoder.get_feature_names_out(cat_features)

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
    data_path = '../data/zijian/jiaolvyiyu'
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
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

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
    cat_names = cat_encoder.get_feature_names_out(cat_features)

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
    data_path = '../data/zijian/jiaolvyiyu'
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
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

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
    cat_names = cat_encoder.get_feature_names_out(cat_features)

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
    data_path = '../zijian/jiaolvyiyu'
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
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

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
    cat_names = cat_encoder.get_feature_names_out(cat_features)

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
def build_knn_graph(features, k=10):
    adj = kneighbors_graph(features, n_neighbors=k, mode='connectivity', include_self=False)
    adj = adj + adj.T  # 对称化
    adj.setdiag(0)
    edge_index = torch.tensor(np.vstack(adj.nonzero()), dtype=torch.long)
    return edge_index
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
if __name__ == '__main__':
    # load_airport()
    # load_wikipedia('squirrel')
    # load_wiki_new('chameleon')
    pass
