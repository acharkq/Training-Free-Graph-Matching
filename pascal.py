'''
part of the code is borrowed from https://github.com/rusty1s/deep-graph-matching-consensus
'''
import os
import random
import argparse
import itertools
import time
import torch
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
from models import SplineCNN, GraphSAGE, RandDGMC, split_sim_batches, act_func_dict
from utils import timeit, cosine_similarity_nbyn, set_random_seed
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F


class KNN(object):
    def __init__(self, k):
        self.k = k
    
    def knn_feature(self, b_s, g_pos_t, y_t):
        '''
        b_s: shape = [B_s * N_s, B_t, N_t]
        g_pos_t: shape = [B_t, 2]
        y_t: shape = [B_t * N_t]
        Return 
            predictions: shape = [test_sum_node, train_g_num] contain the id from the train_graph
        '''
        N = int(torch.max(y_t)) + 1
        device = b_s.device
        ## inter graph node alignment
        b_s, n_ids = torch.max(b_s, dim=-1) # shape = [B_s * N_s, B_t], [B_s * N_s, B_t]
        ## intra graph knn
        _, knn_g_ids = torch.topk(b_s, self.k, dim=-1) # shape = [B_s * N_s, k]
        knn_n_ids = torch.gather(n_ids, 1, knn_g_ids) # shape = [B_s * N_s, k]
        g_start_pos_t = torch.tensor([start for start, end in g_pos_t], device=device) # shape = [B_t]
        knn_n_ids = knn_n_ids + g_start_pos_t[knn_g_ids] # shape = [B_s * N_s, k]
        knn_labels = y_t[knn_n_ids] # shape = [B_s * N_s, k]
        knn_feature = torch.zeros((knn_labels.shape[0], N), device=device)
        knn_feature = torch.scatter_add(knn_feature, 1, knn_labels, torch.ones_like(knn_labels, dtype=torch.float))
        return knn_feature

    def knn_classification(self, b_s, g_pos_t, y_t):
        '''
        b_s: shape = [B_s * N_s, B_t, N_t]
        g_pos_t: shape = [B_t, 2]
        y_t: shape = [B_t * N_t]
        Return 
            predictions: shape = [test_sum_node, train_g_num] contain the id from the train_graph
        '''
        device = b_s.device
        ## inter graph node alignment
        b_s, n_ids = torch.max(b_s, dim=-1) # shape = [B_s * N_s, B_t], [B_s * N_s, B_t]
        ## intra graph knn
        _, knn_g_ids = torch.topk(b_s, self.k, dim=-1) # shape = [B_s * N_s, k]
        knn_n_ids = torch.gather(n_ids, 1, knn_g_ids) # shape = [B_s * N_s, k]
        g_start_pos_t = torch.tensor([start for start, end in g_pos_t], device=device) # shape = [B_t]
        knn_n_ids = knn_n_ids + g_start_pos_t[knn_g_ids] # shape = [B_s * N_s, k]
        knn_labels = y_t[knn_n_ids] # shape = [B_s * N_s, k]
        predictions, _ = torch.mode(knn_labels, dim=-1) # shape = [B_s * N_s]
        return predictions


class Collater(object):
    def collate(self, batch):
        x_list, y_list = list(zip(*batch))
        return torch.tensor(x_list), torch.cat(y_list,dim=0)
    
    def __call__(self, batch):
        return self.collate(batch)


class ValidPairDataset(torch.utils.data.Dataset):
    r"""Combines two datasets, a source dataset and a target dataset, by
    building valid pairs between separate dataset examples.
    A pair is valid if each node class in the source graph also exists in the
    target graph.

    Args:
        dataset_s (torch.utils.data.Dataset): The source dataset.
        dataset_t (torch.utils.data.Dataset): The target dataset.
        sample (bool, optional): If set to :obj:`True`, will sample exactly
            one target example for every source example instead of holding the
            product of all source and target examples. (default: :obj:`False`)
    """
    def __init__(self, dataset_s, dataset_t, sample=False):
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t
        self.sample = sample
        self.pairs, self.cumdeg = self.__compute_pairs__()

    def __compute_pairs__(self):
        num_classes = 0
        for data in itertools.chain(self.dataset_s, self.dataset_t):
            num_classes = max(num_classes, data.y.max().item() + 1)

        y_s = torch.zeros((len(self.dataset_s), num_classes), dtype=torch.bool)
        y_t = torch.zeros((len(self.dataset_t), num_classes), dtype=torch.bool)

        for i, data in enumerate(self.dataset_s):
            y_s[i, data.y] = 1
        for i, data in enumerate(self.dataset_t):
            y_t[i, data.y] = 1

        y_s = y_s.view(len(self.dataset_s), 1, num_classes)
        y_t = y_t.view(1, len(self.dataset_t), num_classes)

        # tt = (y_s * y_t).sum(dim=-1) # shape = [len(s), len(t)], nonzeros are the number of shared classes
        # tt == y_s.sum(dim=-1) # shape = [len(s), len(t)], nonzeros are graph pairs that gt is inclusive of gs
        pairs = ((y_s * y_t).sum(dim=-1) == y_s.sum(dim=-1)).nonzero(as_tuple=False)
        cumdeg = pairs[:, 0].bincount().cumsum(dim=0)
        '''
        pairs: validate source graph and target graph pairs that the node set in target graph is inclusive of that in source graph
        cumdeg: the pos of source graph's targets in the pairs
        '''
        return pairs.tolist(), [0] + cumdeg.tolist()

    def __len__(self):
        return len(self.dataset_s) if self.sample else len(self.pairs)

    def __getitem__(self, idx):
        
        data_s = self.dataset_s[idx]
        i = random.randint(self.cumdeg[idx], self.cumdeg[idx + 1] - 1)
        
        data_t_id = self.pairs[i][1]
        data_t = self.dataset_t[data_t_id]
        '''
        data_t.y shape = [t_num_nodes]
        data_s.y shape = [s_num_nodes]
        '''
        y = data_s.y.new_full((data_t.y.max().item() + 1, ), -1)
        y[data_t.y] = torch.arange(data_t.num_nodes) # shape = [t_max_class], class2t_id
        y = y[data_s.y] # data_s.y s_id2class, y s_id2t_id shape = [s_num_nodes]
        return data_t_id, y


def pack_graphs(graphs, device):
    node2graph_ids = []
    batch_feat_nodes = []
    batch_feat_edges = []
    batch_edges = []
    batch_y = []
    batch_g_pos = []
    
    start = 0 
    i = 0
    for graph in graphs:
        node_num = graph.x.shape[0]
        node2graph_ids.append(torch.ones(node_num, dtype=torch.long) * i)
        batch_feat_nodes.append(graph.x)
        batch_feat_edges.append(graph.edge_attr)
        batch_edges.append(graph.edge_index + start)
        batch_y.append(graph.y)
        batch_g_pos.append((start, start+node_num))
        
        start += node_num
        i += 1

    node2graph_ids = torch.cat(node2graph_ids, dim=0)
    batch_feat_nodes = torch.cat(batch_feat_nodes, dim=0)
    batch_feat_edges = torch.cat(batch_feat_edges, dim=0)
    batch_edges = torch.cat(batch_edges, dim=1)
    batch_y = torch.cat(batch_y, dim=0)

    ## move device
    node2graph_ids = node2graph_ids.to(device)
    batch_feat_nodes = batch_feat_nodes.to(device)
    batch_feat_edges = batch_feat_edges.to(device)
    batch_edges = batch_edges.to(device)
    batch_y = batch_y.to(device)
    return node2graph_ids, batch_feat_nodes, batch_feat_edges, batch_edges, batch_y, batch_g_pos


def generate_y(y_col, device):
    y_row = torch.arange(y_col.size(0), device=device)
    return torch.stack([y_row, y_col.to(device)], dim=0)


def acc(S, y, reduction='mean'):
    assert reduction in ['mean', 'sum']
    if not S.is_sparse:
        pred = S[y[0]].argmax(dim=-1)
    else:
        assert S.__idx__ is not None and S.__val__ is not None
        pred = S.__idx__[y[0], S.__val__[y[0]].argmax(dim=-1)]

    correct = (pred == y[1]).sum().item()
    return correct / y.size(1) if reduction == 'mean' else correct


def extract_graphs(g_ids, g_pos, feat_nodes, graphs, device):
    node2graph_ids = []
    extracted_feat_nodes = []
    extracted_feat_edges = []
    extracted_edges = []
    extracted_y = []
    extracted_g_pos = []

    start = 0
    for i, g_id in enumerate(g_ids):
        pos = g_pos[g_id]
        node_num = pos[1] - pos[0]
        graph = graphs[g_id]
        assert node_num == graph.x.shape[0]
        node2graph_ids.append(torch.ones(node_num, dtype=torch.long) * i)
        extracted_feat_nodes.append(feat_nodes[pos[0]: pos[1]])
        extracted_feat_edges.append(graph.edge_attr)
        extracted_edges.append(graph.edge_index + start)
        extracted_y.append(graph.y)
        extracted_g_pos.append((start, start+node_num))
        start += node_num

    node2graph_ids = torch.cat(node2graph_ids, dim=0)
    extracted_feat_nodes = torch.cat(extracted_feat_nodes, dim=0)
    extracted_feat_edges = torch.cat(extracted_feat_edges, dim=0)
    extracted_edges = torch.cat(extracted_edges, dim=1)
    extracted_y = torch.cat(extracted_y, dim=0)

    ## move device
    node2graph_ids = node2graph_ids.to(device)
    extracted_feat_nodes = extracted_feat_nodes.to(device)
    extracted_feat_edges = extracted_feat_edges.to(device)
    extracted_edges = extracted_edges.to(device)
    extracted_y = extracted_y.to(device)
    return node2graph_ids, extracted_feat_nodes, extracted_feat_edges, extracted_edges, extracted_y, extracted_g_pos


@torch.no_grad()
def run_pascal(args, psi_1, dgmc, test_graphs):
    test_n2g_ids, test_feat_nodes, test_feat_edges, test_edges, test_y, test_g_pos = pack_graphs(test_graphs, args.device)
    sr_test_embed = psi_1(test_feat_nodes, test_edges, test_feat_edges)

    sample_dataset = ValidPairDataset(test_graphs, test_graphs, sample=True)
    test_loader = DataLoader(sample_dataset, len(sample_dataset), shuffle=False, collate_fn=Collater())

    correct = 0
    num_examples = 0
    while (num_examples < args.test_samples):
        '''
        the batch size is set to be the size of the dataset, so there is only one batch
        '''
        i = -1
        tg_g_ids = None
        for i, data in enumerate(test_loader):
            tg_g_ids, y = data  # shape = [batch_size], shape = [sr_test_sum_nodes]
        assert i == 0
        tg_g_ids = tg_g_ids.tolist()
        
        tg_test_n2g_ids, tg_test_embed, tg_test_feat_edges, tg_test_edges, _, _ = extract_graphs(tg_g_ids, test_g_pos, sr_test_embed, test_graphs, args.device)

        if args.use_dgmc:
            S = dgmc.batch_forward(sr_test_embed, test_edges, test_feat_edges, test_n2g_ids, tg_test_embed, tg_test_edges, tg_test_feat_edges,tg_test_n2g_ids)
        else:
            h_s, s_mask = to_dense_batch(sr_test_embed, test_n2g_ids, fill_value=0) # shape = [B, N_s, C_out]; shape = [B, N_s]
            h_t, _ = to_dense_batch(tg_test_embed, tg_test_n2g_ids, fill_value=0) # shape = [B, N_t, C_out]; shape = [B, N_t]
            S_hat = h_s @ h_t.transpose(-1, -2)  # [B, N_s, N_t]
            S = S_hat[s_mask]

        y = generate_y(y, args.device)
        correct += acc(S, y, reduction='sum')   
        num_examples += y.size(1)
        if num_examples >= args.test_samples:
            return correct / num_examples


@torch.no_grad()
def run_pascal_with_knn(args, psi_1, dgmc, train_graphs, test_graphs):
    _, train_feat_nodes, train_feat_edges, train_edges, train_y, train_g_pos = pack_graphs(train_graphs, args.device)
    test_n2g_ids, test_feat_nodes, test_feat_edges, test_edges, test_y, test_g_pos = pack_graphs(test_graphs, args.device)
    train_embed = psi_1(train_feat_nodes, train_edges, train_feat_edges) # shape = [train_sum_nodes, D]
    sr_test_embed = psi_1(test_feat_nodes, test_edges, test_feat_edges) # shape = [test_sum_nodes, D]
    
    '''
    sample_dataset[i] returns:
    (1) data_t_id: a random valid target graph of graph i
    (2) y: a mapping from s_id to t_id
    '''
    sample_dataset = ValidPairDataset(test_graphs, test_graphs, sample=True)
    test_loader = DataLoader(sample_dataset, len(sample_dataset), shuffle=False, collate_fn=Collater())

    knn = KNN(10)
    
    correct = 0
    num_examples = 0
    while (num_examples < args.test_samples):
        '''
        the batch size is set to be the size of the dataset, so there is only one batch
        '''
        i = -1
        tg_g_ids = None
        for i, data in enumerate(test_loader):
            tg_g_ids, y = data  # shape = [batch_size], shape = [sr_test_sum_nodes]
        assert i == 0
        tg_g_ids = tg_g_ids.tolist()

        tg_test_n2g_ids, tg_test_embed, tg_test_feat_edges, tg_test_edges, _, tg_test_g_pos = extract_graphs(tg_g_ids, test_g_pos, sr_test_embed, test_graphs, args.device)

        if args.use_dgmc:
            sr_b_s = dgmc.pairwise_forward(sr_test_embed, test_edges, test_feat_edges, test_g_pos, train_embed, train_edges, train_feat_edges, train_g_pos) # shape = [sr_test_sum_nodes, B_t, N_t]
            tg_b_s = dgmc.pairwise_forward(tg_test_embed, tg_test_edges, tg_test_feat_edges, tg_test_g_pos, train_embed, train_edges, train_feat_edges, train_g_pos) # shape = [tg_test_sum_nodes, B_t, N_t]
        else:
            sr_s = cosine_similarity_nbyn(sr_test_embed, train_embed) # shape = [sr_test_sum_nodes, train_sum_nodes] = [B0 * N_s, B_t * N_t]
            tg_s = cosine_similarity_nbyn(tg_test_embed, train_embed) # shape = [tg_test_sum_nodes, train_sum_nodes]
            sr_b_s, _ = split_sim_batches(sr_s, train_g_pos)
            tg_b_s, _ = split_sim_batches(tg_s, train_g_pos)
        
        sr_knn_feature = knn.knn_feature(sr_b_s, train_g_pos, train_y) # shape = [sr_test_sum_nodes]
        tg_knn_feature = knn.knn_feature(tg_b_s, train_g_pos, train_y) # shape = [tg_test_sum_nodes]
        
        if args.use_dgmc:
            S = dgmc.batch_forward(sr_knn_feature, test_edges, test_feat_edges, test_n2g_ids, tg_knn_feature, tg_test_edges, tg_test_feat_edges,tg_test_n2g_ids) # shape = [B, N_s, N_t]
        else:
            sr_knn_feature = F.normalize(sr_knn_feature, dim=-1, p=2)
            tg_knn_feature = F.normalize(tg_knn_feature, dim=-1, p=2)
            sr_knn_feature, sr_mask = to_dense_batch(sr_knn_feature, test_n2g_ids) # shape = [B, N_s, N], [B, N_s]
            tg_knn_feature, tg_mask = to_dense_batch(tg_knn_feature, tg_test_n2g_ids) # shape = [B, N_t, N], [B, N_t]
            S = sr_knn_feature @ tg_knn_feature.transpose(-1, -2) # shape = [B, N_s, N_t]
            S = S[sr_mask] # shape = [B * N_s, N_t]

        y = generate_y(y, args.device)
        correct += acc(S, y, reduction='sum')
        num_examples += y.shape[1]
        if num_examples >= args.test_samples:
            return correct / num_examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--rnd_dim', type=int, default=128)
    parser.add_argument('--num_layer', type=int, default=5)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--test_samples', type=int, default=1000)
    parser.add_argument('--dgmc_layer', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--act_func', type=str, default='identity')
    parser.add_argument('--use_splinecnn', action='store_true', help='Use SplineCNN as the backbone GNN? The default backbone is GraphSAGE.')
    parser.add_argument('--use_knn', action='store_true', help='Use kNN search to utilize annotation without training?')
    parser.add_argument('--use_dgmc', action='store_true')
    args = parser.parse_args()
    args.device = 'cuda:%d' % args.gpu_id
    args.act_func = act_func_dict[args.act_func]

    pre_filter = lambda data: data.pos.size(0) > 0  # noqa
    transform = T.Compose([
        T.Delaunay(),
        T.FaceToEdge(),
        T.Cartesian(),
    ])

    '''
    Load dataset
    '''
    train_datasets = []
    test_datasets = []
    path = os.path.join('./', 'data', 'PascalVOC')
    for category in PascalVOC.categories:
        dataset = PascalVOC(path, category, train=True, transform=transform,
                            pre_filter=pre_filter)
        dataset.data['x'] = dataset.data['x'][:, :512]
        train_datasets.append(dataset)
        dataset = PascalVOC(path, category, train=False, transform=transform,
                            pre_filter=pre_filter)
        dataset.data['x'] = dataset.data['x'][:, :512]
        test_datasets.append(dataset)
    
    args.in_dim = train_datasets[0].num_node_features
    args.edge_dim = train_datasets[0].num_edge_features


    '''
    Intialize model
    '''
    if args.use_splinecnn:
        psi_1 = SplineCNN(args.in_dim, args.h_dim, args.edge_dim, args.act_func, args.num_layer)
        psi_2 = SplineCNN(args.rnd_dim, args.rnd_dim, args.edge_dim, args.act_func, args.dgmc_layer)
    else:
        psi_1 = GraphSAGE(args.in_dim, args.h_dim, args.num_layer, args.act_func, True, weight_free=True, bias=False)
        psi_2 = GraphSAGE(args.rnd_dim, args.rnd_dim, args.dgmc_layer, args.act_func, True, weight_free=True, bias=False)
    dgmc = RandDGMC(psi_2, -1, args.num_steps, False)
    
    psi_1.to(args.device)    
    psi_1.eval()
    dgmc.to(args.device)
    dgmc.eval()

    start_time = time.time()
    if args.use_knn:
        accs = [100 * run_pascal_with_knn(args, psi_1, dgmc, train_set, test_set) for train_set, test_set in zip(train_datasets, test_datasets)]
    else:
        accs = [100 * run_pascal(args, psi_1, dgmc, test_set) for test_set in test_datasets]

    '''
    print performance
    '''
    accs += [sum(accs) / len(accs)]
    print(' '.join([c[:5].ljust(5) for c in PascalVOC.categories] + ['mean']))
    print(' '.join([f'{acc:.2f}'.ljust(5) for acc in accs]))
    print('Inference totally cost %.2f second.' % (time.time() - start_time))
 