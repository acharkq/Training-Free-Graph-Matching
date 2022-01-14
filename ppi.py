import re
import math
import time
import torch
import random
from collections import Counter
from utils import set_random_seed, evaluate, Performance, cosine_similarity_nbyn, batch_median, l1_similarity_nbyn
from scipy.optimize import linear_sum_assignment
from models import RandDGMC, GraphSAGE, GIN, act_func_dict
import argparse


def s3_evaluation(alignment, src_edges, tgt_edges, print_info=True):
    # from the magna paper
    # alignment: shape = [N, 2]
    # src_edges: shape = [E0, 2]
    # tgt_edges: shape = [E1, 2]
    N = int(torch.max(alignment)) + 1
    
    src_edges_set = {(i,j) for i,j in src_edges.tolist()}
    tgt_edges_set = {(i,j) for i,j in tgt_edges.tolist()}

    src2tgt = torch.zeros(N, dtype=torch.long)
    tgt2src = torch.zeros(N, dtype=torch.long)
    src2tgt[alignment[:,0]] = alignment[:,1]
    tgt2src[alignment[:,1]] = alignment[:,0]
    
    ## EC 
    f_e1 = src2tgt[src_edges].tolist()
    f_e1 = {(i,j) for i, j in f_e1 if (i,j) in tgt_edges_set}
    EC = len(f_e1) / len(src_edges)
    if print_info:
        print('EC %.2f%%' %(EC * 100))
    
    ## ICS
    mapped_tgt_node_set = {j for i,j in alignment.tolist()}
    e_g2_f_v1_set = {(i,j) for i,j in tgt_edges_set if i in mapped_tgt_node_set and j in mapped_tgt_node_set}
    ICS  = len(f_e1) / len(e_g2_f_v1_set)
    if print_info:
        print('ICS %.2f%%' %(ICS * 100))
    
    ## S3
    S3 = len(f_e1) / (len(src_edges_set) + len(e_g2_f_v1_set) - len(f_e1))
    if print_info:
        print('S3 %.2f%%' %(S3 * 100))
    return EC, ICS, S3
        

def read_gw(path, shuffle=False):
    node_regex = re.compile('\|\{(.+)\}\|')
    edge_regex = re.compile('([0-9]+) ([0-9]+) 0 ')
    with open(path) as f:
        for i in range(4):
            f.readline()
        node_num = int(f.readline().strip())
        id2name = {}
        for i in range(node_num):
            line = f.readline()
            node_name = node_regex.match(line).group(1)
            id2name[i] = node_name
        edge_num = int(f.readline().strip())
        edges = []
        for i in range(edge_num):
            line = f.readline()
            result = edge_regex.match(line)
            src_node = int(result.group(1)) - 1
            tgt_node = int(result.group(2)) - 1
            assert src_node in id2name and tgt_node in id2name
            edges.append((src_node, tgt_node))
        edges_inv = [(j,i) for i, j in edges]
        edges.extend(edges_inv)
    if shuffle:
        new_mapping = list(range(len(id2name)))
        random.seed(0)
        random.shuffle(new_mapping)
        id2name = {new_mapping[i]:j for i, j in id2name.items()}
        edges = [(new_mapping[i],new_mapping[j]) for i, j in edges]
    return id2name, edges


def load_synthetic_data(noise_ratio, rewirement=False, rewirement_id=-1):
    folder_path = './data/networks/synthetic_nets_known_node_mapping'
    src_path = '%s/0Krogan_2007_high.gw' % folder_path
    if noise_ratio == 0:
        tgt_path = '%s/0Krogan_2007_high.gw' % folder_path
    else:
        if rewirement:
            tgt_path = '%s/rewired/yeast+%drw_%d.gw' % (folder_path, noise_ratio, rewirement_id)
        else:
            tgt_path = '%s/low_confidence/0Krogan_2007_high+%de.gw' % (folder_path, noise_ratio)
    src_id2name, src_edges = read_gw(src_path)
    tgt_id2name, tgt_edges = read_gw(tgt_path, True)
    return src_id2name, src_edges, tgt_id2name, tgt_edges


@torch.no_grad()
def run_ppi(args, src_id2name, src_edges, tgt_id2name, tgt_edges, print_info):
    device = 'cuda:%d' % args.gpu_id

    src_name2id = {j:i for i,j in src_id2name.items()}
    tgt_name2id = {j:i for i,j in tgt_id2name.items()}

    seeds = []
    for name in src_name2id.keys():
        seeds.append((src_name2id[name], tgt_name2id[name]))
    seeds = torch.tensor(seeds).to(device)

    src_node_num = len(src_id2name)
    tgt_node_num = len(tgt_id2name)
    th_src_edges = torch.tensor(src_edges).T
    th_tgt_edges = torch.tensor(tgt_edges).T
    src_adj = torch.sparse.FloatTensor(th_src_edges, torch.ones(th_src_edges.shape[1]), size=(src_node_num, src_node_num))
    tgt_adj = torch.sparse.FloatTensor(th_tgt_edges, torch.ones(th_tgt_edges.shape[1]), size=(tgt_node_num, tgt_node_num))
    src_degrees = torch.sparse.sum(src_adj, dim=1).to_dense()
    tgt_degrees = torch.sparse.sum(tgt_adj, dim=1).to_dense()
    
    if args.use_one_hot:
        ## use one-hot encoding of node degrees
        def make_degree_feature_mapping(degrees):
            degrees = sorted(Counter(degrees).items(), key=lambda x:x[0])
            degree2id = {degree: i for i, (degree, count) in enumerate(degrees)}
            return degree2id

        _src_degrees = src_degrees.int().tolist()
        _tgt_degrees = tgt_degrees.int().tolist()
        degree2id = make_degree_feature_mapping(_src_degrees + _tgt_degrees)

        src_degree_ids = torch.tensor([degree2id[d] for d in _src_degrees])
        tgt_degree_ids = torch.tensor([degree2id[d] for d in _tgt_degrees])

        src_feats_n = torch.zeros((src_node_num, len(degree2id)))
        tgt_feats_n = torch.zeros((tgt_node_num, len(degree2id)))

        src_feats_n[torch.arange(src_node_num), src_degree_ids] = 1
        tgt_feats_n[torch.arange(tgt_node_num), tgt_degree_ids] = 1
    else:
        f_dim = 512
        use_gaussian_pe = False
        src_feats_n = torch.zeros((src_node_num, f_dim))
        tgt_feats_n = torch.zeros((tgt_node_num, f_dim))
        if use_gaussian_pe:
            ## use Gaussian Position Embedding
            #### Normalize degrees
            w = torch.randn((f_dim // 2))# * std
            for i in range(f_dim):
                if i % 2 == 0:
                    src_feats_n[:, i] = torch.sin(src_degrees * w[i // 2])
                    tgt_feats_n[:, i] = torch.sin(tgt_degrees * w[i // 2])
                else:
                    src_feats_n[:, i] = torch.cos(src_degrees * w[i // 2])
                    tgt_feats_n[:, i] = torch.cos(tgt_degrees * w[i // 2])
        else:
            ## use position embedding from transformer    
            kkkk = 10000
            for i in range(f_dim):
                if i % 2 == 0:
                    src_feats_n[:, i] = torch.sin(src_degrees / (kkkk ** (i/ f_dim)))
                    tgt_feats_n[:, i] = torch.sin(tgt_degrees / (kkkk ** (i/ f_dim)))
                else:
                    src_feats_n[:, i] = torch.cos(src_degrees / (kkkk ** ((i-1)/ f_dim)))
                    tgt_feats_n[:, i] = torch.cos(tgt_degrees / (kkkk ** ((i-1)/ f_dim)))

    f_dim = src_feats_n.shape[1]
    src_feats_n = src_feats_n.to(device)
    tgt_feats_n = tgt_feats_n.to(device)
    th_src_edges = th_src_edges.to(device)
    th_tgt_edges = th_tgt_edges.to(device)

    if args.node_matching:
        S = cosine_similarity_nbyn(src_feats_n, tgt_feats_n)
        row_ind, col_ind = linear_sum_assignment(-S.cpu().numpy())
        row_ind, col_ind = torch.from_numpy(row_ind), torch.from_numpy(col_ind)
        S[row_ind, col_ind] += 1000
        alignment = torch.stack((row_ind, col_ind), dim=-1)
        if print_info:
            print('Evaluate with hungarian')
        pfm = evaluate(S, seeds, torch.arange(src_node_num, device=device), torch.arange(tgt_node_num, device=device), print_info=print_info)
        EC, ICS, S3 = s3_evaluation(alignment, th_src_edges.T, th_tgt_edges.T, print_info=print_info)
        return pfm , EC, ICS, S3

    if args.gnn_type == 'gin':
        gnn = GIN(f_dim, args.h_dim, args.num_layer, use_node_feature=False)
    elif args.gnn_type == 'sage':
        gnn = GraphSAGE(f_dim, args.h_dim, args.num_layer, args.act_func, False, args.weight_free, bias=True)
    gnn.eval()
    gnn.to(device)
    src_g_feats = gnn(src_feats_n, th_src_edges)
    tgt_g_feats = gnn(tgt_feats_n, th_tgt_edges)
    S = cosine_similarity_nbyn(src_g_feats, tgt_g_feats)

    if print_info:
        print('Evaluate without refinement')
    pfm = evaluate(S, seeds, torch.arange(src_node_num, device=device), torch.arange(tgt_node_num, device=device), print_info=print_info)

    if args.use_dgmc:
        if args.gnn_type == 'gin':
            rnd_gnn = GIN(args.rnd_dim, args.rnd_dim, args.dgmc_layer, use_node_feature=False)
        elif args.gnn_type == 'sage':
            rnd_gnn = GraphSAGE(args.rnd_dim, args.rnd_dim, args.dgmc_layer, args.act_func, False, args.weight_free, bias=False)
        dgmc = RandDGMC(rnd_gnn, 100, args.num_steps, weight_free=False)
        dgmc.eval()
        dgmc.to(device)
        S0 = dgmc(S, th_src_edges, th_tgt_edges)
        S1 = dgmc(S.T, th_tgt_edges, th_src_edges).transpose(0, 1)
        S = (S0 + S1).to_dense()
        if print_info:
            print('Evaluate with DGMC')
        pfm = evaluate(S, seeds, torch.arange(src_node_num, device=device), torch.arange(tgt_node_num, device=device), print_info=print_info)


    # if args.use_hungarian:
    row_ind, col_ind = linear_sum_assignment(- S.cpu().numpy())
    row_ind, col_ind = torch.from_numpy(row_ind), torch.from_numpy(col_ind)
    S[row_ind, col_ind] += 1000
    alignment = torch.stack((row_ind, col_ind), dim=-1)
    if print_info:
        print('Evaluate with hungarian')
    pfm = evaluate(S, seeds, torch.arange(src_node_num, device=device), torch.arange(tgt_node_num, device=device), print_info=print_info)
    # else:
    #     row_ind = torch.arange(S.shape[0])
    #     col_ind = torch.argmax(S, dim=-1).cpu()
    #     alignment = torch.stack((row_ind, col_ind), dim=-1)

    EC, ICS, S3 = s3_evaluation(alignment, th_src_edges.T, th_tgt_edges.T, print_info=print_info)
    return pfm , EC, ICS, S3
    

def average_pfm(list_of_pfm):
    N = len(list_of_pfm)
    mr0 = 0
    mrr0 = 0
    top_x0 = [0 for _ in range(len(list_of_pfm[0].top_x0))]
    mr1 = 0
    mrr1 = 0
    top_x1 = [0 for _ in range(len(list_of_pfm[0].top_x1))]

    for pfm in list_of_pfm:
        mr0 += pfm.mr0
        mrr0 += pfm.mrr0
        mr1 += pfm.mr1
        mrr1 += pfm.mrr1
        for i in range(len(pfm.top_x0)):
            top_x0[i] += pfm.top_x0[i]
            top_x1[i] += pfm.top_x1[i]
    mr0 /= N
    mrr0 /= N
    mr1 /= N
    mrr1 /= N
    for i in range(len(list_of_pfm[0].top_x0)):
        top_x0[i] /= N
        top_x1[i] /= N
    pfm = Performance(top_x0, mr0, mrr0, top_x1, mr1, mrr1)
    return pfm


def run_for_rewirement(args, noise_ratio):
    # hit1 = 0
    EC = 0
    ICS = 0
    S3 = 0
    pfm_list = []
    
    synthetic_num = 10
    for i in range(synthetic_num):
        src_id2name, src_edges, tgt_id2name, tgt_edges = load_synthetic_data(noise_ratio, True, i)
        pfm, _EC, _ICS, _S3 = run_ppi(args, src_id2name, src_edges, tgt_id2name, tgt_edges, print_info=False)
        pfm_list.append(pfm)
        EC += _EC
        ICS += _ICS
        S3 += _S3
    EC /= synthetic_num
    ICS /= synthetic_num
    S3 /= synthetic_num
    
    pfm = average_pfm(pfm_list)
    pfm.print_performance()
    print('EC', EC * 100)
    print('ICS', ICS * 100)
    print('S3', S3 * 100)


def run_for_noise(args, noise_ratio, timer=False):
    EC = 0
    ICS = 0
    S3 = 0
    pfm_list = []
    
    synthetic_num = 10
    for _ in range(synthetic_num):
        src_id2name, src_edges, tgt_id2name, tgt_edges = load_synthetic_data(noise_ratio, False, -1)
        start = time.time()
        pfm, _EC, _ICS, _S3 = run_ppi(args, src_id2name, src_edges, tgt_id2name, tgt_edges, print_info=False)
        end = time.time()
        if timer:
            print('Cost', end - start, 'seconds.')
        pfm_list.append(pfm)
        EC += _EC
        ICS += _ICS
        S3 += _S3
    EC /= synthetic_num
    ICS /= synthetic_num
    S3 /= synthetic_num
    
    pfm = average_pfm(pfm_list)
    pfm.print_performance()
    print('EC', EC * 100)
    print('ICS', ICS * 100)
    print('S3', S3 * 100)



if __name__ == '__main__':
    # run_for_real_data('meso-syne')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='extra_edge')
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--rnd_dim', type=int, default=512)
    parser.add_argument('--num_layer', type=int, default=10)
    parser.add_argument('--dgmc_layer', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--gnn_type', type=str, default='sage')
    parser.add_argument('--act_func', type=str, default='identity')
    parser.add_argument('--weight_free', action='store_true')
    parser.add_argument('--node_matching', action='store_true')
    parser.add_argument('--use_one_hot', action='store_true')
    parser.add_argument('--use_dgmc', action='store_true')
    args = parser.parse_args()
    args.act_func = act_func_dict[args.act_func]
    

    print('Dataset', args.dataset)
    # run_for_noise(args, 25)
    for noise_ratio in [5, 10, 15, 20, 25]:
        # run_for_noise(args, r)
        start = time.time()
        print('-------------------------')
        print('Noise ratio', noise_ratio)
        if args.dataset == 'extra_edge':
            run_for_noise(args, noise_ratio)
        elif args.dataset == 'rewirement':
            run_for_rewirement(args, noise_ratio)
        else:
            raise NotImplementedError
        print('Cost seconds', time.time() - start)
        print('-------------------------')
    
