import torch
import torch.nn.functional as F
import numpy as np
import argparse
from models import GraphSAGE, RandDGMC, act_func_dict
import time
from pathlib import Path
import string
from utils import Performance
from tqdm import tqdm


def sp_indexing(sp_tensor, indexing):
    '''
    sp_tensor: shape = [N, M]
    indexing: shape = [C]
    '''
    sp_tensor = sp_tensor.coalesce()
    N, M = sp_tensor.shape
    indices = sp_tensor._indices() # shape = [2, E]
    x, y = indices # shape = [E]
    select = torch.from_numpy(np.isin(x.cpu(), indexing.cpu()))
    x = x[select]
    x_mapping = torch.zeros(N, dtype=torch.long)
    x_mapping[indexing] = torch.arange(len(indexing), dtype=torch.long)
    x = x_mapping[x].to(indexing.device)
    y = y[select]
    indices = torch.stack((x, y), dim=0)
    values = sp_tensor._values()[select] # shape = [E]
    sp_tensor = torch.sparse.FloatTensor(indices, values, size=(len(indexing), M))
    return sp_tensor


@torch.no_grad()
def sparse_similarity(feats0, feats1, k=100):
    N, M = feats0.shape[0], feats1.shape[0]
    batch_size = 1024
    value_list = []
    indice_list = []
    for i in range(0, N, batch_size):
        values, indices = torch.topk(feats0[i:i+batch_size] @ feats1.T, k=k, dim=-1) # shape = [B, 100]
        value_list.append(values)
        indice_list.append(indices)
    knn_values = torch.cat(value_list, dim=0) # shape = [N, 100]
    knn_idx = torch.cat(indice_list, dim=0) # shape = [N, 100]
    sp_S = knn_out2sparse(knn_values, knn_idx, N, M)
    return knn_values, knn_idx, sp_S


def knn_out2sparse(knn_values, knn_indices, N, M):
    device = knn_values.device
    xs = torch.arange(N, device=device).reshape(-1, 1).expand(N, knn_values.shape[1])
    edges = torch.stack((xs, knn_indices), dim=-1)
    edges = edges.reshape(-1, 2).T
    sp_S = torch.sparse.FloatTensor(edges, knn_values.flatten(), size=(N, M))
    return sp_S



def evaluate_sparse(sp_S, seeds, cands0, cands1, print_info=True):
    sp_S = sp_S.coalesce()
    sp_S0 = sp_indexing(sp_S, seeds[:, 0])
    sp_S1 = sp_indexing(sp_S.transpose(0, 1), seeds[:, 1])
    top_x0, mr0, mrr0= _evaluate_sparse(sp_S0, seeds[:, 1], cands1)
    top_x1, mr1, mrr1= _evaluate_sparse(sp_S1, seeds[:, 0], cands0)
    pfm = Performance(top_x0, mr0, mrr0, top_x1, mr1, mrr1)
    if print_info:
        pfm.print_performance()
    return pfm


@torch.no_grad()
def _evaluate_sparse(sp_S, seeds1, cands1, top_k=(1, 10)):
    def _get_hits(S, gt, top_k):
        '''
        :param S: [seed num0, node num1]
        :param gt: [seed num0]
        '''
        sorted_indice = torch.argsort(S, dim=1, descending=True)  # shape = [seed_num0, node num1]
        is_gt = (sorted_indice == gt.reshape(-1, 1)).int()  # shape = [seed_num0, node_num1]
        gt_rank = torch.argmax(is_gt, dim=1) + 1  # shape = [seed_num0]
        top_x = [0.0 for _ in range(len(top_k))]
        for i, k in enumerate(top_k):
            top_x[i] = float(torch.sum((gt_rank <= k).float())) * 100
        mr = float(torch.sum(gt_rank.float()))
        mrr = float(torch.sum(1 / gt_rank.float()))
        return top_x, mr, mrr
    
    def sparse_evaluate(sp_S, gt_seed, cands):
        N, M = sp_S.shape
        indices = sp_S._indices()
        values = sp_S._values()
        ## batched_evaluation 
        batch_size = 4096
        indice_bs = [None for _ in range(0, N, batch_size)]
        values_bs = [None for _ in range(0, N, batch_size)]
        batch_ids = indices[0] // batch_size
        for i in range(len(indice_bs)):
            x, y = indices[:, batch_ids==i]
            x -= i * batch_size
            indice_bs[i] = torch.stack((x, y), dim=0)
            values_bs[i] = values[batch_ids==i]
        
        top_x, mr, mrr = [0, 0], 0, 0
        for idx, (indice, values) in enumerate(zip(indice_bs, values_bs)):
            start = idx * batch_size
            end = min((idx + 1) * batch_size, N)
            local_sp_S = torch.sparse.FloatTensor(indice, values, size=(end - start, M))
            local_S = local_sp_S.to_dense().to(device)
            local_S[:, cands] += 1000
            local_top_x, local_mr, local_mrr = _get_hits(local_S, gt_seed[start:end], top_k)
            top_x[0] += local_top_x[0]
            top_x[1] += local_top_x[1]
            mr += local_mr
            mrr += local_mrr
        top_x[0] /= N
        top_x[1] /= N
        mr /= N
        mrr /= N
        return top_x, mr, mrr

    device = seeds1.device
    sp_S = sp_S.coalesce()
    top_x0, mr0, mrr0 = sparse_evaluate(sp_S, seeds1, cands1)
    return top_x0, mr0, mrr0


@torch.no_grad()
def evaluate(feats0, feats1, seeds, cands0, cands1, print_info=True, top_k=(1, 10)):
    def _get_hits(S, gt, top_k):
        '''
        :param S: [seed num0, node num1]
        :param gt: [seed num0]
        '''
        sorted_indice = torch.argsort(S, dim=1, descending=True)  # shape = [seed_num0, node num1]
        is_gt = (sorted_indice == gt.reshape(-1, 1)).int()  # shape = [seed_num0, node_num1]
        gt_rank = torch.argmax(is_gt, dim=1) + 1  # shape = [seed_num0]
        top_x = [0.0 for _ in range(len(top_k))]
        for i, k in enumerate(top_k):
            top_x[i] = float(torch.sum((gt_rank <= k).float())) * 100
        mr = float(torch.sum(gt_rank.float()))
        mrr = float(torch.sum(1 / gt_rank.float()))
        return top_x, mr, mrr
    
    def batched_evaluate(feats0, feats1, seeds0, seeds1, cands1):
        S0 = feats0[seeds0] @ feats1.T
        N, M = S0.shape
        batch_size = 256
        for i in range(0, len(cands1), batch_size):
            S0[:, cands1[i:i+batch_size]] += 1000
        
        batch_size = 1024
        top_x0, mr0, mrr0 = [0, 0], 0, 0
        for i in range(0, len(seeds), batch_size):
            local_top_x0, local_mr0, local_mrr0 = _get_hits(S0[i:i+batch_size], seeds1[i:i+batch_size], top_k)
            top_x0[0] += local_top_x0[0]
            top_x0[1] += local_top_x0[1]
            mr0 += local_mr0
            mrr0 += local_mrr0
        top_x0[0] /= N
        top_x0[1] /= N
        mr0 /= N
        mrr0 /= N
        return top_x0, mr0, mrr0
    
    top_x0, mr0, mrr0 = batched_evaluate(feats0, feats1, seeds[:, 0], seeds[:, 1], cands1)
    top_x1, mr1, mrr1 = batched_evaluate(feats1, feats0, seeds[:, 1], seeds[:, 0], cands0)

    pfm = Performance(top_x0, mr0, mrr0, top_x1, mr1, mrr1)
    if print_info:
        pfm.print_performance()
    return pfm


def remove_punctruation(s):
    s = s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    return s


def read_seeds(path):
    with open(str(path), 'r', encoding='utf8') as f:
        lines = f.readlines()
        lines = [line.strip().split('\t') for line in lines]
        lines = [(int(sr), int(tg)) for sr, tg in lines]
    return lines


def read_mapping(path):
    with open(str(path), 'r', encoding='utf8') as f:
        lines = f.readlines()
        lines = [line.strip().split('\t') for line in lines]
    id2item = []
    for line in lines:
        idx = int(line[0])
        if len(line) == 1:
            id2item.append((int(idx), ''))
        else:
            id2item.append((int(idx), line[1]))
    mapping = dict(id2item)
    assert len(mapping) == len(id2item)
    return mapping


def read_triples(path):
    with open(str(path), 'r', encoding='utf8') as f:
        lines = f.readlines()
        lines = [line.strip().split('\t') for line in lines]
        triples = [(int(h), int(r), int(t)) for h, t, r in lines]
    return triples


class DWY100k(object):
    def __init__(self, directory, device):
        self.device = device
        self.directory = Path(directory)
        sr, tg = self.directory.name.split('_')

        '''Load mapping'''
        self.id2ent0 = read_mapping(self.directory / ('id2entity_%s.txt' % sr))
        self.id2ent1 = read_mapping(self.directory / ('id2entity_%s.txt' % tg))

        '''Load triples'''
        triples0 = read_triples(self.directory / ('triples_%s.txt' % sr))
        triples1 = read_triples(self.directory / ('triples_%s.txt' % tg))
        self.prepare_for_ea(triples0, triples1)        
        # self.load_glove_embedding()
        self.load_bert_embedding(True)


    def prepare_for_ea(self, triples0, triples1):
        self.triples0 = torch.tensor(triples0)
        self.triples1 = torch.tensor(triples1)
        '''Load seeds'''

        split_dir = self.directory
        train_seeds = read_seeds(split_dir / 'train_entity_seeds.txt')
        valid_seeds = read_seeds(split_dir / 'valid_entity_seeds.txt')
        test_seeds = read_seeds(split_dir / 'test_entity_seeds.txt')
        cand_ents0, cand_ents1 = zip(*test_seeds)

        # '''read relation seeds'''
        # relation_seeds = read_seeds(self.directory / 'relation_seeds.txt')
        # self.relation_seeds = torch.tensor(relation_seeds)
        self.cand_ents0 = torch.tensor(cand_ents0)
        self.cand_ents1 = torch.tensor(cand_ents1)
        self.train_seeds = torch.tensor(train_seeds)
        self.valid_seeds = torch.tensor(valid_seeds)
        self.test_seeds = torch.tensor(test_seeds)
        
    def load_bert_embedding(self, use_char_embedding=False):
        feats0 = np.load(self.directory / 'feats0.npy')
        feats1 = np.load(self.directory / 'feats1.npy')
        feats0 = torch.from_numpy(feats0)
        feats1 = torch.from_numpy(feats1)
        self.feats0 = F.normalize(feats0, p=2, dim=-1)
        self.feats1 = F.normalize(feats1, p=2, dim=-1)

        if use_char_embedding:
            id2chars0 = {i: remove_punctruation(name).lower() for i, name in self.id2ent0.items()}
            id2chars1 = {i: remove_punctruation(name).lower() for i, name in self.id2ent1.items()}
            bigram_d = {}
            ent_names = set(id2chars0.values()).union(id2chars1.values())
            for name in ent_names:
                # name = remove_punctruation(name).lower()
                for word in name.split():
                    for idx in range(len(word)-1):
                        if word[idx:idx+2] not in bigram_d:
                            bigram_d[word[idx:idx+2]] = len(bigram_d)
            
            char_feats0 = torch.zeros((self.feats0.shape[0], len(bigram_d)),)
            char_feats1 = torch.zeros((self.feats1.shape[0], len(bigram_d)),)
            for i in range(len(id2chars0)):
                name = id2chars0[i]
                for word in name.split():
                    for idx in range(len(word)-1):
                        char_feats0[i, bigram_d[word[idx:idx+2]]] += 1
            for i in range(len(id2chars1)):
                name = id2chars1[i]
                for word in name.split():
                    for idx in range(len(word)-1):
                        char_feats1[i, bigram_d[word[idx:idx+2]]] += 1
            
            char_feats0, char_feats1 = char_feats0.to(self.device), char_feats1.to(self.device)
            self.feats0, self.feats1 = self.feats0.to(self.device), self.feats1.to(self.device)

            char_feats0 = F.normalize(char_feats0, p=2, dim=-1)
            char_feats1 = F.normalize(char_feats1, p=2, dim=-1)
            self.feats0 = torch.cat((self.feats0, char_feats0), dim=1)
            self.feats1 = torch.cat((self.feats1, char_feats1), dim=1)


    def load_glove_embedding(self):
        '''
        Load glove word embedding
        Embedding from https://github.com/syxu828/Crosslingula-KG-Matching
        '''
        word2id = {}
        with open(self.directory.parent / 'sub.glove.6B.300d.txt', 'r', encoding='utf8') as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
        glove_embedding = []
        for idx, line in enumerate(lines):
            if len(line) == 301:
                word2id[line[0]] = idx
                glove_embedding.append(line[1:])
            elif len(line) == 300:
                word2id['**<UNK>**'] = idx
                glove_embedding.append(line)
            else:
                continue

        glove_embedding = np.asarray(glove_embedding, dtype=np.float32)
        glove_embedding = torch.from_numpy(glove_embedding)
        N, D = glove_embedding.shape
        glove_embedding = torch.cat([glove_embedding, torch.zeros(1, D)], dim=0)
        assert N == len(word2id)

        '''Transform entity name into embedding'''

        def _transform(id2name, word2id, glove_embedding):
            padding_id = word2id.get('**<UNK>**', N)
            zero_padding_id = len(word2id)
            id2name = [remove_punctruation(id2name[i]) for i in range(len(id2name))]
            id2name = [id2name[i].split() for i in range(len(id2name))]
            id2name_id = [[word2id.get(word.lower(), padding_id) for word in name] for name in id2name]
            max_len = max(len(name) for name in id2name_id)
            id2name_id = [name + [zero_padding_id] * (max_len - len(name)) for name in id2name_id]
            id2name_id = torch.tensor(id2name_id)
            id2embed = glove_embedding[id2name_id]  # shape = [N, max_len, D]
            # id2embed, _ = torch.max(id2embed, dim=1)
            id2embed = id2embed.sum(dim=1)
            return id2embed


        self.feats0 = _transform(self.id2ent0, word2id, glove_embedding)
        self.feats1 = _transform(self.id2ent1, word2id, glove_embedding)
        self.feats0 = F.normalize(self.feats0, p=2, dim=-1)
        self.feats1 = F.normalize(self.feats1, p=2, dim=-1)


@torch.no_grad()
def run_dwy100k_nodematch(args):
    directory = './data/DWY100k/%s/' % args.dataset
    dataset = DWY100k(directory, device=args.device)
    feats0 = dataset.feats0.to(args.device)
    feats1 = dataset.feats1.to(args.device)
    test_seeds = dataset.test_seeds.to(args.device)
    cand_ents0 = dataset.cand_ents0.to(args.device)
    cand_ents1 = dataset.cand_ents1.to(args.device)
    knn_values0, knn_idx0, sp_S0 = sparse_similarity(feats0, feats1)
    knn_values1, knn_idx1, sp_S1 = sparse_similarity(feats1, feats0)
    sp_S = (sp_S0 + sp_S1.transpose(0, 1)) / 2
    pfm = evaluate_sparse(sp_S, test_seeds, cand_ents0, cand_ents1, print_info=True)
    return pfm, 0


@torch.no_grad()
def run_dwy100k(args):
    directory = './data/DWY100k/%s/' % args.dataset
    dataset = DWY100k(directory, device=args.device)
    in_dim = dataset.feats0.shape[-1]

    feats0 = dataset.feats0.to(args.device)
    feats1 = dataset.feats1.to(args.device)
    N0 = feats0.shape[0]
    N1 = feats1.shape[0]

    ## set the edges of two graphs. use undirected edges and add selfloops
    triples0 = dataset.triples0.tolist()
    triples1 = dataset.triples1.tolist()
    edges0 = list({(h, t) for h, r, t in triples0 if h!=t})
    edges1 = list({(h, t) for h, r, t in triples1 if h!=t})
    edges0.extend([(t, h) for h, t in edges0])
    edges1.extend([(t, h) for h, t in edges1])
    edges0.extend([(i, i) for i in range(N0)])
    edges1.extend([(i, i) for i in range(N1)])
    edges0 = torch.tensor(edges0, device=args.device)
    edges1 = torch.tensor(edges1, device=args.device)

    train_seeds = dataset.train_seeds.to(args.device)
    test_seeds = dataset.test_seeds.to(args.device)
    cand_ents0 = dataset.cand_ents0.to(args.device)
    cand_ents1 = dataset.cand_ents1.to(args.device)

    start = time.time()
    if args.use_supervision:
        syn_feats0 = feats0.clone().detach()
        syn_feats1 = feats1.clone().detach()

        ## set seed nodes to have same features
        syn_feats1[train_seeds[:, 1]] = feats0[train_seeds[:, 0]]
        syn_feats0[train_seeds[:, 0]] = feats1[train_seeds[:, 1]]
    
    ## initialize model
    rand_gnn = GraphSAGE(in_dim, args.dim, args.num_layer, args.act_func, use_node_feature=True, weight_free=args.weight_free, bias=False)
    rand_gnn.basic_tfgm = args.use_basic_tfgm
    rand_gnn.eval()
    rand_gnn.to(args.device)

    feats_g0 = rand_gnn(feats0, edges0.T)
    feats_g1 = rand_gnn(feats1, edges1.T)
    if args.use_supervision:
        syn_feats_g0 = rand_gnn(syn_feats0, edges0.T)
        syn_feats_g1 = rand_gnn(syn_feats1, edges1.T)
        # knn_values0, knn_idx0, sp_S0 = sparse_similarity(feats_g0, syn_feats_g1)
        # knn_values1, knn_idx1, sp_S1 = sparse_similarity(feats_g1, syn_feats_g0)
        knn_values0, knn_idx0, sp_S0 = sparse_similarity(feats_g0, syn_feats_g1)
        knn_values1, knn_idx1, sp_S1 = sparse_similarity(syn_feats_g0, feats_g1)
        
        sp_S = (sp_S0 + sp_S1) / 2
    else:
        knn_values0, knn_idx0, sp_S0 = sparse_similarity(feats_g0, feats_g1)
        knn_values1, knn_idx1, sp_S1 = sparse_similarity(feats_g1, feats_g0)
        sp_S = (sp_S0 + sp_S1.transpose(0, 1)) / 2
    pfm = evaluate_sparse(sp_S, test_seeds, cand_ents0, cand_ents1, print_info=True)
    del sp_S
    
    
    if args.use_dgmc:
        if args.use_supervision:
            knn_values1, knn_idx1, _ = sparse_similarity(feats_g1, syn_feats_g0)

        del feats_g0, feats_g1, feats0, feats1
        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()

        rnd_gnn = GraphSAGE(args.rnd_dim, args.rnd_dim, args.dgmc_layer, args.act_func, True, weight_free=args.weight_free)
        dgmc = RandDGMC(rnd_gnn, 100, args.num_steps, weight_free=False)
        dgmc.to(args.device)
            
        sp_S0 = dgmc.sp_forward(knn_values0, knn_idx0, edges0.T, edges1.T, N0, N1)
        
        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()

        if args.symmetric_align:
            sp_S1 = dgmc.sp_forward(knn_values1, knn_idx1, edges1.T, edges0.T, N1, N0)

            with torch.cuda.device(args.device):
                torch.cuda.empty_cache()
            sp_S = (sp_S0 + sp_S1.transpose(0, 1)) / 2
        else:
            sp_S = sp_S0
        pfm = evaluate_sparse(sp_S, test_seeds, cand_ents0, cand_ents1, print_info=True)

    time_spend = time.time() - start
    print('Inference costs', time_spend, 'seconds.')
    return pfm, time_spend



if __name__ == '__main__':
    '''Select dataset'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wd_dbp', help='The dataset used for evaluation.')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--rnd_dim', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--dgmc_layer', type=int, default=1)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='identity')
    parser.add_argument('--use_dgmc', action='store_true')
    parser.add_argument('--weight_free', action='store_true')
    parser.add_argument('--use_supervision', action='store_true')
    parser.add_argument('--use_char_embedding', action='store_true')
    parser.add_argument('--symmetric_align', action='store_true')
    parser.add_argument('--use_basic_tfgm', action='store_true')
    parser.add_argument('--use_node_match', action='store_true')
    args = parser.parse_args()
    args.act_func = act_func_dict[args.act_func]
    args.device = 'cuda:%d' % (args.gpu_id)
    pfm, time_spend = run_dwy100k(args)
    exit()
    # pfms = []
    # for data in ['wd_dbp']:
    #     args.dataset = data
    #     if args.use_node_match:
    #         pfm, time_spend = run_dwy100k_nodematch(args)
    #     else:
    #         pfm, time_spend = run_dwy100k(args)
    #     pfms.append(pfm)
    # print('SR-TG: %.1f %.1f %.3f %.1f %.1f %.3f' % (pfms[0].top_x0[0], pfms[0].top_x0[1], pfms[0].mrr0, pfms[1].top_x0[0], pfms[1].top_x0[1], pfms[1].mrr0))