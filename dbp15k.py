import torch
import torch.nn.functional as F
import numpy as np
import argparse
from utils import evaluate, cosine_similarity_nbyn
from models import GraphSAGE, RandDGMC, act_func_dict
import time
from pathlib import Path


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


class DBP15k(object):
    def __init__(self, directory, load_new_split, use_char_embedding, device):
        self.device = device
        self.directory = Path(directory)
        sr, tg = self.directory.name.split('_')

        '''Load mapping'''
        self.id2ent0 = read_mapping(self.directory / ('id2entity_%s.txt' % sr))
        self.id2ent1 = read_mapping(self.directory / ('id2entity_%s.txt' % tg))
        self.id2rel0 = read_mapping(self.directory / ('id2relation_%s.txt' % sr))
        self.id2rel1 = read_mapping(self.directory / ('id2relation_%s.txt' % tg))

        '''Load triples'''
        triples0 = read_triples(self.directory / ('triples_%s.txt' % sr))
        triples1 = read_triples(self.directory / ('triples_%s.txt' % tg))
        self.prepare_for_ea(triples0, triples1, load_new_split)        
        self.load_glove_embedding(use_char_embedding) 


    def prepare_for_ea(self, triples0, triples1, load_new_split):
        self.triples0 = torch.tensor(triples0)
        self.triples1 = torch.tensor(triples1)
        '''Load seeds'''
        if load_new_split:
            split_dir = self.directory / 'split'
            train_seeds = read_seeds(split_dir / 'train_entity_seeds.txt')
            valid_seeds = read_seeds(split_dir / 'valid_entity_seeds.txt')
            test_seeds = read_seeds(split_dir / 'test_entity_seeds.txt')
            unk_seeds = valid_seeds + test_seeds
            cand_ents0, cand_ents1 = zip(*unk_seeds)
        else:
            split_dir = self.directory / 'jape'
            train_seeds = read_seeds(split_dir / 'train_entity_seeds.txt')
            valid_seeds = read_seeds(split_dir / 'test_entity_seeds.txt')
            test_seeds = read_seeds(split_dir / 'test_entity_seeds.txt')
            cand_ents0, cand_ents1 = zip(*valid_seeds)

        '''read relation seeds'''
        relation_seeds = read_seeds(self.directory / 'relation_seeds.txt')
        self.relation_seeds = torch.tensor(relation_seeds)
        self.cand_ents0 = torch.tensor(cand_ents0)
        self.cand_ents1 = torch.tensor(cand_ents1)
        self.train_seeds = torch.tensor(train_seeds)
        self.valid_seeds = torch.tensor(valid_seeds)
        self.test_seeds = torch.tensor(test_seeds)
        

    def load_glove_embedding(self, use_char_embedding):
        '''
        Load glove word embedding
        Embedding from https://github.com/syxu828/Crosslingula-KG-Matching
        '''
        word2id = {}
        # with open(self.directory.parent / 'sub.glove.300d', 'r', encoding='utf8') as f:
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
            id2name = [id2name[i].split() for i in range(len(id2name))]
            id2name_id = [[word2id.get(word.lower(), padding_id) for word in name] for name in id2name]
            max_len = max(len(name) for name in id2name_id)
            id2name_id = [name + [zero_padding_id] * (max_len - len(name)) for name in id2name_id]
            id2name_id = torch.tensor(id2name_id)
            id2embed = glove_embedding[id2name_id]  # shape = [N, max_len, D]
            id2embed = id2embed.sum(dim=1)
            return id2embed

        def load_feats(path):
            with open(str(path), 'r', encoding='utf8') as f:
                lines = [line.strip().split('\t') for line in f.readlines()]
                id2trans = [(int(idx), trans) for idx, trans, ent in lines]
            mapping = dict(id2trans)
            assert len(mapping) == len(id2trans)
            return mapping

        sr, tg = self.directory.name.split('_')
        use_seu = True
        if use_seu:
            id2feats0 = read_mapping(self.directory / ('id2ent_%s_seu.txt' % sr))
            id2feats1 = read_mapping(self.directory / ('id2ent_%s_seu.txt' % tg))
        else:
            id2feats0 = load_feats(self.directory / 'graph_match' / ('id2feats_%s.txt' % sr))
            id2feats1 = load_feats(self.directory / 'graph_match' / ('id2feats_%s.txt' % tg))
        self.feats0 = _transform(id2feats0, word2id, glove_embedding)
        self.feats1 = _transform(id2feats1, word2id, glove_embedding)
        self.feats0 = F.normalize(self.feats0, p=2, dim=-1)
        self.feats1 = F.normalize(self.feats1, p=2, dim=-1)

        ## use the random projection of onehot encoding word appearance as feature
        if False:
            rand_onehot = torch.randn_like(glove_embedding)
            rand_onehot[-1] = 0
            randn_feats0 = _transform(id2feats0, word2id, rand_onehot)
            randn_feats1 = _transform(id2feats1, word2id, rand_onehot)
            randn_feats0, randn_feats1 = randn_feats0.to(self.device), randn_feats1.to(self.device)
            self.feats0, self.feats1 = self.feats0.to(self.device), self.feats1.to(self.device)

            randn_feats0 = F.normalize(randn_feats0, p=2, dim=-1)
            randn_feats1 = F.normalize(randn_feats1, p=2, dim=-1)
            self.feats0 = F.normalize(self.feats0, p=2, dim=-1)
            self.feats1 = F.normalize(self.feats1, p=2, dim=-1)
            self.feats0 = torch.cat((self.feats0, randn_feats0), dim=1)
            self.feats1 = torch.cat((self.feats1, randn_feats1), dim=1)
        
        ## Use bi-gram as feature
        if use_char_embedding:
            bigram_d = {}
            ent_names = set(id2feats0.values()).union(id2feats1.values())
            for name in ent_names:
                for word in name.split():
                    for idx in range(len(word)-1):
                        if word[idx:idx+2] not in bigram_d:
                            bigram_d[word[idx:idx+2]] = len(bigram_d)
            
            char_feats0 = torch.zeros((self.feats0.shape[0], len(bigram_d)),)
            char_feats1 = torch.zeros((self.feats1.shape[0], len(bigram_d)),)
            for i in range(len(id2feats0)):
                name = id2feats0[i]
                for word in name.split():
                    for idx in range(len(word)-1):
                        char_feats0[i, bigram_d[word[idx:idx+2]]] += 1
            for i in range(len(id2feats1)):
                name = id2feats1[i]
                for word in name.split():
                    for idx in range(len(word)-1):
                        char_feats1[i, bigram_d[word[idx:idx+2]]] += 1
            
            char_feats0, char_feats1 = char_feats0.to(self.device), char_feats1.to(self.device)
            self.feats0, self.feats1 = self.feats0.to(self.device), self.feats1.to(self.device)

            char_feats0 = F.normalize(char_feats0, p=2, dim=-1)
            char_feats1 = F.normalize(char_feats1, p=2, dim=-1)
            self.feats0 = torch.cat((self.feats0, char_feats0), dim=1)
            self.feats1 = torch.cat((self.feats1, char_feats1), dim=1)

@torch.no_grad()
def run_dbp15k_node_match(args):
    directory = './data/DBP15k/%s/' % args.dataset
    dataset = DBP15k(directory, args.load_hard_split, args.use_char_embedding, device=args.device)
    feats0 = dataset.feats0.to(args.device)
    feats1 = dataset.feats1.to(args.device)

    test_seeds = dataset.test_seeds.to(args.device)
    cand_ents0 = dataset.cand_ents0.to(args.device)
    cand_ents1 = dataset.cand_ents1.to(args.device)

    S = feats0 @ feats1.T    
    pfm = evaluate(S, test_seeds, cand_ents0, cand_ents1, print_info=False)
    torch.cuda.empty_cache()
    return pfm, 0


@torch.no_grad()
def run_dbp15k(args):
    directory = './data/DBP15k/%s/' % args.dataset
    dataset = DBP15k(directory, args.load_hard_split, args.use_char_embedding, device=args.device)
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
        S0 = cosine_similarity_nbyn(feats_g0, syn_feats_g1)
        S1 = cosine_similarity_nbyn(syn_feats_g0, feats_g1)
        S = (S0 + S1) / 2
        del S0, S1
        torch.cuda.empty_cache()
    else:
        S = cosine_similarity_nbyn(feats_g0, feats_g1)
    
    pfm = evaluate(S, test_seeds, cand_ents0, cand_ents1, print_info=False)
    torch.cuda.empty_cache()

    if args.use_dgmc:
        rnd_gnn = GraphSAGE(args.rnd_dim, args.rnd_dim, args.dgmc_layer, args.act_func, True, weight_free=args.weight_free)
        dgmc = RandDGMC(rnd_gnn, 100, args.num_steps, weight_free=False)
        dgmc.to(args.device)
        
        S0 = dgmc(S, edges0.T, edges1.T)
        torch.cuda.empty_cache()

        if args.symmetric_align:
            S1 = dgmc(S.T, edges1.T, edges0.T).transpose(0, 1)
            torch.cuda.empty_cache()
            S = (S0 + S1).to_dense()
            del S0, S1
        else:
            S = S0.to_dense()
        torch.cuda.empty_cache()
        pfm = evaluate(S, test_seeds, cand_ents0, cand_ents1, print_info=False)

    time_spend = time.time() - start
    print('Inference costs', time_spend, 'seconds.')
    return pfm, time_spend


@torch.no_grad()
def run_dbp15k_new(args):
    directory = './data/DBP15k/%s/' % args.dataset
    dataset = DBP15k(directory, args.load_hard_split, args.use_char_embedding, device=args.device)
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
    
    ## initialize model
    rand_gnn = GraphSAGE(in_dim, args.dim, args.num_layer, args.act_func, use_node_feature=True, weight_free=args.weight_free, bias=False)
    rand_gnn.eval()
    rand_gnn.to(args.device)

    feats_g0, feats_g1 = rand_gnn.pair_forward(feats0, edges0.T, feats1, edges1.T, train_seeds)
    S0 = cosine_similarity_nbyn(feats_g0, feats_g1)
    feats_g0, feats_g1 = rand_gnn.pair_forward(feats1, edges1.T, feats0, edges0.T, train_seeds.flip(dims=(1,)))
    S1 = cosine_similarity_nbyn(feats_g0, feats_g1)
    S = S0 + S1.T
    
    pfm = evaluate(S, test_seeds, cand_ents0, cand_ents1, print_info=False)
    torch.cuda.empty_cache()

    if args.use_dgmc:
        rnd_gnn = GraphSAGE(args.rnd_dim, args.rnd_dim, args.dgmc_layer, args.act_func, True, weight_free=args.weight_free)
        dgmc = RandDGMC(rnd_gnn, 100, args.num_steps, weight_free=False)
        dgmc.to(args.device)
        
        S0 = dgmc(S, edges0.T, edges1.T)
        torch.cuda.empty_cache()

        if args.symmetric_align:
            S1 = dgmc(S.T, edges1.T, edges0.T).transpose(0, 1)
            torch.cuda.empty_cache()
            S = (S0 + S1).to_dense()
            del S0, S1
        else:
            S = S0.to_dense()
        torch.cuda.empty_cache()
        pfm = evaluate(S, test_seeds, cand_ents0, cand_ents1, print_info=False)

    time_spend = time.time() - start
    print('Inference costs', time_spend, 'seconds.')
    return pfm, time_spend

if __name__ == '__main__':
    '''Select dataset'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='zh_en', help='The dataset used for evaluation. Options: zh_en (Chinese-English), ja_en (Japanese-English), fr_en (French-English)')
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
    parser.add_argument('--load_hard_split', action='store_true')
    parser.add_argument('--use_char_embedding', action='store_true')
    parser.add_argument('--symmetric_align', action='store_true')
    parser.add_argument('--use_basic_tfgm', action='store_true')
    parser.add_argument('--use_node_match', action='store_true')
    args = parser.parse_args()
    args.act_func = act_func_dict[args.act_func]
    args.device = 'cuda:%d' % (args.gpu_id)
    run_dbp15k(args)
    exit()
    # pfms = []
    # for data in ['zh_en', 'ja_en', 'fr_en']:
    #     args.dataset = data
    #     if args.use_node_match:
    #         pfm, time_spend = run_dbp15k_node_match(args)
    #     else:
    #         pfm, time_spend = run_dbp15k(args)
    #     pfms.append(pfm)
    #     torch.cuda.empty_cache()
    # print('SR-TG: %.1f %.1f %.3f %.1f %.1f %.3f %.1f %.1f %.3f' % (pfms[0].top_x0[0], pfms[0].top_x0[1], pfms[0].mrr0, pfms[1].top_x0[0], pfms[1].top_x0[1], pfms[1].mrr0, pfms[2].top_x0[0], pfms[2].top_x0[1], pfms[2].mrr0))