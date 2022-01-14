import random
import functools
import time
import torch
import numpy as np
from collections import namedtuple
import torch.nn.functional as F


class Performance(namedtuple('Performance', ['top_x0', 'mr0', 'mrr0', 'top_x1', 'mr1', 'mrr1'])):
    def print_performance(self, top_k=(1,10)):
        print_time_info('For each source:', dash_top=True)
        print_time_info('MR: %.2f; MRR: %.4f.' % (self.mr0, self.mrr0))
        for i in range(len(self.top_x0)):
            print_time_info('Hits@%d: %.2f%%' % (top_k[i], self.top_x0[i]))
        print('')
        print_time_info('For each target:')
        print_time_info('MR: %.2f; MRR: %.4f.' % (self.mr1, self.mrr1))
        for i in range(len(self.top_x1)):
            print_time_info('Hits@%d: %.2f%%' % (top_k[i], self.top_x1[i]))
        print('-------------------------------------')

    def print(self):
        print_time_info('SR-TG: HT@1 HT@10 MRR')
        print_time_info('SR-TG: %.2f %.2f %.4f' % (self.top_x0[0], self.top_x0[1], self.mrr0))# self.top_x1[0], self.top_x1[1], self.mrr1))

def evaluate(S, seeds, cands0, cands1, print_info=True, top_k=(1, 10)):
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
            top_x[i] = float(torch.mean((gt_rank <= k).float())) * 100
        mr = float(torch.mean(gt_rank.float()))
        mrr = float(torch.mean(1 / gt_rank.float()))
        return top_x, mr, mrr
    
    S0 = S[seeds[:, 0]]
    S1 = S.T[seeds[:, 1]]
    '''make candidates stand out'''
    S0[:, cands1] += 1000
    S1[:, cands0] += 1000

    top_x0, mr0, mrr0 = _get_hits(S0, seeds[:, 1], top_k)
    top_x1, mr1, mrr1 = _get_hits(S1, seeds[:, 0], top_k)
    
    pfm = Performance(top_x0, mr0, mrr0, top_x1, mr1, mrr1)
    
    if print_info:
        pfm.print_performance()
    return pfm

def l2_similarity_nbyn(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    a2 = torch.sum(a ** 2, dim=-1) # shape = [num_item_1]
    b2 = torch.sum(b ** 2, dim=-1) # shape = [num_item_2]
    ab = a @ b.T # shape = [num_item_1, num_item_2]
    l2_distance = torch.pow(a2.unsqueeze(-1) + b2.unsqueeze(0) - 2 * ab, 0.5) # shape = [num_item_1, num_item_2]
    return 1 / (l2_distance + 1)

def l1_similarity_nbyn(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    a = a.unsqueeze(1) # shape = [num_item_1, 1, embedding_dim]
    b = b.unsqueeze(0) # shape = [1, num_item_2, embedding_dim]
    dis = torch.sum(torch.abs(a - b), dim=-1)
    return -dis

def cosine_similarity_nbyn(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    a = F.normalize(a, dim=-1, p=2)
    b = F.normalize(b, dim=-1, p=2)
    return torch.mm(a, b.t())

def batch_median(list_of_sim, device, batch_size=64):
    # shape = [N_sim, N0, N1]
    list_of_partial_median = []
    for i in range(0, list_of_sim.shape[1], batch_size):
        list_of_partial_sim = list_of_sim[:, i:i+batch_size].to(device) # shape = [N_sim, batch_size, N1]
        partial_median, _ = torch.median(list_of_partial_sim, dim=0) # shape = [batch_size, N1]
        list_of_partial_median.append(partial_median)
    median_sim = torch.cat(list_of_partial_median, dim=0)
    return median_sim


def print_time_info(string, end='\n', dash_top=False, dash_bot=False, file=None):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(time.time())))
    string = "[%s] %s" % (times, str(string))
    if dash_top:
        print(len(string) * '-', file=file)
    print(string, end=end, file=file)
    if dash_bot:
        print(len(string) * '-', file=file)


def set_random_seed(seed_value=1, print_info=False):
    if print_info:
        print_time_info('Random seed is set to %d.' % (seed_value))
    torch.manual_seed(seed_value)  # cpu  vars
    np.random.seed(seed_value)  # cpu vars
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)


def get_hits(sim, top_k=(1, 10), print_info=True):
    if isinstance(sim, np.ndarray):
        sim = torch.from_numpy(sim)
    top_lr, mr_lr, mrr_lr = topk(sim, top_k)
    top_rl, mr_rl, mrr_rl = topk(sim.t(), top_k)

    if print_info:
        print_time_info('For each source:', dash_top=True)
        print_time_info('MR: %.2f; MRR: %.2f%%.' % (mr_lr, mrr_lr))
        for i in range(len(top_lr)):
            print_time_info('Hits@%d: %.2f%%' % (top_k[i], top_lr[i]))
        print('')
        print_time_info('For each target:')
        print_time_info('MR: %.2f; MRR: %.2f%%.' % (mr_rl, mrr_rl))
        for i in range(len(top_rl)):
            print_time_info('Hits@%d: %.2f%%' % (top_k[i], top_rl[i]))
        print('-------------------------------------')
    return top_lr, top_rl, mr_lr, mr_rl, mrr_lr, mrr_rl


def topk(sim, top_k=(1, 10, 50, 100)):
    # Sim shape = [num_ent, num_ent]
    assert sim.shape[0] == sim.shape[1]
    test_num = sim.shape[0]
    batched = True
    if sim.shape[0] * sim.shape[1] < 20000 * 128:
        batched = False
        sim = sim.cuda() if torch.cuda.is_available() else sim

    def _opti_topk(sim):
        sorted_arg = torch.argsort(sim)
        true_pos = torch.arange(test_num).reshape((-1, 1))
        if torch.cuda.is_available():
            true_pos = true_pos.cuda()
        locate = sorted_arg - true_pos
        # del sorted_arg, true_pos
        locate = torch.nonzero(locate == 0)
        cols = locate[:, 1]  # Cols are ranks
        cols = cols.float()
        top_x = [0.0] * len(top_k)
        for i, k in enumerate(top_k):
            top_x[i] = float(torch.sum(cols < k)) / test_num * 100
        mr = float(torch.sum(cols + 1)) / test_num
        mrr = float(torch.sum(1.0 / (cols + 1))) / test_num * 100
        return top_x, mr, mrr

    def _opti_topk_batched(sim):
        mr = 0.0
        mrr = 0.0
        top_x = [0.0] * len(top_k)
        batch_size = 1024
        for i in range(0, test_num, batch_size):
            batch_sim = sim[i:i + batch_size].cuda()
            sorted_arg = torch.argsort(batch_sim)
            true_pos = torch.arange(
                batch_sim.shape[0]).reshape((-1, 1)).cuda() + i
            locate = sorted_arg - true_pos
            # del sorted_arg, true_pos
            locate = torch.nonzero(locate == 0)
            cols = locate[:, 1]  # Cols are ranks
            cols = cols.float()
            mr += float(torch.sum(cols + 1))
            mrr += float(torch.sum(1.0 / (cols + 1)))
            for i, k in enumerate(top_k):
                top_x[i] += float(torch.sum(cols < k))
        mr = mr / test_num
        mrr = mrr / test_num * 100
        for i in range(len(top_x)):
            top_x[i] = top_x[i] / test_num * 100
        return top_x, mr, mrr

    with torch.no_grad():
        if not batched:
            return _opti_topk(sim)
        return _opti_topk_batched(sim)


def timeit(func):
    @functools.wraps(func)
    def timed(*args, **kw):
        ts = time.time()
        print_time_info('Method: %s started!' % (func.__name__), dash_top=True)
        result = func(*args, **kw)
        te = time.time()
        print_time_info('Method: %s cost %.2f sec!' % (func.__name__, te - ts), dash_bot=True)
        return result

    return timed


def batch_sinkhorn(S, max_iter=10, tau = 0.1):
    '''
    sinkhorn algorithm
    credit to https://github.com/Thinklab-SJTU/ThinkMatch/blob/master/src/lap_solvers/sinkhorn.py
    S shape = [B, N, M]
    '''
    log_S = S / tau
    for i in range(max_iter):
        if i % 2 == 0:
            log_S_sum = torch.logsumexp(log_S, 1, keepdim=True) # shape = [B, 1, M]
            log_S -= log_S_sum
        else:
            log_S_sum = torch.logsumexp(log_S, 2, keepdim=True) # shape = [B, N, 1]
            log_S -= log_S_sum
    # S = torch.exp(log_S)
    return S

def sinkhorn(S, max_iter=10, tau = 0.1):
    '''
    sinkhorn algorithm
    credit to https://github.com/Thinklab-SJTU/ThinkMatch/blob/master/src/lap_solvers/sinkhorn.py
    S shape = [N, M]
    '''
    log_S = S / tau
    for i in range(max_iter):
        if i % 2 == 0:
            log_S_sum = torch.logsumexp(log_S, 0, keepdim=True)
            log_S -= log_S_sum
        else:
            log_S_sum = torch.logsumexp(log_S, 1, keepdim=True)
            log_S -= log_S_sum
    # S = torch.exp(log_S)
    return S


def topk_sinkhorn(S, max_iter=10, k=10, tau=0.1):
    '''
    select only top-k entries for normalization
    S shape = [N, M]
    '''
    def sparse_normalization(sp_S, dim):
        # sp_S shape = [N, M]
        sp_S_exp = torch.sparse.FloatTensor(sp_S._indices(), torch.exp(sp_S._values()), size=(N, M))
        dim_sum = torch.sparse.sum(sp_S_exp, dim=dim).to_dense()

        ### normalize dimension sum
        sp_ones = torch.sparse.FloatTensor(sp_S._indices(), torch.ones_like(sp_S._values()), size=(N, M))
        lengths = torch.sparse.sum(sp_ones, dim=dim).to_dense() # shape = [M] or shape = [N]
        lengths = lengths / torch.mean(lengths)
        dim_sum /= lengths

        log_sum = torch.log(dim_sum) # shape = [M] or shape = [N]
        row_idx, col_idx = sp_S._indices() # shape = [2, E]
        if dim == 0:
            minus_sp_S = torch.sparse.FloatTensor(sp_S._indices(), log_sum[col_idx], size=(N, M))
        else:
            minus_sp_S = torch.sparse.FloatTensor(sp_S._indices(), log_sum[row_idx], size=(N, M))
        sp_S = (sp_S - minus_sp_S).coalesce()
        return sp_S

    N, M = S.shape
    device = S.device

    ## select topk entries
    topk_col, topk_col_idx = torch.topk(S, k=k, dim=1) # shape = [N, k]
    topk_row, topk_row_idx = torch.topk(S, k=k, dim=0) # shape = [k, M]
    row_idx = torch.arange(N, dtype=torch.long, device=device).reshape(-1, 1).repeat(1, k) # shape = [N, k]
    col_idx = torch.arange(M, dtype=torch.long, device=device).reshape(1, -1).repeat(k, 1) # shape = [k, M]
    coord0 = torch.stack((row_idx.flatten(), topk_col_idx.flatten()), dim=1).tolist() # shape = [N*k, 2]
    coord1 = torch.stack((topk_row_idx.flatten(), col_idx.flatten()), dim=1).tolist() # shape = [M*k, 2]
    values0 = topk_col.flatten().tolist()
    values1 = topk_row.flatten().tolist()
    
    ## de-duplicates
    coord2value = {}
    for coord, val in zip(coord0, values0):
        coord2value[tuple(coord)] = val
    for coord, val in zip(coord1, values1):
        coord2value[tuple(coord)] = val
    coords, values = zip(*coord2value.items())
    coords, values = torch.LongTensor(coords).to(device).T, torch.FloatTensor(values).to(device)
    
    ## transform to sparse tensor
    values /= tau
    sp_S = torch.sparse.FloatTensor(coords, values, size=(N, M))

    ## normalization
    for i in range(max_iter):
        if i % 2 == 0:
            sp_S = sparse_normalization(sp_S, 0)
        else:
            sp_S = sparse_normalization(sp_S, 1)
    # torch.exp_(sp_S._values())
    S = sp_S.to_dense()
    return S
