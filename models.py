import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SplineConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GINConv
from utils import cosine_similarity_nbyn


act_func_dict = {'identity': lambda x:x, 'relu': F.relu, 'sigmoid': F.sigmoid}

def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out

def to_sparse(x, mask):
    return x[mask]

def to_dense(x, mask):
    out = x.new_zeros(tuple(mask.size()) + (x.size(-1), ))
    out[mask] = x
    return out

class SplineCNN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dim, act_func, num_layers):
        super(SplineCNN, self).__init__()
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.act_func = act_func
        self.convs = torch.nn.ModuleList()
        h_dim = in_dim
        for _ in range(num_layers):
            conv = SplineConvNormal(h_dim, out_dim, dim, kernel_size=5, root_weight=False)
            self.convs.append(conv)
            h_dim = out_dim
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        xs = [F.normalize(x, p=2, dim=-1)]
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            xs.append(F.normalize(x, dim=-1, p=2))
            x = self.act_func(x)
        x = torch.cat(xs, dim=-1)
        return x

class SplineConvNormal(SplineConv):
    def reset_parameters(self):
        size = self.weight.size(0) * self.weight.size(1)
        if self.weight is not None:
            nn.init.normal_(self.weight, 0, 1)
            self.weight.data /= math.sqrt(size)
        if self.root is not None:
            nn.init.normal_(self.root, 0, 1)
            self.root.data /= math.sqrt(size)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


def split_sim_batches(batch_sim, batch_pos1):
    '''
    batch_sim: shape = [sum_nodes_batch0, sum_nodes_batch1]
    batch_pos1: shape = [batch_size1, 2]
    return:
        batched_sim: shape = [sum_nodes_batch0, batch_size1, N1]
    '''
    device = batch_sim.device
    A0 = batch_sim.shape[0] # sum of node number in batch0
    B1 = len(batch_pos1) # number of graphs in batch1
    N1 = max([end - start for start, end in batch_pos1]) # maximum node num of graphs in batch1
    batched_sim = torch.zeros((A0, B1, N1), device=device)
    mask = torch.zeros_like(batched_sim, device=device, dtype=torch.bool)
    for i, (start, end) in enumerate(batch_pos1):
        batched_sim[:, i, :end-start] += batch_sim[:, start: end]
        mask[:, i, :end-start] = True
    batched_sim = batched_sim.masked_fill(~mask, float('-inf'))
    return batched_sim, mask

class RandDGMC(nn.Module):
    def __init__(self, psi_2, k, num_steps, weight_free=False):
        super(RandDGMC, self).__init__()
        self.weight_free = weight_free
        self.k = k  # keep only the topk element of the corresponding matrix while refinement
        self.psi_2 = psi_2
        dim = psi_2.in_dim
        self.rnd_dim = dim
        self.num_steps = num_steps

    def pairwise_forward(self, h_s, edges_s, feat_edge_s, g_pos_s, h_t, edges_t, feat_edge_t, g_pos_t):
        '''
        RandDGMC for pairwise similarities of two batches of graphs
        h_s: shape = [B_s * N_s, D]
        h_t: shape = [B_t * N_t, D]
        '''
        rnd_dim = self.rnd_dim
        device = h_s.device

        S_hat = cosine_similarity_nbyn(h_s, h_t) # shape = [B_s * N_s, B_t * N_t]
        S_hat *= self.num_steps
        b_s_hat, mask = split_sim_batches(S_hat, g_pos_t) # shape = [B_s * N_s, B_t, N_t]
        b_s = masked_softmax(b_s_hat, mask, dim=-1) # shape = [B_s * N_s, B_t, N_t]; S[i,j,:] is the i-th node's similarities with respect to nodes in the j-th graph 
        A0, B_t, N_t = b_s.shape
        
        for _ in range(self.num_steps):
            r_s = torch.randn((A0, rnd_dim), device=device) # shape = [B_s * N_s, rnd_dim]
            # sp_s.reshape(-1, B_t * N_t).transpose(-1, -2) shape = [B_t * N_t, B_s * N_s]; sp_s[i,j] is the i-th node's similarities with respect to all the nodes in target graphs
            r_t = b_s.reshape(A0, -1).transpose(0, 1) @ r_s # shape = [B_t * N_t, rnd_dim]; contains zero-entries caused by padding
            r_t = r_t.reshape(B_t, N_t, rnd_dim)[mask[0]] # shape = [B_t * N_t, rnd_dim]
            o_s = self.psi_2(r_s, edges_s, feat_edge_s) # shape = [B_s * N_s, rnd_dim]
            o_t = self.psi_2(r_t, edges_t, feat_edge_t) # shape = [B_t * N_t, rnd_dim]
            sim = cosine_similarity_nbyn(o_s, o_t)
            b_sim, _ = split_sim_batches(sim, g_pos_t) # shape = [B_s * N_s, B_t, N_t]
            b_s_hat += b_sim
            b_s = masked_softmax(b_s_hat, mask, dim=-1) # shape = [B_s * N_s, B_t, N_t]
        return b_s_hat


    def batch_forward(self, h_s, edges_s, feat_edge_s, batch_s, h_t, edges_t, feat_edge_t, batch_t):
        '''
        RandDGMC for a batch of graph pairs
        h_s: shape = [B * N_s, D]
        h_t: shape = [B * N_t, D]
        '''
        h_s = F.normalize(h_s, p=2, dim=-1)
        h_t = F.normalize(h_t, p=2, dim=-1)
        h_s, s_mask = to_dense_batch(h_s, batch_s, fill_value=0) # shape = [B, N_s, C_out]; shape = [B, N_s]
        h_t, t_mask = to_dense_batch(h_t, batch_t, fill_value=0) # shape = [B, N_t, C_out]; shape = [B, N_t]
        rnd_dim = self.rnd_dim
        (B, N_s, C_out), N_t = h_s.shape, h_t.shape[1]
        S_hat = h_s @ h_t.transpose(-1, -2)  # [B, N_s, N_t]

        # Scaling, so that the initial assignment is as important as the refinement phase
        S_hat *= self.num_steps
        S_mask = s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t) # shape = [B, N_s, N_t]
        S = masked_softmax(S_hat, S_mask, dim=-1) # shape = [B, N_s, N_t]

        for _ in range(self.num_steps):
            r_s = torch.randn((B, N_s, rnd_dim), dtype=h_s.dtype,
                                device=h_s.device)
            r_t = S.transpose(-1, -2) @ r_s
            r_s, r_t = to_sparse(r_s, s_mask), to_sparse(r_t, t_mask) # shape = [B * N_s, rnd_dim]
            o_s = self.psi_2(r_s, edges_s, feat_edge_s)
            o_t = self.psi_2(r_t, edges_t, feat_edge_t)
            o_s = F.normalize(o_s, p=2, dim=-1) # shape = [B * N_s, rnd_dim]
            o_t = F.normalize(o_t, p=2, dim=-1) # shape = [B * N_t, rnd_dim]
            o_s, o_t = to_dense(o_s, s_mask), to_dense(o_t, t_mask) # shape = [B, N_s, rnd_dim], [B, N_t, rnd_dim]
            S_hat += o_s @ o_t.transpose(-1, -2)
            S = masked_softmax(S_hat, S_mask, dim=-1) # shape = [B, N_s, N_t]

        S_hat = S_hat[s_mask] # shape = [B * N_s, N_t]
        return S_hat

    def forward(self, S_hat, edges_s, edges_t, feat_edge_s=None, feat_edge_t=None):
        '''
        RandDGMC for two graphs, a sparse implementation
        '''
        device = S_hat.device
        rnd_dim = self.rnd_dim
        N0, N1 = S_hat.shape
        S_hat, knn_idx0 = torch.topk(S_hat, self.k + 1, dim=1)  # shape = [N0, nega_num + 1]
        
        # Scaling, so that the initial assignment is as important as the refinement phase
        S_hat *= self.num_steps
        S = F.softmax(S_hat, dim=1)
        
        # cache the edges
        h = torch.arange(N0, device=device).reshape(-1, 1).expand(N0, knn_idx0.shape[-1])
        edges = torch.stack((h, knn_idx0), dim=-1)
        edges = edges.reshape(-1, 2).T
        if self.weight_free:
            eye = torch.eye(N0, device=device) 
        
        '''refinement: a sparse implementation'''
        for _ in range(self.num_steps):
            '''construct sparse matrix for multiplication'''
            sp_s_hat = torch.sparse.FloatTensor(edges, S.flatten(), size=(N0, N1))
            if self.weight_free:
                r_s = eye
            else:
                r_s = torch.randn((N0, rnd_dim), device=device)  # shape [N0, rnd_dim]
            r_t = torch.sparse.mm(sp_s_hat.transpose(0, 1), r_s)  # shape [N1, rnd_dim]
            o_s = self.psi_2(r_s, edges_s, feat_edge_s)  # shape [N0, rnd_dim]
            o_t = self.psi_2(r_t, edges_t, feat_edge_t)  # shape [N1, rnd_dim]
            o_s = o_s.unsqueeze(-2)  # shape [N0, 1, rnd_dim]
            o_t = o_t[knn_idx0]  # shape [N0, nega_num + 1, rnd_dim]
            o_s = F.normalize(o_s, dim=-1, p=2)
            o_t = F.normalize(o_t, dim=-1, p=2)
            sim = torch.sum(o_s * o_t, dim=-1) # shape = [N0, nega_num + 1]

            S_hat = S_hat + sim
            S = F.softmax(S_hat, dim=1)
        
        S = torch.sparse.FloatTensor(edges, S.flatten(), size=(N0, N1))
        return S


    def sp_forward(self, S_hat, knn_idx0, edges_s, edges_t, N0, N1):
        '''
        RandDGMC for two graphs, a sparse implementation
        '''
        device = S_hat.device
        rnd_dim = self.rnd_dim
        
        # Scaling, so that the initial assignment is as important as the refinement phase
        S_hat *= self.num_steps
        S = F.softmax(S_hat, dim=1)
        
        # cache the edges
        h = torch.arange(N0, device=device).reshape(-1, 1).expand(N0, knn_idx0.shape[-1])
        edges = torch.stack((h, knn_idx0), dim=-1)
        edges = edges.reshape(-1, 2).T
        if self.weight_free:
            eye = torch.eye(N0, device=device) 
        
        '''refinement: a sparse implementation'''
        for _ in range(self.num_steps):
            '''construct sparse matrix for multiplication'''
            sp_s_hat = torch.sparse.FloatTensor(edges, S.flatten(), size=(N0, N1))
            if self.weight_free:
                r_s = eye
            else:
                r_s = torch.randn((N0, rnd_dim), device=device)  # shape [N0, rnd_dim]
            r_t = torch.sparse.mm(sp_s_hat.transpose(0, 1), r_s)  # shape [N1, rnd_dim]
            o_s = self.psi_2(r_s, edges_s)  # shape [N0, rnd_dim]
            o_t = self.psi_2(r_t, edges_t)  # shape [N1, rnd_dim]
            o_s = F.normalize(o_s, dim=-1, p=2)
            o_t = F.normalize(o_t, dim=-1, p=2)
            o_s = o_s.unsqueeze(-2)  # shape [N0, 1, rnd_dim]
            o_t = o_t[knn_idx0]  # shape [N0, nega_num + 1, rnd_dim]
            sim = torch.bmm(o_t, o_s.transpose(-1, -2)).squeeze(-1)
            S_hat = S_hat + sim
            S = F.softmax(S_hat, dim=1)
            
            with torch.cuda.device(device):
                del sim, o_t, o_s
                torch.cuda.empty_cache()

        S = torch.sparse.FloatTensor(edges, S.flatten(), size=(N0, N1))
        return S


class Linear(nn.Module):
    '''
    My implemented Linear layer
    1. without bias
    2. initialize weight with normal distribution
    3. scale the weight so that the randomized version can preserve the norm of inputs
    '''
    def __init__(self, in_dim, out_dim, bias=False):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0, std=1)
        self.weight.data /= math.sqrt(self.weight.shape[0])
        if self.bias is not None:
            nn.init.normal_(self.bias, 0, 1)
            self.bias.data /= math.sqrt(self.weight.shape[0])

    def forward(self, inputs):
        outputs = F.linear(inputs, self.weight, self.bias)
        return outputs


class Identity(nn.Module):
    def __init__(self, bias):
        super(Identity, self).__init__()
        self.use_bias = bias

    def reset_parameters(self):
        if hasattr(self, 'bias'):
            nn.init.normal_(self.bias, 0, 1)
            self.bias.data /= math.sqrt(self.in_dim)

    def forward(self, inputs):
        if not self.use_bias:
            return inputs
        in_dim = inputs.shape[-1]
        if not hasattr(self, 'bias'):
            device = inputs.device
            self.in_dim = in_dim
            self.bias = nn.Parameter(torch.Tensor(in_dim))
            self.reset_parameters()
            self.bias.data = self.bias.data.to(device)
        assert in_dim == self.in_dim
        inputs = inputs + self.bias.reshape(1, -1)
        return inputs

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, act_func=F.relu, weight_free=False, bias=False):
        super(GCNLayer, self).__init__()
        self.weight_free = weight_free
        if not self.weight_free:
            self.linear = Linear(in_dim, out_dim, bias)
        else:
            self.identity = Identity(bias)
        self.act_func = act_func

    def reset_parameters(self):
        if not self.weight_free:
            self.linear.reset_parameters()
        else:
            self.identity.reset_parameters()

    def message_passing(self, feats_n, hs, ts, linear_layer):
        N_n = feats_n.shape[0]
        device = feats_n.device
        indice = torch.stack((hs, ts), dim=0)  # shape = [2, N]
        ones = torch.ones(indice.shape[1], device=device)
        adj = torch.sparse.FloatTensor(indice, ones, size=(N_n, N_n))
        D = torch.sparse.sum(adj, dim=1,).to_dense().reshape(-1, 1)
        sqrt_d_inv = D ** (-1/2)
        to_feats_n = sqrt_d_inv * torch.sparse.mm(adj, sqrt_d_inv * feats_n)  # shape = [N_n, out_dim]
        to_feats_n = linear_layer(to_feats_n)
        return to_feats_n

    def forward(self, feats_n, edges):
        '''
        :param edges: [2, N_e]
        :param feats_n: [N_n, n_dim]
        :return:
        '''
        hs, ts = edges
        if self.weight_free:
            message = self.message_passing(feats_n, hs, ts, self.identity)
        else:
            message = self.message_passing(feats_n, hs, ts, self.linear)
        to_feats_n = self.act_func(message)
        return to_feats_n

class SAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, weight_free=False, bias=False):
        super(SAGELayer, self).__init__()
        self.weight_free = weight_free
        if not self.weight_free:
            self.linear = Linear(in_dim, out_dim, bias)
        else:
            self.identity = Identity(bias)

    def reset_parameters(self):
        if not self.weight_free:
            self.linear.reset_parameters()
        else:
            self.identity.reset_parameters()

    def message_passing(self, feats_n, hs, ts, linear_layer):
        N_n = feats_n.shape[0]
        device = feats_n.device
        indice = torch.stack((hs, ts), dim=0)  # shape = [2, N]
        ones = torch.ones(indice.shape[1], device=device)
        adj = torch.sparse.FloatTensor(indice, ones, size=(N_n, N_n))
        to_feats_n = linear_layer(torch.sparse.mm(adj, feats_n))  # shape = [N_n, out_dim]
        # degree = torch.clamp(torch.sparse.sum(adj, dim=1).to_dense().reshape(-1, 1), min=1)
        # to_feats_n = to_feats_n / degree
        return to_feats_n

    def forward(self, feats_n, edges):
        '''
        :param edges: [2, N_e]
        :param feats_n: [N_n, n_dim]
        :return:
        '''
        hs, ts = edges
        if self.weight_free:
            message = self.message_passing(feats_n, hs, ts, self.identity)
        else:
            message = self.message_passing(feats_n, hs, ts, self.linear)
        to_feats_n = message
        return to_feats_n

    
class GraphSAGE(nn.Module):
    def __init__(self, dim_n, hid_dim, num_layer, act_func=lambda x:x, use_node_feature=True, weight_free=False, bias=False):
        super(GraphSAGE, self).__init__()
        self.in_dim = dim_n
        self.weight_free = weight_free
        self.layers = nn.ModuleList()
        self.act_func = act_func
        for i in range(num_layer):
            if i == 0:
                self.layers.append(SAGELayer(dim_n, hid_dim, weight_free=weight_free, bias=bias))
            else:
                self.layers.append(SAGELayer(hid_dim, hid_dim, weight_free=weight_free, bias=bias))
        self.use_node_feature = use_node_feature
        self.basic_tfgm = False
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, feats_n, edges, edge_attr=None):
        if self.basic_tfgm:
            feats_n_list = [F.normalize(feats_n, p=2, dim=-1)]
            for gnn in self.layers:
                feats_n = gnn(feats_n, edges) # shape = [\sum_i N_{g_i}, D]
                feats_n = F.normalize(feats_n, dim=-1, p=2)
                feats_n += feats_n_list[-1]
                feats_n_list.append(feats_n)
                feats_n = self.act_func(feats_n)
            return feats_n_list[-1]
        else:
            if self.use_node_feature:
                feats_n_list = [F.normalize(feats_n, p=2, dim=-1)]
            else:
                feats_n_list = []
            for gnn in self.layers:
                feats_n = gnn(feats_n, edges) # shape = [\sum_i N_{g_i}, D]
                feats_n = F.normalize(feats_n, dim=-1, p=2)
                feats_n_list.append(feats_n)
                feats_n = self.act_func(feats_n)

            feats_n = torch.cat(feats_n_list,dim=-1)
            return feats_n
    
    def pair_forward(self, feats_n0, edges0, feats_n1, edges1, seeds):
        '''
        seeds: shape = [S, 2]
        '''
        feats_n1[seeds[:, 1]] = feats_n0[seeds[:, 0]]

        if self.use_node_feature:
            feats_n_list0 = [F.normalize(feats_n0, p=2, dim=-1)]
            feats_n_list1 = [F.normalize(feats_n1, p=2, dim=-1)]
        else:
            feats_n_list0 = []
            feats_n_list1 = []
        
        for gnn in self.layers:
            feats_n0 = gnn(feats_n0, edges0) # shape = [\sum_i N_{g_i}, D]
            feats_n1 = gnn(feats_n1, edges1) # shape = [\sum_i N_{g_i}, D]    
            
            feats_n1[seeds[:, 1]] = feats_n0[seeds[:, 0]]

            feats_n0 = F.normalize(feats_n0, dim=-1, p=2)
            feats_n_list0.append(feats_n0)
            feats_n0 = self.act_func(feats_n0)
            feats_n1 = F.normalize(feats_n1, dim=-1, p=2)
            feats_n_list1.append(feats_n1)
            feats_n1 = self.act_func(feats_n1)

        feats_n0 = torch.cat(feats_n_list0, dim=-1)
        feats_n1 = torch.cat(feats_n_list1, dim=-1)
        return feats_n0, feats_n1
        

class GIN(nn.Module):
    def __init__(self, dim_n, hid_dim, num_layer, use_node_feature=True):
        super(GIN, self).__init__()
        self.in_dim = dim_n
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            if i == 0:
                gin = GINConv(nn.Sequential(Linear(dim_n, hid_dim, True), nn.ReLU(), Linear(hid_dim, hid_dim, True), nn.ReLU()))
            else:
                gin = GINConv(nn.Sequential(Linear(hid_dim, hid_dim, True), nn.ReLU(), Linear(hid_dim, hid_dim, True),nn.ReLU()))
            self.layers.append(gin)
        self.use_node_feature = use_node_feature

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
    
    def forward(self, feats_n, edges, edge_attr=None):
        if self.use_node_feature:
            feats_n_list = [F.normalize(feats_n, p=2, dim=-1)]
        else:
            feats_n_list = []
        for gnn in self.layers:
            feats_n = gnn(feats_n, edges) # shape = [\sum_i N_{g_i}, D]
            feats_n = F.normalize(feats_n, dim=-1, p=2)
            feats_n_list.append(feats_n)
        feats_n = torch.cat(feats_n_list,dim=-1)
        return feats_n