import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl
from torch.nn.parameter import Parameter
from Utils.sparse_mat_interface import GraphSparseMatInterface
from Utils.spmm_utils import SpecialSpmm
from Mul_Par import args


def eliminate_zeros(x):
    indices = x.coalesce().indices()
    values = x.coalesce().values()

    mask = values.nonzero()
    nv = values.index_select(0, mask.view(-1))
    ni = indices.index_select(1, mask.view(-1))
    return torch.sparse.FloatTensor(ni, nv, x.shape)


class Discriminator(nn.Module):

    def __init__(self, in_features):
        super(Discriminator, self).__init__()

        self.d = nn.Linear(in_features, in_features, bias=True)
        self.wd = nn.Linear(in_features, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ft):
        ft = F.elu(ft)
        ft = F.dropout(ft, 0.5, training=self.training)

        fc = F.elu(self.d(ft))
        prob = self.wd(fc)

        return prob


class Relation(nn.Module):

    def __init__(self, in_features, ablation):
        super(Relation, self).__init__()

        self.gamma_1 = nn.Linear(in_features, in_features, bias=False)
        self.gamma_2 = nn.Linear(in_features, in_features, bias=False)

        self.beta_1 = nn.Linear(in_features, in_features, bias=False)
        self.beta_2 = nn.Linear(in_features, in_features, bias=False)

        self.r = Parameter(torch.FloatTensor(1, in_features))

        self.elu = nn.ELU()
        self.lrelu = nn.LeakyReLU(0.2)

        self.sigmoid = nn.Sigmoid()
        self.reset_parameter()
        self.ablation = ablation

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.r.size(1))
        self.r.data.uniform_(-stdv, stdv)

    def forward(self, ft, neighbor):

        ft = ft.detach()
        neighbor = neighbor.detach()
        if self.ablation == 3:
            m = ft + self.r - neighbor
        else:
            gamma = self.gamma_1(ft) + self.gamma_2(neighbor)
            gamma = self.lrelu(gamma) + 1.0
            beta = self.beta_1(ft) + self.beta_2(neighbor)
            beta = self.lrelu(beta)
            r_v = gamma * self.r + beta
            m = ft + r_v - neighbor
        return m


class RelationF(nn.Module):

    def __init__(self, nfeat):
        super(RelationF, self).__init__()
        self.fc1 = nn.Linear(nfeat * 4, nfeat)
        self.fc2 = nn.Linear(nfeat, nfeat)

    def forward(self, x, neighbor, masked_neighbor):
        x = x.detach().clone()
        neighbor = neighbor.detach().clone()
        masked_neighbor = masked_neighbor.detach().clone()

        ngb_seq = torch.stack(
            [x, neighbor, neighbor * x, (neighbor + x) / 2.0], dim=1)

        missing_info = self.fc1(ngb_seq.reshape(len(ngb_seq), -1))
        missing_info = F.relu(missing_info, inplace=False)
        missing_info = self.fc2(missing_info)
        support_out = missing_info - masked_neighbor
        return missing_info, support_out


class LightTailGCN(nn.Module):

    def __init__(self, nfeat, global_r=None, use_relation_rf=False):
        super(LightTailGCN, self).__init__()
        self.use_relation_rf = use_relation_rf
        if self.use_relation_rf:
            self.trans_relation = RelationF(nfeat)
        else:
            self.trans_relation = Relation(nfeat, 0)
        if global_r is not None:
            self.trans_relation = global_r
        self.special_spmm = SpecialSpmm()

    def forward(self, x, adj, adj_norm, adj_node_degree, adj_with_loop,
                adj_with_loop_norm, adj_with_loop_norm_plus_1, head, res_adj,
                res_adj_norm):

        neighbor = self.special_spmm(adj_norm, x)

        if self.use_relation_rf:
            masked_neighbor = torch.sparse.mm(res_adj_norm, x)
            missing_info, output = self.trans_relation(x, neighbor,
                                                       masked_neighbor)
        else:
            missing_info = self.trans_relation(x, neighbor)
            output = missing_info

        if head:
            h_k = self.special_spmm(adj_with_loop_norm, x)
        else:
            h_s = missing_info.clone()

            denominator = (adj_node_degree + 2).reshape(-1, 1).float()
            denominator = torch.clamp(denominator, min=1e-8)
            term1 = self.special_spmm(adj_with_loop_norm_plus_1, x)
            term2 = h_s / denominator

            h_k = term1 + term2
        return h_k, output


class LAGCLEncoder(nn.Module):

    def __init__(self):

        super().__init__()
        self.neighbor_norm_type = args.neighbor_norm_type
        self.tail_k_threshold = args.tail_k_threshold
        self.m_h_eps = args.m_h_eps
        self.kl_eps = args.kl_eps
        self.noise_eps = args.noise_eps
        self.agg_function = args.agg_function
        self.use_relation_rf = args.use_relation_rf
        self.agg_w_exp_scale = args.agg_w_exp_scale

        self.lin = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.hidden_dim = args.hidden_dim

        self.n_layers = args.layer_gnn
        self.x_weights = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)

        self.rel_layers = nn.ModuleList([
            LightTailGCN(args.hidden_dim,
                         global_r=None,
                         use_relation_rf=self.use_relation_rf)
            for _ in range(self.n_layers)
        ])

    def encode(self, subgraph: dgl.DGLGraph, x: torch.Tensor):

        ego_embeddings = self.lin(x)
        edge_index = torch.stack(subgraph.edges())
        node_degrees = subgraph.ndata['node_degree']
        node_types = subgraph.ndata['node_type']
        drop_node_mask = node_types == 0
        node_is_head_mask = node_degrees > self.tail_k_threshold
        node_is_head_mask[
            ~drop_node_mask] = True

        edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        n_nodes = len(ego_embeddings)
        n_edges = len(edge_index[0])

        adj = torch.sparse.FloatTensor(edge_index,
                                       torch.ones(n_edges,
                                                  device=edge_index.device),
                                       size=(n_nodes, n_nodes)).coalesce()
        adj_norm, adj_node_degree = GraphSparseMatInterface.normalize_graph_mat(
            adj, norm=self.neighbor_norm_type, return_node_degree=True)
        adj_with_loop = adj + torch.sparse.FloatTensor(
            torch.arange(len(adj)).repeat(2, 1), torch.ones(len(adj))).to(
                adj.device)
        adj_with_loop_norm = GraphSparseMatInterface.normalize_graph_mat(
            adj_with_loop)
        adj_with_loop_norm_plus_1 = GraphSparseMatInterface.normalize_graph_mat(
            adj_with_loop, add_virtual_node_num=1)
        res_adj = torch.sparse.FloatTensor(
            torch.tensor([[0], [0]], dtype=torch.int64), torch.tensor([0.0]),
            adj.shape).to(adj.device)
        res_adj_norm = GraphSparseMatInterface.normalize_graph_mat(res_adj)

        enable_lagcl = self.tail_k_threshold >= 0

        emb_h, support_h, _ = self.gather_pre(ego_embeddings,
                                              adj,
                                              adj_norm,
                                              adj_node_degree,
                                              adj_with_loop,
                                              adj_with_loop_norm,
                                              adj_with_loop_norm_plus_1,
                                              res_adj,
                                              res_adj_norm,
                                              head=True,
                                              use_auto_drop=False,
                                              drop_node_mask=None,
                                              node_is_head_mask=None,
                                              add_other_status=False)

        if enable_lagcl:
            emb_t, support_t, emb_nt = self.gather_pre(
                ego_embeddings,
                adj,
                adj_norm,
                adj_node_degree,
                adj_with_loop,
                adj_with_loop_norm,
                adj_with_loop_norm_plus_1,
                res_adj,
                res_adj_norm,
                head=False,
                use_auto_drop=True if self.training else False,
                drop_node_mask=drop_node_mask,
                node_is_head_mask=node_is_head_mask,
                add_other_status=True)


            node_emb = emb_h * node_is_head_mask.long().reshape(
                -1, 1) + emb_t * (1 - node_is_head_mask.long().reshape(-1, 1))
        else:
            node_emb = emb_h
            emb_nt = emb_t = emb_h
            support_t = support_h

        other_embs_dict = {

            'head_true_drop_false': emb_h,
            'head_true_drop_true': emb_nt,
            'head_false_drop_true': emb_t,
            'support_h': support_h,
            'support_t': support_t,
        }

        return node_emb, other_embs_dict

    def gather_pre(self,
                   ego_embeddings,
                   adj,
                   adj_norm,
                   adj_node_degree,
                   adj_with_loop,
                   adj_with_loop_norm,
                   adj_with_loop_norm_plus_1,
                   res_adj,
                   res_adj_norm,
                   head,
                   use_auto_drop,
                   drop_node_mask,
                   node_is_head_mask,
                   add_other_status=False):

        tail_k_threshold = self.tail_k_threshold
        assert tail_k_threshold != 0
        if use_auto_drop and tail_k_threshold > 0:
            indices = adj.indices()
            node_need_drop = drop_node_mask[indices[0]]
            indices = indices.t()[node_need_drop].t()
            if self.agg_function == 0:

                ego_norm = torch.max(
                    ego_embeddings.norm(dim=1)[:, None],
                    torch.zeros(
                        len(ego_embeddings), 1, device=ego_embeddings.device) +
                    1e-8)
                normd_emb = ego_embeddings / ego_norm

                agg_w = (self.x_weights.weight[0] * normd_emb[indices[0]] *
                         normd_emb[indices[1]]).sum(dim=1)

                agg_w = torch.nn.Softsign()(agg_w)
                agg_w = torch.nn.Softsign()(torch.exp(agg_w *
                                                      self.agg_w_exp_scale))
            else:

                sims = F.cosine_similarity(ego_embeddings[indices[0]],
                                           ego_embeddings[indices[1]])
                sims = torch.nn.Softsign()(torch.exp(sims *
                                                     self.agg_w_exp_scale))
                agg_w = sims

            head_to_tail_sample_type = 'top-k'
            if head_to_tail_sample_type == 'top-k':
                drop_node_is_head_mask = node_is_head_mask[drop_node_mask]

                k = {i: i for i in range(1, tail_k_threshold + 1)}

                node_type = torch.randint(1,
                                          tail_k_threshold + 1,
                                          (len(ego_embeddings), ),
                                          device=indices.device)

                node_type[drop_node_mask][
                    ~drop_node_is_head_mask] = tail_k_threshold
                edge_type = node_type[indices[0]]
                data_dict = {}
                edata = {}
                for edge_type_idx in k.keys():
                    select_mask = edge_type == edge_type_idx
                    nen_type = ('node', edge_type_idx, 'node')
                    data_dict[nen_type] = (indices[0][select_mask],
                                           indices[1][select_mask])
                    edata[nen_type] = agg_w[select_mask]
                g = dgl.heterograph(data_dict)
                g.edata['weight'] = edata

                sampled_g = dgl.sampling.sample_neighbors(
                    g,
                    nodes=g.nodes(),
                    fanout=k,
                    edge_dir='out',
                    prob='weight',
                    output_device=g.device)
                all_edges = []
                all_agg_w = []
                for etype in k.keys():
                    all_edges.append(torch.stack(sampled_g.edges(etype=etype)))
                    all_agg_w.append(sampled_g.edata['weight'][('node', etype,
                                                                'node')])
                all_edges = torch.cat(all_edges, dim=1)
                all_agg_w = torch.cat(all_agg_w)
            elif head_to_tail_sample_type == 'mantail-k':
                g = dgl.graph((indices[0], indices[1]))
                g.edata['weight'] = agg_w
                sampled_g = dgl.sampling.sample_neighbors(
                    g,
                    nodes=g.nodes(),
                    fanout=tail_k_threshold,
                    edge_dir='out',
                    prob='weight')
                all_edges = sampled_g.edges()
                all_agg_w = sampled_g.edata['weight']


            tail_indices = torch.stack([
                torch.cat([all_edges[0], all_edges[1]]),
                torch.cat([all_edges[1], all_edges[0]])
            ])
            tail_values = torch.cat([all_agg_w, all_agg_w])

            tail_adj = torch.sparse.FloatTensor(
                tail_indices, tail_values, adj.shape).coalesce().to(adj.device)
            tail_adj_norm, tail_adj_node_degree = GraphSparseMatInterface.normalize_graph_mat(
                tail_adj,
                norm=self.neighbor_norm_type,
                return_node_degree=True)
            tail_adj_with_loop = tail_adj + torch.sparse.FloatTensor(
                torch.arange(len(tail_adj)).repeat(2, 1),
                torch.ones(len(tail_adj))).to(adj.device)
            tail_adj_with_loop_norm = GraphSparseMatInterface.normalize_graph_mat(
                tail_adj_with_loop)
            tail_adj_with_loop_norm_plus_1 = GraphSparseMatInterface.normalize_graph_mat(
                tail_adj_with_loop, add_virtual_node_num=1)
            tail_res_adj = eliminate_zeros(adj - tail_adj)
            tail_res_adj_norm = GraphSparseMatInterface.normalize_graph_mat(
                tail_res_adj)

            adj, adj_norm, adj_node_degree = tail_adj, tail_adj_norm, tail_adj_node_degree
            adj_with_loop, adj_with_loop_norm = tail_adj_with_loop, tail_adj_with_loop_norm
            adj_with_loop_norm_plus_1 = tail_adj_with_loop_norm_plus_1
            res_adj, res_adj_norm = tail_res_adj, tail_res_adj_norm

        all_status_embeddings = {True: [], False: []}
        all_status_support_outs = {True: [], False: []}
        ego_embeddings1 = ego_embeddings2 = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings1, output1 = self.rel_layers[k](
                ego_embeddings1, adj, adj_norm, adj_node_degree, adj_with_loop,
                adj_with_loop_norm, adj_with_loop_norm_plus_1, head, res_adj,
                res_adj_norm)
            if add_other_status:
                ego_embeddings2, output2 = self.rel_layers[k](
                    ego_embeddings2, adj, adj_norm, adj_node_degree,
                    adj_with_loop, adj_with_loop_norm,
                    adj_with_loop_norm_plus_1, not head, res_adj, res_adj_norm)
            else:
                ego_embeddings2, output2 = ego_embeddings1, output1
            all_status_embeddings[head].append(ego_embeddings1)
            all_status_embeddings[not head].append(ego_embeddings2)
            all_status_support_outs[head].append(output1)
            all_status_support_outs[not head].append(output2)

        def agg_all_layers_out(all_embeddings, backbone_name='lightgcn'):
            if backbone_name == 'lightgcn':
                all_embeddings = torch.stack(all_embeddings, dim=1)
                all_embeddings = torch.mean(all_embeddings, dim=1)
            elif backbone_name == 'gcn':
                all_embeddings = all_embeddings[-1]
            return all_embeddings

        all_embeddings = agg_all_layers_out(all_status_embeddings[head])
        all_embeddings_other = agg_all_layers_out(
            all_status_embeddings[not head])

        return all_embeddings, all_status_support_outs[
            head], all_embeddings_other

    def forward(
        self,
        subgraph: dgl.DGLGraph,
        x: torch.Tensor,
    ):
        return self.encode(subgraph, x)