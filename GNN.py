import torch
import torch.nn as nn
from dgl import graph
from torch.nn import init
from torch.autograd import Variable

from Utils.graph_builder import build_graph
from Utils.loss_utils import l2_reg_loss
from lagcl_encoder import LAGCLEncoder
from Mul_Par import args

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class myModel(nn.Module):
    def __init__(self, g:dict, userNum, itemNum, behavior, behavior_mats):
        super(myModel, self).__init__()  
        
        self.userNum = userNum
        self.itemNum = itemNum
        self.behavior = behavior
        self.behavior_mats = behavior_mats
        
        self.embedding_dict = self.init_embedding() 
        self.weight_dict = self.init_weight()
        self.gcn = GCN(g, self.userNum, self.itemNum, self.behavior, self.behavior_mats).to(device)


    def init_embedding(self):
        
        embedding_dict = {  
            'user_embedding': None,
            'item_embedding': None,
            'user_embeddings': None,
            'item_embeddings': None,
        }
        return embedding_dict

    def init_weight(self):  
        initializer = nn.init.xavier_uniform_
        
        weight_dict = nn.ParameterDict({
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_self_attention_cat': nn.Parameter(initializer(torch.empty([args.head_num*args.hidden_dim, args.hidden_dim]))),
            'alpha': nn.Parameter(torch.ones(2)),
        })      
        return weight_dict  


    def forward(self):
        user_embed, item_embed, user_embeds, item_embeds = self.gcn()

        return user_embed, item_embed, user_embeds, item_embeds 

    def para_dict_to_tenser(self, para_dict):
        tensors = []
        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)

        return tensors.float()

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_parameters(), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_parameters()(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  
                    self.set_param(self, name, param)

class GCN(nn.Module):
    def __init__(self, g,  userNum, itemNum, behavior, behavior_mats):
        super(GCN, self).__init__()  
        self.userNum = userNum
        self.itemNum = itemNum
        self.hidden_dim = args.hidden_dim

        self.behavior = behavior
        self.behavior_mats = behavior_mats
        
        self.alpha, self.i_concatenation_w, self.u_concatenation_w, self.i_input_w, self.u_input_w = self.init_weight()
        
        self.sigmoid = torch.nn.Sigmoid()
        self.act = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(args.drop_rate)

        self.gnn_layer = eval(args.gnn_layer)
        self.layers = nn.ModuleList()
        for i in range(0, len(self.gnn_layer)):  
            self.layers.append(GCNLayer(args.hidden_dim, args.hidden_dim, self.userNum, self.itemNum, self.behavior, self.behavior_mats)).to(device)

        self.noise_eps = args.noise_eps
        self.l2_reg = args.l2_reg
        self.use_relation_rf = args.use_relation_rf
        self.annealing_type = args.annealing_type
        self.lambda_gp = args.lambda_gp
        self.hidden_dim = args.hidden_dim

        graph,_,_,_,_ = build_graph(g['train_u'], g['train_i'], g['test_u'], g['test_i'])

        self.graph = graph.to(device)
        self.node_ids = self.graph.ndata['node_feature']
        self.node_types = self.graph.ndata['node_type']
        self.is_user_node = self.node_types == 0
        self.global_head_mask = self.graph.ndata[
                                    'node_degree'] > args.tail_k_threshold
        self.user_is_head_mask = self.global_head_mask[self.is_user_node]
        self.item_is_head_mask = self.global_head_mask[~self.is_user_node]
        self.encoder = LAGCLEncoder().cuda()

        self.user_embedding, self.item_embedding = self.init_embedding()

    def init_embedding(self):
        user_embedding = torch.nn.Embedding(self.userNum, args.hidden_dim).to(device)
        item_embedding = torch.nn.Embedding(self.itemNum, args.hidden_dim).to(device)
        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)

        return user_embedding, user_embedding

    def init_weight(self):
        alpha = nn.Parameter(torch.ones(2))

        i_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim))
        u_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim))
        i_input_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        u_input_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        init.xavier_uniform_(i_concatenation_w)
        init.xavier_uniform_(u_concatenation_w)
        init.xavier_uniform_(i_input_w)
        init.xavier_uniform_(u_input_w)

        return alpha, i_concatenation_w, u_concatenation_w, i_input_w, u_input_w

    def forward(self, user_embedding_input=None, item_embedding_input=None):

        all_user_embeddings = []
        all_item_embeddings = []
        all_user_embeddingss = []
        all_item_embeddingss = []

        user_embedding = self.user_embedding.weight
        item_embedding = self.item_embedding.weight

        node_initial = torch.cat([user_embedding, item_embedding], dim=0)
        x = node_initial[self.node_ids]
        node_embeddings, other_embs_dict = self.encoder(self.graph, x)
        rec_user_emb, rec_item_emb = self.split_user_item_embs(node_embeddings)

        for i, layer in enumerate(self.layers):
            user_embedding, item_embedding, user_embeddings, item_embeddings = layer(rec_user_emb, rec_item_emb)

            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)
            all_user_embeddingss.append(user_embeddings)
            all_item_embeddingss.append(item_embeddings)


        user_embedding = torch.cat(all_user_embeddings, dim=1)
        item_embedding = torch.cat(all_item_embeddings, dim=1)
        user_embeddings = torch.cat(all_user_embeddingss, dim=2)
        item_embeddings = torch.cat(all_item_embeddingss, dim=2)


        user_embedding = torch.matmul(user_embedding , self.u_concatenation_w)
        item_embedding = torch.matmul(item_embedding , self.i_concatenation_w)
        user_embeddings = torch.matmul(user_embeddings , self.u_concatenation_w)
        item_embeddings = torch.matmul(item_embeddings , self.i_concatenation_w)
            

        return user_embedding, item_embedding, user_embeddings, item_embeddings

    def split_user_item_embs(self, node_embs):

        user_node_ids = self.node_ids[self.is_user_node]
        item_node_ids = self.node_ids[~self.is_user_node]

        user_embs = torch.zeros(self.userNum,
                                self.hidden_dim,
                                device=device)
        item_embs = torch.zeros(self.itemNum,
                                self.hidden_dim,
                                device=device)

        user_embs[user_node_ids] = node_embs[self.is_user_node]
        item_embs[item_node_ids - self.userNum] = node_embs[~self.is_user_node]

        return user_embs, item_embs

    def process_one_subgraph(self):

        node_initial = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        x = node_initial[self.node_ids]

        node_embeddings, other_embs_dict = self.encoder(self.graph, x)

        rec_user_emb, rec_item_emb = self.split_user_item_embs(node_embeddings)

        emb_h, emb_nt, emb_t = other_embs_dict[
            'head_true_drop_false'], other_embs_dict[
                'head_true_drop_true'], other_embs_dict['head_false_drop_true']
        rec_user_emb_h, rec_item_emb_h = self.split_user_item_embs(emb_h)
        rec_user_emb_t, rec_item_emb_t = self.split_user_item_embs(emb_t)
        rec_u_emb_nt, rec_i_emb_nt = self.split_user_item_embs(emb_nt)

        support_h, support_t = other_embs_dict['support_h'], other_embs_dict[
            'support_t']

        return (rec_user_emb, rec_item_emb), (support_h, support_t), {

            'user_head_true_drop_false': rec_user_emb_h,
            'item_head_true_drop_false': rec_item_emb_h,

            'user_head_true_drop_true': rec_u_emb_nt,
            'item_head_true_drop_true': rec_i_emb_nt,

            'user_head_false_drop_true': rec_user_emb_t,
            'item_head_false_drop_true': rec_item_emb_t,
        }

    def lt_loss(self, uid, pid, nid):
        user_idx = uid
        pos_idx = pid.clone().detach()
        neg_idx = nid.clone().detach()

        (rec_user_emb, rec_item_emb), (
            support_h,
            support_t), other_embs_dict = self.process_one_subgraph()
        u_emb_h, u_emb_t, u_emb_nt = other_embs_dict[
            'user_head_true_drop_false'], other_embs_dict[
                'user_head_false_drop_true'], other_embs_dict[
                    'user_head_true_drop_true']
        i_emb_h, i_emb_t, i_emb_nt = other_embs_dict[
            'item_head_true_drop_false'], other_embs_dict[
                'item_head_false_drop_true'], other_embs_dict[
                    'item_head_true_drop_true']

        user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

        l2_loss = l2_reg_loss(self.l2_reg, user_emb, pos_item_emb)


        L_kl_corr_user = L_kl_corr_item = torch.tensor(0.0)
        L_kl_corr_user = torch.nn.functional.kl_div(
            u_emb_t[user_idx].log_softmax(dim=-1),
            u_emb_h[user_idx].softmax(dim=-1),
            reduction='batchmean',
            log_target=False) * self.encoder.kl_eps

        L_kl_corr = L_kl_corr_user + L_kl_corr_item


        m_regularizer = self.cal_m_regularizer_loss([i[self.is_user_node]
            for i in (support_t if self.use_relation_rf else support_h)
        ], self.global_head_mask[self.is_user_node]) * self.encoder.m_h_eps

        loss =  l2_loss + L_kl_corr  + m_regularizer
        return loss

    def cal_m_regularizer_loss(self, support_out, head_mask):
        m_reg_loss = 0
        for m in support_out:
            m_reg_loss += torch.mean(torch.norm(m[head_mask], dim=1))
        return m_reg_loss

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, userNum, itemNum, behavior, behavior_mats):
        super(GCNLayer, self).__init__()

        self.behavior = behavior
        self.behavior_mats = behavior_mats

        self.userNum = userNum
        self.itemNum = itemNum

        self.act = torch.nn.Sigmoid()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.ii_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)

    def forward(self, user_embedding, item_embedding):

        user_embedding_list = [None]*len(self.behavior)
        item_embedding_list = [None]*len(self.behavior)

        for i in range(len(self.behavior)):

            user_embedding_list[i] = torch.spmm(self.behavior_mats[i]['A'], item_embedding)

            item_embedding_list[i] = torch.spmm(self.behavior_mats[i]['AT'], user_embedding)


        user_embeddings = torch.stack(user_embedding_list, dim=0) 
        item_embeddings = torch.stack(item_embedding_list, dim=0)

        user_embedding = self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w))
        item_embedding = self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w))

        user_embeddings = self.act(torch.matmul(user_embeddings, self.u_w))
        item_embeddings = self.act(torch.matmul(item_embeddings, self.i_w))

        return user_embedding, item_embedding, user_embeddings, item_embeddings             


