import os

import numpy as np
import wandb

import Kg_Par
import torch

from Utils.graph_builder import build_graph
from Utils.loss_utils import l2_reg_loss
from lagcl_encoder import LAGCLEncoder
from Kg_Data import BasicDataset
from torch import nn
from GAT import GAT
from Kg_Par import args, device
import torch.nn.functional as F
from utils import _L2_loss_mean

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError

class Model(BasicModel):
    def __init__(self,config: dict, dataset: BasicDataset, kg_dataset):

        super(Model, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.kg_dataset = kg_dataset
        self.__init_weight()

        self.gat = GAT(self.latent_dim, self.latent_dim, dropout=0.4, alpha=0.2).train()

        self.noise_eps = args.noise_eps
        self.l2_reg = args.l2_reg
        self.use_relation_rf = args.use_relation_rf
        self.annealing_type = args.annealing_type
        self.lambda_gp = args.lambda_gp
        self.hidden_dim = args.hidden_dim

        graph, _, _, _, _ = build_graph(self.dataset.g['train_u'], self.dataset.g['train_i'], self.dataset.g['test_u'], self.dataset.g['test_i'])
        self.userNum = self.dataset.g["userNum"]
        self.itemNum = self.dataset.g["itemNum"]
        self.graph = graph.to(device)
        self.node_ids = self.graph.ndata['node_feature']
        self.node_types = self.graph.ndata['node_type']
        self.is_user_node = self.node_types == 0
        self.global_head_mask = self.graph.ndata[
                                    'node_degree'] > args.tail_k_threshold
        self.user_is_head_mask = self.global_head_mask[self.is_user_node]
        self.item_is_head_mask = self.global_head_mask[~self.is_user_node]
        self.encoder = LAGCLEncoder().cuda()


    def __init_weight(self):

        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.num_entities = self.kg_dataset.entity_count
        self.num_relations = self.kg_dataset.relation_count
        print("user:{}, item:{}, entity:{}".format(self.num_users,
                                                   self.num_items,
                                                   self.num_entities))
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        self.embedding_user = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.emb_item_list = nn.ModuleList([
            torch.nn.Embedding(self.num_items, self.latent_dim),
            torch.nn.Embedding(self.num_items, self.latent_dim)
        ])
        self.emb_entity_list = nn.ModuleList([
            nn.Embedding(self.num_entities + 1, self.latent_dim),
            nn.Embedding(self.num_entities + 1, self.latent_dim)
        ])
        self.emb_relation_list = nn.ModuleList([
            nn.Embedding(self.num_relations + 1, self.latent_dim),
            nn.Embedding(self.num_relations + 1, self.latent_dim)
        ])

        for i in range(2):
            nn.init.normal_(self.emb_item_list[i].weight, std=0.1)
            nn.init.normal_(self.emb_entity_list[i].weight, std=0.1)
            nn.init.normal_(self.emb_relation_list[i].weight, std=0.1)

        self.transR_W = nn.Parameter(torch.Tensor(self.num_relations + 1, self.latent_dim, self.latent_dim))
        self.TATEC_W = nn.Parameter(torch.Tensor(self.num_relations + 1, self.latent_dim, self.latent_dim))

        nn.init.xavier_uniform_(self.transR_W, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.TATEC_W, gain=nn.init.calculate_gain('relu'))

        self.W_R = nn.Parameter(
            torch.Tensor(self.num_relations, self.latent_dim, self.latent_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))
        nn.init.normal_(self.embedding_user.weight, std=0.1)

        self.co_user_score = nn.Linear(self.latent_dim, 1)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(
            self.num_items)

    def cal_item_embedding_from_kg(self, kg: dict = None, index=0):
        if kg is None:
            kg = self.kg_dict

        return self.cal_item_embedding_rgat(kg, index)

    def cal_item_embedding_rgat(self, kg: dict, index):
        user_embs = self.embedding_user.weight

        item_embs = self.emb_item_list[index](
            torch.IntTensor(list(kg.keys())).to(
                Kg_Par.device))

        item_entities = torch.stack(list(
            kg.values()))

        item_relations = torch.stack(list(self.item2relations.values()))

        entity_embs = self.emb_entity_list[index](
            item_entities)

        relation_embs = self.emb_relation_list[index](
            item_relations)

        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(item_entities), torch.zeros_like(item_entities)).float()

        node_initial = torch.cat([user_embs, item_embs], dim=0)
        x = node_initial[self.node_ids]
        node_embeddings, other_embs_dict = self.encoder(self.graph, x)
        user_embs, item_embs = self.split_user_item_embs(node_embeddings)

        return self.gat.forward_relation(item_embs, entity_embs, relation_embs, padding_mask)

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb_ego = self.embedding_user(users)

        pos_emb_ego0 = self.emb_item_list[0](pos_items)
        pos_emb_ego1 = self.emb_item_list[1](pos_items)
        neg_emb_ego0 = self.emb_item_list[0](neg_items)
        neg_emb_ego1 = self.emb_item_list[1](neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego0, pos_emb_ego1, neg_emb_ego0, neg_emb_ego1

    def getAll(self):
        all_users, all_items = self.computer()
        return all_users, all_items

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0, pos_emb_ego0,
         pos_emb_ego1, neg_emb_ego0, neg_emb_ego1) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) + pos_emb_ego0.norm(2).pow(2) + pos_emb_ego1.norm(2).pow(2)
                              + neg_emb_ego0.norm(2).pow(2) + neg_emb_ego1.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        return loss, reg_loss

    def computer(self):

        users_emb = self.embedding_user.weight

        items_emb0 = self.cal_item_embedding_from_kg(index=0)
        items_emb1 = self.cal_item_embedding_from_kg(index=1)

        items_emb = (items_emb0 + items_emb1) / 2


        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def view_computer_all(self, g_droped, index):
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(index=index)
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def calc_kg_loss_transR(self, h, r, pos_t, neg_t, index):
        r_embed = self.emb_relation_list[index](r).unsqueeze(-1)
        h_embed = self.emb_item_list[index](h).unsqueeze(-1)
        pos_t_embed = self.emb_entity_list[index](pos_t).unsqueeze(-1)
        neg_t_embed = self.emb_entity_list[index](neg_t).unsqueeze(-1)

        r_matrix = self.transR_W[r]
        h_embed = torch.matmul(r_matrix, h_embed)
        pos_t_embed = torch.matmul(r_matrix, pos_t_embed)
        neg_t_embed = torch.matmul(r_matrix, neg_t_embed)

        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2),
                              dim=1)
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2),
                              dim=1)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)
        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(
            r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed) + torch.norm(self.transR_W)

        loss = kg_loss + 1e-3 * l2_loss

        return loss

    def calc_kg_loss_TATEC(self, h, r, pos_t, neg_t, index):
        r_embed = self.emb_relation_list[index](r).unsqueeze(-1)
        h_embed = self.emb_item_list[index](h).unsqueeze(-1)
        pos_t_embed = self.emb_entity_list[index](pos_t).unsqueeze(-1)
        neg_t_embed = self.emb_entity_list[index](neg_t).unsqueeze(-1)

        r_matrix = self.TATEC_W[r]
        pos_mrt = torch.matmul(r_matrix, pos_t_embed)
        neg_mrt = torch.matmul(r_matrix, neg_t_embed)

        pos_hmrt = torch.sum(h_embed * pos_mrt, dim=1)
        neg_hmrt = torch.sum(h_embed * neg_mrt, dim=1)

        hr = torch.sum(h_embed * r_embed, dim=1)
        pos_tr = torch.sum(pos_t_embed * r_embed, dim=1)
        neg_tr = torch.sum(neg_t_embed * r_embed, dim=1)

        pos_ht = torch.sum(h_embed * pos_t_embed, dim=1)
        neg_ht = torch.sum(h_embed * neg_t_embed, dim=1)

        pos_score = pos_hmrt + hr + pos_tr + pos_ht
        neg_score = neg_hmrt + hr + neg_tr + neg_ht

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)
        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(
            r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed) + torch.norm(self.TATEC_W)

        loss = kg_loss + 1e-3 * l2_loss

        return loss


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
        item_embedding = (self.emb_item_list[0].weight + self.emb_item_list[1].weight) / 2
        node_initial = torch.cat([self.embedding_user.weight, item_embedding], dim=0)
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

        loss = l2_loss + L_kl_corr + m_regularizer
        return loss

    def cal_m_regularizer_loss(self, support_out, head_mask):
        m_reg_loss = 0
        for m in support_out:
            m_reg_loss += torch.mean(torch.norm(m[head_mask], dim=1))
        return m_reg_loss

    def train_disc(self, embed_model, disc_model, optimizer_D, disc_pseudo_real, optimizer_D_pseudo_real):
        disc_model.train()
        disc_pseudo_real.train()

        embed_model.train()

        for p in disc_model.parameters():
            p.requires_grad = True
        for p in disc_pseudo_real.parameters():
            p.requires_grad = True
        with torch.no_grad():
            _, _, other_embs_dict = embed_model.process_one_subgraph()
            u_emb_h, u_emb_t, u_emb_nt = other_embs_dict[
                'user_head_true_drop_false'], other_embs_dict[
                    'user_head_false_drop_true'], other_embs_dict[
                        'user_head_true_drop_true']
            i_emb_h, i_emb_t, i_emb_nt = other_embs_dict[
                'item_head_true_drop_false'], other_embs_dict[
                    'item_head_false_drop_true'], other_embs_dict[
                        'item_head_true_drop_true']

        all_head_emb_h = u_emb_h[self.user_is_head_mask]
        all_emb_t = u_emb_t
        all_emb_nt = u_emb_nt
        cell_mask = self.user_is_head_mask


        if True:
            if self.annealing_type == 0:

                noise_eps = self.noise_eps
            else:

                noise_eps = 0.0

            def exec_perturbed(x, noise_eps):
                random_noise = torch.rand_like(x).to(self.device)
                x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * noise_eps
                return x


            all_head_emb_h = exec_perturbed(all_head_emb_h,
                                            noise_eps=noise_eps)
            all_emb_t = exec_perturbed(all_emb_t, noise_eps=noise_eps)
            all_emb_nt = exec_perturbed(all_emb_nt, noise_eps=noise_eps)

        prob_h = disc_model(all_head_emb_h)
        prob_t = disc_model(all_emb_t)

        errorD = -prob_h.mean()
        errorG = prob_t.mean()

        def get_select_idx(max_value, select_num, strategy='uniform'):

            select_idx = None
            if strategy == 'uniform':
                select_idx = torch.randperm(max_value).repeat(
                    int(np.ceil(select_num / max_value)))[:select_num]
            elif strategy == 'random':
                select_idx = np.random.randint(0, max_value, select_num)
            return select_idx

        def calc_gradient_penalty(netD, real_data, fake_data, lambda_gp):
            alpha = torch.rand(real_data.shape[0], 1).to(self.device)
            alpha = alpha.expand(real_data.size())

            interpolates = alpha * real_data + ((1 - alpha) * fake_data)
            interpolates.requires_grad_(True)

            disc_interpolates = netD(interpolates)

            import torch.autograd as autograd
            gradients = autograd.grad(outputs=disc_interpolates,
                                      inputs=interpolates,
                                      grad_outputs=torch.ones(
                                          disc_interpolates.size(),
                                          device=self.device),
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)[0]

            gradient_penalty = (
                (gradients.norm(2, dim=1) - 1)**2).mean() * lambda_gp
            return gradient_penalty

        # disc 1
        gp_fake_data = all_emb_t
        gp_real_data = all_head_emb_h[get_select_idx(len(all_head_emb_h),
                                                     len(gp_fake_data),
                                                     strategy='random')]
        gradient_penalty = calc_gradient_penalty(netD=disc_model,
                                                 real_data=gp_real_data,
                                                 fake_data=gp_fake_data,
                                                 lambda_gp=self.lambda_gp)
        L_d = errorD + errorG + gradient_penalty

        optimizer_D.zero_grad()
        L_d.backward()
        optimizer_D.step()


        pseudo_embs = all_emb_nt[cell_mask]
        real_tail_embs = all_emb_nt[~cell_mask]
        if len(pseudo_embs) > len(real_tail_embs):
            gp_fake_data = pseudo_embs
            gp_real_data = real_tail_embs[get_select_idx(len(real_tail_embs),
                                                         len(gp_fake_data),
                                                         strategy='random')]
        else:
            gp_real_data = real_tail_embs
            gp_fake_data = pseudo_embs[get_select_idx(len(pseudo_embs),
                                                      len(gp_real_data),
                                                      strategy='random')]
        L_gp2 = calc_gradient_penalty(netD=disc_pseudo_real,
                                      real_data=gp_real_data,
                                      fake_data=gp_fake_data,
                                      lambda_gp=self.lambda_gp)

        prob_t_with_miss = disc_pseudo_real(all_emb_nt)
        errorR_pseudo = prob_t_with_miss[cell_mask].mean()
        errorR_real_tail = -prob_t_with_miss[~cell_mask].mean()
        L_d2 = errorR_pseudo + errorR_real_tail + L_gp2

        optimizer_D_pseudo_real.zero_grad()
        L_d2.backward()
        optimizer_D_pseudo_real.step()

        log = {
            'loss/disc1_errorD': errorD.item(),
            'loss/disc1_errorG': errorG.item(),
            'loss/disc1_errorG_real': prob_t[~cell_mask].mean().item(),
            'loss/disc1_errorG_pseudo': prob_t[cell_mask].mean().item(),
            'loss/disc1_gp': gradient_penalty.item(),
            'loss/disc1_full': L_d.item(),
            'loss/disc2_full': L_d2.item(),
            'loss/disc2_gp': L_gp2.item(),
            'loss/disc2_errorR_pseudo': errorR_pseudo.item(),
            'loss/disc2_errorR_real_tail': errorR_real_tail.item(),
            'noise_eps': noise_eps,
        }
        if os.environ.get('use_wandb'):
            wandb.log(log)
        return L_d

    def train_gen(self, embed_model, optimizer, disc_model, disc_pseudo_real):
        embed_model.train()
        disc_model.train()
        disc_pseudo_real.train()
        for p in disc_model.parameters():
            p.requires_grad = False
        for p in disc_pseudo_real.parameters():
            p.requires_grad = False

        _, _, other_embs_dict = embed_model.process_one_subgraph()
        u_emb_h, u_emb_t, u_emb_nt = other_embs_dict[
            'user_head_true_drop_false'], other_embs_dict[
                'user_head_false_drop_true'], other_embs_dict[
                    'user_head_true_drop_true']
        i_emb_h, i_emb_t, i_emb_nt = other_embs_dict[
            'item_head_true_drop_false'], other_embs_dict[
                'item_head_false_drop_true'], other_embs_dict[
                    'item_head_true_drop_true']

        all_emb_t = u_emb_t

        prob_t = disc_model(all_emb_t)
        L_disc1 = -prob_t.mean() * 0.1

        all_emb_nt = u_emb_nt[self.user_is_head_mask]

        prob_t_with_miss = disc_pseudo_real(all_emb_nt)

        L_disc2 = -prob_t_with_miss.mean() * 0.1

        L_d = L_disc1 + L_disc2

        optimizer.zero_grad()
        L_d.backward()
        optimizer.step()

        log = {
            'loss/discG_full': L_d.item(),
            'loss/discG_1': L_disc1.item(),
            'loss/discG_2': L_disc2.item(),
        }
        if os.environ.get('use_wandb'):
            wandb.log(log)
        return L_d
