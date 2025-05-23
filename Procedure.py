import os

import numpy as np
import torch.nn.functional as F
import wandb

from contrast import Contrast
from torch.utils.data.dataloader import DataLoader
import Kg_Par
import torch
import utils
from utils import timer
from tqdm import tqdm
import Kg_Model
device = ('cuda' if torch.cuda.is_available() else "cpu")

def next_batch_pairwise(training_data, node_table, training_set_u, training_set_i, batch_size):
    from random import shuffle, choice
    n_negs = 1

    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)

    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [
                training_data[idx][0]
                for idx in range(batch_id, batch_size + batch_id)
            ]
            items = [
                training_data[idx][1]
                for idx in range(batch_id, batch_size + batch_id)
            ]
            batch_id += batch_size
        else:
            users = [
                training_data[idx][0] for idx in range(batch_id, data_size)
            ]
            items = [
                training_data[idx][1] for idx in range(batch_id, data_size)
            ]
            batch_id = data_size

        u_idx, i_idx, j_idx = [], [], []
        item_list = list(training_set_i.keys())
        for i, user_id in enumerate(users):
            i_idx.append(node_table[items[i]]['node_feature'])
            u_idx.append(node_table[user_id]['node_feature'])
            for _ in range(n_negs):
                neg_item_id = choice(item_list)
                while neg_item_id in training_set_u[user_id]:
                    neg_item_id = choice(item_list)
                j_idx.append(node_table[neg_item_id]['node_feature'])
        yield u_idx, i_idx, j_idx


def kg_init_transR(kgdataset, recommend_model, opt,index ):
    Recmodel = recommend_model
    Recmodel.train()
    kgloader = DataLoader(kgdataset, batch_size=4096, drop_last=True)
    trans_loss = 0.
    for data in tqdm(kgloader, total=len(kgloader), disable=True):
        heads = data[0].to(Kg_Par.device)
        relations = data[1].to(Kg_Par.device)
        pos_tails = data[2].to(Kg_Par.device)
        neg_tails = data[3].to(Kg_Par.device)
        kg_batch_loss = Recmodel.calc_kg_loss_transR(heads, relations, pos_tails, neg_tails, index)
        trans_loss += kg_batch_loss / len(kgloader)
        opt.zero_grad()
        kg_batch_loss.backward()
        opt.step()


def kg_init_TATEC(kgdataset, recommend_model, opt, index):
    Recmodel = recommend_model
    Recmodel.train()
    kgloader = DataLoader(kgdataset, batch_size=4096, drop_last=True)
    trans_loss = 0.
    for data in tqdm(kgloader, total=len(kgloader), disable=True):
        heads = data[0].to(Kg_Par.device)
        relations = data[1].to(Kg_Par.device)
        pos_tails = data[2].to(Kg_Par.device)
        neg_tails = data[3].to(Kg_Par.device)
        kg_batch_loss = Recmodel.calc_kg_loss_TATEC(heads, relations, pos_tails, neg_tails, index)
        trans_loss += kg_batch_loss / len(kgloader)
        opt.zero_grad()
        kg_batch_loss.backward()
        opt.step()


def BPR_train_contrast(dataset, recommend_model, loss_class, contrast_model: Contrast, contrast_views, optimizer, neg_k=1, w=None, ssl_reg=0.1):
    Recmodel: Kg_Model.Model = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    batch_size = Kg_Par.config['bpr_batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    aver_loss = 0.
    aver_loss_main = 0.
    aver_loss_ssl = 0.

    uiv1, uiv2 = contrast_views["uiv1"], contrast_views["uiv2"]

    for batch_i, train_data in tqdm(enumerate(dataloader), total=len(dataloader), disable=True):

        batch_users = train_data[0].long().to(Kg_Par.device)
        batch_pos = train_data[1].long().to(Kg_Par.device)
        batch_neg = train_data[2].long().to(Kg_Par.device)

        l_main = bpr.compute(batch_users, batch_pos, batch_neg)
        l_ssl = list()
        items = batch_pos


        usersv1_ro, itemsv1_ro = Recmodel.view_computer_all(uiv1, index=0)
        usersv2_ro, itemsv2_ro = Recmodel.view_computer_all(uiv2, index=1)

        items_uiv1 = itemsv1_ro[items]
        items_uiv2 = itemsv2_ro[items]
        l_item = contrast_model.grace_loss(items_uiv1, items_uiv2)

        users = batch_users
        users_uiv1 = usersv1_ro[users]
        users_uiv2 = usersv2_ro[users]
        l_user = contrast_model.grace_loss(users_uiv1, users_uiv2)
        l_ssl.extend([l_user * ssl_reg, l_item * ssl_reg])

        if l_ssl:
            l_ssl = torch.stack(l_ssl).sum()

            l_all = l_main + l_ssl
            aver_loss_ssl += l_ssl.cpu().item()
        else:
            l_all = l_main

        optimizer.zero_grad()
        l_all.backward()
        optimizer.step()

        aver_loss_main += l_main.cpu().item()
        aver_loss += l_all.cpu().item()

    timer.zero()





