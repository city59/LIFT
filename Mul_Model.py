import os
import torch.nn.functional as F
import numpy as np
import torch
import wandb
import pickle
import gc
import datetime
import torch as t
import torch.nn as nn
import torch.utils.data as dataloader
import utils
import Mul_Data
import GNN
from lagcl_encoder import Discriminator
from Mul_Par import args
from Utils.TimeLogger import log
from tqdm import tqdm
import Procedure
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


t.backends.cudnn.benchmark = True
if t.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False
t.autograd.set_detect_anomaly(True)


class Model():
    def __init__(self):

        self.trn_file = args.path + args.dataset + '/trn_'
        self.tst_file = args.path + args.dataset + '/tst_int'

        self.g = {}
        self.user_num = -1
        self.item_num = -1
        self.behavior_mats = {}
        self.behaviors = []
        self.behaviors_data = {}

        self.train_loss = []
        self.his_hr = []
        self.his_ndcg = []
        gc.collect()

        self.relu = t.nn.ReLU()
        self.sigmoid = t.nn.Sigmoid()
        self.curEpoch = 0
        self.u = []
        self.i = []

        if args.dataset == 'Tmall':
            self.behaviors_SSL = ['pv', 'fav', 'cart', 'buy']
            self.behaviors = ['pv', 'fav', 'cart', 'buy']

        elif args.dataset == 'Yelp':
            self.behaviors = ['tip', 'neg', 'neutral', 'pos']
            self.behaviors_SSL = ['tip', 'neg', 'neutral', 'pos']

        elif args.dataset == 'retail':
            self.behaviors = ['fav', 'cart', 'buy']
            self.behaviors_SSL = ['fav', 'cart', 'buy']

        for i in range(0, len(self.behaviors)):
            with open(self.trn_file + self.behaviors[i], 'rb') as fs:
                data = pickle.load(fs)
                self.behaviors_data[i] = data
                self.u.append(data.get_shape()[0])
                self.i.append(data.get_shape()[1])

                if data.get_shape()[0] > self.user_num:

                    self.user_num = data.get_shape()[0]
                if data.get_shape()[1] > self.item_num:

                    self.item_num = data.get_shape()[1]

                if self.behaviors[i] == args.target:
                    self.trainMat = data
                    self.trainLabel = 1 * (self.trainMat != 0)
                    self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0)))

        for i in range(0, len(self.behaviors)):
            self.behaviors_data[i] = self.behaviors_data[i][:,: max(self.trainMat.indices)+1]
        self.trainMat = self.trainMat[:,: max(self.trainMat.indices)+1]
        self.item_num = max(self.trainMat.indices) + 1

        for i in range(0, len(self.behaviors)):
            self.behavior_mats[i] = utils.get_use(self.behaviors_data[i])

        train_u, train_v = self.trainMat.nonzero()
        train_data = np.hstack((train_u.reshape(-1, 1), train_v.reshape(-1, 1))).tolist()
        train_dataset = Mul_Data.RecDataset_beh(self.behaviors, train_data, self.item_num, self.behaviors_data, True)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0,
                                                  pin_memory=True)

        with open(self.tst_file, 'rb') as fs:
            data = pickle.load(fs)

        test_user = np.array([idx for idx, i in enumerate(data) if i is not None])
        test_item = np.array([i for idx, i in enumerate(data) if i is not None])
        test_data = np.hstack((test_user.reshape(-1, 1), test_item.reshape(-1, 1))).tolist()
        test_dataset = Mul_Data.RecDataset(test_data, self.item_num, self.trainMat, 0, False)
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0,
                                                 pin_memory=True)

        self.g["train_u"] = train_u
        self.g["train_i"] = train_v
        self.g["test_u"] = test_user
        self.g["test_i"] = test_item

    def prepareModel(self):
        self.gnn_layer = eval(args.gnn_layer)
        self.hidden_dim = args.hidden_dim
        self.model = GNN.myModel(self.g, self.user_num, self.item_num, self.behaviors, self.behavior_mats).cuda()
        self.opt = t.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.opt_weight_decay)
        self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=5,
                                                       step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None,
                                                       scale_mode='cycle', cycle_momentum=False, base_momentum=0.8,
                                                       max_momentum=0.9, last_epoch=-1)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.disc_model = Discriminator(args.hidden_dim).to(device)
        self.disc_pseudo_real = Discriminator(args.hidden_dim).to(device)
        self.optimizer_D = torch.optim.RMSprop(self.disc_model.parameters(), lr=args.learning_rate * 0.1)
        self.optimizer_D_pseudo_real = torch.optim.RMSprop(self.disc_pseudo_real.parameters(), lr=args.learning_rate * 0.1)

        if use_cuda:
            self.model = self.model.cuda()

    def innerProduct(self, u, i, j):
        pred_i = t.sum(t.mul(u, i), dim=1) * args.inner_product_mult
        pred_j = t.sum(t.mul(u, j), dim=1) * args.inner_product_mult
        return pred_i, pred_j

    def SSL(self, user_embeddings, user_step_index):

        def multi_neg_sample_pair_index(batch_index, step_index, embedding1,
                                        embedding2):

            index_set = set(np.array(step_index.cpu()))
            batch_index_set = set(np.array(batch_index.cpu()))
            neg2_index_set = index_set - batch_index_set
            neg2_index = t.as_tensor(np.array(list(neg2_index_set))).long().cuda()
            neg2_index = t.unsqueeze(neg2_index, 0)
            neg2_index = neg2_index.repeat(len(batch_index), 1)
            neg2_index = t.reshape(neg2_index, (1, -1))
            neg2_index = t.squeeze(neg2_index)

            neg1_index = batch_index.long().cuda()
            neg1_index = t.unsqueeze(neg1_index, 1)
            neg1_index = neg1_index.repeat(1, len(neg2_index_set))
            neg1_index = t.reshape(neg1_index, (1, -1))
            neg1_index = t.squeeze(neg1_index)

            neg_score_pre = t.sum(
                compute(embedding1, embedding2, neg1_index, neg2_index).squeeze().view(len(batch_index), -1),
                -1)
            return neg_score_pre

        def compute(x1, x2, neg1_index=None, neg2_index=None, τ=0.05):

            if neg1_index != None:
                x1 = x1[neg1_index]
                x2 = x2[neg2_index]

            N = x1.shape[0]
            D = x1.shape[1]

            x1 = x1
            x2 = x2

            scores = t.exp(t.div(t.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1), np.power(D, 1) + 1e-8))

            return scores

        def single_infoNCE_loss_one_by_one(embedding1, embedding2, step_index):
            N = step_index.shape[0]
            D = embedding1.shape[1]

            pos_score = compute(embedding1[step_index], embedding2[step_index]).squeeze()
            neg_score = t.zeros((N,), dtype=t.float64).cuda()


            steps = int(np.ceil(N / args.SSL_batch))
            for i in range(steps):
                st = i * args.SSL_batch
                ed = min((i + 1) * args.SSL_batch, N)
                batch_index = step_index[st: ed]

                neg_score_pre = multi_neg_sample_pair_index(batch_index, step_index, embedding1, embedding2)
                if i == 0:
                    neg_score = neg_score_pre
                else:
                    neg_score = t.cat((neg_score, neg_score_pre), 0)


            con_loss = -t.log(1e-8 + t.div(pos_score, neg_score + 1e-8))

            assert not t.any(t.isnan(con_loss))
            assert not t.any(t.isinf(con_loss))

            return t.where(t.isnan(con_loss), t.full_like(con_loss, 0 + 1e-8), con_loss)

        user_con_loss_list = []

        SSL_len = int(user_step_index.shape[0] / 10)
        user_step_index = t.as_tensor(
            np.random.choice(user_step_index.cpu(), size=SSL_len, replace=False, p=None)).cuda()

        for i in range(len(self.behaviors_SSL)):
            user_con_loss_list.append(
                single_infoNCE_loss_one_by_one(user_embeddings[-1], user_embeddings[i], user_step_index))

        return user_con_loss_list, user_step_index

    def run(self, Kg_model, contrast_model, optimizer, bpr):

        self.Kg_model = Kg_model
        self.contrast_model = contrast_model
        self.optimizer = optimizer
        self.bpr = bpr
        self.prepareModel()

        cvWait = 0
        self.best_HR = 0
        self.best_NDCG = 0

        self.user_embed = None
        self.item_embed = None
        self.user_embeds = None
        self.item_embeds = None

        for e in range(self.curEpoch, args.epoch + 1):
            self.curEpoch = e

            log("*****************Start epoch: %d ************************" % e)

            Procedure.kg_init_transR(self.Kg_model.kg_dataset, Kg_model, optimizer, index=0)
            Procedure.kg_init_TATEC(self.Kg_model.kg_dataset, Kg_model, optimizer, index=1)

            if args.isJustTest == False:
                epoch_loss = self.trainEpoch()
                self.train_loss.append(epoch_loss)
                print(f"epoch {e / args.epoch},  epoch loss{epoch_loss}")
                self.train_loss.append(epoch_loss)
            else:
                break

            HR, NDCG = self.testEpoch(self.test_loader)
            self.his_hr.append(HR)
            self.his_ndcg.append(NDCG)

            self.scheduler.step()

            if HR > self.best_HR:
                self.best_HR = HR
                self.best_epoch = self.curEpoch
                cvWait = 0
                print("----------------------------------------------------------------------------------------------------best_HR",self.best_HR)

            if NDCG > self.best_NDCG:
                self.best_NDCG = NDCG
                self.best_epoch = self.curEpoch
                cvWait = 0
                print("----------------------------------------------------------------------------------------------------best_NDCG",self.best_NDCG)

            if (HR < self.best_HR) and (NDCG < self.best_NDCG):
                cvWait += 1

            if cvWait == args.patience:
                print(f"Early stop at {self.best_epoch} :  best HR: {self.best_HR}, best_NDCG: {self.best_NDCG} \n")
                break

        HR, NDCG = self.testEpoch(self.test_loader)
        self.his_hr.append(HR)
        self.his_ndcg.append(NDCG)

    def trainEpoch(self):

        contrast_views = self.contrast_model.get_ui_kg_view()
        log("Drop done")
        Procedure.BPR_train_contrast(self.Kg_model.dataset, self.Kg_model, self.bpr, self.contrast_model, contrast_views, self.optimizer, neg_k=1)

        self.train_disc(self.model.gcn, self.disc_model, self.optimizer_D, self.disc_pseudo_real, self.optimizer_D_pseudo_real)
        self.train_gen(self.model.gcn, self.optimizer, self.disc_model, self.disc_pseudo_real)

        train_loader = self.train_loader
        time = datetime.datetime.now()
        print("start_pos_neg_sample:  ", time)
        train_loader.dataset.pos_neg_sample()
        time = datetime.datetime.now()
        print("end_pos_neg_sample:  ", time)

        epoch_loss = 0

        self.behavior_loss_list = [None] * len(self.behaviors)

        self.user_id_list = [None] * len(self.behaviors)
        self.item_id_pos_list = [None] * len(self.behaviors)
        self.item_id_neg_list = [None] * len(self.behaviors)


        cnt = 0
        for user, item_i, item_j in tqdm(train_loader):

            user = user.long().cuda()
            self.user_step_index = user

            mul_behavior_loss_list = [None] * len(self.behaviors)
            mul_user_index_list = [None] * len(self.behaviors)

            mul_model = GNN.myModel(self.g, self.user_num, self.item_num, self.behaviors, self.behavior_mats).cuda()
            mul_opt = t.optim.AdamW(mul_model.parameters(), lr=args.lr, weight_decay=args.opt_weight_decay)
            mul_model.load_state_dict(self.model.state_dict())
            #
            mul_user_embed, mul_item_embed, mul_user_embeds, mul_item_embeds = mul_model()

            for index in range(len(self.behaviors)):

                not_zero_index = np.where(item_i[index].cpu().numpy() != -1)[0]

                self.user_id_list[index] = user[not_zero_index].long().cuda()

                mul_user_index_list[index] = self.user_id_list[index]
                self.item_id_pos_list[index] = item_i[index][not_zero_index].long().cuda()
                self.item_id_neg_list[index] = item_j[index][not_zero_index].long().cuda()

                mul_userEmbed = mul_user_embed[self.user_id_list[index]]
                mul_posEmbed = mul_item_embed[self.item_id_pos_list[index]]
                mul_negEmbed = mul_item_embed[self.item_id_neg_list[index]]

                mul_pred_i, mul_pred_j = self.innerProduct(mul_userEmbed, mul_posEmbed, mul_negEmbed)

                mul_behavior_loss_list[index] = - (mul_pred_i.view(-1) - mul_pred_j.view(-1)).sigmoid().log()

            mul_infoNCELoss_list, SSL_user_step_index = self.SSL(mul_user_embeds, self.user_step_index)


            for i in range(len(self.behaviors)):
                mul_infoNCELoss_list[i] = (mul_infoNCELoss_list[i]).sum()
                mul_behavior_loss_list[i] = (mul_behavior_loss_list[i]).sum()

            mul_bprloss = sum(mul_behavior_loss_list) / len(mul_behavior_loss_list)
            mul_infoNCELoss = sum(mul_infoNCELoss_list) / len(mul_infoNCELoss_list)
            mul_regLoss = (t.norm(mul_userEmbed) ** 2 + t.norm(mul_posEmbed) ** 2 + t.norm(mul_negEmbed) ** 2)

            mul_model_loss = (mul_bprloss + args.reg * mul_regLoss + args.beta * mul_infoNCELoss) / args.batch

            epoch_loss = epoch_loss + mul_model_loss.item()

            mul_opt.zero_grad(set_to_none=True)
            mul_model_loss.backward()
            nn.utils.clip_grad_norm_(mul_model.parameters(), max_norm=20, norm_type=2)
            mul_opt.step()

            user_embed, item_embed, user_embeds, item_embeds = self.model()

            with t.no_grad():
                user_embed1, item_embed1 = self.Kg_model.getAll()

            user_embed = 0.9*user_embed + 0.1*user_embed1
            item_embed = 0.9*item_embed + 0.1*item_embed1

            for index in range(len(self.behaviors)):
                userEmbed = user_embed[self.user_id_list[index]]
                posEmbed = item_embed[self.item_id_pos_list[index]]
                negEmbed = item_embed[self.item_id_neg_list[index]]

                pred_i, pred_j = self.innerProduct(userEmbed, posEmbed, negEmbed)

                self.behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()

            infoNCELoss_list, SSL_user_step_index = self.SSL(user_embeds, self.user_step_index)

            for i in range(len(self.behaviors)):
                infoNCELoss_list[i] = (infoNCELoss_list[i]).sum()
                self.behavior_loss_list[i] = (self.behavior_loss_list[i]).sum()


            bprloss = sum(self.behavior_loss_list) / len(self.behavior_loss_list)
            infoNCELoss = sum(infoNCELoss_list) / len(infoNCELoss_list)
            regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)

            lt_mul_loss = self.model.gcn.lt_loss(user, item_i[3], item_j[3])
            loss = (bprloss + args.reg * regLoss + args.beta * infoNCELoss + lt_mul_loss) / args.batch

            epoch_loss = epoch_loss + loss.item()

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()

            cnt += 1

        return epoch_loss

    def testEpoch(self, data_loader):

        epochHR, epochNDCG = [0] * 2
        with t.no_grad():
            user_embed, item_embed, user_embeds, item_embeds = self.model()
            user_embed1, item_embed1 = self.Kg_model.getAll()

        user_embed = 0.9 * user_embed + 0.1 * user_embed1
        item_embed = 0.9 * item_embed + 0.1 * item_embed1

        cnt = 0
        tot = 0
        for user, item_i in data_loader:
            user_compute, item_compute, user_item1, user_item100 = self.sampleTestBatch(user, item_i)
            userEmbed = user_embed[user_compute]
            itemEmbed = item_embed[item_compute]

            pred_i = t.sum(t.mul(userEmbed, itemEmbed), dim=1)

            hit, ndcg = self.calcRes(t.reshape(pred_i, [user.shape[0], 100]), user_item1, user_item100)
            epochHR = epochHR + hit
            epochNDCG = epochNDCG + ndcg
            cnt += 1
            tot += user.shape[0]

        result_HR = epochHR / tot
        result_NDCG = epochNDCG / tot
        print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}")

        return result_HR, result_NDCG

    def calcRes(self, pred_i, user_item1, user_item100):

        hit = 0
        ndcg = 0

        for j in range(pred_i.shape[0]):

            _, shoot_index = t.topk(pred_i[j], args.shoot)
            shoot_index = shoot_index.cpu()
            shoot = user_item100[j][shoot_index]
            shoot = shoot.tolist()

            if type(shoot) != int and (user_item1[j] in shoot):
                hit += 1
                ndcg += np.reciprocal(np.log2(shoot.index(user_item1[j]) + 2))
            elif type(shoot) == int and (user_item1[j] == shoot):
                hit += 1
                ndcg += np.reciprocal(np.log2(0 + 2))

        return hit, ndcg

    def sampleTestBatch(self, batch_user_id, batch_item_id):

        batch = len(batch_user_id)
        tmplen = (batch * 100)

        sub_trainMat = self.trainMat[batch_user_id].toarray()
        user_item1 = batch_item_id
        user_compute = [None] * tmplen
        item_compute = [None] * tmplen
        user_item100 = [None] * (batch)

        cur = 0
        for i in range(batch):
            pos_item = user_item1[i]
            negset = np.reshape(np.argwhere(sub_trainMat[i] == 0), [-1])

            random_neg_sam = np.random.permutation(negset)[:99]
            user_item100_one_user = np.concatenate((random_neg_sam, np.array([pos_item])))
            user_item100[i] = user_item100_one_user

            for j in range(100):
                user_compute[cur] = batch_user_id[i]
                item_compute[cur] = user_item100_one_user[j]
                cur += 1

        return user_compute, item_compute, user_item1, user_item100

    def train_disc(self, embed_model, disc_model, optimizer_D, disc_pseudo_real, optimizer_D_pseudo_real):
        disc_model.train()
        disc_pseudo_real.train()
        # embed_model.eval()
        embed_model.train()

        for p in disc_model.parameters():
            p.requires_grad = True  # to avoid computation
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

        all_head_emb_h = u_emb_h[self.model.gcn.user_is_head_mask]
        all_emb_t = u_emb_t
        all_emb_nt = u_emb_nt
        cell_mask = self.model.gcn.user_is_head_mask

        if True:
            if self.model.gcn.annealing_type == 0:
                noise_eps = self.model.gcn.noise_eps
            else:
                noise_eps = 0.0

            def exec_perturbed(x, noise_eps):
                random_noise = torch.rand_like(x).to(device)
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
            alpha = torch.rand(real_data.shape[0], 1).to(device)
            alpha = alpha.expand(real_data.size())

            interpolates = alpha * real_data + ((1 - alpha) * fake_data)
            interpolates.requires_grad_(True)

            disc_interpolates = netD(interpolates)

            import torch.autograd as autograd
            gradients = autograd.grad(outputs=disc_interpolates,
                                      inputs=interpolates,
                                      grad_outputs=torch.ones(
                                          disc_interpolates.size(),
                                          device=device),
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
                                                 lambda_gp=self.model.gcn.lambda_gp)
        L_d = errorD + errorG + gradient_penalty

        optimizer_D.zero_grad()
        L_d.backward()
        optimizer_D.step()

        # disc 2
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
                                      lambda_gp=self.model.gcn.lambda_gp)

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

        all_emb_nt = u_emb_nt[self.model.gcn.user_is_head_mask]

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


