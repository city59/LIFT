import torch.utils.data as data
import numpy as np


class RecDataset(data.Dataset):
    def __init__(self, data, num_item, train_mat=None, num_ng=1, is_training=True):  
        super(RecDataset, self).__init__()

        self.data = np.array(data)
        self.num_item = num_item
        self.train_mat = train_mat
        self.is_training = is_training

    def ng_sample(self):
        dok_trainMat = self.train_mat.todok()  
        length = self.data.shape[0]  
        self.neg_data = np.random.randint(low=0, high=self.num_item, size=length)  

        for i in range(length):
            uid = self.data[i][0]
            iid = self.neg_data[i]
            if (uid, iid) in dok_trainMat:
                while (uid, iid) in dok_trainMat:  
                    iid = np.random.randint(low=0, high=self.num_item)
                    self.neg_data[i] = iid
                self.neg_data[i] = iid  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.data[idx][1]

        if self.is_training:  
            neg_data = self.neg_data
            item_j = neg_data[idx]
            return user, item_i, item_j
        else:  
            return user, item_i

    def getMatrix(self):
        pass
    
    def getAdj(self):
        pass
    
    def sampleLargeGraph(self):
        def makeMask():
            pass
        def updateBdgt():
            pass
        def sample():
            pass
    
    def constructData(self):
        pass


class RecDataset_beh(data.Dataset):
    def __init__(self, beh, data, num_item, behaviors_data=None, num_ng=1, is_training=True):  
        super(RecDataset_beh, self).__init__()

        self.data = np.array(data)
        self.num_item = num_item
        self.is_training = is_training
        self.beh = beh
        self.behaviors_data = behaviors_data

        self.length = self.data.shape[0] 
        self.neg_data = [None]*self.length  
        self.pos_data = [None]*self.length  

    def pos_neg_sample(self):
        for i in range(self.length):
            self.neg_data[i] = [None]*len(self.beh)
            self.pos_data[i] = [None]*len(self.beh)

        for index in range(len(self.beh)):
            train_u, train_v = self.behaviors_data[index].nonzero()
            beh_dok = self.behaviors_data[index].todok()
            set_pos = np.array(list(set(train_v)))
            self.pos_data_index = np.random.choice(set_pos, size=self.length, replace=True, p=None)
            for i in range(self.length):
                uid = self.data[i][0]
                iid_neg = self.get_negative_sample(uid, beh_dok)
                self.neg_data[i][index] = iid_neg
                iid_pos = self.pos_data_index[i]
                if index == (len(self.beh) - 1):
                    iid_pos = train_v[i]
                elif (uid, iid_pos) not in beh_dok:
                    iid_pos = self.get_positive_sample(uid, index)
                self.pos_data[i][index] = iid_pos

    def get_negative_sample(self, uid, beh_dok):

        iid_neg = np.random.randint(low=0, high=self.num_item)

        while (uid, iid_neg) in beh_dok:
            iid_neg = np.random.randint(low=0, high=self.num_item)
        return iid_neg

    def get_positive_sample(self, uid, index):


        if len(self.behaviors_data[index][uid].data) == 0:
            return -1
        t_array = self.behaviors_data[index][uid].toarray()
        pos_index = np.where(t_array != 0)[1]
        return np.random.choice(pos_index, size=1, replace=True, p=None)[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.pos_data[idx]

        if self.is_training:  
            item_j = self.neg_data[idx]
            return user, item_i, item_j
        else:  
            return user, item_i

