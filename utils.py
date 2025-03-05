#coding=utf8
'''
    utils
'''

import numpy as np
from scipy.sparse import csr_matrix as csr
import pandas as pd

def reverse_map(m):
    return {v:k for k,v in m.items()}

def generate_adj_mat(relation, row_map, col_map, is_weight=False):
    data, rows, cols = [],[],[]
    for r in relation:
        if is_weight:
            data.append(r[2])
        else:
            data.append(1)
        rows.append(row_map[r[0]])
        cols.append(col_map[r[1]])
    adj = csr((data,(rows,cols)),shape=[len(row_map), len(col_map)])
    adj_t = csr((data,(cols, rows)),shape=[len(col_map), len(row_map)])
    return adj, adj_t

def load_rand_data():
    '''
        return the features, labels, and the group inds
    '''
    S, N = 1000, 80
    X = np.random.normal(size=[S,N])
    Y = np.random.uniform(size=[S])

    test_X = np.random.normal(size=[200, N])
    test_Y = np.random.uniform(size=[200])
    logger.info('train_data: (%.4f,%.4f), test_data: (%.4f,%.4f)', np.mean(Y), np.std(Y), np.mean(test_Y), np.std(test_Y))
    return X, Y, test_X, test_Y

def save_lines(filename, res):
    fw = open(filename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print('save %s lines in %s' % (len(res), filename))

def save_triplets(filename, triplets, is_append=False):
    if is_append:
        fw = open(filename, 'a+')
        fw.write('\n')
    else:
        fw = open(filename, 'w+')
    fw.write('\n'.join(['%s\t%s\t%s' % (h,t,v) for h,t,v in triplets]))
    fw.close()
    print('save %s triplets in %s' % (len(triplets), filename))

def test_save_triplets():
    a = [(i,i**2, i**3) for i in range(10)]
    filename = 'log/test_appending_mode2.txt'
    for ind in xrange(0, len(a), 3):
        tri = a[ind:ind+3]
        save_triplets(filename, tri, is_append=True)

def split_data(data, train_ratio):
    obs_num = len(data)
    rand_inds = np.random.permutation(obs_num)
    train_num = int(obs_num * train_ratio)

    train_data = data.iloc[rand_inds[:train_num]]
    test_data = data.iloc[rand_inds[train_num:]]
    test_num = len(test_data)
    del rand_inds

    return train_data, test_data

def load_data(filename):
    # self.data = np.loadtxt(self.filename, dtype=np.float64)
    # #self.data[:,2] -= self.data[:,2].mean()
    # #self.data[:,2] /= self.data[:,2].std()
    # self.obs_num = len(self.data)
    # self.split_data()
    column_names = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    data = pd.read_csv(filename, sep = "::", names = column_names, engine='python')
    tmp = data['Rating']-data['Rating'].mean()
    data['Rating'] = tmp/tmp.std()
    # self.obs_num = len(self.data)
    return data
    # self.split_data()

def load_data_netflix(filename):
    data = pd.read_csv(filename)
    tmp = data['Rating']-data['Rating'].mean()
    data['Rating'] = tmp/tmp.std()
    return data



def split_devices(df,num_devices = None):
    
    # num_divices = 1000
    if num_devices == None:
        num_devices = df['UserID'].max()
    dict_data = {}
    dict_data['user'] = [i for i in range(num_devices)]
    dict_data['data'] = {}
    all_idx = [i for i in range(1,df['UserID'].max()+1)]
    num_users_per_device = int(df['UserID'].max()/num_devices)
    for i in range(num_devices):
        choices = np.random.choice(all_idx, num_users_per_device, replace=False)
        dict_data['data'][i] = df[df['UserID'].isin(choices)]
        dict_data['data'][i].loc[:,'UserID'] = pd.factorize(dict_data['data'][i]['UserID'])[0] + 1
        all_idx = list(set(all_idx) - set(choices))
    return dict_data, num_users_per_device

# if __name__ == '__main__':
#     test_save_triplets()

