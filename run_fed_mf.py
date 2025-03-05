import numpy as np 
from fd_mf import Server
import pandas as pd 
from utils import *
from fd_mavg import Server_mavg
import matplotlib.pyplot as plt
import pickle
import json

datasets = ['Movielens 1M', 'Movielens 10M', 'Netflix']
ndevices = 100
nactive = 10
max_iter = 100
inner_iter = 10
K = {'Movielens 1M':5, 'Movielens 10M':8, 'Netflix':13}
for dataset in datasets:
    if dataset == 'Movielens 1M':
        data = load_data('data\ml-1m\ratings.dat')
    if dataset == 'Movielens 10M':
        data = load_data('data\ml-10m\ml-10M100K\ratings.dat')
    if dataset == 'Netflix':
        data = load_data_netflix('data\netflix\netflix.csv')
    num_items = data['MovieID'].max()
    dict_data, num_users_per_device = split_devices(data,ndevices)
    ##Split training and testing
    train_data = {}
    test_data = {}
    train_data['user'] = dict_data['user']
    train_data['data'] = {}
    for i in dict_data['user']:                        
        train_data['data'][i], test_data[i] = split_data(dict_data['data'][i], 0.8)

    fed_mc = Server(max_iter=max_iter,num_users_per_device = num_users_per_device, num_items = num_items, beta=1e4,lamb=1e-6)

    fed_mc.fit(train_data=train_data, test_data=test_data, num_selected_defvices=nactive, inner_iter=inner_iter)

    fed_amc = Server_mavg(max_iter=max_iter,num_users_per_device = num_users_per_device, num_items = num_items, Q2 = 10, beta=1e4,lamb=1e-6, K = K[dataset])

    fed_amc.fit(train_data=train_data, test_data=test_data, num_selected_defvices=nactive)

    with open('./logs/'+dataset+'_'+str(ndevices)+'_'+str(nactive)+'_'+str(max_iter)+'_'+str(inner_iter)+'_FedMC_ADMM.json', 'w') as file:
        json.dump({'loss':fed_mc.objs_1,'time':fed_mc.costs,'test_rmse':fed_mc.rmses}, file, indent=4)
    
    with open('./logs/'+dataset+'_'+str(ndevices)+'_'+str(nactive)+'_'+str(max_iter)+'_'+str(inner_iter)+'_FedMAvg.json', 'w') as file:
        json.dump({'loss':fed_amc.objs_1,'time':fed_amc.costs,'test_rmse':fed_amc.rmses}, file, indent=4)
    x = np.array(fed_mc.costs).cumsum()

    plt.figure()
    plt.plot(range(len(fed_mc.objs_1)), fed_mc.objs_1, color='teal', linewidth =3,label='FedMC-ADMM')
    plt.plot(range(len(fed_amc.objs_1)), fed_amc.objs_1, color='red', linewidth =3,label='FedMAvg')

    plt.xlabel('Communication rounds',fontsize=14 )
    plt.ylabel('Objective value',fontsize=14 )
    plt.title(dataset,fontsize=14 )
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig('./images/'+dataset+'_obj_'+str(ndevices)+'_'+str(nactive)+'_'+str(max_iter)+'_'+str(inner_iter)+'.png', format='png', dpi=300)

    plt.figure()
    plt.plot(range(len(fed_mc.rmses)), fed_mc.rmses, color='teal',linewidth =3, label='FedMC-ADMM')
    plt.plot(range(len(fed_amc.rmses)), fed_amc.rmses, color='red', linewidth =3,label='FedMAvg')

    plt.xlabel('Communication rounds', fontsize=14)
    plt.ylabel('Testing RMSE', fontsize=14)
    plt.title(dataset, fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.savefig('./images/'+dataset+'_RMSE_'+str(ndevices)+'_'+str(nactive)+'_'+str(max_iter)+'_'+str(inner_iter)+'.png', format='png', dpi=300)
