
import time
import ctypes
import itertools
import sys
import logging

import pandas as pd 
import copy
import random
# import json

import numpy as np
from scipy.sparse import csr_matrix as cm
from numpy.linalg import norm
from numpy.linalg import svd
from numpy import power

import matplotlib.pyplot as plt

from logging_util import init_logger


def print_cost(func):
    def wrapper(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print('%s: %.1fs' % (func.__name__, time.time() - t))
        return res
    return wrapper

class Server_mavg(object):

    def __init__(self, num_users_per_device = 6, num_items = 3000, max_iter=500, K=50, lamb=0.1, eps=0.0001, beta = 1000, Qhat = 100, Q2 = None ,silent_run=False, save_uv=False, call_logger=None):
        if call_logger:
            global logger
            logger = call_logger
        self.K = K
        self.lamb = lamb
        self.eps = eps
        self.ite = max_iter
        self.tol = 1e-8
        self.silent_run=silent_run
        self.save_uv = save_uv
        self.beta = beta
        self.M = num_users_per_device
        self.N = num_items
        self.Qhat = Qhat
        self.Q2 = Q2
        self.load_lib()

    def load_lib(self):
        part_dot_lib = ctypes.cdll.LoadLibrary('partXY.dll')
        set_val_lib = ctypes.cdll.LoadLibrary('setVal.dll')
        self.part_dot = part_dot_lib.partXY
        self.set_val = set_val_lib.setVal

    

    def get_obs_inds(self):
        return self.train_data[:,0].astype(int), self.train_data[:,1].astype(int)

    def part_uv(self, U, V, rows, cols, k):
        num = len(rows)
        output = np.zeros((num,1), dtype=np.float64)

        up = U.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        vp = V.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        op = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        rsp = rows.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
        csp = cols.ctypes.data_as(ctypes.POINTER(ctypes.c_long))

        nc = ctypes.c_int(num)
        rc = ctypes.c_int(k)
        self.part_dot(up, vp, rsp, csp, op, nc, rc)
        return output

    def p_omega(self, mat, rows, cols):
        mat_t = mat.copy()
        mat_t[rows, cols] = 0.0
        return mat - mat_t

    def cal_omega(self, omega, U, V, rows, cols, bias, obs, train_num):
        puv = self.part_uv(U, V, rows, cols, self.K)
        puv = obs - puv -  bias
        puvp = puv.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        odp = omega.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        nc = ctypes.c_int(train_num)
        self.set_val(puvp, odp, nc)

    def obj(self, Us, V, omegas):
        ob = 0
        for i in range(self.p):
            ob += 1.0 / (self.p*2) * (power(norm(omegas[i].data),2) + self.lamb * power(norm(Us[i],'fro'),2))
        return ob + self.lamb / 2.0 * power(norm(V,'fro'),2)
    def train_rmse(self, Us, Vs, biass, omegas):
        tmp = 0
        for i in range(self.p):
            tmp += power(norm(omegas[i].data),2)
        return np.sqrt(tmp / self.total_num_train)

    def get_grad(self, omega, U, V):
        du = - 1.0/self.p * omega.dot(V)
        dv = - 1.0/self.p * omega.T.dot(U) 
        return du, dv

    def fit(self, train_data=[], test_data=[], num_selected_defvices = 50):
        omegas = {}
        Us = {}
        Vs = {}
        Ys = {}
        rowss = {}
        colss = {}
        obss = {}
        biass = {}
        train_num = {}
        test_num = 0
        self.p = len(train_data['user'])
        self.total_num_train = 0
        
        V = np.random.rand(self.N, self.K) * 0.0002

        for i in range(self.p):
            omegas[i] = cm((train_data['data'][i]['Rating'], (train_data['data'][i]['UserID']-1, train_data['data'][i]['MovieID']-1)), shape=(self.M,self.N)) #index starting from 0
            Us[i] = np.random.rand(self.M, self.K) * 0.0002
            Vs[i] = copy.deepcopy(V) 
            Ys[i] = np.zeros((self.N, self.K))
            biass[i] = 0 

            train_num[i] = len(train_data['data'][i])
            self.total_num_train += train_num[i]
            rowss[i], colss[i] = omegas[i].tocoo().row.astype(np.int32), omegas[i].tocoo().col.astype(np.int32)
            obss[i] = omegas[i].copy().data.astype(np.float64).reshape(train_num[i], 1)
            self.cal_omega(omegas[i], Us[i], Vs[i], rowss[i], colss[i], biass[i], obss[i],train_num[i])

            
        if len(test_data):
            trowss, tcolss = {}, {}
            for i in range(self.p):
                trowss[i], tcolss[i] = (test_data[i]['UserID']-1).array.astype(np.int32), (test_data[i]['MovieID']-1).array.astype(np.int32)
                test_num += len(test_data[i])
        objs_1 = [self.obj(Us, V, omegas)]
        eps_1 = eps_2 = self.eps

        
        objs_2 = []
        trmses = []
        rmses, maes, costs, acu_cost = [], [], [], []

        run_start = time.time()

        tmp = 0
        for i in range(self.p):
            tmp += Vs[i] 
        V = 1/num_selected_defvices * tmp

        for rnd in range(0, self.ite):
            start = time.time()
            
            ## select a subset of devices:
            S = random.sample(range(self.p),num_selected_defvices)
            ## update local parameters
            for i in range(self.p):
                Q1 = 10
                if self.Q2:
                    Q2s = self.Q2
                else:
                    Q2s = int(np.floor(self.Qhat/(rnd + 1)) + 1)
                
                # Compute the eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eig(Vs[i].T.dot(Vs[i]))
                # Find the largest eigenvalue
                step_u = 2.0/np.max(eigenvalues)
                for _ in range(Q1):
                    du = 1.0/(self.p) * (-omegas[i].dot(Vs[i])  + self.lamb * Us[i])
                    
                    Us[i] = Us[i] - step_u*du

                    self.cal_omega(omegas[i], Us[i], Vs[i], rowss[i], colss[i], biass[i], obss[i],train_num[i])
                
                eigenvalues, eigenvectors = np.linalg.eig(Us[i].T.dot(Us[i]))
                # Find the largest eigenvalue
                step_v = 1.0/(np.max(eigenvalues)*5)
                for _ in range(Q2s): 
                    dv = 1.0/self.p * (-omegas[i].T.dot(Us[i]) + self.lamb * Vs[i])             
                    Vs[i] = Vs[i] - step_v*dv

                    self.cal_omega(omegas[i], Us[i], Vs[i], rowss[i], colss[i], biass[i], obss[i],train_num[i])
            
            ## Server updates the global parameter: U
            tmp = 0
            for i in S:
                tmp += Vs[i] 
            V = V = 1/num_selected_defvices * tmp

            end = time.time()

            l_obj = self.obj(Us, V, omegas)
            objs_1.append(l_obj)
            
            trmses.append(self.cal_rmse(train_data['data'], self.total_num_train, Us, V, rowss, colss))

            lrate = (objs_1[rnd] - objs_1[rnd+1]) / objs_1[rnd]

            
            costs.append(round(end-start, 1))
            acu_cost.append(int(end-run_start))

            if len(test_data):
                
                rmses.append(self.cal_rmse(test_data, test_num, Us, V, trowss, tcolss))
                
                if not self.silent_run:
                
                    print('iter: ', rnd, "obj: ", objs_1[rnd], "rmse train: ",trmses[rnd], "rmse test: ", rmses[rnd])
            else:
                
                print('iter: ', rnd, "obj: ", objs_1[rnd], "rmse train: ",trmses[rnd])
    


            

        self.rmses = rmses if rmses else 99.0
        self.maes = maes if maes else 99.0
        self.costs = costs
        self.objs_1 = objs_1

    def get_test_rmse(self):
        return np.mean(self.rmses[-5:])

    def get_test_mae(self):
        return np.mean(self.maes[-5:])

    def cal_rmse(self, test_data, test_num, Us, V, trowss, tcolss):
        tmp = 0
        for i in range(self.p):
            preds = self.part_uv(Us[i], V, trowss[i], tcolss[i], self.K)
            delta = preds - test_data[i]['Rating'].array.astype(np.int32).reshape(preds.shape) #self.test_data[:,2].reshape(preds.shape)
            tmp += np.square(delta).sum()
        rmse = np.sqrt(tmp / test_num)
        return rmse

    def cal_mae(self, predss,test_data, test_num):
        tmp = 0
        for i in range(self.p):
            delta = predss[i] - test_data[i]['Rating'].array.astype(np.int32).reshape(predss[i].shape) #self.test_data[:,2].reshape(preds.shape)
            tmp += np.abs(delta).sum()
        mae = tmp / test_num
        return mae
