"""
updated
EF21 with bidirectional compression
experiment for logistic regression function with non-convex regularizer
"""
import numpy as np
from sklearn.model_selection import train_test_split
import time
import sys
import os
import argparse
from numpy.random import normal, uniform
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix, load_svmlight_file, dump_svmlight_file
from numpy.linalg import norm
import itertools
from scipy.special import binom
from scipy.stats import ortho_group
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
from matplotlib import pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
import datetime
from IPython import display
from logreg_functions_fast import *

#np.random.seed(23)
def myrepr(x):
    return repr(round(x, 4)).replace('.',',') if isinstance(x, float) else repr(x)

def stopping_criterion(sq_norm, eps, it, Nsteps):
    #return (R_k > eps * R_0) and (it <= Nsteps)
    return (it <= Nsteps) and (sq_norm >=eps)

def top_k_matrix (X,k):
    output = np.zeros(X.shape)
    for i in range (X.shape[0]):
        output[i] = top_k_compressor(X[i],k)
    return output
        
def top_k_compressor(x, k):
    output = np.zeros(x.shape)
    x_abs = np.abs(x)
    idx = np.argpartition(x_abs, -k)[-k:]  # Indices not sorted
    inds = idx[np.argsort(x_abs[idx])][::-1]
    output[inds] = x[inds]
    return output

def compute_full_grads (A, x, b, la,n_workers):
    grad_ar = np.zeros((n_workers, x.shape[0]))
    for i in range(n_workers):
        grad_ar[i] = logreg_grad(x, A[i], b[i], la).copy()
    return grad_ar

def ef21_worker_compressor(A, x, b, la, k, wg_ar, n_workers):
    grads = compute_full_grads(A, x, b, la, n_workers)
    assert(grads.shape==(n_workers,x.shape[0]))
    wg_ar_new = np.zeros((n_workers, x.shape[0]))
    delta = grads - wg_ar
    wg_ar_new = wg_ar + top_k_matrix(delta, k)
    size_value_sent = 32
    return wg_ar_new, size_value_sent, np.mean(grads, axis=0)

def ef21_master_compressor(wg, g, k):
    delta = wg - g
    g_new = g + top_k_compressor(delta, k)
    return g_new

def ef21_bc(x_0, A, b, A_0, b_0, stepsize, eps,la, k_od, k_bd, n_workers, experiment_name, project_path, dataset, Nsteps=100000):
    wg_ar = compute_full_grads(A, x_0, b, la, n_workers)
    g = np.mean(wg_ar, axis=0)
    sq_norm_ar = [np.linalg.norm(x=g, ord=2) ** 2]
    its_bits_od_ar = [0]
    its_bits_bd_ar = [0]
    it_comm_ar = [0]
    x = x_0.copy()
    it = 0
    PRINT_EVERY = 1000
    while stopping_criterion(sq_norm_ar[-1], eps, it, Nsteps):
        x = x - stepsize*g 
        wg_ar, size_value_sent, grad = ef21_worker_compressor(A, x, b, la, k_od, wg_ar, n_workers)
        wg = np.mean(wg_ar, axis=0)
        g = ef21_master_compressor(wg, g, k_bd)
        sq_norm_ar.append(np.linalg.norm(x=grad, ord=2) ** 2)
        it += 1
        its_bits_od_ar.append(it*k_od*size_value_sent)
        its_bits_bd_ar.append(it*(k_od+k_bd)*size_value_sent)
        it_comm_ar.append(it)
        if it%PRINT_EVERY ==0:
            display.clear_output(wait=True)
            print(it, sq_norm_ar[-1])
            its_bits_od_ef21_bc = np.array(its_bits_od_ar)
            its_bits_bd_ef21_bc = np.array(its_bits_bd_ar)
            its_comm_ef21_bc = np.array(it_comm_ar)
            norms_ef21_bc = np.array(sq_norm_ar)
            sol_ef21_bc = x.copy()
            its_epochs_ef21_bc = its_comm_ef21_bc.copy()

            save_data(its_bits_od_ef21_bc, its_bits_bd_ef21_bc, its_epochs_ef21_bc, its_comm_ef21_bc, norms_ef21_bc, sol_ef21_bc, k_od, k_bd, experiment_name, project_path, dataset)
    return np.array(its_bits_od_ar), np.array(its_bits_bd_ar), np.array(it_comm_ar), np.array(sq_norm_ar), x

def save_data(its_bits_od, its_bits_bd, its_epochs, its_comm, f_grad_norms, x_solution, k_od, k_bd, experiment_name, project_path, dataset):
    experiment = '{0}_{1}_{2}'.format(experiment_name, k_od, k_bd)
    logs_path = project_path + "logs/logs_{0}_{1}/".format(dataset, experiment)
    
    if not os.path.exists(project_path + "logs/"):
        os.makedirs(project_path + "logs/")
    
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    np.save(logs_path + 'iteration_bits_od' + '_' + experiment, np.array(its_bits_od))
    np.save(logs_path + 'iteration_bits_bd' + '_' + experiment, np.array(its_bits_bd))
    np.save(logs_path + 'iteration_epochs' + '_' + experiment, np.array(its_epochs))
    np.save(logs_path + 'iteration_comm' + '_' + experiment, np.array(its_comm))
    np.save(logs_path + 'solution' + '_' + experiment, x_solution)
    np.save(logs_path + 'norms' + '_' + experiment, np.array(f_grad_norms))


parser = argparse.ArgumentParser(description='Run top-k algorithm')
parser.add_argument('--max_it', action='store', dest='max_it', type=int, default=None, help='Maximum number of iteration')
parser.add_argument('--k_od', action='store', dest='k_od', type=int, default=1, help='Worker-master compresion parameter')
parser.add_argument('--k_bd', action='store', dest='k_bd', type=int, default=1, help='Master-worker compresion parameter')
parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=1, help='Number of workers that will be used')
parser.add_argument('--factor', action='store', dest='factor', type=float, default=1, help='Stepsize factor')
parser.add_argument('--tol', action='store', dest='tol', type=float, default=1e-5, help='tolerance')
parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='mushrooms',help='Dataset name for saving logs')

args = parser.parse_args()
nsteps = args.max_it
k_od = args.k_od
k_bd = args.k_bd
n_w = args.num_workers
dataset = args.dataset
loss_func = "log-reg"
factor = args.factor
eps = args.tol
'''
nsteps = 2000
n_w = 20
dataset = "phishing"
loss_func = "log-reg"
factor = 1.0
eps = 1e-9
k_od = 1
k_bd = 1
'''
la = 0.1

user_dir = os.path.expanduser('~/')
project_path = os.getcwd() + "/"

data_path = project_path + "data_{0}/".format(dataset)

if not os.path.exists(data_path):
    os.mkdir(data_path)

X_0 = np.load(data_path + 'X.npy') #whole dateset
y_0 = np.load(data_path + 'y.npy')

n_0, d_0 = X_0.shape

hess_f_0 = (1 / (4*n_0)) * (X_0.T @ X_0) + 2*la*np.eye(d_0)
L_0 = np.max(np.linalg.eigvals(hess_f_0))
L_0 = L_0.astype(np.float)

#c = subprocess.call(f"python3 generate_data.py --dataset mushrooms --num_starts 1 --num_workers {n_w} --loss_func log-reg --is_homogeneous 0", shell=True)
X = []
y = []
L = np.zeros(n_w)
n = np.zeros(n_w, dtype=int)
d = np.zeros(n_w, dtype=int)
for j in range(n_w):
    X.append(np.load(data_path + 'X_{0}_nw{1}_{2}.npy'.format(dataset, n_w, j)))
    y.append(np.load(data_path + 'y_{0}_nw{1}_{2}.npy'.format(dataset, n_w, j)))
    n[j], d[j] = X[j].shape

    currentDT = datetime.datetime.now()
    print (currentDT.strftime("%Y-%m-%d %H:%M:%S"))
    print (X[j].shape)

    hess_f_j = (1 / (4*n[j])) * (X[j].T @ X[j]) + 2*la*np.eye(d[j])
    L[j] = np.max(np.linalg.eigvals(hess_f_j))
L = L.astype(np.float)

if not os.path.isfile(data_path + 'w_init_{0}.npy'.format(loss_func)):
    # create a new w_0
    x_0 = np.random.normal(loc=0.0, scale=1.0, size=d_0)
    np.save(data_path + 'w_init_{0}.npy'.format(loss_func), x_0)
    x_0 = np.array(np.load(data_path + 'w_init_{0}.npy'.format(loss_func)))
else:
    # load existing w_0
    x_0 = np.array(np.load(data_path + 'w_init_{0}.npy'.format(loss_func)))

al = k_od/d_0
t = -1 + np.sqrt(1/(1-al))
theta = 1 - (1 - al)*(1 + t)
beta = (1 - al)*(1 + 1/t)
Lt = np.sqrt (np.mean (L**2))
step_size_diana_tpc = (1/(L_0 + Lt*np.sqrt(beta/theta)))*factor

experiment_name = "ef21-bc_nw-{0}_{1}x".format(n_w, myrepr(factor))

results = ef21_bc(x_0, X, y, X_0, y_0, step_size_diana_tpc, eps,la, k_od, k_bd, n_w,experiment_name, project_path, dataset, Nsteps=nsteps)
print (experiment_name + f" with k={k_od} finished in {results[0].shape[0]} iterations" )
its_bits_od_ef21_bc = results[0]
its_bits_bd_ef21_bc = results[1]
its_comm_ef21_bc = results[2]
norms_ef21_bc = results[3]
sol_ef21_bc = results[4]
its_epochs_ef21_bc = its_comm_ef21_bc.copy()

save_data(its_bits_od_ef21_bc, its_bits_bd_ef21_bc, its_epochs_ef21_bc, its_comm_ef21_bc, norms_ef21_bc, sol_ef21_bc, k_od, k_bd, experiment_name, project_path, dataset)
