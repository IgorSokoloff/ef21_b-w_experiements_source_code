"""
to be tested on cluster
EF21 with heavy ball acceleration
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
    return repr(round(x, 2)).replace('.',',') if isinstance(x, float) else repr(x)

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

def ef21_hb_estimator(A, x, b, la, k, g_ar, n_workers):
    grads = compute_full_grads(A, x, b, la, n_workers)
    assert(grads.shape==(n_workers, x.shape[0]))
    g_ar_new = np.zeros((n_workers, x.shape[0]))
    delta = grads - g_ar
    g_ar_new = g_ar + top_k_matrix(delta, k)
    size_value_sent = 32
    return g_ar_new, size_value_sent, np.mean(grads, axis=0)

def ef21_hb(x_0, A, b, A_0, b_0, stepsize, eta, eps,la,k, n_workers,experiment_name, project_path, dataset, Nsteps=100000):
    g_ar = compute_full_grads(A, x_0, b, la, n_workers)
    g = np.mean(g_ar, axis=0)
    v = g.copy()
    sq_norm_ar = [np.linalg.norm(x=g, ord=2) ** 2]
    it_bits_ar = [0]
    it_comm_ar = [0]
    x = x_0.copy()
    it = 0
    PRINT_EVERY = 1000
    COMPUTE_FG_EVERY = 10
    while stopping_criterion(sq_norm_ar[-1], eps, it, Nsteps):
        x = x - stepsize*v
        g_ar, size_value_sent, grad = ef21_hb_estimator(A, x, b, la, k, g_ar, n_workers)
        g = np.mean(g_ar, axis=0)
        v = eta*v + g
        it += 1
        sq_norm_ar.append(np.linalg.norm(x=grad, ord=2) ** 2)
        it_bits_ar.append(it*k*size_value_sent)
        it_comm_ar.append(it)
        if it%PRINT_EVERY ==0:
            print(it, sq_norm_ar[-1])
            its_bits_ef21_hb = np.array(it_bits_ar)
            its_comm_ef21_hb = np.array(it_comm_ar)      # communication complexity
            norms_ef21_hb = np.array(sq_norm_ar)
            sol_ef21_hb = x
            its_epochs_ef21_hb = its_comm_ef21_hb.copy()

            save_data(its_bits_ef21_hb, its_epochs_ef21_hb, its_comm_ef21_hb, norms_ef21_hb, sol_ef21_hb, k, experiment_name, project_path, dataset)
    return np.array(it_bits_ar), np.array(it_comm_ar), np.array(sq_norm_ar), x

def save_data(its_bits, its_epochs, its_comm, f_grad_norms, x_solution, k_size, experiment_name, project_path, dataset):
    experiment = '{0}_{1}'.format(experiment_name, k_size)
    logs_path = project_path + "logs/logs_{0}_{1}/".format(dataset, experiment)

    if not os.path.exists(project_path + "logs/"):
        os.makedirs(project_path + "logs/")

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    np.save(logs_path + 'iteration_bits_od' + '_' + experiment, np.array(its_bits))
    np.save(logs_path + 'iteration_epochs' + '_' + experiment, np.array(its_epochs))
    np.save(logs_path + 'iteration_comm' + '_' + experiment, np.array(its_comm))
    np.save(logs_path + 'solution' + '_' + experiment, x_solution)
    np.save(logs_path + 'norms' + '_' + experiment, np.array(f_grad_norms))

parser = argparse.ArgumentParser(description='Run top-k algorithm')
parser.add_argument('--max_it', action='store', dest='max_it', type=int, default=None, help='Maximum number of iteration')
parser.add_argument('--k', action='store', dest='k', type=int, default=1, help='Sparcification parameter')
parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=1, help='Number of workers that will be used')
parser.add_argument('--factor', action='store', dest='factor', type=float, default=1, help='Stepsize factor')
parser.add_argument('--eta', action='store', dest='eta', type=float, default=0.99, help='eta parameter')
parser.add_argument('--tol', action='store', dest='tol', type=float, default=1e-5, help='tolerance')
parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='mushrooms',help='Dataset name for saving logs')

args = parser.parse_args()
nsteps = args.max_it
k_tk = args.k
n_w = args.num_workers
dataset = args.dataset
loss_func = "log-reg"
factor = args.factor
eps = args.tol
eta = args.eta

# test
'''
nsteps = 2000
k_tk = 1
n_w = 20
dataset = "phishing"
loss_func = "log-reg"
factor = 1
eps = 1e-7
eta = 0.5
'''
la = 0.1

user_dir = os.path.expanduser('~/')
project_path = os.getcwd() + "/"

data_path = project_path + "data_{0}/".format(dataset)

if not os.path.exists(data_path):
    os.mkdir(data_path)

A_0 = np.load(data_path + 'X.npy') #whole dateset
b_0 = np.load(data_path + 'y.npy')

n_0, d_0 = A_0.shape
hess_f_0 = (1 / (4*n_0)) * (A_0.T @ A_0) + 2*la*np.eye(d_0)
L_0 = np.max(np.linalg.eigvals(hess_f_0))
L_0 = L_0.astype(np.float)

A = []
b = []
L = np.zeros(n_w)
n = np.zeros(n_w, dtype=int)
d = np.zeros(n_w, dtype=int)
for j in range(n_w):
    A.append(np.load(data_path + 'X_{0}_nw{1}_{2}.npy'.format(dataset, n_w, j)))
    b.append(np.load(data_path + 'y_{0}_nw{1}_{2}.npy'.format(dataset, n_w, j)))
    n[j], d[j] = A[j].shape

    currentDT = datetime.datetime.now()
    print (currentDT.strftime("%Y-%m-%d %H:%M:%S"))
    print (A[j].shape)

    hess_f_j = (1 / (4*n[j])) * (A[j].T @ A[j]) + 2*la*np.eye(d[j])
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

al = k_tk/d_0
#theory
t = -1 + np.sqrt(1/(1-al))
theta = 1 - (1 - al)*(1 + t)
beta = (1 - al)*(1 + 1/t)
Lt = np.sqrt (np.mean (L**2))
step_size_ef21_hb = (1/(L_0 + Lt*np.sqrt(beta/theta)))*factor

experiment_name = "ef21-hb_nw-{0}_{1}x_e-{2}".format(n_w, myrepr(factor), myrepr(eta))

results = ef21_hb(x_0, A, b, A_0, b_0, step_size_ef21_hb, eta, eps, la, k_tk, n_w,experiment_name, project_path, dataset, Nsteps=nsteps)
print (experiment_name + f" with k={k_tk} finished in {results[0].shape[0]} iterations")

its_bits_ef21_hb = results[0]
its_comm_ef21_hb = results[1]      # communication complexity
norms_ef21_hb = results[2]
sol_ef21_hb = results[3]
its_epochs_ef21_hb = its_comm_ef21_hb.copy()

save_data(its_bits_ef21_hb, its_epochs_ef21_hb, its_comm_ef21_hb, norms_ef21_hb, sol_ef21_hb, k_tk, experiment_name, project_path, dataset)
