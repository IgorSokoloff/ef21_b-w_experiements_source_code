"""
updated:  16.09.2021
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

def ef21_compressor(v_ar, g_ar, x, k, n_workers):
    assert(g_ar.shape==(n_workers,x.shape[0]))
    assert(v_ar.shape==(n_workers,x.shape[0]))
    delta = v_ar - g_ar
    g_ar_new = g_ar + top_k_matrix(delta, k)
    size_value_sent = 32
    num_bits_sent = size_value_sent*k
    return g_ar_new, num_bits_sent

def compute_stoch_grads_diff (A, x, x_prev, b, la, n_workers, tau_ar, m_ar):
    sgrad_diff_ar = np.zeros((n_workers, x.shape[0]))
    #check  m_ar and A.shape[0]
    for i in range(n_workers):
        probs = np.full(m_ar[i], 1 / m_ar[i])
        inds = np.random.choice(a=np.arange(m_ar[i]), size=tau_ar[i], replace=False, p=probs)
        sgrad_diff_ar[i] = logreg_grad(x, A[i][inds], b[i][inds], la) - logreg_grad(x_prev, A[i][inds], b[i][inds], la)
    return sgrad_diff_ar

def page_estimator(A, x,x_prev, b, la, v_ar, n_workers, prob, tau_ar, m_ar, tau_sum, m_total):

    v_ar_new = np.zeros((n_workers, x.shape[0]))
    c_k = np.random.binomial(1, prob, 1)[0] #sample bernoully random varinable
    if c_k == 1:
        v_ar_new = compute_full_grads(A, x, b, la, n_workers)
        num_oracle_epochs = 1
    elif c_k ==0:
        v_ar_new = v_ar + compute_stoch_grads_diff (A, x, x_prev, b, la, n_workers, tau_ar, m_ar)
        num_oracle_epochs = 2*(tau_sum/m_total) # because we compute stoch grad on points x and x_prev
    else:
        raise ValueError ("c_k has to be either 0 or 1")

    return v_ar_new, num_oracle_epochs, c_k

def ef21_page(x_0, A, b, A_0, b_0, stepsize,prob, tau_ar, eps,la,k, n_workers, m_ar, Nsteps=100000):
    #m_ar - number of datapoints per worker
    m_total = A_0.shape[0] #total number of datapoints
    tau_sum = np.sum(tau_ar)
    COMPUTE_GD_EVERY = int(m_total/tau_sum)
    v_ar = compute_full_grads(A, x_0, b, la, n_workers)
    g_ar = v_ar.copy()
    g = np.mean(g_ar, axis=0)
    sq_norm_ar = [np.linalg.norm(x=g, ord=2) ** 2]
    it_bits_ar = [0]
    it_epochs_ar = [0]
    it_comm_ar = [0]
    x_prev = x_0.copy()
    x = x_0.copy()
    it = 0
    it_bits_cum_sum = 0
    it_epochs_cum_sum = 0
    PRINT_EVERY = 1000
    while stopping_criterion(sq_norm_ar[-1], eps, it, Nsteps):
        x = x_prev - stepsize*g

        v_ar_new, num_oracle_epochs, c_k = page_estimator(A, x,x_prev, b, la, v_ar, n_workers, prob, tau_ar, m_ar, tau_sum, m_total)
        g_ar_new, num_bits_sent = ef21_compressor (v_ar_new, g_ar, x, k, n_workers)
        g = np.mean(g_ar_new, axis=0)
        it_epochs_cum_sum += num_oracle_epochs
        it_bits_cum_sum += num_bits_sent
        it += 1
        if (it%COMPUTE_GD_EVERY == 0) or (c_k == 1):
            if (it%COMPUTE_GD_EVERY == 0) and (c_k == 0):
                grad = logreg_grad(x, A_0, b_0, la)
            if c_k == 1:
                grad = np.mean(v_ar_new, axis=0)
            sq_norm_ar.append(np.linalg.norm(x=grad, ord=2) ** 2)
            it_bits_ar.append(it_bits_ar[-1] + it_bits_cum_sum)
            it_comm_ar.append(it)
            it_epochs_ar.append(it_epochs_ar[-1] + it_epochs_cum_sum)
            it_bits_cum_sum = 0
            it_epochs_cum_sum = 0

        if it%PRINT_EVERY ==0:
            print(it, sq_norm_ar[-1])
        x_prev = x.copy()
        v_ar = v_ar_new.copy()
        g_ar = g_ar_new.copy()
    return np.array(it_bits_ar), np.array(it_epochs_ar), np.array(it_comm_ar), np.array(sq_norm_ar), x

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
parser.add_argument('--tol', action='store', dest='tol', type=float, default=1e-5, help='tolerance')
parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='mushrooms',help='Dataset name for saving logs')
parser.add_argument('--prb', action='store', dest='prb', type=float, default=0.5, help='Proportion of batchsize')

args = parser.parse_args()
nsteps = args.max_it
k_tk = args.k
n_w = args.num_workers
dataset = args.dataset
loss_func = "log-reg"
factor = args.factor
eps = args.tol
prb = args.prb


nsteps = 1000
n_w = 20
dataset = "realsim"
loss_func = "log-reg"
factor = 1.0
eps = 1e-9
prb = 0.25
#prb = 0.125
#prb = 0.015

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
step_size_ef21_page = (1/(L_0 + Lt*np.sqrt(beta/theta)))*factor
tau_ar = np.array(n*prb).astype(dtype=int)

experiment_name = "ef21-page_nw-{0}_{1}x_{2}".format(n_w, myrepr(factor), myrepr(prb))

prob = np.mean(tau_ar/(tau_ar + n))
print (round(prob, 4))
results = ef21_page(x_0, A, b, A_0, b_0, step_size_ef21_page, prob, tau_ar, eps, la, k_tk, n_w, n, Nsteps=nsteps)
print (experiment_name + f" with k={k_tk} finished in {results[0].shape[0]} iterations")
its_bits_ef21_page = results[0]
its_epochs_ef21_page = results[1]    # number of epochs
its_comm_ef21_page = results[2]      # communication complexity
norms_ef21_page = results[3]
sol_ef21_page = results[4]

save_data(its_bits_ef21_page, its_epochs_ef21_page, its_comm_ef21_page, norms_ef21_page, sol_ef21_page, k_tk, experiment_name, project_path, dataset)
