"""
EF21 with heavy ball acceleration
experiment for least squares function
"""

import numpy as np
import time
import sys
import os
import argparse
from numpy.random import normal, uniform
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix, load_svmlight_file, dump_svmlight_file
from numpy.linalg import norm
import itertools
from scipy.special import binom
import pandas as pd
from matplotlib import pyplot as plt
import math
from sklearn.datasets import load_svmlight_file
import datetime
from IPython import display
from least_squares_functions_fast import *

#np.random.seed(23)
def myrepr(x):
    return repr(round(x, 4)).replace('.',',') if isinstance(x, float) else repr(x)

def stopping_criterion(func_diff, eps, it, Nsteps):
    #return (R_k > eps * R_0) and (it <= Nsteps)
    return (it <= Nsteps) and (func_diff >=eps)

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
        grad_ar[i] = least_squares_grad(x, A[i], b[i], la).copy()
    return grad_ar

def ef21_hb_estimator(A, x, b, la, k, g_ar, n_workers):
    grads = compute_full_grads(A, x, b, la, n_workers)
    assert(grads.shape==(n_workers, x.shape[0]))
    g_ar_new = np.zeros((n_workers, x.shape[0]))
    delta = grads - g_ar
    g_ar_new = g_ar + top_k_matrix(delta, k)
    size_value_sent = 32
    return g_ar_new, size_value_sent, np.mean(grads, axis=0)

def ef21_hb(x_0, x_star, f_star, A, b, A_0, b_0, stepsize, eta, eps,la,k, n_workers, experiment_name, project_path,dataset, Nsteps=100000):
    g_ar = compute_full_grads(A, x_0, b, la, n_workers)
    g = np.mean(g_ar, axis=0)
    v = g.copy()
    dim = x_0.shape[0]
    f_x = least_squares_loss(x_0, A_0, b_0, la)
    sq_norm_ar = [np.linalg.norm(x=g, ord=2) ** 2]
    its_bits_od_ar = [0]
    its_bits_bd_ar = [0]
    its_comm_ar = [0]
    its_arg_res_ar = [np.linalg.norm(x=(x_0 - x_star), ord=2) ** 2] #argument residual \sqnorm{x^t - x_star}
    func_diff_ar = [f_x - f_star]
    x = x_0.copy()
    it = 0
    PRINT_EVERY = 1000
    COMPUTE_FG_EVERY = 10
    while stopping_criterion(func_diff_ar[-1], eps, it, Nsteps):
        x = x - stepsize*v
        g_ar, size_value_sent, grad = ef21_hb_estimator(A, x, b, la, k, g_ar, n_workers)
        g = np.mean(g_ar, axis=0)
        v = eta*v + g
        it += 1
        f_x = least_squares_loss(x, A_0, b_0, la)
        sq_norm_ar.append(np.linalg.norm(x=grad, ord=2) ** 2)
        its_bits_od_ar.append(it*k*size_value_sent)
        its_bits_bd_ar.append(it*(k+dim)*size_value_sent)
        its_comm_ar.append(it)
        its_arg_res_ar.append(np.linalg.norm(x=(x - x_star), ord=2) ** 2)
        func_diff_ar.append(f_x - f_star)
        if it%PRINT_EVERY ==0:
            print(it, sq_norm_ar[-1], func_diff_ar[-1])
            its_bits_od = np.array(its_bits_od_ar)
            its_bits_bd = np.array(its_bits_bd_ar)
            its_comm = np.array(its_comm_ar)
            its_arg_res = np.array(its_arg_res_ar)
            func_diff = np.array(func_diff_ar)
            norms = np.array(sq_norm_ar)
            sol = x.copy()
            its_epochs = its_comm.copy()

            save_data(its_bits_od, its_bits_bd, its_epochs, its_comm, its_arg_res, func_diff, norms, sol, k, experiment_name, project_path,dataset)
    return np.array(its_bits_od_ar), np.array(its_bits_bd_ar), np.array(its_comm_ar), np.array(its_arg_res_ar), np.array(func_diff_ar), np.array(sq_norm_ar), x,

def save_data(its_bits_od, its_bits_bd, its_epochs, its_comm, its_arg_res, func_diff, f_grad_norms, x_solution, k_size, experiment_name, project_path, dataset):
    experiment = '{0}_{1}'.format(experiment_name, k_size)
    logs_path = project_path + "logs/logs_{0}_{1}/".format(dataset, experiment)

    if not os.path.exists(project_path + "logs/"):
        os.makedirs(project_path + "logs/")

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    np.save(logs_path + 'iteration_bits_od' + '_' + experiment, np.array(its_bits_od))
    np.save(logs_path + 'iteration_bits_bd' + '_' + experiment, np.array(its_bits_bd))
    np.save(logs_path + 'iteration_epochs' + '_' + experiment, np.array(its_epochs))
    np.save(logs_path + 'iteration_comm' + '_' + experiment, np.array(its_comm))
    np.save(logs_path + 'iteration_arg_res' + '_' + experiment, np.array(its_arg_res))
    np.save(logs_path + 'func_diff' + '_' + experiment, np.array(func_diff))
    np.save(logs_path + 'norms' + '_' + experiment, np.array(f_grad_norms))
    np.save(logs_path + 'solution' + '_' + experiment, x_solution)
##}

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
loss_func = "least-sq"
factor = args.factor
eps = args.tol
eta = args.eta
'''
nsteps = 2000
k_tk = 1
n_w = 20
dataset = "phishing"
loss_func = "least-sq"
factor = 1
eps = 1e-7
eta = 0.5
'''
la = 0

user_dir = os.path.expanduser('~/')
project_path = os.getcwd() + "/"

data_path = project_path + "data_{0}/".format(dataset)

if not os.path.exists(data_path):
    os.mkdir(data_path)

X_0 = np.load(data_path + 'X.npy') #whole dateset
y_0 = np.load(data_path + 'y.npy')

n_0, d_0 = X_0.shape

hess_f_0 = (2 /n_0) * (X_0.T @ X_0) + 2*la*np.eye(d_0)
eigvs = np.linalg.eigvals(hess_f_0)
mu_0 = eigvs[np.where(eigvs > 0, eigvs, np.inf).argmin()]  #returns smallest positive number

L_0 = np.max(np.linalg.eigvals(hess_f_0))
L_0 = L_0.astype(float)

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

    hess_f_j = (2 / (n[j])) * (X[j].T @ X[j]) + 2*la*np.eye(d[j])
    L[j] = np.max(np.linalg.eigvals(hess_f_j))
L = L.astype(float)

if not os.path.isfile(data_path + 'w_init_{0}.npy'.format(loss_func)):
    # create a new w_0
    x_0 = np.random.normal(loc=0.0, scale=1.0, size=d_0)
    np.save(data_path + 'w_init_{0}.npy'.format(loss_func), x_0)
    x_0 = np.array(np.load(data_path + 'w_init_{0}.npy'.format(loss_func)))
else:
    # load existing w_0
    x_0 = np.array(np.load(data_path + 'w_init_{0}.npy'.format(loss_func)))

x_star_path = data_path + 'x_star_{0}.npy'.format(loss_func)
f_star_path = data_path + 'f_star_{0}.npy'.format(loss_func)
if (not os.path.isfile(x_star_path)) or (not os.path.isfile(f_star_path)):
    f = lambda w: least_squares_loss(w, X_0, y_0, la)
    grad = lambda w: least_squares_grad (w, X_0, y_0, la)
    minimize_result = minimize(fun=f, x0=x_0, jac=grad, method="BFGS", tol=1e-16, options={"maxiter": 10000000})
    x_star, f_star = minimize_result.x, minimize_result.fun
    np.save(x_star_path, x_star)
    np.save(f_star_path, f_star)
else:
    x_star = np.load(x_star_path)
    f_star = np.load(f_star_path)

al = k_tk/d_0
#theory
t = -1 + np.sqrt(1/(1-al))
theta = 1 - (1 - al)*(1 + t)
beta = (1 - al)*(1 + 1/t)
Lt = np.sqrt (np.mean (L**2))
left_part = float(1/(L_0 + Lt*np.sqrt(2*beta/theta)))
right_part = float(theta/(2*mu_0))

step_size_pl_ef21_hb = min(left_part, right_part) *factor

experiment_name = "pl-ef21-hb_nw-{0}_{1}x_e-{2}".format(n_w, myrepr(factor), myrepr(eta))
begin_time = datetime.datetime.now()
results = ef21_hb(x_0,x_star, f_star, X, y, X_0, y_0, step_size_pl_ef21_hb, eta, eps, la, k_tk, n_w, experiment_name, project_path,dataset, Nsteps=nsteps)
print (experiment_name + f" with k={k_tk} finished in {results[0].shape[0]} iterations; running time: {datetime.datetime.now() - begin_time}")

its_bits_od = results[0]
its_bits_bd = results[1]
its_comm = results[2]
its_arg_res = results[3]
func_diff = results[4]
norms = results[5]
sol = results[6]
its_epochs = its_comm.copy()

save_data(its_bits_od, its_bits_bd, its_epochs, its_comm, its_arg_res, func_diff, norms, sol, k_tk, experiment_name, project_path,dataset)
