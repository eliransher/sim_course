# imports
import simpy
import numpy as np
import sys
sys.path.append('../sim_course_pycharm')

import pandas as pd
import os
from IPython.display import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from scipy.linalg import expm, sinm, cosm
import matplotlib.pyplot as plt

from numpy.linalg import matrix_power
from scipy.stats import rv_discrete
from scipy.special import factorial
from transeint_throughput import *
from single_station_transient import *
from multiple_servers_transient import *
      

def mm1_steady(rho):
    x = np.arange(500)
    return (1-rho)*rho**x

def print_steady_state_bar_chart(ground_truth, results):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    width = 0.25
    max_probs = 11
    rects1 = ax1.bar( 1.5*width+np.arange(max_probs), results[0][0][:11], width, label='Sim')
    rects2 = ax1.bar(np.arange(max_probs) , ground_truth[:11], width, label='Label')
    plt.rcParams['font.size'] = '20'

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_ylabel('PMF', fontsize=21)
    ax1.set_xlabel('Number of customers', fontsize=20)
    ax1.set_title( 'error: '+ str(results[0][1])[:5] + ', runtime: '+ str(results[0][2])[:5] + ' sec', fontsize=21,       fontweight="bold")
    ax1.set_xticks(np.linspace(0,12,16).astype(int))
    ax1.set_xticklabels(np.linspace(0,12,16).astype(int), fontsize=19)
    ax1.legend(fontsize=22)


    rects1 = ax2.bar( 1.5*width+np.arange(max_probs), results[1][0][:11], width, label='Sim')
    rects2 = ax2.bar(np.arange(max_probs) , ground_truth[:11], width, label='Label')
    plt.rcParams['font.size'] = '20'

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    ax2.set_ylabel('PMF', fontsize=21)
    ax2.set_xlabel('Number of customers', fontsize=20)
    ax2.set_title(  'error: '+ str(results[1][1])[:5] + ', runtime: '+ str(results[1][2])[:5]+ ' sec' , fontsize=21,       fontweight="bold")
    ax2.set_xticks(np.linspace(0,12,16).astype(int))
    ax2.set_xticklabels(np.linspace(0,12,16).astype(int), fontsize=19)
    ax2.legend(fontsize=22)


    rects1 = ax3.bar( 1.5*width+np.arange(max_probs), results[2][0][:11], width, label='Sim')
    rects2 = ax3.bar(np.arange(max_probs) , ground_truth[:11], width, label='Label')
    plt.rcParams['font.size'] = '20'

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    ax3.set_ylabel('PMF', fontsize=21)
    ax3.set_xlabel('Number of customers', fontsize=20)
    ax3.set_title(  'error: '+ str(results[2][1])[:5] + ', runtime: '+ str(results[2][2])[:5] + ' sec', fontsize=21,       fontweight="bold")
    ax3.set_xticks(np.linspace(0,12,16).astype(int))
    ax3.set_xticklabels(np.linspace(0,12,16).astype(int), fontsize=19)
    ax3.legend(fontsize=22)
    plt.show()

    
def plot_run_time_accuracy_plot(x_vals, error_list, runtime_list, error_label = 'Error'):

    # Create some mock data
    t = x_vals
    data1 = error_list
    data2 = runtime_list

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Simulation time (sec)')
    ax1.set_ylabel(error_label, color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Runtime (sec)', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Accuracy Runtime trade-off', fontsize = 20)
    plt.show()
    
    
def get_cost(service_rate, arrival_rate, c1, c2):
    ## In this case we compute this analytical
    sojourn_time = get_sojourn(service_rate, arrival_rate)
    return c1*service_rate + c2*sojourn_time

def get_sojourn(service_rate, arrival_rate):
    
    return (1/service_rate)/(1-arrival_rate/service_rate)



import math
def c_func(c,lam,mu):
    rho = lam/mu
    parta = (1-rho)
    partb = math.factorial(c)/(c*rho)**c
    if c > 1:
        partc = np.array([(((c*rho)**k)/math.factorial(k)) for k in range(c-1)]).sum()
    else:
        partc  =1
    tot = 1/(1+parta*partb*partc)
    return tot
    
def sojourn_multiple(c,lam,mu):
    return c_func(c,lam,mu)/(c*mu-lam)+1/mu

def reward(n1, lam1, mu1, n2, lam2, mu2,  r1, r2, c, c1, c2):
    
    sojourn1 =sojourn_multiple(n1,lam1,mu1)
    sojourn2 =sojourn_multiple(n2,lam2,mu2)
    
    return  r1*lam1+r2*lam2 - ( c1*sojourn_1+c2*sojourn_2+ c*(n1+n2))


def compute_pdf_within_range(x_vals, s, A):
    '''
    compute_pdf_within_range
    :param x_vals:
    :param s:
    :param A:
    :return:
    '''
    pdf_list = []
    for x in x_vals:
        pdf_list.append(compute_pdf(x, s, A).flatten())

    return pdf_list

def compute_cdf_within_range(x_vals, s, A):
    '''
    compute_cdf_within_range
    :param x_vals:
    :param s:
    :param A:
    :return:
    '''
    pdf_list = []
    for x in x_vals:
        pdf_list.append(compute_cdf(x, s, A).flatten())

    return pdf_list

def compute_pdf(x, s, A):
    '''
    x: the value of pdf
    s: inital probs
    A: Generative matrix
    '''
    A0 = -np.dot(A, np.ones((A.shape[0], 1)))
    return np.dot(np.dot(s, expm(A * x)), A0)


def compute_first_n_moments(s, A, n=3):
    moment_list = []
    for moment in range(1, n + 1):
        moment_list.append(ser_moment_n(s, A, moment))
    return np.array(moment_list).flatten()



def ser_moment_n(s, A, mom):
    e = np.ones((A.shape[0], 1))
    try:
        mom_val = ((-1) ** mom) *factorial(mom)*np.dot(np.dot(s, matrix_power(A, -mom)), e)
        if mom_val > 0:
            return mom_val
        else:
            return False
    except:
        return False
    
    
def create_erlang_row(rate, ind, size):
    aa = np.zeros(size)
    aa[ind] = -rate
    if ind < size - 1:
        aa[ind + 1] = rate
    return aa

def generate_erlang_given_rates(rate, ph_size):
    '''
    generate_erlang_given_rates
    :param rate:
    :param ph_size:
    :return:
    '''
    A = np.identity(ph_size)
    A_list = [create_erlang_row(rate, ind, ph_size) for ind in range(ph_size)]
    A = np.concatenate(A_list).reshape((ph_size, ph_size))
    return A

def create_Erlang_given_ph_size(ph_size):
    '''
    create_Erlang_given_ph_size
    :param ph_size:
    :return:
    '''
    s = np.zeros(ph_size)
    s[0] = 1
    rate = ph_size
    A = generate_erlang_given_rates(rate, ph_size)
    # A = A*compute_first_n_moments(s, A, 1)[0][0]
    return (s,A)


def compute_trans_probs_tandem_queues(dict_sch, sim_time, capacities, num_stations, mu):

    arrival_scd_pd = convert_dict_to_pd(dict_sch)
    stations_res_list = {}

    for station in range(1, 1 + num_stations):
        stations_res_list[station] = []

    for ind in tqdm(range(5000)):

        sim_res = single_sim_rate_arrival(arrival_scd_pd, mu, sim_time, capacities, num_stations)

        for station in range(1, 1 + num_stations):
            stations_res_list[station].append(sim_res[station])

    probs = {}
    for station in range(1, 1 + num_stations):
        probs[station] = compute_prob_arr(stations_res_list[station], sim_time, max_num_customers=100)

    return probs


def compute_counts(num_stations, sim_time, capacities, mu, num_jobs, num_simulations):
    stations_res_list = {}

    counts_thought = []

    for station in range(1, 1 + num_stations):
        stations_res_list[station] = []

    for ind in tqdm(range(num_simulations)):

        sim_res, count_num_jobs_complete = single_sim_rate_arrival_throughtput(mu, sim_time, capacities, num_stations,
                                                                               num_jobs)
        counts_thought.append(count_num_jobs_complete)
        for station in range(1, 1 + num_stations):
            stations_res_list[station].append(sim_res[station])

    return counts_thought

def compute_prob_arr(list_of_sims, sim_time, max_num_customers=100):
    time_dict = {}


    for hour in range(sim_time + 1):
        time_dict[hour] = np.zeros(max_num_customers)

    for resultDictionary in list_of_sims:
        for hour in resultDictionary.keys():
            time_dict[hour][resultDictionary[hour]] += 1

        # phase 2:
    prob_queue_arr = np.array([])

    for t in range(len(time_dict)):

        probs = (time_dict[t] / time_dict[t].sum())

        if t == 0:
            prob_queue_arr = np.array(probs).reshape(1, probs.shape[0])
        else:
            prob_queue_arr = np.concatenate((prob_queue_arr, np.array(probs).reshape(1, probs.shape[0])), axis=0)

    return prob_queue_arr



# def compute_transient_probs( mu, arrival_rate, sim_time, num_sims, shut_down = 8):
#     dict_sch = {0: arrival_rate, shut_down: 0.0000000000000005}
#     arrival_scd_pd = convert_dict_to_pd(dict_sch)
#     list_of_sims = []
#     for ind in tqdm(range(num_sims)):
#         list_of_sims.append(single_sim(arrival_scd_pd, mu, sim_time))
#
#     prob_queue_arr = compute_prob_arr(list_of_sims, sim_time)
#
#     return prob_queue_arr
# num_stations = 3
# capacities = {1: 1, 2:2, 3:3}
#
# dict_sch = {0: 1/20, 481:1/12.5}
# sim_time = 240
# arrival_scd_pd = convert_dict_to_pd(dict_sch)
# num_stations = 3
# mu = np.zeros(num_stations)
# mu[0] = 1/10
# mu[1] = 1/15
# mu[2] = 1/60
#
# compute_trans_probs_tandem_queues(dict_sch, sim_time, capacities, num_stations, mu)