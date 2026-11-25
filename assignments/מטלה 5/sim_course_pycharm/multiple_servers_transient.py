# imports
import simpy
import numpy as np
import sys
import pandas as pd
import os
from IPython.display import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

sys.path.append('../sim_course_pycharm')
from triage import Customer
from triage import Queue_triage
from two_stations2 import Customer
from two_stations2 import Queue_two_stations
from utils import *


def give_num_cust_dist(sim_time, list_of_sims):
    # phase 1:
    max_num_customers = 100
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


def give_rate_per_time(df, time):
    ind_rate = df.loc[df['LB_time'] <= time, :].index[-1]
    return df.loc[ind_rate, 'arrival_rate']


def convert_dict_to_pd(dict_sch):
    arrival_sch = pd.DataFrame([])

    for ind, key in enumerate(dict_sch.keys()):
        arrival_sch.loc[ind, 'LB_time'] = key
        arrival_sch.loc[ind, 'arrival_rate'] = dict_sch[key]

    return arrival_sch


def find_num_cust_time_stamp(df, time):
    try:
        ind = df.loc[df['Time_stamp'] < time, :].index[-1]

        num_cust = df.loc[ind, 'num_cust']
    except:
        num_cust = 0

    return num_cust



def single_sim_rate_arrival(arrival_scd_pd, mu, sim_time, capacities, num_stations):

    sim_time += 1
    sim_instance = Queue_multi_stations(arrival_scd_pd, mu, sim_time, capacities,  num_stations)
    sim_instance.run()

    resultDictionary = {}

    for station in range(1, num_stations+1):

        event_log = pd.DataFrame({'Customer_id': sim_instance.event_log_customer_id_list[station],
                                  'Time_stamp': sim_instance.event_log_time_stamp[station],
                                  'Type': sim_instance.event_log_type_list[station],
                                  'num_cust': sim_instance.event_log_num_cust_list[station]})

        result = [(time_epoch, find_num_cust_time_stamp(event_log, time_epoch)) for time_epoch in range(sim_time)]
        resultDictionary[station] = dict((x, y) for x, y in result)

    return resultDictionary


is_print = False


## Defining a new class: of customers
class Customer:
    def __init__(self, p_id, arrival_time, type_cust):
        self.id = p_id
        self.arrival_time = arrival_time
        self.type = type_cust




class Queue_multi_stations:

    def __init__(self, arrival_scd_pd, mu, sim_time, capacities, num_stations):



        self.env = simpy.Environment()  # initializing simpy enviroment
        self.end_time = sim_time
        self.id_current = 1
        self.arrival_pd = arrival_scd_pd
        self.num_stations = num_stations

        self.server = {}
        self.mu = {}

        self.num_cust_durations = {}
        self.last_event_time = {}
        self.num_cust_sys = {}

        self.event_log_customer_id_list = {}
        self.event_log_num_cust_list = {}
        self.event_log_type_list = {}
        self.event_log_time_stamp = {}



        for station in range(1, num_stations + 1):

            self.server[station] = simpy.Resource(self.env, capacity=capacities[station])  # Defining a resource with capacity 1
            self.mu[station] = mu[station-1]  # service rate
            self.num_cust_durations[station] = np.zeros(500)  ## the time duration of each each state (state= number of cusotmers in the system)
            self.last_event_time[station] = 0  # the time of the last event -
            self.num_cust_sys[station] = 0  # keeping track of number of customers in the system

            self.event_log_customer_id_list[station] = []
            self.event_log_num_cust_list[station] = []
            self.event_log_type_list[station] = []
            self.event_log_time_stamp[station] = []



    def run(self):

        self.env.process(self.customer_arrivals(1))  ## Initializing a process
        self.env.run(until=self.end_time)  ## Running the simulaiton until self.end_time

    def give_rate_per_time(self, time):

        ind_rate = self.arrival_pd.loc[self.arrival_pd['LB_time'] <= time, :].index[-1]
        return self.arrival_pd.loc[ind_rate, 'arrival_rate']

    #########################################################
    ################# Service block #########################
    #########################################################

    def service(self, station, customer):

        with self.server[station].request() as req:
            yield req

            yield self.env.timeout(np.random.exponential(1 / self.mu[station]))

            tot_time = self.env.now - self.last_event_time[station]  # keeping track of the last event
            self.num_cust_durations[station][self.num_cust_sys[station]] += tot_time

            self.num_cust_sys[station] -= 1  # updating number of cusotmers in the system
            self.last_event_time[station] = self.env.now


            self.event_log_customer_id_list[station].append(customer.id)
            self.event_log_time_stamp[station].append(self.env.now)
            self.event_log_num_cust_list[station].append(self.num_cust_sys[station])
            self.event_log_type_list[station].append('Departure')

            if is_print:
                print('Departed customer {} at {}'.format(customer.id, self.env.now))


        if station < self.num_stations:

            station += 1

            tot_time = self.env.now - self.last_event_time[station]
            self.num_cust_durations[station][self.num_cust_sys[station]] += tot_time

            curr_id = self.id_current
            arrival_time = self.env.now
            customer = Customer(curr_id, arrival_time, 1)

            self.event_log_customer_id_list[station].append(customer.id)
            self.event_log_time_stamp[station].append(self.env.now)

            self.last_event_time[station] = self.env.now

            self.num_cust_sys[station] += 1
            self.event_log_num_cust_list[station].append(self.num_cust_sys[station])
            self.event_log_type_list[station].append('Arrival')

            self.env.process(self.service(station, customer))



    #########################################################
    ################# Arrival block #########################
    #########################################################

    def customer_arrivals(self, station):

        while True:
            if self.give_rate_per_time(self.env.now) == 0:
                yield self.env.timeout(np.random.exponential(99999999999))
            else:
                yield self.env.timeout(np.random.exponential(1 / self.give_rate_per_time(self.env.now)))

            tot_time = self.env.now - self.last_event_time[station]
            self.num_cust_durations[station][self.num_cust_sys[station]] += tot_time

            curr_id = self.id_current
            arrival_time = self.env.now
            customer = Customer(curr_id, arrival_time, 1)

            self.event_log_customer_id_list[station].append(customer.id)
            self.event_log_time_stamp[station].append(self.env.now)

            self.last_event_time[station] = self.env.now

            self.num_cust_sys[station] += 1
            self.event_log_num_cust_list[station].append(self.num_cust_sys[station])
            self.event_log_type_list[station].append('Arrival')

            self.id_current += 1

            if is_print:
                print('Arrived customer {} at {}'.format(customer.id, self.env.now))

            self.env.process(self.service(station, customer))


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


#
# dict_sch = {0: 15, 481:1/12.5}
# sim_time = 120
# arrival_scd_pd = convert_dict_to_pd(dict_sch)
# num_stations = 3
# capacities = {1: 2, 2:2, 3:2}
# mu = 2*np.ones(num_stations)
# mu[0] = 10
# mu[1] = 15
# mu[2] = 20
#
#
# stations_res_list = {}
#
# for station in range(1,1+num_stations):
#     stations_res_list[station] = []
#
# for ind in tqdm(range(500)):
#
#     sim_res = single_sim_rate_arrival(arrival_scd_pd, mu, sim_time, capacities,  num_stations)
#
#     for station in range(1, 1 + num_stations):
#         stations_res_list[station].append(sim_res[station])
#
# probs = {}
# for station in range(1, 1 + num_stations):
#
#     probs[station] = compute_prob_arr(stations_res_list[station], sim_time,  max_num_customers=100)