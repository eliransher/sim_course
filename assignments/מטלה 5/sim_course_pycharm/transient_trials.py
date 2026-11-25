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








def single_sim_rate_arrival_service(arrival_scd_pd, mu_schedule, sim_time, num_stations):

    sim_time += 1
    sim_instance = Service_arrival_change(arrival_scd_pd, mu_schedule, sim_time, num_stations)
    sim_instance.run()
    resultDictionary = {}

    for station in range(1,1+num_stations):

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


is_print = False





class Service_arrival_change:

    def __init__(self, arrival_scd_pd, mu_schedule, sim_time, num_stations):

        self.env = simpy.Environment()  # initializing simpy enviroment

        self.servers = {}
        self.mu_schedule_per_station = {}
        self.num_cust_durations = {}
        self.last_event_time = {}  # the time of the last event -
        self.last_time = {}
        self.num_cust_sys = {}
        self.event_log_customer_id_list = {}
        self.event_log_num_cust_list = {}
        self.event_log_type_list = {}
        self.event_log_time_stamp = {}



        for ind in range(1,1+num_stations):
            self.servers[ind] = simpy.Resource(self.env, capacity=1)  # Defining a resource with capacity 1
            self.mu_schedule_per_station[ind] = mu_schedule[ind]
            self.num_cust_durations[ind] = np.zeros(500)
            self.last_time[ind] = 0
            self.last_event_time[ind] = 0  # the time of the last event -
            self.num_cust_sys[ind] = 0
            self.event_log_customer_id_list[ind] = []
            self.event_log_num_cust_list[ind] = []
            self.event_log_type_list[ind] = []
            self.event_log_time_stamp[ind] = []



        self.end_time = sim_time  # The time simulation terminate
        self.id_current = 1

         # keeping track of number of customers in the system
        self.num_stations = num_stations

        self.arrival_pd = arrival_scd_pd

    def run(self):

        self.env.process(self.customer_arrivals(1))  ## Initializing a process
        self.env.run(until=self.end_time)  ## Running the simulaiton until self.end_time

    def give_rate_per_time(self, time, df_schedule):

        ind_rate = df_schedule.loc[df_schedule['LB_time'] <= time, :].index[-1]
        return df_schedule.loc[ind_rate, 'arrival_rate'].item()

    #########################################################
    ################# Service block #########################
    #########################################################

    def service(self, customer, station):

        with self.servers[station].request() as req:
            yield req

            yield self.env.timeout(np.random.exponential(1 / self.give_rate_per_time(self.env.now, self.mu_schedule_per_station[station])))

            tot_time = self.env.now - self.last_event_time[station]  # keeping track of the last event
            self.num_cust_durations[station][self.num_cust_sys[station]] += tot_time

            self.num_cust_sys[station] -= 1  # updating number of cusotmers in the system
            self.last_event_time[station] = self.env.now
            self.last_time[station] = self.env.now
            self.event_log_customer_id_list[station].append(customer.id)
            self.event_log_time_stamp[station].append(self.env.now)
            self.event_log_num_cust_list[station].append(self.num_cust_sys[station])
            self.event_log_type_list[station].append('Departure')

            if is_print:
                print('Departed customer {} at {}'.format(customer.id, self.env.now))

        if station < self.num_stations:
            station += 1
            self.event_log_customer_id_list[station].append(customer.id)
            self.event_log_time_stamp[station].append(self.env.now)

            self.last_event_time[station] = self.env.now
            self.last_time[station] = self.env.now

            self.num_cust_sys[station] += 1
            self.event_log_num_cust_list[station].append(self.num_cust_sys[station])
            self.event_log_type_list[station].append('Arrival')

            self.env.process(self.service(customer, station))


    #########################################################
    ################# Arrival block #########################
    #########################################################

    def customer_arrivals(self, station):

        while True:

            yield self.env.timeout(np.random.exponential(1 / self.give_rate_per_time(self.env.now, self.arrival_pd)))

            tot_time = self.env.now - self.last_event_time[station]
            self.num_cust_durations[station][self.num_cust_sys[station]] += tot_time

            curr_id = self.id_current
            arrival_time = self.env.now
            customer = Customer(curr_id, arrival_time, 1)

            self.event_log_customer_id_list[station].append(customer.id)
            self.event_log_time_stamp[station].append(self.env.now)

            self.last_event_time[station] = self.env.now
            self.last_time[station] = self.env.now

            self.num_cust_sys[station] += 1
            self.event_log_num_cust_list[station].append(self.num_cust_sys[station])
            self.event_log_type_list[station].append('Arrival')

            self.id_current += 1

            if is_print:
                print('Arrived customer {} at {}'.format(customer.id, self.env.now))

            self.env.process(self.service(customer, station))


is_print = False


## Defining a new class: of customers
class Customer:
    def __init__(self, p_id, arrival_time, type_cust):
        self.id = p_id
        self.arrival_time = arrival_time
        self.type = type_cust


is_print = False

dict_sch_arrival = {0: 1, 12: 0.5, 20: 0.9}

arrival_scd_pd = convert_dict_to_pd(dict_sch_arrival)

mu_scheduler = {}
for station in range(1,3):
    dict_sch_service = {0: 2, 2: 1}
    mu_scheduler[station] = convert_dict_to_pd(dict_sch_service)

sim_time = 20
num_stations = 2
list_of_sims_1 = []
for station in tqdm(range(100)):

    list_of_sims_1.append(single_sim_rate_arrival_service(arrival_scd_pd, mu_scheduler, sim_time,num_stations))

list_per_station = {}
for station in range(1, 1+num_stations):
    list_per_station[station] = []
    for ind in range(len(list_of_sims_1)):
        list_per_station[station].append(list_of_sims_1[ind][station])

for station in range(1, num_stations):
    dist_option_1 = give_num_cust_dist(sim_time, list_per_station[station])

print('cd')





