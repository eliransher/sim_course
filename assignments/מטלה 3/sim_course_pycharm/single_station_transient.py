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
    for hour in range(sim_time+1):
        time_dict[hour] = np.zeros(max_num_customers)

    for resultDictionary in list_of_sims:
        for hour in resultDictionary.keys():
            time_dict[hour][resultDictionary[hour]] += 1
            
            
    # phase 2: 
    prob_queue_arr = np.array([])

    for t in range(len(time_dict)):

        probs = (time_dict[t]/time_dict[t].sum())

        if t == 0:
            prob_queue_arr = np.array(probs).reshape(1, probs.shape[0])
        else:
            prob_queue_arr = np.concatenate((prob_queue_arr, np.array(probs).reshape(1,probs.shape[0])), axis = 0)
    
    return prob_queue_arr


def give_rate_per_time(df,  time):
        
        ind_rate = df.loc[df['LB_time']<=time, :].index[-1]
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

def single_sim(lamda, mu, sim_time):
    sim_time +=1
    queue_single_station = Queue_single_station(lamda, mu, sim_time)
    queue_single_station.run()
    event_log = pd.DataFrame({'Customer_id': queue_single_station.event_log_customer_id_list,
                                  'Time_stamp': queue_single_station.event_log_time_stamp,
                                  'Type': queue_single_station.event_log_type_list,
                                  'num_cust': queue_single_station.event_log_num_cust_list})
    
    result = [(time_epoch, find_num_cust_time_stamp(event_log, time_epoch)) for time_epoch in range(sim_time)]
    resultDictionary = dict((x, y) for x, y in result)
    
    return resultDictionary


def single_sim_single_rate(arrivals, mu, sim_time):
    sim_time +=1
    sim_instance = Trans_single_rate(arrivals, mu, sim_time)
    sim_instance.run()
    event_log = pd.DataFrame({'Customer_id': sim_instance.event_log_customer_id_list,
                                  'Time_stamp': sim_instance.event_log_time_stamp,
                                  'Type': sim_instance.event_log_type_list,
                                  'num_cust': sim_instance.event_log_num_cust_list})
    
    result = [(time_epoch, find_num_cust_time_stamp(event_log, time_epoch)) for time_epoch in range(sim_time)]
    resultDictionary = dict((x, y) for x, y in result)
    
    return resultDictionary


def single_sim_rate_arrival_service(arrival_scd_pd, mu_schedule, sim_time):
    
    sim_time +=1
    sim_instance = Service_arrival_change(arrival_scd_pd, mu_schedule, sim_time)
    sim_instance.run()
    event_log = pd.DataFrame({'Customer_id': sim_instance.event_log_customer_id_list,
                                  'Time_stamp': sim_instance.event_log_time_stamp,
                                  'Type': sim_instance.event_log_type_list,
                                  'num_cust': sim_instance.event_log_num_cust_list})
    
    result = [(time_epoch, find_num_cust_time_stamp(event_log, time_epoch)) for time_epoch in range(sim_time)]
    resultDictionary = dict((x, y) for x, y in result)
    
    return resultDictionary


is_print = False

## Defining a new class: of customers
class Customer:
    def __init__(self, p_id, arrival_time, type_cust):
        self.id = p_id
        self.arrival_time = arrival_time
        self.type = type_cust

is_print = False

class Trans_single_rate:

    def __init__(self, lamda, mu, sim_time):

        self.env = simpy.Environment() # initializing simpy enviroment 
        self.server = simpy.Resource(self.env, capacity=1) # Defining a resource with capacity 1
        self.end_time = sim_time # The time simulation terminate
        self.id_current = 1 # keeping track of cusotmers id
        self.df_events = pd.DataFrame([]) # is a dataframe the holds all information of the queue dynamic:
        #an event can one of three: (1. arrival, 2. entering service 3. service completion)
        
        self.df_waiting_times = pd.DataFrame([]) # is a dataframe the holds all information waiting time
        self.mu = mu # service rate
        self.lamda = lamda # arrival rate
        self.num_cust_durations = np.zeros(500) ## the time duration of each each state (state= number of cusotmers in the system)
        self.last_event_time = 0 # the time of the last event - 
        self.num_cust_sys = 0 # keeping track of number of customers in the system
                
        self.event_log_customer_id_list = []
        self.event_log_num_cust_list = []
        self.event_log_type_list = []
        self.event_log_time_stamp = []
                
    def run(self):

        self.env.process(self.customer_arrivals()) ## Initializing a process
        self.env.run(until=self.end_time) ## Running the simulaiton until self.end_time

   
    
    def give_rate_per_time(self, time):
        
        ind_rate = self.arrival_pd.loc[self.arrival_pd['LB_time']<=time, :].index[-1]
        return self.arrival_pd.loc[ind_rate, 'arrival_rate']
        
    
    #########################################################
    ################# Service block ######################### 
    ######################################################### 
    
    
    def service(self, customer):

        with self.server.request() as req:
                       

            yield req

            yield self.env.timeout(np.random.exponential(1/self.mu))

            tot_time = self.env.now - self.last_event_time  # keeping track of the last event
            self.num_cust_durations[self.num_cust_sys] += tot_time # Since the number of customers in the system changes
            #we compute how much time the system had this number of customers
            
            self.num_cust_sys -= 1 # updating number of cusotmers in the system
            self.last_event_time = self.env.now

            self.last_time = self.env.now

            self.event_log_customer_id_list.append(customer.id)
            self.event_log_time_stamp.append(self.env.now)
            self.event_log_num_cust_list.append(self.num_cust_sys)
            self.event_log_type_list.append('Departure')
                       

            if is_print:
                print('Departed customer {} at {}'.format(customer.id, self.env.now))
           

    #########################################################
    ################# Arrival block ######################### 
    ######################################################### 
    
    
    def customer_arrivals(self):

        while True:

            yield self.env.timeout(np.random.exponential(1/self.lamda))

            tot_time = self.env.now - self.last_event_time
            self.num_cust_durations[self.num_cust_sys] += tot_time
            
            curr_id = self.id_current
            arrival_time = self.env.now
            customer = Customer(curr_id,  arrival_time, 1)
            
            self.event_log_customer_id_list.append(customer.id)
            self.event_log_time_stamp.append(self.env.now)
            
            self.last_event_time = self.env.now
            self.last_time = self.env.now         
            
            self.num_cust_sys += 1
            self.event_log_num_cust_list.append(self.num_cust_sys)
            self.event_log_type_list.append('Arrival')

            self.id_current += 1

            if is_print:
                print('Arrived customer {} at {}'.format(customer.id, self.env.now))

            self.env.process(self.service(customer))
            



is_print = False

## Defining a new class: of customers
class Customer:
    def __init__(self, p_id, arrival_time, type_cust):
        self.id = p_id
        self.arrival_time = arrival_time
        self.type = type_cust

is_print = False

class Service_arrival_change:

    def __init__(self, arrival_scd_pd, mu_schedule, sim_time):

        self.env = simpy.Environment() # initializing simpy enviroment 
        self.server = simpy.Resource(self.env, capacity=1) # Defining a resource with capacity 1
        self.end_time = sim_time # The time simulation terminate
        self.id_current = 1 # keeping track of cusotmers id
        self.df_events = pd.DataFrame([]) # is a dataframe the holds all information of the queue dynamic:
        #an event can one of three: (1. arrival, 2. entering service 3. service completion)
        
        self.df_waiting_times = pd.DataFrame([]) # is a dataframe the holds all information waiting time
        self.mu_schedule = mu_schedule # service rate
        self.num_cust_durations = np.zeros(500) ## the time duration of each each state (state= number of cusotmers in the system)
        self.last_event_time = 0 # the time of the last event - 
        self.num_cust_sys = 0 # keeping track of number of customers in the system
        
        
        self.event_log_customer_id_list = []
        self.event_log_num_cust_list = []
        self.event_log_type_list = []
        self.event_log_time_stamp = []
        
        self.arrival_pd = arrival_scd_pd

    def run(self):

        self.env.process(self.customer_arrivals()) ## Initializing a process
        self.env.run(until=self.end_time) ## Running the simulaiton until self.end_time

   
    
    def give_rate_per_time(self, time, df_schedule):
        
        ind_rate = df_schedule.loc[df_schedule['LB_time']<=time, :].index[-1]
        return df_schedule.loc[ind_rate, 'arrival_rate']
        
    
    #########################################################
    ################# Service block ######################### 
    ######################################################### 
    
    
    def service(self, customer):

        with self.server.request() as req:
                       

            yield req

            yield self.env.timeout(np.random.exponential(1/self.give_rate_per_time(self.env.now, self.mu_schedule)))

            tot_time = self.env.now - self.last_event_time  # keeping track of the last event
            self.num_cust_durations[self.num_cust_sys] += tot_time # Since the number of customers in the system changes
            #we compute how much time the system had this number of customers
            
            self.num_cust_sys -= 1 # updating number of cusotmers in the system
            self.last_event_time = self.env.now
            self.last_time = self.env.now
            self.event_log_customer_id_list.append(customer.id)
            self.event_log_time_stamp.append(self.env.now)
            self.event_log_num_cust_list.append(self.num_cust_sys)
            self.event_log_type_list.append('Departure')
                       

            if is_print:
                print('Departed customer {} at {}'.format(customer.id, self.env.now))
           

    #########################################################
    ################# Arrival block ######################### 
    ######################################################### 
    
    
    def customer_arrivals(self):

        while True:

            yield self.env.timeout(np.random.exponential(1/self.give_rate_per_time(self.env.now,self.arrival_pd)))

            tot_time = self.env.now - self.last_event_time
            self.num_cust_durations[self.num_cust_sys] += tot_time
            
            curr_id = self.id_current
            arrival_time = self.env.now
            customer = Customer(curr_id,  arrival_time, 1)
            
            self.event_log_customer_id_list.append(customer.id)
            self.event_log_time_stamp.append(self.env.now)
            
            self.last_event_time = self.env.now
            self.last_time = self.env.now         
            
            self.num_cust_sys += 1
            self.event_log_num_cust_list.append(self.num_cust_sys)
            self.event_log_type_list.append('Arrival')

            self.id_current += 1

            if is_print:
                print('Arrived customer {} at {}'.format(customer.id, self.env.now))

            self.env.process(self.service(customer))
            



is_print = False

## Defining a new class: of customers
class Customer:
    def __init__(self, p_id, arrival_time, type_cust):
        self.id = p_id
        self.arrival_time = arrival_time
        self.type = type_cust

is_print = False

class Queue_single_station:

    def __init__(self, arrival_scd_pd, mu, sim_time):

        self.env = simpy.Environment() # initializing simpy enviroment 
        self.server = simpy.Resource(self.env, capacity=1) # Defining a resource with capacity 1
        self.end_time = sim_time # The time simulation terminate
        self.id_current = 1 # keeping track of cusotmers id
        self.df_events = pd.DataFrame([]) # is a dataframe the holds all information of the queue dynamic:
        #an event can one of three: (1. arrival, 2. entering service 3. service completion)
        
        self.df_waiting_times = pd.DataFrame([]) # is a dataframe the holds all information waiting time
        self.mu = mu # service rate
        self.num_cust_durations = np.zeros(500) ## the time duration of each each state (state= number of cusotmers in the system)
        self.last_event_time = 0 # the time of the last event - 
        self.num_cust_sys = 0 # keeping track of number of customers in the system
        
        
        self.event_log_customer_id_list = []
        self.event_log_num_cust_list = []
        self.event_log_type_list = []
        self.event_log_time_stamp = []
        
        self.arrival_pd = arrival_scd_pd

    def run(self):

        self.env.process(self.customer_arrivals()) ## Initializing a process
        self.env.run(until=self.end_time) ## Running the simulaiton until self.end_time

   
    
    def give_rate_per_time(self, time):
        
        ind_rate = self.arrival_pd.loc[self.arrival_pd['LB_time']<=time, :].index[-1]
        return self.arrival_pd.loc[ind_rate, 'arrival_rate']
        
    
    #########################################################
    ################# Service block ######################### 
    ######################################################### 
    
    
    def service(self, customer):

        with self.server.request() as req:
                       

            yield req

            yield self.env.timeout(np.random.exponential(1/self.mu))

            tot_time = self.env.now - self.last_event_time  # keeping track of the last event
            self.num_cust_durations[self.num_cust_sys] += tot_time # Since the number of customers in the system changes
            #we compute how much time the system had this number of customers
            
            self.num_cust_sys -= 1 # updating number of cusotmers in the system
            self.last_event_time = self.env.now

            self.last_time = self.env.now

            self.event_log_customer_id_list.append(customer.id)
            self.event_log_time_stamp.append(self.env.now)
            self.event_log_num_cust_list.append(self.num_cust_sys)
            self.event_log_type_list.append('Departure')
                       

            if is_print:
                print('Departed customer {} at {}'.format(customer.id, self.env.now))
           

    #########################################################
    ################# Arrival block ######################### 
    ######################################################### 
    
    
    def customer_arrivals(self):

        while True:

            yield self.env.timeout(np.random.exponential(1/self.give_rate_per_time(self.env.now)))

            tot_time = self.env.now - self.last_event_time
            self.num_cust_durations[self.num_cust_sys] += tot_time
            
            curr_id = self.id_current
            arrival_time = self.env.now
            customer = Customer(curr_id,  arrival_time, 1)
            
            self.event_log_customer_id_list.append(customer.id)
            self.event_log_time_stamp.append(self.env.now)
            
            self.last_event_time = self.env.now
            self.last_time = self.env.now         
            
            self.num_cust_sys += 1
            self.event_log_num_cust_list.append(self.num_cust_sys)
            self.event_log_type_list.append('Arrival')

            self.id_current += 1

            if is_print:
                print('Arrived customer {} at {}'.format(customer.id, self.env.now))

            self.env.process(self.service(customer))
            



class Queue_multi_stations:

    def __init__(self, arrival_scd_pd, mu, sim_time, num_stations):

        self.env = simpy.Environment() # initializing simpy enviroment 
        self.server = simpy.Resource(self.env, capacity=1) # Defining a resource with capacity 1
        self.end_time = sim_time # The time simulation terminate
        self.id_current = 1 # keeping track of cusotmers id
        self.df_events = pd.DataFrame([]) # is a dataframe the holds all information of the queue dynamic:
        #an event can one of three: (1. arrival, 2. entering service 3. service completion)
        
        self.df_waiting_times = pd.DataFrame([]) # is a dataframe the holds all information waiting time
        self.mu = mu # service rate
        self.num_cust_durations = np.zeros(500) ## the time duration of each each state (state= number of cusotmers in the system)
        self.last_event_time = 0 # the time of the last event - 
        self.num_cust_sys = 0 # keeping track of number of customers in the system
        
        
        self.event_log_customer_id_list = []
        self.event_log_num_cust_list = []
        self.event_log_type_list = []
        self.event_log_time_stamp = []
        
        self.arrival_pd = arrival_scd_pd

    def run(self):

        self.env.process(self.customer_arrivals()) ## Initializing a process
        self.env.run(until=self.end_time) ## Running the simulaiton until self.end_time

   
    
    def give_rate_per_time(self, time):
        
        ind_rate = self.arrival_pd.loc[self.arrival_pd['LB_time']<=time, :].index[-1]
        return self.arrival_pd.loc[ind_rate, 'arrival_rate']
        
    
    #########################################################
    ################# Service block ######################### 
    ######################################################### 
    
    
    def service(self, customer):

        with self.server.request() as req:
                       

            yield req

            yield self.env.timeout(np.random.exponential(1/self.mu))

            tot_time = self.env.now - self.last_event_time  # keeping track of the last event
            self.num_cust_durations[self.num_cust_sys] += tot_time # Since the number of customers in the system changes
            #we compute how much time the system had this number of customers
            
            self.num_cust_sys -= 1 # updating number of cusotmers in the system
            self.last_event_time = self.env.now

            self.last_time = self.env.now

            self.event_log_customer_id_list.append(customer.id)
            self.event_log_time_stamp.append(self.env.now)
            self.event_log_num_cust_list.append(self.num_cust_sys)
            self.event_log_type_list.append('Departure')
                       

            if is_print:
                print('Departed customer {} at {}'.format(customer.id, self.env.now))
           

    #########################################################
    ################# Arrival block ######################### 
    ######################################################### 
    
    
    def customer_arrivals(self):

        while True:

            yield self.env.timeout(np.random.exponential(1/self.give_rate_per_time(self.env.now)))

            tot_time = self.env.now - self.last_event_time
            self.num_cust_durations[self.num_cust_sys] += tot_time
            
            curr_id = self.id_current
            arrival_time = self.env.now
            customer = Customer(curr_id,  arrival_time, 1)
            
            self.event_log_customer_id_list.append(customer.id)
            self.event_log_time_stamp.append(self.env.now)
            
            self.last_event_time = self.env.now
            self.last_time = self.env.now         
            
            self.num_cust_sys += 1
            self.event_log_num_cust_list.append(self.num_cust_sys)
            self.event_log_type_list.append('Arrival')

            self.id_current += 1

            if is_print:
                print('Arrived customer {} at {}'.format(customer.id, self.env.now))

            self.env.process(self.service(customer))
            

def compute_transient_probs( mu, arrival_rate, sim_time, num_sims, shut_down = 8):
    dict_sch = {0: arrival_rate, shut_down: 0.0000000000000005}
    arrival_scd_pd = convert_dict_to_pd(dict_sch)
    list_of_sims = []
    for ind in tqdm(range(num_sims)):
        list_of_sims.append(single_sim(arrival_scd_pd, mu, sim_time))

    prob_queue_arr = compute_prob_arr(list_of_sims, sim_time)

    return prob_queue_arr
            
            