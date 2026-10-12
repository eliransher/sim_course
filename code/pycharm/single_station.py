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


is_print = False

## Defining a new class: of customers
class Customer:
    def __init__(self, p_id, arrival_time, type_cust):
        self.id = p_id
        self.arrival_time = arrival_time
        self.type = type_cust

is_print = False

class Queue_single_station:

    def __init__(self, lamda, mu, sim_time):

        self.env = simpy.Environment() # initializing simpy enviroment 
        self.server = simpy.Resource(self.env, capacity=1) # Defining a resource with capacity 1
        self.end_time = sim_time # The time simulation terminate
        self.id_current = 1 # keeping track of cusotmers id
        self.df_events = pd.DataFrame([]) # is a dataframe the holds all information of the queue dynamic:
        #an event can one of three: (1. arrival, 2. entering service 3. service completion)
        
        self.df_waiting_times = pd.DataFrame([]) # is a dataframe the holds all information waiting time
        self.mu = mu # service rate
        self.lamda = lamda # inter-arrival rate
        self.num_cust_durations = np.zeros(500) ## the time duration of each each state (state= number of cusotmers in the system)
        self.last_event_time = 0 # the time of the last event - 
        self.num_cust_sys = 0 # keeping track of number of customers in the system


    def run(self):

        self.env.process(self.customer_arrivals()) ## Initializing a process
        self.env.run(until=self.end_time) ## Running the simulaiton until self.end_time

    
    def update_new_row(self, customer, event):
        
        new_row = {'Event': event, 'Time': self.env.now, 'Customer': customer.id,
                       'Queue lenght': len(self.server.queue), 'System lenght': self.num_cust_sys}

        self.df_events = pd.concat([self.df_events, pd.DataFrame([new_row])], ignore_index=True)
    
    
    #########################################################
    ################# Service block ######################### 
    ######################################################### 
    
    
    def service(self, customer):

        with self.server.request() as req:
                       
            # Updating the a new cusotmer arrived
            self.update_new_row(customer, 'Arrival')

            yield req
            
            # Updating the a new cusotmer entered service 
            self.update_new_row(customer, 'Enter service')

            yield self.env.timeout(np.random.exponential(1/self.mu))

            tot_time = self.env.now - self.last_event_time  # keeping track of the last event
            self.num_cust_durations[self.num_cust_sys] += tot_time # Since the number of customers in the system changes
            #we compute how much time the system had this number of customers
            
            self.num_cust_sys -= 1 # updating number of cusotmers in the system
            self.last_event_time = self.env.now

            # Updating the a cusotmer departed the system
            self.update_new_row( customer, 'Departure')
            

            if is_print:
                print('Departed customer {} at {}'.format(customer.id, self.env.now))

            new_waiting_row = {'Customer': customer.id, 'WaitingTime': self.env.now-customer.arrival_time}

            self.df_waiting_times = pd.concat([self.df_waiting_times, pd.DataFrame([new_waiting_row])], ignore_index=True)
            
            


    #########################################################
    ################# Arrival block ######################### 
    ######################################################### 
    
    
    def customer_arrivals(self):

        while True:

            yield self.env.timeout(np.random.exponential(1/self.lamda))

            tot_time = self.env.now - self.last_event_time
            self.num_cust_durations[self.num_cust_sys] += tot_time
            self.num_cust_sys += 1
            self.last_event_time = self.env.now
            self.last_time = self.env.now


            curr_id = self.id_current
            arrival_time = self.env.now
            customer = Customer(curr_id,  arrival_time, 1)

            self.id_current += 1

            if is_print:
                print('Arrived customer {} at {}'.format(customer.id, self.env.now))

            self.env.process(self.service(customer))
            
    def get_steady_single_station(self):
        return self.num_cust_durations/self.num_cust_durations.sum()


