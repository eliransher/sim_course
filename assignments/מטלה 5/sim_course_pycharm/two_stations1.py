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



class Queue_two_stations:

    def __init__(self, arrival_rates, service_rates, sim_time):
        self.env = simpy.Environment()  # initializing simpy enviroment
        self.server = simpy.Resource(self.env, capacity=1)  # Defining a resource with capacity 1
        self.end_time = sim_time  # The time simulation terminate
        self.id_current = 1  # keeping track of cusotmers id
        self.df_events = pd.DataFrame([])  # is a dataframe the holds all information of the queue dynamic:
        # an event can one of three: (1. arrival, 2. entering service 3. service completion)

        self.df_waiting_times = pd.DataFrame([])  # is a dataframe the holds all information waiting time
        self.service_rates = service_rates  # service rate
        self.arrival_rates = arrival_rates  # inter-arrival rate
        self.num_cust_durations = []
        self.df_waiting_times = []
        self.server = []
        self.last_event_time = []  # the time of the last event -
        self.num_cust_sys = []  # keeping track of number of customers in the system
        self.df_events = []  # is a dataframe the holds all information of the queue dynamic:

        for ind in range(2):
            self.server.append(simpy.Resource(self.env, capacity=1))  # Defining a resource with capacity 1
            self.num_cust_durations.append(
                np.zeros(500))  ## the time duration of each each state (state= number of cusotmers in the system)
            self.df_waiting_times.append(pd.DataFrame([]))  # is a dataframe the holds all information waiting time
            self.num_cust_sys.append(0)
            self.last_event_time.append(0)
            self.df_events.append(pd.DataFrame([]))  # is a dataframe the holds all information of the queue dynamic:

    def run(self):
        print('Starting simulating two stations system...')

        for station in range(2):
            for type_ in range(2):
                if self.arrival_rates[station, type_] > 0:
                    self.env.process(self.customer_arrivals(station, type_))  ## Initializing a process
        self.env.run(until=self.end_time)  ## Running the simulaiton until self.end_time
        print('Simulation end')

    def update_new_row(self, customer, event, station):

        new_row = {'Event': event, 'Time': self.env.now, 'Customer': customer.id,
                   'Queue lenght': len(self.server[station].queue), 'System lenght': self.num_cust_sys[station]
            , 'Customer type': customer.type}

        self.df_events[station] = pd.concat([self.df_events[station], pd.DataFrame([new_row])], ignore_index=True)

    #########################################################
    ################# Service block #########################
    #########################################################

    def service(self, customer, station):

        tot_time = self.env.now - self.last_event_time[station]
        self.num_cust_durations[station][self.num_cust_sys[station]] += tot_time
        self.num_cust_sys[station] += 1
        self.last_event_time[station] = self.env.now
        self.last_time = self.env.now

        with self.server[station].request() as req:
            # Updating the a new cusotmer arrived
            self.update_new_row(customer, 'Arrival', station)

            yield req

            # Updating the a new cusotmer entered service
            self.update_new_row(customer, 'Enter service', station)

            yield self.env.timeout(np.random.exponential(1 / self.service_rates[station, customer.type]))

            tot_time = self.env.now - self.last_event_time[station]  # keeping track of the last event
            self.num_cust_durations[station][
                self.num_cust_sys[station]] += tot_time  # Since the number of customers in the system changes
            # we compute how much time the system had this number of customers

            self.num_cust_sys[station] -= 1  # updating number of cusotmers in the system
            self.last_event_time[station] = self.env.now

            # Updating the a cusotmer departed the system
            self.update_new_row(customer, 'Departure', station)

            if is_print:
                print('Departed customer {} at {}'.format(customer.id, self.env.now))

            new_waiting_row = {'Customer': customer.id, 'WaitingTime': self.env.now - customer.arrival_time}

            self.df_waiting_times[station] = pd.concat(
                [self.df_waiting_times[station], pd.DataFrame([new_waiting_row])],
                ignore_index=True)

        ## if type 0 customer arrived at station 1 then he is moved to station 1
        if (station == 1) & (customer.type == 0):
            customer.arrival_time = self.env.now
            self.env.process(self.service(customer, 0))

    #########################################################
    ################# Arrival block #########################
    #########################################################

    def customer_arrivals(self, station, customer_type):

        while True:

            yield self.env.timeout(np.random.exponential(1 / self.arrival_rates[station, customer_type]))

            curr_id = self.id_current
            arrival_time = self.env.now
            customer = Customer(curr_id, arrival_time, customer_type)

            self.id_current += 1

            if is_print:
                print('Arrived customer {} at {}'.format(customer.id, self.env.now))

            self.env.process(self.service(customer, station))

    def get_average_waiting_time_two_station(self):
        return (self.df_waiting_times[0]['WaitingTime'].mean(), self.df_waiting_times[1]['WaitingTime'].mean())

arrival_rates = np.array([[1,1],[1,1]])
service_rates = np.array([[3,4],[6,4]])

sim_time = 7000
queue_two_stations = Queue_two_stations(arrival_rates, service_rates, sim_time)
queue_two_stations.run()

print('Stop')