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


class Queue_two_stations_markov_trans:

    def __init__(self, arrival_rates, service_rates, probs, sim_time, df_event_track = False, capacities = [1,1]):
        self.env = simpy.Environment()  # initializing simpy enviroment
        self.end_time = sim_time  # The time simulation terminate
        self.id_current = 1  # keeping track of cusotmers id
        self.df_events = pd.DataFrame([])  # is a dataframe the holds all information of the queue dynamic:
        # an event can one of three: (1. arrival, 2. entering service 3. service completion)

        self.df_event_track = df_event_track
        self.capacities = capacities
        self.df_waiting_times = pd.DataFrame([])  # is a dataframe the holds all information waiting time
        self.service_rates = service_rates  # service rate
        self.arrival_rates = arrival_rates  # inter-arrival rate
        self.probs = probs # transition probs
        self.num_cust_durations = []
        self.df_waiting_times = []
        self.server = []
        self.last_event_time = []  # the time of the last event -
        self.num_cust_sys = []  # keeping track of number of customers in the system
        self.df_events = []  # is a dataframe the holds all information of the queue dynamic:

        self.waiting_times = {}

        for ind in range(2):
            self.server.append(simpy.Resource(self.env, capacity=self.capacities[ind]))  # Defining a resource with capacity 1
            self.num_cust_durations.append(
                np.zeros(5000000))  ## the time duration of each each state (state= number of cusotmers in the system)
            self.df_waiting_times.append(pd.DataFrame([]))  # is a dataframe the holds all information waiting time
            self.num_cust_sys.append(0)
            self.last_event_time.append(0)
            self.df_events.append(pd.DataFrame([]))  # is a dataframe the holds all information of the queue dynamic:
            self.waiting_times[ind] = []
    def run(self):
        print('Starting simulating two stations system...')

        for station in range(2):
            if self.arrival_rates[station] > 0:
                self.env.process(self.customer_arrivals(station, 1))  ## Initializing a process

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
            if self.df_event_track:
                self.update_new_row(customer, 'Arrival', station)

            yield req

            # Updating the a new cusotmer entered service
            if self.df_event_track:
                self.update_new_row(customer, 'Enter service', station)

            yield self.env.timeout(np.random.exponential(1 / self.service_rates[station]))

            tot_time = self.env.now - self.last_event_time[station]  # keeping track of the last event
            self.num_cust_durations[station][
                self.num_cust_sys[station]] += tot_time  # Since the number of customers in the system changes
            # we compute how much time the system had this number of customers

            self.num_cust_sys[station] -= 1  # updating number of cusotmers in the system
            self.last_event_time[station] = self.env.now

            # Updating the a cusotmer departed the system
            if self.df_event_track:
                self.update_new_row(customer, 'Departure', station)

            if is_print:
                print('Departed customer {} at {}'.format(customer.id, self.env.now))

            # new_waiting_row = {'Customer': customer.id, 'WaitingTime': self.env.now - customer.arrival_time}
            #
            # self.df_waiting_times[station] = pd.concat(
            #     [self.df_waiting_times[station], pd.DataFrame([new_waiting_row])],
            #     ignore_index=True)

            self.waiting_times[station].append(self.env.now - customer.arrival_time)

        ## next station - if 2 means exit the system:
        all_probs = np.append(self.probs[station], 1 - self.probs[station].sum())

        next_station = np.random.choice(3, 1, p=all_probs).item()
        if next_station < all_probs.shape[0]-1:
            customer.arrival_time = self.env.now
            self.env.process(self.service(customer, next_station))

    #########################################################
    ################# Arrival block #########################
    #########################################################

    def customer_arrivals(self, station, customer_type):

        while True:

            yield self.env.timeout(np.random.exponential(1 / self.arrival_rates[station]))

            curr_id = self.id_current
            arrival_time = self.env.now
            customer = Customer(curr_id, arrival_time, customer_type)

            self.id_current += 1

            if is_print:
                print('Arrived customer {} at {}'.format(customer.id, self.env.now))

            self.env.process(self.service(customer, station))

    def get_average_waiting_time_two_station(self):
        return (self.df_waiting_times[0]['WaitingTime'].mean(), self.df_waiting_times[1]['WaitingTime'].mean())

# arrival_rates = np.array([1, 1])
# service_rates = np.array([1.25, 1.25])
# probs = np.array([[0, 0.5], [0, 0]])
#
# sim_time = 1000
# queue_two_stations_markov_trans = Queue_two_stations_markov_trans(arrival_rates, service_rates, probs, sim_time)
# queue_two_stations_markov_trans.run()
#
#
# print(queue_two_stations_markov_trans.get_average_waiting_time_two_station())
# print('Stop')