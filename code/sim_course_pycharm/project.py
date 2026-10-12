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
import pickle as pkl
import scipy.stats as st
is_print = False


## Defining a new class: of customers
class Customer:
    def __init__(self, p_id, arrival_time,   type_cust):
        self.id = p_id
        self.arrival_time = arrival_time
        self.original_arrival = arrival_time
        self.type = type_cust


class Queue_open_network_markov_trans:

    def __init__(self, arrivals, services, probs, arrival_rates, sim_time, df_event_track = False, capacities = [1,1,1]):
        self.env = simpy.Environment()  # initializing simpy enviroment
        self.end_time = sim_time  # The time simulation terminate
        self.id_current = 1  # keeping track of cusotmers id
        self.df_events = pd.DataFrame([])  # is a dataframe the holds all information of the queue dynamic:
        # an event can one of three: (1. arrival, 2. entering service 3. service completion)
        self.num_stations = arrival_rates.shape[0]
        self.df_event_track = df_event_track
        self.capacities = capacities
        self.df_waiting_times = pd.DataFrame([])  # is a dataframe the holds all information waiting time
        self.arrival_rates = arrival_rates
        self.services = services  # service rate
        self.arrivals = arrivals  # inter-arrival rate
        self.sojourn_tot = []
        self.probs = probs  # transition probs
        self.num_cust_durations = []
        self.df_waiting_times = []
        self.server = []
        self.last_event_time = []  # the time of the last event -
        self.num_cust_sys = []  # keeping track of number of customers in the system
        self.df_events = []  # is a dataframe the holds all information of the queue dynamic:

        self.waiting_times = {}

        for ind in range(self.num_stations):
            self.server.append(simpy.Resource(self.env, capacity=self.capacities[ind]))  # Defining a resource with capacity 1
            self.num_cust_durations.append(
                np.zeros(500000))  ## the time duration of each each state (state= number of cusotmers in the system)
            self.df_waiting_times.append(pd.DataFrame([]))  # is a dataframe the holds all information waiting time
            self.num_cust_sys.append(0)
            self.last_event_time.append(0)
            self.df_events.append(pd.DataFrame([]))  # is a dataframe the holds all information of the queue dynamic:
            self.waiting_times[ind] = []

    def run(self):
        print('Starting simulating open network...')

        for station in range(self.num_stations):
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

            ind_ser = np.random.randint(self.services[station].shape[0])
            yield self.env.timeout(self.services[station][ind_ser])


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


            self.waiting_times[station].append(self.env.now - customer.arrival_time)

        ## next station - if 3 means exit the system:
        all_probs = np.append(self.probs[station], 1 - self.probs[station].sum())

        next_station = np.random.choice(self.num_stations+1, 1, p=all_probs).item()
        if next_station < all_probs.shape[0]-1:
            customer.arrival_time = self.env.now
            self.env.process(self.service(customer, next_station))

        else:
            self.sojourn_tot.append(self.env.now-customer.original_arrival)

    #########################################################
    ################# Arrival block #########################
    #########################################################

    def customer_arrivals(self, station, customer_type):

        while True:

            ind_arrival = np.random.randint(self.arrivals.shape[0])
            yield self.env.timeout(self.arrivals[ind_arrival])

            curr_id = self.id_current
            arrival_time = self.env.now
            customer = Customer(curr_id, arrival_time, customer_type)

            self.id_current += 1

            if is_print:
                print('Arrived customer {} at {}'.format(customer.id, self.env.now))

            self.env.process(self.service(customer, station))

    def get_average_waiting_time_two_station(self):

        avg_waiting_time = []

        for ind in range(self.num_stations):
            avg_waiting_time.append(np.array(self.waiting_times[ind]).mean())

        return avg_waiting_time

    def get_steady_single_station(self, station):
        return self.num_cust_durations[station] / self.num_cust_durations[station].sum()


def give_CI_length(data):
    low, high = st.t.interval(alpha=0.95, df=len(data)-1,
              loc=np.mean(data),
              scale=st.sem(data))
    return high - low


# arrivals_3_stations, services_3_stations, p_trans,  arrival_rates = pkl.load(open('../pkls/3_stations_data_sim_input.pkl', 'rb'))
# # p_trans[0,1] = 0.0
# sim_time = 10000
# sojourn_time_station_tot = []
#
# for ind in tqdm(range(10)):
#
#     queue_open_network_markov_trans = Queue_open_network_markov_trans(arrivals_3_stations, services_3_stations, p_trans, arrival_rates, sim_time, df_event_track = False)
#     queue_open_network_markov_trans.run()
#     sojourn_time_station_tot.append(np.array(queue_open_network_markov_trans.sojourn_tot).mean())
#     print(queue_open_network_markov_trans.get_average_waiting_time_two_station())
#     print(queue_open_network_markov_trans.get_steady_single_station(0)[0])
#     print(queue_open_network_markov_trans.get_steady_single_station(1)[0])
#     print(queue_open_network_markov_trans.get_steady_single_station(2)[0])
#     print(np.array(queue_open_network_markov_trans.sojourn_tot).mean())
#
# print(give_CI_length(sojourn_time_station_tot))
#
#
# # pkl.dump((queue_open_network_markov_trans.df_events), open('../pkls/df_data.pkl', 'wb'))