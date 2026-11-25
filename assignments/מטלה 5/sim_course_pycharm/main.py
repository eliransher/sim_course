
# imports
import simpy
import numpy as np
import sys
import pandas as pd
import os



class Customer:
    def __init__(self, p_id,  arrival_time):
        self.id = p_id
        self.arrival_time = arrival_time


is_print = False


class Customer:
        def __init__(self, p_id,  arrival_time):
            self.id = p_id
            self.arrival_time = arrival_time

class Queue_example:

    def __init__(self, service_rate):

        self.env = simpy.Environment()
        self.server = simpy.Resource(self.env, capacity=1)
        self.end_time = 1000
        self.id_current = 1
        self.df_events = pd.DataFrame([])
        self.df_waiting_times = pd.DataFrame([])
        self.service_rate = service_rate
        self.num_cust_durations = np.zeros(500)
        self.last_event_time = 0
        self.num_cust_sys = 0


    def run(self):

        self.env.process(self.customer_arrivals())
        self.env.run(until=self.end_time)

    def service(self, customer):

        with self.server.request() as req:
            new_row = [{'Event': 'Arrival', 'Time': self.env.now, 'Customer': customer.id,
                       'Queue_lenght': len(self.server.queue)}]


            self.df_events = pd.concat([self.df_events, pd.DataFrame([new_row])], ignore_index=True)

            yield req
            new_row = {'Event': 'Enter queue', 'Time': self.env.now, 'Customer': customer.id,
                       'Queue_lenght': len(self.server.queue)}
            self.df_events = pd.concat([self.df_events, pd.DataFrame([new_row])], ignore_index=True)

            yield self.env.timeout(np.random.exponential(1/self.service_rate))

            tot_time = self.env.now - self.last_event_time
            self.num_cust_durations[self.num_cust_sys] += tot_time
            self.num_cust_sys -= 1
            self.last_event_time = self.env.now


            new_row = {'Event': 'Departure', 'Time': self.env.now, 'Customer': customer.id, 'Queue_lenght': len(self.server.queue)}
            self.df_events = pd.concat([self.df_events, pd.DataFrame([new_row])], ignore_index=True)

            if is_print:
                print('Departed customer {} at {}'.format(customer.id, self.env.now))

            new_waiting_row = {'Customer': customer.id, 'WaitingTime': self.env.now-customer.arrival_time}

            self.df_waiting_times = pd.concat([self.df_waiting_times, pd.DataFrame([new_waiting_row])], ignore_index=True)


    def customer_arrivals(self):

        while True:

            yield self.env.timeout(np.random.exponential(1))

            tot_time = self.env.now - self.last_event_time
            self.num_cust_durations[self.num_cust_sys] += tot_time
            self.num_cust_sys += 1
            self.last_event_time = self.env.now
            self.last_time = self.env.now


            curr_id = self.id_current
            arrival_time = self.env.now
            customer = Customer(curr_id,  arrival_time)

            self.id_current += 1

            if is_print:
                print('Arrived customer {} at {}'.format(customer.id, self.env.now))

            self.env.process(self.service(customer))


queue_example = Queue_example(2)
queue_example.run()

print('Stop')