#!/usr/bin/python

from simpledtw import simpledtw
import numpy as np



def simple_test():
    c1 = np.array([[1, 1], [4, 5], [6, 5], [10, 11]])
    c2 = np.array([[1, 5], [4, 5], [6, 5], [10, 11]])
    _, x, _, _ = simpledtw(c1, c2)

def batch_test():
    create_data()
    C1 = np.load('./C1.npy')
    C2 = np.load('./C2.npy')
    # print(C1)
    batch_size = C1.shape[0]
    my_dist = -1*np.ones((1, batch_size))
    # my_times = -1*np.ones((1, batch_size))
    my_path = []

    for i in range(batch_size):
        _, my_dist[0,i], _, _ = simpledtw(C1[i, :, :], C2[i, :, :], warping_window=0, verbose=True)

def create_data(batch_size = 100):
    n1 = np.random.randint(low=35, high=40)
    n2 = np.random.randint(low=35, high=40)

    C1 = np.random.uniform(low=0.0, high=100.0, size=(batch_size, n1, 2))
    C2 = np.random.uniform(low=0.0, high=100.0, size=(batch_size, n2, 2))
    np.save('C1.npy', C1)
    np.save('C2.npy', C2)

if __name__ == '__main__':
    # simple_test()
    batch_test()
