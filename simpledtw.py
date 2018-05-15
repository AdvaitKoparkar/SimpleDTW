#!/usr/bin/python

import numpy as np


def simpledtw(c1, c2, warping_window=None, warping_strength=1000, verbose=False):
    return estimate(find_dist_mat(c1, c2), warping_window, warping_strength, verbose)

def find_dist_mat(c1, c2):
    '''
        Helper function for finding D_mat

        Input:
            (*) c1: n1x2 array - contour1
            (*) c2: n2x2 array - contour2

        Output:
            (*) D: Distance matrix
    '''

    D = -1*np.ones((c1.shape[0], c2.shape[0]))
    for i in range(c1.shape[0]):
        for j in range(c2.shape[0]):
            D[i, j] = np.linalg.norm(c1[i, :]-c2[j, :])**2
    return D

def estimate(D_mat, warping_window=None, warping_strength=100, verbose=False):
    '''
        A function for finding the DTW distance between two curves

        Input:
            (*) D_mat: Distance matrix between each pair of points between two curves
            (*) warping_window: Size of warping window for the path on each side of
                               diagonal
            (*) verbose: echo progress

        Output:
            (*) dist: minumium distance between curves
            (*) cost: dtw distance
            (*) acc: accumulated distance along each path
            (*) path: path taken through accumulated distance
    '''
    acc = -1*np.ones_like(D_mat)
    path_mat = -1*np.ones_like(D_mat)
    path = []
    prev_pt = []

    if verbose:
        import time
        import matplotlib.pyplot as plt
        t = time.time()
        print("Searching paths...")

    if warping_window > 0 and warping_window < min(D_mat.shape)/2.0:
        if verbose:
            print("Adding warping constraint...")
        D_mat[warping_window:, :D_mat.shape[1]-warping_window] = D_mat[warping_window:, :D_mat.shape[1]-warping_window] + warping_strength*np.tril(D_mat[warping_window:, :D_mat.shape[1]-warping_window])
        D_mat[:D_mat.shape[0]-warping_window, warping_window:] = D_mat[:D_mat.shape[0]-warping_window, warping_window:] + warping_strength*np.triu(D_mat[:D_mat.shape[0]-warping_window, warping_window:])

    # Initialization
    acc[0, 0] = D_mat[0, 0]
    path_mat[0, 0] = -1
    for i in range(1, acc.shape[0]):
        acc[i, 0] = acc[i-1, 0] + D_mat[i, 0]
        path_mat[i, 0] = 1
    for i in range(1, acc.shape[1]):
        acc[0, i] = acc[0, i-1] + D_mat[0, i]
        path_mat[0, i] = 2

    # Inductive Step
    for i in range(1, acc.shape[0]):
        for j in range(1, acc.shape[1]):
            prev_pt = [acc[i-1, j-1], acc[i-1, j], acc[i, j-1]]
            acc[i, j] = D_mat[i, j] + np.min(prev_pt)
            path_mat[i, j] = np.argmin(prev_pt)

    if verbose:
        print("Finding optimal path...")
    # Backtracking
    m_ind, n_ind = path_mat.shape[0] - 1, path_mat.shape[1] - 1
    while path_mat[m_ind, n_ind] != -1:
        path.append((m_ind, n_ind))
        if path_mat[m_ind, n_ind] == 0:
            m_ind -= 1
            n_ind -= 1
        elif path_mat[m_ind, n_ind] == 1:
            m_ind -= 1
        elif path_mat[m_ind, n_ind] == 2:
            n_ind -= 1
    path.append((0, 0))

    path.reverse()
    path = np.array(path)
    path = path
    cost = acc[acc.shape[0] - 1, acc.shape[1] - 1] / len(path)
    dist = np.min(D_mat)

    if verbose:
        print("Completed in : %fs" %(time.time() - t))
        plt.imshow(D_mat.T, origin='lower')
        plt.plot(path[:, 0], path[:, 1], 'w')
        plt.show()

    return dist, cost, acc, path
