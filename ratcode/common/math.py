import numpy as np


def decay_conv(array, tau = 1000):
    '''
    Return an array

    Parameters:
    array: input to be convolved. Typically an array of zeros and ones
    tau: time constant of the exponential. Default set for miliseconds units
    '''

    t = np.arange(0,len(array))
    g = np.exp(-t/tau)
    conv = np.convolve(array,g)[:len(t)] * (t[1]-t[0])

    return conv

