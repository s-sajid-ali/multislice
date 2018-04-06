import pyfftw
import numpy as np
import os,pickle
from os.path import dirname as up


try:
    pyfftw.import_wisdom(pickle.load(open(up(os.getcwd())+'/wisdom/wisdom.pickle','rb')))
    print('Wisdom loaded!')
except:
    pass


__all__ = ['FFT2',
           'IFFT2']

'''
pyfftw builder interface is used to access the FFTW class and is optimized with pyfftw_wisdom. Multithreading is used to speed up the calculation. The input is destroyed as making a copy of the input is time consuming and is not useful.

'''

def FFT2(a,flag = 'ESTIMATE',threads = 10):
    A = pyfftw.empty_aligned((np.shape(a)),dtype='complex128', n = pyfftw.simd_alignment)
    
    fft2_ = pyfftw.FFTW(A,A, axes=(0,1), direction='FFTW_FORWARD', flags=('FFTW_'+str(flag), ), 
                         threads=threads, planning_timelimit=None)
    np.copyto(A,a)
    fft2_()
    np.copyto(a,A)
    del(A)
    return None


def IFFT2(a,flag = 'ESTIMATE',threads = 10):
    A = pyfftw.empty_aligned((np.shape(a)),dtype='complex128', n =  pyfftw.simd_alignment)
    
    ifft2_ = pyfftw.FFTW(A,A, axes=(0,1), direction='FFTW_BACKWARD', flags=('FFTW_'+str(flag), ), 
                         threads=threads, planning_timelimit=None)
    ifft2_.__call__(normalise_idft='False')
    np.copyto(A,a)
    ifft2_()
    np.copyto(a,A)
    del(A)
    return None
