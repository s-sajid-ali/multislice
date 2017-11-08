import pyfftw
import numpy as np

pyfftw.import_wisdom(pyfftw.export_wisdom())

__all__ = ['FFT2',
           'IFFT2']

'''
pyfftw builder interface is used to access the FFTW class and is optimized with pyfftw_wisdom. Multithreading is used to speed up the calculation. The input is destroyed as making a copy of the input is time consuming and is not useful.

'''
def FFT2(a,threads = 6):
    A = pyfftw.empty_aligned((np.shape(a)),dtype='complex128',n=32)
    
    fft2_ = pyfftw.FFTW(A,A, axes=(0,1), direction='FFTW_FORWARD', flags=('FFTW_MEASURE', ), 
                         threads=threads, planning_timelimit=None)
    np.copyto(A,a)
    fft2_()
    np.copyto(a,A)
    del(A)
    return None
def IFFT2(a,threads = 6):
    A = pyfftw.empty_aligned((np.shape(a)),dtype='complex128',n=32)
    
    ifft2_ = pyfftw.FFTW(A,A, axes=(0,1), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', ), 
                         threads=threads, planning_timelimit=None)
    ifft2_.__call__(normalise_idft='False')
    np.copyto(A,a)
    ifft2_()
    np.copyto(a,A)
    del(A)
    return None
