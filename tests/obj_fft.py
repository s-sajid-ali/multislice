import numpy as np
import pyfftw,time
import os,pickle
from os.path import dirname as up

pyfftw.import_wisdom(pickle.load(open(up(os.getcwd())+'/wisdom/wisdom.pickle','rb')))


class FFT_2d_Obj(object):
        
    def __init__(self,dimension,direction='FORWARD',flag='ESTIMATE',threads = 10):
        
        self.pyfftw_array = pyfftw.empty_aligned(dimension,dtype='complex128', n = pyfftw.simd_alignment)
        
        self.fft2_ = pyfftw.FFTW(self.pyfftw_array,self.pyfftw_array, axes=(0,1), direction='FFTW_FORWARD',
                                 flags=('FFTW_'+str(flag), ), threads=threads, planning_timelimit=None )
        self.ifft2_ = pyfftw.FFTW(self.pyfftw_array,self.pyfftw_array, axes=(0,1), direction='FFTW_BACKWARD',
                                  flags=('FFTW_'+str(flag), ), threads=threads, planning_timelimit=None)
        
        self.ifft2_.__call__(normalise_idft='False')
        
    def run_fft2(self,A):
        pa = self.pyfftw_array
        np.copyto(pa,A)
        self.fft2_()
        np.copyto(A,pa)
        del(pa)
        return None
    
    def run_ifft2(self,A):
        pa = self.pyfftw_array
        np.copyto(pa,A)
        self.ifft2_()
        np.copyto(A,pa)
        del(pa)
        return None