import numpy as np
import pyfftw
import os,psutil,json,pickle,time
from os.path import dirname as up
from multislice import fft_utils


try :
    os.chdir(up(os.getcwd())+str('/wisdom'))
except :
    os.mkdir(up(os.getcwd())+str('/wisdom'))
    os.chdir(up(os.getcwd())+str('/wisdom'))


for N in np.array([5000,10000,15000,25000,30000,36000,40000,45000,50000]):
    for t in range(1,psutil.cpu_count()+1):
        a = np.random.random((N,N)) + 1j*np.random.random((N,N))
        t0 = time.time()
        fft_utils.FFT2(a,threads=t)
        t1 = time.time()
        print(N,t, t1 - t0)


for N in np.array([5000,10000,15000,25000,30000,36000,40000,45000,50000]):
    for t in range(1,psutil.cpu_count()+1):
        a = np.random.random((N,N)) + 1j*np.random.random((N,N))
        t0 = time.time()
        fft_utils.IFFT2(a,threads=t)
        t1 = time.time()
        print(N,t,t1 - t0)


wisdom = pyfftw.export_wisdom()
pickle.dump(wisdom,open('wisdom.pickle','wb'))

