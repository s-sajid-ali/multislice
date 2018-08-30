import numpy as np
import pyfftw
import os
import psutil
import json
import pickle
import time
from os.path import dirname as up
from multislice.fft_utils import FFT_2d_Obj


try:
    os.chdir(up(os.getcwd()) + str('/wisdom'))
except BaseException:
    os.mkdir(up(os.getcwd()) + str('/wisdom'))
    os.chdir(up(os.getcwd()) + str('/wisdom'))


for N in np.array([10000, 15000, 30000, 40000]):
    for t in range(1, psutil.cpu_count() + 1):
        a = np.random.random((N, N)) + 1j * np.random.random((N, N))
        t0 = time.time()
        fft_obj = FFT_2d_Obj(np.shape(a), threads=t)
        fft_obj.run_fft2(a)
        t1 = time.time()
        print(N, t, t1 - t0)


for N in np.array([0000, 15000, 30000, 40000]):
    for t in range(1, psutil.cpu_count() + 1):
        a = np.random.random((N, N)) + 1j * np.random.random((N, N))
        t0 = time.time()
        fft_obj = FFT_2d_Obj(np.shape(a), threads=t)
        fft_obj.run_ifft2(a)
        t1 = time.time()
        print(N, t, t1 - t0)


wisdom = pyfftw.export_wisdom()
pickle.dump(wisdom, open('wisdom.pickle', 'wb'))
