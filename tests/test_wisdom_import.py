
# coding: utf-8

# In[1]:


import numpy as np
import pyfftw,os,time,pickle


# In[2]:


def FFT2(a,flag = 'ESTIMATE',threads = 10):
    A = pyfftw.empty_aligned((np.shape(a)),dtype='complex128', n = pyfftw.simd_alignment)
    
    fft2_ = pyfftw.FFTW(A,A, axes=(0,1), direction='FFTW_FORWARD', flags=('FFTW_'+str(flag), ), 
                         threads=threads, planning_timelimit=None)
    np.copyto(A,a)
    fft2_()
    np.copyto(a,A)
    del(A)
    return None


# In[3]:


N = 10000
t = 4
a = np.random.random((N,N)) + 1j*np.random.random((N,N))
t0 = time.time()
FFT2(a,flag='MEASURE',threads=t)
t1 = time.time()
print(N,t, t1 - t0)


# In[4]:


wisdom = pyfftw.export_wisdom()
pickle.dump(wisdom,open('wisdom.pickle','wb'))


# In[5]:


pyfftw.forget_wisdom()


# In[6]:


pyfftw.import_wisdom(pickle.load(open(os.getcwd()+'/wisdom.pickle','rb')))


# In[7]:


N = 10000
t = 4
a = np.random.random((N,N)) + 1j*np.random.random((N,N))
t0 = time.time()
FFT2(a,flag='WISDOM_ONLY',threads=t)
t1 = time.time()
print(N,t, t1 - t0)

