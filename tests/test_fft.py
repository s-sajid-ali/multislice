
# coding: utf-8

# Evaluating performance of FFT2 and IFFT2 and checking for accuracy. <br><br>
# Note that the ffts from fft_utils perform the transformation in place to save memory!<br><br>
# As a rule of thumb, it's good to increase the number of threads as the size of the transform increases <br>

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from multislice import fft_utils
import pyfftw,os
import scipy.fftpack as sfft


# In[2]:


get_ipython().magic('load_ext memory_profiler')


# Loading libraries and the profiler to be used

# In[3]:


N = 40000
a = np.random.random((N,N)) + 1j*np.random.random((N,N))


# In[4]:


t = 24 #number of threads.


# Creating a test signal to perform on which we will perform 2D FFT 

# In[5]:


fft_obj = fft_utils.FFT_2d_Obj(a,threads=t)


# In[6]:


get_ipython().magic('timeit np.fft.fft2(a)')


# In[7]:


get_ipython().magic("timeit sfft.fft2(a,overwrite_x='True')")


# In[8]:


get_ipython().magic('timeit fft_obj.run_fft(a)')


# In[9]:


get_ipython().magic('memit np.fft.fft2(a)')


# In[10]:


get_ipython().magic('memit fft_obj.run_fft(a)')


# pyfftw is clearly faster and uses lesser memory !

# In[11]:


a = np.random.random((N,N)) + 1j*np.random.random((N,N))


# In[12]:


t = 20
ifft_obj = fft_utils.IFFT_2d_Obj(a,threads=t)


# Creating a test signal to perform on which we will perform 2D IFFT.

# In[13]:


get_ipython().magic('timeit np.fft.ifft2(a)')


# In[14]:


get_ipython().magic("timeit sfft.ifft2(a,overwrite_x='True')")


# In[15]:


get_ipython().magic('timeit ifft_obj.run_ifft(a)')


# In[16]:


get_ipython().magic('memit np.fft.ifft2(a)')


# In[17]:


get_ipython().magic('memit ifft_obj.run_ifft(a)')


# pyfftw is clearly faster and uses lesser memory !

# Testing for accuracy of 2D FFT: 

# In[18]:


N = 10000
a = np.random.random((N,N)) + 1j*np.random.random((N,N))
A1 = np.fft.fft2(a)
fft_utils.FFT2(a,threads=6)
np.allclose(A1,a)


# Testing for accuracy of 2D IFFT: 

# In[19]:


N = 10000
a = np.random.random((N,N)) + 1j*np.random.random((N,N))
A1 = np.fft.ifft2(a)
fft_utils.IFFT2(a,threads=6)
np.allclose(A1,a)


# In[ ]:




