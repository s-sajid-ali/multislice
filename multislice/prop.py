# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:38:01 2017

@author: sajid
"""

import numpy as np
import numexpr as ne
import pyfftw
from multislice.fft_utils import FFT2,IFFT2

'''
contains functions propTF, propIR, propFF, prop1FT
'''

__all__ = ['propTF',
           'propIR',
           'propFF',
           'prop1FT']


'''
Propogation using the Transfer function method. Note that fftfreq has been used from the numpy.fft library. Using this means that we no longer perform an fftshift after transforming u1 to frequency domain.

u1 is the profile of the beam at the input plane. 
step is the sampling step size at the input plane.
L is the side length of the support.
wavel is the wavelength of the light
z is the propogation distance

u2 is the beam profile at the output plane
'''
def propTF(u,step,L,wavel,z) :
    M,N = np.shape(u)
    pi = np.pi
    fx = np.fft.fftfreq(M,d=step)
    fy = np.fft.fftfreq(N,d=step)
    FX,FY = np.meshgrid((fx),(fy))
    FX = pyfftw.interfaces.numpy_fft.fftshift(FX)
    FY = pyfftw.interfaces.numpy_fft.fftshift(FY)
    
    H = ne.evaluate('exp(-1j*pi*wavel*z*(FX**2+FY**2))')
    FFT2(u)
    u = np.fft.fftshift(u)
    u = ne.evaluate('H*u')
    u = np.fft.ifftshift(u)
    IFFT2(u)
    
    return u
'''
Propogation using the Impulse Response function. The convention of shiftinng a function in realspace before performing the fourier transform which is used in the reference is followed here. Input convention as above
'''
def propIR(u,step,L,wavel,z):
    M,N = np.shape(u)
    k = 2*np.pi/wavel
    x = np.linspace(-L/2.0,L/2.0-step,M)
    y = np.linspace(-L/2.0,L/2.0-step,N)
    X,Y = np.meshgrid(x,y)
    
    h = ne.evaluate('(exp(1j*k*z)/(1j*wavel*z))*exp(1j*k*(1/(2*z))*(X**2+Y**2))')
    h_in = pyfftw.empty_aligned((np.shape(h)))
    h = pyfftw.interfaces.numpy_fft.fftshift(h)
    h_in = h
    FFT2(h)
    H = h*step*step
    
    u = pyfftw.interfaces.numpy_fft.fftshift(u)
    FFT2(u)
    u = ne.evaluate('H * u')
    IFFT2(u)
    u = pyfftw.interfaces.numpy_fft.ifftshift(u)
    
    return u
'''
Fraunhofer propogation. Note that we now output two variables since the side length of the observation plane is no longer the same as the side length of the input plane.
'''
def propFF(u,step,L1,wavel,z):
    M,N = np.shape(u)
    k = 2*np.pi/wavel
    L_out = wavel*z/step
    step2 = wavel*z/L1
    n = M #number of samples
    x2 = np.linspace(-L_out/2.0,L_out/2.0,n)
    X2,Y2 = np.meshgrid(x2,x2) 
    c =ne.evaluate('exp((1j*k*(1/(2*z)))*(X2**2+Y2**2))')*(1/(1j*wavel*z))
    u = pyfftw.interfaces.numpy_fft.fftshift(u)
    
    FFT2(u)    
    u = pyfftw.interfaces.numpy_fft.ifftshift(u)
    u = ne.evaluate('c*u')
    u *= step*step
    
    return u,L_out

def prop1FT(u,step,L1,wavel,z):
    M,N = np.shape(u)
    k = 2*np.pi/wavel
    x = np.linspace(-L1/2.0,L1/2.0-step,M)
    y = np.linspace(-L1/2.0,L1/2.0-step,N)
    X,Y = np.meshgrid(x,y)
    L_out = wavel*z/step
    step2 = wavel*z/L1
    n = M #number of samples
    x2 = np.linspace(-L_out/2.0,L_out/2.0,n)
    X2,Y2 = np.meshgrid(x2,x2)

    c = ne.evaluate('exp(1j*k*(1/(2*z))*(X2**2+Y2**2))')*(1/(1j*wavel*z))
    c0 = ne.evaluate('exp((1j*k)/(2*z)*(X**2 + Y**2))')
    
    u = ne.evaluate('c0*u')
    u = pyfftw.interfaces.numpy_fft.fftshift(u)    
    FFT2(u)
    u = pyfftw.interfaces.numpy_fft.ifftshift(u)
    u = ne.evaluate('c*u')    
    u *= step*step
    
    return u,L_out
