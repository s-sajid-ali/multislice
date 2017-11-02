import pyfftw

pyfftw.import_wisdom(pyfftw.export_wisdom())

__all__ = ['FFT2',
           'IFFT2']

'''
pyfftw builder interface is used to access the FFTW class and is optimized with pyfftw_wisdom. Multithreading is used to speed up the calculation. The input is destroyed as making a copy of the input is time consuming and is not useful.

'''
def FFT2(a,threads = 6):
    fft2 = pyfftw.builders.fft2(a,overwrite_input=True, threads=threads)
    return fft2()
def IFFT2(a,threads = 6):
    ifft2 = pyfftw.builders.ifft2(a,overwrite_input=True, threads=threads)
    return ifft2()
