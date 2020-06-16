## Multislice

Multislice wave propagation in Python. 

#### Contents
* `prop.py` - propogation routines including Transfer Function, Single Fourer Transform, Fraunhofer, Impulse Response <br><br>
  * propTF  - Transfer function propogator
  * prop1FT - Single fourier transform propogator
  * propFF  - Fraunhofer propogator
  * propIR  - Impulse Response propogator
  <br>
* `prop_utils.py`   - utilities for multilsice propogation <br><br>
  * modify  - modify the wavefront by the material 
  * modify_two_materials_case_1, modify_two_materials_case_2 - mixing two materials
  * decide - Decide which propogator to use
  * number_of_steps - Compute the number of steps to be used for propogation within an object. 
  * plot_2d_complex - Plot the wavefront
  * optic_illumination - Utility function that takes an input wavefront and optic and produces the focal spot. 
  <br>
* `fft_utils.py`    - pyfftw wrapper (numpy.fft can be used as well) <br><br> 
  * FFT2 - 2D fft
  * IFFT2 - 2D ifft
  <br>
* `wisdom.pickle`   - file containing pre planned wisdom to elimnate the need for planning at runtime <br><br>

#### Installation : 
Clone the directory via git clone and run `pip install .`. Note that this project uses [flit](https://github.com/takluyver/flit/) as it's build system.

#### Testing
The tests folder has ipython notebooks which compare the results with [this](https://github.com/mdw771/xdesign/blob/master/tests/test_tube_particles.py) test case. Additional tests evaluate the performance of ffts.<br>

#### Credits
This work is based on `Multislice does it all : calculating the performance of nanofocusing x-ray optics, Optics Express, vol. 25, 2017 `.
