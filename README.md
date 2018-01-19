## multislice

Codes for multislice propogation

**Contents** 
* *fft_utils.py*    - pyfftw wrapper <br><br>
  * FFT2 - 2D fft
  * IFFT2 - 2D ifft
  <br>
* *prop.py*         - propogation routines including Transfer Function, Single Fourer Transform, Fraunhofer <br><br>
  * propTF  - Transfer function propogation
  * prop1FT - Single fourier transform propogation
  * propFF  - Fraunhofer propogation 
  <br>
* *prop_utils.py*   - utilities for multilsice propogation <br><br>
  * modify  - modify the wavefront by the material 
  * modify_two_materials_case_1, modify_two_materials_case_2 - mixing two materials
  * decide - Decide which propogator to use
  * number_of_steps - Compute the number of steps to be used for propogation within an object. 
  * plot_2d_complex - Plot the wavefront
  * optic_illumination - Utility function that takes an input wavefront and optic and produces the focal spot. 
  <br>
* *wisdom.pickle*   - file containing pre planned wisdom to elimnate the need for planning at runtime <br><br>

**Testing**
* The tests folder has ipython notebooks which compare the results with [this](https://github.com/mdw771/xdesign/blob/master/tests/test_tube_particles.py) test case(Note that at the moment the ffts use 6 threads and the test case has dimensions of 255x255). Additional tests evaluate the performance of ffts.<br>

Also, note that one must configure the pyfftw parameters and wisdom for whichever machine this package is going to be used on.
