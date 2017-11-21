## multislice

Codes for multislice propogation

**Contents** 
* *fft_utils.py*    - pyfftw wrapper <br>
* *prop.py*         - propogation routines including Transfer Function, Impulse Response, Single Fourer Transform, Fraunhofer<br>
* *prop_utils.py*   - utilities for multilsice propogation <br>
* *wisdom.pickle*   - file containing pre planned wisdom to elimnate the need for planning at runtime <br>

**Testing**
* The tests folder has ipython notebooks which compare the results with [this](https://github.com/mdw771/xdesign/blob/master/tests/test_tube_particles.py) test case. Additional tests evaluate the performance of ffts.

Usage can be gleaned from looking at the zone plate simulation ipython notebooks at : https://github.com/sajid-ali-nu/zone_plate_testing<br>
Also, note that one must configure the pyfftw parameters and wisdom for whichever machine this package is going to be used on.
