import numpy as np
from multislice import prop
from multislice.fft_utils import FFT_2d_Obj
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import time
import numexpr as ne
from skimage.restoration import unwrap_phase

__all__ = ['modify',
           'modify_two_materials_case_1',
           'modify_two_materials_case_2',
           'decide',
           'plot_2d_complex',
           'number_of_steps',
           'optic_illumination',]


'''
decide : decide whether to use TF or IR approach depending on the distance

Inputs - step size in z, step size in xy, support length, wavelength

Outputs - propogator 
'''
def decide(step_z,step_xy,L,wavel):
    dist = step_z
    sampling = step_xy
    critical = wavel*dist/L
    if ((L**2)/(wavel*step_z)) > 0.1 :
        
        if sampling > critical :
            p = prop.propTF
            print('propogator to be used : Transfer Function')
        else :
            p = prop.prop1FT
            print('propogator to be used : Single Fourier Transform')
    else :
        p = prop.FF
        print('propogator to be used : Fraunhofer')
    return p   



'''
modify : wavefront is modified according to the material present

Inputs - wavefront, slice properties (here the zone plate),step size in z , wavelength

Outputs - modified wavefront

(used as part of the multislice loop)
'''
def modify(wavefront,zp_delta,zp_beta,step_z,wavel):
    dist = step_z
    kz = 2 * np.pi * dist /wavel
    beta_slice = zp_beta
    delta_slice = zp_delta
    new_wavefront = wavefront * np.exp((kz * delta_slice) * 1j) * np.exp(-kz * beta_slice)
    return new_wavefront



'''
modify_two_materials_case_1 : wavefront is modified according to the materials present that are horizontally stacked
                              (Fig 8, a-(i) in Optics Express Vol. 25, Issue 3, pp. 1831-1846)

Inputs  - wavefront, propogation distance,wavelength, pattern of first material, delta, beta(first material),
          pattern_2,delta, beta(second material) 
          
Outputs - modified wavefront

(used as part of the multislice loop)
'''
def modify_two_materials_case_1(wavefront,step_z,wavel,frac_1,frac_2,pattern_1,delta_1,beta_1,pattern_2,delta_2,beta_2):
    dist = step_z
    kz_1 = 2 * np.pi * dist * frac_1 /wavel
    kz_2 = 2 * np.pi * dist * frac_2 /wavel
        
    '''
    Collapsing the following into one numexpr statement : 
    modulation_1 = pattern_1*np.exp((kz_1 * delta_1)*1j -kz_1*beta_1)
    modulation_2 = pattern_2*np.exp((kz_2 * delta_2)*1j -kz_2*beta_2)
    output = wavefront * ( modulation_1 * modulation_2 )
    '''
    
    return ne.evaluate('wavefront*(pattern_1*exp((kz_1*delta_1)*1j -kz_1*beta_1)*pattern_2*exp((kz_2*delta_2)*1j -kz_2*beta_2))')


'''
modify_two_materials_case_2 : wavefront is modified according to the materials present that are vertically stacked
                              (Fig 8, a-(ii) in Optics Express Vol. 25, Issue 3, pp. 1831-1846)

Inputs  - wavefront, propogation distance,wavelength, pattern of first material, delta, beta(first material),
          pattern_2,delta, beta(second material) 
          
Outputs - modified wavefront

(used as part of the multislice loop)
'''
def modify_two_materials_case_2(wavefront,step_z,wavel,pattern_1,delta_1,beta_1,pattern_2,delta_2,beta_2):
    pi = np.pi
    kz = ne.evaluate('2 * pi * step_z /wavel')
    
    '''
    Collapsing the following into one numexpr statement : 
    modulation_1 = exp((kz*delta_1)*1j - kz*beta_1)
    modulation_2 = exp((kz*delta_2)*1j - kz*beta_2)
    output = wavefront * ( pattern_1*modulation_1 + pattern_2*modulation_2 )
    '''
    
    return ne.evaluate('wavefront * ( pattern_1*exp((kz*delta_1)*1j - kz*beta_1)+pattern_2*exp((kz*delta_2)*1j - kz*beta_2) )')


'''
plot_2d_complex : function used to plot complex 2d array

Inputs  - input_array : input array to be plotted, name : name of the variable being plotted, mode : used to specify linear or log plot , coords : used to plot the boundaries(as **kwargs)

Outputs - plots of magnitude and phase!
'''
def plot_2d_complex(input_array,mode='linear',name='input_array',*args,**kwargs):
    fig, (ax1,ax2) = plt.subplots(1,2)
    if 'coords' in kwargs:
        if mode == 'linear' : 
            im1 = ax1.imshow((np.abs(input_array)),extent = kwargs['coords'])
        if mode == 'log' :
            im1 = ax1.imshow(np.log((np.abs(input_array))),extent = kwargs['coords'])
        ax1.title.set_text('magnitude of '+str(name)+' ( in '+str(mode)+' scale)')
        ax1.title.set_y(1.08)
        fig.colorbar(im1,ax = ax1)
        scaling = np.round(np.log10(np.abs(kwargs['coords'][0])))
        scaling = np.int(scaling)
        ax1.set_xlabel('axes in 10^('+str(scaling)+')')
        im2 = ax2.imshow(unwrap_phase(np.angle(input_array)),extent = kwargs['coords'])
        ax2.title.set_text('phase of '+str(name))
        ax2.title.set_y(1.08)
        fig.subplots_adjust(right=1.75)
        ax2.set_xlabel('axes in 10^('+str(scaling)+')')
        fig.colorbar(im2,ax = ax2)
    else :
        if mode == 'linear' : 
            im1 = ax1.imshow((np.abs(input_array)))
        if mode == 'log' :
            im1 = ax1.imshow(np.log((np.abs(input_array)))) 
        ax1.title.set_text('magnitude  of '+str(name)+' ( in '+str(mode)+' scale)')
        ax1.title.set_y(1.08)
        fig.colorbar(im1,ax = ax1)
        ax1.set_xlabel('axes in 10^('+str(scaling)+')')
        im2 = ax2.imshow(unwrap_phase(np.angle(input_array)))
        ax2.title.set_text('phase of '+str(name))
        ax2.title.set_y(1.08)
        ax2.set_xlabel('axes in 10^('+str(scaling)+')')
        fig.subplots_adjust(right=1.75)
        fig.colorbar(im2,ax = ax2) 
    plt.show()
    if 'print_max' in args :
        print('maximum value of '+str(name)+': ',np.max(abs(input_array)))
        print('minimum value of '+str(name)+': ',np.min(abs(input_array)))
        print('location of maxima in '+str(name)+': ',np.where(abs(input_array)==np.max(abs(input_array))))

        
        
'''
number_of_steps : calculate number of steps required for propogation along direction of beam

Inputs  - step_xy : sampling size in xy plane, wavel : wavelength, thickness : thickness of object

Outputs - number of steps : number of steps for propogation through the object 
(As per the meteric described in Optics Express Vol. 25, Issue 3, pp. 1831-1846)
'''
def number_of_steps(step_xy,wavel,thickness):
    eps1 = 0.1
    eps2 = 0.1
    delta_z_suggested = ((eps2*(step_xy**2))/(eps1**2*wavel))
    number_of_steps = int(np.ceil(thickness/delta_z_suggested))+1
    print('suggested step size :',delta_z_suggested)
    print('number of steps required for propogation through the zone plate :',number_of_steps)
    return number_of_steps



'''
optic_illumination : calculate illumination from zone plate (or any other xray optic)

Inputs  - wavefront : input wave, pattern : pattern of the optic along one plane,delta,beta : refractive index of the optic, thickness : thickness of zone plate, number_of_steps_zp : number of steps for propogation through zp, d1 : propogation distance before zp, d2 : propogation distance after zp (typically the focal length) 

Outputs - wavefront : output wave
'''
def optic_illumination(wavefront_input,
                       pattern,delta,beta,
                       thickness,step_xy,wavel,
                       number_of_steps,d1,d2,use_fftw='True',**kwargs):
    
    wavefront = np.copy(wavefront_input)
    L = np.shape(wavefront_input)[0]*step_xy
    xray_object = str('zone plate')
    mode = str('serial')
    if use_fftw == 'True':
        fft_obj = FFT_2d_Obj(np.shape(wavefront))
    else :
        fft_obj = None

    if 'xray_object' in kwargs :
        xray_object = kwargs['xray_object']
    if 'mode' in kwargs : 
        mode = kwargs['mode']    
      
    
    #pre object
    if d1 != 0 :
        print('Free space propogation before '+str(xray_object)+'...')
        step_z = d1
        p = decide(step_z,step_xy,L,wavel)
        print('Fresnel Number :',((L**2)/(wavel*step_z)))
        wavefront,L  = p(wavefront,step_xy,L,wavel,step_z,fft_obj)
        
    
    
    #through object
    step_z = thickness/number_of_steps
    p = decide(step_z,step_xy,L,wavel)
    print('Fresnel Number :',((L**2)/(wavel*step_z)))
    time.sleep(1) 
    if mode == 'parallel':
        for i in range(number_of_steps):    
            t0 = time.time()
            wavefront = modify_two_materials_case_2(wavefront,step_z,wavel,pattern,delta,beta,np.ones(np.shape(pattern))-pattern,0,0)
            t1 = time.time()
           
            wavefront,L  = p(wavefront,step_xy,L,wavel,step_z,fft_obj)
            t2 = time.time()
            print(i,'modify : ',t1-t0,'prop :',t2-t1)
    else : 
        for i in tqdm(range(number_of_steps),desc='Propogation through '+str(xray_object)+'...'):
            wavefront = modify_two_materials_case_2(wavefront,step_z,wavel,pattern,delta,beta,np.ones(np.shape(pattern))-pattern,0,0)
            wavefront,L  = p(wavefront,step_xy,L,wavel,step_z)

    #post object
    if d2 !=0 :
        step_z = d2
        print('Free space propogation after '+str(xray_object)+'...')
        p = decide(step_z,step_xy,L,wavel)
        print('Fresnel Number :',((L**2)/(wavel*step_z)))
        wavefront,L  = p(wavefront,step_xy,L,wavel,step_z)
    
    wavefront_out = np.copy(wavefront)
    del wavefront
    return wavefront_out,L
