import numpy as np

hbar = 1.054571800e-27  # постоянная Планка
me = 9.10938356e-28     # масса электрона
c = 2.99792458e+10      # скорость света
el = 4.803204673e-10    # заряд электрона
mp = 1.83615267261e+3   # отношение массы протона к массе электрона
from numpy import pi    # число пи

re = el ** 2 / (me * c ** 2)         # классический радиус электрона
ES = me ** 2 * c ** 3 / (hbar * el)  # поле Швингера
alpha = el ** 2 / (hbar * c)         # постоянная тонкой структуры
R0 = hbar ** 2 / (me * el ** 2)      # радиус первой Боровской орбиты
Ry = el ** 2 / (2 * R0)              # постоянная Ридберга (энергия электрона на первой Боровской орбите)

def omega(lmbda):
    """ Converts wavelength to angular frequency """
    return 2 * pi * c / lmbda

def Ncr(lmbda):
    """ Returns critical plasma density for given wavelength """
    return me * omega(lmbda) ** 2 / (4 * pi * el ** 2)

def E0(lmbda, m=me, q = el):
    """ Returns relativistic field amplitude for given particle charge and mass and wavelength """
    return m * c * omega(lmbda) / q

def j0(lmbda):
    """ Returns normalizing current used in PIC """
    return el * c * Ncr(lmbda)

def omega_pl(Ne, gamma = 1):
    """ Returns plasma frequency for given density """
    return np.sqrt( 4 * pi * el ** 2 * Ne / me / gamma )

def to_C(q):
    """ Converts charge from CGS to Coulomb """
    return q * 1e1 / c

def to_eV(erg):
    """ Converts energy from CGS to eV """
    return erg / to_C(el) * 1e-7

def to_erg(eV):
    """ Converts energy from eV to erg """
    return eV * to_C(el) * 1e7

def to_Vm(E):
    """ Converts field amplitude from CGS to V/m """
    return E * c * 1e-6

def to_A(j):
    """ Converts current density from CGS to Ampere """
    return j * 1e1 / c

def to_Wcm(a0, lmbda):
    """ Converts field amplitude from CGS to intensity in W/cm^2 """
    con = me**2 * c**5 * np.pi / (2 * el**2) * 1e-7
    return con * a0**2 / lmbda**2

def to_W(I, width):
    """ Converts intensity (W cm^-2) to power (W) for gaussian beam """
    return 0.5 * I * np.pi * width * width

#####################################
### TRUNCATE INSIGNIFICANT DIGITS ###
#####################################

def trunc(x, n):
    mantissas, binaryExponents = np.frexp( x )
    decimalExponents = 3.0103e-1 * binaryExponents
    omags = np.floor(decimalExponents)

    mantissas *= 10**(decimalExponents - omags)

    if mantissas < 1.0:
        mantissas *= 10.0
        omags -= 1.0

    return np.around( mantissas, decimals = n - 1 ) * 10**omags

#####################
### BIAS AND GAIN ###
#####################

def bias_scalar(x, bias):
    return (x / ((((1.0/bias) - 2.0)*(1.0 - x))+1.0))

bias = np.vectorize(bias_scalar,excluded=['bias'])

def gain_scalar(x,gain):
    if x < 0.5:
        return bias(x * 2.0,gain)/2.0
    else:
        return bias(x * 2.0 - 1.0,1.0 - gain)/2.0 + 0.5

gain = np.vectorize(gain_scalar,excluded=['gain'])

##########################
### CONVOLUTION SMOOTH ###
#########################

def smooth(x, window, center=2, edges=4):
    xc = np.linspace(-1,1,window)
    filt = np.power(np.cos(0.5*np.pi*np.power(xc,edges)), center)
    filt /= filt.sum()
    return np.convolve(x,filt,'same')

################################################
### MATPLOTLIB HELPER FUNCTIONS AND DEFAULTS ###
################################################

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, PowerNorm, ListedColormap, hsv_to_rgb

def subplots(x=1,y=1,closeall='auto',auto_aspect=True,**kwargs):
    if closeall == 'auto' and len(plt.get_fignums())>10:
        closeall = True
        print('Deleting unused figures')
    if closeall is True:
        plt.close('all')
    fig, ax = plt.subplots(y,x,**kwargs)
    if x==1 and y==1 and auto_aspect:
        ax.set_aspect('auto')
    elif (x>1 or y>1) and auto_aspect:
        for a in ax.flatten():
            a.set_aspect('auto')
    # if figsize:
    #     w, h = figsize
    # else:
    #     w, h = plt.rcParams['figure.figsize']
    # fig.canvas.layout.height = str(h) + 'in'
    # fig.canvas.layout.width = str(w) + 'in'
    return fig, ax

def plot_vcolorbar(im, axes, fig=None, divider=None, name='',pad=0.3,**kwargs):
    if fig is None:
        fig = plt.gcf()
    if divider is None:
        divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size='5%', pad=pad)
    cb = fig.colorbar(im, cax=cax, orientation='vertical', fraction=0.05)
    # cb.set_alpha(1)
    # cb.draw_all()
    cb.set_label(name)
    # cax.ticklabel_format(useMathText=True)
    return divider, cb

def seq_cmap(hue = 0.0, sat = 1.0, value_norm = lambda x : x, alpha_norm = lambda x : x, data_norm = lambda x :x, name='new', register=False):
    """
    Generates (and registers) sequential colormap from single color. Smaller values get more transparent, larger values get darker
    
    Parameters
    ----------
    hue : float, optional
        Color hue, by default 0.0
    sat : float, optional
        Color saturation, by default 1.0
    value_norm : callable, optional
        Function which describes luminosity profile. Should transform range [0, 1] to the range [0, 1]. value_norm(0) gives luminosity of the largest value, value_norm(1) - of the smallest. By default profile is linear.
    alpha_norm : callable, optional
        Same as value norm, but for transparency and flipped.
    name : str, optional
        Name of colormap, by default 'new'.
    register : bool, optional
        Whether to register new colormap in matplotlib, by default False.

    Returns
    -------
    Matplotlib colormap
    """
    
    v = np.linspace(1, 0, 256)
    hsv = np.repeat(v.copy()[:,np.newaxis], 3, axis = 1)
    hsv[:, 0] = hue
    hsv[:, 1] = sat
    hsv[:, 2] = value_norm(hsv[:, 2])
    
    colors = hsv_to_rgb(hsv)
    m = ListedColormap(colors, name)
    m._init()
    m._lut[:-3,3] = alpha_norm(1-v)
    
    # Equating out of boundary colors to on boundary ones
    m._lut[-3] = m._lut[0]
    m._lut[-2] = m._lut[-4]
    return m

def div_cmap(hue_s = 0.8, sat_s = 1.0, hue_l = 0.0, sat_l = 1.0, value_norm = lambda x : x, alpha_norm = lambda x : x, name='new', register=False):
    """
    Generates (and registers) diverging colormap from two colors. Extreme values (the smallest and the largest) are most dark, saturated and opaque. Middle values are white and transparent.
    
    Parameters
    ----------
    hue_s : float, optional
        Color hue of small values, by default 0.0
    sat_s : float, optional
        Color saturation of small values, by default 1.0
    hue_l : float, optional
        Color hue of large values, by default 0.0
    sat_l : float, optional
        Color saturation of large values, by default 1.0
    value_norm : callable, optional
        Function which describes luminosity profile from the middle value to the largest (smallest) value. Should transform range [0, 1] to the range [0, 1]. value_norm(0) gives luminosity of the largest (smallest) value, value_norm(1) - of the middle. The resulting profile is mirrored around middle value. By default profile is linear.
    alpha_norm : callable, optional
        Same as value norm, but for transparency and flipped.
    name : str, optional
        Name of colormap, by default 'new'.
    register : bool, optional
        Whether to register new colormap in matplotlib, by default False.
    
    Returns
    -------
    Matplotlib colormap
    """
    
    v = np.linspace(0,1,128)
    hsv_s = np.repeat(v.copy()[:,np.newaxis], 3, axis = 1)
    hsv_l = np.flip(hsv_s.copy())
    
    hsv_s[:, 0] = hue_s
    hsv_s[:, 1] = sat_s

    hsv_l[:, 0] = hue_l
    hsv_l[:, 1] = sat_l

    hsv_s[:, 2] = value_norm(hsv_s[:, 2])
    hsv_l[:, 2] = value_norm(hsv_l[:, 2])

    hsv = np.concatenate((hsv_s,hsv_l))

    colors = hsv_to_rgb(hsv)
    m = ListedColormap(colors, name)
    m._init()
    
    m._lut[:-3,3] = alpha_norm(np.abs(np.linspace(-1,1,256)))
    
    # Equating out of boundary colors to on boundary ones
    m._lut[-3] = m._lut[0]
    m._lut[-2] = m._lut[-4]
    return m
    
lw = 0.7 # linewidth for border
lwl = 1.0 # linewidth for lines in plots
font = {'family' : 'serif', 'serif' : 'cmr10', 'size' : 12}
mpl.rc('font', **font)
mpl.rc('lines', linewidth=lwl)
mpl.rc('axes', linewidth=lw)
mpl.rc('axes', unicode_minus=False)
mpl.rc('figure', autolayout=True)
mpl.rc('mathtext', fontset='cm')
mpl.rc('figure', figsize=(8,6))
mpl.rc('image', aspect='auto')
mpl.rc('image', origin='lower')

def mm_to_inch(x):
    return 0.0393701 * x