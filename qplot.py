import numpy as np
import xarray as xr
from utilities import *

#####################
### BIAS AND GAIN ###
#####################

def bias_scalar(x, bias):
    return x / ( 1 + ( 1 / bias - 2 ) * ( 1 - x ) )

bias = np.vectorize(bias_scalar,excluded=['bias'])

def gain_scalar(x,gain):
    if x < 0.5:
        return bias(x * 2, gain) / 2
    else:
        return bias(x * 2 - 1, 1 - gain) / 2.0 + 0.5

gain = np.vectorize(gain_scalar,excluded=['gain'])

################################################
### MATPLOTLIB HELPER FUNCTIONS AND DEFAULTS ###
################################################

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, PowerNorm, ListedColormap, hsv_to_rgb
from hsluv import hsluv_to_rgb

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
    cb = fig.colorbar(im, cax=cax, orientation='vertical', **kwargs)
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
    
    colors = np.array([hsluv_to_rgb([360 * hsv[i][0], 100 * hsv[i][1], 100 * hsv[i][2]]) for i in range(256)])
#     colors = hsv_to_rgb(hsv)
    alpha = alpha_norm(1-v)
    m = ListedColormap(colors, name)
    m._init()
    m._lut[:-3,3] = alpha
    
    white = np.ones((256, 3))
    alpha = np.broadcast_to(alpha,(3, 256)).T
    mo = ListedColormap(colors * alpha + white * (1 - alpha), name + '_opaque')
    mo._init()
    
    # Equating out of boundary colors to on boundary ones
    m._lut[-3] = m._lut[0]
    m._lut[-2] = m._lut[-4]
    
    mo._lut[-3] = mo._lut[0]
    mo._lut[-2] = mo._lut[-4]
    return m, mo

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

cm_w = dict(hue = 0.045, sat = 0.85, 
            value_norm = lambda x : 0.45 + 0.3*x, 
            alpha_norm = lambda x : np.clip(x / 0.3, 0, 1))
cm_e = dict(hue = 0.33, sat = 0.85,
            value_norm = lambda x : 0.45 + 0.3*np.power(x, 2), 
            alpha_norm = lambda x : np.clip(x / 0.3, 0, 1))
cm_i = dict(hue = 0.66, sat = 0.85,
            value_norm = lambda x : 0.45 + 0.3*np.power(x, 2), 
            alpha_norm = lambda x : np.clip(x / 0.3, 0, 1))
cm_p = dict(hue = 0.02, sat = 1.0,
            value_norm = lambda x : 0.5, #0.9, 
            alpha_norm = lambda x : np.clip(x / 0.05, 0, 1))
cm_p2 = dict(hue = 0.2, sat = 1.0,
             value_norm = lambda x : 0.9, 
             alpha_norm = lambda x : np.clip(x / 0.05, 0, 1))
cm_g = dict(hue = 0.85, sat = 1.0,
            value_norm = lambda x : 0.3, 
            alpha_norm = lambda x : np.clip(x / 0.2, 0, 1))
cms = dict(e = cm_w, b = cm_w, w = cm_w, n_E = cm_e, n_I = cm_i, n_P = cm_p, n_G = cm_g)

masks = dict(e = [[1],[1]],
             w = [[1],[1]],
             b = [[1],[1]],
             n_E = [[1],[0]],
             n_I = [[0],[1]],
             n_P = [[1],[1]],
             n_G = [[1],[1]])

def plot_2d(ax, div, qres, t, field, plane = 'xy', mask = None, cbar = False, cbax = None, norm = None, cmap = 'viridis', interpolation = 'nearest'):
    f = qres.read_field(t, field)[plane]
    if field == 'n_E':
        f = -f
    if field in ["n_E", "n_P"]:
        f *= Ncr(qres.a.lamda)
    if mask is None:
        mask = [[1,1],[1,1]]
    mask = np.array(mask)
    if len(mask.shape) == 1 or (len(mask.shape) == 2 and mask.shape[1] == 1):
        mask = np.broadcast_to(mask,(2,2))
    a1, a2 = plane[0], plane[1]
    x0, x1, y0, y1 = f[a1].min(),f[a1].max(),f[a2].min(),f[a2].max()
    mnx, mny = mask.shape
    mask = xr.DataArray(mask,coords=[(a2,np.linspace(y1, y0, len(mask))),
                                     (a1,np.linspace(x0, x1, len(mask[0])))]).interp({a1:f[a1],a2:f[a2]},method='nearest')
    f = f.where(mask)
    test = xr.DataArray(np.ones_like(f.values),coords=f.coords).where(mask)
    if norm is None:
        norm = Normalize(f.min(),max(f.max(),f.min()+0.01))
    if isinstance(cmap, tuple):
        cm, cm_o = cmap
    else:
        cm = cm_o = cmap
    ax.imshow(f.values.T,extent=[x0, x1, y0, y1], norm = norm, cmap = cm, interpolation = interpolation, aspect='equal')
    if cbar:
        if cbax is None:
            cbax = div.append_axes("right", size="7.5%", pad="20%")
        fig = plt.gcf()
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cm_o), cax=cbax, orientation='vertical', fraction=0.05)
#         cbax.set_title(rf"${field}$")

def plot_pic(qres, t, fields = ['e','n_E','n_I','n_G','n_P'], plane = 'xy', cbar = False, gs = None, norms = None):
    if isinstance(qres, QRES):
        qres = [qres]
        
    if gs is None:
        gs = gridspec.GridSpec(len(qres), 1)
    
    gl_axes = []
    for i, q in enumerate(qres):
        t = min(t, q.t_max())
        print(t)

        sgs = gs[i].subgridspec(1, 3)
        if i == 0:
            axes = [fig.add_subplot(s) for s in sgs]
        else:
            axes = [fig.add_subplot(s, sharex = gl_axes[0][j], sharey = gl_axes[0][j]) for j, s in enumerate(sgs)]
        gl_axes.append(axes)

        ax = axes[0]
        div = make_axes_locatable(ax)
        norm_dict = {f : None for f in fields}
        if norms is not None and isinstance(norms, dict):
            norm_dict.update(norms)
        else:
            norm_dict.update({'w':Normalize(0,2*(q.a.a0y**2))})
        for f in fields:
            plot_2d(ax, div, q, t, f, plane, masks[f], cbar = cbar, norm = norm_dict[f], **cms[f])
        ax.set_aspect('auto')

        ax = axes[2]
#         ax.set_title(q.df.split('/')[-2].replace('_',' '))
        q.read_energy(norm_to = 't').plot(ax=ax)
        ax.grid()
        
        ax = axes[1]
        
#         ff = q.read_field(t,'bx')
#         ff['xy'].isel()
#         fxy, fxz = ff['xy'], ff['xz']
#         ly, lz = int(0.5 * bx.y.size), int(0.5 * bx.z.size)
#         f1 = 0.5*fxy.isel({'y':slice(ly,-1)}) + 0.5*fxy.assign_coords({"y":fxy.y.values[::-1]}).isel({'y':slice(0,ly)})
#         f2 = 0.5*fxz.isel({'z':slice(lz,-1)}) + 0.5*fxz.assign_coords({"z":fxz.z.values[::-1]}).isel({'z':slice(0,lz)})
#         f = 0.5 * (f1 + f2.rename({'z':'y'}))
#         f = f.assign_coords({"y":f.y-f.y.isel(y=0)}).rolling(x=20).mean().interp(x=np.linspace(f.x.min(),f.x.max(),int(f.x.size/20)))
#         mx = max(-f.min(),f.max())
#         f.plot(ax=ax,x='x',y='y',cmap='bwr',vmin=-mx,vmax=mx,add_colorbar=False)
#         twax = ax.twinx()
#         onax = -(f.isel(y=0) + 0.5 * f.isel(y=1) + 0.25 * f.isel(y=2) + 0.125 * f.isel(y=3)) / (1 + 0.5 + 0.25 + 0.125)
#         onax.plot(ax=twax,color='k')
#         twax.axhline(0,linestyle='--',color='k')
#         ax.set_aspect('auto')
        
        try:
            spec = q.binned_statistics(t,'i','q','g','sum',(250),None)
            spec = spec.assign_coords({'g':(spec['g']-1)/q.a.icmr*0.511})
            spec = q.binned_statistics(t,'p','q','g','sum',(250),None)
            spec = spec.assign_coords({'g':(spec['g']-1)*0.511})
            spec.plot(ax=ax)
        except:
            pass
        ax.set_yscale('log')
        ax.set_xlabel('E, MeV')
        ax.grid()
    for ax in gl_axes:
        for a in ax:
            a.yaxis.set_tick_params(which='both', labelbottom=True)