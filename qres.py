import os
import re
import time
import scipy
import scipy.stats
import numpy as np
import xarray as xr
import pandas as pd
import threading
import ipywidgets as widgets
from tqdm.notebook import tqdm
from IPython.display import display
from expression_parser import Parser
from itertools import product
from collections import namedtuple

# For some reason np.sqrt raises warning when applied to long array
np.seterr(invalid='ignore')

def get_ioloop():
    import IPython, zmq
    ipython = IPython.get_ipython()
    if ipython and hasattr(ipython, 'kernel'):
        return zmq.eventloop.ioloop.IOLoop.instance()

#The IOloop is shared
ioloop = get_ioloop()

@xr.register_dataset_accessor("field")
class FieldAccessor:
    def __init__(self, dataset, smooth = [5,5]):
        self._obj = dataset
        self.onaxis('x', smooth)
        self.onaxis('y', smooth)
        self.onaxis('z', smooth)

    def onaxis(self, axis='x', smooth=[5,5]):
        dim1, dim2 = 'xyz'.replace(axis, '')
        ds = self._obj
        n1, n2 = len(ds.coords[dim1]), len(ds.coords[dim2])
        f1, f2, = ds['xyz'.replace(dim2,'')], ds['xyz'.replace(dim1,'')]
        mids = [int(0.5*n1), int(0.5*n2)]
        slices = [None, None]
        for i, s in enumerate(smooth):
            if s == -1:
                slices[i] = slice(0, -1)
            else:
                slices[i] = slice( mids[i] - s - 1, mids[i] + s )
        f =  0.5 * f1[{f'{dim1}':slices[0]}].mean(dim=dim1, skipna=True) + \
             0.5 * f2[{f'{dim2}':slices[1]}].mean(dim=dim2, skipna=True)
        if axis == 'x':
            self.xx = f
            ds['xx'] = f
        elif axis == 'y':
            self.yy = f
            ds['yy'] = f
        elif axis == 'z':
            self.zz = f
            ds['zz'] = f
        return f

class Binarizer(threading.Thread):
    def __init__(self, ioloop, qres):
        super().__init__()
        self.qres = qres
        self.ioloop = ioloop
        self.output = widgets.HTML(value='Binarizing data in the background')
        self._quit = threading.Event()
        self.start()
        display(self.output)
    
    def run(self):
        i = 0
        res = 1
        ready_fields = []
        fnames = ''.join([f'{f}|' for f in self.qres.readable_fields])
        pattern = f'({fnames})  '
        while not self._quit.isSet():
            all_fields = [file for file in os.listdir(self.qres.df) if re.match(pattern+'$', file) is not None]
            ready_fields = [file[:-3] for file in os.listdir(self.qres.df) if re.match(pattern+'\.nc$', file) is not None]
            for f in all_fields:
                field, t = re.split(f'({fnames})',f,maxsplit=1)[-2:]
                t = float(t)
                # print(field, t)
                if f not in ready_fields:
                    try:
                        self.qres.read_field(t, field) # Reading and saving data to .nc file to allow lazy access
                    except:
                        continue
                    # ready_fields.append(f)
                    res = f'{field} at {t}'
                    break
                res = 'is done'
            def update_output(i, res):
                self.output.value = f'Binarizing data in the background. Proccesing {res}'+''.join(['.' for j in range(i%4)])
            time.sleep(0.05)
            # self.qres.binarize_fields()
            i += 1
            self.ioloop.add_callback(update_output, i, res)
        self.output.value = 'Quitted'
            
    def quit(self):
        self._quit.set()

class QRES:
    df = None
    t_current = 0
    particle_space = []
    readable_fields = []
    readable_particle_data = []
    computables = []
    debug = True
    parser = None

    def __init__(self, df, binarize_fields = False, binarize_particles = True, load=['e','b','n_P','n_E','n_G','n_I']):
        self.df = df
        log = pd.read_csv(df+'log',header=None,sep='\t',na_values='$').dropna()
        log = log.iloc[:(log[0].str.contains('#')).idxmax()-1]
        log = pd.DataFrame(log.values.reshape(-1,2)).transpose()
        log.columns = log.iloc[0]
        log.drop(log.index[0], inplace=True)
        log = log.apply(pd.to_numeric, errors='ignore')
        try: 
            log['icmr']
        except KeyError:
            log['icmr'] = -9999
        for c in 'xyz':
            log[f'l{c}']=log[f'd{c}']*log[f'n{c}']
        self.conf = log
        d = log.rename(columns={'lambda':'lamda'}).to_dict('records')[0]
        attrs = namedtuple('Attributes', d)
        self.a = attrs(**d)
        fields = ['e','b','j']
        self.readable_fields = [''.join(i) for i in product(fields,['x','y','z'],['','_beam'])]+ ['w','inv','n_E','n_P','n_I','n_G'] + fields
        self.computables += self.readable_fields

        self.particle_space = ['q','x','y','z','ux','uy','uz','g','chi','ID']
        self.readable_particle_data = [''.join(i) for i in product([''.join(i) for i in product(['','u','v','theta','phi'],['x','y','z'])] + ['q','g','chi','ID','vperp','uperp','u','v'], ['_E','_I','_P','_G'])]
        self.computables += self.readable_particle_data

        self.ts = pd.Index(np.arange(0, self.t_max() + self.a.output_period, self.a.output_period), name='t')
        
        self.parser = Parser(self.computables, self.read_grid_data)
#         if binarize_fields:
#             self.binarizer = self.Binarizer(get_ioloop(), self)

        # if len(load) > 0:
        #     self.fields = self.load_fields(self.ts, load)
        # self.energy = self.read_energy()
        # self.n = self.read_N()

    def rename_outputs(self):
        pars = {'_p':'_P','_ph':'_G',f"_{self.a.icmr:g}_":'_I'}
        fields = {'rho':'n','phasespace':'par','deleted':'del'}
        fin = {'n':'n_E','par':'par_E','in_':'n_'}
        arr = [(pars, '\w*', '(_beam)*([0-9]+)(\.[0-9]*)*'), (fields, '\w*', ''), (fin, '', '(_beam)*[^_]')]
        files_orig = os.listdir(self.df)
        files = files_orig.copy()
        for d, pre, post in arr:
            for i, f in enumerate(files):
                for key, value in d.items():
                    if re.match(pre+key+post, files[i]) is not None:
                        files[i] = re.sub(key, value, files[i])
        for i, f in enumerate(files):
            if files_orig[i] != f:
                os.rename(f'{self.df}'+files_orig[i], f'{self.df}'+f)


    def t_max(self):
        """
        Returns maximum time instance for which data is availiable
        """
        for f in self.readable_fields:
            try:
                return max([float(file[len(f):]) for file in os.listdir(self.df) if re.match(f'{f}([0-9]+)(\.[0-9]*)*', file) is not None])
            except ValueError:
                continue
        print('Cannot find any field data')
        return 999

    def attr(self,a):
        return getattr(self.a, a)
        
    def read_particles(self, t = None, p = 'e', space = None, chunks = 'auto', amount = 1.0, save = True):
        """
        Reads particles data.
        
        Parameters
        -------
        t : {'max', None} or number, default None
            Time instance at which data is read. If None, uses the most recent one used. If 'max', uses the lates time instance availiable.
        
        p : str or list of str, default 'e'
            Type of particles to read. Multiple types are allowed.
        
        space : str or list of str
            Additional data to compute. Supported spaces are 'vx','vy','vz','vperp','v','uperp','u','thetax','thetay','thetaz','phix','phiy','phiz','n','density'. 
            - 'thetax', 'thetay', 'thetaz' are the angles between momentum vector and X, Y, Z axes correspondingly. 
            - 'phix' is the angle of momentum vector in YZ plane counting from Y axis, 'phiy' - in XZ plane counting from Z axis, 'phiz' - in XY plane counting from X axis. 
            - 'n' is number of real particles in  quasi-particle. 'density' is average density of quasi-particle (in cm^-3).
            
        chunks : {'auto', True, False}, default 'auto'
            Whether to read data in chunks. May be more efficient when small number of particles is needed to be read.
            
        amount : int or float, default 1.0
            If int sets total number of particles to read. If float < 1 sets percentage of total particles.
        """

        h5file = f'{self.df}{p}{t:g}.h5'
        try:
            with pd.HDFStore(h5file) as store:
                data = store['df']
                if self.debug:
                    print('Reading from file '+h5file)
        except KeyError:
            particle_space = self.particle_space
            s = 'phasespace'
            d = {'e':'','p':'_p','g':'_ph','i':f"_{self.a.icmr:g}_"}
            try:
                s += d[p]
            except KeyError:
                print('Unsupported type of particles')
                return
            if p == 'g':
                s = 'deleted_ph'
                particle_space = ['ID','q','x','y','z','ux','uy','uz','g','chi']
            df = self.df+s+f'{t:g}'
            
            nchunks = 1 # number of chunks
            if chunks: # !! REDO MORE CONSCIOUSLY !!
                nchunks = 10
                def file_len(fname):
                    with open(fname) as f:
                        for i, _ in enumerate(f):
                            pass
                    return i + 1
                
                chunksize = int(file_len(df) / len(particle_space) / 10) * len(particle_space)
            else:
                chunksize = None
            
            if amount <= 1.0:
                arg = {'frac':amount}
            else:
                arg = {'n':int(np.ceil(amount/nchunks))}
            iter_csv = pd.read_csv(df, header=None,sep='\t', chunksize=chunksize, iterator=True)
            data = pd.concat( [pd.DataFrame(chunk.values.reshape(-1,len(particle_space)),columns=particle_space).sample(**arg) for chunk in iter_csv] )
            data.reset_index(inplace = True, drop = True)
            if data.shape[0] > amount and amount > 1:
                data = data.iloc[:int(amount)]
            if save:
                if self.debug:
                    print('Writing to file '+h5file)
                with pd.HDFStore(h5file) as store:
                    store['df'] = data

        if space is not None:
            space = np.array(space).reshape(-1)
            for s in space:
                if s == 'vx':
                    data['vx'] = data['ux'] / data['g']
                elif s == 'vy':
                    data['vy'] = data['uy'] / data['g']
                elif s == 'vz':
                    data['vz'] = data['uz'] / data['g']
                elif s == 'vperp':
                    data['vperp'] = np.sqrt( data['uy']*data['uy'] + data['uz']*data['uz']  ) / data['g']
                elif s == 'v':
                    data['v'] = np.sqrt( data['ux']*data['ux'] + data['uy']*data['uy'] + data['uz']*data['uz']  ) / data['g']
                elif s == 'uperp':
                    data['uperp'] = np.sqrt( data['uy']*data['uy'] + data['uz']*data['uz']  )
                elif s == 'u':
                    data['u'] = np.sqrt( data['ux']*data['ux'] + data['uy']*data['uy'] + data['uz']*data['uz']  )
                elif s == 'thetax':
                    data['thetax'] = np.arctan2( np.sqrt( data['uy']*data['uy'] + data['uz']*data['uz']  ) , data['ux'] )
                elif s == 'thetay':
                    data['thetay'] = np.arctan2( np.sqrt( data['uz']*data['uz'] + data['ux']*data['ux']  ) , data['uy'] )
                elif s == 'thetay':
                    data['thetay'] = np.arctan2( np.sqrt( data['ux']*data['ux'] + data['uy']*data['uy']  ) , data['uz'] )
                elif s == 'phix':
                    data['phix'] = np.arctan2( data['uz'], data['uy'] )
                elif s == 'phiy':
                    data['phiy'] = np.arctan2( data['ux'], data['uz'] )
                elif s == 'phiz':
                    data['phiz'] = np.arctan2( data['uy'], data['ux'] )
                elif s == 'n':
                    data['n'] = np.abs(data['q'])*self.a.dx*self.a.dy*self.a.dz*1.11485e13*self.a.lamda
                elif s == 'density':
                    data['density'] = np.abs(data['q'])*1.11485e13*np.power(self.a.lamda, 2)
                elif s not in self.particle_space:
                    raise AttributeError(f"Particles don't have {s} property")
        return data

    def binned_statistics(self, t=None, p='e', weights=None, space=['x','y'], stat='mean', target_shape='auto', range='full', chunks=True, amount=1.0, smooth=1, filter='', add_space=[], data = None):
        space = np.ravel(space).tolist()         #
        weights = np.ravel(weights).tolist()     # To ensure that they are lists
        add_space = np.ravel(add_space).tolist() #
        if isinstance(data, pd.DataFrame):
            particles = data
        else:
            particles = self.read_particles(t, p, space + weights + add_space, chunks, amount)
        if filter != '':
            particles = particles.query(filter)
        bins = ['auto'] * len(space) if target_shape == 'auto' else target_shape
        rng  = ['full'] * len(space) if range == 'full' else range
        for i, s in enumerate(space):
            if s in 'xyz':
                ns = self.attr(f"n{s}")
                ds = self.attr(f"d{s}")
                ls = self.attr(f"l{s}")
                if bins[i] == 'auto':
                    bins[i] = np.linspace(-ds*0.5, ls + ds*0.5, ns+1)
                if rng[i] == 'full':
                    rng[i] = [0,ls]
            else:
                mn, mx = particles[s].min(), particles[s].max()
                if bins[i] == 'auto':
                    bins[i] = np.linspace(mn, mx, 250)
                if rng[i] == 'full':
                    rng[i] = (mn, mx)
        if stat == 'mean' and weights != 'q':
            sumw, edges, _ = scipy.stats.binned_statistic_dd(particles[space].values, [particles[w].values * particles['q'].values for w in weights], statistic = 'sum', bins = bins, range = rng)
            sumq, edges, _ = scipy.stats.binned_statistic_dd(particles[space].values, particles['q'].values, statistic = 'sum', bins = bins, range = rng)
            stats = [st / sumq for st in sumw]
        else:
            stats, edges, _ = scipy.stats.binned_statistic_dd(particles[space].values, [particles[w].values for w in weights], statistic = stat, bins = bins, range = rng)
        ss = [xr.DataArray(st, coords=[(s,0.5*(edges[i][:-1]+edges[i][1:])) for i, s in enumerate(space)]) for st in stats]
        if len(weights) == 1:
            return ss[0]
        else:
            return xr.Dataset({w:ss[i] for i, w in enumerate(weights)})

    def load_fields(self, ts, fields, update = True):
        ts = pd.Index(ts, name='t')
        return xr.Dataset({f : xr.concat([self.read_field(t, f).field.onaxis() for t in tqdm(ts)], dim=ts) for f in fields})

    def field_to_filename(self,field):
        d = {'n_E':'rho','n_P':'rho_p','n_G':'rho_ph','n_I':f"irho_{self.a.icmr:g}_"}
        try:
            return d[field]
        except:
            return field

    def read_field(self, t=None, field='ey', save = True):
        """
        Reads fields data in 1d or 2d.
        
        Parameters
        -------
        t : {'max', None} or number, default None
            Time instance at which data is read. If None, uses the most recent one used. If 'max', uses the lates time instance availiable.
        
        field : str 
            List of availiable fields:
             - ex, ey, ez, bx, by, bz, jx, jy, jz - cartesian components of electric electric field, magnetic field, current density
             - e, b, j - amplitude of electric field, magnetic field, current density
             - w, inv - electromagnetic energy density, invariant E^2-B^2
             - ne, np, ni, ng - electron, positron, ion, photon density
        """
        
        ncfile = f'{self.df}{field}{t:g}.nc'
        try:
            f = xr.open_dataset(ncfile)
            if self.debug:
                print('Reading from file '+ncfile)
            return f
        except FileNotFoundError:
            if field in ['e','b','j']:
                fx, fy, fz = self.read_field(t,f"{field}x"), self.read_field(t,f"{field}y"), self.read_field(t,f"{field}z")
                return np.sqrt( fx * fx + fy * fy + fz * fz )
            elif field == 'q':
                return self.read_field(t,'n_P') + self.read_field(t,'n_I') - self.read_field(t,'n_E')
            elif field in self.readable_fields:
                file = f"{self.df}{self.field_to_filename(field)}{t:g}"
                nx, ny, nz = self.a.nx, self.a.ny, self.a.nz
                if self.a.output_mode == 1 and field != 'w': # Binary mode
                    f = np.fromfile(file)
                else: # Text mode
                    f = pd.read_csv(file,header=None,sep='\t').values
                f, fyz = np.split(f, [nx*(ny+nz)])
                xs = np.linspace(0, self.a.lx, nx)
                ys = np.linspace(0, self.a.ly, ny)
                zs = np.linspace(0, self.a.lz, nz)
                fxy = xr.DataArray(f.reshape((nx,ny+nz))[:,:-nz], coords=[('x',xs),('y',ys)])
                fxz = xr.DataArray(f.reshape((nx,ny+nz))[:,ny:], coords=[('x',xs),('z',zs)])
                fyz = xr.DataArray(fyz.reshape((ny,nz)), coords=[('y',ys),('z',zs)])
                f = xr.Dataset({'xy':fxy,'xz':fxz,'yz':fyz})
                if save:
                    if self.debug:
                        print('Writing to file '+ncfile)
                    f.to_netcdf(ncfile, format='NETCDF4', engine='h5netcdf')
                return f
    
    def read_grid_data(self,t=None,field='ex',space=['x','y'],stat='mean',target_shape='auto',range='full',smooth=[5,5]):
        smooth = np.reshape(smooth,(-1))
        if field in self.readable_fields:
            space = "".join(space)
            if space in ['xy','xz','yz']:
                return self.read_field(t,field)[space]
            elif space in ['x','y','z']:
                pass
            else:
                raise AttributeError(f"No data available for {field} in {space} space")
        elif field in self.readable_particle_data:
            split = field.split('_')
            p = split[1].lower()
            weight = split[0]
            depth = 'xyz'
            for s in space:
                depth = depth.replace(s,'')
            filter = ''
            for i, d in enumerate(depth):
                delta = self.attr(f"d{d}")
                mid = self.attr(f"n{d}")*delta*0.5
                filter += f"{d} < {mid+smooth[i]*delta:.2f} and {d} > {mid-smooth[i]*delta:.2f} and "
            filter = filter[:-5]
            print(t, p, weight, space, stat, target_shape, range)
            return self.binned_statistics(t, p, weight, space, stat, target_shape, range, chunks=True, amount=1.0, smooth=1, filter="")

    def compute(self, t, expression, space = ['x','y'], stat = 'mean', target_shape = 'auto', range='full', order = 'cr', smooth = [5,5]):
        """
        Computes any expression envolving fields and/or particles on 1d or 2d regular grid. 
        
        Parameters
        ----------
        t : [type]
            [description]
        expression : [type]
            [description]
        target_shape: 'auto', list of ints or int, default 'auto'
            Desired shape of returned array.
        order : {'cr' | 'rc'}, default 'cr'
            Order of computation: 'cr': compute expression first in field shape, then resample to target shape; 'rc': resample fields to target_shape first, then compute expression. Ignored if target_shape is set to 'auto'
        """

        if order == 'cr':
            res = self.parser.evaluate(expression,'field',t=t,space=space,stat=stat,target_shape='auto',range=range,smooth=smooth)
            return res
        elif order == 'rc':
            return self.parser.evaluate(expression,'field',t=t,space=space,stat=stat,target_shape=target_shape,range=range,smooth=smooth)

    def read_tracks(self, p = 'e', amount = 1.0, vx=0, seed = 0, condition='', space = None, save = True):
        ncfile = f'{self.df}tracks_{p}.nc'
        try:
            tracks = xr.open_dataset(ncfile)
            if self.debug:
                print('Reading from file '+h5file)
        except FileNotFoundError:
            if space is None:
                space = self.particle_space + ['ex', 'ey', 'ez', 'bx', 'by', 'bz']
            cmr = {'e':-1,'p':1,'g':0,'i':self.a.icmr}[p]
            files = [file for file in os.listdir(self.df) if re.match(f"track_{cmr}_[0-9]",file) is not None]
            if amount <= 1:
                amount = amount * len(files)
    #             amount = amount * len([file for file in os.listdir(self.df) if re.match(f"track_[0-9]",file) is not None])
            amount = int(amount)
            print(amount)
            trs=[]
            i = seed
            for f in np.roll(files, seed)[:amount]:
                tmp=pd.DataFrame(pd.read_csv(self.df+f,header=None,sep='\t').values.reshape(-1,len(space)),columns=space).to_xarray()
    #                 tmp=pd.DataFrame(pd.read_csv(self.df+f'track_{i}',header=None,sep='\t').values.reshape(-1,len(self.particle_space)),columns=self.particle_space)
                # tmp['n'] = i
                tmp=tmp.rename({'index':'t'})
                tmp.coords['t'] = tmp.coords['t'] * self.a.dt
                tmp['x'] = tmp['x'] - vx * tmp.coords['t']
                # if condition != '':
                #     if tmp.shape == tmp.query(condition).shape:
                #         trs.append(tmp)
                # else:
                #     trs.append(tmp)
                trs.append(tmp)
            tracks = xr.concat(trs, xr.IndexVariable('n', np.arange(0,len(trs))))
            if save:
                if self.debug:
                    print('Writing to file '+ncfile)
                tracks.to_netcdf(ncfile, format='NETCDF4', engine='h5netcdf')
        return tracks
    
    def read_energy(self,norm_to='t',cut_instability=True):
        """
        Reads energy including deleted particles if availiable.
        
        Parameters
        -------
        norm_to : {'em','e','p','g','i','t','max', None} or number, default None
            Divide result by initial energy, maximum total energy or number
        
        cut_instability : bool, default True
            Delete data points after occurance of numeric instability

        Returns
        -------
        out : pandas DataFrame
        """
        
        energy = pd.read_csv(self.df+'energy',header=None,sep='\t',names=['w','e','p','g','i'])
        try:
            deleted = pd.read_csv(self.df+'energy_deleted',header=None,sep='\t',names=['e_del','p_del','g_del','i_del'])
#             energy = energy.add(deleted, fill_value=0)
            energy = energy.join(deleted)
        except FileNotFoundError:
            if self.debug:
                print('Deleted energy file not found')
        energy['t'] = energy.sum(axis=1)
        energy.index *= self.a.dt
        energy.index.name = 't'
        if cut_instability:
            energy = energy.iloc[0:np.argmax(np.gradient(energy['t'])/energy['t'].iloc[0] > 0.5)-10]
        if norm_to in energy.columns:
            energy /= energy[norm_to][0]
        elif norm_to == 'max':
            energy /= energy['t'].max()
        elif norm_to is not None:
            energy /= norm_to
        return energy
    
    def read_N(self):
        """
        Reads particles number.
        """
        
        N = pd.read_csv(self.df+'N',header=None,sep='\t',names=['e','p','g'])
        N.index *= self.a.dt
        N.index.name = 't'
        return N

    def read_spectrum(self, t, p):
        d = {'e':'','p':'_p','g':'_ph','i':f"_{self.a.icmr:g}_"}
        suff = d[p]
        s = pd.read_csv(self.df+'spectrum'+suff+f"{t:g}",header=None,sep='\t',names=['n'])
        s.index *= self.a.__getattribute__("deps"+suff)
        return s
    
    def text_to_bin(self):
        pass