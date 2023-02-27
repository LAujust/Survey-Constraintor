'''
NS Properties Constrainter module as well as Hubble constant constrainter sub-module,
fully base on observation survey and kilonova surrogate model.
'''

'--Copyright by Aujust--'
import os
home_dir = os.getcwd()

# Please enter the path to where you have placed the Schlegel, Finkbeiner & Davis (1998) dust map files
# You can also set the environment variable SFD_DIR to this path (in that case the variable below should be None)
sfd98_dir = os.path.join(home_dir, 'data/sfd98')
sfd98_dir = '/home/Aujust/data/Kilonova/Constraint/data/sfd98'

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import pickle
from scipy.interpolate import interp1d, griddata
import numpy as np
import pandas as pd
import pickle
import simsurvey
import toolkit as tkk
from multiprocessing import Pool
from astropy.time import Time
import simsurvey_tools as sst
from scipy.interpolate import RectBivariateSpline as Spline2d
import sncosmo
import astropy.units as u
from astropy.cosmology import Planck18, z_at_value
import dynesty
from tqdm import tqdm


pc10 = tkk.pc10

class Survey_constraintor(object):
    def __init__(self,plan_file=None,load_ccd_fun=sst.load_ztf_ccds,ccd_file='/home/Aujust/data/Kilonova/Constraint/data/ZTF_corners_rcid.txt',width=2.6,height=2.6):
        '''
        Setup Telescope information. This module was designed for calculate efficiency map with one or multiple EM-Counterpart observations
        of ONE/SPECIFIC telescope.
        In addition, it provide a new novel NS property constraint program with post-processed efficiency map. In this way, you can combine
        one or multiple observations of ONE or MULTIPLE telescopes!
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        This module also provide a Bayesian analysis scheme to constraint NS property, in which I treat detection efficiency as likelihhood of
        detection for a given model and survey. So the effciency map will be interpolated using Neural Network which shows better profermance 
        when there are more than three model parameters.
        '''
        try:
            self.wfst_bands = tkk.load_wfst_bands()
        except:
            print('WFST Bands already registered! ^_^')
        # Load the ZTF CCD corners and ZTF quadrants corners*********Will be replaced by WFST ccds
        self.ccds = load_ccd_fun()
        self.ccds = load_ccd_fun(filename=ccd_file, num_segs=64)
        self.width = width   #2.6 is default FoV of WFST
        self.height = height

        self.plan = plan_file
        #self.survey_all = dict()
        self.effcy_map_all = dict()

    def set_models(self,**kwargs):
        '''
        kn_model:     Kilonova model you use
        lc_model:     Model to generate wave,flux as input of sncosmo.Model
        cosmo_model:  sncosmo model
        random_p:     random_parameter
        '''
        self.kn_model = kwargs['kn_model']
        self.lc_model = kwargs['lc_model']
        self.cosmo_model = kwargs['cosmo_model']
        self.random_p = kwargs['random_p']

    def generate_plan(self,survey_file,field_file,zp=26):
        #field_file should be a dict contains 'field_id','ra','dec'
        #survey_file should be a dict or pandas.Dataframe contains 'time','field','band','maglim'


        self.survey_file = survey_file
        self.field_file = field_file
        obs = {'time': [], 'field': [], 'band': [], 'maglim': [], 'skynoise': [], 'comment': [], 'zp': []}

        for k in self.survey_file.keys():
            obs[k] = self.survey_file[k]

        obs['zp'] = [zp for i in range(len(self.survey_file.index))]
        obs['comment'] = ['' for i in range(len(self.survey_file.index))]
        obs['skynoise'] = 10**(-0.4 * (np.array(obs['maglim']) - zp)) / 5

        plan = simsurvey.SurveyPlan(time=obs['time'],
                            band=obs['band'],
                            skynoise=obs['skynoise'],
                            obs_field=obs['field'],
                            obs_ccd=None,
                            zp=obs['zp'],
                            comment=obs['comment'],
                            fields=self.field_file,
                            ccds=self.ccds,
                            width=2.6,
                            height=2.6
                            )
        self.plan = plan
        print(plan.pointings)

    def generate_transient(self,ntransient,rate,**kwargs):

        tr = simsurvey.get_transient_generator([self.z_min,self.z_max],
                                            ntransient=ntransient,
                                            ratefunc=lambda z: rate,
                                            sfd98_dir=sfd98_dir,
                                            **kwargs
                                            )
        survey = simsurvey.SimulSurvey(generator=tr, plan=self.plan, n_det=2, threshold=5., sourcenoise=False)

        return survey

    def _cal_effcy(self,param):
        #transientprop
        out = self.lc_model(param,model=self.kn_model,phase=np.linspace(0,7,100))
        if len(out) == 3:
            phase, wave, flux = out
            source = TimeSeriesSource(phase, wave, flux)
        elif len(out) == 4:
            phase, wave, cos_theta, flux = out
            source = AngularTimeSeriesSource(phase, wave, cos_theta, flux)
        dust = sncosmo.CCM89Dust()
        model = sncosmo.Model(source=source,effects=[dust, dust], effect_names=['host', 'MW'], effect_frames=['rest', 'obs'])
        transientprop = dict(lcmodel=model, lcsimul_func=self.random_p)
        self.kwargs['transientprop'] = transientprop

        survey = self.generate_transient(self.ntransient,self.rate,**self.kwargs)
        lcs = survey.get_lightcurves()
        efficy = len(lcs.lcs)/len(lcs.meta_full)
        out = np.concatenate((param,[efficy]))
        return out


    def get_effcy_map(self,param_flat,nprocess=20,event_id=None,rate=3e-5,ntransient=1000,out_dir=None,skymap_file=False,**kwargs):
        '''
        ntransient: [int]               override number of transient calculated from integrated 
                                    rate, ratefunc is still used to calculate shape of the 
                                    redshift distribution
    
        transientprop: [dict]      dict(lcmodel=model, lcsimul_func=random_parameters)
        dL: [float]    Luminosity Distance of the event
        ddL:[float]   Deviation of luminosity distance of the event
        mjd_range  MJD range for transients
        '''
        #Give each event a identified ID
        if event_id is None:
            event_id = np.random.randint(1000)
        if kwargs.get('dL',None) is None:
            self.z_min,self.z_max = kwargs['z_min'],kwargs['z_max']
        else:
            dL,ddL = kwargs['dL'],kwargs['ddL']
            self.z_min = z_at_value(Planck18.angular_diameter_distance,(dL-ddL)*u.Mpc,zmax=1,method='Bounded').value
            self.z_max = z_at_value(Planck18.angular_diameter_distance,(dL+ddL)*u.Mpc,zmax=1,method='Bounded').value
            del kwargs['dL']
            del kwargs['ddL']

        if kwargs.get('mjd_range',None) is None:
            kwargs['mjd_range'] = (self.plan.pointings['time'].min()-3, self.plan.pointings['time'].min()-0.1)

        if skymap_file:
            try:
                import healpy as hp
                prob, distmu,distsigma,distnorm = hp.fitsfunc.read_map(skymap_file,field=[0,1,2,3])
                self.lal_dict = {
                    'prob':prob,
                    'distmu':distmu,
                    'distsigma':distsigma
                }
                kwargs['skymap'] = self.lal_dict
            except:
                raise EOFError('Please install healpy first!')
        else:
            self.dec_range=(self.plan.pointings['Dec'].min()-5,self.plan.pointings['Dec'].max()+5)
            self.ra_range=(self.plan.pointings['RA'].min()-5,self.plan.pointings['RA'].max()+5)
            kwargs['ra_range'] = self.ra_range
            kwargs['dec_range'] = self.dec_range

        self.ntransient = ntransient
        self.rate = rate
        self.kwargs = kwargs
        print('Setting is done')

        total = int(param_flat.shape[0] * param_flat.shape[1])
        #with Pool(nprocess) as pool:
        #    result = list(tqdm(pool.imap(self._cal_effcy,param_flat), total=total))

        pool = Pool(processes=nprocess)
        result = pool.map(self._cal_effcy,param_flat)
        pool.close()
        pool.join()

        result = np.array(result)
        self.result = result
        self.effcy_map_all[event_id] = result 
    
    def _map_interpolator_train(self):
        params = self.result[:,:-1]
        effcy = self.result[:,-1]

        self.params_min = np.min(params,axis=0)
        self.params_max = np.max(params,axis=0)

        params_postprocess = (params-self.params_min)/(self.params_max-self.params_min)

        self.effcy_postprocess = effcy

        #NN- K Neighbours Regressor
        knn = KNeighborsRegressor(5,weights='distance')
        knn.fit(params_postprocess,self.effcy_postprocess)
        self.map_predictor = knn

    'Define log-likelihood and prior transformation'
    def _prior_transform(self,u_params):
        u_params = np.array(u_params)
        factor = self.bounds[:,1]-self.bounds[:,0]
        return factor*u_params+self.bounds[:,0]

    def _dlog_like(self,params):
        params_post_ = (params-self.params_min)/(self.params_max-self.params_min)
        deffcy = self.map_predictor.predict([params_post_])
        return np.log10(deffcy[0]+1e-10)
    

    def get_posterior(self,bounds,ndim,nlive=1000,out_dir=None,param_names=None):
        self._map_interpolator_train()
        self.bounds = bounds
        dsampler = dynesty.NestedSampler(self._dlog_like, self._prior_transform, ndim=ndim,nlive=nlive)
        dsampler.run_nested()
        self.posterior = dsampler.results
        self.param_names = param_names

        if out_dir is None:
            return self.posterior
        else:
            with open(out_dir,'wb') as f:
                pickle.dump(self.posterior,f)
            f.close()
                



    def load_plan(self,filename):
        with open(filename,'rb') as f:
            plan = pickle.load(f)
        f.close()
        print('Load plan from '+filename)
        self.plan = plan

    def _update(self):
        self.plan = None
        del self.survey_file
        del self.posterior




'------------------------------------------------------------------------------------------------'

def Bullamodel(dynwind=False, dataDir='./kilonova_models/bns_m3_3comp/', m_dyn=0.01, m_pm=0.110,phi=30):
    l = dataDir+'nph1.0e+06_mejdyn'+'{:.3f}'.format(m_dyn)+'_mejwind'+'{:.3f}'.format(m_pm)+'_phi'+'{:.0f}'.format(phi)+'.txt'
    f = open(l)
    lines = f.readlines()
    nobs = int(lines[0])
    nwave = float(lines[1])
    line3 = (lines[2]).split(' ')
    ntime = int(line3[0])
    t_i = float(line3[1])
    t_f = float(line3[2])
    cos_theta = np.linspace(0, 1, nobs)  # 11 viewing angles
    phase = np.linspace(t_i, t_f, ntime)  # epochs
    file_ = np.genfromtxt(l, skip_header=3)
    wave = file_[0:int(nwave),0]
    flux = []
    for i in range(int(nobs)):
        flux.append(file_[i*int(nwave):i*int(nwave)+int(nwave),1:])
    flux = np.array(flux).T

    return phase, wave, cos_theta, flux

# phase, wave, cos_theta, flux = Bullamodel()

def Possismodel(param_,model,phase):
    #M_dyn,M_wind,phi
    wave = np.linspace(1e2,9.99e4,500)
    cos_theta_list = np.linspace(0,1,11)
    #This is kilonovanet model
    flux = []
    for ii, cos_theta in enumerate(cos_theta_list):
        param = np.concatenate((param_,[cos_theta]))
        flux.append(model.predict_spectra(param,phase)[0].T/(4*np.pi*pc10**2))

    return phase, wave, cos_theta_list, np.array(flux).T

def Possismodel_tf(param_,model,phase):
    #M_dyn,M_wind,phi
    wave = np.linspace(1e2,9.99e4,500)
    cos_theta_list = np.linspace(0,1,11)
    #This is kilonovanet model
    flux = []
    for ii, cos_theta in enumerate(cos_theta_list):
        param = np.concatenate((param_,[cos_theta]))
        flux.append(model.calc_spectra(tt=phase,param_list=param)[-1])
    return phase, wave, cos_theta_list, np.array(flux).T

def Possismodel_angular(m_dyn,m_pm,phi,cos_theta,model,phase):

    wave = np.linspace(1e2,9.99e4,500)
    #This is kilonovanet model
    param = np.array([m_dyn,m_pm,phi,cos_theta])
    flux = model.predict_spectra(param,phase)[0].T/(4*np.pi*pc10**2)
    return phase, wave, np.array(flux).T

def BNS_model(param,model,phase=np.linspace(0,7,100)):
    #Mc,q,lambda_s,Mtov,eta_disk
    Mc,q,Ls,Mtov,eta = param[0],param[1],param[2],param[3],param[4]
    m1,m2 = _mc2ms(Mc,q)
    m_total = m1 + m2

    #Yagi and Yunes 2016 QUR to get L1 and L2 from Lsym
    a_ = 0.07550
    b_ = np.array([[-2.235, 0.8474],[10.45, -3.251],[-15.70, 13.61]])
    c_ = np.array([[-2.048, 0.5976],[7.941, 0.5658],[-7.360, -1.320]])
    n_ave = 0.743

    Fq = (1-(m2/m1)**(10./(3-n_ave)))/(1+(m2/m1)**(10./(3-n_ave)))

    nume = a_
    denom = a_

    for i in np.arange(3):
        for j in np.arange(2):
            nume += b_[i,j]*(m2/m1)**(j+1)*Ls**(-(i+1)/5)

    for i in np.arange(3):
        for j in np.arange(2):
            denom += c_[i,j]*(m2/m1)**(j+1)*Ls**(-(i+1)/5)

    La = Ls * Fq * nume / denom

    L1 = Ls - La
    L2 = Ls + La

    Lambda = 16./13 * ((m1 + 12*m2) * m1**4 * L1 +
                (m2 + 12*m1) * m2**4 * L2) / m_total**5

    # Approx radius of NS from De+ 2018
    radius_ns = 11.2 * Mc * (Lambda/800)**(1./6.)
    R14 = radius_ns

    # Compactness of each component (Maselli et al. 2013; Yagi & Yunes 2017)
    # C = (GM/Rc^2)
    C1 = 0.360 - 0.0355*np.log(L1) + 0.000705*np.log(L1)**2
    C2 = 0.360 - 0.0355*np.log(L2) + 0.000705*np.log(L2)**2

    #R1 = (tkk.G_CGS * m1 * tkk.M_SUN_CGS / (C1 * tkk.C_CGS**2)) / 1e5
    #R2 = (tkk.G_CGS * m2 * tkk.M_SUN_CGS / (C2 * tkk.C_CGS**2)) / 1e5

    M_dyn = _dyn_mass(m1,m2,C1,C2)
    M_wind = eta * _wind_mass(m1,m2,Mtov,R14)
    fred = _cal_fred(m1,m2)
    Phi = np.rad2deg(np.arcsin(fred))

    param_ = np.array([M_dyn,M_wind,Phi])
    phase, wave, cos_theta, flux = Possismodel(param_,model,phase)
    return phase,wave,cos_theta,flux




#NS property TO ejecta property
#some constant
a,b,d,n = -0.0719,0.2116,-2.42,-2.905
a0,da,b0,db,cc,dd,beta,qtrans = -1.581,-2.439,-0.538,-0.406,0.953,0.0417,3.910,0.9
G,c,Msun = 6.6743e-11,3e8,2e30
A,B,C = 14.8609,-28.6148,13.9597
a_vd,b_vd,c_vd = 0.309,0.657,-1.879

def _mc2ms(Mc,q):
    m1 = Mc * q ** -0.6 * (q + 1.0) ** 0.2
    m2 = m1 * q  #q<1
    return m1,m2


def _dyn_mass(m1,m2,c1,c2):
    s = (a*(1-2*c1)*m1/c1 + b*m2*(m1/m2)**n + d/2)
    s = s + (a*(1-2*c2)*m2/c2 + b*m1*(m2/m1)**n + d/2)
    return 10**s

def _wind_mass(m1,m2,Mtov,R14):
    q = min([m1/m2,m2/m1])
    xi = np.tanh(beta*(q-qtrans))/2
    aa = a0+da*xi
    bb = b0+db*xi
    Mthr = (2.38-3.606*Mtov/R14)*Mtov
    log_disk = max([-3,aa*(1+bb*np.tanh((cc-(m1+m2)/Mthr)/dd))])
    return 0.2 * 10**log_disk

def _cal_fred(m1,m2):
    q = min([m1/m2,m2/m1])
    return min([1,A*q**2+B*q+C])  # Phi = arcsin(fred)




def Linearmodel(peak_M, alpha, mode):
    phase = np.linspace(0,20,100)
    wave = np.linspace(1e3,1e5,500)

    def M2flux(M):
        #at 10pc
        f_wave = 350e-11*10**(-M/2.5)
        return [f_wave for i in range(len(wave))]

    flux = []
    if mode == 'monochro':
        for t in phase:
            if alpha >= 0:
                f = M2flux(peak_M+alpha*t)
                flux.append(f)
            elif alpha < 0:
                f = M2flux(peak_M+alpha*(t-3))
                flux.append(f)
    elif mode == 'updown':
        for t in phase:
            if t < 0.5:
                f = M2flux(peak_M+alpha*(0.5-t))
                flux.append(f)
            elif t >= 0.5:
                f = M2flux(peak_M+alpha*(t-0.5))
                flux.append(f)
    else:
        print("Mode dosen't match!")
        return None

    return phase, wave, np.array(flux)


def random_parameters(redshifts, model,r_v=2., ebv_rate=0.11,**kwargs):
    # Amplitude
    amp = []
    for z in redshifts:
        #amp.append(10**(-0.4*Planck18.distmod(z).value))
        d_l = Planck18.luminosity_distance(z).value * 1e5
        amp.append(d_l**-2)

    return {
        'amplitude': np.array(amp),
        'hostr_v': r_v * np.ones(len(redshifts)),
        'hostebv': np.random.exponential(ebv_rate, len(redshifts))
        }

def random_parameters_ang(redshifts, model,r_v=2., ebv_rate=0.11,**kwargs):
    # Amplitude
    amp = []
    for z in redshifts:
        amp.append(10**(-0.4*Planck18.distmod(z).value))
    
    theta = np.arccos(np.random.random(len(redshifts))) / np.pi * 180

    return {
        'amplitude': np.array(amp),
        'theta': theta, 
        'hostr_v': r_v * np.ones(len(redshifts)),
        'hostebv': np.random.exponential(ebv_rate, len(redshifts))
        }


class TimeSeriesSource(sncosmo.Source):
    """A single-component spectral time series model.
    The spectral flux density of this model is given by
    .. math::
       F(t, \\lambda) = A \\times M(t, \\lambda)
    where _M_ is the flux defined on a grid in phase and wavelength
    and _A_ (amplitude) is the single free parameter of the model. The
    amplitude _A_ is a simple unitless scaling factor applied to
    whatever flux values are used to initialize the
    ``TimeSeriesSource``. Therefore, the _A_ parameter has no
    intrinsic meaning. It can only be interpreted in conjunction with
    the model values. Thus, it is meaningless to compare the _A_
    parameter between two different ``TimeSeriesSource`` instances with
    different model data.
    Parameters
    ----------
    phase : `~numpy.ndarray`
        Phases in days.
    wave : `~numpy.ndarray`
        Wavelengths in Angstroms.
    flux : `~numpy.ndarray`
        Model spectral flux density in erg / s / cm^2 / Angstrom.
        Must have shape ``(num_phases, num_wave)``.
    zero_before : bool, optional
        If True, flux at phases before minimum phase will be zeroed. The
        default is False, in which case the flux at such phases will be equal
        to the flux at the minimum phase (``flux[0, :]`` in the input array).
    time_spline_degree : int, optional
        Degree of the spline used for interpolation in the time (phase)
        direction. By default this is set to 3 (i.e. cubic spline). For models
        that are defined with sparse time grids this can lead to large
        interpolation uncertainties and negative fluxes. If this is a problem,
        set time_spline_degree to 1 to use linear interpolation instead.
    name : str, optional
        Name of the model. Default is `None`.
    version : str, optional
        Version of the model. Default is `None`.
    """

    _param_names = ['amplitude']
    param_names_latex = ['A']

    def __init__(self, phase, wave, flux, zero_before=False,
                 time_spline_degree=3, name=None, version=None):
        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._parameters = np.array([1.])
        self._model_flux = Spline2d(phase, wave, flux, kx=time_spline_degree,
                                    ky=3)
        self._zero_before = zero_before

    def _flux(self, phase, wave):
        f = self._parameters[0] * self._model_flux(phase, wave)
        if self._zero_before:
            mask = np.atleast_1d(phase) < self.minphase()
            f[mask, :] = 0.
        return f





# AngularTimeSeriesSource classdefined to create an angle dependent time serie source.
class AngularTimeSeriesSource(sncosmo.Source):
    """A single-component spectral time series model.
        The spectral flux density of this model is given by
        .. math::
        F(t, \lambda) = A \\times M(t, \lambda)
        where _M_ is the flux defined on a grid in phase and wavelength
        and _A_ (amplitude) is the single free parameter of the model. The
        amplitude _A_ is a simple unitless scaling factor applied to
        whatever flux values are used to initialize the
        ``TimeSeriesSource``. Therefore, the _A_ parameter has no
        intrinsic meaning. It can only be interpreted in conjunction with
        the model values. Thus, it is meaningless to compare the _A_
        parameter between two different ``TimeSeriesSource`` instances with
        different model data.
        Parameters
        ----------
        phase : `~numpy.ndarray`
        Phases in days.
        wave : `~numpy.ndarray`
        Wavelengths in Angstroms.
        cos_theta: `~numpy.ndarray`
        Cosine of
        flux : `~numpy.ndarray`
        Model spectral flux density in erg / s / cm^2 / Angstrom.
        Must have shape ``(num_phases, num_wave, num_cos_theta)``.
        zero_before : bool, optional
        If True, flux at phases before minimum phase will be zeroed. The
        default is False, in which case the flux at such phases will be equal
        to the flux at the minimum phase (``flux[0, :]`` in the input array).
        name : str, optional
        Name of the model. Default is `None`.
        version : str, optional
        Version of the model. Default is `None`.
        """

    _param_names = ['amplitude', 'theta']
    param_names_latex = ['A', r'\theta']

    def __init__(self, phase, wave, cos_theta, flux, zero_before=True, zero_after=True, name=None,
                 version=None):
        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._cos_theta = cos_theta
        self._flux_array = flux
        self._parameters = np.array([1., 0.])
        self._current_theta = 0.
        self._zero_before = zero_before
        self._zero_after = zero_after
        self._set_theta()

    def _set_theta(self):
        logflux_ = np.zeros(self._flux_array.shape[:2])
        for k in range(len(self._phase)):
            adding = 1e-10 # Here we are adding 1e-10 to avoid problems with null values
            f_tmp = Spline2d(self._wave, self._cos_theta, np.log(self._flux_array[k]+adding),
                             kx=1, ky=1)
            logflux_[k] = f_tmp(self._wave, np.cos(self._parameters[1]*np.pi/180)).T

        self._model_flux = Spline2d(self._phase, self._wave, logflux_, kx=1, ky=1)

        self._current_theta = self._parameters[1]

    def _flux(self, phase, wave):
        if self._current_theta != self._parameters[1]:
            self._set_theta()
        f = self._parameters[0] * (np.exp(self._model_flux(phase, wave)))
        if self._zero_before:
            mask = np.atleast_1d(phase) < self.minphase()
            f[mask, :] = 0.
        if self._zero_after:
            mask = np.atleast_1d(phase) > self.maxphase()
            f[mask, :] = 0.
        return f



'######################################'