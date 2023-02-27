#aujust@mail.ustc.edu.cn
#Toolkit for Aujust

import numpy as np
import sncosmo
import astropy.units as u
import astropy.constants as c

def __check__():
    print('Check~')


def load_filter_list():
    all = True
    filt_list = ['sdssu','sdssg','sdssr','sdssi','sdssz','desg','desr','desi','desz','desy','f435w','f475w','f555w','f606w','f625w',
        'f775w','nicf110w','nicf160w','f098m','f105w','f110w','f125w','f127m','f139m','f140w','f153m','f160w','f218w','f225w',
        'f275w','f300x','f336w','f350lp','f390w','f689m','f763m','f845m','f438w','uvf555w','uvf475w',
        'uvf606w','uvf625w','uvf775w','uvf814w','uvf850lp','cspb','csphs','csphd','cspjs','cspjd','cspv3009',
        'cspv3014','cspv9844','cspys','cspyd','cspg','cspi','cspk','cspr','cspu','f070w','f090w','f115w','f150w',
        'f200w','f277w','f356w','f444w','f140m','f162m','f182m','f210m','f250m','f300m','f335m','f360m','f410m','f430m',
        'f460m','f480m','lsstu','lsstg','lsstr','lssti','lsstz','lssty','keplercam::us','keplercam::b','keplercam::v','keplercam::v',
        'keplercam::r','keplercam::i','4shooter2::us','4shooter2::b','4shooter2::v','4shooter2::r','4shooter2::i','f062','f087',
        'f106','f129','f158','f184','f213','f146','ztfg','ztfr','ztfi','uvot::b','uvot::u','uvot::uvm2','uvot::uvw1','uvot::uvw2',
        'uvot::v','uvot::white','ps1::open','ps1::g','ps1::r','ps1::i','ps1::z','ps1::y','ps1::w','atlasc','atlaso','2massJ',
        '2massH','2massKs','wfst_u','wfst_g','wfst_r','wfst_i','wfst_z','wfst_w'
    ]
    for filt in filt_list:
        try:
            _x = sncosmo.get_bandpass(filt)
        except:
            print('Fail for '+filt)
            if all:
                all = False
    if all:
        print('Load all filters successfully!')
    return filt_list

def load_wfst_bands():
    add_bands = ['u','g','r','i','w','z']
    wfst_bands = ['wfst_'+i for i in add_bands]
    for add_band in add_bands:
        data = np.loadtxt('/home/Aujust/data/Kilonova/WFST/transmission/WFST_WFST.'+add_band+'_AB.dat')
        wavelength = data[:,0]
        trans = data[:,1]
        band = sncosmo.Bandpass(wavelength, trans, name='wfst_'+add_band)
        sncosmo.register(band, 'wfst_'+add_band)
    return wfst_bands

def mab2flux(mab):
    #erg s^-1 cm^-2
    return 10**(-(mab+48.6)/2.5)

def flux2mab(f):
    #erg s^-1 cm^-2
    return -2.5*np.log10(f)-48.6

def sumab(mab_list):
    _flux_all = 0
    for mab in mab_list:
        _flux_all += mab2flux(mab)
    return flux2mab(_flux_all)


#--------------------------------------#
#               Constant               #
#--------------------------------------#

day2sec = u.day.cgs.scale
MPC_CGS = u.Mpc.cgs.scale
C_CGS = c.c.cgs.value
M_SUN_CGS = c.M_sun.cgs.value
G_CGS = c.G.cgs.value
Jy = u.Jy.cgs.scale
ANG_CGS = u.Angstrom.cgs.scale
pi = np.pi
pc10 = 10 * u.pc.cgs.scale