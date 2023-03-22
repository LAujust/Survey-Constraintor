'''
script foer Survey_constraintor_demo.ipynb
'''
print('---?----')
from Survey_constraintor import *
from Survey_constraintor import Survey_constraintor as SSR
import simsurvey_tools as sst
import argparse


ztf_fields = sst.load_ztf_fields()

#event_name = 'S200105ae'
event_name = 'S200213t'

#with open('/home/Aujust/data/Kilonova/Constraint/data/ZTF_{}_survey.pkl'.format(event_name),'rb') as f2:
#    ztf_survey = pickle.load(f2)
#f2.close()

plan_dir = '/home/Aujust/data/Kilonova/Constraint/plans/{}_plan.pkl'.format(event_name)
# with open('/home/Aujust/data/Kilonova/Constraint/plans/{}_plan.pkl'.format(event_name),'rb') as handle:
#     plan = pickle.load(handle)
# handle.close()

import sys
sys.path.append('/home/Aujust/data/Kilonova/GPR')
from Knust import Knust
import kilonovanet

model_bns = kilonovanet.Model('/home/Aujust/data/Kilonova/possis/data/metadata_bulla_bns.json',
                 '/home/Aujust/data/Kilonova/possis/models/bulla-bns-latent-20-hidden-1000-CV-4-2021-04-21-epoch-200.pt',
                 filter_library_path='/home/Aujust/data/Kilonova/possis/data/filter_data')

svd_path = '/home/Aujust/data/Kilonova/GPR/NN/'
model_name = 'Bulla_3comp_spectra'
spec_model = Knust(model_type='tensorflow',model_dir=svd_path,model_name=model_name)

with open('/home/Aujust/data/Kilonova/WFST/WFST_Field.pkl','rb') as f:
    wfst_field = pickle.load(f)
f.close()

with open('/home/Aujust/data/Kilonova/WFST/sim_survey/survey_file.pkl','rb') as f2:
    wfst_survey = pickle.load(f2)
f2.close()

skymap_file = '/home/Aujust/data/Kilonova/Constraint/Skymaps/BNS/LALInference_{}.fits'.format(event_name)

#Generate a survey plan / load plan
kws = dict(width=2.6,height=2.6)  #WFST
ssr = SSR()
#ssr.generate_plan(survey_file=ztf_survey,field_file=ztf_fields)
ssr.load_plan(plan_dir)

#Set models we used
Models = dict(lc_model=BNS_model,kn_model=spec_model,cosmo_model=AngularTimeSeriesSource,random_p=random_parameters_ang,model_dim=4)
ssr.set_models(**Models)
print('Models setted')

#--------# POSSIS BNS
M_dyn_list = np.linspace(0.001,0.02,10)
M_pm_list = np.linspace(0.01,0.13,10)
phi_list = np.linspace(0,90,10)
M_dyn_, M_pm_, phi_= np.meshgrid(M_dyn_list,M_pm_list,phi_list)

M_dyn_flat = M_dyn_.flatten()
M_pm_flat = M_pm_.flatten()
phi_flat = phi_.flatten()
param_flat = np.array([M_dyn_flat,M_pm_flat,phi_flat]).T
#---------# BNS_Model
Mc_list = np.linspace(1,2,10)
q_list = np.linspace(0.6,1,10)
lambda_list = np.linspace(110,500,10)
Mtov_list = np.linspace(2,2.3,10)
Mc_,q_,lambda_,Mtov_ = np.meshgrid(Mc_list,q_list,lambda_list,Mtov_list)

Mc_flat = Mc_.flatten()
q_flat = q_.flatten()
lambda_flat = lambda_.flatten()
Mtov_flat = Mtov_.flatten()
eta_flat = np.array([0.3 for i in range(len(Mc_flat))]).T
param_bns_flat = np.array([Mc_flat,q_flat,lambda_flat,Mtov_flat,eta_flat]).T

#---------# POSSIS BHNS
M_dyn_list = np.linspace(0.02,0.09,10)
M_pm_list = np.linspace(0.02,0.09,10)
M_dyn_, M_pm_= np.meshgrid(M_dyn_list,M_pm_list)

M_dyn_flat = M_dyn_.flatten()
M_pm_flat = M_pm_.flatten()
param_bhns_flat = np.array([M_dyn_flat,M_pm_flat]).T



def cal_effcy(param_):
        #transientprop
        out = ssr.lc_model(param_,model=ssr.kn_model,phase=np.linspace(0,7,100))
        if ssr.model_dim == 3:
            phase, wave, flux = out
            source = ssr.cosmo_model(phase, wave, flux)
        elif ssr.model_dim == 4:
            phase, wave, cos_theta, flux = out
            source = ssr.cosmo_model(phase, wave, cos_theta, flux)
        else:
            raise IndexError('Check model you use.')
        model = sncosmo.Model(source=source,effects=[ssr.dust, ssr.dust], effect_names=['host', 'MW'], effect_frames=['rest', 'obs'])
        transientprop = dict(lcmodel=model, lcsimul_func=ssr.random_p)

        Kwargs = copy.deepcopy(ssr.kwargs)
        Kwargs['transientprop'] = transientprop
        lc_kwgs = {'progress_bar':True}
        survey = ssr.generate_transient(ssr.ntransient,ssr.rate,**Kwargs)
        lcs = survey.get_lightcurves(**lc_kwgs)
        efficy = len(lcs.lcs)/len(lcs.meta_full)
        output = np.concatenate((param_,[efficy]))
        print(efficy)
        return output


#MJD range
mjd_dict = dict(S200105ae=58853.683634,S200115j=58863.182754,
                S190425z=58598.346134259256,S190901ap=58727.98013888889,
                S190910h=58736.354409722226,S200213t=58892.174363425926)
dL_dict = dict(S200105ae=(283,74),S200115j=(340,79),S190425z=(156,41),S190901ap=(241,79),
               S190910h=(230,88),S200213t=(201,80))
mjd_start = mjd_dict[event_name]
dL,ddL = dL_dict[event_name]

mjd_range = (mjd_start-0.1,mjd_start+0.1)
event_kwargs = dict(dL=dL,ddL=ddL,mjd_range=mjd_range)
ssr.set_kwargs(ntransient=3000,skymap_file=skymap_file,**event_kwargs)

print('-----EVENT: {}------'.format(event_name))

with Pool(60) as pool:
    result = pool.map(cal_effcy,param_bns_flat)
    pool.close()
    pool.join()

#ssr.get_effcy_map(param_flat=param_flat,ntransient=1000,**event_kwargs)
#result = ssr.result

with open('/home/Aujust/data/Kilonova/Constraint/ZTF_{}_bns_efcymap.pkl'.format(event_name),'wb') as f:
    pickle.dump(result,f)
f.close()


print('Done')