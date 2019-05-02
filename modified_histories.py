import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from classy import Class
import healpy as hp
from scipy.integrate import trapz
import scipy.optimize as op
import emcee
from time import time

from tools import get_spectra, get_tau, get_twotau, lnprob_EE_ell

import sys
from schwimmbad import MPIPool


lmax = 100
params = {
    'output': 'tCl pCl lCl',
    'l_max_scalars': lmax,
    'lensing': 'yes',
    'A_s': 2.3e-9,
    'n_s': 0.965}
params['reio_parametrization'] ='reio_many_tanh'
params['many_tanh_num'] = 3
params['many_tanh_width'] = 0.5
cosmo = Class()

def lnprob(args, Clhat):
    zre, x_e = args
    if (zre < 4) | (x_e < 0) | (x_e > 0.5):
        return -np.inf
    return sum(lnprob_EE_ell(zre, x_e, Clhat)[2:])

#def lnprob_EE_ell(zre, x_e, Clhat):
#    ell, Cl, TE = get_spectra(zre, x_e, lmax=len(Clhat)-1, spectra=True)
#    chi2_ell = (2*ell+1)*(Clhat/Cl + np.log(Cl/Clhat)-1)
#    return -chi2_ell#-chi2_exp_ell


if __name__ == '__main__':

    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit()
    seed = 2

    params['many_tanh_z'] = '3.5,7,28'
    params['many_tanh_xe'] = '-2,-1,0.2'
    
    
    
    
    ell, ee, te = get_spectra(6, 0.00, spectra=True, lmax=lmax)
    np.random.seed(seed)
    eehat = hp.alm2cl(hp.synalm(ee, lmax=lmax))
    
    
    chi2_ell = lnprob_EE_ell(6, 0.00, eehat)
    
    
    
    ndim, nwalkers = 2, 24
    
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=([eehat]),
            pool=pool)
    
    # roughly 20 cpu-seconds/step
    try:
        data = np.loadtxt('chain_{0}.dat'.format(seed))
        pos = data[-nwalkers:,1:]
        lnprob = np.loadtxt('lnprob_{0}.dat'.format(seed))
    except IOError:
        nll = lambda *args: -lnprob(*args)
        result = op.minimize(nll, [7, 0.2], args=(eehat))
        pos = [result['x'] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

    nsteps = 1000
    t0 = time()
    for i, result in enumerate(sampler.sample(pos, iterations=nsteps, storechain=False)):
        if (i+1) % 10 == 0:
            print("{0:5.1%}".format(float(i)/nsteps))
            print(time()-t0)
            t0 = time()
        position = result[0]
        f = open('chain_{0}.dat'.format(seed), 'a')
        for k in range(position.shape[0]):
            f.write("{0:4d} {1:s}\n".format(k, np.array2string(position[k]).strip("[]").replace('\n','')))
        f.close()
    
    np.save('chain_{0}'.format(seed), sampler.chain)
    np.save('lnprob_{0}'.format(seed), sampler.lnprobability)
