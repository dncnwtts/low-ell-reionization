import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from classy import Class
import healpy as hp
from scipy.integrate import trapz
import scipy.optimize as op
import emcee
from time import time

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

def lnprob_EE_ell(zre, x_e, Clhat):
    ell, Cl, TE = get_spectra(zre, x_e, lmax=len(Clhat)-1, spectra=True)
    chi2_ell = (2*ell+1)*(Clhat/Cl + np.log(Cl/Clhat)-1)
    return -chi2_ell#-chi2_exp_ell

def get_tau(thermo, zmax=100, xmin=2e-4):
    eta = thermo['conf. time [Mpc]']
    z = thermo['z']
    x_e = thermo['x_e']
    dtaudeta = thermo["kappa' [Mpc^-1]"]
    sigmaTan_p = dtaudeta/x_e
    integrand = -sigmaTan_p*x_e
    return trapz(integrand[(x_e>xmin) & (z<zmax)], x=eta[(x_e>xmin) & (z<zmax)])


def get_twotau(thermo, zmax=100, xmin=2e-4):
    eta = thermo['conf. time [Mpc]']
    z = thermo['z']
    x_e = thermo['x_e']
    dtaudeta = thermo["kappa' [Mpc^-1]"]
    sigmaTan_p = dtaudeta/x_e
    integrand = -sigmaTan_p*x_e
    zre = z[np.where(x_e < 0.5)[0][0]]
    zsplit = 1+zre
    tau_lo = trapz(integrand[(x_e>xmin) & (z<zsplit)], x=eta[(x_e>xmin) & (z<zsplit)])
    tau_hi = trapz(integrand[(x_e>xmin) & (z>zsplit) & (z<zmax)], x=eta[(x_e>xmin) & (z>zsplit) & (z<zmax)])
    return zsplit, tau_lo, tau_hi

def get_spectra(zreio, x_e, history=False, spectra=False, both=False, lmax=40, therm=False):
    params['many_tanh_z'] = '3.5,' + str(zreio) +',28'
    params['many_tanh_xe'] = '-2,-1,'+str(max(x_e, 2e-4))
    cosmo.set(params)
    cosmo.compute()
    thermo = cosmo.get_thermodynamics()
    tau = get_tau(thermo)
    params['A_s'] = 2.3e-9*np.exp(-2*0.06)/np.exp(-2*tau)
    cosmo.set(params)
    cosmo.compute()

    thermo = cosmo.get_thermodynamics()
    cls = cosmo.lensed_cl(lmax)
    cosmo.struct_cleanup()
    if both:
        z, xe = thermo['z'], thermo['x_e']
        cls = cosmo.lensed_cl(lmax)
        ell, EE, TE = cls['ell'], cls['ee'], cls['te']
        return z, xe, ell, EE, TE
    elif therm:
        return thermo
    elif spectra:
        ell, EE, TE = cls['ell'], cls['ee'], cls['te']
        return ell, EE, TE
    elif history:
        z, xe = thermo['z'], thermo['x_e']
        return z, xe
    else:
        return

if __name__ == '__main__':

    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit()
    seed = 2

    params['many_tanh_z'] = '3.5,7,28'
    params['many_tanh_xe'] = '-2,-1,0.2'
    
    
    
    
    ell, ee, te = get_spectra(6, 0.05, spectra=True, lmax=lmax)
    np.random.seed(seed)
    eehat = hp.alm2cl(hp.synalm(ee, lmax=lmax))
    
    
    chi2_ell = lnprob_EE_ell(6, 0.05, eehat)
    
    
    
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
