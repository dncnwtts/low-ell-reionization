import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from classy import Class
import healpy as hp
from scipy.integrate import trapz
import scipy.optimize as op
import emcee
from time import time
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
    if both:
        thermo = cosmo.get_thermodynamics()
        z, xe = thermo['z'], thermo['x_e']
        cls = cosmo.lensed_cl(lmax)
        ell, EE, TE = cls['ell'], cls['ee'], cls['te']
        return z, xe, ell, EE, TE
    elif therm:
        return cosmo.get_thermodynamics()
    elif spectra:
        cls = cosmo.lensed_cl(lmax)
        ell, EE, TE = cls['ell'], cls['ee'], cls['te']
        return ell, EE, TE
    elif history:
        thermo = cosmo.get_thermodynamics()
        z, xe = thermo['z'], thermo['x_e']
        return z, xe
    else:
        return

if __name__ == '__main__':
    seed = 0

    # Define your cosmology (what is not specified will be set to CLASS default parameters)
    
    params['many_tanh_z'] = '3.5,7,28'
    params['many_tanh_xe'] = '-2,-1,0.2'
    
    
    
    
    ell, ee, te = get_spectra(7, 0.2, spectra=True, lmax=lmax)
    plt.loglog(ell, ee)
    np.random.seed(seed)
    eehat = hp.alm2cl(hp.synalm(ee, lmax=lmax))
    plt.loglog(ell, eehat)
    
    def lnprob_EE_ell(zre, x_e, Clhat):
        ell, Cl, TE = get_spectra(zre, x_e, lmax=len(Clhat)-1, spectra=True)
        chi2_ell = (2*ell+1)*(Clhat/Cl + np.log(Cl/Clhat)-1)
        return -chi2_ell#-chi2_exp_ell
    
    chi2_ell = lnprob_EE_ell(7, 0.2, eehat)
    plt.plot(ell[2:], chi2_ell[2:])
    
    def lnprob(args, Clhat):
        zre, x_e = args
        if (zre < 4) | (x_e < 0) | (x_e > 0.5):
            return -np.inf
        return sum(lnprob_EE_ell(zre, x_e, Clhat)[2:])
    
    print(sum(chi2_ell[2:]), lnprob(np.array([7, 0.2]), eehat))
    
    
    nll = lambda *args: -lnprob(*args)
    result = op.minimize(nll, [7, 0.2], args=(eehat))
    print(result)
    
    ndim, nwalkers = 2, 6
    pos = [result['x'] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=([eehat]))
    
    
    nsteps = 50
    t0 = time()
    for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
        if (i+1) % 10 == 0:
            print("{0:5.1%}".format(float(i)/nsteps))
            print(time()-t0)
            t0 = time()
    
    
    nsteps = 750
    print(sampler.chain[:,-1,:])
    t0 = time()
    for i, result in enumerate(sampler.sample(sampler.chain[:,-1,:], iterations=nsteps)):
        if (i+1) % 10 == 0:
            print("{0:5.1%}".format(float(i)/nsteps))
            print(time()-t0)
            t0 = time()
    
    for i in range(nwalkers):
        for j in range(ndim):
            plt.figure(j)
            plt.plot(sampler.chain[i,:,j], color='k')
        plt.figure(ndim+1)
        plt.plot(sampler.lnprobability[i], color='k')
    
    
    plt.figure(2)
    plt.loglog(ell, eehat, '.')
    for i in range(len(sampler.chain[:,-1])):
        ell, ee, te = get_spectra(*sampler.chain[i,-1], spectra=True, lmax=lmax)
        plt.figure(1)
        plt.plot(ell,-lnprob_EE_ell(*sampler.chain[i,-1],eehat), color=plt.cm.viridis(i/6))
        print(sum(lnprob_EE_ell(*sampler.chain[i,-1],eehat)[2:]))
        plt.figure(2)
        plt.loglog(ell[2:], ee[2:], color=plt.cm.viridis(i/6))
        
    plt.figure(1)
    plt.plot(ell, -lnprob_EE_ell(7, 0.2, eehat), color='k')
    print(sum(lnprob_EE_ell(7, 0.2,eehat)[2:]))
    np.save('chain', sampler.chain)
    np.save('lnprob', sampler.lnprobability)
