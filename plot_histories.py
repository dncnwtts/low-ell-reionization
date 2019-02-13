import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from modified_histories import get_spectra, get_twotau
import healpy as hp

data = np.load('chain.npy')
nwalkers, nsteps, ndim = data.shape

burn = 100

for i in range(ndim):
    plt.subplot(3,1,i+1)
    for j in range(nwalkers):
        plt.plot(data[j,:,i], color=plt.cm.viridis(j/nwalkers))

plt.subplot(3,1,3)
lnprob = np.load('lnprob.npy')
for j in range(nwalkers):
    plt.plot(np.exp(lnprob[j]-max(lnprob.flatten())), color=plt.cm.viridis(j/nwalkers))

corner(data[:,100:,:].reshape(-1, ndim), labels=[r'$z_\mathrm{re}$',\
    r'$x_e(\mathrm{high-}z)$'], truths=[7, 0.2], show_titles=True, quantiles=[0.5])

seed = 0
lmax = 100

plt.figure()
ell, ee, te = get_spectra(7, 0.2, spectra=True, lmax=100)
plt.loglog(ell[2:], ee[2:], zorder=5, color='r')
np.random.seed(seed)
eehat = hp.alm2cl(hp.synalm(ee, lmax=lmax))
plt.loglog(ell[2:], eehat[2:], 'o')

chain = data[:,100:,:].reshape(-1,ndim)
lnp = lnprob[:,100:].reshape(-1)

inds = np.random.randint(low=0,high=len(chain), size=1000)
taulo, tauhi, tau = [], [], []
zbla, xebla = [], []
for i in range(len(inds)):
    print(i)
    therm = get_spectra(*chain[i], therm=True)
    zsplit, tau_lo, tau_hi = get_twotau(therm)
    zbla.append(chain[i][0])
    xebla.append(chain[i][1])
    taulo.append(tau_lo)
    tauhi.append(tau_hi)
    tau.append(tau_lo+tau_hi)
    ell, ee, te = get_spectra(*chain[i], spectra=True, lmax=100)
    plt.loglog(ell[2:], ee[2:], alpha=0.1, color='k')
            #color=plt.cm.viridis(np.exp(lnp[j]-max(lnp.flatten()))), zorder=-1)
plt.show()

therm = get_spectra(7, 0.2, therm=True)
zsplit, taulot, tauhit = get_twotau(therm)
taut = taulot + tauhit

plt.figure()
plt.plot(taulo, tauhi, 'o')
plt.axvline(taulot)
plt.axhline(tauhit)
print(taulot, tauhit)
plt.xlabel(r'$\tau_\mathrm{lo}$')
plt.ylabel(r'$\tau_\mathrm{hi}$')


derived = np.array([taulo, tauhi, tau]).T
corner(derived, truths=[taulot, tauhit, taut, 7, 0.2], \
        labels=[r'$\tau_\mathrm{lo}$', r'$\tau_\mathrm{hi}$', r'$\tau$', r'$z$', r'$x_e$'])

plt.show()


