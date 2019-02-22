import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from modified_histories import get_spectra, get_twotau
import healpy as hp
from time import time

seed = 2
lmax = 100

plt.figure()
ell, ee, te = get_spectra(6, 0.05, spectra=True, lmax=100)
plt.loglog(ell[2:], ee[2:], zorder=5, color='r')
np.random.seed(seed)
eehat = hp.alm2cl(hp.synalm(ee, lmax=lmax))
plt.loglog(ell[2:], eehat[2:], 'o')

chain = np.loadtxt('chain_{0}.dat'.format(seed))[:,1:]
t0 = time()
inds = np.random.randint(low=0,high=len(chain), size=500)
taulo, tauhi, tau = [], [], []
zbla, xebla = [], []
try:
    data = np.loadtxt('auxvars_{0}.dat'.format(seed))
except IOError:
    f = open('auxvars_{0}.dat'.format(seed), 'w')
    f.close()
for i in range(len(inds)):
    print(i, time()-t0, (time()-t0)*len(inds)/60)
    t0 = time()
    therm = get_spectra(*chain[i], therm=True)
    zsplit, tau_lo, tau_hi = get_twotau(therm)
    zbla.append(chain[i][0])
    xebla.append(chain[i][1])
    taulo.append(tau_lo)
    tauhi.append(tau_hi)
    tau.append(tau_lo+tau_hi)
    bla = np.array([chain[i][0], chain[i][1], tau_lo, tau_hi, tau_lo+tau_hi])
    f = open('auxvars.dat', 'a')
    print(np.array2string(bla))
    f.write("{0:4d} {1:s}\n".format(i, np.array2string(bla).strip("[]")))

    ell, ee, te = get_spectra(*chain[i], spectra=True, lmax=100)
    plt.loglog(ell[2:], ee[2:], alpha=0.1, color='k')
            #color=plt.cm.viridis(np.exp(lnp[j]-max(lnp.flatten()))), zorder=-1)
plt.show()
plt.savefig('ps.png')

therm = get_spectra(6, 0.05, therm=True)
zsplit, taulot, tauhit = get_twotau(therm)
taut = taulot + tauhit

plt.figure()
plt.plot(taulo, tauhi, 'o')
plt.axvline(taulot)
plt.axhline(tauhit)
print(taulot, tauhit)
plt.xlabel(r'$\tau_\mathrm{lo}$')
plt.ylabel(r'$\tau_\mathrm{hi}$')
plt.savefig('taus.png')


derived = np.array([taulo, tauhi, tau]).T
corner(derived, truths=[taulot, tauhit, taut, 7, 0.2], \
        labels=[r'$\tau_\mathrm{lo}$', r'$\tau_\mathrm{hi}$', r'$\tau$', r'$z$', r'$x_e$'])
plt.savefig('corner_reparameterized.png')

plt.show()


