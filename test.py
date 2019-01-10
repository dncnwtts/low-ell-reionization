import numpy as np
import matplotlib.pyplot as plt

from classy import Class

from tools import get_EE, get_TE

import healpy as hp

'''
Uses CLASS to plot standard power spectrum, the baseline for our analysis.
'''

def lnprob_EE_ell(tau, taus, EE_arr, tau0=0.06):
    # return log(P(Clhat | C_l))
    # Working off of equation (8) of HL08
    i = (np.abs(taus - tau)).argmin()
    f = (np.abs(taus - tau0)).argmin()
    ell = np.arange(len(EE_arr[i]))
    Clhat = EE_arr[f] # the observed, "true" power spectrum. Might be worth
                      # actually doing a "real" PS.
    np.random.seed(5)
    Clhat = hp.alm2cl(hp.synalm(EE_arr[f]))
    Cl = EE_arr[i] # the theory power spectrum that we're comparing to.
    chi2_ell = (2*ell+1)*(Clhat/Cl + \
            np.log(Cl)-(2*ell-1)/(2*ell+1)*np.log(Clhat))
    chi2_0 = (2*ell+1)*(Clhat/Clhat + \
            np.log(Clhat)-(2*ell-1)/(2*ell+1)*np.log(Clhat))
    return chi2_ell# - chi2_0

# Define your cosmology (what is not specified will be set to CLASS default parameters)
params = {
    'output': 'tCl pCl lCl',
    'l_max_scalars': 2500,
    'lensing': 'yes',
    'A_s': 2.3e-9,
    'n_s': 0.965,
    'tau_reio':0.06}

# Create an instance of the CLASS wrapper
cosmo = Class()

# Set the parameters to the cosmological code
cosmo.set(params)

# Run the whole code. Depending on your output, it will call the
# CLASS modules more or less fast. For instance, without any
# output asked, CLASS will only compute background quantities,
# thus running almost instantaneously.
# This is equivalent to the beginning of the `main` routine of CLASS,
# with all the struct_init() methods called.
cosmo.compute()

# Access the lensed cl until l=2500
cls = cosmo.lensed_cl(2500)

# Clean CLASS (the equivalent of the struct_free() in the `main`
# of CLASS. This step is primordial when running in a loop over different
# cosmologies, as you will saturate your memory very fast if you ommit
# it.
cosmo.struct_cleanup()

# If you want to change completely the cosmology, you should also
# clean the arguments, otherwise, if you are simply running on a loop
# of different values for the same parameters, this step is not needed
cosmo.empty()
Z = cls['ell']*(cls['ell']+1)/(2*np.pi)*(cosmo.T_cmb()*1e6)**2
plt.loglog(cls['ell'][2:], cls['ee'][2:]*Z[2:])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$D_\ell\ \mathrm{[\mu K^2]}$')


plt.figure()
lmax = 100
ell = np.arange(lmax+1)
plt.loglog(ell[2:], get_EE(tau=0.06, lmax=lmax)[2:])
plt.loglog(ell[2:], get_TE(0.06, lmax=lmax)[2:])
plt.loglog(ell[2:], -get_TE(0.06, lmax=lmax)[2:], 'C1--')




plt.figure()
try:
    EE_arr = np.loadtxt('ee.txt')
    TE_arr = np.loadtxt('te.txt')
    taus = np.loadtxt('taus.txt')
    num = len(EE_arr)
except IOError:
    num = 500
    taus = np.linspace(0.02, 0.1, num)
    EE_arr = np.zeros((num, lmax+1))
    for i in range(num):
        EE_arr[i] = get_EE(tau=taus[i], lmax=lmax)
    
    TE_arr = np.zeros((num, lmax+1))
    for i in range(num):
        TE_arr[i] = get_TE(taus[i], lmax=lmax)

cinds = np.linspace(0,1,num)
Z = ell*(ell+1)/(2*np.pi)
Z[:2] = 1.
cm = plt.cm.viridis_r
cm = plt.cm.cool

for i in range(num):
    plt.loglog(ell[2:], (EE_arr[i])[2:], color=cm(cinds[i]))
sm = plt.cm.ScalarMappable(cmap=cm,
        norm=plt.Normalize(vmin=taus[0], vmax=taus[-1]))
sm._A = []
#cbaxes = fig.add_axes([1, 0.15, 0.03, 0.7])
cbar = plt.colorbar(mappable=sm, label=r'$\tau$, $A_s e^{-2\tau}$ fixed',
        orientation='vertical', ticklocation='right')

i = num //2
print(taus[i])
sigma = np.sqrt(2/(2*ell+1))*EE_arr[i]
plt.fill_between(ell[2:], (EE_arr[i]-sigma)[2:], (EE_arr[i]+sigma)[2:],
        alpha=1, color='k',zorder=-1)

for i in range(num):
    plt.loglog(ell[2:], (TE_arr[i])[2:], color=cm(cinds[i]))
i = num //2
print(taus[i])
sigma = np.sqrt(2/(2*ell+1))*TE_arr[i]
plt.fill_between(ell[2:], (TE_arr[i]-sigma)[2:], (TE_arr[i]+sigma)[2:],
        alpha=1, color='k',zorder=-1)


plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^\mathrm{EE}$ [$\mathrm{\mu K^2}$]')
plt.savefig('cv_v_theoryv.pdf', bbox_inches='tight')

np.savetxt('ee.txt', EE_arr)
np.savetxt('te.txt', TE_arr)
np.savetxt('taus.txt', taus)

plt.figure()

taus = np.linspace(0.03, 0.09, 21)
for i in range(len(taus)):
    chi2 = lnprob_EE_ell(taus[i], taus, EE_arr, tau0=0.06)
    plt.semilogx(ell, chi2, color=plt.cm.coolwarm(i/30))
    print(taus[i], sum(chi2[2:]))

taus = np.loadtxt('taus.txt')
f = (np.abs(taus - 0.06)).argmin()
plt.figure()
np.random.seed(0)
chi2s = []
Clhat = hp.alm2cl(hp.synalm(EE_arr[f]))
for i in range(len(EE_arr)):
    plt.loglog(ell, EE_arr[i], color=plt.cm.coolwarm(i/len(EE_arr)))
    chi2 = lnprob_EE_ell(taus[i], taus, EE_arr, tau0=0.06)
    chi2s.append(sum(chi2[2:]))
plt.plot(ell, Clhat, 'k')
plt.figure()
chi2s = np.array(chi2s)
plt.plot(taus, chi2s)
print(taus[np.argmin(chi2s)], chi2s.min())

plt.figure()

plt.plot(taus, np.exp(-(chi2s-chi2s.min())/2))
plt.axvline(0.06)



plt.show()
