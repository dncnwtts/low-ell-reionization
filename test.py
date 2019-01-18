import numpy as np
import matplotlib.pyplot as plt

from scipy.special import polygamma, gamma, kv
seed = 5

from classy import Class

from tools import get_EE, get_TE, get_TT

import healpy as hp

'''
Uses CLASS to plot standard power spectrum, the baseline for our analysis.
'''

def lnprob_EE_ell(tau, taus, EE_arr, Clhat):
    # clhat is the observed spectrum
    # return log(P(Clhat | C_l))
    # Working off of equation (8) of HL08
    i = (np.abs(taus - tau)).argmin()
    ell = np.arange(len(EE_arr[i]))
    Cl = EE_arr[i] # the theory power spectrum that we're comparing to.
    #chi2_ell = (2*ell+1)*(Clhat/Cl + \
    #        np.log(Cl)-(2*ell-1)/(2*ell+1)*np.log(Clhat))
    # If you add an arbitrary constant, you get this;
    chi2_ell = (2*ell+1)*(Clhat/Cl + np.log(Cl/Clhat)-1)
    chi2_exp_ell = (2*ell+1)*(np.log(ell+1/2) - polygamma(0,ell+1/2))
    return chi2_ell#-chi2_exp_ell

def prob_TE_ell(tau, taus, theory_arrs, TEhat):
    c = TEhat
    taui = (np.abs(taus - tau)).argmin()
    TT, TE, EE = theory_arrs
    TTi = TT[taui]
    TEi = TE[taui]
    EEi = EE[taui]
    rho = TEi/np.sqrt(TTi*EEi)
    z = (1-rho**2)*np.sqrt(TTi*EEi)
    ell = np.arange(len(TTi))
    N = 2*ell+1
    num = N**((N+1)/2)*abs(c)**((N-1)/2)*np.exp(N*rho*c/z)*kv((N-1)//2, N*abs(c)/z)
    den = 2**((N-1)/2)*np.sqrt(np.pi)*gamma(N/2)*np.sqrt(z)*(TTi*EEi)**(N/4)
    return num/den

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
    TT_arr = np.loadtxt('tt.txt')
    taus = np.loadtxt('taus.txt')
    num = len(EE_arr)
except IOError:
    #num = 500
    #taus = np.linspace(0.02, 0.1, num)
    taus = np.arange(0.02, 0.1, 0.0001)
    num = len(taus)
    EE_arr = np.zeros((num, lmax+1))
    for i in range(num):
        print(i, num)
        EE_arr[i] = get_EE(tau=taus[i], lmax=lmax)
    
    TE_arr = np.zeros((num, lmax+1))
    for i in range(num):
        print(i, num)
        TE_arr[i] = get_TE(taus[i], lmax=lmax)

    TT_arr = np.zeros((num, lmax+1))
    for i in range(num):
        print(i, num)
        TT_arr[i] = get_TT(taus[i], lmax=lmax)

cinds = np.linspace(0,1,num)
Z = ell*(ell+1)/(2*np.pi)
Z[:2] = 1.
cm = plt.cm.viridis_r
cm = plt.cm.cool


chi2_eff = (ell.max() - 2 + 1)*1*(2)/2 + 1*(2 + 3 - 1)/24*np.log(ell.max()/2)
chi2_eff_ell = (2*ell+1)*(np.log(ell+1/2) - polygamma(0,ell+1/2))
chi2_eff = sum(chi2_eff_ell[2:])
varchi2_ell = (2*ell+1)*((2*ell+1)*polygamma(1,ell+1/2) -2)
varchi2 = sum(varchi2_ell[2:])

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
np.savetxt('tt.txt', TT_arr)
np.savetxt('taus.txt', taus)

plt.figure()

np.random.seed(seed)
f = (np.abs(taus - 0.06)).argmin()
Clhat = hp.alm2cl(hp.synalm(EE_arr[f]))
taus = np.linspace(0.03, 0.09, 21)
for i in range(len(taus)):
    chi2 = lnprob_EE_ell(taus[i], taus, EE_arr, Clhat)
    plt.semilogx(ell, chi2, color=plt.cm.coolwarm(i/30))
    print(taus[i], sum(chi2[2:]))


taus = np.loadtxt('taus.txt')
f = (np.abs(taus - 0.06)).argmin()
plt.figure()
chi2s = []
np.random.seed(seed)
Clhat = hp.alm2cl(hp.synalm(EE_arr[f]))
for i in range(len(EE_arr)):
    plt.loglog(ell, EE_arr[i], color=plt.cm.coolwarm(i/len(EE_arr)))
    chi2 = lnprob_EE_ell(taus[i], taus, EE_arr, Clhat)
    chi2s.append(sum(chi2[2:]))
plt.plot(ell, Clhat, 'k')
plt.figure()
chi2s = np.array(chi2s)
plt.plot(taus, chi2s)
chi2_exp_ell = (2*ell+1)*(np.log(ell+1/2) - polygamma(0,ell+1/2))
#plt.axhline(sum(chi2_exp_ell[2:]))
plt.fill_between(taus, chi2_eff-varchi2**0.5, chi2_eff+varchi2**0.5)
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\chi^2$')
print(taus[np.argmin(chi2s)], chi2s.min())

plt.figure()

plt.plot(taus, np.exp(-(chi2s-chi2s.min())/2))
plt.axvline(0.06)
plt.xlabel(r'$\tau$')
plt.ylabel(r'$e^{-(\chi^2-\chi^2_\mathrm{min})}$')

L = np.exp(-(chi2s-chi2s.min())/2)
mu = sum(taus*L)/sum(L)
var = sum(taus**2*L)/sum(L) - mu**2
plt.title(r'$\hat\tau={0}\pm{1}$'.format(np.round(mu,4), np.round(var**0.5,4)))

chi2_exp_ell = (2*ell+1)*(np.log(ell+1/2) - polygamma(0,ell+1/2))
#plt.figure('curves')
#for i in range(len(EE_arr)):
#    plt.loglog(ell, EE_arr[i], color=plt.cm.coolwarm(i/len(EE_arr)))

tauhats = []
minchis = []

for s in range(100):
    chi2s = []
    np.random.seed(s)
    Clhat = hp.alm2cl(hp.synalm(EE_arr[f]))
    #plt.figure('curves')
    for i in range(len(EE_arr)):
        chi2 = lnprob_EE_ell(taus[i], taus, EE_arr, Clhat)
        chi2s.append(sum(chi2[2:]))
    chi2_exp = sum(chi2_exp_ell[2:])
    #cind = np.exp((min(chi2s) - chi2_exp)/2)
    #cind = min(cind, 1)
    #print(taus[np.argmin(chi2s)], cind, min(chi2s), chi2_exp)
    #cind = deltachi2/chi2_exp # between 0 to + infty
    #plt.plot(ell, Clhat, color=plt.cm.viridis(cind))
    chi2s = np.array(chi2s)
    #plt.figure('chi2s')
    #plt.plot(taus, chi2s, color=plt.cm.viridis(cind))
    
    #plt.figure('likelihood')
    
    #plt.plot(taus, np.exp(-(chi2s-chi2s.min())/2), color=plt.cm.viridis(cind),
    #        zorder=cind)
    #plt.axvline(0.06)
    tauhats.append(taus[np.argmin(chi2s)])
    minchis.append(min(chi2s))


plt.figure()
plt.hist(tauhats, 20)
plt.xlabel(r'$\hat\tau$')
print(np.mean(tauhats), np.std(tauhats))
plt.title(r'$\hat\tau={0}\pm{1}$'.format(np.round(np.mean(tauhats),4), np.round(np.std(tauhats),4)))


plt.figure()
plt.hist(minchis, 20)
plt.axvline(chi2_eff)
plt.axvline(chi2_eff - varchi2**0.5)
plt.axvline(chi2_eff + varchi2**0.5)
plt.xlabel(r'$\chi^2$')
print(np.mean(minchis), np.std(minchis))



'''
I more or less have the TT stuff figured out. The TE stuff mostly, I want to
check that I get the right answer, that I have a good "goodness-of-fit"
parameter to compare against theory.
'''

theory_arrs = np.array([TT_arr, TE_arr, EE_arr])
clth = np.array([TT_arr[f], EE_arr[f], EE_arr[f]*0, TE_arr[f]])
np.random.seed(seed)
Clhat = hp.alm2cl(hp.synalm(clth, new=True))
TEhat = Clhat[3]

taus = np.linspace(0.03, 0.09, 21)
f = (np.abs(taus - 0.06)).argmin()
clth = np.array([TT_arr[f], EE_arr[f], EE_arr[f]*0, TE_arr[f]])
np.random.seed(seed)
Clhat = hp.alm2cl(hp.synalm(clth, new=True))
TEhat = Clhat[3]
for i in range(len(taus)):
    chi2 = -2*np.log(prob_TE_ell(taus[i], taus, theory_arrs, TEhat))
    plt.semilogx(ell, chi2, color=plt.cm.coolwarm(i/21))
    print(taus[i], sum(chi2[2:]))

taus = np.loadtxt('taus.txt')
f = (np.abs(taus - 0.06)).argmin()
clth = np.array([TT_arr[f], EE_arr[f], EE_arr[f]*0, TE_arr[f]])
np.random.seed(seed)
Clhat = hp.alm2cl(hp.synalm(clth, new=True))
TEhat = Clhat[3]
chi2s = []
plt.figure()
plt.loglog(Clhat[0])
plt.loglog(Clhat[1])
plt.loglog(Clhat[3])

plt.loglog(clth[0],color='k')
plt.loglog(clth[1],color='k')
plt.loglog(clth[3],color='k')


plt.figure()
for i in range(len(TE_arr)):
    plt.loglog(ell, TE_arr[i], color=plt.cm.coolwarm(i/len(EE_arr)))
    plt.loglog(ell, TT_arr[i], color=plt.cm.coolwarm(i/len(EE_arr)))
    plt.loglog(ell, TE_arr[i], color=plt.cm.coolwarm(i/len(EE_arr)))
    chi2 = -2*np.log(prob_TE_ell(taus[i], taus, theory_arrs, TEhat))
    chi2s.append(sum(chi2[2:]))
plt.plot(ell, TEhat, 'k')
plt.figure()
chi2s = np.array(chi2s)
plt.plot(taus, chi2s)
plt.xlabel(r'$\tau$')
plt.ylabel(r'$-2\ln\mathcal L$')
print(taus[np.argmin(chi2s)], chi2s.min())

plt.figure()

plt.plot(taus, np.exp(-(chi2s-chi2s.min())/2))
plt.axvline(0.06)
plt.xlabel(r'$\tau$')
plt.ylabel(r'$e^{-(\chi^2-\chi^2_\mathrm{min})}$')

L = np.exp(-(chi2s-chi2s.min())/2)
mu = sum(taus*L)/sum(L)
var = sum(taus**2*L)/sum(L) - mu**2
plt.title(r'$\hat\tau={0}\pm{1}$'.format(np.round(mu,4), np.round(var**0.5,4)))

tauhats = []
minchis = []

for s in range(100):
    print(s)
    chi2s = []
    np.random.seed(s)
    Clhat = hp.alm2cl(hp.synalm(clth, new=True))
    TEhat = Clhat[3]
    #plt.figure('curves')
    for i in range(len(EE_arr)):
        chi2 = -2*np.log(prob_TE_ell(taus[i], taus, theory_arrs, TEhat))
        chi2s.append(sum(chi2[2:]))
    chi2s = np.array(chi2s)
    tauhats.append(taus[np.argmin(chi2s)])
    minchis.append(min(chi2s))


plt.figure()
plt.hist(tauhats, 20)
plt.xlabel(r'$\hat\tau$')
print(np.mean(tauhats), np.std(tauhats))
plt.title(r'$\hat\tau={0}\pm{1}$'.format(np.round(np.mean(tauhats),4), np.round(np.std(tauhats),4)))

minchis = np.array(minchis)
minchis = minchis[np.isfinite(minchis)]
plt.figure()
plt.hist(minchis, 20)
plt.xlabel(r'$-2\ln\mathcal L$')
print(np.mean(minchis), np.std(minchis))
plt.close('all')
plt.show()



'''
Simultaneous versus joint fits.
'''
taus = np.loadtxt('taus.txt')
f = (np.abs(taus - 0.06)).argmin()
clth = np.array([TT_arr[f], EE_arr[f], EE_arr[f]*0, TE_arr[f]])
np.random.seed(seed)
Clhat = hp.alm2cl(hp.synalm(clth, new=True))
TEhat = Clhat[3]
EEhat = Clhat[1]
plt.figure()
plt.loglog(Clhat[0])
plt.loglog(Clhat[1])
plt.loglog(Clhat[3])

plt.loglog(clth[0],color='k')
plt.loglog(clth[1],color='k')
plt.loglog(clth[3],color='k')


plt.figure()
chi2TEs = []
chi2EEs = []
for i in range(len(TE_arr)):
    plt.loglog(ell, TE_arr[i], color=plt.cm.coolwarm(i/len(EE_arr)))
    plt.loglog(ell, TT_arr[i], color=plt.cm.coolwarm(i/len(EE_arr)))
    plt.loglog(ell, EE_arr[i], color=plt.cm.coolwarm(i/len(EE_arr)))
    chi2TE = -2*np.log(prob_TE_ell(taus[i], taus, theory_arrs, TEhat))
    chi2EE = lnprob_EE_ell(taus[i], taus, EE_arr, EEhat)
    chi2TEs.append(sum(chi2TE[2:]))
    chi2EEs.append(sum(chi2EE[2:]))
plt.plot(ell, TEhat, 'k')
plt.plot(ell, EEhat, 'k')
plt.plot(ell, Clhat[0], 'k')
#plt.show()

plt.figure()
chi2EEs = np.array(chi2EEs)
chi2TEs = np.array(chi2TEs)
chi2s = chi2EEs + chi2TEs
plt.plot(taus, chi2EEs - chi2EEs.min())
plt.plot(taus, chi2TEs - chi2TEs.min())
plt.plot(taus, chi2EEs+chi2TEs - chi2EEs.min()-chi2TEs.min())
plt.xlabel(r'$\tau$')
plt.ylabel(r'$-2\ln\mathcal L$')
#plt.show()


plt.figure()

plt.axvline(0.06)
plt.xlabel(r'$\tau$')
plt.ylabel(r'$e^{-(\chi^2-\chi^2_\mathrm{min})}$')
#plt.show()

L = np.exp(-(chi2s-chi2s.min())/2)
inds = np.isfinite(L)
mu = sum(taus[inds]*L[inds])/sum(L[inds])
var = sum(taus[inds]**2*L[inds])/sum(L[inds]) - mu**2
print(mu, var**0.5, 'All')
#plt.title(r'$\hat\tau={0}\pm{1}$'.format(np.round(mu,4), np.round(var**0.5,4)))
plt.plot(taus, np.exp(-(chi2s-chi2s.min())/2),
        label=r'${0}\pm{1}$'.format(np.round(mu,5),np.round(var**0.5,5)))

L = np.exp(-(chi2EEs-chi2EEs.min())/2)
inds = np.isfinite(L)
mu = sum(taus[inds]*L[inds])/sum(L[inds])
var = sum(taus[inds]**2*L[inds])/sum(L[inds]) - mu**2
print(mu, var**0.5, 'EE')
plt.plot(taus, np.exp(-(chi2EEs-chi2EEs.min())/2),
        label=r'${0}\pm{1}$'.format(np.round(mu,5),np.round(var**0.5,5)))

L = np.exp(-(chi2TEs-chi2TEs.min())/2)
inds = np.isfinite(L)
mu = sum(taus[inds]*L[inds])/sum(L[inds])
var = sum(taus[inds]**2*L[inds])/sum(L[inds]) - mu**2
print(mu, var**0.5, 'TE')
plt.plot(taus, np.exp(-(chi2TEs-chi2TEs.min())/2),
        label=r'${0}\pm{1}$'.format(np.round(mu,5),np.round(var**0.5,5)))
plt.legend(loc='best')
plt.show()


tauhats = []
minchis = []
tauhatsEE = []
minchisEE = []
tauhatsTE = []
minchisTE = []

for s in range(100):
    chi2EEs = []
    chi2TEs = []
    np.random.seed(s)
    Clhat = hp.alm2cl(hp.synalm(clth, new=True))
    TEhat = Clhat[3]
    EEhat = Clhat[1]
    plt.figure('curves')
    plt.loglog(Clhat[1], color='C0', alpha=0.1)
    plt.loglog(Clhat[3], color='C1', alpha=0.1)
    for i in range(len(EE_arr)):
        chi2TE = -2*np.log(prob_TE_ell(taus[i], taus, theory_arrs, TEhat))
        chi2TEs.append(sum(chi2TE[2:]))
        chi2EE = lnprob_EE_ell(taus[i], taus, theory_arrs[2], EEhat)
        chi2EEs.append(sum(chi2EE[2:]))
    chi2EEs = np.array(chi2EEs)
    chi2TEs = np.array(chi2TEs)
    chi2s = chi2EEs + chi2TEs
    tauhatsEE.append(taus[np.argmin(chi2EEs)])
    minchisEE.append(min(chi2EEs))
    tauhatsTE.append(taus[np.argmin(chi2TEs)])
    minchisTE.append(min(chi2TEs))
    tauhats.append(taus[np.argmin(chi2s)])
    minchis.append(min(chi2s))


plt.figure()
bins = np.linspace(0, 0.1, 25)
plt.xlabel(r'$\hat\tau$')
print(np.mean(tauhats), np.std(tauhats))
plt.title(r'$\hat\tau={0}\pm{1}$'.format(np.round(np.mean(tauhats),4), np.round(np.std(tauhats),4)))
plt.hist(tauhatsTE, bins)
plt.hist(tauhatsEE, bins)
plt.hist(tauhats, bins)

minchis = np.array(minchis)
minchis = minchis[np.isfinite(minchis)]
plt.figure()
plt.hist(minchis, 20)
plt.xlabel(r'$-2\ln\mathcal L$')
print(np.mean(minchis), np.std(minchis))
#plt.close('all')
plt.show()
