import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import subprocess

from classy import Class
from scipy.stats import gmean


# Define your cosmology (what is not specified will be set to CLASS default parameters)
params = {
    'output': 'tCl pCl lCl',
    'l_max_scalars': 2500,
    'lensing': 'yes',
    'A_s': 2.3e-9,
    'n_s': 0.965,
    'tau_reio':0.06,
    'r': 0.,
    'modes': 't, s'}

cosmo = Class()
cosmo.set(params)
cosmo.compute()
cls = cosmo.lensed_cl(2500)
cosmo.struct_cleanup()
cosmo.empty()

def get_TE(tau, lmax=100, rescale=True):
    params['tau_reio'] = tau
    if rescale:
        amp = 2.3e-9*np.exp(-2*0.06)
        As = amp/np.exp(-2*tau)
        #As = 2.3e-9
        params['A_s' ] = As

    cosmo.set(params)
    cosmo.compute()
    powers = cosmo.lensed_cl(lmax=lmax)
    Z = np.ones_like(powers['ell'])*(cosmo.T_cmb()*1e6)**2
    return powers['te'][:lmax+1]*Z[:lmax+1]

def get_TT(tau, lmax=100, rescale=True):
    params['tau_reio'] = tau
    if rescale:
        amp = 2.3e-9*np.exp(-2*0.06)
        As = amp/np.exp(-2*tau)
        #As = 2.3e-9
        params['A_s' ] = As

    cosmo.set(params)
    cosmo.compute()
    powers = cosmo.lensed_cl(lmax=lmax)
    Z = np.ones_like(powers['ell'])*(cosmo.T_cmb()*1e6)**2
    return powers['tt'][:lmax+1]*Z[:lmax+1]

def get_EE(r=0.01, tau=0.06, lmax=100, rescale=False):
    params['tau_reio'] = tau
    if rescale:
        amp = 2.3e-9*np.exp(-2*0.06)
        As = amp/np.exp(-2*tau)
        #As = 2.3e-9
        params['A_s' ] = As
    params['r'] = r

    cosmo.set(params)
    cosmo.compute()
    powers = cosmo.lensed_cl(lmax=lmax)
    Z = np.ones_like(powers['ell'])*(1e6*cosmo.T_cmb())**2
    return powers['ee'][:lmax+1]*Z[:lmax+1]


def get_BB(r=0.01, tau=0.06, lmax=100, rescale=False):
    params['tau_reio'] = tau
    if rescale:
        amp = 2.3e-9*np.exp(-2*0.06)
        As = amp/np.exp(-2*tau)
        #As = 2.3e-9
        params['A_s' ] = As
    params['r'] = r

    cosmo.set(params)
    cosmo.compute()
    powers = cosmo.lensed_cl(lmax=lmax)
    #Z = powers['ell']*(powers['ell']+1)/(2*np.pi)*(cosmo.T_cmb())**2
    Z = np.ones_like(powers['ell'])*(1e6*cosmo.T_cmb())**2
    return powers['bb'][:lmax+1]*Z[:lmax+1]


def test_BB():
    for r in np.arange(0, 0.24, 0.04):
        BB = get_BB(r=r, lmax=2000)
        ell = np.arange(len(BB))
        Z = ell*(ell+1)/(2*np.pi)
        plt.loglog(ell[2:], (BB*Z)[2:], label=r'$r={0}$'.format(r))
    plt.xlim([2, 2000])
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_\ell/(2\pi)\,[\mathrm{\mu K^2}]$')
    plt.legend(loc='best')
    plt.show()
    return




def med_subtract(map_orig):
    nside = hp.npix2nside(len(map_orig))
    pixinds = np.arange(len(map_orig))
    theta, phi = hp.pix2ang(nside, pixinds)
    theta_unique = np.unique(theta)
    med_map = np.zeros_like(map_orig)
    for theta_i in theta_unique:
        inds = np.where(theta == theta_i)
        med = np.median(map_orig[inds])
        med_map[inds] = med

    med_subtract = map_orig - med_map
    inds = np.where(map_orig == hp.UNSEEN)
    med_subtract[inds] = hp.UNSEEN
    return med_subtract

def bin_spectra(Cl, nbins=20, var=False, bins=False):
    ell = np.arange(len(Cl))
    l_bins = np.copy(ell)
    if bins:
        lb = bins
    else:
        lb = np.unique(np.floor(np.logspace(np.log10(2), np.log10(ell.max()), nbins)))
    l_step = np.copy(ell)
    Cl_b = np.zeros(len(lb)-1)
    l_plot = np.zeros(len(lb)-1)
    s_b = np.zeros(len(lb)-1)
    for i in range(len(lb)-1):
        inds = (ell >= lb[i]) & (ell < lb[i+1])
        lib = ell[inds]
        Clib = Cl[inds]
        l_step[inds] = lb[i]
        #l_plot[i] = gmean(ell[(ell >= lb[i]) & (ell < lb[i+1])])
        l_plot[i] = np.average(lib, weights=abs(Clib))
        #l_plot[i] = (lb[i]*lb[i+1])**0.5
        Cl_b[i] = np.mean(Cl[inds])
        #Cl_b[i] = mean(Cl[(ell >= lb[i]) & (ell < lb[i+1])])
        N = len(Cl[inds])
        s_b[i] = np.std(Cl[(ell >= lb[i]) & (ell < lb[i+1])])/N**0.5
    if var:
        return (l_plot, Cl_b, s_b)
    else:
        return (l_plot, Cl_b)

def bin_noisy_spectra(Cl, Nl, nbins=20, var=False, bins=False):
    ell = np.arange(len(Cl))
    l_bins = np.copy(ell)
    if bins:
        lb = bins
    else:
        lb = np.unique(np.floor(np.logspace(np.log10(2), np.log10(ell.max()), nbins)))
    l_step = np.copy(ell)
    Cl_b = np.zeros(len(lb)-1)
    l_plot = np.zeros(len(lb)-1)
    s_b = np.zeros(len(lb)-1)
    for i in range(len(lb)-1):
        inds = (ell >= lb[i]) & (ell < lb[i+1])
        lib = ell[inds]
        Clib = Cl[inds]
        l_step[inds] = lb[i]
        #l_plot[i] = gmean(ell[(ell >= lb[i]) & (ell < lb[i+1])])
        #l_plot[i] = np.average(lib, weights=abs(Clib))
        l_plot[i] = (lb[i]*lb[i+1])**0.5
        Cl_b[i] = np.sum((Cl/Nl**2)[inds])/np.sum(1/Nl[inds]**2)
        s_b[i] = 1/np.sum(1/Nl[inds]**2)**0.5
    return (l_plot, Cl_b, s_b)

def plot_alm(alm, vmin=None, vmax=None, lmin=None, lmax=None,\
        mmin=None, mmax=None, func=None, almind=1, size=20):
    ls = []
    ms = []
    for i in range(len(alm[0])):
        l, m = hp.Alm.getlm(3*128-1, i=i)
        ls.append(l)
        ms.append(m)

    if func:
        c = func(abs(alm[almind]**2))
        if vmin and vmin:
            vmin = func(vmin)
            vmax = func(vmax)
    else:
        c = np.log10(abs(alm[almind])**2)

    plt.scatter(ls, ms, c=c, s=size, vmin=vmin, vmax=vmax)
    plt.xlim([lmin, lmax])
    plt.ylim([mmin, mmax])
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$m$')
    plt.colorbar(label=r'$\log (|a_{\ell m}|^2)$')
    return

def spice_wrap(m1, m2, mask=None, lmax=199):
    '''
    Should work on omar.
    '''
    clfile = 'cls.txt'
    if mask is None:
        weightfile = 'no_mask.fits'
        hp.write_map(weightfile, np.ones(hp.nside2npix(128)), overwrite=True)
    else:
        weightfile = 'mask.fits'
        mask[mask < 0] = 0
        hp.write_map(weightfile, mask, overwrite=True)
    nside = hp.npix2nside(len(m1[0]))
    hp.write_map('mapfile1.fits', m1, overwrite=True)
    hp.write_map('mapfile2.fits', m2, overwrite=True)
    command = ['/home/dwatts/PolSpice_v03-05-01/bin/spice', 
        '-mapfile', 'mapfile1.fits',
        '-mapfile2', 'mapfile2.fits',
        '-verbosity', '1',
        '-weightfile', weightfile,
        '-nlmax', str(lmax),
        '-overwrite', 'YES',
        '-polarization', 'YES',
        '-decouple', 'YES',
        '-symmetric_cl', 'YES', # averages T_map1*E_map2 and T_map2*E_map1, etc.
        '-clfile', clfile,
        '-tolerance', '1e-6',
        ]
    subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    ell, *clhat = np.loadtxt(clfile).T


    return np.array(clhat)


# at ell=80, nu=353 GHz


# C_l = A_s*(nu/9 GHz)**(2*beta_S), but let's just make nu = 2.3 GHz to match S-PASS.

from astropy import constants as c
from astropy import units as u

def g(nu):
    x = nu/56.78
    return (np.exp(x)-1)**2/(x**2*np.exp(x))

# Rescales in thermodynamic units from nu0 to nu using antenna temperature power
# law index.
def alpha(nu, nu0, beta):
    return (nu/nu0)**beta * g(nu)/g(nu0)

def alpha_S(nu, nu0, beta_S):
    return (nu/nu0)**beta_S * g(nu)/g(nu0)

def B_nu(nu, T=22):
    const = 2*c.h*(nu*u.GHz)**3/c.c**2
    x = c.h*nu*u.GHz/(c.k_B*T*u.K)
    return const/(np.exp(x)-1)


def alpha_D(nu, nu0, beta_D):
    return (nu/nu0)**(beta_D-2)*B_nu(nu)/B_nu(nu0)*g(nu)/g(nu0)


def get_synch(ell, nu, beta_S=-3):
    a_S = -3
    AE_S = 4e3 #uK^2
    AB_S = 2e3
    C_lE = AE_S*(ell/80)**a_S
    C_lB = AB_S*(ell/80)**a_S
    D_lSE = C_lE*ell*(ell+1)/(2*np.pi)
    D_lSB = C_lB*ell*(ell+1)/(2*np.pi)
    D_lSE = D_lSE*alpha_S(nu, 2.5, beta_S)**2
    D_lSB = D_lSB*alpha_S(nu, 2.5, beta_S)**2
    return D_lSE, D_lSB

def get_dust(ell, nu, beta_D=1.6):
    AE_D = 315
    AB_D = AE_D/2.
    a_D = -2.4+2
    D_lDE = AE_D*(ell/80)**a_D
    D_lDB = AB_D*(ell/80)**a_D
    D_lDE = D_lDE*alpha_D(nu, 353, beta_D)**2
    D_lDB = D_lDB*alpha_D(nu, 353, beta_D)**2
    return D_lDE, D_lDB



def test_bin():
    ell = np.arange(200)
    Cl = np.zeros(200)
    Cl[2:] = 1/ell[2:]**2
    plt.loglog(ell, Cl)
    l_plot, Cl_b = bin_spectra(Cl)
    plt.plot(l_plot, Cl_b, '.')
    plt.show()
    return

def test_plotalm():
    nside = 128
    ell = np.arange(3.*128)
    Cl = np.zeros_like(ell)
    Cl[2:] = 1./ell[2:]**2
    Cls = np.array([Cl, 0.1*Cl, 0.01*Cl, 0.3*Cl])
    alms = hp.synalm(Cls)
    plot_alm(alms, func=np.log10, vmin=None, vmax=None,\
            lmin=-1, lmax=100, mmin=-1, mmax=100)
    return

def test_binning():
    ell = np.arange(502.)
    Cl = np.zeros(len(ell))
    Cl[2:] = 1./ell[2:]**2
    l, Cl_b = bin_spectra(Cl, bins=[2,3,4,6,9,16,33,102,252,502])
    plt.loglog(ell, Cl)
    plt.plot(l, Cl_b, 'o')
    plt.show()



def test_spectra():
    ell = np.arange(200.)
    Cl0 = np.zeros_like(ell)
    Cl0[2:] = 1./ell[2:]**2
    Cl = np.array([Cl0, 0.1*Cl0, 0.01*Cl0, 0.3*Cl0])
    np.random.seed(0)
    m = hp.synfast(Cl, 128)

    nomask = np.ones_like(m[0])
    clfull = spice_wrap(m, m, mask=nomask)

    theta, phi = hp.pix2ang(128, np.arange(hp.nside2npix(128)))
    lat = -theta*180/np.pi+90
    
    class_mask = hp.smoothing(np.where( (lat > 25) | (lat < -65), 0, 1), fwhm=15*np.pi/180, verbose=False)
    class_mask = np.where(class_mask < 0.01, 0, class_mask)
    
    inv_mask = 1-class_mask

    clnorm = spice_wrap(m, m, mask=class_mask)
    clinv = spice_wrap(m, m, mask=inv_mask)

    plt.loglog(clfull[1], '.', label='Full')
    plt.loglog(clnorm[1], '.', label='Normal')
    plt.loglog(clinv[1], '.', label='Inverse')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    #test_bin()
    #test_plotalm()
    #test_BB()
    #test_binning()
    test_spectra()

