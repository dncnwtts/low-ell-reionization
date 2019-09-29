import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import healpy as hp
import subprocess

from classy import Class
from scipy.stats import gmean
from scipy.integrate import trapz, cumtrapz
from scipy.integrate import quad
from scipy.special import gammaln, gamma

from scipy.special import kv, kvp

from functools import lru_cache
# lru_cache can be used as a decorator for functions that need to be called
# repeatedly with the same expected results, and makes sense for a
# computationally expensive call.
# Note that the results are stored in a dict, a mutable structure, so if you
# modify the output, you will change the output in future calls.

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

def twinplot(ell, Cl, axes=None, label=None, color='k', marker='.',
        linestyle=' ', ymin=None, ymax=None, alpha=1, lw=None, spec='EE',
            xlabels=True, ylabels_l=True, ylabels_r=True):
    if axes == None:
        ax = plt.gca()
        axes = [ax, ax.twinx()]

    axes[0].semilogx(ell, np.log10(Cl), label=label, color=color, marker=marker,
            linestyle=linestyle, alpha=alpha, lw=lw)

    if (ymin == None) & (ymax == None):
        ymin, ymax = axes[0].get_ylim()
    else:
        ymin, ymax = np.log10(ymin), np.log10(ymax)
        axes[0].set_ylim(ymin, ymax)
    xmin, xmax = axes[0].get_xlim()
    values = np.arange(int(ymin), int(ymax)+1)
    labels = 10.**values
    axes[0].set_yticks(values, minor=False)

    # If you do this, you get tick labels like $10^{-16}$.
    f = mpl.ticker.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % 10.**x))
    axes[0].get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(g))
    # If you do this, you get tick labels like 1e-16
    #axes[0].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    xticks = np.array([2, 5, 10, 20, 50, 100, 200])
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xticks)


    l0 = np.array([], dtype='float')
    labels = np.arange(1, 10)
    for i in range(values[0]-1, values[-1]+2):
        l0 = np.concatenate((l0, 10**i*labels))
    axes[0].set_yticks(np.log10(l0), minor=True)
    axes[0].set_ylim(ymin, ymax)

    base = np.array([3, 5, 10, 15])
    labels = np.array([300, 500, 1000, 1500, 3000, 5000,
        10000, 15000, 30000, 50000, int(1e5), int(1.5e5), int(3e5), int(5e5)]).astype('str')
    labels = np.concatenate(((10*base).astype('str'), labels))
    labels = np.concatenate((base.astype('str'), labels))
    labels = np.concatenate((np.round((10**-1*base),2).astype('str'), labels))
    labels = np.concatenate(((np.round(10**-2*base,2)).astype('str'), labels))

    # Convert from uKarcmin to uK^2 sr
    values = np.log10((np.pi*labels.astype('float')/180/60)**2)
    axes[1].set_yticks(values, minor=False)
    axes[1].get_yaxis().set_major_formatter(mpl.ticker.FixedFormatter(labels))


    l0 = np.array([], dtype='float')
    labels = np.arange(1, 10, 0.5)
    for i in range(-2, 6):
        l0 = np.concatenate((l0, 10**i*labels))
    axes[1].set_yticks(np.log10((np.pi*l0/180/60)**2), minor=True)
    axes[1].set_ylim(axes[0].get_ylim())

    if ylabels_l:
        spec = r'\mathrm{{ {0} }}'.format(spec)
        axes[0].set_ylabel(r'$C_\ell^' + spec + r'$ [$\mathrm{\mu K^2\,sr}$]')
    if ylabels_r:
        axes[1].set_ylabel(r'$w_p^{-1/2}$ [$\mathrm{\mu K\ arcmin}$]',
                rotation=270, labelpad=20)

    if xlabels:
        axes[0].set_xlabel(r'$\ell$')
        axes[0].set_xlim(xmin, 95)


    return axes

def get_tau(thermo, zmax=100, xmin=2e-4):
    eta = thermo['conf. time [Mpc]']
    z = thermo['z']
    x_e = thermo['x_e']
    dtaudeta = thermo["kappa' [Mpc^-1]"]
    sigmaTan_p = dtaudeta/x_e
    integrand = -sigmaTan_p*x_e
    return trapz(integrand[(x_e>xmin) & (z<zmax)], x=eta[(x_e>xmin) & (z<zmax)])

def get_tau_z(thermo, zmax=100, xmin=2e-4):
    eta = thermo['conf. time [Mpc]']
    z = thermo['z']
    x_e = thermo['x_e']
    dtaudeta = thermo["kappa' [Mpc^-1]"]
    sigmaTan_p = dtaudeta/x_e
    integrand = -sigmaTan_p*x_e
    return z, cumtrapz(integrand, x=eta)


def get_twotau(thermo, zmax=100, xmin=2e-4, zsplit=False):
    eta = thermo['conf. time [Mpc]']
    z = thermo['z']
    x_e = thermo['x_e']
    dtaudeta = thermo["kappa' [Mpc^-1]"]
    sigmaTan_p = dtaudeta/x_e
    integrand = -sigmaTan_p*x_e
    zre = z[np.where(x_e < 0.5)[0][0]]
    if not zsplit:
        zsplit = 1+zre
    tau_lo = trapz(integrand[(x_e>xmin) & (z<zsplit)], x=eta[(x_e>xmin) & (z<zsplit)])
    tau_hi = trapz(integrand[(x_e>xmin) & (z>zsplit) & (z<zmax)], x=eta[(x_e>xmin) & (z>zsplit) & (z<zmax)])
    return zsplit, tau_lo, tau_hi

#@profile
#@memoize
@lru_cache(maxsize=2**10)
def get_spectra(zreio, x_e, dz=0.5, z_t=28, history=False, spectra=False, both=False, 
                all_spectra=False, lmax=100, therm=False, zstartmax=50,
                verbose=False, rescale=True, only_BB=False, r=0):
    if verbose: print(get_spectra.cache_info())
    cosmo = Class()

    # Params from Table 1 of Pagano et al. 2019, TT,TE,EE+lowE
    #tau0 = 0.0544
    #ln1010A_s = 3.045
    #A_s0 = np.exp(ln1010A_s-10*np.log(10))
    #params = {
    #    'output': 'tCl pCl lCl',
    #    'l_max_scalars': lmax,
    #    'lensing': 'yes',
    #    'A_s': A_s0,
    #    'n_s': 0.9649,
    #    'omega_b': 0.02236,
    #    'omega_cdm': 0.1202,
    #    'H0': 67.27}

    # The Planck baseline results are TT,TE,EE+lowE+lensing
    tau0 = 0.0544
    ln1010A_s = 3.044
    A_s0 = np.exp(ln1010A_s-10*np.log(10))
    params = {
        'modes': 's, t',
        'output': 'tCl pCl lCl',
        'l_max_scalars': lmax,
        'lensing': 'yes',
        'A_s': A_s0,
        'n_s': 0.9649,
        'omega_b': 0.02237,
        'omega_cdm': 0.1200,
        'H0': 67.36,
        'r': r,
        'reio_parametrization' : 'reio_many_tanh',
        'many_tanh_num': 3}

    params['many_tanh_z'] = '3.5,' + str(zreio) +',' + str(z_t)
    params['many_tanh_xe'] = '-2,-1,'+str(max(x_e, 2e-4))
    params['many_tanh_width'] = dz
    params['reionization_z_start_max'] = zstartmax


    
    params['hyper_flat_approximation_nu'] = 7000. # The higher this is, the more exact
    params['transfer_neglect_delta_k_S_t0'] = 0.17 # The higher these are, the more exact
    params['transfer_neglect_delta_k_S_t1'] = 0.05
    params['transfer_neglect_delta_k_S_t2'] = 0.17
    params['transfer_neglect_delta_k_S_e'] = 0.13
    params['delta_l_max'] = 1000 # difference between l_max in unlensed and lensed spectra


    # You HAVE TO run struct_cleanup() after every compute step. It adds 20 MB
    # per compute call otherwise.
    cosmo.set(params)
    cosmo.compute()
    thermo = cosmo.get_thermodynamics()
    if rescale:
        tau = get_tau(thermo)
        params['A_s'] = A_s0*np.exp(-2*tau0)/np.exp(-2*tau)
        cosmo.struct_cleanup()
        cosmo.set(params)
        cosmo.compute()
    Z = (cosmo.T_cmb()*1e6)**2
    if both:
        thermo = cosmo.get_thermodynamics()
        z, xe = thermo['z'], thermo['x_e']
        cls = cosmo.lensed_cl(lmax)
        cosmo.struct_cleanup()
        ell, EE, TE, TT = cls['ell'], cls['ee'], cls['te'], cls['tt']
        if all_spectra:
            return z, xe, ell, EE*Z, TE*Z, TT*Z
        else:
            return z, xe, ell, EE*Z, TE*Z
    elif only_BB:
        cls = cosmo.lensed_cl(lmax)
        cosmo.struct_cleanup()
        ell, BB = cls['ell'], cls['bb']*Z
        return ell, BB

    elif therm:
        therm = cosmo.get_thermodynamics()
        cosmo.struct_cleanup()
        return therm
    elif spectra:
        cls = cosmo.lensed_cl(lmax)
        cosmo.struct_cleanup()
        ell, TT, EE, TE = cls['ell'], cls['tt']*Z, cls['ee']*Z, cls['te']*Z
        if all_spectra:
            return ell, EE, TE, TT
        else:
            return ell, EE, TE

    elif history:
        thermo = cosmo.get_thermodynamics()
        cosmo.struct_cleanup()
        z, xe = thermo['z'], thermo['x_e']
        return z, xe
    else:
        return

@lru_cache(maxsize=2**10)
def get_spectra_tau(tau, dz=0.5, z_t=28, history=False, spectra=False, both=False, 
                all_spectra=False, lmax=100, therm=False, zstartmax=50,
                rescale=True, only_BB=False, r=0):
    cosmo = Class()
    # Params from Table 1 of Pagano et al. 2019, TT,TE,EE+lowE
    #tau0 = 0.0544
    #ln1010A_s = 3.045
    #A_s0 = np.exp(ln1010A_s-10*np.log(10))
    #params = {
    #    'output': 'tCl pCl lCl',
    #    'l_max_scalars': lmax,
    #    'lensing': 'yes',
    #    'A_s': A_s0,
    #    'n_s': 0.9649,
    #    'omega_b': 0.02236,
    #    'omega_cdm': 0.1202,
    #    'H0': 67.27,
    #    'tau_reio': tau}

    # The Planck baseline results are TT,TE,EE+lowE+lensing
    tau0 = 0.0544
    ln1010A_s = 3.044
    A_s0 = np.exp(ln1010A_s-10*np.log(10))
    params = {
        'modes': 's, t',
        'output': 'tCl pCl lCl',
        'l_max_scalars': lmax,
        'lensing': 'yes',
        'A_s': A_s0,
        'n_s': 0.9649,
        'omega_b': 0.02237,
        'omega_cdm': 0.1200,
        'H0': 67.36,
        'tau_reio': tau,
        'r': r}

    params['hyper_flat_approximation_nu'] = 7000. # The higher this is, the more exact
    params['transfer_neglect_delta_k_S_t0'] = 0.17 # The higher these are, the more exact
    params['transfer_neglect_delta_k_S_t1'] = 0.05
    params['transfer_neglect_delta_k_S_t2'] = 0.17
    params['transfer_neglect_delta_k_S_e'] = 0.13
    params['delta_l_max'] = 1000 # difference between l_max in unlensed and lensed spectra


    params['tol_thermo_integration'] = 1e-10


    # You HAVE TO run struct_cleanup() after every compute step. It adds 20 MB
    # per compute call otherwise.
    cosmo.set(params)
    cosmo.compute()
    thermo = cosmo.get_thermodynamics()
    if rescale:
        tau = get_tau(thermo)
        params['A_s'] = A_s0*np.exp(-2*tau0)/np.exp(-2*tau)
        cosmo.struct_cleanup()
        cosmo.set(params)
        cosmo.compute()
    Z = (cosmo.T_cmb()*1e6)**2
    if both:
        thermo = cosmo.get_thermodynamics()
        z, xe = thermo['z'], thermo['x_e']
        cls = cosmo.lensed_cl(lmax)
        cosmo.struct_cleanup()
        ell, EE, TE = cls['ell'], cls['ee'], cls['te']
        if all_spectra:
            return z, xe, ell, EE*Z, TE*Z, cls['tt']*Z
        else:
            return z, xe, ell, EE*Z, TE*Z
    elif only_BB:
        cls = cosmo.lensed_cl(lmax)
        cosmo.struct_cleanup()
        ell, BB = cls['ell'], cls['bb']*Z
        return ell, BB

    elif therm:
        therm = cosmo.get_thermodynamics()
        cosmo.struct_cleanup()
        return therm
    elif spectra:
        cls = cosmo.lensed_cl(lmax)
        cosmo.struct_cleanup()
        ell, TT, EE, TE = cls['ell'], cls['tt']*Z, cls['ee']*Z, cls['te']*Z, 
        if all_spectra:
            return ell, EE, TE, TT
        else:
            return ell, EE, TE

    elif history:
        thermo = cosmo.get_thermodynamics()
        cosmo.struct_cleanup()
        z, xe = thermo['z'], thermo['x_e']
        return z, xe
    else:
        return

def get_spectra_complex(zarr, x_earr, dz=0.5, history=False, spectra=False, both=False, 
                all_spectra=False, lmax=100, therm=False, zstartmax=50):
    cosmo = Class()
    #cosmo.struct_cleanup()
    #cosmo.empty()
    ln1010A_s = 3.0448
    #10*np.log(10)+np.log(A_s) = 3.0448
    A_s0 = np.exp(ln1010A_s-10*np.log(10))
    tau0 = 0.0568
    params = {
        'output': 'tCl pCl lCl',
        'l_max_scalars': lmax,
        'lensing': 'yes',
        'A_s': A_s0,
        'n_s': 0.96824,
        'omega_b': 0.022447,
        'omega_cdm': 0.11923,
        'H0': 67.70,
        'reio_parametrization' : 'reio_inter'}

    inds = (zarr > 6)
    zarr = zarr[inds]
    x_earr = x_earr[inds]
    #der = np.diff(x_earr)/np.diff(zarr)
    #inds = (abs(der)/x_earr[:-1] > 0.1)
    #zarr = zarr[:-1][inds]
    #x_earr = x_earr[:-1][inds]
    print(len(zarr)+4)
    params['reio_inter_num'] = len(zarr) + 4
    params['reio_inter_z'] = '0, 3, 4, 6,' + ','.join([str(num) for num in zarr])
    #params['reio_inter_z'] =   '0,  3,  4,   8,   9,  10,  11, 12'
    #params['reio_inter_xe']= '-2, -2, -1,  -1, 0.9, 0.5, 0.1,  0'
    params['reio_inter_xe'] = '-2, -2, -1, -1,' + ','.join([str(num) for num in x_earr])

    print(len(params['reio_inter_z'].split(',')))
    print(len(params['reio_inter_xe'].split(',')))
    print(params['reio_inter_num'])



    
    params['hyper_flat_approximation_nu'] = 7000. # The higher this is, the more exact
    params['transfer_neglect_delta_k_S_t0'] = 0.17 # The lower these are, the more exact
    params['transfer_neglect_delta_k_S_t1'] = 0.05
    params['transfer_neglect_delta_k_S_t2'] = 0.17
    params['transfer_neglect_delta_k_S_e'] = 0.13
    params['delta_l_max'] = 1000 # difference between l_max in unlensed and lensed spectra


    # You HAVE TO run struct_cleanup() after every compute step. It adds 20 MB
    # per compute call otherwise.
    cosmo.set(params)
    cosmo.compute()
    thermo = cosmo.get_thermodynamics()
    tau = get_tau(thermo)
    params['A_s'] = A_s0*np.exp(-2*tau0)/np.exp(-2*tau)
    cosmo.struct_cleanup()
    cosmo.set(params)
    cosmo.compute()
    Z = (2.7e6)**2
    if both:
        thermo = cosmo.get_thermodynamics()
        z, xe = thermo['z'], thermo['x_e']
        cls = cosmo.lensed_cl(lmax)
        cosmo.struct_cleanup()
        ell, EE, TE = cls['ell'], cls['ee'], cls['te']
        if all_spectra:
            return z, xe, ell, EE*Z, TE*Z, cls['tt']*Z
        else:
            return z, xe, ell, EE*Z, TE*Z

    elif therm:
        therm = cosmo.get_thermodynamics()
        cosmo.struct_cleanup()
        return therm
    elif spectra:
        cls = cosmo.lensed_cl(lmax)
        cosmo.struct_cleanup()
        ell, TT, EE, TE = cls['ell'], cls['tt']*Z, cls['ee']*Z, cls['te']*Z
        if all_spectra:
            return ell, EE, TE, TT
        else:
            return ell, EE, TE

    elif history:
        thermo = cosmo.get_thermodynamics()
        cosmo.struct_cleanup()
        z, xe = thermo['z'], thermo['x_e']
        return z, xe
    else:
        return

def get_spectra_simple(zreio, x_e, dz=0.5, history=False, spectra=False, both=False, 
                all_spectra=False, lmax=100, therm=False, zstartmax=50):
    #cosmo.struct_cleanup()
    #cosmo.empty()
    cosmo = Class()
    params = {
        'output': 'tCl pCl lCl',
        'l_max_scalars': lmax,
        'lensing': 'yes',
        'A_s': 2.3e-9,
        'n_s': 0.965,
        'reio_parametrization' : 'reio_camb'}

    params['z_reio'] = zreio
    params['reionization_width'] = dz
    params['reionization_z_start_max'] = zstartmax


    
    params['hyper_flat_approximation_nu'] = 7000. # The higher this is, the more exact
    params['transfer_neglect_delta_k_S_t0'] = 0.17 # The lower these are, the more exact
    params['transfer_neglect_delta_k_S_t1'] = 0.05
    params['transfer_neglect_delta_k_S_t2'] = 0.17
    params['transfer_neglect_delta_k_S_e'] = 0.13
    params['delta_l_max'] = 1000 # difference between l_max in unlensed and lensed spectra


    # You HAVE TO run struct_cleanup() after every compute step. It adds 20 MB
    # per compute call otherwise.
    cosmo.set(params)
    cosmo.compute()
    thermo = cosmo.get_thermodynamics()
    tau = get_tau(thermo)
    params['A_s'] = 2.3e-9*np.exp(-2*0.06)/np.exp(-2*tau)
    cosmo.struct_cleanup()
    cosmo.set(params)
    cosmo.compute()
    Z = (2.7e6)**2
    if both:
        thermo = cosmo.get_thermodynamics()
        z, xe = thermo['z'], thermo['x_e']
        cls = cosmo.lensed_cl(lmax)
        cosmo.struct_cleanup()
        ell, EE, TE = cls['ell'], cls['ee'], cls['te']
        if all_spectra:
            return z, xe, ell, EE*Z, TE*Z, cls['tt']*Z
        else:
            return z, xe, ell, EE*Z, TE*Z

    elif therm:
        therm = cosmo.get_thermodynamics()
        cosmo.struct_cleanup()
        return therm
    elif spectra:
        cls = cosmo.lensed_cl(lmax)
        cosmo.struct_cleanup()
        ell, TT, EE, TE = cls['ell'], cls['tt']*Z, cls['ee']*Z, cls['te']*Z
        if all_spectra:
            return ell, EE, TE, TT
        else:
            return ell, EE, TE

    elif history:
        thermo = cosmo.get_thermodynamics()
        cosmo.struct_cleanup()
        z, xe = thermo['z'], thermo['x_e']
        return z, xe
    else:
        return

def lnprob_BB_ell(zre, x_e, r, Clhat, N_l=0):
    # This returns log(P)
    ell, BB = get_spectra(zre, x_e,  r=r, spectra=True, lmax=len(Clhat)-1, only_BB=True)
    Cl = BB + N_l
    chi2_ell = (2*ell[2:]+1)*(Clhat[2:]/Cl[2:] + np.log(Cl[2:]) - np.log(Clhat[2:])-1)
    chi2_ell = np.insert(chi2_ell, [0,0], 0)
    return -chi2_ell/2

def lnprob_BB_ell_tau(tau, r, Clhat, N_l=0):
    # This returns log(P)
    ell, B = get_spectra_tau(tau, r=r, lmax=len(Clhat)-1, only_BB=True)
    Cl = BB + N_l
    chi2_ell = (2*ell[2:]+1)*(Clhat[2:]/Cl[2:] + np.log(Cl[2:]) - np.log(Clhat[2:])-1)
    chi2_ell = np.insert(chi2_ell, [0,0], 0)
    return -chi2_ell/2

def lnprob_EE_ell(zre, x_e, Clhat, N_l=0):
    # This returns log(P)
    ell, EE, TE, TT = get_spectra(zre, x_e,  spectra=True, lmax=len(Clhat)-1, all_spectra=True)
    Cl = EE + N_l
    chi2_ell = (2*ell[2:]+1)*(Clhat[2:]/Cl[2:] + np.log(Cl[2:]) - np.log(Clhat[2:])-1)
    chi2_ell = np.insert(chi2_ell, [0,0], 0)
    return -chi2_ell/2

def lnprob_EE_ell_tau(tau, Clhat, N_l=0):
    # This returns log(P)
    ell, EE, TE = get_spectra_tau(tau, lmax=len(Clhat)-1, spectra=True)
    Cl = EE + N_l
    chi2_ell = (2*ell[2:]+1)*(Clhat[2:]/Cl[2:] + np.log(Cl[2:]) - np.log(Clhat[2:])-1)
    chi2_ell = np.insert(chi2_ell, [0,0], 0)
    return -chi2_ell/2

def VG(x, r, theta, sigma, mu):
    # a VG variable is distributed as 
    # Y = mu + theta*S + sigma*S**0.5*T
    # where S~ Gamma(r/2, 1/2) and T is a standard normal.
    f = np.exp(theta*(x-mu)/sigma**2)*(abs(x-mu)/(2*(sigma**2+theta**2)**0.5))**((r-1)/2)
    f *= kv((r-1)/2, (theta**2+sigma**2)**0.5*abs(x-mu)/sigma**2)/(sigma*np.pi**0.5*gamma(r/2))
    return f
    #lnf = theta*(x-mu)/sigma**2 + (r-1)/2*np.log(abs(x-mu)/(2*(sigma**2+theta**2)**0.5))
    #lnf += np.log(kv((r-1)/2, (theta**2+sigma**2)**0.5*abs(x-mu)/sigma**2))
    #lnf -= np.log(sigma) + 0.5*np.log(np.pi) + gammaln(r/2)
    ##lnf[~np.isfinite(lnf)] = 0
    #return np.exp(lnf)

def test_VG():
    x = np.linspace(-1, 1, 100000)
    rhos = np.arange(0, 1, 0.2)
    ell = 2
    N = 2*ell+1
    sigmas = 0.1
    for i in range(len(rhos)):
        rho = rhos[i]
        Zbar = VG(x, N, rho*sigmas/N, sigmas*np.sqrt(1-rho**2)/N, 0)
        plt.plot(x, Zbar, color=plt.cm.gnuplot_r(i/len(rhos)))
        mu = rho*sigmas
        plt.axvline(mu, color=plt.cm.gnuplot_r(i/len(rhos)), linestyle='--')
        plt.axvline(x[np.argmax(Zbar)], color=plt.cm.gnuplot_r(i/len(rhos)),
                linestyle=':')
    plt.xlim([-0.2, 0.3])
    plt.show()
    return

def test_chi2():
    x = np.linspace(0,5,100000)
    ells = [2,5,10,25]
    for i in range(len(ells)):
        ell = ells[i]
        lnL = -(2*ell+1)/2*(x-(2*ell-1)/(2*ell+1)*np.log(x))
        plt.plot(x, np.exp(lnL-lnL.max()), color=f'C{i}')
        plt.axvline(1, color=f'C{i}', linestyle='--')
        plt.axvline((2*ell-1)/(2*ell+1), color=f'C{i}', linestyle=':')
    plt.show()


def lnprob_TE_ell_tau(tau, TEhat, N_lT=0, N_lE=0):
    ell, ee, te, tt = get_spectra_tau(tau,  spectra=True, lmax=len(TEhat)-1, all_spectra=True)
    sigmas = np.sqrt((ee+N_lE)*(tt+N_lT))
    rho = te/sigmas
    z = (1-rho**2)*sigmas
    N = 2*ell+1
    L = VG(TEhat, N, rho*sigmas/N, sigmas*np.sqrt(1-rho**2)/N, 0)
    return np.log(L)

def lnprob_TE_ell(zre, x_e, TEhat, N_lT=0, N_lE=0):
    ell, ee, te, tt = get_spectra(zre, x_e,  spectra=True, lmax=len(TEhat)-1, all_spectra=True)
    sigmas = np.sqrt((ee+N_lE)*(tt+N_lT))
    rho = te/sigmas
    z = (1-rho**2)*sigmas
    N = 2*ell+1
    L = VG(TEhat, N, rho*sigmas/N, sigmas*np.sqrt(1-rho**2)/N, 0)
    return np.log(L)
    #den = np.zeros_like(L)
    #for i in range(len(ell)):
    #    #integrand = lambda x: VG(x, N[i], (rho*sigmas/N)[i], (sigmas*np.sqrt(1-rho**2)/N)[i], 0)
    #    #den[i] = quad(integrand, 1e-3, 1)[1]
    #    x = np.linspace(-1, 1, 1e4)
    #    integrand = VG(x, N[i], (rho*sigmas/N)[i], (sigmas*np.sqrt(1-rho**2)/N)[i], 0)
    #    den[i] = trapz(integrand, x)
    #return np.log(L/den)

def lnprob_wish_ell(zre, x_e, Clhat, N_lT=0, N_lE=0):
    # This returns log(P)
    TThat, EEhat, BBhat, TEhat, TBhat, EBhat = Clhat
    ell, EE, TE, TT = get_spectra(zre, x_e,  spectra=True, lmax=len(EEhat)-1, all_spectra=True)

    tt = TT + N_lT
    ee = EE + N_lE
    te = TE + 0*N_lT
    n = 2
    Cl = np.array([[tt, te], [te, ee]])
    Cl = np.swapaxes(Cl, 1,2)
    Cl = np.swapaxes(Cl, 0,1)
    bla2 = np.linalg.inv(Cl[2:])

    Clhat = np.array([[Clhat[0], Clhat[3]], [Clhat[3], Clhat[1]]])
    Clhat = np.swapaxes(Clhat, 1,2)
    Clhat = np.swapaxes(Clhat, 0,1)
    bla = Clhat[2:]
    
    x = np.einsum('ijk,ikl->ijl',bla, bla2)
    ls = ell[2:]
    chi2_ell = (2*ls+1)*(np.trace(x, axis1=1, axis2=2) - np.linalg.slogdet(x)[1] - n)

    chi2_ell = np.insert(chi2_ell, [0,0], 0)


    return -chi2_ell/2

def lnprob_wish_ell_tau(tau, Clhat, N_lT=0, N_lE=0):
    # This returns log(P)
    TThat, EEhat, BBhat, TEhat, TBhat, EBhat = Clhat
    ell, EE, TE, TT = get_spectra_tau(tau,  spectra=True, lmax=len(EEhat)-1, all_spectra=True)

    tt = TT + N_lT
    ee = EE + N_lE
    te = TE + N_lE*0
    n = 2
    Cl = np.array([[tt, te], [te, ee]])
    Cl = np.swapaxes(Cl, 1,2)
    Cl = np.swapaxes(Cl, 0,1)
    bla2 = np.linalg.inv(Cl[2:])

    Clhat = np.array([[Clhat[0], Clhat[3]], [Clhat[3], Clhat[1]]])
    Clhat = np.swapaxes(Clhat, 1,2)
    Clhat = np.swapaxes(Clhat, 0,1)
    bla = Clhat[2:]
    
    x = np.einsum('ijk,ikl->ijl',bla, bla2)
    ls = ell[2:]
    chi2_ell = (2*ls+1)*(np.trace(x, axis1=1, axis2=2) - np.linalg.slogdet(x)[1] - n)

    chi2_ell = np.insert(chi2_ell, [0,0], 0)


    return -chi2_ell/2

def get_F_ell(zre, x_e, dzre=5e-5, dxre=1e-5, ell_arr=False, lmin=2, lmax=100,
        N_lT=0, N_lE=0, test=False, test2=False, tau_vars=False):
    ell = np.arange(lmax+1)
    if type(N_lT) != np.ndarray:
        N_lT = N_lT * np.ones_like(ell)
    if type(N_lE) != np.ndarray:
        N_lE = N_lE * np.ones_like(ell)
    Fs = []
    Fs_EE = []
    ell, EE, TE, TT = get_spectra(zre, x_e, lmax=lmax, spectra=True, all_spectra=True)
    ell, dEEx, dTEx, dTTx = get_spectra(zre, x_e+dxre, lmax=lmax, spectra=True, all_spectra=True)
    ell, dEEz, dTEz, dTTz = get_spectra(zre+dzre, x_e, lmax=lmax, spectra=True, all_spectra=True)
    dEEdz = (dEEz - EE)/dzre
    dTEdz = (dTEz - TE)/dzre
    dTTdz = (dTTz - TT)/dzre
    dEEdx = (dEEx - EE)/dxre
    dTEdx = (dTEx - TE)/dxre
    dTTdx = (dTTx - TT)/dxre

    if tau_vars:
        zsplit, taulo, tauhi = get_twotau(get_spectra(zre, x_e, therm=True))
        zsplit, d_taulo, _tauhi = get_twotau(get_spectra(zre+dzre, x_e, therm=True))
        zsplit, _taulo, d_tauhi = get_twotau(get_spectra(zre, x_e+dxre, therm=True))

        dtaulo = d_taulo-taulo
        dtauhi = d_tauhi-tauhi

        dEEdz *= dzre/dtaulo
        dTEdz *= dzre/dtaulo
        dTTdz *= dzre/dtaulo
        dEEdx *= dxre/dtauhi
        dTEdx *= dxre/dtauhi
        dTTdx *= dxre/dtauhi

    ders = [[ [], [], [] ],
            [ [], [], [] ]]
    for l in np.arange(lmin, lmax+1):
        if test:
            ders[0][0].append(dTTdx[l])
            ders[0][1].append(dEEdx[l])
            ders[0][2].append(dTEdx[l])
            ders[1][0].append(dTTdz[l])
            ders[1][1].append(dEEdz[l])
            ders[1][2].append(dTEdz[l])
            return dTTdx, dEEdx, dTEdx, dTTdz, dEEdz, dTEdz
        F = np.zeros((2,2))
        Cl = np.array([[TT[l] + N_lT[l], TE[l]],[TE[l],EE[l] + N_lE[l]]])
        dCldx = np.array([[dTTdx[l], dTEdx[l]], [dTEdx[l], dEEdx[l]]])
        dCldz = np.array([[dTTdz[l], dTEdz[l]], [dTEdz[l], dEEdz[l]]])
        Clinv = np.linalg.inv(Cl)
        xx = np.trace(Clinv.dot(dCldx.dot(Clinv.dot(dCldx))))*(2*l+1)/2
        xz = np.trace(Clinv.dot(dCldx.dot(Clinv.dot(dCldz))))*(2*l+1)/2
        zz = np.trace(Clinv.dot(dCldz.dot(Clinv.dot(dCldz))))*(2*l+1)/2
        F[0,0] += zz
        F[1,0] += xz
        F[0,1] += xz
        F[1,1] += xx
        Fs.append(F)

        Cl = EE[l] + N_lE[l]
        dCldx = dEEdx[l]
        dCldz = dEEdz[l]
        Clinv = 1/Cl
        xx = Clinv*dCldx*Clinv*dCldx*(2*l+1)/2
        xz = Clinv*dCldx*Clinv*dCldz*(2*l+1)/2
        zz = Clinv*dCldz*Clinv*dCldz*(2*l+1)/2
        FEE = np.array([[zz, xz],
                        [xz, xx]])

        Fs_EE.append(FEE)
    if test2:
        return Fs, Fs_EE, dTTdx[lmin:lmax+1], dEEdx[lmin:lmax+1],\
    dTEdx[lmin:lmax+1], dTTdz[lmin:lmax+1], dEEdz[lmin:lmax+1],\
    dTEdz[lmin:lmax+1]
    return Fs

def get_F_ell_forcezsplit(zre, x_e, dzre=5e-5, dxre=1e-5, ell_arr=False, lmin=2, lmax=100,
        N_lT=0, N_lE=0, test=False, test2=False, tau_vars=True):
    Fs = []
    Fs_EE = []
    ell, EE, TE, TT = get_spectra(zre, x_e, lmax=lmax, spectra=True, all_spectra=True)
    ell, dEEx, dTEx, dTTx = get_spectra(zre, x_e+dxre, lmax=lmax, spectra=True, all_spectra=True)
    ell, dEEz, dTEz, dTTz = get_spectra(zre+dzre, x_e, lmax=lmax, spectra=True, all_spectra=True)
    dEEdz = (dEEz - EE)/dzre
    dTEdz = (dTEz - TE)/dzre
    dTTdz = (dTTz - TT)/dzre
    dEEdx = (dEEx - EE)/dxre
    dTEdx = (dTEx - TE)/dxre
    dTTdx = (dTTx - TT)/dxre

    if tau_vars:
        zsplit, taulo, tauhi = get_twotau(get_spectra(zre, x_e, therm=True),
                zsplit=15)
        zsplit, d_taulo, _tauhi = get_twotau(get_spectra(zre+dzre, x_e,
            therm=True), zsplit=15)
        zsplit, _taulo, d_tauhi = get_twotau(get_spectra(zre, x_e+dxre,
            therm=True), zsplit=15)

        dtaulo = d_taulo-taulo
        dtauhi = d_tauhi-tauhi

        dEEdz *= dzre/dtaulo
        dTEdz *= dzre/dtaulo
        dTTdz *= dzre/dtaulo
        dEEdx *= dxre/dtauhi
        dTEdx *= dxre/dtauhi
        dTTdx *= dxre/dtauhi

    ders = [[ [], [], [] ],
            [ [], [], [] ]]
    for l in np.arange(lmin, lmax+1):
        F = np.zeros((2,2))
        Cl = np.array([[TT[l] + N_lT, TE[l]],[TE[l],EE[l] + N_lE]])
        dCldx = np.array([[dTTdx[l], dTEdx[l]], [dTEdx[l], dEEdx[l]]])
        dCldz = np.array([[dTTdz[l], dTEdz[l]], [dTEdz[l], dEEdz[l]]])
        Clinv = np.linalg.inv(Cl)
        xx = np.trace(Clinv.dot(dCldx.dot(Clinv.dot(dCldx))))*(2*l+1)/2
        xz = np.trace(Clinv.dot(dCldx.dot(Clinv.dot(dCldz))))*(2*l+1)/2
        zz = np.trace(Clinv.dot(dCldz.dot(Clinv.dot(dCldz))))*(2*l+1)/2
        F[0,0] += zz
        F[1,0] += xz
        F[0,1] += xz
        F[1,1] += xx
        Fs.append(F)

        Cl = EE[l] + N_lE
        dCldx = dEEdx[l]
        dCldz = dEEdz[l]
        Clinv = 1/Cl
        xx = Clinv*dCldx*Clinv*dCldx*(2*l+1)/2
        xz = Clinv*dCldx*Clinv*dCldz*(2*l+1)/2
        zz = Clinv*dCldz*Clinv*dCldz*(2*l+1)/2
        FEE = np.array([[zz, xz],
                        [xz, xx]])

        Fs_EE.append(FEE)
    dTEdx[lmin:lmax+1], dTTdz[lmin:lmax+1], dEEdz[lmin:lmax+1],\
    dTEdz[lmin:lmax+1]
    return Fs, taulo, tauhi

def get_F_ell_2(zre, x_e, dzre=5e-5, dxre=1e-5, ell_arr=False, lmin=2, lmax=100,
        N_lT=0, N_lE=0, test=False, test2=False, tau_vars=False):
    Fs = []
    Fs_EE = []
    ell, EE, TE, TT = get_spectra(zre, x_e, lmax=lmax, spectra=True, all_spectra=True)
    ell, dEEx, dTEx, dTTx = get_spectra(zre, x_e+dxre, lmax=lmax, spectra=True, all_spectra=True)
    ell, dEEz, dTEz, dTTz = get_spectra(zre+dzre, x_e, lmax=lmax, spectra=True, all_spectra=True)
    dEEdz = (dEEz - EE)/dzre
    dTEdz = (dTEz - TE)/dzre
    dTTdz = (dTTz - TT)/dzre
    dEEdx = (dEEx - EE)/dxre
    dTEdx = (dTEx - TE)/dxre
    dTTdx = (dTTx - TT)/dxre

    if tau_vars:
        zsplit, taulo, tauhi = get_twotau(get_spectra(zre, x_e, therm=True))
        zsplit, d_taulo, _tauhi = get_twotau(get_spectra(zre+dzre, x_e, therm=True))
        zsplit, _taulo, d_tauhi = get_twotau(get_spectra(zre, x_e+dxre, therm=True))

        dtaulo = d_taulo-taulo
        dtauhi = d_tauhi-tauhi

        dEEdz *= dzre/dtaulo
        dTEdz *= dzre/dtaulo
        dTTdz *= dzre/dtaulo
        dEEdx *= dxre/dtauhi
        dTEdx *= dxre/dtauhi
        dTTdx *= dxre/dtauhi

    ders = [[ [], [], [] ],
            [ [], [], [] ]]
    for l in np.arange(lmin, lmax+1):
        if test:
            ders[0][0].append(dTTdx[l])
            ders[0][1].append(dEEdx[l])
            ders[0][2].append(dTEdx[l])
            ders[1][0].append(dTTdz[l])
            ders[1][1].append(dEEdz[l])
            ders[1][2].append(dTEdz[l])
            return dTTdx, dEEdx, dTEdx, dTTdz, dEEdz, dTEdz
        F = np.zeros((2,2))
        Cl = np.array([[TT[l] + N_lT, 0*TE[l]],[0*TE[l],EE[l] + N_lE]])
        dCldx = np.array([[dTTdx[l], 0*dTEdx[l]], [0*dTEdx[l], dEEdx[l]]])
        dCldz = np.array([[dTTdz[l], 0*dTEdz[l]], [0*dTEdz[l], dEEdz[l]]])
        Clinv = np.linalg.inv(Cl)
        xx = np.trace(Clinv.dot(dCldx.dot(Clinv.dot(dCldx))))*(2*l+1)/2
        xz = np.trace(Clinv.dot(dCldx.dot(Clinv.dot(dCldz))))*(2*l+1)/2
        zz = np.trace(Clinv.dot(dCldz.dot(Clinv.dot(dCldz))))*(2*l+1)/2
        F[0,0] += zz
        F[1,0] += xz
        F[0,1] += xz
        F[1,1] += xx
        Fs.append(F)

        Cl = EE[l] + N_lE
        dCldx = dEEdx[l]
        dCldz = dEEdz[l]
        Clinv = 1/Cl
        xx = Clinv*dCldx*Clinv*dCldx*(2*l+1)/2
        xz = Clinv*dCldx*Clinv*dCldz*(2*l+1)/2
        zz = Clinv*dCldz*Clinv*dCldz*(2*l+1)/2
        FEE = np.array([[zz, xz],
                        [xz, xx]])

        Fs_EE.append(FEE)
    if test2:
        return Fs, Fs_EE, dTTdx, dEEdx, dTEdx, dTTdz, dEEdz, dTEdz
    return Fs

def get_F_ell_3(zre, x_e, dzre=5e-5, dxre=1e-5, ell_arr=False, lmin=2, lmax=100,
        N_lT=0, N_lE=0, test=False, test2=False, tau_vars=False):
    '''
    Sets the derivative of T and E with respect to tau equal to zero.
    '''
    Fs = []
    Fs_EE = []
    ell, EE, TE, TT = get_spectra(zre, x_e, lmax=lmax, spectra=True, all_spectra=True)
    ell, dEEx, dTEx, dTTx = get_spectra(zre, x_e+dxre, lmax=lmax, spectra=True, all_spectra=True)
    ell, dEEz, dTEz, dTTz = get_spectra(zre+dzre, x_e, lmax=lmax, spectra=True, all_spectra=True)
    dEEdz = (dEEz - EE)/dzre
    dTEdz = (dTEz - TE)/dzre
    dTTdz = (dTTz - TT)/dzre
    dEEdx = (dEEx - EE)/dxre
    dTEdx = (dTEx - TE)/dxre
    dTTdx = (dTTx - TT)/dxre

    if tau_vars:
        zsplit, taulo, tauhi = get_twotau(get_spectra(zre, x_e, therm=True))
        zsplit, d_taulo, _tauhi = get_twotau(get_spectra(zre+dzre, x_e, therm=True))
        zsplit, _taulo, d_tauhi = get_twotau(get_spectra(zre, x_e+dxre, therm=True))

        dtaulo = d_taulo-taulo
        dtauhi = d_tauhi-tauhi

        dEEdz *= dzre/dtaulo
        dTEdz *= dzre/dtaulo
        dTTdz *= dzre/dtaulo
        dEEdx *= dxre/dtauhi
        dTEdx *= dxre/dtauhi
        dTTdx *= dxre/dtauhi

    ders = [[ [], [], [] ],
            [ [], [], [] ]]
    for l in np.arange(lmin, lmax+1):
        if test:
            ders[0][0].append(dTTdx[l])
            ders[0][1].append(dEEdx[l])
            ders[0][2].append(dTEdx[l])
            ders[1][0].append(dTTdz[l])
            ders[1][1].append(dEEdz[l])
            ders[1][2].append(dTEdz[l])
            return dTTdx, dEEdx, dTEdx, dTTdz, dEEdz, dTEdz
        F = np.zeros((2,2))
        Cl = np.array([[TT[l] + N_lT, TE[l]],[TE[l],EE[l] + N_lE]])
        dCldx = np.array([[0*dTTdx[l], dTEdx[l]], [dTEdx[l], 0*dEEdx[l]]])
        dCldz = np.array([[0*dTTdz[l], dTEdz[l]], [dTEdz[l], 0*dEEdz[l]]])
        Clinv = np.linalg.inv(Cl)
        xx = np.trace(Clinv.dot(dCldx.dot(Clinv.dot(dCldx))))*(2*l+1)/2
        xz = np.trace(Clinv.dot(dCldx.dot(Clinv.dot(dCldz))))*(2*l+1)/2
        zz = np.trace(Clinv.dot(dCldz.dot(Clinv.dot(dCldz))))*(2*l+1)/2
        F[0,0] += zz
        F[1,0] += xz
        F[0,1] += xz
        F[1,1] += xx
        Fs.append(F)

        Cl = EE[l] + N_lE
        dCldx = dEEdx[l]
        dCldz = dEEdz[l]
        Clinv = 1/Cl
        xx = Clinv*dCldx*Clinv*dCldx*(2*l+1)/2
        xz = Clinv*dCldx*Clinv*dCldz*(2*l+1)/2
        zz = Clinv*dCldz*Clinv*dCldz*(2*l+1)/2
        FEE = np.array([[zz, xz],
                        [xz, xx]])

        Fs_EE.append(FEE)
    if test2:
        return Fs, Fs_EE, dTTdx, dEEdx, dTEdx, dTTdz, dEEdz, dTEdz
    return Fs

def get_F_ell_4(zre, x_e, dzre=5e-5, dxre=1e-5, ell_arr=False, lmin=2, lmax=100,
        N_lT=0, N_lE=0, test=False, test2=False, tau_vars=False):
    '''
    Sets the derivative of E with respect to tau equal to zero.
    '''
    Fs = []
    Fs_EE = []
    ell, EE, TE, TT = get_spectra(zre, x_e, lmax=lmax, spectra=True, all_spectra=True)
    ell, dEEx, dTEx, dTTx = get_spectra(zre, x_e+dxre, lmax=lmax, spectra=True, all_spectra=True)
    ell, dEEz, dTEz, dTTz = get_spectra(zre+dzre, x_e, lmax=lmax, spectra=True, all_spectra=True)
    dEEdz = (dEEz - EE)/dzre
    dTEdz = (dTEz - TE)/dzre
    dTTdz = (dTTz - TT)/dzre
    dEEdx = (dEEx - EE)/dxre
    dTEdx = (dTEx - TE)/dxre
    dTTdx = (dTTx - TT)/dxre

    if tau_vars:
        zsplit, taulo, tauhi = get_twotau(get_spectra(zre, x_e, therm=True))
        zsplit, d_taulo, _tauhi = get_twotau(get_spectra(zre+dzre, x_e, therm=True))
        zsplit, _taulo, d_tauhi = get_twotau(get_spectra(zre, x_e+dxre, therm=True))

        dtaulo = d_taulo-taulo
        dtauhi = d_tauhi-tauhi

        dEEdz *= dzre/dtaulo
        dTEdz *= dzre/dtaulo
        dTTdz *= dzre/dtaulo
        dEEdx *= dxre/dtauhi
        dTEdx *= dxre/dtauhi
        dTTdx *= dxre/dtauhi

    ders = [[ [], [], [] ],
            [ [], [], [] ]]
    for l in np.arange(lmin, lmax+1):
        if test:
            ders[0][0].append(dTTdx[l])
            ders[0][1].append(dEEdx[l])
            ders[0][2].append(dTEdx[l])
            ders[1][0].append(dTTdz[l])
            ders[1][1].append(dEEdz[l])
            ders[1][2].append(dTEdz[l])
            return dTTdx, dEEdx, dTEdx, dTTdz, dEEdz, dTEdz
        F = np.zeros((2,2))
        Cl = np.array([[TT[l] + N_lT, TE[l]],[TE[l],EE[l] + N_lE]])
        dCldx = np.array([[dTTdx[l], dTEdx[l]], [dTEdx[l], 0*dEEdx[l]]])
        dCldz = np.array([[dTTdz[l], dTEdz[l]], [dTEdz[l], 0*dEEdz[l]]])
        Clinv = np.linalg.inv(Cl)
        xx = np.trace(Clinv.dot(dCldx.dot(Clinv.dot(dCldx))))*(2*l+1)/2
        xz = np.trace(Clinv.dot(dCldx.dot(Clinv.dot(dCldz))))*(2*l+1)/2
        zz = np.trace(Clinv.dot(dCldz.dot(Clinv.dot(dCldz))))*(2*l+1)/2
        F[0,0] += zz
        F[1,0] += xz
        F[0,1] += xz
        F[1,1] += xx
        Fs.append(F)

        Cl = EE[l] + N_lE
        dCldx = dEEdx[l]
        dCldz = dEEdz[l]
        Clinv = 1/Cl
        xx = Clinv*dCldx*Clinv*dCldx*(2*l+1)/2
        xz = Clinv*dCldx*Clinv*dCldz*(2*l+1)/2
        zz = Clinv*dCldz*Clinv*dCldz*(2*l+1)/2
        FEE = np.array([[zz, xz],
                        [xz, xx]])

        Fs_EE.append(FEE)
    if test2:
        return Fs, Fs_EE, dTTdx, dEEdx, dTEdx, dTTdz, dEEdz, dTEdz
    return Fs

def get_F_ell_5(zre, x_e, dzre=5e-5, dxre=1e-5, ell_arr=False, lmin=2, lmax=100,
        N_lT=0, N_lE=0, test=False, test2=False, tau_vars=False):
    '''
    Sets the derivative of T with respect to tau equal to zero.
    '''
    Fs = []
    Fs_EE = []
    ell, EE, TE, TT = get_spectra(zre, x_e, lmax=lmax, spectra=True, all_spectra=True)
    ell, dEEx, dTEx, dTTx = get_spectra(zre, x_e+dxre, lmax=lmax, spectra=True, all_spectra=True)
    ell, dEEz, dTEz, dTTz = get_spectra(zre+dzre, x_e, lmax=lmax, spectra=True, all_spectra=True)
    dEEdz = (dEEz - EE)/dzre
    dTEdz = (dTEz - TE)/dzre
    dTTdz = (dTTz - TT)/dzre
    dEEdx = (dEEx - EE)/dxre
    dTEdx = (dTEx - TE)/dxre
    dTTdx = (dTTx - TT)/dxre

    if tau_vars:
        zsplit, taulo, tauhi = get_twotau(get_spectra(zre, x_e, therm=True))
        zsplit, d_taulo, _tauhi = get_twotau(get_spectra(zre+dzre, x_e, therm=True))
        zsplit, _taulo, d_tauhi = get_twotau(get_spectra(zre, x_e+dxre, therm=True))

        dtaulo = d_taulo-taulo
        dtauhi = d_tauhi-tauhi

        dEEdz *= dzre/dtaulo
        dTEdz *= dzre/dtaulo
        dTTdz *= dzre/dtaulo
        dEEdx *= dxre/dtauhi
        dTEdx *= dxre/dtauhi
        dTTdx *= dxre/dtauhi

    ders = [[ [], [], [] ],
            [ [], [], [] ]]
    for l in np.arange(lmin, lmax+1):
        if test:
            ders[0][0].append(dTTdx[l])
            ders[0][1].append(dEEdx[l])
            ders[0][2].append(dTEdx[l])
            ders[1][0].append(dTTdz[l])
            ders[1][1].append(dEEdz[l])
            ders[1][2].append(dTEdz[l])
            return dTTdx, dEEdx, dTEdx, dTTdz, dEEdz, dTEdz
        F = np.zeros((2,2))
        Cl = np.array([[TT[l] + N_lT, TE[l]],[TE[l],EE[l] + N_lE]])
        dCldx = np.array([[0*dTTdx[l], dTEdx[l]], [dTEdx[l], dEEdx[l]]])
        dCldz = np.array([[0*dTTdz[l], dTEdz[l]], [dTEdz[l], dEEdz[l]]])
        Clinv = np.linalg.inv(Cl)
        xx = np.trace(Clinv.dot(dCldx.dot(Clinv.dot(dCldx))))*(2*l+1)/2
        xz = np.trace(Clinv.dot(dCldx.dot(Clinv.dot(dCldz))))*(2*l+1)/2
        zz = np.trace(Clinv.dot(dCldz.dot(Clinv.dot(dCldz))))*(2*l+1)/2
        F[0,0] += zz
        F[1,0] += xz
        F[0,1] += xz
        F[1,1] += xx
        Fs.append(F)

        Cl = EE[l] + N_lE
        dCldx = dEEdx[l]
        dCldz = dEEdz[l]
        Clinv = 1/Cl
        xx = Clinv*dCldx*Clinv*dCldx*(2*l+1)/2
        xz = Clinv*dCldx*Clinv*dCldz*(2*l+1)/2
        zz = Clinv*dCldz*Clinv*dCldz*(2*l+1)/2
        FEE = np.array([[zz, xz],
                        [xz, xx]])

        Fs_EE.append(FEE)
    if test2:
        return Fs, Fs_EE, dTTdx, dEEdx, dTEdx, dTTdz, dEEdz, dTEdz
    return Fs

def get_F_ell_3_tau(tau, dtau=1e-4, ell_arr=False, lmin=2, lmax=100,
        N_lT=0, N_lE=0, test=False, test2=False, tau_vars=False, TT_fac=0,
        TE_fac=1, EE_fac=0):
    '''
    Sets the derivative of T and E with respect to tau equal to zero.
    '''
    Fs = []
    Fs_EE = []
    ell, EE, TE, TT = get_spectra_tau(tau, lmax=lmax, spectra=True, all_spectra=True)
    ell, dEEx, dTEx, dTTx = get_spectra_tau(tau+dtau, lmax=lmax, spectra=True, all_spectra=True)
    dEEdx = (dEEx - EE)/dtau
    dTEdx = (dTEx - TE)/dtau
    dTTdx = (dTTx - TT)/dtau

    # second derivatives
    ell, mEEx, mTEx, mTTx = get_spectra_tau(tau-dtau, lmax=lmax, spectra=True, all_spectra=True)

    d2EEdx2 = (dEEx + mEEx - 2*EE)/dtau**2
    d2TEdx2 = (dTEx + mTEx - 2*TE)/dtau**2
    d2TTdx2 = (dTTx + mTTx - 2*TT)/dtau**2

    plt.figure()
    plt.loglog(ell[2:], EE[2:], label='E', color='C0')
    plt.loglog(ell[2:], dEEdx[2:], label="E'", color='C1')
    plt.loglog(ell[2:], -dEEdx[2:], linestyle=':', color='C1')
    plt.loglog(ell[2:], d2EEdx2[2:], label="E''", color='C2')
    plt.loglog(ell[2:], -d2EEdx2[2:], linestyle=':', color='C2')
    plt.legend(loc='best')
    plt.savefig('test.pdf')
    plt.close()

    for l in np.arange(lmin, lmax+1):
        F = 0
        Cl = np.array([[TT[l] + N_lT, TE[l]],[TE[l],EE[l] + N_lE]])
        Clhat = np.array([[TT[l] + N_lT, TE[l]], [TE[l], EE[l] + N_lE]])
        Clhat = np.array([[0*(TT[l] + N_lT), TE[l]], [TE[l], 0*(EE[l] + N_lE)]])
        dCldx = np.array([[dTTdx[l], dTEdx[l]], 
                          [dTEdx[l], dEEdx[l]]])
        d2Cldx2 = np.array([[d2TTdx2[l], d2TEdx2[l]],
                            [d2TEdx2[l], d2EEdx2[l]]])
        Clinv = np.linalg.inv(Cl)

        xx1 = np.trace(-Clinv.dot(dCldx.dot(Clinv.dot(dCldx))) + Clinv.dot(d2Cldx2))
        xx2 = np.trace(Cl.dot(2*Clinv.dot(dCldx.dot(Clinv.dot(dCldx.dot(Clinv)))) \
                - Clinv.dot(d2Cldx2.dot(Clinv))))
        print('Cldot')
        print(int(xx1), int(xx2))
        xx2 = np.trace(Clhat.dot(2*Clinv.dot(dCldx.dot(Clinv.dot(dCldx.dot(Clinv)))) \
                - Clinv.dot(d2Cldx2.dot(Clinv))))
        print('Clhatdot')
        print(int(xx1), int(xx2))
        print(xx2)
        F += (xx1+xx2)*(2*l+1)/2
        print('\n')

        #xx1 = -np.trace(Clinv.dot(dCldx.dot(Clinv.dot(dCldx))))
        #xx2 = np.trace(Clhat.dot(2*Clinv.dot(dCldx.dot(Clinv.dot(dCldx.dot(Clinv))))))
        #xx3 = np.trace(Clinv.dot(d2Cldx2) - Clhat.dot(Clinv.dot(d2Cldx2.dot(Clinv))))
        ##print(xx3)
        #F += (xx1+xx2+xx3)*(2*l+1)/2
        Fs.append(F)

    return Fs

def get_F_ell_tau(tau, dtau=1e-4, ell_arr=False, lmin=2, lmax=100,
        N_lT=0, N_lE=0, test=False, test2=False, tau_vars=False,
        TT_fac=1, TE_fac=1, EE_fac=1):
    Fs = []
    Fs_EE = []
    ell, EE, TE, TT = get_spectra_tau(tau, lmax=lmax, spectra=True, all_spectra=True)
    ell, dEEx, dTEx, dTTx = get_spectra_tau(tau+dtau, lmax=lmax, spectra=True, all_spectra=True)
    dEEdx = (dEEx - EE)/dtau
    dTEdx = (dTEx - TE)/dtau
    dTTdx = (dTTx - TT)/dtau

    for l in np.arange(lmin, lmax+1):
        F = 0
        Cl = np.array([[TT[l] + N_lT, TE[l]],[TE[l],EE[l] + N_lE]])
        dCldx = np.array([[TT_fac*dTTdx[l], TE_fac*dTEdx[l]], 
                          [TE_fac*dTEdx[l], EE_fac*dEEdx[l]]])
        Clinv = np.linalg.inv(Cl)
        xx = np.trace(Clinv.dot(dCldx.dot(Clinv.dot(dCldx))))*(2*l+1)/2
        F += xx
        Fs.append(F)

    return Fs

from classtools.users.djw.tools import (med_subtract, spice_wrap, bin_spectra,
        bin_noisy_spectra, twinx_whitenoise, read_classmap,
        get_TT, get_TE, get_EE, rotate_map, alpha)
from scipy.integrate import trapz

def test_lnL():
    lmin, lmax = 2, 30
    zre = 7
    zres = np.linspace(6,10,50)
    ell, EE, TE = get_spectra(zre, 0, lmax=lmax, spectra=True)
    plt.figure()
    ell = np.arange(len(TE))
    lnLs = np.zeros_like(zres)
    for i in range(len(zres)):
        chi2_TE = lnprob_TE_ell(zres[i], 0, TE)
        plt.plot(ell, chi2_TE, color=plt.cm.viridis(i/len(zres)))
        lnLs[i] = sum(chi2_TE[lmin:lmax+1])
    chi2_TE_True = lnprob_TE_ell(zre, 0, TE)
    plt.plot(ell, chi2_TE_True, color='r')
    plt.xscale('log')
    plt.figure()
    L = np.exp(lnLs - lnLs.max())
    plt.plot(zres, L)
    title = trapz(L*zres, zres)/trapz(L, zres)
    plt.title(r'$\bar z={0}$'.format(title))
    plt.axvline(zre)

    plt.figure()
    ell = np.arange(len(EE))
    lnLs = np.zeros_like(zres)
    for i in range(len(zres)):
        chi2_EE = lnprob_EE_ell(zres[i], 0, EE)
        plt.plot(ell, chi2_EE, color=plt.cm.viridis(i/len(zres)))
        lnLs[i] = sum(chi2_EE[lmin:lmax+1])
    chi2_EE_True = lnprob_EE_ell(zre, 0, EE)
    plt.plot(ell, chi2_EE_True, color='r')
    plt.xscale('log')
    plt.figure()
    plt.plot(zres, np.exp(lnLs-lnLs.max()))
    plt.axvline(zre)

    plt.show()


#def get_F_ell_TE(zre, x_e, dzre=5e-5, dxre=1e-5, ell_arr=False, lmin=2, lmax=100,
#        N_lT=0, N_lE=0, test=False, test2=False, tau_vars=False):
def get_F_ell_TE_tau(tau, dtau=1e-3, ell_arr=False, lmin=2, lmax=100,
        N_lT=0, N_lE=0):
    ell, EE, TE, TT = get_spectra_tau(tau, lmax=lmax, spectra=True, all_spectra=True)
    ell, EE_p, TE_p, TT_p = get_spectra_tau(tau + dtau, lmax=lmax, spectra=True, all_spectra=True)
    ell, EE_m, TE_m, TT_m = get_spectra_tau(tau - dtau, lmax=lmax, spectra=True, all_spectra=True)
    F = []
    sigmas = np.sqrt((EE+N_lE)*(TT+N_lT))
    sE = np.sqrt(EE+N_lE)
    sT = np.sqrt(TT+N_lT)
    rho = TE/sigmas
    z = (1-rho**2)*sigmas
    N = 2*ell+1

    Cl_th = np.array([TT, EE, 0*EE, TE])

    for i in range(n_real):
        Cl_hat = hp.alm2cl(hp.synalm(Cl_th, new=True))

        chat = Cl_hat[3]


        dEE = (EE_p - EE)/dtau
        dTE = (TE_p - TE)/dtau
        dTT = (TT_p - TT)/dtau

        d2EE = (EE_p + EE_m - 2*EE)/dtau**2
        d2TE = (TE_p + TE_m - 2*TE)/dtau**2
        d2TT = (TT_p + TT_m - 2*TT)/dtau**2

        dsE = sE*dEE/(1+N_lE/EE)/EE/2
        dsT = sE*dTT/(1+N_lT/TT)/TT/2

        ds2E = sE*d2EE/(1+N_lE/EE)/EE/2 + dsE*dEE/(1+N_lE/EE)/2 + sE*(dEE/EE)**2*N_lE/EE/(1+N_lE/EE)**2
        ds2T = sT*d2TT/(1+N_lT/TT)/TT/2 + dsT*dTT/(1+N_lT/TT)/2 + sT*(dTT/TT)**2*N_lT/TT/(1+N_lT/TT)**2

        drho = rho*(dTE/TE - dEE/EE/(1+N_lE/EE) - dTT/TT/(1+N_lT/TT))

        d2rho = drho**2/rho + rho*(d2TE/TE - d2EE/EE/(1+N_lE/EE)-dEE/EE*N_lE/EE/(1+N_lE/EE)**2 \
                                           - d2TT/TT/(1+N_lT/TT)-dTT/TT*N_lT/TT/(1+N_lT/TT)**2)

        dz = -2*TE*drho + z*dEE/EE/(1+N_lE/EE)/2 + z*dTT/TT/(1+N_lT/TT)/2
        d2z = -2*dTE*drho - TE*d2rho + dz*dEE/EE/(1+N_lE/EE)/2 + (dEE/EE)**2*z*(N_lE/EE)*z/(1+N_lE/EE)**2/2 + z*d2EE/EE/(1+N_lE/EE)\
                                     + dz*dTT/TT/(1+N_lT/TT)/2 + (dTT/TT)**2*z*(N_lT/TT)*z/(1+N_lT/TT)**2/2 + z*d2TT/TT/(1+N_lT/TT)


        # should be -2 lnL_{1,2}
        # the bessel function kvp(v, z, n=1) where v is the order, z is the
        # argument, n is the order of the derivative.
        x = N*chat/z
        der2 = d2z/z*(1+2*x*rho + 2*abs(x)*kvp(ell, abs(x), n=1)/kv(ell,abs(x)))\
                + dz/z*(2*x*drho - 2*x*rho*dz - 2*abs(z)*dz*kvp(ell, abs(x), n=1)/kv(ell, abs(x)) \
                - 2*x**2*dz*(kvp(ell, abs(x), n=2) - kvp(ell, abs(x), n=1))*kvp(ell, abs(x), n=1)/kp(ell, abs(x))**2)\
                + N*(ds2E/sE + ds2T/sT - 2*x/z*dz/z*drho - 2*x/z*d2rho)

    return np.mean(der2/2)

def test_lnL_BB(r=0):
    lmin, lmax = 2, 30
    zre = 7
    zres = np.linspace(6,10,50)
    ell, BB = get_spectra(zre, 0, r=r, lmax=lmax, spectra=True, only_BB=True)
    plt.figure()
    ell = np.arange(len(BB))
    lnLs = np.zeros_like(zres)
    for i in range(len(zres)):
        chi2_BB = lnprob_BB_ell(zres[i], 0, r, BB)
        plt.plot(ell, chi2_BB, color=plt.cm.viridis(i/len(zres)))
        lnLs[i] = sum(chi2_BB[lmin:lmax+1])
    chi2_BB_True = lnprob_BB_ell(zre, 0, r, BB)
    plt.plot(ell, chi2_BB_True, color='r')
    plt.xscale('log')
    plt.figure()
    L = np.exp(lnLs - lnLs.max())
    plt.plot(zres, L)
    title = trapz(L*zres, zres)/trapz(L, zres)
    plt.title(r'$\bar z={0}$'.format(title))
    plt.axvline(zre)

    plt.show()


#def get_F_ell_TE(zre, x_e, dzre=5e-5, dxre=1e-5, ell_arr=False, lmin=2, lmax=100,
#        N_lT=0, N_lE=0, test=False, test2=False, tau_vars=False):
def get_F_ell_TE_tau(tau, dtau=1e-3, ell_arr=False, lmin=2, lmax=100,
        N_lT=0, N_lE=0):
    ell, EE, TE, TT = get_spectra_tau(tau, lmax=lmax, spectra=True, all_spectra=True)
    ell, EE_p, TE_p, TT_p = get_spectra_tau(tau + dtau, lmax=lmax, spectra=True, all_spectra=True)
    ell, EE_m, TE_m, TT_m = get_spectra_tau(tau - dtau, lmax=lmax, spectra=True, all_spectra=True)
    F = []
    sigmas = np.sqrt((EE+N_lE)*(TT+N_lT))
    sE = np.sqrt(EE+N_lE)
    sT = np.sqrt(TT+N_lT)
    rho = TE/sigmas
    z = (1-rho**2)*sigmas
    N = 2*ell+1

    Cl_th = np.array([TT, EE, 0*EE, TE])

    for i in range(n_real):
        Cl_hat = hp.alm2cl(hp.synalm(Cl_th, new=True))

        chat = Cl_hat[3]


        dEE = (EE_p - EE)/dtau
        dTE = (TE_p - TE)/dtau
        dTT = (TT_p - TT)/dtau

        d2EE = (EE_p + EE_m - 2*EE)/dtau**2
        d2TE = (TE_p + TE_m - 2*TE)/dtau**2
        d2TT = (TT_p + TT_m - 2*TT)/dtau**2

        dsE = sE*dEE/(1+N_lE/EE)/EE/2
        dsT = sE*dTT/(1+N_lT/TT)/TT/2

        ds2E = sE*d2EE/(1+N_lE/EE)/EE/2 + dsE*dEE/(1+N_lE/EE)/2 + sE*(dEE/EE)**2*N_lE/EE/(1+N_lE/EE)**2
        ds2T = sT*d2TT/(1+N_lT/TT)/TT/2 + dsT*dTT/(1+N_lT/TT)/2 + sT*(dTT/TT)**2*N_lT/TT/(1+N_lT/TT)**2

        drho = rho*(dTE/TE - dEE/EE/(1+N_lE/EE) - dTT/TT/(1+N_lT/TT))

        d2rho = drho**2/rho + rho*(d2TE/TE - d2EE/EE/(1+N_lE/EE)-dEE/EE*N_lE/EE/(1+N_lE/EE)**2 \
                                           - d2TT/TT/(1+N_lT/TT)-dTT/TT*N_lT/TT/(1+N_lT/TT)**2)

        dz = -2*TE*drho + z*dEE/EE/(1+N_lE/EE)/2 + z*dTT/TT/(1+N_lT/TT)/2
        d2z = -2*dTE*drho - TE*d2rho + dz*dEE/EE/(1+N_lE/EE)/2 + (dEE/EE)**2*z*(N_lE/EE)*z/(1+N_lE/EE)**2/2 + z*d2EE/EE/(1+N_lE/EE)\
                                     + dz*dTT/TT/(1+N_lT/TT)/2 + (dTT/TT)**2*z*(N_lT/TT)*z/(1+N_lT/TT)**2/2 + z*d2TT/TT/(1+N_lT/TT)


        # should be -2 lnL_{1,2}
        # the bessel function kvp(v, z, n=1) where v is the order, z is the
        # argument, n is the order of the derivative.
        x = N*chat/z
        der2 = d2z/z*(1+2*x*rho + 2*abs(x)*kvp(ell, abs(x), n=1)/kv(ell,abs(x)))\
                + dz/z*(2*x*drho - 2*x*rho*dz - 2*abs(z)*dz*kvp(ell, abs(x), n=1)/kv(ell, abs(x)) \
                - 2*x**2*dz*(kvp(ell, abs(x), n=2) - kvp(ell, abs(x), n=1))*kvp(ell, abs(x), n=1)/kp(ell, abs(x))**2)\
                + N*(ds2E/sE + ds2T/sT - 2*x/z*dz/z*drho - 2*x/z*d2rho)

    return np.mean(der2/2)


if __name__ == '__main__':
    #test_bin()
    #test_plotalm()
    #test_BB()
    #test_binning()
    #test_spectra()
    #test_VG()
    #test_chi2()
    #test_lnL()
    test_lnL_BB(r=0.01)

