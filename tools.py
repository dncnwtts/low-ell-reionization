import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import healpy as hp
import subprocess

from classy import Class
from scipy.stats import gmean
from scipy.integrate import trapz, cumtrapz

from memory_profiler import profile



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
        axes[1].set_ylabel(r'$w_p^{-1/2}$ [$\mathrm{\mu K\ arcmin}$]')

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

#@profile
cosmo = Class()
def get_spectra(zreio, x_e, dz=0.5, history=False, spectra=False, both=False, 
                all_spectra=False, lmax=100, therm=False, zstartmax=50):
    #cosmo.struct_cleanup()
    #cosmo.empty()
    params = {
        'output': 'tCl pCl lCl',
        'l_max_scalars': lmax,
        'lensing': 'yes',
        'A_s': 2.3e-9,
        'n_s': 0.965,
        'reio_parametrization' : 'reio_many_tanh',
        'many_tanh_num': 3}

    params['many_tanh_z'] = '3.5,' + str(zreio) +',28'
    params['many_tanh_xe'] = '-2,-1,'+str(max(x_e, 2e-4))
    params['many_tanh_width'] = dz
    params['reionization_z_start_max'] = zstartmax


    
    params['hyper_flat_approximation_nu'] = 7000. # The higher this is, the more exact
    params['hyper_flat_approximation_nu'] = 1e6 # The higher this is, the more exact
    params['transfer_neglect_delta_k_S_t0'] = 0.0017 # The lower these are, the more exact
    params['transfer_neglect_delta_k_S_t1'] = 0.0005
    params['transfer_neglect_delta_k_S_t2'] = 0.0017
    params['transfer_neglect_delta_k_S_e'] = 0.0013
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
    params['hyper_flat_approximation_nu'] = 1e6 # The higher this is, the more exact
    params['transfer_neglect_delta_k_S_t0'] = 0.0017 # The lower these are, the more exact
    params['transfer_neglect_delta_k_S_t1'] = 0.0005
    params['transfer_neglect_delta_k_S_t2'] = 0.0017
    params['transfer_neglect_delta_k_S_e'] = 0.0013
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

def lnprob_EE_ell(zre, x_e, Clhat, N_l=0):
    # This returns log(P)
    ell, Cl, TE = get_spectra(zre, x_e, lmax=len(Clhat)-1, spectra=True)
    Cl += N_l
    chi2_ell = (2*ell[2:]+1)*(Clhat[2:]/Cl[2:] + np.log(Cl[2:]) - np.log(Clhat[2:])-1)
    chi2_ell = np.insert(chi2_ell, [0,0], 0)
    return -chi2_ell/2

def lnprob_wish_ell(zre, x_e, Clhat, N_lT=0, N_lE=0):
    # This returns log(P)
    TThat, EEhat, BBhat, TEhat, TBhat, EBhat = Clhat
    ell, ee, te, tt = get_spectra(zre, x_e,  spectra=True, lmax=len(EEhat)-1, all_spectra=True)

    tt += N_lT
    ee += N_lE
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
        N_lT=0, N_lE=0, test=False, test2=False):
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
        if l == 10:
            print('\n')
            print('l==10')
            print(Cl)
            print(Clinv)
            print(zz, xz, xx)
            print('\n')
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


if __name__ == '__main__':
    #test_bin()
    #test_plotalm()
    #test_BB()
    #test_binning()
    test_spectra()

