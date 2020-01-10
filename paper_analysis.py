import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (4,3)
mpl.rcParams['figure.dpi'] = 300
#mpl.rcParams['axes.labelsize'] = 'medium'
#mpl.rcParams['xtick.labelsize'] = 8
#mpl.rcParams['ytick.labelsize'] = 8

from matplotlib.ticker import (FormatStrFormatter, AutoMinorLocator)

from scipy.optimize import curve_fit
from scipy.integrate import trapz
from scipy.interpolate import interp1d

import healpy as hp

from getdist.gaussian_mixtures import GaussianND
from getdist import plots

from classy import Class
import functools

from tools import (twinplot, get_tau, get_tau_z, get_twotau, get_spectra,
        lnprob_EE_ell, lnprob_wish_ell, lnprob_TE_ell,
        lnprob_EE_ell_tau, lnprob_wish_ell_tau, lnprob_TE_ell_tau,
        get_spectra_tau,
        get_F_ell, get_F_ell_forcezsplit, get_F_ell_2, get_F_ell_3,
        get_F_ell_4, get_F_ell_5, get_F_ell_tau, get_spectra_simple,
        get_F_ell_3_tau)

from classtools.users.djw.tools import (med_subtract, spice_wrap, bin_spectra,
        bin_noisy_spectra, twinx_whitenoise, read_classmap,
        get_TT, get_TE, get_EE, rotate_map, alpha)

lmax = 30

def get_camb_EE(tau):
    import camb
    from camb import model, initialpower
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0,
            tau=tau)
    pars.InitPower.set_params(As=2e-9*np.exp(2*tau), ns=0.965, r=0)
    pars.set_for_lmax(400, lens_potential_accuracy=4);
    pars.set_accuracy(AccuracyBoost=1)
    pars.Accuracy.SourcekAccuracyBoost=1
    pars.Accuracy.IntkAccuracyBoost=1
    pars.Accuracy.lSampleBoost=1
    pars.Accuracy.AccuratePolarization=True
    pars.Accuracy.AccurateReionization=True
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    tot_Cl = powers['total']
    EE = tot_Cl[:,1]
    ell = np.arange(len(EE))
    Z = ell*(ell+1)/(2*np.pi)
    return ell, EE/Z

def get_camb_TE(tau):
    import camb
    from camb import model, initialpower
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0,
            tau=tau)
    pars.InitPower.set_params(As=2e-9*np.exp(2*tau), ns=0.965, r=0)
    pars.set_for_lmax(400, lens_potential_accuracy=4);
    pars.set_accuracy(AccuracyBoost=1)
    pars.Accuracy.SourcekAccuracyBoost=1
    pars.Accuracy.IntkAccuracyBoost=1
    pars.Accuracy.lSampleBoost=1
    pars.Accuracy.AccuratePolarization=True
    pars.Accuracy.AccurateReionization=True
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    tot_Cl = powers['total']
    TE = tot_Cl[:,3]
    ell = np.arange(len(TE))
    Z = ell*(ell+1)/(2*np.pi)
    return ell, TE/Z

def cartoons(tau=0.06):
    ell, EEtau = get_camb_EE(tau)
    ell, EE0 = get_camb_EE(0.00)

    reio = EEtau - EE0
    reco = EE0

    l = np.logspace(np.log10(2), np.log10(ell.max()), 1000)

    f0 = interp1d(ell[2:], EEtau[2:], kind='quadratic')
    f1 = interp1d(ell[2:], reio[2:], kind='quadratic')
    f2 = interp1d(ell[2:], reco[2:], kind='quadratic')

    plt.loglog(l, f0(l), 'k--', lw=0.9, zorder=3)
    plt.loglog(l, f1(l), 'C0')
    plt.loglog(l, f2(l), 'C1')
    #plt.loglog(ell[2:], reio[2:], 'o', 'C0')
    #plt.loglog(ell[2:], reco[2:], 'o', 'C1')


    plt.text(3, 4e-2, 'Reionization', color='C0')
    plt.text(30, 1e-3, 'Recombination', color='C1')
    plt.savefig('cartoon1.pdf', bbox_inches='tight')
    plt.xlim(xmax=180)
    plt.savefig('cartoon0.pdf', bbox_inches='tight')
    plt.ylim([1.5e-6, 1e-1])
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_\ell^\mathrm{EE}\ [\mathrm{\mu K^2\,sr}]$')
    ax = plt.gca()
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)

    plt.savefig('cartoon1.pdf', bbox_inches='tight')
    plt.savefig('cartoon1.png', bbox_inches='tight')
    plt.close()


    ell, TEtau = get_camb_TE(tau)
    ell, TE0 = get_camb_TE(0.00)

    reio = TEtau - TE0
    reco = TE0

    l = np.logspace(np.log10(2), np.log10(ell.max()), 1000)

    f0 = interp1d(ell[2:], TEtau[2:], kind='quadratic')
    f1 = interp1d(ell[2:], reio[2:], kind='quadratic')
    f2 = interp1d(ell[2:], reco[2:], kind='quadratic')

    plt.loglog(l, f0(l), 'k--', lw=0.9, zorder=3)
    plt.loglog(l, f1(l), 'C0')
    plt.loglog(l, f2(l), 'C1')
    plt.savefig('cartoon1.5.pdf', bbox_inches='tight')
    plt.savefig('cartoon1.5.png', bbox_inches='tight')
    plt.close()


    fontsize = 12
    z, xe, ell, EE, TE = get_spectra(7.5, 0.2, both=True)

    #fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(4,5))
    fig0, ax0 = plt.subplots(1)
    ax0.plot(z, xe, color='k')
    ax0.set_xlabel(r'$z$')
    ax0.set_ylabel(r'$x_e(z)$')

    ax0.annotate(r'$\boldsymbol{z_\mathrm{re}}$',
            xy=(7.5, 0.6), xytext=(7.5,-0.15), ha='center', va='center',
            arrowprops=dict(facecolor='C2',edgecolor='C2',
                arrowstyle="-|>", shrinkA=0.05, lw=2),
            color='C2', fontsize=fontsize)

    ax0.text(14, 0.08, r'$\boldsymbol{x_e^\mathrm{min}}$', va='center',
    color='C0', fontsize=fontsize)
    ax0.annotate('', xy=(19, 0.2), xytext=(19,0), \
            arrowprops=dict(facecolor='C0',edgecolor='C0',
                arrowstyle="<|-|>", shrinkA=0.05,
                shrinkB=0.05, lw=2),
            fontsize=fontsize)


    ax0.text(25.5, 0.1-0.02, r'$\boldsymbol{\delta z_\mathrm{re}}$', ha='center', va='center',
            color='C8', fontsize=fontsize)
    ax0.annotate('', xy=(27, 0.1), xytext=(29,0.1), \
            arrowprops=dict(facecolor='C8',edgecolor='C8',
                arrowstyle="<|-|>", shrinkA=0.05,
                shrinkB=0.05, mutation_scale=5),
            fontsize=fontsize)

    ax0.text(5, 0.7-0.02, r'$\boldsymbol{\delta z_\mathrm{re}}$', ha='center',
            color='C8', va='center', fontsize=fontsize)
    ax0.annotate('', xy=(6.5, 0.7), xytext=(8.5,0.7), \
            arrowprops=dict(facecolor='C8',edgecolor='C8',
                arrowstyle="<|-|>", shrinkA=0.01,
                shrinkB=0.01, mutation_scale=5),
            fontsize=fontsize)

    ax0.annotate(r'$\boldsymbol{z_\mathrm t}$',
            xy=(28, 0.1), xytext=(28,-0.15), ha='center', va='center',
            arrowprops=dict(facecolor='C3',edgecolor='C3',
                arrowstyle="-|>", shrinkA=0.05, lw=2),
            color='C3', fontsize=fontsize)


    ax0.set_xlim([0, 35])
    ax0.set_ylim([0, 1.3])

    ax0.set_xticks([],[])
    #plt.yticks([],[])
    ax0.set_yticks([0, 0.5, 1], [0, 0.5, 1])
    ax0.yaxis.set_minor_locator(AutoMinorLocator(5))
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)

    #plt.subplots_adjust(hspace=0.2)


    fig0.savefig('cartoon2.pdf', bbox_inches='tight')
    fig0.savefig('cartoon2.png', bbox_inches='tight')


    import pandas
    names=['z_l', 'z', 'z_u', 'xe_l', 'x_e', 'xe_u', 'technique', 'ref']
    dtype = dict()
    for i in range(6):
        dtype[names[i]] = np.float32
    for i in range(6,8):
        dtype[names[i]] = str
    d = pandas.read_csv('direct_observations.csv', header=1, names=names,
                       dtype=dtype, verbose=True, skipinitialspace=True)
    
    d = d.sort_values(['technique', 'z'])
    
    i1 = ~np.isnan(d['xe_l']) & ~np.isnan(d['x_e']) & ~np.isnan(d['z'])
    i2 = np.isnan(d['x_e']) & ~np.isnan(d['xe_u']) & ~np.isnan(d['z'])
    i3 = np.isnan(d['x_e']) & ~np.isnan(d['xe_l'])
    i4 = np.isnan(d['x_e']) & ~np.isnan(d['xe_u']) & np.isnan(d['z'])
    i5 = ~np.isnan(d['xe_l']) & ~np.isnan(d['x_e']) & np.isnan(d['z'])
    #d.loc[d['xe_l'] is np.nan]
    d1 = d[i1]
    d2 = d[i2]
    d3 = d[i3]
    d4 = d[i4]
    d5 = d[i5]
    alpha = 1

    #fig1, ax1 = plt.subplots(1)
    z, xe, ell, EE, TE = get_spectra(6.75, 0.05, both=True)
    ax1 = ax0.inset_axes([0.45, 0.45, 0.55, 0.55], facecolor='0.9')

    f = ax1.figure
    f.set_facecolor('0.9')
    b = ax0.transData.inverted().transform(ax1.get_tightbbox(fig0.canvas.get_renderer()))
    ax0.add_patch(mpl.patches.Rectangle( (0.95*b[0][0], 0.95*b[0][1]), 
        b[1][0]-0.95*b[0][0], b[1][1]-0.95*b[0][1], facecolor='0.9', edgecolor='0.8'))

    ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    ms = 5
    lw = 1
    ax1.errorbar(d1['z'], 1.08*d1['x_e'], 
            xerr=[1.08*d1['z_l'], 1.08*d1['z_u']],
            yerr=[1.08*d1['xe_l'], 1.08*d1['xe_u']], 
            fmt='.', color='C0', zorder=6, ms=ms, elinewidth=lw)
    ax1.errorbar(d2['z'], 1.08*d2['xe_u'], yerr=0.2, uplims=True,
            fmt='.', color='C1', alpha=alpha, ms=ms, elinewidth=lw)
    ax1.errorbar(d3['z'], 1.08*d3['xe_l'], yerr=0.2, lolims=True,
            fmt='.', color='C2', alpha=alpha, ms=ms, elinewidth=lw)
    ax1.errorbar((d4['z_l']+d4['z_u'])/2, 1.08*d4['xe_u'], xerr=(d4['z_l']-d4['z_u'])/2, 
                 yerr=0.2, uplims=True, fmt='.', color='C1',
                 alpha=alpha, ms=ms, elinewidth=lw)
    ax1.errorbar((d5['z_l']+d5['z_u'])/2, 1.08*d5['x_e'], xerr=(d5['z_l']-d5['z_u'])/2,
                yerr = [1.08*d5['xe_l'], 1.08*d5['xe_u']], fmt='.', color='C0',
                ms=ms, elinewidth=lw)


    ax1.set_yticks([0, 0.5, 1], [0, 0.5, 1])
    ax1.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax1.plot(z, xe, color='k', zorder=-5)
    #ax1.set_xlabel(r'$z$')
    #ax1.set_ylabel(r'$x_e(z)$')

    ax1.set_xlim([5.1, 8.2])
    ax1.set_ylim([0, 1.3])

    fig0.savefig('cartoon2.5.pdf', bbox_inches='tight')
    fig0.savefig('cartoon2.5.png', bbox_inches='tight')



    plt.close()



    return

def fig1(num=50):
    lw = 1
    ymin = 3.2e-5
    ymax = 1.2e-1

    zs = np.linspace(6, 10, num)
    xes = np.linspace(0, 0.2, num)
    z0 = 6
    xe0 = 0
    
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12,3), 
                             gridspec_kw={"width_ratios":[1,1,1,0.05], "wspace":0.6})
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    cax = axes[3]
    axs2 = [ax2, ax2.twinx()]
    axs3 = [ax3, ax3.twinx()]
    for i in range(len(zs))[::-1]:
        z, xe, ell, EE, TE = get_spectra(zs[i], xe0, both=True)
        thermo = get_spectra(zs[i], xe0, therm=True)
        ax1.plot(z, xe, color=plt.cm.magma(i/len(zs)), lw=lw)
        ax1.set_xlim([0, 40])
        ax1.set_xlabel(r'$z$')
        ax1.set_ylabel(r'$x_e(z)$')
        twinplot(ell, EE, linestyle='-', marker=None, color=plt.cm.magma(i/len(zs)), 
                 axes=axs2, spec='EE', lw=lw, ymin=ymin, ymax=ymax, ylabels_r=False)
        twinplot(ell, TE, linestyle='-', marker=None, color=plt.cm.magma(i/len(zs)), 
                 axes=axs3, spec='TE', lw=lw, ymin=5.5e-3, ymax=5.5)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.magma, norm=plt.Normalize(vmin=zs.min(), vmax=zs.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$z_\mathrm{re}$')
    
    plt.tight_layout()
    plt.savefig('f1_zre.pdf', bbox_inches='tight')
    plt.savefig('f1_zre.png', bbox_inches='tight')
    plt.close('all')
    
        
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3), 
                             gridspec_kw={"width_ratios":[1,1,1,0.05], "wspace":0.6})
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    cax = axes[3]
    axs2 = [ax2, ax2.twinx()]
    axs3 = [ax3, ax3.twinx()]
    for i in range(len(zs))[::-1]:
        z, xe, ell, EE, TE = get_spectra(z0, xes[i], both=True)
        thermo = get_spectra(z0, xes[i], therm=True)
        _, taulo, tauhi = get_twotau(thermo, zmax=100, xmin=2e-4)
    
        ax1.plot(z, xe, color=plt.cm.plasma(i/len(zs)), lw=lw)
        ax1.set_xlim([0, 40])
        ax1.set_xlabel(r'$z$')
        ax1.set_ylabel(r'$x_e$')
        twinplot(ell, EE, linestyle='-', marker=None, color=plt.cm.plasma(i/len(zs)), 
                 axes=axs2, spec='EE', lw=lw, ymin=ymin, ymax=ymax, ylabels_r=False)
        twinplot(ell, TE, linestyle='-', marker=None, color=plt.cm.plasma(i/len(zs)), 
                 axes=axs3, spec='TE', lw=lw, ymin=5.5e-3, ymax=5.5)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=xes.min(), vmax=xes.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$x_e^\mathrm{min}$', ticks=[0, 0.1, 0.2])
    
    #plt.tight_layout()
    plt.savefig('f1_xmin.pdf', bbox_inches='tight')
    plt.savefig('f1_xmin.png', bbox_inches='tight')
    plt.close('all')
    
    
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3), 
                             gridspec_kw={"width_ratios":[1,1,1,0.05], "wspace":0.6})
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    cax = axes[3]
    axs2 = [ax2, ax2.twinx()]
    axs3 = [ax3, ax3.twinx()]
    dzs = np.linspace(0.5, 10, len(xes))
    #z0 = 8
    for i in range(len(dzs))[::-1]:
        #z, xe, ell, EE, TE = get_spectra(z0, xe0, dz=dzs[i], both=True,
                #zstartmax=200)
        z, xe, ell, EE, TE = get_spectra_simple(z0, xe0, dz=dzs[i], both=True,
                zstartmax=200)
        #thermo = get_spectra(z0, xe0, dz=dzs[i], therm=True)
        #_, taulo, tauhi = get_twotau(thermo, zmax=100, xmin=2e-4)
    
        ax1.plot(z, xe, color=plt.cm.inferno(i/len(dzs)),lw=lw)
        ax1.set_xlim([0, 40])
        ax1.set_xlabel(r'$z$')
        ax1.set_ylabel(r'$x_e$')
        twinplot(ell, EE, linestyle='-', marker=None, color=plt.cm.inferno(i/len(dzs)), 
                 axes=axs2, spec='EE', lw=lw, ymin=ymin, ymax=ymax, ylabels_r=False, ylabels_l=True)
        twinplot(ell, TE, linestyle='-', marker=None, color=plt.cm.inferno(i/len(dzs)), 
                 axes=axs3, spec='TE', lw=lw, ymin=5.5e-3, ymax=5.5, ylabels_l=True)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmin=dzs.min(), vmax=dzs.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$\delta z_\mathrm{re}$')
    #plt.tight_layout()
    plt.savefig('f1_dz.pdf', bbox_inches='tight')
    plt.savefig('f1_dz.png', bbox_inches='tight')
    plt.close('all')

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3), 
                             gridspec_kw={"width_ratios":[1,1,1,0.05], "wspace":0.6})
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    cax = axes[3]
    axs2 = [ax2, ax2.twinx()]
    axs3 = [ax3, ax3.twinx()]
    dzs = np.linspace(0.5, 10, len(xes))
    zts = np.linspace(20, 35, len(xes))
    #z0 = 8
    for i in range(len(zts))[::-1]:
        #z, xe, ell, EE, TE = get_spectra(z0, xe0, dz=dzs[i], both=True,
                #zstartmax=200)
        z, xe, ell, EE, TE = get_spectra(z0, 0.1, z_t=zts[i], both=True,
                zstartmax=200)
        #thermo = get_spectra(z0, xe0, dz=dzs[i], therm=True)
        #_, taulo, tauhi = get_twotau(thermo, zmax=100, xmin=2e-4)
    
        ax1.plot(z, xe, color=plt.cm.cividis(i/len(dzs)),lw=lw)
        twinplot(ell, EE, linestyle='-', marker=None, color=plt.cm.cividis(i/len(dzs)), 
                 axes=axs2, spec='EE', lw=lw, ymin=ymin, ymax=ymax,
                 ylabels_r=False, ylabels_l=False)
        twinplot(ell, TE, linestyle='-', marker=None, color=plt.cm.cividis(i/len(dzs)), 
                 axes=axs3, spec='TE', lw=lw, ymin=5.5e-3, ymax=5.5,
                 ylabels_l=False, ylabels_r=False)

    ax1.set_xlim([0, 40])
    ax1.minorticks_on()
    ax1.set_xlabel(r'$z$')
    ax1.set_ylabel(r'$x_e$')
    ax2.set_title(r'$C_\ell^\mathrm{EE}$')
    ax3.set_title(r'$C_\ell^\mathrm{TE}$')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis, norm=plt.Normalize(vmin=zts.min(), vmax=zts.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$z_\mathrm t$')
    #plt.tight_layout()
    plt.savefig('f1_zt.pdf', bbox_inches='tight')
    plt.savefig('f1_zt.png', bbox_inches='tight')
    plt.close('all')
    return


def fig1_alt(num=50):
    lw = 1
    ymin = 3.2e-5
    ymax = 1.2e-1

    zs = np.linspace(6, 10, num)
    xes = np.linspace(0, 0.2, num)
    z0 = 6
    xe0 = 0
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9,3), 
                             gridspec_kw={"width_ratios":[1,1,0.05], "wspace":0.6})
    ax1 = axes[0]
    ax2 = axes[1]
    cax = axes[2]
    axs2 = [ax2, ax2.twinx()]
    for i in range(len(zs))[::-1]:
        z, xe, ell, EE, TE = get_spectra(zs[i], xe0, both=True)
        thermo = get_spectra(zs[i], xe0, therm=True)
        ax1.plot(z, xe, color=plt.cm.magma(i/len(zs)), lw=lw)
        ax1.set_xlim([0, 40])
        ax1.set_xlabel(r'$z$')
        ax1.set_ylabel(r'$x_e$')
        twinplot(ell, EE, linestyle='-', marker=None, color=plt.cm.magma(i/len(zs)), 
                 axes=axs2, spec='EE', lw=lw, ymin=ymin, ymax=ymax,
                 ylabels_r=True)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.magma, norm=plt.Normalize(vmin=zs.min(), vmax=zs.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$z_\mathrm{re}$')
    
    plt.tight_layout()
    plt.savefig('f1_zre_alt.pdf', bbox_inches='tight')
    plt.savefig('f1_zre_alt.png', bbox_inches='tight')
    plt.close('all')
    
        
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), 
                             gridspec_kw={"width_ratios":[1,1,0.05], "wspace":0.6})
    ax1 = axes[0]
    ax2 = axes[1]
    cax = axes[2]
    axs2 = [ax2, ax2.twinx()]
    for i in range(len(zs))[::-1]:
        z, xe, ell, EE, TE = get_spectra(z0, xes[i], both=True)
        thermo = get_spectra(z0, xes[i], therm=True)
        _, taulo, tauhi = get_twotau(thermo, zmax=100, xmin=2e-4)
    
        ax1.plot(z, xe, color=plt.cm.plasma(i/len(zs)), lw=lw)
        ax1.set_xlim([0, 40])
        ax1.set_xlabel(r'$z$')
        ax1.set_ylabel(r'$x_e$')
        twinplot(ell, EE, linestyle='-', marker=None, color=plt.cm.plasma(i/len(zs)), 
                 axes=axs2, spec='EE', lw=lw, ymin=ymin, ymax=ymax,
                 ylabels_r=True)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=xes.min(), vmax=xes.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$x_e^\mathrm{min}$', ticks=[0, 0.1, 0.2])
    
    #plt.tight_layout()
    plt.savefig('f1_xmin_alt.pdf', bbox_inches='tight')
    plt.savefig('f1_xmin_alt.png', bbox_inches='tight')
    plt.close('all')
    
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), 
                             gridspec_kw={"width_ratios":[1,1,0.05], "wspace":0.6})
    ax1 = axes[0]
    ax2 = axes[1]
    cax = axes[2]
    axs2 = [ax2, ax2.twinx()]
    dzs = np.linspace(0.5, 10, len(xes))
    #z0 = 8
    for i in range(len(dzs))[::-1]:
        #z, xe, ell, EE, TE = get_spectra(z0, xe0, dz=dzs[i], both=True,
                #zstartmax=200)
        z, xe, ell, EE, TE = get_spectra_simple(z0, xe0, dz=dzs[i], both=True,
                zstartmax=200)
        #thermo = get_spectra(z0, xe0, dz=dzs[i], therm=True)
        #_, taulo, tauhi = get_twotau(thermo, zmax=100, xmin=2e-4)
    
        ax1.plot(z, xe, color=plt.cm.inferno(i/len(dzs)),lw=lw)
        ax1.set_xlim([0, 40])
        ax1.set_xlabel(r'$z$')
        ax1.set_ylabel(r'$x_e$')
        twinplot(ell, EE, linestyle='-', marker=None, color=plt.cm.inferno(i/len(dzs)), 
                 axes=axs2, spec='EE', lw=lw, ymin=ymin, ymax=ymax,
                 ylabels_r=True, ylabels_l=True)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmin=dzs.min(), vmax=dzs.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$\delta z_\mathrm{re}$')
    #plt.tight_layout()
    plt.savefig('f1_dz_alt.pdf', bbox_inches='tight')
    plt.savefig('f1_dz_alt.png', bbox_inches='tight')
    plt.close('all')

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), 
                             gridspec_kw={"width_ratios":[1,1,0.05], "wspace":0.6})
    ax1 = axes[0]
    ax2 = axes[1]
    cax = axes[2]
    axs2 = [ax2, ax2.twinx()]
    dzs = np.linspace(0.5, 10, len(xes))
    zts = np.linspace(20, 35, len(xes))
    #z0 = 8
    for i in range(len(zts))[::-1]:
        #z, xe, ell, EE, TE = get_spectra(z0, xe0, dz=dzs[i], both=True,
                #zstartmax=200)
        z, xe, ell, EE, TE = get_spectra(z0, 0.1, z_t=zts[i], both=True,
                zstartmax=200)
        #thermo = get_spectra(z0, xe0, dz=dzs[i], therm=True)
        #_, taulo, tauhi = get_twotau(thermo, zmax=100, xmin=2e-4)
    
        ax1.plot(z, xe, color=plt.cm.cividis(i/len(dzs)),lw=lw)
        twinplot(ell, EE, linestyle='-', marker=None, color=plt.cm.cividis(i/len(dzs)), 
                 axes=axs2, spec='EE', lw=lw, ymin=ymin, ymax=ymax,
                 ylabels_r=True, ylabels_l=False)

    ax1.set_xlim([0, 40])
    ax1.minorticks_on()
    ax1.set_xlabel(r'$z$')
    ax1.set_ylabel(r'$x_e$')
    ax2.set_title(r'$C_\ell^\mathrm{EE}$')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis, norm=plt.Normalize(vmin=zts.min(), vmax=zts.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$z_\mathrm t$')
    #plt.tight_layout()
    plt.savefig('f1_zt_alt.pdf', bbox_inches='tight')
    plt.savefig('f1_zt_alt.png', bbox_inches='tight')
    plt.close('all')
    return

def fig1_big(num=50):
    lw = 1
    ymin = 3.2e-5
    ymax = 1.2e-1

    zs = np.linspace(6, 10, num)
    xes = np.linspace(0, 0.2, num)
    z0 = 6
    xe0 = 0
    
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12,9), 
                             gridspec_kw={"width_ratios":[1,1,1,0.05], "wspace":0.6})
    ax1 = axes[0][0]
    ax2 = axes[0][1]
    ax3 = axes[0][2]
    cax = axes[0][3]
    axs2 = [ax2, ax2.twinx()]
    axs3 = [ax3, ax3.twinx()]
    for i in range(len(zs))[::-1]:
        z, xe, ell, EE, TE = get_spectra(zs[i], xe0, both=True)
        thermo = get_spectra(zs[i], xe0, therm=True)
        ax1.plot(z, xe, color=plt.cm.magma(i/len(zs)), lw=lw)
        ax1.set_xlim([0, 40])
        ax1.set_xlabel(r'$z$')
        ax1.set_ylabel(r'$x_e$')
        twinplot(ell, EE, linestyle='-', marker=None, color=plt.cm.magma(i/len(zs)), 
                 axes=axs2, spec='EE', lw=lw, ymin=ymin, ymax=ymax, ylabels_r=False)
        twinplot(ell, TE, linestyle='-', marker=None, color=plt.cm.magma(i/len(zs)), 
                 axes=axs3, spec='TE', lw=lw, ymin=5.5e-3, ymax=5.5)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.magma, norm=plt.Normalize(vmin=zs.min(), vmax=zs.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$z_\mathrm{re}$')
    
    #plt.tight_layout()
    #plt.savefig('f1_zre.pdf', bbox_inches='tight')
    #plt.savefig('f1_zre.png', bbox_inches='tight')
    #plt.close('all')
    
        
    ax1 = axes[1][0]
    ax2 = axes[1][1]
    ax3 = axes[1][2]
    cax = axes[1][3]
    axs2 = [ax2, ax2.twinx()]
    axs3 = [ax3, ax3.twinx()]
    for i in range(len(zs))[::-1]:
        z, xe, ell, EE, TE = get_spectra(z0, xes[i], both=True)
        thermo = get_spectra(z0, xes[i], therm=True)
        _, taulo, tauhi = get_twotau(thermo, zmax=100, xmin=2e-4)
    
        ax1.plot(z, xe, color=plt.cm.plasma(i/len(zs)), lw=lw)
        ax1.set_xlim([0, 40])
        ax1.set_xlabel(r'$z$')
        ax1.set_ylabel(r'$x_e$')
        twinplot(ell, EE, linestyle='-', marker=None, color=plt.cm.plasma(i/len(zs)), 
                 axes=axs2, spec='EE', lw=lw, ymin=ymin, ymax=ymax, ylabels_r=False)
        twinplot(ell, TE, linestyle='-', marker=None, color=plt.cm.plasma(i/len(zs)), 
                 axes=axs3, spec='TE', lw=lw, ymin=5.5e-3, ymax=5.5)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=xes.min(), vmax=xes.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$x_e^\mathrm{min}$', ticks=[0, 0.1, 0.2])
    
    #plt.tight_layout()
    #plt.savefig('f1_xmin.pdf', bbox_inches='tight')
    #plt.savefig('f1_xmin.png', bbox_inches='tight')
    #plt.close('all')
    
    
    ax1 = axes[2][0]
    ax2 = axes[2][1]
    ax3 = axes[2][2]
    cax = axes[2][3]
    axs2 = [ax2, ax2.twinx()]
    axs3 = [ax3, ax3.twinx()]
    dzs = np.linspace(0.5, 10, len(xes))
    #z0 = 8
    for i in range(len(dzs))[::-1]:
        #z, xe, ell, EE, TE = get_spectra(z0, xe0, dz=dzs[i], both=True,
                #zstartmax=200)
        z, xe, ell, EE, TE = get_spectra_simple(z0, xe0, dz=dzs[i], both=True,
                zstartmax=200)
        #thermo = get_spectra(z0, xe0, dz=dzs[i], therm=True)
        #_, taulo, tauhi = get_twotau(thermo, zmax=100, xmin=2e-4)
    
        ax1.plot(z, xe, color=plt.cm.inferno(i/len(dzs)),lw=lw)
        ax1.set_xlim([0, 40])
        ax1.set_xlabel(r'$z$')
        ax1.set_ylabel(r'$x_e$')
        twinplot(ell, EE, linestyle='-', marker=None, color=plt.cm.inferno(i/len(dzs)), 
                 axes=axs2, spec='EE', lw=lw, ymin=ymin, ymax=ymax, ylabels_r=False, ylabels_l=True)
        twinplot(ell, TE, linestyle='-', marker=None, color=plt.cm.inferno(i/len(dzs)), 
                 axes=axs3, spec='TE', lw=lw, ymin=5.5e-3, ymax=5.5, ylabels_l=True)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmin=dzs.min(), vmax=dzs.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$\delta z_\mathrm{re}$')
    plt.tight_layout()
    #plt.savefig('f1_dz.pdf', bbox_inches='tight')
    #plt.savefig('f1_dz.png', bbox_inches='tight')
    #plt.close('all')

    plt.savefig('f1_big.pdf', bbox_inches='tight')

    return

def fig1_transp(num=50):
    lw = 1
    ymin = 3.2e-5
    ymax = 1.2e-1

    zs = np.linspace(6, 10, num)
    xes = np.linspace(0, 0.2, num)
    z0 = 6
    xe0 = 0
    
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8,8), 
                             gridspec_kw={"height_ratios":[0.05, 1,1,1], 
                                          "width_ratios": [1,1,1],
                                          "wspace":0.2, 'hspace':0.2})
    ax1 = axes[1][0]
    ax2 = axes[2][0]
    ax3 = axes[3][0]
    cax = axes[0][0]
    axs2 = [ax2, ax2.twinx()]
    axs3 = [ax3, ax3.twinx()]
    c1 = plt.cm.magma
    c2 = plt.cm.copper
    c3 = plt.cm.cividis
    for i in range(len(zs))[::-1]:
        z, xe, ell, EE, TE = get_spectra(zs[i], xe0, both=True)
        thermo = get_spectra(zs[i], xe0, therm=True)
        ax1.plot(z, xe, color=c1(i/len(zs)), lw=lw)
        ax1.set_xlim([0, 30])
        ax1.minorticks_on()
        ax1.set_xlabel(r'$z$', labelpad=-7)
        ax1.set_ylabel(r'$x_e$')
        twinplot(ell, EE, linestyle='-', marker=None, color=c1(i/len(zs)), 
                 axes=axs2, spec='EE', lw=lw, ymin=ymin, ymax=ymax, ylabels_r=False)
        twinplot(ell, TE, linestyle='-', marker=None, color=c1(i/len(zs)), 
                 axes=axs3, spec='TE', lw=lw, ymin=5.5e-4, ymax=5.5,
                 ylabels_r=False)
        if i == 0:
            twinplot(ell, -TE, linestyle='--', marker=None, color=c1(i/len(zs)), 
                     axes=axs3, spec='TE', lw=lw, ymin=5.5e-4, ymax=5.5,
                     ylabels_r=False)
        axs2[1].set_yticklabels([])
        axs2[0].set_xticklabels([])
        axs3[1].set_yticklabels([])
        axs2[0].set_xlabel('')
    sm = plt.cm.ScalarMappable(cmap=c1, norm=plt.Normalize(vmin=zs.min(), vmax=zs.max()))
    sm._A = []
    fig.colorbar(sm, orientation='horizontal', cax=cax, label=r'$z_\mathrm{re}$')
    cax.xaxis.set_label_position('top')
    cax.xaxis.set_ticks_position('top')
    l, b, w, h, = cax.get_position().bounds
    cax.set_position([l, b+0.02, w, h])
    l, b, w, h, = ax1.get_position().bounds
    ax1.set_position([l, b+0.04, w, h])

    
    
        
    ax1 = axes[1][1]
    ax2 = axes[2][1]
    ax3 = axes[3][1]
    cax = axes[0][1]
    axs2 = [ax2, ax2.twinx()]
    axs3 = [ax3, ax3.twinx()]
    for i in range(len(zs))[::-1]:
        z, xe, ell, EE, TE = get_spectra(z0, xes[i], both=True)
        thermo = get_spectra(z0, xes[i], therm=True)
        _, taulo, tauhi = get_twotau(thermo, zmax=100, xmin=2e-4)
    
        ax1.plot(z, xe, color=c2(i/len(zs)), lw=lw)
        ax1.minorticks_on()
        ax1.set_xlim([0, 30])
        ax1.set_xlabel(r'$z$', labelpad=-7)
        #ax1.set_ylabel(r'$x_e$')
        ax1.set_yticklabels([])
        twinplot(ell, EE, linestyle='-', marker=None, color=c2(i/len(zs)), 
                 axes=axs2, spec='EE', lw=lw, ymin=ymin, ymax=ymax,
                 ylabels_r=False, ylabels_l=False)
        twinplot(ell, TE, linestyle='-', marker=None, color=c2(i/len(zs)), 
                 axes=axs3, spec='TE', lw=lw, ymin=5.5e-4, ymax=5.5,
                 ylabels_r=False, ylabels_l=False)
        if i == 0:
            twinplot(ell, -TE, linestyle='--', marker=None, color=c2(i/len(zs)), 
                     axes=axs3, spec='TE', lw=lw, ymin=5.5e-4, ymax=5.5,
                     ylabels_r=False, ylabels_l=False)
        axs2[1].set_yticklabels([])
        axs3[1].set_yticklabels([])
        axs2[0].set_yticklabels([])
        axs3[0].set_yticklabels([])
        axs2[0].set_xlabel('')
        axs2[0].set_xticklabels([])
    sm = plt.cm.ScalarMappable(cmap=c2, norm=plt.Normalize(vmin=xes.min(), vmax=xes.max()))
    sm._A = []
    fig.colorbar(sm, orientation='horizontal', cax=cax, label=r'$x_e^\mathrm{min}$', ticks=[0, 0.1, 0.2])
    cax.xaxis.set_label_position('top')
    cax.xaxis.set_ticks_position('top')
    l, b, w, h, = cax.get_position().bounds
    cax.set_position([l, b+0.02, w, h])
    l, b, w, h, = ax1.get_position().bounds
    ax1.set_position([l, b+0.04, w, h])
    
    
    
    ax1 = axes[1][2]
    ax2 = axes[2][2]
    ax3 = axes[3][2]
    cax = axes[0][2]
    axs2 = [ax2, ax2.twinx()]
    axs3 = [ax3, ax3.twinx()]
    dzs = np.linspace(0.5, 10, len(xes))
    #z0 = 8
    for i in range(len(dzs))[::-1]:
        #z, xe, ell, EE, TE = get_spectra(z0, xe0, dz=dzs[i], both=True,
                #zstartmax=200)
        z, xe, ell, EE, TE = get_spectra_simple(z0, xe0, dz=dzs[i], both=True,
                zstartmax=200)
        #thermo = get_spectra(z0, xe0, dz=dzs[i], therm=True)
        #_, taulo, tauhi = get_twotau(thermo, zmax=100, xmin=2e-4)
    
        ax1.plot(z, xe, color=c3(i/len(dzs)),lw=lw)
        ax1.minorticks_on()
        ax1.set_xlim([0, 30])
        ax1.set_xlabel(r'$z$', labelpad=-7)
        #ax1.set_ylabel(r'$x_e$')
        ax1.set_yticklabels([])
        twinplot(ell, EE, linestyle='-', marker=None, color=c3(i/len(dzs)), 
                 axes=axs2, spec='EE', lw=lw, ymin=ymin, ymax=ymax,
                 ylabels_r=True, ylabels_l=False)
        twinplot(ell, TE, linestyle='-', marker=None, color=c3(i/len(dzs)), 
                 axes=axs3, spec='TE', lw=lw, ymin=5.5e-4, ymax=5.5,
                 ylabels_l=False)
        if i == 0:
            twinplot(ell, -TE, linestyle='--', marker=None,
                    color=c3(i/len(dzs)), 
                     axes=axs3, spec='TE', lw=lw, ymin=5.5e-4, ymax=5.5,
                     ylabels_l=False)
        axs2[0].set_yticklabels([])
        axs3[0].set_yticklabels([])
        axs2[0].set_xlabel('')
        axs2[0].set_xticklabels([])
        axs2[1].yaxis.labelpad = 25
        axs3[1].yaxis.labelpad = 25

    sm = plt.cm.ScalarMappable(cmap=c3, norm=plt.Normalize(vmin=dzs.min(), vmax=dzs.max()))
    sm._A = []
    fig.colorbar(sm, orientation='horizontal', cax=cax, label=r'$\delta z_\mathrm{re}$', 
            ticks=[0.5, 5, 10])
    cax.xaxis.set_label_position('top')
    cax.xaxis.set_ticks_position('top')
    l, b, w, h, = cax.get_position().bounds
    cax.set_position([l, b+0.02, w, h])
    l, b, w, h, = ax1.get_position().bounds
    ax1.set_position([l, b+0.04, w, h])

    plt.savefig('f1_transp.pdf', bbox_inches='tight')


    return

def fig2_1(num=50, n_noises=30):
    '''
    z_re constraints only, EE and Wishart. What do I want this plot to look
    like? How about N_l^TT = 0, crank up N_l^EE.

    alphas increase as noise decreases
    C0 for Wishart, C1 for EE only.
    '''
    zres = np.linspace(6, 10, num)
    wps = np.logspace(0, np.log10(200), n_noises) 
    N_lEEs = (wps*np.pi/180/60)**2

    fig, axes = plt.subplots(nrows=1, ncols=2, 
                             gridspec_kw={"width_ratios":[1,0.05], "wspace":0.1})
    ax = axes[0]
    cax = axes[1]
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
            norm=mpl.colors.LogNorm(vmin=wps.min(), vmax=wps.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$w_p^{-1/2}$ [$\mathrm{\mu K\,arcmin}$]')

    lnPWs = []
    for i, N_lEE in enumerate(N_lEEs):
        print(i)
        ell, EE, TE, TT = get_spectra(8, 0, lmax=lmax, spectra=True, all_spectra=True)
        Clhat = np.array([TT, EE+N_lEE, EE*0, TE, EE*0, EE*0])
        lnPE = []
        lnPW = []
        for zre in zres:
            lnPE.append(lnprob_EE_ell(zre, 0, Clhat[1], N_l=N_lEE))
            lnPW.append(lnprob_wish_ell(zre, 0, Clhat, N_lT=0, N_lE=N_lEE))
        ax.plot(zres, np.exp(np.sum(lnPW, axis=1)),
                color=plt.cm.viridis(i/len(N_lEEs)), zorder=1/N_lEE)
        #ax.plot(zres, np.exp(np.sum(lnPW, axis=1)), color='C0',
        #        alpha=1-i/len(N_lEEs))
        #ax.plot(zres, np.exp(np.sum(lnPE, axis=1)), color='C1',
        #        alpha=1-i/len(N_lEEs))
        # lnPW is an array of length len(zres), 11.
        lnPWs.append(np.exp(np.sum(lnPW, axis=1)))
    lnPWs = np.array(lnPWs)
    print(lnPWs.shape)
    print(zres.shape)
    bla = np.vstack((zres, lnPWs))
    np.save('z_lnL', bla)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel(r'$z_\mathrm{re}$')
    ax.set_ylabel(r'$P(z_\mathrm{re}|x_e^\mathrm{min}=0,\{\hat{\boldsymbol C}_\ell\})$')
    fig.savefig('f2.pdf', bbox_inches='tight')
    fig.savefig('f2.png', bbox_inches='tight')
    plt.close('all')
    return

def fig2_2(num=50, n_noises=30):
    '''
    z_re constraints only, EE and Wishart. What do I want this plot to look
    like? How about N_l^TT = 0, crank up N_l^EE.

    alphas increase as noise decreases
    C0 for Wishart, C1 for EE only.
    '''
    zres = np.linspace(6, 10, num)
    wps = np.logspace(0, np.log10(2000), n_noises) 
    N_lEEs = (wps*np.pi/180/60)**2

    fig, axes = plt.subplots(nrows=1, ncols=2, 
                             gridspec_kw={"width_ratios":[1,0.05], "wspace":0.1})
    ax = axes[0]
    cax = axes[1]
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
            norm=mpl.colors.LogNorm(vmin=wps.min(), vmax=wps.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$w_p^{-1/2}$ [$\mathrm{\mu K\,arcmin}$]')

    lnPWs = []
    for i, N_lEE in enumerate(N_lEEs):
        print(i)
        ell, EE, TE, TT = get_spectra(8, 0, lmax=lmax, spectra=True, all_spectra=True)
        Clhat = np.array([TT, EE+N_lEE, EE*0, TE, EE*0, EE*0])
        lnPE = []
        lnPW = []
        for zre in zres:
            lnPE.append(lnprob_EE_ell(zre, 0, Clhat[1], N_l=N_lEE))
            lnPW.append(lnprob_wish_ell(zre, 0, Clhat, N_lT=0, N_lE=N_lEE))
        ax.plot(zres, np.exp(np.sum(lnPW, axis=1)),
                color=plt.cm.viridis(i/len(N_lEEs)), zorder=1/N_lEE)
        #ax.plot(zres, np.exp(np.sum(lnPW, axis=1)), color='C0',
        #        alpha=1-i/len(N_lEEs))
        #ax.plot(zres, np.exp(np.sum(lnPE, axis=1)), color='C1',
        #        alpha=1-i/len(N_lEEs))
        # lnPW is an array of length len(zres), 11.
        lnPWs.append(np.exp(np.sum(lnPW, axis=1)))
    lnPWs = np.array(lnPWs)
    print(lnPWs.shape)
    print(zres.shape)
    bla = np.vstack((zres, lnPWs))
    np.save('z_lnL', bla)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel(r'$z_\mathrm{re}$')
    ax.set_ylabel(r'$P(z_\mathrm{re}|x_e^\mathrm{min}=0,\{\hat{\boldsymbol C}_\ell\})$')
    fig.savefig('f2.pdf', bbox_inches='tight')
    fig.savefig('f2.png', bbox_inches='tight')
    plt.close('all')
    return

def fig1_alt2(num=50):
    lw = 1
    ymin = 3.2e-5
    ymax = 1.2e-1

    zs = np.linspace(6, 10, num)
    xes = np.linspace(0, 0.2, num)
    z0 = 6
    xe0 = 0
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9,3), 
                             gridspec_kw={"width_ratios":[1,1,0.05], "wspace":0.6})
    ax1 = axes[0]
    ax2 = axes[1]
    cax = axes[2]
    axs2 = [ax2, ax2.twinx()]
    for i in range(len(zs))[::-1]:
        z, xe, ell, EE, TE = get_spectra(zs[i], xe0, both=True)
        thermo = get_spectra(zs[i], xe0, therm=True)
        ax1.plot(z, xe, color=plt.cm.magma(i/len(zs)), lw=lw)
        ax1.set_xlim([0, 40])
        ax1.set_xlabel(r'$z$')
        ax1.set_ylabel(r'$x_e$')
        twinplot(ell, TE, linestyle='-', marker=None, color=plt.cm.magma(i/len(zs)), 
                 axes=axs2, spec='TE', lw=lw, ymin=5.5e-3, ymax=5.5,
                 ylabels_r=True)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.magma, norm=plt.Normalize(vmin=zs.min(), vmax=zs.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$z_\mathrm{re}$')
    
    plt.tight_layout()
    plt.savefig('f1_zre_alt_TE.pdf', bbox_inches='tight')
    plt.savefig('f1_zre_alt_TE.png', bbox_inches='tight')
    plt.close('all')
    
        
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), 
                             gridspec_kw={"width_ratios":[1,1,0.05], "wspace":0.6})
    ax1 = axes[0]
    ax2 = axes[1]
    cax = axes[2]
    axs2 = [ax2, ax2.twinx()]
    for i in range(len(zs))[::-1]:
        z, xe, ell, EE, TE = get_spectra(z0, xes[i], both=True)
        thermo = get_spectra(z0, xes[i], therm=True)
        _, taulo, tauhi = get_twotau(thermo, zmax=100, xmin=2e-4)
    
        ax1.plot(z, xe, color=plt.cm.plasma(i/len(zs)), lw=lw)
        ax1.set_xlim([0, 40])
        ax1.set_xlabel(r'$z$')
        ax1.set_ylabel(r'$x_e$')
        twinplot(ell, TE, linestyle='-', marker=None, color=plt.cm.magma(i/len(zs)), 
                 axes=axs2, spec='TE', lw=lw, ymin=5.5e-3, ymax=5.5,
                 ylabels_r=True)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=xes.min(), vmax=xes.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$x_e^\mathrm{min}$', ticks=[0, 0.1, 0.2])
    
    #plt.tight_layout()
    plt.savefig('f1_xmin_alt_TE.pdf', bbox_inches='tight')
    plt.savefig('f1_xmin_alt_TE.png', bbox_inches='tight')
    plt.close('all')
    
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), 
                             gridspec_kw={"width_ratios":[1,1,0.05], "wspace":0.6})
    ax1 = axes[0]
    ax2 = axes[1]
    cax = axes[2]
    axs2 = [ax2, ax2.twinx()]
    dzs = np.linspace(0.5, 10, len(xes))
    #z0 = 8
    for i in range(len(dzs))[::-1]:
        #z, xe, ell, EE, TE = get_spectra(z0, xe0, dz=dzs[i], both=True,
                #zstartmax=200)
        z, xe, ell, EE, TE = get_spectra_simple(z0, xe0, dz=dzs[i], both=True,
                zstartmax=200)
        #thermo = get_spectra(z0, xe0, dz=dzs[i], therm=True)
        #_, taulo, tauhi = get_twotau(thermo, zmax=100, xmin=2e-4)
    
        ax1.plot(z, xe, color=plt.cm.inferno(i/len(dzs)),lw=lw)
        ax1.set_xlim([0, 40])
        ax1.set_xlabel(r'$z$')
        ax1.set_ylabel(r'$x_e$')
        twinplot(ell, TE, linestyle='-', marker=None, color=plt.cm.magma(i/len(zs)), 
                 axes=axs2, spec='TE', lw=lw, ymin=5.5e-3, ymax=5.5,
                 ylabels_r=True)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmin=dzs.min(), vmax=dzs.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$\delta z_\mathrm{re}$')
    #plt.tight_layout()
    plt.savefig('f1_dz_alt_TE.pdf', bbox_inches='tight')
    plt.savefig('f1_dz_alt_TE.png', bbox_inches='tight')
    plt.close('all')

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), 
                             gridspec_kw={"width_ratios":[1,1,0.05], "wspace":0.6})
    ax1 = axes[0]
    ax2 = axes[1]
    cax = axes[2]
    axs2 = [ax2, ax2.twinx()]
    dzs = np.linspace(0.5, 10, len(xes))
    zts = np.linspace(20, 35, len(xes))
    #z0 = 8
    for i in range(len(zts))[::-1]:
        #z, xe, ell, EE, TE = get_spectra(z0, xe0, dz=dzs[i], both=True,
                #zstartmax=200)
        z, xe, ell, EE, TE = get_spectra(z0, 0.1, z_t=zts[i], both=True,
                zstartmax=200)
        #thermo = get_spectra(z0, xe0, dz=dzs[i], therm=True)
        #_, taulo, tauhi = get_twotau(thermo, zmax=100, xmin=2e-4)
    
        ax1.plot(z, xe, color=plt.cm.cividis(i/len(dzs)),lw=lw)
        twinplot(ell, TE, linestyle='-', marker=None, color=plt.cm.magma(i/len(zs)), 
                 axes=axs2, spec='TE', lw=lw, ymin=5.5e-3, ymax=5.5,
                 ylabels_r=True)

    ax1.set_xlim([0, 40])
    ax1.minorticks_on()
    ax1.set_xlabel(r'$z$')
    ax1.set_ylabel(r'$x_e$')
    ax2.set_title(r'$C_\ell^\mathrm{EE}$')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis, norm=plt.Normalize(vmin=zts.min(), vmax=zts.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$z_\mathrm t$')
    #plt.tight_layout()
    plt.savefig('f1_zt_alt_TE.pdf', bbox_inches='tight')
    plt.savefig('f1_zt_alt_TE.png', bbox_inches='tight')
    plt.close('all')
    return

def fig2_3(num=50, n_noises=30):
    '''
    z_re constraints only, EE and Wishart. What do I want this plot to look
    like? How about N_l^TT = 0, crank up N_l^EE.

    alphas increase as noise decreases
    C0 for Wishart, C1 for EE only.
    '''
    zres = np.linspace(6, 10, num)
    wps = np.logspace(0, np.log10(200), n_noises) 
    N_lEEs = (wps*np.pi/180/60)**2

    fig, axes = plt.subplots(nrows=1, ncols=2, 
                             gridspec_kw={"width_ratios":[1,0.05], "wspace":0.1})
    ax = axes[0]
    cax = axes[1]
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
            norm=mpl.colors.LogNorm(vmin=wps.min(), vmax=wps.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$w_p^{-1/2}$ [$\mathrm{\mu K\,arcmin}$]')

    lnPWs = []
    for i, N_lEE in enumerate(N_lEEs):
        print(i)
        ell, EE, TE, TT = get_spectra(8, 0, lmax=lmax, spectra=True, all_spectra=True)
        Clhat = np.array([TT, EE+N_lEE, EE*0, TE, EE*0, EE*0])
        lnPE = []
        lnPW = []
        for zre in zres:
            lnPE.append(lnprob_EE_ell(zre, 0, Clhat[1], N_l=N_lEE))
            lnPW.append(lnprob_wish_ell(zre, 0, Clhat, N_lT=0, N_lE=N_lEE))
        ax.plot(zres, np.exp(np.sum(lnPW, axis=1)),
                color=plt.cm.viridis(i/len(N_lEEs)), zorder=1/N_lEE)
        #ax.plot(zres, np.exp(np.sum(lnPW, axis=1)), color='C0',
        #        alpha=1-i/len(N_lEEs))
        #ax.plot(zres, np.exp(np.sum(lnPE, axis=1)), color='C1',
        #        alpha=1-i/len(N_lEEs))
        # lnPW is an array of length len(zres), 11.
        lnPWs.append(np.exp(np.sum(lnPW, axis=1)))
    lnPWs = np.array(lnPWs)
    print(lnPWs.shape)
    print(zres.shape)
    bla = np.vstack((zres, lnPWs))
    np.save('z_lnL', bla)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel(r'$z_\mathrm{re}$')
    ax.set_ylabel(r'$P(z_\mathrm{re}|x_e^\mathrm{min}=0,\{\hat{\boldsymbol C}_\ell\})$')
    fig.savefig('f2.pdf', bbox_inches='tight')
    fig.savefig('f2.png', bbox_inches='tight')
    plt.close('all')
    return

def fig2_alt(num=50, n_noises=30):
    '''
    same as fig2, but for tau.
    '''
    zres = np.linspace(4, 24, num)
    wps = np.logspace(1, np.log10(5e4), n_noises) 
    N_lEEs = (wps*np.pi/180/60)**2

    fig, axes = plt.subplots(nrows=1, ncols=2, 
                             gridspec_kw={"width_ratios":[1,0.05], "wspace":0.1})
    ax = axes[0]
    cax = axes[1]
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
            norm=mpl.colors.LogNorm(vmin=wps.min(), vmax=wps.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$w_p^{-1/2}$ [$\mathrm{\mu K\,arcmin}$]')

    taus = []
    lnPWs = []
    for i, N_lEE in enumerate(N_lEEs):
        print(i)
        ell, EE, TE, TT = get_spectra(14, 0, lmax=lmax, spectra=True, all_spectra=True)
        Clhat = np.array([TT, EE+N_lEE, EE*0, TE, EE*0, EE*0])
        lnPE = []
        lnPW = []
        taus = []
        for zre in zres:
            lnPE.append(lnprob_EE_ell(zre, 0, Clhat[1], N_l=N_lEE))
            lnPW.append(lnprob_wish_ell(zre, 0, Clhat, N_lT=0, N_lE=N_lEE))
            zsplit, tau_lo, tau_hi = get_twotau(get_spectra(zre, 0, therm=True))
            taus.append(tau_lo)
        ax.plot(taus, np.exp(np.sum(lnPW, axis=1)),
                color=plt.cm.viridis(i/len(N_lEEs)), zorder=1/N_lEE)
        #ax.plot(zres, np.exp(np.sum(lnPW, axis=1)), color='C0',
        #        alpha=1-i/len(N_lEEs))
        #ax.plot(zres, np.exp(np.sum(lnPE, axis=1)), color='C1',
        #        alpha=1-i/len(N_lEEs))
        lnPWs.append(np.exp(np.sum(lnPW, axis=1)))
    taus = np.array(taus)
    lnPWs = np.array(lnPWs)
    bla = np.vstack((taus, lnPWs))
    np.save('tau_lnL', bla)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$P(\tau|x_e^\mathrm{min}=0,\{\hat{\boldsymbol C}_\ell\})$')
    fig.savefig('f2_alt.pdf', bbox_inches='tight')
    fig.savefig('f2_alt.png', bbox_inches='tight')
    plt.close('all')
    return

def fig2_wrong(num=50, n_noises=30):
    '''
    same as fig2, but for tau.
    '''
    zres = np.linspace(6, 10, num)
    wps = np.logspace(0, np.log10(200), n_noises) 
    N_lEEs = (wps*np.pi/180/60)**2

    fig, axes = plt.subplots(nrows=1, ncols=2, 
                             gridspec_kw={"width_ratios":[1,0.05], "wspace":0.1})
    ax = axes[0]
    cax = axes[1]
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
            norm=mpl.colors.LogNorm(vmin=wps.min(), vmax=wps.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$w_p^{-1/2}$ [$\mathrm{\mu K\,arcmin}$]')

    taus = []
    lnPWs = []
    xe0 = 0.01
    for i, N_lEE in enumerate(N_lEEs):
        print(i)
        ell, EE, TE, TT = get_spectra(8, xe0, lmax=lmax, spectra=True, all_spectra=True)
        Clhat = np.array([TT, EE+N_lEE, EE*0, TE, EE*0, EE*0])
        lnPE = []
        lnPW = []
        taus = []
        for zre in zres:
            lnPE.append(lnprob_EE_ell(zre, 0, Clhat[1], N_l=N_lEE))
            lnPW.append(lnprob_wish_ell(zre, 0, Clhat, N_lT=0, N_lE=N_lEE))
            zsplit, tau_lo, tau_hi = get_twotau(get_spectra(zre, 0, therm=True))
            taus.append(tau_lo)
        ax.plot(taus, np.exp(np.sum(lnPW, axis=1)),
                color=plt.cm.viridis(i/len(N_lEEs)), zorder=1/N_lEE)
        #ax.plot(zres, np.exp(np.sum(lnPW, axis=1)), color='C0',
        #        alpha=1-i/len(N_lEEs))
        #ax.plot(zres, np.exp(np.sum(lnPE, axis=1)), color='C1',
        #        alpha=1-i/len(N_lEEs))
        lnPWs.append(np.exp(np.sum(lnPW, axis=1)))
    taus = np.array(taus)
    lnPWs = np.array(lnPWs)
    bla = np.vstack((taus, lnPWs))
    np.save('tau_lnL_wrong', bla)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$P(\tau|x_e^\mathrm{min}=0,\{\hat{\boldsymbol C}_\ell\})$')
    fig.savefig('f2_wrong.pdf', bbox_inches='tight')
    fig.savefig('f2_wrong.png', bbox_inches='tight')
    plt.close('all')
    return

def fig3(zre=7, xe=0.05, lmin=2, lmax=100, ntests=2):
    '''
    Contour plots
    '''
    F = []
    wps = np.array([10, 20, 50, 100, 200, 500, 1000])
    #wps = np.array([1e3, 5e3, 1e4, 5e4])
    #wps = np.array([0, np.inf])
    # 70 uK-arcmin corresponds to the projected white noise level and the
    # reported one for 143 GHz, both in the blue book and in the final results.
    # But 100 is a bit closer to the red  noise sensitivity...
    wps = np.array([0, 10, 70])
    wps = np.array([0, 10, 100, 200])
    wps = np.array([0, 10, 60, 100])
    #wps = np.array([0, 10, 50, 200])
    plt.figure()
    ell = np.arange(lmin, lmax+1)
    fig, axes = plt.subplots(3,1)
    fig2, axes2 = plt.subplots(3,1)
    fig3, axes3 = plt.subplots(3,1)
    for wp in wps:
        #F.append(np.sum(get_F_ell(zre, xe,\
        #    N_lT=(wp*np.pi/180/60)**2,\
        #    N_lE=(wp*np.pi/180/60)**2),axis=0))
        F_ell, F_ell1d, dTTdx, dEEdx, dTEdx, dTTdz, dEEdz, dTEdz = get_F_ell(zre, xe,\
            N_lT=(0*np.pi/180/60)**2,\
            N_lE=(wp*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
            test2=True, dzre=5e-5, dxre=1e-5)
        F_ell = np.array(F_ell)
        axes[0].loglog(ell, abs(F_ell[:,0,0]))
        axes[0].set_ylabel(r'$|F_{zz}|$')
        axes[1].loglog(ell, abs(F_ell[:,1,0]))
        axes[1].set_ylabel(r'$|F_{xz}|$')
        axes[2].loglog(ell, abs(F_ell[:,1,1]))
        axes[2].set_ylabel(r'$|F_{xx}|$')
        axes[2].set_xlabel(r'$\ell$')

        F.append(np.sum(F_ell,axis=0))
        axes2[0].loglog(ell, abs(dTTdx))
        axes2[0].set_ylabel(r'$\partial C_\ell^\mathrm{TT}/\partial x$')
        axes2[1].loglog(ell, abs(dTEdx))
        axes2[1].set_ylabel(r'$\partial C_\ell^\mathrm{TE}/\partial x$')
        axes2[2].loglog(ell, abs(dEEdx))
        axes2[2].set_ylabel(r'$\partial C_\ell^\mathrm{EE}/\partial x$')
        axes2[2].set_xlabel(r'$\ell$')

        axes3[0].loglog(ell, abs(dTTdz))
        axes3[0].set_ylabel(r'$\partial C_\ell^\mathrm{TT}/\partial z$')
        axes3[1].loglog(ell, abs(dTEdz))
        axes3[1].set_ylabel(r'$\partial C_\ell^\mathrm{TE}/\partial z$')
        axes3[2].loglog(ell, abs(dEEdz))
        axes3[2].set_ylabel(r'$\partial C_\ell^\mathrm{EE}/\partial z$')
        axes3[2].set_xlabel(r'$\ell$')

    
    fig.savefig('test1.pdf', bbox_inches='tight')
    fig2.savefig('test2.pdf', bbox_inches='tight')
    fig3.savefig('test3.pdf', bbox_inches='tight')
    plt.close('all')


    mean = np.array([zre, xe])
    gausses = []
    for i in range(len(F)):
        print(wps[i])
        cov = np.linalg.inv(F[i])
        print(cov[0,0]**0.5, cov[1,1]**0.5)
        gauss = GaussianND(mean, cov, labels=[r'$z_\mathrm{re}$', r'$x_e^\mathrm{min}$'],\
            label=r'${0}\,\mathrm{{\mu K\,arcmin}}$'.format(wps[i]))
        gausses.append(gauss)

    g = plots.getSubplotPlotter()
    #print(colors, type(colors))
    #colors = tuple(colors)
    colors = [r'C{0}'.format(i) for i in range(len(F))]
    g.triangle_plot(gausses[::-1], filled=True, colors=colors[::-1],
            contour_colors=colors[::-1], contour_lws=[1]*len(gausses))


    fig = plt.gcf()
    for ax in fig.axes[:2]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])

    fig.axes[0].set_xlim(4.5, 9.5)
    fig.axes[2].set_xlim(4.5, 9.5)

    fig.axes[0].set_xticks([5,6,7,8,9])
    fig.axes[2].set_xticks([5,6,7,8,9])
    
    fig.axes[1].set_xlim(0, 0.175)
    fig.axes[2].set_ylim(0, 0.175)
    
    #fig.axes[2].set_yticks([0.02, 0.04])
    #fig.axes[2].set_xticks([0.06, 0.08])

    fig.axes[1].set_xticks([0.05, 0.10, 0.15])
    fig.axes[2].set_yticks([0.05, 0.10, 0.15])


    plt.savefig(f'f3_{lmin}_{lmax}.pdf', bbox_inches='tight')
    plt.savefig(f'f3_{lmin}_{lmax}.png', bbox_inches='tight')
    plt.close('all')
    return

def fig3_rednoise(zre=8, xe=0.05, lmin=2, lmax=100, ntests=2):
    '''
    Contour plots
    '''
    F = []

    l, NlD_P = np.loadtxt('planck_FFP8_spec.txt', delimiter=',').T

    f = interp1d(l, NlD_P)
    ell = np.arange(lmin, lmax+1)
    D_l = f(ell)
    Z = ell*(ell+1)/(2*np.pi)
    N_ell = D_l / Z
    N_ell = np.insert(N_ell, [0], [0,1])
    print(N_ell.shape)
    N_ells = [(0*np.pi/180/60)**2, (100*np.pi/180/60)**2, N_ell]

    ell = np.arange(lmin, lmax+1)
    fig, axes = plt.subplots(3,1)
    fig2, axes2 = plt.subplots(3,1)
    fig3, axes3 = plt.subplots(3,1)
    for Nl in N_ells:
        F_ell, F_ell1d, dTTdx, dEEdx, dTEdx, dTTdz, dEEdz, dTEdz = get_F_ell(zre, xe,\
            N_lE=Nl, lmin=lmin, lmax=lmax,
            test2=True, dzre=5e-5, dxre=1e-5)
        F_ell = np.array(F_ell)
        F.append(np.sum(F_ell,axis=0))

    mean = np.array([zre, xe])
    gausses = []
    labels = ['Noiseless', r'$100\,\mathrm{\mu K\,arcmin}$', r'FFP8']
    for i in range(len(F)):
        cov = np.linalg.inv(F[i])
        print(labels[i])
        print(cov[0,0]**0.5, cov[1,1]**0.5)
        gauss = GaussianND(mean, cov, labels=[r'$z_\mathrm{re}$', r'$x_e^\mathrm{min}$'],\
            label=labels[i])
        gausses.append(gauss)

    g = plots.getSubplotPlotter()
    colors = [r'C{0}'.format(i) for i in range(len(F))]
    g.triangle_plot(gausses[::-1], filled=True, colors=colors[::-1],
            contour_colors=colors[::-1], contour_lws=[1]*len(gausses))


    fig = plt.gcf()
    for ax in fig.axes[:2]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])

    fig.axes[0].set_xlim(5., 11)
    fig.axes[2].set_xlim(5., 11)

    fig.axes[0].set_xticks([6,8,10])
    fig.axes[2].set_xticks([6,8,10])
    
    fig.axes[1].set_xlim(0, 0.18)
    fig.axes[2].set_ylim(0, 0.18)
    

    fig.axes[1].set_xticks([0.05, 0.10, 0.15])
    fig.axes[2].set_yticks([0.05, 0.10, 0.15])


    plt.savefig(f'f3_rednoise_{lmin}_{lmax}.pdf', bbox_inches='tight')
    plt.savefig(f'f3_rednoise_{lmin}_{lmax}.png', bbox_inches='tight')
    plt.close('all')
    return

def fig3_ell_var(zre=7, xe=0.05, noise=0):
    '''
    Contour plots
    '''
    F = []
    #pairs = [(30,99), (20, 29), (2, 9), (10, 19), (2, 99)][::-1]
    pairs = [(2,99), (2, 9), (10, 19), (20, 29), (30, 99)]
    colors = [r'C{0}'.format(i) for i in range(len(pairs))]
    for p in pairs:
        lmin, lmax = p
        ell = np.arange(lmin, lmax+1)
        F_ell, F_ell1d, dTTdx, dEEdx, dTEdx, dTTdz, dEEdz, dTEdz = get_F_ell(zre, xe,\
            N_lT=(0*np.pi/180/60)**2,\
            N_lE=(noise*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
            test2=True, dzre=5e-5, dxre=1e-5)
        F_ell = np.array(F_ell)
        F.append(np.sum(F_ell,axis=0))

    mean = np.array([zre, xe])
    gausses = []
    sizes = []
    for i in range(len(F)):
        cov = np.linalg.inv(F[i])
        # rho = np.cos(theta)
        theta = np.arccos(cov[1,0])
        sizes.append(abs(cov[0,0]*cov[1,1]/np.sin(theta)/np.cos(theta)))
        gauss = GaussianND(mean, cov, labels=[r'$z_\mathrm{re}$', r'$x_e^\mathrm{min}$'],\
            label=f'${pairs[i][0]}\leqslant\ell<{pairs[i][1]+1}$')
        gausses.append(gauss)
        #print(pairs[i], sizes[i])
    sizes = np.array(sizes)
    idx = np.arange(len(F))
    print('\n')
    print('noise')
    print(noise)
    print('sizes')
    print(sizes)
    inds = np.argsort(sizes)
    print('index to reorder sizes')
    print(inds)
    print('reordered sizes')
    print(sizes[inds])
    aa = np.argsort(inds)
    print('sizes in original order')
    print(sizes[inds][aa])
    print('list of the original ordering names')
    print(idx[inds])

    g = plots.getSubplotPlotter()
    colors = tuple(colors)
    gausses = [x for _,x in sorted(zip(sizes, gausses))][::-1]
    colors = [x for _,x in sorted(zip(sizes, colors))][::-1]
    g.triangle_plot(gausses, filled=True,
            colors=colors,
            contour_colors=colors,
            label_order=idx[::-1][inds],
            contour_lws=[1]*len(F), legend_loc=None)
    g.finish_plot(no_tight=True)

    fig = plt.gcf()
    for ax in fig.axes[:2]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])

    fig.axes[0].set_xlim(4.5, 9.5)
    fig.axes[2].set_xlim(4.5, 9.5)

    fig.axes[0].set_xticks([5,6, 7,8, 9])
    fig.axes[2].set_xticks([5,6, 7,8, 9])
    
    fig.axes[1].set_xlim(0, 0.175)
    fig.axes[2].set_ylim(0, 0.175)
    
    #fig.axes[2].set_yticks([0.02, 0.04])
    #fig.axes[2].set_xticks([0.06, 0.08])

    fig.axes[1].set_xticks([0.05, 0.10, 0.15])
    fig.axes[2].set_yticks([0.05, 0.10, 0.15])


    noise = str(noise).zfill(3)
    plt.savefig(f'f3_ellranges_{noise}uKarcmin.pdf', bbox_inches='tight')
    plt.suptitle(r'${0}\,\mathrm{{\mu K\,arcmin}}$'.format(noise), x=0.1, y=1.05)
    plt.savefig(f'f3_ellranges_{noise}uKarcmin.png', bbox_inches='tight')
    plt.close('all')
    return

def swap(*line_list):
    """
    Example
    -------
    line = plot(linspace(0, 2, 10), rand(10))
    swap(line)
    """
    for lines in line_list:
        try:
            iter(lines)
        except:
            lines = [lines]

        for line in lines:
            xdata, ydata = line.get_xdata(), line.get_ydata()
            line.set_xdata(ydata)
            line.set_ydata(xdata)
            line.axes.autoscale_view()

def fig3_taus(zre=7, xe=0.05, lmin=2, lmax=100, ntests=2):
    '''
    Contour plots
    '''
    F = []
    wps = np.array([10, 20, 50, 100, 200, 500, 1000])
    #wps = np.array([1e3, 5e3, 1e4, 5e4])
    #wps = np.array([0, np.inf])
    # 70 uK-arcmin corresponds to the projected white noise level and the
    # reported one for 143 GHz, both in the blue book and in the final results.
    # But 100 is a bit closer to the red  noise sensitivity...
    wps = np.array([0, 10, 70])
    wps = np.array([0, 10, 100, 200])
    wps = np.array([0, 10, 60, 100])
    #wps = np.array([0, 10, 50, 200])
    plt.figure()
    ell = np.arange(lmin, lmax+1)
    fig, axes = plt.subplots(3,1)
    fig2, axes2 = plt.subplots(3,1)
    fig3, axes3 = plt.subplots(3,1)
    for wp in wps:
        F_ell  = get_F_ell(zre, xe,\
            N_lT=(0*np.pi/180/60)**2,\
            N_lE=(wp*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
            tau_vars=True)
        F_ell = np.array(F_ell)
        F.append(np.sum(F_ell,axis=0))


    mean = np.array([0.06, 0.01])
    mean = np.array([0.057202082194763124, 0.013024054184239887])
    _, taulo, tauhi = get_twotau(get_spectra(zre, xe, therm=True))
    mean = np.array([taulo, tauhi])
    gausses = []
    for i in range(len(F)):
        #print(F[i], i, wps[i])
        cov = np.linalg.inv(F[i])
        print(wps[i])
        print('taulo_uncertainty', 'tauhi_uncertainty', 'cov')
        print(cov[0,0]**0.5, cov[1,1]**0.5, cov[0,1]/(cov[0,0]**0.5*cov[1,1]**0.5))
        gauss = GaussianND(mean, cov, labels=[r'$\tau_\mathrm{lo}$', r'$\tau_\mathrm{hi}$'],\
            label=r'${0}\,\mathrm{{\mu K\,arcmin}}$'.format(wps[i]))
        gausses.append(gauss)

    g = plots.getSubplotPlotter()
    colors = [r'C{0}'.format(i) for i in range(len(F))]
    g.triangle_plot(gausses[::-1], filled=True, colors=colors[::-1],
            contour_colors=colors[::-1], contour_lws=[1]*len(gausses))


    fig = plt.gcf()
    for ax in fig.axes[:2]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
    '''
    fig.axes[2].spines['right'].set_visible(False)
    fig.axes[2].spines['top'].set_visible(False)
    fig.axes[2].yaxis.set_ticks_position('left')
    fig.axes[2].xaxis.set_ticks_position('bottom')
    '''

    taulos = np.linspace(0, 0.1)
    tauhis = 0.06 - taulos
    fig.axes[2].plot(taulos, 0.08-taulos, 'k', lw=1, dashes=[5,2]) 
    fig.axes[2].plot(taulos, 0.07-taulos, 'k', lw=1, dashes=[4,2]) 
    fig.axes[2].plot(taulos, 0.06-taulos, 'k', lw=1, dashes=[3,2]) 
    fig.axes[2].plot(taulos, 0.05-taulos, 'k', lw=1, dashes=[2,2]) 
    fig.axes[2].plot(taulos, 0.04-taulos, 'k', lw=1, dashes=[1,2]) 

    
    fig.axes[2].set_xticks([0.03, 0.04, 0.05,0.06])
    fig.axes[0].set_xticks([0.03, 0.04, 0.05,0.06])


    fig.axes[2].set_yticks([0, 0.01, 0.02, 0.03, 0.04])
    fig.axes[1].set_xticks(   [0.01, 0.02, 0.03, 0.04])


    fig.axes[0].set_xlim(0.033-0.007, 0.058+0.007)
    fig.axes[2].set_xlim(0.033-0.007, 0.058+0.007)
    
    fig.axes[1].set_xlim(0, 0.025+0.007*2)
    fig.axes[2].set_ylim(0, 0.025+0.007*2)

    '''
    lines = fig.axes[1].get_lines()
    swap(lines)
    fig.axes[1].set_xlim(fig.axes[0].get_ylim())
    fig.axes[1].set_ylim(0, 0.035+0.02)
    fig.axes[1].spines['bottom'].set_visible(False)
    fig.axes[1].spines['left'].set_visible(True)

    fig.axes[1].set_yticks([0, 0.01, 0.02, 0.03])
    fig.axes[1].yaxis.set_ticklabels([])
    fig.axes[1].set_xticks([])
    fig.axes[1].set_xlabel('')


    yticks = fig.axes[2].yaxis.get_major_ticks()
    yticks[0].tick1line.set_visible(False)
    '''

    plt.savefig('f3_taus.pdf', bbox_inches='tight')
    plt.savefig('f3_taus.png', bbox_inches='tight')
    plt.close('all')
    return

def fig3_taus_forcezsplit(zre=8, xe=0.05, lmin=2, lmax=100, ntests=2):
    '''
    Contour plots
    '''
    F = []
    wps = np.array([10, 20, 50, 100, 200, 500, 1000])
    #wps = np.array([1e3, 5e3, 1e4, 5e4])
    #wps = np.array([0, np.inf])
    # 70 uK-arcmin corresponds to the projected white noise level and the
    # reported one for 143 GHz, both in the blue book and in the final results.
    # But 100 is a bit closer to the red  noise sensitivity...
    wps = np.array([0, 10, 70])
    wps = np.array([0, 10, 100, 200])
    wps = np.array([0, 10, 60, 100])
    #wps = np.array([0, 10, 50, 200])
    plt.figure()
    ell = np.arange(lmin, lmax+1)
    fig, axes = plt.subplots(3,1)
    fig2, axes2 = plt.subplots(3,1)
    fig3, axes3 = plt.subplots(3,1)
    for wp in wps:
        F_ell, taulo, tauhi  = get_F_ell_forcezsplit(zre, xe,\
            N_lT=(0*np.pi/180/60)**2,\
            N_lE=(wp*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
            tau_vars=True)
        F_ell = np.array(F_ell)
        F.append(np.sum(F_ell,axis=0))


    mean = np.array([0.06, 0.01])
    mean = np.array([0.057202082194763124, 0.013024054184239887])
    print(mean)
    mean = np.array([taulo, tauhi])
    print(mean)
    gausses = []
    for i in range(len(F)):
        print(F[i], i, wps[i])
        cov = np.linalg.inv(F[i])
        print(cov[0,0]**0.5, cov[1,1]**0.5, cov[0,1])
        gauss = GaussianND(mean, cov, labels=[r'$\tau_\mathrm{lo}$', r'$\tau_\mathrm{hi}$'],\
            label=r'${0}\,\mathrm{{\mu K\,arcmin}}$'.format(wps[i]))
        gausses.append(gauss)


    g = plots.getSubplotPlotter()
    colors = [r'C{0}'.format(i) for i in range(len(F))]
    g.triangle_plot(gausses[::-1], filled=True, colors=colors[::-1],
            contour_colors=colors[::-1], contour_lws=[1]*len(gausses))


    fig = plt.gcf()
    for ax in fig.axes[:2]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])

    taulos = np.linspace(0, 0.1)
    tauhis = (taulo+tauhi) - taulos
    cent = taulo+tauhi
    fig.axes[2].plot(taulos, cent+0.02-taulos, 'k', lw=1, dashes=[5,2]) 
    fig.axes[2].plot(taulos, cent+0.01-taulos, 'k', lw=1, dashes=[4,2]) 
    fig.axes[2].plot(taulos, cent+0.00-taulos, 'k', lw=1, dashes=[3,2]) 
    fig.axes[2].plot(taulos, cent-0.01-taulos, 'k', lw=1, dashes=[2,2]) 
    fig.axes[2].plot(taulos, cent-0.02-taulos, 'k', lw=1, dashes=[1,2]) 

    
    #fig.axes[0].set_xlim(0.038-0.007, 0.073+0.007)
    #fig.axes[2].set_xlim(0.038-0.007, 0.073+0.007)
    
    fig.axes[1].set_xlim(0, 0.02)
    fig.axes[2].set_ylim(0, 0.02)

    fig.axes[2].set_yticks([0, 0.01, 0.02])
    fig.axes[1].set_xticks(   [0.01, 0.02])

    plt.savefig('f3_taus_forcezsplit.pdf', bbox_inches='tight')
    plt.savefig('f3_taus_forcezsplit.png', bbox_inches='tight')
    plt.close('all')
    return




def fig3_alt(zre=8, xe=0.05, lmin=2, lmax=100, ntests=2):
    '''
    Contour plots with and without temperature
    '''
    F = []
    FEs = []
    wps = np.array([10, 20, 50, 100, 200, 500, 1000])
    #wps = np.array([1e3, 5e3, 1e4, 5e4])
    #wps = np.array([0, np.inf])
    wps = np.array([0, np.inf])
    wplabs = np.array(['Noiseless Measurement of T', 
        'Infinitely Noisy T'])
    plt.figure()
    ell = np.arange(lmin, lmax+1)
    fig, axes = plt.subplots(3,1)
    fig2, axes2 = plt.subplots(3,1)
    fig3, axes3 = plt.subplots(3,1)
    for wp in wps:
        F_ell, F_ellEE, dTTdx, dEEdx, dTEdx, dTTdz, dEEdz, dTEdz = get_F_ell(zre, xe,\
            N_lT=(wp*np.pi/180/60)**2,\
            N_lE=(0*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
            test2=True, dzre=5e-5, dxre=1e-5)
        F_ell = np.array(F_ell)
        F.append(np.sum(F_ell,axis=0))
        FEs.append(np.sum(F_ellEE, axis=0))
    

    mean = np.array([zre, xe])
    gausses = []
    for i in range(len(F)):
        print(F[i], i)
        cov = np.linalg.inv(F[i])
        gauss = GaussianND(mean, cov, labels=[r'$z_\mathrm{re}$', r'$x_e^\mathrm{min}$'],\
            label=wplabs[i])
        gausses.append(gauss)

    g = plots.getSubplotPlotter()
    #print(colors, type(colors))
    #colors = tuple(colors)
    colors = [r'C{0}'.format(i) for i in range(len(F))]
    g.triangle_plot(gausses[::-1], filled=True, colors=colors[::-1],
            contour_colors=colors[::-1], contour_lws=[1]*len(gausses))


    fig = plt.gcf()
    for ax in fig.axes[:2]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])

    #fig.axes[0].set_xlim(5., 11)
    #fig.axes[2].set_xlim(5., 11)
    #
    #fig.axes[1].set_xlim(0, 0.15)
    #fig.axes[2].set_ylim(0, 0.15)
    #
    ##fig.axes[2].set_yticks([0.02, 0.04])
    ##fig.axes[2].set_xticks([0.06, 0.08])

    #fig.axes[1].set_xticks([0.05, 0.10])
    #fig.axes[2].set_yticks([0.05, 0.10])


    plt.savefig('f3_alt.pdf', bbox_inches='tight')
    plt.savefig('f3_alt.png', bbox_inches='tight')
    plt.close('all')

    gausses = []
    for i in range(len(FEs)-1):
        print(FEs[i], i)
        cov = np.linalg.inv(FEs[i])
        gauss = GaussianND(mean, cov, labels=[r'$z_\mathrm{re}$', r'$x_e^\mathrm{min}$'],\
            label='EE Constraint Only')
        gausses.append(gauss)

    g = plots.getSubplotPlotter()
    #print(colors, type(colors))
    #colors = tuple(colors)
    colors = [r'C3']
    g.triangle_plot(gausses[-1], filled=True, colors=colors[-1],
            contour_colors=colors[-1], contour_lws=[1]*len(gausses))


    fig = plt.gcf()
    for ax in fig.axes[:2]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])

    plt.suptitle('EE Constraints Only')

    plt.savefig('f3_altEE.pdf', bbox_inches='tight')
    plt.savefig('f3_altEE.png', bbox_inches='tight')
    plt.close('all')

    return

def best_case_versus_worst(zre=8, xe=0.05, lmin=2, lmax=100, ntests=2):
    '''
    Trying to make a plot of sigma_tau as a function of noise, for the Wishart,
    EE-only, and TE-only cases.
    '''
    wps = np.logspace(np.log10(0.65), np.log10(230), 50)
    F = []
    sigma_tautot = []
    for wp in wps:
        F_ell  = get_F_ell_tau(0.06,\
            N_lT=(0*np.pi/180/60)**2,\
            N_lE=(wp*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
            tau_vars=True)
        F_ell = np.array(F_ell)
        F.append(np.sum(F_ell,axis=0))
    for i in range(len(F)):
        sigma_tautot.append(1/F[i]**0.5)
    plt.plot(wps, sigma_tautot, label=r'$\sigma_{\tau}$ from $\hat C_\ell^\mathrm{TT}+\hat C_\ell^\mathrm{TE}+\hat C_\ell^\mathrm{EE}$')

    F = []
    sigma_tautot = []
    for wp in wps:
        F_ell  = get_F_ell_tau(0.06,\
            N_lT=(10000*np.pi/180/60)**2,\
            N_lE=(wp*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
            tau_vars=True)
        F_ell = np.array(F_ell)
        F.append(np.sum(F_ell,axis=0))
    for i in range(len(F)):
        #cov = np.linalg.inv(F[i])
        sigma_tautot.append(1/F[i]**0.5)
    plt.plot(wps, sigma_tautot, label=r'$\sigma_{\tau}$ from $\hat C_\ell^\mathrm{EE}$')

    F = []
    sigma_tautot = []
    for wp in wps:
        F_ell  = get_F_ell_3_tau(0.06,\
            N_lT=(0*np.pi/180/60)**2,\
            N_lE=(wp*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
            tau_vars=True)
        F_ell = np.array(F_ell)
        F.append(np.sum(F_ell,axis=0))
    for i in range(len(F)):
        sigma_tautot.append(1/F[i]**0.5)
    print(F)
    #print(sigma_tautot)
    plt.plot(wps, sigma_tautot, label=r'$\sigma_{\tau}$ from $\hat C_\ell^\mathrm{TE}$')

    #plt.axhline(0.007, color='k', linestyle='--',
    #        label=r'$\sigma_\tau$ from \textit{Planck} $\hat C_\ell^\mathrm{EE}$')
    plt.xscale('log')
    #plt.yscale('log')
    plt.yscale('linear')
    plt.xlim([0.65, 230])
    plt.legend(loc='best')
    plt.xlabel(r'$w_p^{-1/2}$ [$\mathrm{\mu K\,arcmin}$]')
    plt.ylabel(r'$\sigma_\tau$')
    #yticks = [0.002, 0.005, 0.01, 0.02]
    yticks = np.arange(0.002, plt.gca().get_ylim()[1], 0.002)
    plt.yticks(yticks, yticks)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    plt.xticks([1,10,100],[1,10,100])


    plt.savefig('comp_sigma_taus.pdf', bbox_inches='tight')
    plt.savefig('comp_sigma_taus.png', bbox_inches='tight', dpi=300)


def noise_vs_uncertainty(zre=7, xe=0.05, lmin=2, lmax=100, ntests=2):
    '''
    Contour plots
    '''
    F = []
    wps = np.logspace(np.log10(0.65), np.log10(230), 50)
    plt.figure()
    F_tautot = []
    for wp in wps:
        F_ell  = get_F_ell(zre, xe,\
            N_lT=(0*np.pi/180/60)**2,\
            N_lE=(wp*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
            tau_vars=True)
        F_ell = np.array(F_ell)
        F_tautot_ell = F_ell[:,0,0] + F_ell[:,1,1] + 2*F_ell[:,0,1]
        F_tautot.append(np.sum(F_tautot_ell))
        F.append(np.sum(F_ell,axis=0))
    
    sigma_tauhi = []
    sigma_taulo = []
    sigma_tautot = []
    rhos = []

    for i in range(len(F)):
        print(F[i], i, wps[i])
        cov = np.linalg.inv(F[i])
        sigma_taulo.append(cov[0,0]**0.5)
        sigma_tauhi.append(cov[1,1]**0.5)
        rhos.append(cov[0,1])
        sigma_tautot.append(1/F_tautot[i]**0.5)

    sigma_taulo = np.array(sigma_taulo)
    sigma_tauhi = np.array(sigma_tauhi)
    rhos = np.array(rhos)

    sigma_tautot = np.sqrt(sigma_taulo**2+sigma_tauhi**2)

    plt.plot(wps, sigma_taulo, 'C0-', label=r'$\sigma_{\tau_\mathrm{lo}}$')
    plt.plot(wps, sigma_tauhi, 'C1-', label=r'$\sigma_{\tau_\mathrm{hi}}$')
    plt.plot(wps, sigma_tautot, 'C2-', label=r'$\sigma_{\tau_\mathrm{tot}}$')
    F = []
    sigma_tautot = []
    for wp in wps:
        F_ell  = get_F_ell(zre+1, xe*0,\
            N_lT=(0*np.pi/180/60)**2,\
            N_lE=(wp*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
            tau_vars=True)
        F_ell = np.array(F_ell)
        F.append(np.sum(F_ell,axis=0))
    for i in range(len(F)):
        cov = np.linalg.inv(F[i])
        sigma_tautot.append(1/F[i][0,0]**0.5)
    plt.plot(wps, sigma_tautot, 'C3', label=r'$\sigma_{\tau}$ (tanh-like)',
            zorder=-1)


    #plt.axhline(0.007, color='k', linestyle='--',
    #        label=r'\textit{Planck} $\sigma_\tau$')

    plt.xscale('log')
    #plt.yscale('log')
    plt.legend(loc='best')

    plt.xlabel(r'$w_p^{-1/2}$ [$\mathrm{\mu K\,arcmin}$]')
    plt.ylim(ymin=0, ymax=0.01)
    yticks = np.round(np.arange(0.000,  0.01+0.002, 0.002), 3)

    plt.yticks(yticks, yticks)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    plt.xticks([1,10,100],[1,10,100])
    plt.xlim([0.65, 230])
    plt.ylabel(r'$\sigma_\tau$')


    plt.savefig('sigma_taus.pdf', bbox_inches='tight')
    plt.savefig('sigma_taus.png', bbox_inches='tight', dpi=300)

    plt.figure()
    F = []
    for wp in wps:
        F_ell  = get_F_ell(zre, xe,\
            N_lT=(0*np.pi/180/60)**2,\
            N_lE=(wp*np.pi/180/60)**2, lmin=lmin, lmax=lmax)
        F_ell = np.array(F_ell)
        F.append(np.sum(F_ell,axis=0))
    
    sigma_zre = []
    sigma_xe = []
    sigma_z = []
    rhos = []
    
    for i in range(len(F)):
        print(F[i], i, wps[i])
        cov = np.linalg.inv(F[i])
        sigma_zre.append(cov[0,0]**0.5)
        sigma_xe.append(cov[1,1]**0.5)
        sigma_z.append(1/F[i][0,0]**0.5)
        rhos.append(cov[0,1])

    sigma_zre = np.array(sigma_zre)
    sigma_xe = np.array(sigma_xe)
    rhos = np.array(rhos)

    plt.plot(wps, sigma_zre/zre, 'C0-', label=r'$\sigma_{z_\mathrm{re}}/z_\mathrm{re}$')
    plt.plot(wps, sigma_xe/xe, 'C1-', label=r'$\sigma_{x^\mathrm{min}_e}/x^\mathrm{min}_e$')
    plt.plot(wps, (rhos/zre/xe)**0.5, '-',
            label=r'$|\rho/(z_\mathrm{re}x^\mathrm{min}_e)|^{1/2}$',
            color='C2')
    plt.plot(wps, (-rhos/zre/xe)**0.5, 'C2--')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.xlabel(r'$w_p^{-1/2}$ [$\mathrm{\mu K\,arcmin}$]')
    #yticks = [0.002, 0.005, 0.01, 0.02]
    #plt.yticks(yticks, yticks)

    plt.savefig('sigma_zs.pdf', bbox_inches='tight')
    plt.savefig('sigma_zs.png', bbox_inches='tight', dpi=300)
    plt.close('all')

    return

def testfig3(zre=8, xe=0.05, ntests=5):
    '''
    Making sure my derivatives are stable
    '''
    ders = []
    wps = np.array([10, 10, 10, 10])
    wps = np.ones(ntests)*10
    for wp in wps:
        ders_ell = get_F_ell(zre, xe,\
            N_lT=(wp*np.pi/180/60)**2,\
            N_lE=10, test=True)
        ders.append(ders_ell)
        print(ders[-1][0][2])
    print('\n')

    fig, axes = plt.subplots(3,1)
    fig2, axes2 = plt.subplots(3,1)

    for i in range(len(wps)):
        dTTdx, dEEdx, dTEdx, dTTdz, dEEdz, dTEdz = ders[-1]
        axes[0].semilogy(abs(dTTdx))
        axes[0].set_ylabel(r'$\partial C_\ell^\mathrm{TT}/\partial x$')
        axes[1].semilogy(abs(dTEdx))
        axes[1].set_ylabel(r'$\partial C_\ell^\mathrm{TE}/\partial x$')
        axes[2].semilogy(abs(dEEdx))
        axes[2].set_ylabel(r'$\partial C_\ell^\mathrm{EE}/\partial x$')
        axes[2].set_xlabel(r'$\ell$')
        axes2[0].semilogy(abs(dTTdz))
        axes2[0].set_ylabel(r'$\partial C_\ell^\mathrm{TT}/\partial z$')
        axes2[1].semilogy(abs(dTEdz))
        axes2[1].set_ylabel(r'$\partial C_\ell^\mathrm{TE}/\partial z$')
        axes2[2].semilogy(abs(dEEdz))
        axes2[2].set_ylabel(r'$\partial C_\ell^\mathrm{EE}/\partial z$')
        axes2[2].set_xlabel(r'$\ell$')
    fig.savefig('bla1.pdf', bbox_inches='tight')
    fig2.savefig('bla2.pdf', bbox_inches='tight')
    plt.close('all')

    return

def fig4(num=50, n_noises=30):
    xmins = np.linspace(0, 0.08, num)
    wps = np.logspace(0, np.log10(200), n_noises) 
    N_lEEs = (wps*np.pi/180/60)**2

    fig, axes = plt.subplots(nrows=1, ncols=2, 
                             gridspec_kw={"width_ratios":[1,0.05], "wspace":0.1})
    ax = axes[0]
    cax = axes[1]
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
            norm=mpl.colors.LogNorm(vmin=wps.min(), vmax=wps.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$w_p^{-1/2}$ [$\mathrm{\mu K\,arcmin}$]')

    lnPWs = []
    for i, N_lEE in enumerate(N_lEEs):
        print(i)
        ell, EE, TE, TT = get_spectra(8, 0, lmax=lmax, spectra=True, all_spectra=True)
        Clhat = np.array([TT, EE+N_lEE, EE*0, TE, EE*0, EE*0])
        lnPE = []
        lnPW = []
        for xmin in xmins:
            lnPE.append(lnprob_EE_ell(8, xmin, Clhat[1], N_l=N_lEE))
            lnPW.append(lnprob_wish_ell(8, xmin, Clhat, N_lT=0, N_lE=N_lEE))
        #ax.plot(xmins, np.exp(np.sum(lnPW, axis=1)), color='C0',
        #        alpha=1-i/len(N_lEEs))
        ax.plot(xmins, np.exp(np.sum(lnPW, axis=1)),
                color=plt.cm.viridis(i/len(N_lEEs)), zorder=1/N_lEE)
        #ax.plot(xmins, np.exp(np.sum(lnPE, axis=1)), color='C1',
        #        alpha=1-i/len(N_lEEs))
        lnPWs.append(np.exp(np.sum(lnPW, axis=1)))
    lnPWs = np.array(lnPWs)
    bla = np.vstack((xmins, lnPWs))
    np.save('xmin_lnL', bla)

    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel(r'$x_e^\mathrm{min}$')
    ax.set_ylabel(r'$P(x_e^\mathrm{min}|z_\mathrm{re}=8,\{\hat{\boldsymbol C}_\ell\})$')
    fig.savefig('f4.pdf', bbox_inches='tight')
    fig.savefig('f4.png', bbox_inches='tight')
    plt.close('all')

def fig4_alt(num=50, n_noises=30):
    xmins = np.linspace(0, 0.08, num)
    wps = np.logspace(0, np.log10(200), n_noises) 
    N_lEEs = (wps*np.pi/180/60)**2

    fig, axes = plt.subplots(nrows=1, ncols=2, 
                             gridspec_kw={"width_ratios":[1,0.05], "wspace":0.1})
    ax = axes[0]
    cax = axes[1]
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
            norm=mpl.colors.LogNorm(vmin=wps.min(), vmax=wps.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$w_p^{-1/2}$ [$\mathrm{\mu K\,arcmin}$]')

    lnPWs = []
    for i, N_lEE in enumerate(N_lEEs):
        print(i)
        ell, EE, TE, TT = get_spectra(8, 0, lmax=lmax, spectra=True, all_spectra=True)
        Clhat = np.array([TT, EE+N_lEE, EE*0, TE, EE*0, EE*0])
        lnPE = []
        lnPW = []
        tau_his = []
        for xmin in xmins:
            lnPE.append(lnprob_EE_ell(8, xmin, Clhat[1], N_l=N_lEE))
            lnPW.append(lnprob_wish_ell(8, xmin, Clhat, N_lT=0, N_lE=N_lEE))
            zsplit, tau_lo, tau_hi = get_twotau(get_spectra(8, xmin, therm=True))
            tau_his.append(tau_hi)
        #ax.plot(xmins, np.exp(np.sum(lnPW, axis=1)), color='C0',
        #        alpha=1-i/len(N_lEEs))
        ax.plot(tau_his, np.exp(np.sum(lnPW, axis=1)),
                color=plt.cm.viridis(i/len(N_lEEs)), zorder=1/N_lEE)
        #ax.plot(xmins, np.exp(np.sum(lnPE, axis=1)), color='C1',
        #        alpha=1-i/len(N_lEEs))
        lnPWs.append(np.exp(np.sum(lnPW, axis=1)))
    lnPWs = np.array(lnPWs)
    tau_his = np.array(tau_his)
    bla = np.vstack((tau_his, lnPWs))
    np.save('tauhis_lnL', bla)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel(r'$\tau_\mathrm{hi}$')
    ax.set_ylabel(r'$P(\tau_\mathrm{hi}|z_\mathrm{re}=8,\{\hat{\boldsymbol C}_\ell\})$')
    fig.savefig('f4_alt.pdf', bbox_inches='tight')
    fig.savefig('f4_alt.png', bbox_inches='tight')
    plt.close('all')


    return

def fig5(num=50):
    '''
    Seeing what happens when EE is constant and TT gets cranked up.
    '''
    zres = np.linspace(7, 9, num)
    N_lEE = 1e-6
    N_lTTs = np.logspace(-2, 2, 5)
    #fig, axes = plt.subplots(nrows=1, ncols=2, 
    #                         gridspec_kw={"width_ratios":[1,0.05], "wspace":0.1})
    #ax = axes[0]
    #cax = axes[1]
    #sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
    #        norm=mpl.colors.LogNorm(vmin=N_lTTs.min(), vmax=N_lTTs.max()))
    #sm._A = []
    #fig.colorbar(sm, cax=cax, label=r'$N_\ell^\mathrm{TT}$')
    fig, ax = plt.subplots(1)

    N_lTTs = [1e-6]

    for i, N_lTT in enumerate(N_lTTs):
        print(i, len(N_lTTs))
        ell, EE, TE, TT = get_spectra(8, 0, lmax=lmax, spectra=True, all_spectra=True)
        Clhat = np.array([TT+N_lTT, EE+N_lEE, EE*0, TE, EE*0, EE*0])
        lnPE = []
        lnPW = []
        for zre in zres:
            lnPE.append(lnprob_EE_ell(zre, 0, Clhat[1], N_l=N_lEE))
            lnPW.append(lnprob_wish_ell(zre, 0, Clhat, N_lT=N_lTT, N_lE=N_lEE))
        L1 = np.exp(np.sum(lnPW, axis=1))
        ax.plot(zres, L1, color='C0',
                alpha=1-i/len(N_lTTs))
        L2 = np.exp(np.sum(lnPE, axis=1))
        ax.plot(zres, L2, color='C1',
                alpha=1-i/len(N_lTTs))
    ax.set_ylim([-0.05, 1.05])
    ax.plot([],[],color='C0',linestyle='-',label='Wishart Distribution')
    ax.plot([],[],color='C1',linestyle='-',label=r'$\chi^2$ Distribution')
    ax.set_xlabel(r'$z_\mathrm{re}$')
    ax.set_ylabel(r'$P(z_\mathrm{re}|x_e^\mathrm{min}=0,\{\hat{\boldsymbol C}_\ell\})$')
    legend = ax.legend(loc='best')


    fig.savefig('f5.pdf', bbox_inches='tight')
    fig.savefig('f5.png', bbox_inches='tight')
    plt.close('all')

    plt.figure()

    mean = 8
    sigma = 1
    popt,pcov = curve_fit(gaus,zres,L1,p0=[1,mean,sigma])
    s1 = popt[2]
    plt.plot(zres, L1, label=r'$\hat C_\ell^\mathrm{TT}+\hat C_\ell^\mathrm{TE}+\hat C_\ell^\mathrm{EE}$')
    #plt.plot(zres, gaus(zres, *popt))
    print(popt)
    popt,pcov = curve_fit(gaus,zres,L2,p0=[1,mean,sigma])
    plt.plot(zres, L2, label='EE')
    s2 = popt[2]
    #plt.plot(zres, gaus(zres, *popt))
    print(popt)
    print(s1, s2)

    popt = np.copy(popt)
    s3 = np.sqrt(s2**2+s1**2)
    popt[2] = s3
    plt.plot(zres, gaus(zres, *popt), label=r'TT+TE')
    plt.ylabel(r'$P(z_\mathrm{re}|x_e^\mathrm{min}=0,\{\hat{\boldsymbol C}_\ell\})$')
    plt.xlabel(r'$z_\mathrm{re}$')
    legend = plt.legend(loc='upper left')
    plt.savefig('f5.pdf', bbox_inches='tight')
    plt.savefig('f5.png', bbox_inches='tight')

    print(s1, s2, s3)

    plt.show()


    return
def fig5_alt(num=50):
    '''
    Seeing what happens when EE is constant and TT gets cranked up. Same as fig5
    but with tau instead
    '''
    zres = np.linspace(7, 9, num)
    N_lEE = 1e-6
    N_lTTs = np.logspace(-2, 2, 5)
    #fig, axes = plt.subplots(nrows=1, ncols=2, 
    #                         gridspec_kw={"width_ratios":[1,0.05], "wspace":0.1})
    #ax = axes[0]
    #cax = axes[1]
    #sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
    #        norm=mpl.colors.LogNorm(vmin=N_lTTs.min(), vmax=N_lTTs.max()))
    #sm._A = []
    #fig.colorbar(sm, cax=cax, label=r'$N_\ell^\mathrm{TT}$')
    fig, ax = plt.subplots(1)

    N_lTTs = [1e-6]

    for i, N_lTT in enumerate(N_lTTs):
        print(i, len(N_lTTs))
        ell, EE, TE, TT = get_spectra(8, 0, lmax=lmax, spectra=True, all_spectra=True)
        Clhat = np.array([TT+N_lTT, EE+N_lEE, EE*0, TE, EE*0, EE*0])
        lnPE = []
        lnPW = []
        taus = []
        for zre in zres:
            lnPE.append(lnprob_EE_ell(zre, 0, Clhat[1], N_l=N_lEE))
            lnPW.append(lnprob_wish_ell(zre, 0, Clhat, N_lT=N_lTT, N_lE=N_lEE))
            zsplit, tau_lo, tau_hi = get_twotau(get_spectra(zre, 0, therm=True))
            taus.append(tau_lo)
        L1 = np.exp(np.sum(lnPW, axis=1))
        ax.plot(taus, L1, color='C0',
                alpha=1-i/len(N_lTTs))
        L2 = np.exp(np.sum(lnPE, axis=1))
        ax.plot(taus, L2, color='C1',
                alpha=1-i/len(N_lTTs))
    ax.set_ylim([-0.05, 1.05])
    ax.plot([],[],color='C0',linestyle='-',label='Wishart Distribution')
    ax.plot([],[],color='C1',linestyle='-',label=r'$\chi^2$ Distribution')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$P(z_\mathrm{re}|x_e^\mathrm{min}=0,\{\hat{\boldsymbol C}_\ell\})$')
    legend = ax.legend(loc='best')


    fig.savefig('f5_alt.pdf', bbox_inches='tight')
    fig.savefig('f5_alt.png', bbox_inches='tight')
    plt.close('all')

    plt.figure()

    mean = 0.06
    sigma = 0.002
    popt,pcov = curve_fit(gaus,taus,L1,p0=[1,mean,sigma])
    s1 = popt[2]
    plt.plot(zres, L1, label=r'$\hat C_\ell^\mathrm{TT}+\hat C_\ell^\mathrm{TE}+\hat C_\ell^\mathrm{EE}$')
    #plt.plot(zres, gaus(zres, *popt))
    print(popt)
    popt,pcov = curve_fit(gaus,taus,L2,p0=[1,mean,sigma])
    plt.plot(taus, L2, label='EE')
    s2 = popt[2]
    #plt.plot(zres, gaus(zres, *popt))
    print(popt)
    print(s1, s2)

    popt = np.copy(popt)
    s3 = np.sqrt(s2**2+s1**2)
    popt[2] = s3
    popt[0] = 1.
    print(popt)
    print(gaus(taus, *popt))
    plt.plot(taus, gaus(taus, popt[0], popt[1], popt[2]), label=r'TT+TE')
    plt.ylabel(r'$P(\tau|x_e^\mathrm{min}=0,\{\hat{\boldsymbol C}_\ell\})$')
    plt.xlabel(r'$\tau$')
    legend = plt.legend(loc='upper left')
    plt.savefig('f5_alt.pdf', bbox_inches='tight')
    plt.savefig('f5_alt.png', bbox_inches='tight')

    print(s1, s2, s3)

    plt.show()


    return

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def fig6(num=50, lmax=40, wp=0):
    '''
    Make plots of the chi2 per multipole when you change both zre and xe.

    First plot is zre centered at 8, varying between 7 and 9. Remember that
    \chi^2 = 0 by definition when \hat C_\ell = C_\ell, so that'll always be the
    minimum. Have the same colormaps as in figure 1. The \hat C_\ell will be the
    theory specs for z_re = 8 and x_e = 0.
    '''
    N_ell = (wp*np.pi/180/60)**2
    lw = 1.5 # just want to make the lines a bit thicker so they overlap more
    zs = np.linspace(6, 9, num)
    xes = np.linspace(0, 0.05, num)
    #xes = np.linspace(0, 0.1, num)
    z0 = 6
    xe0 = 0
    cm1 = plt.cm.magma
    cm2 = plt.cm.copper

    ell, EE, TE, TT = get_spectra(z0, xe0, lmax=lmax, spectra=True, all_spectra=True)
    Clhat = np.array([TT, EE+N_ell, EE*0, TE, EE*0, EE*0])

    ell = np.arange(lmax+1)

    fig, axes = plt.subplots(nrows=1, ncols=2, 
                             gridspec_kw={"width_ratios":[1,0.05], "wspace":0.1})
    ax = axes[0]
    cax = axes[1]
    sm = plt.cm.ScalarMappable(cmap=cm1,
            norm=plt.Normalize(vmin=zs.min(), vmax=zs.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$z_\mathrm{re}$')

    for i in range(num):
        lnPE = []
        lnPW = []
        lnPW = lnprob_wish_ell(zs[i], xe0, Clhat, N_lE=N_ell)
        ax.plot(ell[2:], -2*lnPW[2:], color=cm1(i/num), lw=lw)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\chi^2_\mathrm{eff,\ell}$')
    if wp == 0: print(ax.get_ylim())
    #ax.set_ylim((-1.6012803451985382, 33.626887249168966))
    ax.set_ylim((0, 25))
    ax.set_xlim(xmin=2)
    ticks = [2,10,20,30,40]
    ax.set_xticks(ticks)
    if wp == 0:
        fig.savefig('f6a.pdf', bbox_inches='tight')
        fig.savefig('f6a.png', bbox_inches='tight')
    plt.title('${0}\ \mathrm{{\mu K\,arcmin}}$'.format(wp), ha='right')
    fig.savefig(f'f6a_{wp}.pdf', bbox_inches='tight')
    fig.savefig(f'f6a_{wp}.png', bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=2, 
                             gridspec_kw={"width_ratios":[1,0.05], "wspace":0.1})
    ax = axes[0]
    cax = axes[1]
    sm = plt.cm.ScalarMappable(cmap=cm2,
            norm=plt.Normalize(vmin=xes.min(), vmax=xes.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$x_e^\mathrm{min}$')

    for i in range(num):
        lnPE = []
        lnPW = []
        lnPW = lnprob_wish_ell(z0, xes[i], Clhat, N_lE=N_ell)
        ax.plot(ell[2:], -2*lnPW[2:], color=cm2(i/num), lw=lw)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\chi^2_\mathrm{eff,\ell}$')
    if wp == 0: print(ax.get_ylim())
    #ax.set_ylim((-1.2331949469091026, 25.89709388509082))
    ax.set_ylim((0, 25))
    ax.set_xlim(xmin=2)
    ax.set_xticks(ticks)
    
    if wp == 0:
        fig.savefig('f6b.pdf', bbox_inches='tight')
        fig.savefig('f6b.png', bbox_inches='tight')
    plt.title('${0}\ \mathrm{{\mu K\,arcmin}}$'.format(wp), ha='right')
    fig.savefig(f'f6b_{wp}.pdf', bbox_inches='tight')
    fig.savefig(f'f6b_{wp}.png', bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=2, 
                             gridspec_kw={"width_ratios":[1,0.05], "wspace":0.1})
    ax = axes[0]
    cax = axes[1]
    sm = plt.cm.ScalarMappable(cmap=cm1,
            norm=plt.Normalize(vmin=zs.min(), vmax=zs.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$z_\mathrm{re}$')

    for i in range(num):
        lnPE = lnprob_EE_ell(zs[i], xe0, Clhat[1], N_l=N_ell)
        ax.plot(ell[2:], -2*lnPE[2:], color=cm1(i/num), lw=lw)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\chi^2_\mathrm{eff,\ell}$')
    plt.title('${0}\ \mathrm{{\mu K\,arcmin}}$'.format(wp), ha='right')
    if wp == 0: print(ax.get_ylim())
    ax.set_ylim((0, 17.538341295423095))
    fig.savefig(f'f6a_EE_{wp}.pdf', bbox_inches='tight')
    fig.savefig(f'f6a_EE_{wp}.png', bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=2, 
                             gridspec_kw={"width_ratios":[1,0.05], "wspace":0.1})
    ax = axes[0]
    cax = axes[1]
    sm = plt.cm.ScalarMappable(cmap=cm2,
            norm=plt.Normalize(vmin=xes.min(), vmax=xes.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$x_e^\mathrm{min}$')

    for i in range(num):
        lnPE = lnprob_EE_ell(z0, xes[i], Clhat[1], N_l=N_ell)
        ax.plot(ell[2:], -2*lnPE[2:], color=cm2(i/num), lw=lw)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\chi^2_\mathrm{eff,\ell}$')
    plt.title('${0}\ \mathrm{{\mu K\,arcmin}}$'.format(wp), ha='right')
    if wp == 0: print(ax.get_ylim())
    ax.set_ylim((0, 11.13736343356942))
    fig.savefig(f'f6b_EE_{wp}.pdf', bbox_inches='tight')
    fig.savefig(f'f6b_EE_{wp}.png', bbox_inches='tight')
    plt.close()


    fig, axes = plt.subplots(nrows=1, ncols=2, 
                             gridspec_kw={"width_ratios":[1,0.05], "wspace":0.1})
    ax = axes[0]
    cax = axes[1]
    sm = plt.cm.ScalarMappable(cmap=cm1,
            norm=plt.Normalize(vmin=zs.min(), vmax=zs.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$z_\mathrm{re}$')

    lnPE0 = lnprob_TE_ell(z0, xe0, Clhat[3], N_lE=N_ell)
    for i in range(num):
        lnPE = lnprob_TE_ell(zs[i], xe0, Clhat[3], N_lE=N_ell) - lnPE0
        ax.plot(ell[2:], -2*lnPE[2:], color=cm1(i/num), lw=lw)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\chi^2_\mathrm{eff,\ell}$')
    plt.title('${0}\ \mathrm{{\mu K\,arcmin}}$'.format(wp), ha='right')
    if wp == 0: print(ax.get_ylim())
    ax.set_ylim((0, 2.7800844991679634))
    fig.savefig(f'f6a_TE_{wp}.pdf', bbox_inches='tight')
    fig.savefig(f'f6a_TE_{wp}.png', bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=2, 
                             gridspec_kw={"width_ratios":[1,0.05], "wspace":0.1})
    ax = axes[0]
    cax = axes[1]
    sm = plt.cm.ScalarMappable(cmap=cm2,
            norm=plt.Normalize(vmin=xes.min(), vmax=xes.max()))
    sm._A = []
    fig.colorbar(sm, cax=cax, label=r'$x_e^\mathrm{min}$')

    for i in range(num):
        lnPE = lnprob_TE_ell(z0, xes[i], Clhat[3], N_lE=N_ell) - lnPE0
        ax.plot(ell[2:], -2*lnPE[2:], color=cm2(i/num), lw=lw)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\chi^2_\mathrm{eff,\ell}$')
    plt.title('${0}\ \mathrm{{\mu K\,arcmin}}$'.format(wp), ha='right')
    if wp == 0: print(ax.get_ylim())
    ax.set_ylim((0, 1.7899325064530514))
    fig.savefig(f'f6b_TE_{wp}.pdf', bbox_inches='tight')
    fig.savefig(f'f6b_TE_{wp}.png', bbox_inches='tight')
    plt.close()
    return


def dertest(zre=8, x_e=0.05, dzre=5e-5, dxre=1e-5, lmin=2, lmax=100):
    for i in range(10):
        ell, EE, TE, TT = get_spectra(zre, x_e, lmax=lmax, spectra=True, all_spectra=True)
        ell, dEEx, dTEx, dTTx = get_spectra(zre, x_e+dxre, lmax=lmax, spectra=True, all_spectra=True)
        ell, dEEz, dTEz, dTTz = get_spectra(zre+dzre, x_e, lmax=lmax, spectra=True, all_spectra=True)
        dEEdz = (dEEz - EE)/dzre
        dTEdz = (dTEz - TE)/dzre
        dTTdz = (dTTz - TT)/dzre
        dEEdx = (dEEx - EE)/dxre
        dTEdx = (dTEx - TE)/dxre
        dTTdx = (dTTx - TT)/dxre
        plt.figure(1)
        plt.subplot(311)
        plt.semilogy(ell[2:], abs(dTTdx[2:]))
        plt.ylabel(r'$\partial C_\ell^\mathrm{TT}/\partial x$')
        print(f'Dertest {i+0}')
        print('dTdx')
        print(min(abs(dTTdx[2:])))
        print(max(abs(dTTdx[2:])))
        print('dTdz')
        print(min(abs(dTTdz[2:])))
        print(max(abs(dTTdz[2:])))
        print('\n')
        plt.subplot(312)
        plt.semilogy(ell[2:], abs(dTEdx[2:]))
        plt.ylabel(r'$\partial C_\ell^\mathrm{TE}/\partial x$')
        plt.subplot(313)
        plt.semilogy(ell[2:], abs(dEEdx[2:]))
        plt.ylabel(r'$\partial C_\ell^\mathrm{EE}/\partial x$')
        plt.xlabel(r'$\ell$')

        plt.figure(2)
        plt.subplot(311)
        plt.semilogy(ell[2:], abs(dTTdz[2:]))
        plt.ylabel(r'$\partial C_\ell^\mathrm{TT}/\partial z$')
        plt.subplot(312)
        plt.semilogy(ell[2:], abs(dTEdz[2:]))
        plt.ylabel(r'$\partial C_\ell^\mathrm{TE}/\partial z$')
        plt.subplot(313)
        plt.semilogy(ell[2:], abs(dEEdz[2:]))
        plt.ylabel(r'$\partial C_\ell^\mathrm{EE}/\partial z$')
        plt.xlabel(r'$\ell$')
    plt.figure(1)
    plt.savefig('dertest1.pdf', bbox_inches='tight')
    plt.figure(2)
    plt.savefig('dertest2.pdf', bbox_inches='tight')
    plt.close('all')

def test_sigma():
    zre = 7.5
    xe = 0.001
    lmin = 2
    lmax = 23
    F_ell  = get_F_ell(zre, xe,\
        N_lT=(0*np.pi/180/60)**2,\
        N_lE=(0*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
        tau_vars=True)
    F = np.sum(F_ell, axis=0)
    print('TT+TE+EE', np.round(1/F[0,0]**0.5, 4))

    F_ell  = get_F_ell_2(zre, xe,\
        N_lT=(0*np.pi/180/60)**2,\
        N_lE=(0*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
        tau_vars=True)
    F = np.sum(F_ell, axis=0)
    print('TT+EE   ', np.round(1/F[0,0]**0.5, 4))

    F_ell  = get_F_ell(zre, xe,\
        N_lT=(100000*np.pi/180/60)**2,\
        N_lE=(0*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
        tau_vars=True)
    F = np.sum(F_ell, axis=0)
    print('EE      ', np.round(1/F[0,0]**0.5, 4))

    F_ell  = get_F_ell_2(zre, xe,\
        N_lT=(0*np.pi/180/60)**2,\
        N_lE=(10000*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
        tau_vars=True)
    F = np.sum(F_ell, axis=0)
    print('TT      ', np.round(1/F[0,0]**0.5, 4))

    F_ell  = get_F_ell_3(zre, xe,\
        N_lT=(0*np.pi/180/60)**2,\
        N_lE=(0*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
        tau_vars=True)
    F = np.sum(F_ell, axis=0)
    print('TE      ', np.round(1/F[0,0]**0.5, 4))

    F_ell  = get_F_ell_4(zre, xe,\
        N_lT=(0*np.pi/180/60)**2,\
        N_lE=(0*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
        tau_vars=True)
    F = np.sum(F_ell, axis=0)
    print('TT+TE   ', np.round(1/F[0,0]**0.5, 4))

    F_ell  = get_F_ell_5(zre, xe,\
        N_lT=(0*np.pi/180/60)**2,\
        N_lE=(0*np.pi/180/60)**2, lmin=lmin, lmax=lmax,
        tau_vars=True)
    F = np.sum(F_ell, axis=0)
    print('TE+EE   ', np.round(1/F[0,0]**0.5, 4))


    return

def TT_constraint():
    lmax = 100
    ell = np.arange(lmax+1)
    tau = 0.05
    dtau = 0.01
    TT = get_TT(tau=tau, rescale=True)
    TT_delta = get_TT(tau=tau+dtau, rescale=True)
    der = (TT_delta + TT)/dtau
    F_ell = (2*ell+1)/2*(der/TT)**2
    plt.loglog(ell[2:], F_ell[2:])
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$F_{\tau\tau,\ell}^\mathrm{TT}$')
    
    plt.close('all')
    plt.loglog(ell[2:], TT[2:])
    plt.loglog(ell[2:], TT_delta[2:])
    plt.loglog(ell[2:], der[2:])
    plt.loglog(ell[2:], (der/TT)[2:]**2)
    plt.savefig('TT_fish.png', bbox_inches='tight')
    plt.savefig('TT_fish.pdf', bbox_inches='tight')
    ivar = sum(F_ell[2:])
    sigma_tau = 1/ivar**2
    print(sigma_tau)
    plt.show()
    return

'''
The big things Graeme pointed out were that I should be taking the means of the
log-likelihoods (or products of the likelihoods) for many different clhat
realizations, and how that would be a good way of showing what the different
distributions look like.
'''

def fig_mean_realizations(n_theta, n_real, zre=8, lmin=2, lmax=100):
    zres = np.linspace(4,12,n_theta)

    ell, EE, TE, TT = get_spectra(zre, 0, lmax=lmax, spectra=True,\
            all_spectra=True)
    Cl_th = np.array([TT, EE, 0*EE, TE])
    Cl_hats = []
    TEhats = []
    lnL_arr_TE = []
    lnL_arr_EE = []
    lnL_arr_wish = []

    plt.figure(figsize=(4,6))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)

    ax1.set_ylabel(r'$\mathcal L(z_\mathrm{re}|\hat C_\ell^\mathrm{TE})$')
    ax3.set_ylabel(r'$\mathcal L(z_\mathrm{re}|\hat{\boldsymbol C}_\ell)$')
    ax3.set_xlabel(r'$z_\mathrm{re}$')
    ax2.set_ylabel(r'$\mathcal L(z_\mathrm{re}|\hat C_\ell^\mathrm{EE})$')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    lnLs_TE = np.zeros_like(zres)
    lnLs_EE = np.zeros_like(zres)
    lnLs_wish = np.zeros_like(zres)
    for i in range(n_real):
        Cl_hat = hp.alm2cl(hp.synalm(Cl_th, new=True))
        Cl_hats.append(Cl_hat)
    for i in range(n_real):
        if i % int(n_real*0.05) == 0:
            print(i, n_real)
        for j in range(len(zres)):
            chi2_TE   = lnprob_TE_ell(zres[j], 0, Cl_hats[i][3])
            chi2_EE   = lnprob_EE_ell(zres[j], 0, Cl_hats[i][1])
            chi2_wish = lnprob_wish_ell(zres[j], 0, Cl_hats[i])

            lnLs_TE[j]   = sum(chi2_TE[lmin:])
            lnLs_EE[j]   = sum(chi2_EE[lmin:])
            lnLs_wish[j] = sum(chi2_wish[lmin:])
        
        lnL_arr_TE.append(lnLs_TE - max(lnLs_TE))
        lnL_arr_EE.append(lnLs_EE - max(lnLs_EE))
        lnL_arr_wish.append(lnLs_wish - max(lnLs_wish))

        a = 0.01
        if i < 500:
            ax1.plot(zres, np.exp(lnL_arr_TE[i]), color='k', alpha=a)
            ax2.plot(zres, np.exp(lnL_arr_EE[i]), color='k', alpha=a)
            ax3.plot(zres, np.exp(lnL_arr_wish[i]), color='k', alpha=a)


    plt.savefig('test2.pdf', bbox_inches='tight')
    plt.close()
    L1 = np.exp(np.nanmean(lnL_arr_TE, axis=0))
    L2 = np.exp(np.nanmean(lnL_arr_EE, axis=0))
    L3 = np.exp(np.nanmean(lnL_arr_wish, axis=0))    
    plt.plot(zres, L1, label=r'$\hat C_\ell^\mathrm{TE}$')
    plt.plot(zres, L2, label=r'$\hat C_\ell^\mathrm{EE}$')
    plt.plot(zres, L3, label=r'$\hat C_\ell^\mathrm{TT}+\hat C_\ell^\mathrm{TE}+\hat C_\ell^\mathrm{EE}$')

    mu1 = trapz(L1*zres, zres)/trapz(L1, zres)
    mu2 = trapz(L2*zres, zres)/trapz(L2, zres)
    mu3 = trapz(L3*zres, zres)/trapz(L3, zres)

    var1 = trapz(L1*(zres-mu1)**2, zres)/trapz(L1, zres)
    var2 = trapz(L2*(zres-mu2)**2, zres)/trapz(L2, zres)
    var3 = trapz(L3*(zres-mu3)**2, zres)/trapz(L3, zres)

    skew1 = trapz(L1*(zres-mu1)**3, zres)/trapz(L1, zres)/var1**1.5
    skew2 = trapz(L2*(zres-mu2)**3, zres)/trapz(L2, zres)/var2**1.5
    skew3 = trapz(L3*(zres-mu3)**3, zres)/trapz(L3, zres)/var3**1.5

    print(mu1, var1**0.5, skew1)
    print(mu2, var2**0.5, skew2)
    print(mu3, var3**0.5, skew3)

    #plt.axvline(8, color='k', linestyle=':')
    plt.xlim([5.5, 10.5])
    plt.xlabel(r'$z_\mathrm{re}$')
    plt.ylabel(r'$\langle\mathcal L(z_\mathrm{re}|\hat C_\ell)\rangle$')
    plt.legend(loc='best')
    plt.savefig('test.pdf', bbox_inches='tight')


def fig_mean_realizations_tau(n_theta, n_real, tau=0.06, lmin=2, lmax=100,
         wp=0):
    taus = np.linspace(0.02,0.0925, n_theta)

    N_l = (wp*np.pi/180/60)**2

    ell, EE, TE, TT = get_spectra_tau(tau, lmax=lmax, spectra=True,\
            all_spectra=True)
    Cl_th = np.array([TT, EE + N_l, 0*EE, TE])
    Cl_hats = []
    TEhats = []
    lnL_arr_TE = []
    lnL_arr_EE = []
    lnL_arr_wish = []

    plt.figure(figsize=(4,6))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)

    ax1.set_ylabel(r'$\mathcal L(\tau|\hat C_\ell^\mathrm{TE})$')
    ax3.set_ylabel(r'$\mathcal L(\tau|\hat{\boldsymbol C}_\ell$')
    ax3.set_xlabel(r'$\tau$')
    ax2.set_ylabel(r'$\mathcal L(\tau|\hat C_\ell^\mathrm{EE})$')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    lnLs_TE = np.zeros_like(taus)
    lnLs_EE = np.zeros_like(taus)
    lnLs_wish = np.zeros_like(taus)
    for i in range(n_real):
        Cl_hat = hp.alm2cl(hp.synalm(Cl_th, new=True))
        Cl_hats.append(Cl_hat)
    for i in range(n_real):
        if i % int(n_real*0.1) == 0:
            print(i, n_real)
        for j in range(len(taus)):
            if i == 0:
                if j % int(len(taus)*0.05) == 0:
                    print(j, len(taus))
            chi2_TE   = lnprob_TE_ell_tau(taus[j], Cl_hats[i][3], N_lE=N_l)
            chi2_EE   = lnprob_EE_ell_tau(taus[j], Cl_hats[i][1], N_l=N_l)
            chi2_wish = lnprob_wish_ell_tau(taus[j], Cl_hats[i], N_lE=N_l)

            lnLs_TE[j]   = sum(chi2_TE[lmin:])
            lnLs_EE[j]   = sum(chi2_EE[lmin:])
            lnLs_wish[j] = sum(chi2_wish[lmin:])

        lnLs_TE = np.array(lnLs_TE)
        lnLs_EE = np.array(lnLs_EE)
        lnLs_wish = np.array(lnLs_wish)

        dtau = np.diff(taus)
        dL = np.diff(lnLs_TE)
        der = dL/dtau
        d1 = der.std()*6  > abs(der).max()
        der = np.diff(lnLs_EE)/dtau
        d2 = der.std()*6  > abs(der).max()
        der = np.diff(lnLs_wish)/dtau
        d3 = der.std()*6  > abs(der).max()
        if (d1 and d2 and d3):
            lnL_arr_TE.append(lnLs_TE - max(lnLs_TE))
            lnL_arr_EE.append(lnLs_EE - max(lnLs_EE))
            lnL_arr_wish.append(lnLs_wish - max(lnLs_wish))

        a = 0.01
        if (i < 500) and (len(lnL_arr_TE) > 0):
            print(i, len(lnL_arr_TE))
            ax1.plot(taus, np.exp(lnL_arr_TE[-1]), color='k', alpha=a)
            ax2.plot(taus, np.exp(lnL_arr_EE[-1]), color='k', alpha=a)
            ax3.plot(taus, np.exp(lnL_arr_wish[-1]), color='k', alpha=a)


    plt.savefig('test2.pdf', bbox_inches='tight')
    plt.close('all')
    L1 = np.exp(np.nanmean(lnL_arr_TE, axis=0))
    L2 = np.exp(np.nanmean(lnL_arr_EE, axis=0))
    L3 = np.exp(np.nanmean(lnL_arr_wish, axis=0))    
    plt.plot(taus, L3/L3.max(), zorder=3, color='C0', label=r'$\hat C_\ell^\mathrm{TT}+\hat C_\ell^\mathrm{TE}+\hat C_\ell^\mathrm{EE}$')
    plt.plot(taus, L2/L2.max(), zorder=2, color='C1', label=r'$\hat C_\ell^\mathrm{EE}$')
    plt.plot(taus, L1/L1.max(), zorder=1, color='C2', label=r'$\hat C_\ell^\mathrm{TE}$')
    plt.plot([0, taus.min()], [0,0], color='C0', zorder=3)

    mu1 = trapz(L1*taus, taus)/trapz(L1, taus)
    mu2 = trapz(L2*taus, taus)/trapz(L2, taus)
    mu3 = trapz(L3*taus, taus)/trapz(L3, taus)

    var1 = trapz(L1*(taus-mu1)**2, taus)/trapz(L1, taus)
    var2 = trapz(L2*(taus-mu2)**2, taus)/trapz(L2, taus)
    var3 = trapz(L3*(taus-mu3)**2, taus)/trapz(L3, taus)

    skew1 = trapz(L1*(taus-mu1)**3, taus)/trapz(L1, taus)/var1**1.5
    skew2 = trapz(L2*(taus-mu2)**3, taus)/trapz(L2, taus)/var2**1.5
    skew3 = trapz(L3*(taus-mu3)**3, taus)/trapz(L3, taus)/var3**1.5

    print(wp)
    print(mu1, var1**0.5, skew1)
    print(mu2, var2**0.5, skew2)
    print(mu3, var3**0.5, skew3)

    #plt.axvline(8, color='k', linestyle=':')
    #plt.xlim([5.5, 10.5])
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$\langle\mathcal L(\tau|\hat C_\ell)\rangle$')
    plt.xlim([0.005, taus.max()])
    plt.minorticks_on()
    plt.legend(loc='best')
    if (wp == 0.65) | (wp == 0):
        plt.savefig(f'f5.pdf', bbox_inches='tight')
    plt.savefig(f'f5_{wp}_uKarcmin.pdf', bbox_inches='tight')

    #      TE    EE    all
    return (mu1, mu2, mu3), (var1, var2, var3), (skew1,skew2,skew3)



def fig_mean_realizations_tau_fixed(n_theta, n_realizations):
    wps = np.logspace(np.log10(0.65), np.log10(230), 25)
    noises = []
    means = []
    skews = []
    for wp in wps:
        mus, vars, sks = fig_mean_realizations_tau(n_theta, n_realizations, wp=wp)
        varTE, varEE, varall = vars
        muTE, muEE, muall = mus
        skewTE, skewEE, skewall = sks
        noises.append([varTE, varEE, varall])
        means.append([muTE, muEE, muall])
        skews.append([skewTE, skewEE, skewall])
    noises = np.array(noises)
    means = np.array(means)
    skews = np.array(skews)

    plt.figure()
    plt.plot(wps, noises[:,2]**0.5, label=r'$\sigma_{\tau}$ from $\hat C_\ell^\mathrm{TT}+\hat C_\ell^\mathrm{TE}+\hat C_\ell^\mathrm{EE}$')
    plt.plot(wps, noises[:,1]**0.5, label=r'$\sigma_{\tau}$ from $\hat C_\ell^\mathrm{EE}$')
    plt.plot(wps, noises[:,0]**0.5, label=r'$\sigma_{\tau}$ from $\hat C_\ell^\mathrm{TE}$')

    plt.xlabel(r'$w_p^{-1/2}\ [\mathrm{\mu K\,arcmin}]$')
    plt.ylabel(r'$\sigma_\tau$')
    plt.ylim(ymin=0)
    plt.xscale('log')
    yticks = np.arange(0.000, plt.gca().get_ylim()[1], 0.002)
    plt.yticks(yticks, yticks)
    #plt.axhline(0.007, color='k', linestyle='--',
    #        label=r'$\sigma_\tau$ from \textit{Planck} $\hat C_\ell^\mathrm{EE}$')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    plt.xticks([1,10,100],[1,10,100])
    plt.xlim([0.65, 230])
    plt.legend(loc='best')

    plt.savefig('test_noises.pdf', bbox_inches='tight')
    plt.savefig('comp_sigma_taus.pdf', bbox_inches='tight')

    plt.figure()

    plt.plot(wps, means[:,2], label=r'$\mu_{\tau}$ from $\hat C_\ell^\mathrm{TT}+\hat C_\ell^\mathrm{TE}+\hat C_\ell^\mathrm{EE}$')
    plt.plot(wps, means[:,1], label=r'$\mu_{\tau}$ from $\hat C_\ell^\mathrm{EE}$')
    plt.plot(wps, means[:,0], label=r'$\mu_{\tau}$ from $\hat C_\ell^\mathrm{TE}$')

    plt.xlabel(r'$w_p^{-1/2}\ [\mathrm{\mu K\,arcmin}]$')
    plt.ylabel(r'$\mu_\tau$')
    plt.xscale('log')

    plt.xticks([1,10,100],[1,10,100])
    plt.xlim([0.65, 230])
    plt.legend(loc='best')
    plt.savefig('test_means.pdf', bbox_inches='tight')

    plt.figure()
    plt.plot(wps, skews[:,2], label=r'$\gamma_{\tau}$ from $\hat C_\ell^\mathrm{TT}+\hat C_\ell^\mathrm{TE}+\hat C_\ell^\mathrm{EE}$')
    plt.plot(wps, skews[:,1], label=r'$\gamma_{\tau}$ from $\hat C_\ell^\mathrm{EE}$')
    plt.plot(wps, skews[:,0], label=r'$\gamma_{\tau}$ from $\hat C_\ell^\mathrm{TE}$')

    plt.xlabel(r'$w_p^{-1/2}\ [\mathrm{\mu K\,arcmin}]$')
    plt.ylabel(r'$\gamma_\tau$')
    plt.xscale('log')

    plt.xticks([1,10,100],[1,10,100])
    plt.xlim([0.65, 230])
    plt.legend(loc='best')
    plt.savefig('test_skews.pdf', bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    zre, xe = 6.75, 0.05
    n_theta = 100
    n_theta = 2**7
    #n_theta = 8
    nnoise = 30
    #nnoise = 4
    #_,taulo,tauhi=get_twotau(get_spectra(7, 0.05, therm=True))
    #print(taulo, tauhi, taulo+tauhi)
    #fig1(num=n_theta)
    #fig1_big(num=n_theta)
    #fig1_transp(num=n_theta)
    #fig1_alt(num=n_theta)
    #fig2(num=n_theta, n_noises=nnoise)
    #fig2_alt(num=n_theta, n_noises=nnoise)
    #fig2_wrong(num=n_theta, n_noises=nnoise)
    #fig3_rednoise(zre=7.2, xe=0.03, lmin=2, lmax=29)
    #for n in np.arange(0, 200, 30):
    #    fig3_ell_var(noise=n)
    #fig3_ell_var(zre=7, noise=0)
    #   fig3_alt(lmin=2, ntests=0)
    #fig3(zre=zre, lmin=2, lmax=100, ntests=0)
    #fig3(zre=7, xe=0.001, lmin=2, lmax=100, ntests=0)
    #fig3_taus(zre=zre)
    #fig3_taus_forcezsplit(xe=0.02)
    #noise_vs_uncertainty(zre=zre, xe=0.05)
    #   fig4(num=n_theta, n_noises=nnoise)
    #   fig4_alt(num=n_theta, n_noises=nnoise)
    #fig5(num=n_theta)
    #fig5_alt(num=n_theta)
    #for wp in np.arange(0, 201, 10):
    #    fig3_ell_var(noise=wp)
    #    fig6(num=n_theta*2-1, wp=wp)
    #fig3_ell_var(zre=zre, noise=0)
    #fig6(num=n_theta*2-1, wp=0)

    #testfig3(ntests=2)
    #dertest()
    #test_sigma()
    #best_case_versus_worst(xe=0)
    #TT_constraint()
    cartoons()
    n_realizations = 50000
    #n_realizations = 50
    n_theta = 251
    #n_theta = 25

    #fig_mean_realizations(n_theta, n_realizations)
    #fig_mean_realizations_tau(n_theta, n_realizations)
    #fig_mean_realizations_tau_fixed(n_theta, n_realizations)
    n_realizations = 5000
    n_theta = 251
    #print('2,9')
    #fig3(zre=zre, lmin=2, lmax=9, ntests=0)
    #print('10,19')
    #fig3(zre=zre, lmin=10, lmax=19, ntests=0)
    #print('20,29')
    #fig3(zre=zre, lmin=20, lmax=29, ntests=0)
    #print('30,99')
    #fig3(zre=zre, lmin=30, lmax=99, ntests=0)
    #print('2,99')
    #fig3(zre=zre, lmin=2, lmax=99, ntests=0)

    #print('2,9')
    #fig3_taus(zre=zre, lmin=2, lmax=9, ntests=0)
    #print('10,19')
    #fig3_taus(zre=zre, lmin=10, lmax=19, ntests=0)
    #print('20,29')
    #fig3_taus(zre=zre, lmin=20, lmax=29, ntests=0)
    #print('30,99')
    #fig3_taus(zre=zre, lmin=30, lmax=99, ntests=0)
    #print('2,99')
    #fig3_taus(zre=zre, lmin=2, lmax=99, ntests=0)
