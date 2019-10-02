import matplotlib.pyplot as plt
import numpy as np


from tools import get_spectra, get_spectra_complex, get_tau_z

skip = 10

zre = 7.25
xre = 0
therm = get_spectra(zre, xre, therm=True)


therm_unbump = get_spectra(6, 0, therm=True)

plt.plot(therm['z'], therm['g [Mpc^-1]'], label='No high-z')

zre = 6
xre = 0.04
thermhiz = get_spectra(zre, xre, therm=True)
plt.plot(thermhiz['z'], thermhiz['g [Mpc^-1]'], label=r'10\%')

plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.xlim([1, 2e3])
plt.ylim([1e-9, 1])


fig, axes =  plt.subplots(1,3)
z = therm_unbump['z']
xe = therm_unbump['x_e']

bump = np.exp(-(z-30)**2/(2*1**2))*0.2
bump += xe


inds = (z > 6)
ind25 = np.argmin(abs(z -25))
z25 = z[ind25]
xe25 = xe[ind25]
ind9 = np.argmin(abs(z - 9))
z9 = z[ind9]
xe9 = xe[ind9]
xe = xe[inds][::skip*2]
z = z[inds][::skip*2]
#z[0] = 0
#xe[-1] = 0
#therm2 = get_spectra_complex(z, xe, therm=True)


bump[-1] = 0
inds = ( (therm_unbump['z'] > 25) & (therm_unbump['z'] < 35))
print(z[:10], xe[:10])
z2 = therm_unbump['z'][inds]
z2 = np.concatenate((z[:10], [z9], [z25], z2, [35]))
bump = bump[inds]
bump = np.concatenate((xe[:10], [xe9], [xe25], bump, [0]))
thermbump = get_spectra_complex(z2, bump, therm=True)

#der = np.diff(xe+bump)/np.diff(z)
#plt.figure()
#plt.plot(z[1:], abs(der)/(xe+bump)[1:])
#plt.plot(z, xe+bump)


#z2, xe2 = thermhiz['z'], thermhiz['x_e']
#inds = (z2 < 40)
#z2 = z2[inds]
#xe2 = xe2[inds]
#der = np.diff(xe2)/np.diff(z2)
#plt.figure()
#plt.plot(z2[1:], abs(der)/xe2[1:])
#plt.plot(z2, xe2)
#
#plt.show()



axes[1].plot(therm['z'], therm['g [Mpc^-1]'], label='Original')
axes[1].plot(thermhiz['z'], thermhiz['g [Mpc^-1]'], label='10\% rise')
axes[1].plot(thermbump['z'], thermbump['g [Mpc^-1]'], label='Bump')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].legend(loc='best')
axes[1].set_xlim([1, 2e3])
axes[1].set_ylim([1e-9, 1])
axes[1].set_xlabel(r'$z$')
axes[1].set_ylabel(r'Visibility function $g(z)$')

#plt.figure()
axes[0].plot(therm['z'], therm['x_e'])
axes[0].plot(thermhiz['z'], thermhiz['x_e'])
axes[0].plot(thermbump['z'], thermbump['x_e'])
#axes[1].plot(z, xe+bump, 'o-')
axes[0].set_xlim([0, 50])
axes[0].set_xlabel(r'$z$')
axes[0].set_ylabel(r'$x_e$')


z, tau = get_tau_z(therm)
inds = (z[1:] < 50)
axes[2].plot(z[1:][inds], tau[inds])
z, tau = get_tau_z(thermhiz)
inds = (z[1:] < 50)
axes[2].plot(z[1:][inds], tau[inds])
z, tau = get_tau_z(thermbump)
inds = (z[1:] < 50)
axes[2].plot(z[1:][inds], tau[inds])
axes[2].set_xlabel(r'$z$')
axes[2].set_ylabel(r'$\tau$')


plt.figure()

ell, EE, TE = get_spectra(zre, 0, spectra=True)
plt.loglog(ell[2:], EE[2:])

ell, EE, TE = get_spectra(zre, xre, spectra=True)
plt.loglog(ell[2:], EE[2:])

ell, EE, TE = get_spectra_complex(z2, bump, spectra=True)
plt.loglog(ell[2:], EE[2:])

plt.figure()
plt.loglog(therm['z'], therm['x_e'])
plt.loglog(thermhiz['z'], thermhiz['x_e'])
plt.loglog(thermbump['z'], thermbump['x_e'])
plt.show()
