import numpy as np 
import exshalos
import pylab as pl
from scipy.interpolate import interp1d, CubicSpline, splrep, splev
from scipy.special import spherical_jn
from scipy.signal import savgol_filter
import hankl
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import simps

#Compute the FFTLog
def fftlog(k, P, q, mu):
    rr, xi = hankl.FFTLog(k, P*np.power(k, q-0.5), q=0.0, mu=mu+0.5)
    xi = xi/np.power(2.0*np.pi*rr, 1.5)

    return np.array([rr, xi])

#Parameters
Lambda = 0.7
kmin = 1e-5
kmax = 0.7
nmin = 2
nmax = 10

#Open the linear power spectrum
kl, Pl = np.loadtxt("MDPL2_z00_matterpower.dat", unpack = True)

k = kl#[(kl>kmin)*(kl<kmax)]
P = Pl#[(kl>kmin)*(kl<kmax)]

P[k>Lambda] = 0.0
fP = splrep(k, P, s=0)

print(len(k), len(k[(k>1.0/400.0)*(k<1.0)]))
Nk = int(len(k))
#k = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)
#P = splev(k, fP, der=0)

#Compute P11
print("Computing P11")
#Pclpt = exshalos.theory.CLPT_Powers(k, P, Lambda = Lambda, kmax = kmax, nmin = nmin, nmax = nmax, verbose = False)

#Compute with pyfftlog
ksi = fftlog(k, P, 2.0, 0.0)[1]
r, ksi2 = hankl.P2xi(k, P, l=0)

zeta = fftlog(k, P, 1.0, 1.0)[1]

psiR = fftlog(k, P, 0.0, 0.0)[1] #- 2.0*fftlog(k, P, -1.0, 1.0)[1]/r

psiT = fftlog(k, P, -1.0, 1.0)[1]/r
                                      
psiB2 = fftlog(k, P, 1.0, 3.0)[1]

A4 = simps(P, k)/3.0
print(A4)
#Compute the correlation function
Xi = exshalos.utils.Compute_Correlation(k, P, direction = 1, verbose = False)

#Plot the spherical bessel functions
pl.clf()

x = np.linspace(0.0, 10.0, 1000)

pl.plot(x, spherical_jn(0, x)*x*x, color = "red")
pl.plot(x, spherical_jn(1, x)*x, color = "blue")
pl.plot(x, spherical_jn(0, x), color = "darkgreen")
pl.plot(x, spherical_jn(1, x)/x, color = "purple")

pl.grid(True)

#pl.show()

#Plot the spectra
pl.clf()
mask = r > 0.0

pl.subplot(321)
pl.plot(r[mask], ksi[mask], color = "red")
pl.plot(r[mask], ksi2[mask], color = "blue")
#pl.plot(Xi["R"][mask], Xi["Xi"][mask], color = "black")
pl.xscale("log")
pl.yscale("log")
pl.ylabel(r"$\xi _{2,0}$", fontsize = 15)
pl.grid(True)
#pl.xlim(0.01, 400)

pl.subplot(322)
pl.plot(r[mask], zeta[mask], color = "red")
pl.plot(r[mask], gaussian_filter1d(zeta[mask], 2), color = "blue")
pl.xscale("log")
pl.yscale("log")
pl.ylabel(r"$\xi _{1,1}$", fontsize = 15)
pl.grid(True)
#pl.xlim(0.01, 400)

pl.subplot(323)
pl.plot(r[mask], psiR[mask], color = "red")
pl.plot(r[mask], gaussian_filter1d(psiR[mask], 2), color = "blue")
pl.xscale("log")
pl.yscale("log")
pl.ylabel(r"$\xi _{0,0}$", fontsize = 15)
pl.grid(True)
#pl.xlim(0.01, 400)

pl.subplot(324)
pl.plot(r[mask], psiT[mask], color = "red")
pl.plot(r[mask], gaussian_filter1d(psiT, 2)[mask], color = "blue")
pl.plot(np.hstack([[0.0], r[mask][r[mask] > 2]]), np.hstack([[A4], gaussian_filter1d(psiT[mask], 2)[r[mask] > 2]]), color = "yellow")
pl.xscale("log")
pl.yscale("linear")
pl.ylim(0.0, 200.0)
pl.ylabel(r"$\xi _{-1,1}$", fontsize = 15)
pl.grid(True)
#pl.xlim(0.01, 400)

pl.subplot(325)
pl.plot(r[mask], psiB2[mask], color = "red")
pl.plot(r[mask], gaussian_filter1d(psiB2, 2)[mask], color = "blue")
pl.xscale("log")
pl.yscale("log")
pl.ylabel(r"$\xi _{1,3}$", fontsize = 15)
pl.grid(True)
#pl.xlim(0.01, 400)

pl.tight_layout()
pl.savefig("P11.pdf")
pl.show()
'''
pl.clf()

mask = psiT > 0.0

pl.plot(r[mask], np.exp(-psiT[mask]), color = "red")
pl.plot(r[mask], np.exp(-gaussian_filter1d(psiT, 2)[mask]), color = "blue")

pl.xscale("log")
pl.yscale("log")
pl.grid(True)
#pl.xlim(0.01, 400)

pl.show()'''