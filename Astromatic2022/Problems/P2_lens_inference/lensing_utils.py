import numpy as np
from astropy import constants as csts
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
import scipy.interpolate as interp
from scipy.stats import truncnorm

# Values calculated from a sample of corrected photometric redshift from SDSS-III imaging data
# (https://www.sdss.org/dr14/data_access/value-added-catalogs/?vac_id=photometric-redshift-distributions)
Z_MEAN = 0.353412
Z_STD = 0.15404023
Z_MIN = (0.08 - Z_MEAN) / Z_STD
Z_MAX = np.inf
ZELL_DIST = lambda size: truncnorm.rvs(Z_MIN, Z_MAX, Z_MEAN, Z_STD, size)

# Redshift distance arbitrarily chosen to replicate SLAC lenses (https://arxiv.org/pdf/1711.00072.pdf)
ZDIFF_MEAN = 0.5
ZDIFF_STD = 0.5
ZDIFF_MIN = (0.3 - ZDIFF_MEAN) / ZDIFF_STD
ZDIFF_MAX = (2 - ZDIFF_MEAN) / ZDIFF_STD


def zs_from_zl(zl):
	return zl + np.random.uniform()


def D_i_j(z_i, z_j):
	"""
    Angular diameter distance in Mpc between redshifts z_i and z_j
    with z_i < z_j
    """
	return cosmo.angular_diameter_distance_z1z2(z_i, z_j).value


def theta_E_from_M(M, zl, zs):
	"""
    Einstein radius in arcsecs of a mass M [Msun] at redshift zl, with source at zs
    """
	mass = M * u.Msun
	Dls = D_i_j(zl, zs) * u.Mpc
	Dl = D_i_j(0, zl) * u.Mpc
	Ds = D_i_j(0, zs) * u.Mpc

	theta_E = (np.sqrt(4 * csts.G * mass / csts.c ** 2 * Dls / Dl / Ds)).to(
		u.dimensionless_unscaled) * u.rad

	return theta_E.to(u.arcsec).value


def sp_ray_tracing(x1, x2, a1, a2):
	y1 = x1 - a1
	y2 = x2 - a2
	return y1, y2


def lens_source(x1, x2, y1, y2, source, npix):
	im = interp.griddata(points=(x1.ravel(), x2.ravel()), values=source.ravel(), xi=(y1.ravel(), y2.ravel()), method="linear", fill_value=0.)
	return im.reshape(npix, npix)


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import matplotlib as mpl
	z = ZELL_DIST(10)

	plt.style.use("science")
	z_ell = np.linspace(0.01, 1, 100)
	zs = z_ell + 0.1
	mass = 10**np.linspace(8.5, 13.5)

	plt.figure(figsize=(5, 5))
	cmap = mpl.cm.Spectral
	norm = mpl.colors.LogNorm(vmin=z_ell.min(), vmax=z_ell.max())
	for i, zl in enumerate(z_ell):
		# for zs in z_source[z_source > zl]:
		te = theta_E_from_M(mass, zl, zs[i])
		plt.semilogx(mass, te, "-", color=cmap(zl))

	ax = plt.gca()
	plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', label='Lens redshift')
	plt.ylim(0.5, 3)
	plt.title("Apparent size of Einstein rings")
	plt.xlabel(r"Lens mass [$M_\odot$]")
	plt.ylabel(r"$\theta_E$")
	plt.show()


