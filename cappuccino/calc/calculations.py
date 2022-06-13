import numpy as np
from cappuccino.utils import constants as const

def calcVirialProductPosterior(tau, dv, nsamp): #veltype=["meanfwhm", "meansigma", "rmssigma", "rmsfwhm"]):
	"""
	Calculates the virial product using the Tauccf parameter, and the line
	widths defined in veltype. Saves a posterior sample of the same length
	as the other posteriors. Errors are propagated by assuming Gaussian
	errors in Tauccf and velocities and drawing random samples.

	:param veltype: List of the line with measurement types to use
	:type veltype: list

	:return: None
	"""
	# Virial Product
	print "-Using lag (days) = %.2f +%.2f/-%.2f" % (tau[0], tau[2], tau[1])
	tau_post = tau[0] + np.mean([tau[1], tau[2]]) * np.random.randn(nsamp)
	# Don't allow negative lags
	for i in range(len(tau_post)):
		while tau_post[i] < 0:
			tau_post[i] = tau[0] + np.mean([tau[1], tau[2]]) * np.random.normal()
	R = tau_post * const.ckm * const.day

	print "-Using %s (km/s) = %.2f +%.2f/-%.2f" % (v, dv[0], dv[2], dv[1])
	dv_post = dv[0] + np.mean([dv[1], dv[2]]) * np.random.randn(nsamp)
	# Don't allow negative line widths
	for i in range(len(dv_post)):
		while dv_post[i] < 0:
			dv_post[i] = dv[0] + np.mean([dv[1], dv[2]]) * np.random.normal()
	log10VP = np.log10((R * dv_post ** 2) / const.G)

	return log10VP