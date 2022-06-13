import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
from scipy import stats
from cappuccino.utils.util import confInt


def findConfInt(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level
    pass


def potatoPlot(x,y,ax,conf_levels=[0.6827, ],**contour_kwargs):
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()

    xmed = np.median(x)
    ymed = np.median(y)
    ax.plot(
        xmed, ymed,
        marker='o',
        markerfacecolor=contour_kwargs["colors"],
        markeredgewidth=0,
        alpha=contour_kwargs["alpha"]
    )

    if np.std(x) > 0 and np.std(y) > 0:
        X,Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(),Y.ravel()])
        values = np.vstack([x,y])
        try:
            kernel = stats.gaussian_kde(values)
            Z = np.reshape(kernel(positions), X.shape)
            Z /= np.sum(Z)  

            if 1 not in [np.isnan(Z[n, m]) for n in range(len(Z))
                         for m in range(len(Z[0]))]:
                levels = [so.brentq(findConfInt, 0., 1., args=(Z, conf))
                         for conf in conf_levels]   

                ax.contour(X, Y, Z, levels=levels, origin="lower",**contour_kwargs)
                return
        except:
            pass

    print "Couldn't produce contour. Reverting to regular errorbars."
    xint = confInt(x, 68.27)
    yint = confInt(y, 68.27)
    ax.errorbar(
        xint[0], yint[0],
        xerr=[[xint[1]], [xint[2]]],
        yerr=[[yint[1]], [yint[2]]],
        fmt='o',
        color=contour_kwargs["colors"]
    )


def densityContour(xdata, ydata, nbins_x, nbins_y, ax,
                   conf_levels=[0.68, ], **contour_kwargs):
    """ Create a density contour plot.
    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """

    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=(nbins_x, nbins_y),
                                       normed=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1, nbins_x))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y, 1))

    pdf = (H * (x_bin_sizes * y_bin_sizes))

    levels = [so.brentq(findConfInt, 0., 1., args=(pdf, conf))
              for conf in conf_levels]

    X, Y = 0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1])
    Z = pdf.T

    contour = ax.contour(X, Y, Z, levels=levels, origin="lower",**contour_kwargs)

    return contour


def angledError(x,y,ax,**plot_kwargs):
    cov = np.cov(x, y)

    # If there are any NaNs in the covariance matrix, ignore point
    if 1 in [np.isnan(cov[n, m]) for n in [0, 1] for m in [0, 1]]:
        print "Covariance matrix contains NaNs. Ignoring this data point."
        return
    xmed = np.median(x)
    ymed = np.median(y)
    ax.plot(
        xmed, ymed,
        marker='o',
        **plot_kwargs
    )
    eigval, eigvec = np.linalg.eig(cov)
    err = np.array([eigvec[:, 0] * eigval[0] ** 0.5,
                    eigvec[:, 1] * eigval[1] ** 0.5])
    ax.plot(
        [xmed - err[0, 0], xmed + err[0, 0]],
        [ymed - err[0, 1], ymed + err[0, 1]],
        **plot_kwargs
    )
    ax.plot(
        [xmed - err[1, 0], xmed + err[1, 0]],
        [ymed - err[1, 1], ymed + err[1, 1]],
        **plot_kwargs
    )

    pass


def normalError(x,y,ax,conf_level=0.6827,**plot_kwargs):
    xint = confInt(x, 100*conf_level)
    yint = confInt(y, 100*conf_level)
    ax.errorbar(
        xint[0],yint[0],
        xerr=[[xint[1]], [xint[2]]],
        yerr=[[yint[1]], [yint[2]]],
        fmt='o',
        **plot_kwargs
    )
    pass


def linearRegression(lm_filename,ax,conf_ints = [68.27], burnin=0):
    lmchain = np.load(lm_filename)
    chain = {
        'alpha': lmchain[burnin:, 0],
        'beta': lmchain[burnin:, 1],
        'sigsqr': lmchain[burnin:, 2],
        'corr': lmchain[burnin:, 3]
    }
    sigsqr50 = np.percentile(chain['sigsqr'], 50)
    intscat = np.sqrt(sigsqr50)

    # Plot the linear regression
    xlim = ax.get_xlim()
    lineXs = np.linspace(xlim[0], xlim[1], 100, endpoint=True)
    for i in range(len(conf_ints)):
        percent = conf_ints[i]
        lineYs = mcmcLineFit(lineXs, chain, percent)
        ax.fill_between(
            lineXs, lineYs[1], lineYs[2],
            color='grey',
            alpha=0.5 - 0.2 * i,
            linestyle='-.'
        )

    lineYsmed = mcmcLineFit(lineXs, chain, 68.)[0]
    ax.plot(lineXs, lineYsmed, 'k--')
    ax.plot(
        lineXs, lineYsmed - intscat,
        color='k',
        linestyle='dotted'
    )
    ax.plot(
        lineXs, lineYsmed + intscat,
        color='k',
        linestyle='dotted'
    )

    pass


def mcmcLineFit(x, chain, percent):
    alphas = np.array(chain['alpha'])
    betas = np.array(chain['beta'])
    ymed, yhigh, ylow = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
    for i in range(len(x)):
        ys = alphas + betas * x[i]
        int = confInt(ys,percent)
        ymed[i],ylow[i],yhigh[i] = int[0],int[0]-int[1],int[0]+int[2]

    return [ymed, ylow, yhigh]