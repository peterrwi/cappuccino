import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import buildFigures as bf
from cappuccino.utils import parInfo, util
import pandas as pd
reload(parInfo)

ndigits = parInfo.paramDigits()
texnames = parInfo.paramTexNames()
texunits = parInfo.paramTexUnits()

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True

def plotPosteriors(posteriors, params=[], logify={}, names=[], truth={}, confints=None, _type='hist', figsize=(13,6), colors=None, dim=[], nbins=15, legendloc=[-1,-1], savename=''):
    """
    Plot one or multiple posterior distributions

    :param posteriors: List of posteriors to plot. If Model objects are provided, will extract the
        posteriors, names, and confidence intervals.
    :type posteriors: List of Pandas DataFrames

    :param params: List of parameters to plot. Defaults to 'log10Mbh', 'Thetao', 'Thetai', 'Rmean',
        'Rmedian', 'Rmin', 'Taumean', 'Taumedian', 'Beta', 'Xi', 'Kappa', 'Gamma', 'Fellip',
        'Fflow', 'Thetae', 'Sigmaturb'
    :type params: List of strings

    :param logify: Dictionary describing if parameters should be plotted on a log scale.
    :type logify: dictionary of bools

    :param names: List of run names. If Model objects are provided, will use Model.runname values
    :type names: list of strings

    :param confints: List of confidence intervals. If Model objects are provided to posteriors,
        will use those computed confidence intervals. Set to False to hide.
    :type confints: List of dictionaries

    :param _type: 'hist' for histogram or 'kde' for kernel density estimate. Defaults to 'hist'.
    :type _type: str

    :param figsize: Size of figure, in inches. Defaults to (width, height) = (13,6)
    :type figsize: tuple, length 2

    :param colors: Colors for the histograms
    :type colors: list of strings

    :param dim: Number of rows, number of colums. Will run util.findDimensions by default
    :type dim: tuple or list, length 2

    :param nbins: Number of bins for histogram. Defaults to 15.
    :type nbins: int

    :param legendloc: Which subplot to place the legend. Defaults to upper left
    :type legendloc: list or tuple, length 2

    :param savename: Filepath to save figure. Default won't save.
    :type savename: str

    :return: ax
    """
    if type(posteriors) != list:
        posteriors = [posteriors]
    
    # If Models are provided, extract information from each
    if not isinstance(posteriors[0], pd.DataFrame):
        models = [p for p in posteriors]
        posteriors = [m.posterior for m in models]
        if confints is None:
            confints = [m.confints for m in models]
        if names == []:
            names = [m.runname for m in models]
    if type(confints) != list:
        if confints == False:
            confints = None
        confints = [confints] * len(posteriors)
    if type(names) != list:
        names = [names] * len(posteriors)
    if type(_type) != list:
        _type = [_type]
    if params==[]:
        params = [
            'log10Mbh',
            'Thetao','Thetai',
            'Rmean','Rmedian','Rmin',
            'Taumean','Taumedian',
            'Beta','Xi','Kappa','Gamma',
            'Fellip','Fflow','Thetae',
            'Sigmaturb',
        ]
    nparams = len(params)
    if dim == []:
        dim = util.findDimensions(nparams)

    if legendloc == [-1,-1]:
        if len(params) == dim[0] * dim[1]:
            legendloc = [0,0]

    if colors is None:
        try:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        except:
            colors = plt.rcParams['axes.color_cycle']

    for param in params:
        if param not in logify.keys():
            logify[param] = False

    fig,ax = plt.subplots(*dim, figsize=figsize)
    ax = np.atleast_2d(ax)
    i_row,i_col = 0,0
    for param in params:
        # Take log if necessary
        to_plot = []
        if logify[param]:
            for i in range(len(posteriors)):
                try:
                    to_plot.append(np.log10(posteriors[i][param]))
                except:
                    to_plot.append([])
                    confints[i][param] = None
        else:
            for i in range(len(posteriors)):
                try:
                    to_plot.append(posteriors[i][param])
                except:
                    to_plot.append([])
                    confints[i][param] = None
        
        if 'hist' in _type:
            ax[i_row,i_col].hist(
                to_plot,
                histtype='step',
                bins=nbins,
                lw=2,
                normed=True,
                label=[name for name in names],
                color=colors[:len(posteriors)],
            )
        if 'kde' in _type:
            from scipy import stats
            xlims = min(min(vals) for vals in to_plot), max(max(vals) for vals in to_plot)
            xvals = np.linspace(
                xlims[0] - 0.05 * (xlims[1]-xlims[0]),
                xlims[1] + 0.05 * (xlims[1]-xlims[0]),
                100
            )
            for i in range(len(to_plot)):
                kernel = stats.gaussian_kde(to_plot[i])
                ax[i_row,i_col].plot(xvals, kernel(xvals), lw=2, color=colors[i], label=names[i])

        for i in range(len(confints)):
            if not confints[i] is None:
                try:
                    if logify[param]:
                        tmp = [
                            np.log10(confints[i][param][0]),
                            np.log10(confints[i][param][0]) - np.log10(confints[i][param][0] - confints[i][param][1]),
                            np.log10(confints[i][param][0] + confints[i][param][2]) - np.log10(confints[i][param][0])           
                        ]
                    else:
                        tmp = [confints[i][param][0], confints[i][param][1], confints[i][param][2]]
                    ax[i_row,i_col].axvspan(
                        tmp[0]-tmp[1],
                        tmp[0]+tmp[2],
                        color=colors[i],
                        alpha=0.1
                    )
                    ax[i_row,i_col].axvline(
                        tmp[0],
                        ls='dashed',
                        color=colors[i],
                    )
                except:
                    pass
        if param in truth.keys():
            ax[i_row,i_col].axvline(truth[param], color='red')
        # Set the axis label
        xlabel = texnames[param]
        if logify[param]:
            xlabel = "$\\log_{10}($%s$)$" % xlabel
        if texunits[param] != "":
            xlabel = "%s (%s)" % (xlabel, texunits[param])
        ax[i_row,i_col].set_xlabel(xlabel)
        
        if len(posteriors) != 1:
            if i_col == legendloc[0] and i_row == legendloc[1]:
                ax[i_col,i_row].legend()

        ax[i_row, i_col].set_yticks([])
        
        if i_col == dim[1]-1:
            i_col = 0
            i_row += 1
        else:
            i_col += 1
    if i_row < dim[0]:
        #ax[i_row, i_col].get_xaxis().set_visible(False)
        #ax[i_row, i_col].get_yaxis().set_visible(False)
        for i in range(len(names)):
            ax[i_row,i_col].hist([], histtype='step',lw=2,label=names[i], color=colors[i])
            ax[i_row,i_col].set_ylim(2,3)
        #ax[i_row, i_col].hist([],label='Test')
        ax[i_row, i_col].legend(loc='upper left')
        ax[i_row, i_col].axis('off')
    plt.tight_layout()
    if savename != "":
        plt.savefig(savename)
    return fig,ax
