file_directory = '/'.join(__file__.split('/')[:-1]) + '/'

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cappuccino.utils import parInfo
from cappuccino.calc import posterior
#from ..calc import posteriors
#import plots
from cappuccino.plotting import plots
import buildFigures as bf
reload(posterior)
reload(parInfo)
reload(plots)


def plotCorrelations(post, varsx, varsy, plot_order=[], x_logify=False, y_logify=False,
                     savename='', usecovariance=False, plotType='normal',
                     fp_linmix='', lmsuffix='', label_agn=False, figwidth=7.,
                     pad_inches = [], transpose=False, _colors={}, contour_kwargs={}):
    """

    :param post: Dictionary of Posterior objects
    :type post: dict

    :param varsx: List of keys for x-axis variables
    :type varsx: list

    :param varsy: List of dict keys for x-axis variables
    :type varsy: list

    :param x_logify: List of booleans saying if the log of the x-axis variable should be plotted
    :type varsy: list

    :param y_logify: List of booleans saying if the log of the y-axis variable should be plotted
    :type varsy: list

    :param _savename: Filepath to save the figure
    :type varsy: str

    :param diagonalerrors: Indicate if should plot diagonal error bars, along axis of covariance
    :type varsy: bool

    :param fp_linmix: Filepath to where previous correlation calculations are located
    :type varsy: str

    :param savenew: Indicates if should save newly calculated LinMix correlation values
    :type varsy: bool

    :param lmsuffix:
    :type varsy: str

    :return:
    """
    if type(x_logify) != list:
        x_logify = [x_logify] * len(varsx)
    if type(y_logify) != list:
        y_logify = [y_logify] * len(varsy)

    if len(plot_order) == 0:
        plot_order = post.keys()
    for p in post.keys():
        if post[p].campaign not in _colors.keys():
            _colors[post[p].campaign] = 'k'
    # Plotting values
    nrows = len(varsy)
    ncols = len(varsx)
    if pad_inches == []:
        pad_inches = [0.6, 0.1, 0.1, 0.4, 0.0, 0.0]

    mpl.rcParams['lines.linewidth'] = 0.5
    mpl.rcParams['lines.markersize'] = 2
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9
    mpl.rcParams['axes.labelsize'] = 9

    Figure = bf.CustomFigure()
    if transpose:
        Figure.fixedWidth(figwidth,pad_inches,(ncols,nrows))
    else:
        Figure.fixedWidth(figwidth,pad_inches,(nrows,ncols))

    ax = np.zeros((len(varsx),len(varsy))).tolist()

    for j in range(len(varsx)):
        varx = varsx[j]
        for l in range(len(varsy)):
            vary = varsy[l]
            print "\nWorking on %s vs. %s" % (vary, varx)

            x,y,cov,campaign,objects = posterior.extractConfIntsAndCov(
                post, varx, vary,
                x_logify=x_logify[j], y_logify=y_logify[l],
                agn_order=plot_order
            )
            colors = list()
            for i in range(len(campaign)):
                colors.append(_colors[campaign[i]])

            # Creat the subplot panel
            if transpose:
                ax[j][l] = Figure.newSubplot(position = (l, j))
            else:
                ax[j][l] = Figure.newSubplot(position = (j, l))
            xlim, ylim = parInfo.paramLims()[varx], parInfo.paramLims()[vary]
            ax[j][l].set_xlim(xlim), ax[j][l].set_ylim(ylim)

            xtexname, ytexname = parInfo.paramTexNames()[varx], parInfo.paramTexNames()[vary]
            xticks, yticks = parInfo.paramTicksCorrelation()[varx], parInfo.paramTicksCorrelation()[vary]

            # Setting the ticks and labels
            if x_logify[j]:
                xtexname = "$\\log_{10}(%s)$" % xtexname[1:-1]
            if y_logify[l]:
                ytexname = "$\\log_{10}(%s)$" % ytexname[1:-1]

            ax[j][l].set_xticks(xticks)
            ax[j][l].set_yticks(yticks)

            if l == len(varsy) - 1:
                ax[j][l].set_xlabel(xtexname)
            else:
                if transpose:
                    ax[j][l].set_xlabel(xtexname)
                else:
                    ax[j][l].set_xticklabels([])
            if j == 0:
                ax[j][l].set_ylabel(ytexname)
            else:
                ax[j][l].set_yticklabels([])

            # Plot the linear regression results from LinMix
            if usecovariance:
                _savenamecov = 'wcov'
            else:
                _savenamecov = 'nocov'
            lm_filename = "%s%s_vs_%s_%s_%s.npy" % (fp_linmix, vary, varx, _savenamecov, lmsuffix)
            plots.linearRegression(
                lm_filename,
                ax = ax[j][l],
                conf_ints=[68.27]
            )

            # Plot the data points
            for agn in plot_order:
                campaign = post[agn].campaign
                x = post[agn].posteriors[varx]
                y = post[agn].posteriors[vary]
                if x_logify[j]:
                    x = np.log10(x)
                if y_logify[l]:
                    y = np.log10(y)
                #contour_kwargs = {'colors': _colors[campaign], 'linewidths':3}
                plot_kwargs = {'color': _colors[campaign]}
                if np.shape(x) != () and np.shape(y) != () and \
                        1 not in [np.isnan(n) for n in y] and \
                        1 not in [np.isnan(n) for n in x]:
                    if plotType.lower() == 'angled':
                        plots.angledError(
                            x, y,
                            ax = ax[j][l],
                            **plot_kwargs
                        )
                    elif plotType.lower() == 'potato':
                        plots.potatoPlot(
                            x, y,
                            ax=ax[j][l],
                            conf_levels=[0.6827, ],
                            **contour_kwargs[campaign]
                        )
                    elif plotType.lower() == 'contour':
                        plots.densityContour(
                            x, y,
                            12, 12,
                            ax=ax[j][l],
                            conf_levels=[0.6827, ],#[0.393, ],
                            **contour_kwargs[campaign]
                        )
                    else:
                        plots.normalError(
                            x, y,
                            ax=ax[j][l],
                            conf_level=0.6827,
                            **plot_kwargs
                        )



    ## Add legend in upper-right plot
    ##if j == len(varsx)-1:
    # if j == 0:
    #	for key in _colors:
    #		ax[0].plot([],[],color=_colors[key],label=key)
    #	ax[0].legend(loc='upper left',fontsize=12,numpoints=1,frameon=False)

    if savename != '':
        print "Saving..."
        plt.savefig(savename + ".pdf")
    return ax

    plt.show()


def plotCorrelationsNoFit2(post, varsx, varsy, plot_order=[], x_logify=False, y_logify=False,
                     savename='', usecovariance=False, plotType='normal',
                    label_agn=False, figwidth=7.,
                     pad_inches = [], transpose=False, _colors={}, contour_kwargs={}):
    """

    :param post: Dictionary of Posterior objects
    :type post: dict

    :param varsx: List of keys for x-axis variables
    :type varsx: list

    :param varsy: List of dict keys for x-axis variables
    :type varsy: list

    :param x_logify: List of booleans saying if the log of the x-axis variable should be plotted
    :type varsy: list

    :param y_logify: List of booleans saying if the log of the y-axis variable should be plotted
    :type varsy: list

    :param _savename: Filepath to save the figure
    :type varsy: str

    :param diagonalerrors: Indicate if should plot diagonal error bars, along axis of covariance
    :type varsy: bool

    :param fp_linmix: Filepath to where previous correlation calculations are located
    :type varsy: str

    :param savenew: Indicates if should save newly calculated LinMix correlation values
    :type varsy: bool

    :param lmsuffix:
    :type varsy: str

    :return:
    """
    if type(x_logify) != list:
        x_logify = [x_logify] * len(varsx)
    if type(y_logify) != list:
        y_logify = [y_logify] * len(varsy)

    if len(plot_order) == 0:
        plot_order = post.keys()
    for p in post.keys():
        if post[p].campaign not in _colors.keys():
            _colors[post[p].campaign] = 'k'
    # Plotting values
    nrows = len(varsy)
    ncols = len(varsx)
    if pad_inches == []:
        pad_inches = [0.6, 0.1, 0.1, 0.4, 0.0, 0.0]

    mpl.rcParams['lines.linewidth'] = 0.5
    mpl.rcParams['lines.markersize'] = 2
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9
    mpl.rcParams['axes.labelsize'] = 9

    Figure = bf.CustomFigure()
    if transpose:
        Figure.fixedWidth(figwidth,pad_inches,(ncols,nrows))
    else:
        Figure.fixedWidth(figwidth,pad_inches,(nrows,ncols))

    ax = np.zeros((len(varsx),len(varsy))).tolist()

    for j in range(len(varsx)):
        varx = varsx[j]
        for l in range(len(varsy)):
            vary = varsy[l]
            print "\nWorking on %s vs. %s" % (vary, varx)

            x,y,cov,campaign,objects = posterior.extractConfIntsAndCov(
                post, varx, vary,
                x_logify=x_logify[j], y_logify=y_logify[l],
                agn_order=plot_order
            )
            colors = list()
            for i in range(len(campaign)):
                colors.append(_colors[campaign[i]])

            # Creat the subplot panel
            if transpose:
                ax[j][l] = Figure.newSubplot(position = (l, j))
            else:
                ax[j][l] = Figure.newSubplot(position = (j, l))
            xlim, ylim = parInfo.paramLims()[varx], parInfo.paramLims()[vary]
            ax[j][l].set_xlim(xlim), ax[j][l].set_ylim(ylim)

            xtexname, ytexname = parInfo.paramTexNames()[varx], parInfo.paramTexNames()[vary]
            xticks, yticks = parInfo.paramTicksCorrelation()[varx], parInfo.paramTicksCorrelation()[vary]

            # Setting the ticks and labels
            if x_logify[j]:
                xtexname = "$\\log_{10}(%s)$" % xtexname[1:-1]
            if y_logify[l]:
                ytexname = "$\\log_{10}(%s)$" % ytexname[1:-1]

            ax[j][l].set_xticks(xticks)
            ax[j][l].set_yticks(yticks)

            if l == len(varsy) - 1:
                ax[j][l].set_xlabel(xtexname)
            else:
                if transpose:
                    ax[j][l].set_xlabel(xtexname)
                else:
                    ax[j][l].set_xticklabels([])
            if j == 0:
                ax[j][l].set_ylabel(ytexname)
            else:
                ax[j][l].set_yticklabels([])

            # Plot the linear regression results from LinMix
            if usecovariance:
                _savenamecov = 'wcov'
            else:
                _savenamecov = 'nocov'

            # Plot the data points
            for agn in plot_order:
                try:
                    campaign = post[agn].campaign
                    x = post[agn].posteriors[varx]
                    y = post[agn].posteriors[vary]
                    if x_logify[j]:
                        x = np.log10(x)
                    if y_logify[l]:
                        y = np.log10(y)
                    #contour_kwargs = {'colors': _colors[campaign], 'linewidths':3}
                    plot_kwargs = {'color': _colors[campaign]}
                    if np.shape(x) != () and np.shape(y) != () and \
                            1 not in [np.isnan(n) for n in y] and \
                            1 not in [np.isnan(n) for n in x]:
                        if plotType.lower() == 'angled':
                            plots.angledError(
                                x, y,
                                ax = ax[j][l],
                                **plot_kwargs
                            )
                        elif plotType.lower() == 'potato':
                            plots.potatoPlot(
                                x, y,
                                ax=ax[j][l],
                                conf_levels=[0.6827, ],
                                **contour_kwargs[campaign]
                            )
                        elif plotType.lower() == 'contour':
                            plots.densityContour(
                                x, y,
                                12, 12,
                                ax=ax[j][l],
                                conf_levels=[0.6827, ],#[0.393, ],
                                **contour_kwargs[campaign]
                            )
                        else:
                            plots.normalError(
                                x, y,
                                ax=ax[j][l],
                                conf_level=0.6827,
                                **plot_kwargs
                            )
                except:
                    pass



    ## Add legend in upper-right plot
    ##if j == len(varsx)-1:
    # if j == 0:
    #   for key in _colors:
    #       ax[0].plot([],[],color=_colors[key],label=key)
    #   ax[0].legend(loc='upper left',fontsize=12,numpoints=1,frameon=False)

    if savename != '':
        print "Saving..."
        plt.savefig(savename + ".pdf")
    return ax

    plt.show()


def plotCorrelationsCorrHists(post, varsx, varsy, x_logify=False, y_logify=False,
                     savename='', usecovariance=False,
                     fp_linmix='', lmsuffix=''):
    """

    :param post: Dictionary of Posterior objects
    :type post: dict

    :param varsx: List of keys for x-axis variables
    :type varsx: list

    :param varsy: List of dict keys for x-axis variables
    :type varsy: list

    :param x_logify: List of booleans saying if the log of the x-axis variable should be plotted
    :type varsy: list

    :param y_logify: List of booleans saying if the log of the y-axis variable should be plotted
    :type varsy: list

    :param _savename: Filepath to save the figure
    :type varsy: str

    :param diagonalerrors: Indicate if should plot diagonal error bars, along axis of covariance
    :type varsy: bool

    :param fp_linmix: Filepath to where previous correlation calculations are located
    :type varsy: str

    :param savenew: Indicates if should save newly calculated LinMix correlation values
    :type varsy: bool

    :param lmsuffix:
    :type varsy: str

    :return:
    """
    if type(x_logify) != list:
        x_logify = [x_logify] * len(varsx)
    if type(y_logify) != list:
        y_logify = [y_logify] * len(varsy)

    # Plotting values
    figheight = 2.0
    nrows = len(varsy)
    ncols = len(varsx)
    pad_inches = [0.47, 0.1, 0.1, 0.32, 0.0, 0.0]

    mpl.rcParams['lines.linewidth'] = 0.5
    mpl.rcParams['lines.markersize'] = 2
    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6
    mpl.rcParams['axes.labelsize'] = 8

    _colors = {"LAMP2011": "red", "LAMP2008": "green", "AGN10": "blue"}

    Figure = bf.CustomFigure()
    Figure.fixedHeight(figheight, pad_inches, (nrows, ncols), spaxisratio=0.5)

    ax = np.zeros((len(varsx),len(varsy))).tolist()

    for j in range(len(varsx)):
        varx = varsx[j]
        for l in range(len(varsy)):
            vary = varsy[l]
            #print "\nWorking on %s vs. %s" % (vary, varx)
            x,y,cov,campaign,objects = posteriors.extractConfIntsAndCov(
                post, varx, vary,
                x_logify=x_logify[j], y_logify=y_logify[l]
            )
            colors = list()
            for i in range(len(campaign)):
                colors.append(_colors[campaign[i]])

            # Creat the subplot panel
            ax[j][l] = Figure.newSubplot(position = (j, l))

            xtexname, ytexname = parInfo.paramTexNames()[varx], parInfo.paramTexNames()[vary]
            xticks, yticks = parInfo.paramTicksCorrelation()[varx], parInfo.paramTicksCorrelation()[vary]

            # Setting the ticks and labels
            if x_logify[j]:
                xtexname = "$\\log_{10}(%s)$" % xtexname[1:-1]
            if y_logify[l]:
                ytexname = "$\\log_{10}(%s)$" % ytexname[1:-1]


            if l == len(varsy) - 1:
                ax[j][l].set_xlabel(xtexname)
            else:
                pass
            if j == 0:
                ax[j][l].set_ylabel(ytexname)
            else:
                pass

            # Plot the linear regression results from LinMix
            if usecovariance:
                _savenamecov = 'wcov'
            else:
                _savenamecov = 'nocov'
            lm_filename = "%s%s_vs_%s_%s_%s.npy" % (fp_linmix, vary, varx, _savenamecov, lmsuffix)
            lmchain = np.load(lm_filename)
            chain = {
                'alpha': lmchain[:, 0],
                'beta': lmchain[:, 1],
                'sigsqr': lmchain[:, 2],
                'corr': lmchain[:, 3]
            }


            # TODO:
            # Experiment with calculating p-values for the correlations

            from scipy.stats import t
            Nsamp = len(cov)
            tstat = chain['corr'] * np.sqrt((Nsamp - 2.)/(1. - chain['corr']**2))
            #tstat = np.abs(tstat)
            #tstat = np.array([tt for tt in tstat if tt < 10])
            pval = t.sf(tstat,Nsamp)
            #ax[j][l].hist(tstat, bins=50)
            #ax[j][l].axvline(np.median(tstat), color='r', lw=1)
            ax[j][l].hist(pval, bins=50)
            ax[j][l].axvline(np.median(pval), color='r', lw=1)
            realcorr05 = np.where(pval < 0.05, 1, 0)
            realcorr01 = np.where(pval < 0.01, 1, 0)
            print "p < 0.05: %.4f" % float(np.sum(realcorr05)) / float(len(realcorr05))
            print "p < 0.01: %.4f" % float(np.sum(realcorr01)) / float(len(realcorr01))
            #print "%s vs. %s: %.4f" % (vary,varx,np.median(pval))

            #Nsamp = len(chain['corr'])
            #ltzero = np.where(chain['corr'] < 0, 1, 0)
            #gtzero = np.where(chain['corr'] > 0, 1, 0)
            #print "%s vs. %s negative: %.4f" % (vary, varx, float(np.sum(ltzero))/float(Nsamp))
            #print "%s vs. %s positive: %.4f" % (vary, varx, float(np.sum(gtzero))/float(Nsamp))
            #ax[j][l].hist(chain['corr'],bins=50)
            #c50,c84,c16 = np.percentile(chain['corr'],[50,84,16])
            #ax[j][l].axvline(x = c50, color='r', lw=1)
            #ax[j][l].axvline(x=c84, color='r', lw=1, ls='dashed')
            #ax[j][l].axvline(x=c16, color='r', lw=1, ls='dashed')
            #ax[j][l].axvline(x=0,color='k',ls='dotted',lw=1)


    if savename != '':
        print "Saving..."
        plt.savefig(savename + "_corr.pdf")

    plt.show()


def plotCorrelationsNoFit(post, varsx, varsy, x_logify=False, y_logify=False, _savename=True, diagonalerrors=False,plotmask=''):
    """

    :param post: Dictionary of Posterior objects
    :type post: dict

    :param varsx: List of keys for x-axis variables
    :type varsx: list

    :param varsy: List of dict keys for x-axis variables
    :type varsy: list

    :param x_logify: List of booleans saying if the log of the x-axis variable should be plotted
    :type varsy: list

    :param y_logify: List of booleans saying if the log of the y-axis variable should be plotted
    :type varsy: list

    :param _savename: Filepath to save the figure
    :type varsy: str

    :param diagonalerrors: Indicate if should plot diagonal error bars, along axis of covariance
    :type varsy: bool

    :param fp_linmix: Filepath to where previous correlation calculations are located
    :type varsy: str

    :param savenew: Indicates if should save newly calculated LinMix correlation values
    :type varsy: bool

    :param lmsuffix:
    :type varsy: str

    :return:
    """
    if type(x_logify) != list:
        x_logify = [x_logify] * len(varsx)
    if type(y_logify) != list:
        y_logify = [y_logify] * len(varsy)

    if np.shape(plotmask) == ():
        plotmask = np.array([[plotmask]] * len(varsx) * len(varsy))
        plotmask = plotmask.reshape((len(varsx), len(varsy)))

    # Plotting values
    figheight = 6.

    _axis_fontparams = {'fontsize': 14}
    _tick_fontparams = {'fontsize': 10}
    _colors = {"LAMP2011": "red", "LAMP2008": "green", "AGN10": "blue", "SDSSJ2222": "orange"}

    nrows = len(varsy)
    ncols = len(varsx)

    _lpadin = 0.8
    _rpadin = 0.1
    _bpadin = 0.55
    _tpadin = 0.1
    _wpadin = 0.0
    _hpadin = 0.0

    _spheightin = (figheight - _tpadin - _bpadin - _hpadin * (nrows - 1.)) / nrows
    _spwidthin = _spheightin
    figwidth = ncols * _spwidthin + (ncols - 1.) * _wpadin + _rpadin + _lpadin

    _lpad = _lpadin / figwidth
    _rpad = _rpadin / figwidth
    _tpad = _tpadin / figheight
    _wpad = _wpadin / figwidth

    _spwidth = _spwidthin / figwidth
    _spheight = _spheightin / figheight

    _panelwidth = _spwidth
    _panelheight = _spheight

    plt.figure(1, figsize=(figwidth, figheight))
    ax = np.zeros((len(varsx),len(varsy))).tolist()

    for j in range(len(varsx)):
        varx = varsx[j]
        for l in range(len(varsy)):
            vary = varsy[l]
            print "\nWorking on %s vs. %s" % (vary, varx)
            x,y,cov,campaign,objects = posterior.extractConfIntsAndCov(
                post, varx, vary,
                x_logify=x_logify[j], y_logify=y_logify[l]
            )
            colors = list()
            for i in range(len(campaign)):
                colors.append(_colors[campaign[i]])

            # Creat the subplot panel
            ax[j][l] = plt.axes((
                _lpad + j * (_spwidth + _wpad),
                1. - _tpad - _spheight * l - 1. * _panelheight,
                _panelwidth,
                _panelheight
            ))
            try:
                xlim, ylim = parInfo.paramLims()[varx], parInfo.paramLims()[vary]
                ax[j][l].set_xlim(xlim), ax[j][l].set_ylim(ylim)
            except:
                pass

            xtexname, ytexname = parInfo.paramTexNames()[varx], parInfo.paramTexNames()[vary]

            # Setting the ticks and labels
            if x_logify[j]:
                xtexname = "$\\log_{10}(%s)$" % xtexname[1:-1]
            if y_logify[l]:
                ytexname = "$\\log_{10}(%s)$" % ytexname[1:-1]

            if l == len(varsy) - 1:
                ax[j][l].set_xlabel(xtexname, **_axis_fontparams)
            else:
                ax[j][l].set_xticklabels([])
            if j == 0:
                ax[j][l].set_ylabel(ytexname, **_axis_fontparams)
            else:
                ax[j][l].set_yticklabels([])

            # Plot the points and the error bars
            if diagonalerrors:
                for i in range(len(x)):
                    if objects[i] not in plotmask[j][l]:
                        plt.plot(
                            x[i,0], y[i,0],
                            marker='o',
                            color=colors[i]
                        )
                        eigval, eigvec = np.linalg.eig(cov[i])
                        err = np.array([eigvec[:, 0] * eigval[0] ** 0.5, eigvec[:, 1] * eigval[1] ** 0.5])
                        plt.plot(
                            [x[i,0] - err[0, 0], x[i,0] + err[0, 0]],
                            [y[i,0] - err[0, 1], y[i,0] + err[0, 1]],
                            color=colors[i]
                        )
                        plt.plot(
                            [x[i,0] - err[1, 0], x[i,0] + err[1, 0]],
                            [y[i,0] - err[1, 1], y[i,0] + err[1, 1]],
                            color=colors[i]
                        )
            else:
                for i in range(len(x)):
                    if objects[i] not in plotmask[j][l]:
                        plt.errorbar(x[i,0], y[i,0],
                            xerr=[[x[i, 1]], [x[i, 2]]],
                            yerr=[[y[i, 1]], [y[i, 2]]],
                            fmt='o',
                            color=colors[i]
                        )

    if _savename != False:
        print "Saving..."
        plt.savefig(_savename + ".pdf")

    plt.show()


def loadLinMix():
    pass