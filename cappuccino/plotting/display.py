import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import buildFigures as bf
from cappuccino.utils.util import errorsum

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True

def plotData(Model, ax=0, no_xlabel=False, cmap='coolwarm', colorbar=True, plotnepochs=-1,
    imshowlims=[-9e9,9e9], epoch_labels=[], figsize=(6,4), **kwargs):
    """
    2d plot of the emission line light curve data

    :param Model: Model object for the CARAMEL run
    :type Model: cappuccino.model.Model object

    :param ax: Axes object to plot on. Creates new fig if none provided
    :type ax: matplotlib.axes object

    :param no_xlabel: Hides the x_label, defaults to False
    :type no_xlabel: bool

    :param cmap: Matplotlib colormap
    :type cmap: str

    :param colorbar: Whether to show a colorbar, defaults to True
    :type colorbar: bool

    :param plotnepochs: Number of epochs to plot. Plots all by default
    :type plotnepochs: int

    :param imshowlims: z-axis limits of imshow, defaults to full limits spanned by data
    :type imshowlims: list, length 2

    :param epoch_labels: y-axis labels of epochs, defaults to auto.
    :type epoch_labels: list

    :param figsize: Size of the figure (if ax not provided)
    :type figsize: tuple, length 2

    :return: ax
    """

    if ax == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if plotnepochs == -1:
        plotnepochs = np.shape(Model.data)[0]
    ylim = [0,plotnepochs]

    if imshowlims[0] != -9e9:
        vmin,vmax = imshowlims[0], imshowlims[1]
    else:
        vmin,vmax = np.min(Model.data[:plotnepochs,:]), np.max(Model.data[:plotnepochs,:])
    ax.imshow(
        Model.data[:plotnepochs,:],
        interpolation='nearest',
        aspect='auto',
        origin='upper',
        cmap=cmap,
        extent=(Model.llim[0], Model.llim[1], ylim[1], ylim[0]),
        vmin=vmin,vmax=vmax
    )
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=np.min(Model.data[:plotnepochs,:]), vmax=np.max(Model.data[:plotnepochs,:]))
        )
        sm._A = []
        plt.colorbar(sm, ax=ax)

    ax.set_xlim(Model.llim)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.text(
        0.03,
        0.94,
        "$\\rm Data$",
        transform=ax.transAxes,
        ha='left',va='top',
        color='white',
    )
    if no_xlabel:
        ax.tick_params(labelbottom='off')
    else:
        ax.set_xlabel(r"$\rm Observed~Wavelength~(\AA)$")
    if epoch_labels != []:
        ax.set_yticks(epoch_labels)
    ax.set_ylabel(r"$\rm Epoch$")
    ax.locator_params('y', nbins=10)

def plotModel(Model, ax=0, index=0, no_xlabel=False, cmap='coolwarm', colorbar=True, plotnepochs=-1,
    imshowlims=[-9e9,9e9], epoch_labels=[], figsize=(6,4),  **kwargs):
    """
    2d plot of the model fit to the emission line light curve data

    :param Model: Model object for the CARAMEL run
    :type Model: cappuccino.model.Model object

    :param ax: Axes object to plot on. Creates new fig if none provided
    :type ax: matplotlib.axes object

    :param index: Index to draw from posterior sample, defaults to 0
    :type index: int

    :param no_xlabel: Hides the x_label, defaults to False
    :type no_xlabel: bool

    :param cmap: Matplotlib colormap
    :type cmap: str

    :param colorbar: Whether to show a colorbar, defaults to True
    :type colorbar: bool

    :param plotnepochs: Number of epochs to plot. Plots all by default
    :type plotnepochs: int

    :param imshowlims: z-axis limits of imshow, defaults to full limits spanned by data
    :type imshowlims: list, length 2

    :param epoch_labels: y-axis labels of epochs, defaults to auto.
    :type epoch_labels: list

    :param figsize: Size of the figure (if ax not provided)
    :type figsize: tuple, length 2

    :return: ax
    """
    if ax == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    rainbow = Model.posterior_sample[index,
              Model.spec_ind[0]:Model.spec_ind[1]]

    if plotnepochs == -1:
        plotnepochs = np.shape(Model.data)[0]
    ylim = [0,plotnepochs]

    # plot the model
    if imshowlims[0] != -9e9:
        vmin,vmax = imshowlims[0], imshowlims[1]
    else:
        vmin,vmax = np.min(rainbow), np.max(rainbow)
    ax.imshow(
        rainbow.reshape(Model.data.shape),
        interpolation='nearest',
        aspect='auto',
        origin='upper',
        cmap=cmap,
        extent=(Model.llim[0], Model.llim[1], ylim[1], ylim[0]),
        vmin=vmin,vmax=vmax
    )
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=min(rainbow), vmax=max(rainbow))
        )
        sm._A = []
        plt.colorbar(sm, ax=ax)

    ax.set_xlim(Model.llim)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.text(
        0.03,
        0.94,
        "$\\rm Model$",
        transform=ax.transAxes,
        ha='left',va='top',
        color='white'
    )
    if no_xlabel:
        ax.tick_params(labelbottom='off')
    else:
        ax.set_xlabel(r"$\rm Observed~Wavelength~(\AA)$")
    if epoch_labels != []:
        ax.set_yticks(epoch_labels)
    ax.set_ylabel(r"$\rm Epoch$")
    ax.locator_params('y', nbins=10)

def plotResidual(Model, ax=0, index=0, normalized=True, scale_by_temp=False, no_xlabel=False,
    cmap='coolwarm', colorbar=True, plotnepochs=-1, imshowlims=[-9e9,9e9], epoch_labels=[],
    figsize=(6,4),  **kwargs):
    """
    2d plot of the residual for the model fit to the emission line light curve data

    :param Model: Model object for the CARAMEL run
    :type Model: cappuccino.model.Model object

    :param ax: Axes object to plot on. Creates new fig if none provided
    :type ax: matplotlib.axes object

    :param index: Index to draw from posterior sample, defaults to 0
    :type index: int
    
    :param normalized: Whether to show normalized or regular residual, defaults to True
    :type normalized: bool
    
    :param scale_by_temp: Whether to scale emission line uncertainties by the temperature, defaults to False
    :type scale_by_temp: bool

    :param no_xlabel: Hides the x_label, defaults to False
    :type no_xlabel: bool

    :param cmap: Matplotlib colormap
    :type cmap: str

    :param colorbar: Whether to show a colorbar, defaults to True
    :type colorbar: bool

    :param plotnepochs: Number of epochs to plot. Plots all by default
    :type plotnepochs: int

    :param imshowlims: z-axis limits of imshow, defaults to full limits spanned by data
    :type imshowlims: list, length 2

    :param epoch_labels: y-axis labels of epochs, defaults to auto.
    :type epoch_labels: list

    :param figsize: Size of the figure (if ax not provided)
    :type figsize: tuple, length 2

    :return: ax
    """
    if ax == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    rainbow = Model.posterior_sample[index,
              Model.spec_ind[0]:Model.spec_ind[1]]
        
    if plotnepochs == -1:
        plotnepochs = np.shape(Model.data)[0]
    ylim = [0,plotnepochs]

    to_plot = Model.data[:plotnepochs,:] - rainbow.reshape(Model.data.shape)[:plotnepochs,:]
    if normalized:
        to_plot /= Model.err[:plotnepochs,:]
    if scale_by_temp:
        to_plot /= np.sqrt(Model.temp)
    
    if imshowlims[0] != -9e9:
        vmin,vmax = imshowlims[0], imshowlims[1]
    else:
        vmin,vmax = np.min(to_plot), np.max(to_plot)
    ax.imshow(
        to_plot,
        interpolation='nearest',
        aspect='auto',
        origin='upper',
        cmap=cmap,
        extent=(Model.llim[0], Model.llim[1], ylim[1], ylim[0]),
        vmin=vmin,vmax=vmax
    )
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=np.min(to_plot), vmax=np.max(to_plot))
        )
        sm._A = []
        plt.colorbar(sm, ax=ax)

    ax.set_xlim(Model.llim)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if normalized: 
        axtext = "${\\rm Normalized~residual}$"
    else:
        axtext = "${\\rm Residual}$"
    ax.text(
        0.03,
        0.94,
        axtext,
        transform=ax.transAxes,
        ha='left',va='top',
        color='k'
    )

    if no_xlabel:
        ax.tick_params(labelbottom='off')
    else:
        ax.set_xlabel(r"$\rm Observed~Wavelength~(\AA)$")
    ax.set_ylabel(r"$\rm Epoch$")
    if epoch_labels != []:
        ax.set_yticks(epoch_labels)

def plotLineProfile(Model, ax=0, index=0, epoch=10, lc_style='samples', scale_by_temp=False,
    linecenter=0, no_xlabel=False, figsize=(6,4),  **kwargs):
    """
    Emission line profile data + model fits.

    :param Model: Model object for the CARAMEL run
    :type Model: cappuccino.model.Model object

    :param ax: Axes object to plot on. Creates new fig if none provided
    :type ax: matplotlib.axes object

    :param index: Index to draw from posterior sample, defaults to 0
    :type index: int

    :param epoch: Epoch to show for the emission line profile, defaults to 10
    :type epoch: int

    :param lc_style: Plot individual samples ('samples') or confidence intervals ('confint'), defaults to 'samples'
    :type lc_style: str

    :param scale_by_temp: Whether to scale emission line uncertainties by the temperature, defaults to False
    :type scale_by_temp: bool

    :param linecenter: If provided, will draw a vertical dashed line at the emission line center
    :type linecenter: float

    :param no_xlabel: Hides the x_label, defaults to False
    :type no_xlabel: bool

    :param colorbar: Whether to show a colorbar, defaults to True
    :type colorbar: bool

    :param plotnepochs: Number of epochs to plot. Plots all by default
    :type plotnepochs: int

    :param imshowlims: z-axis limits of imshow, defaults to full limits spanned by data
    :type imshowlims: list, length 2

    :param epoch_labels: y-axis labels of epochs, defaults to auto.
    :type epoch_labels: list

    :param figsize: Size of the figure (if ax not provided)
    :type figsize: tuple, length 2

    :return: ax
    """
    if ax == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Keep track of the minimum and maximum values of the data
    ymax, ymin = -1e9, 1e9

    # Plot either a range of models or confidence intervals
    if Model.num_samples > 1:
        if lc_style == 'samples':
            # Plot 8 additional samples
            for i in np.arange(3, 25, 3):
                rainbow = Model.posterior_sample[index + i,
                            Model.spec_ind[0]:Model.spec_ind[1]]
                ax.plot(Model.wavelengths,
                        rainbow.reshape(Model.data.shape)[epoch, :], '-c',
                        lw=0.3)
                ymax = max(ymax, max(rainbow.reshape(Model.data.shape)[epoch, :]))
                ymin = min(ymin, min(rainbow.reshape(Model.data.shape)[epoch, :]))
            rainbow = Model.posterior_sample[index,
                            Model.spec_ind[0]:Model.spec_ind[1]]
            # Plot the selected model
            ax.plot(Model.wavelengths,
                rainbow.reshape(Model.data.shape)[epoch, :],
                '-r',
                lw=0.8
            )
        elif lc_style == 'confint':
            # Compute the confidence intervals at each wavelength bin
            tmp = []
            for i in range(0,Model.num_samples,1):
                rainbow = Model.posterior_sample[i,
                            Model.spec_ind[0]:Model.spec_ind[1]]
                tmp.append(rainbow.reshape(Model.data.shape)[epoch, :])
            tmp = np.array(tmp)
            med,high,low = np.percentile(tmp,(50,50+34.135,50-34.135),axis=0)
            ax.fill_between(Model.wavelengths, low, high, color='r', alpha=0.2, lw=0)
            ax.plot(Model.wavelengths, med, 'r', lw=0.6)
            ymax = max(ymax, max(high))
            ymin = min(ymin, min(low))


    # Plot the data
    if scale_by_temp:
        yerr = Model.err[epoch, :] * np.sqrt(Model.temp)
    else:
        yerr = Model.err[epoch, :] * 1.
    ax.errorbar(
        Model.wavelengths,
        Model.data[epoch, :],
        yerr=yerr,
        color='k',
        linewidth=0,
        elinewidth=0.5
    )
    ymax = max(ymax, max(Model.data[epoch, :] + yerr))
    ymin = min(ymin, min(Model.data[epoch, :] - yerr))

    if type(linecenter) != list:
        linecenter = [linecenter]
    if linecenter[0] != 0:
        for l in linecenter:
            ax.axvline(l, color='k',ls='dashed', lw=0.5)

    ax.set_xlim(Model.llim)
        
    if no_xlabel:
        ax.tick_params(labelbottom='off')
    else:
        ax.set_xlabel(r"$\rm Observed~Wavelength~(\AA)$")
    ax.set_ylabel('$\\rm Flux$\\\\$\\rm (arbitrary)$')

    #ax.set_ylim(ymin - 0.05 * (ymax-ymin),ymax + 0.05 * (ymax-ymin))
    #ax.set_yticks(np.linspace(ymin, 0.9 * ymax, 6))
    #ax.set_yticklabels((r'$0$', r'$2$', r'$4$', r'$6$', r'$8$', r'$10$'))

    ax.minorticks_off()

def plotLineLightCurve(Model, ax=0, index=0, times_correct=0, lc_style='samples',
    scale_by_temp=False, plotnepochs=-1, plotstartdate=0, plotenddate=0, no_xlabel=False,
    emission_line_texname='', xlabel='', figsize=(6,4), **kwargs):
    """
    Integrated emission line light curve

    :param Model: Model object for the CARAMEL run
    :type Model: cappuccino.model.Model object

    :param ax: Axes object to plot on. Creates new fig if none provided
    :type ax: matplotlib.axes object

    :param index: Index to draw from posterior sample, defaults to 0
    :type index: int

    :param times_correct: Time correction to be added to dates, e.g., -50000 for HJD - 50,000. Defaults to 0
    :type times_correct: float

    :param lc_style: Plot individual samples ('samples') or confidence intervals ('confint'), defaults to 'samples'
    :type lc_style: str

    :param scale_by_temp: Whether to scale emission line uncertainties by the temperature, defaults to False
    :type scale_by_temp: bool
    
    :param plotstartdate: Start date (xlim[0]) of date axis, in data units (pre-times_correct)
    :type plotstartdate: int

    :param plotenddate: End date (xlim[0]) of date axis, in data units (pre-times_correct)
    :type plotenddate: int

    :param plotnepochs: Number of epochs to plot. Plots all by default
    :type plotnepochs: int

    :param xlabel: x-axis label. Important if data units are not HJD.
    :type xlabel: str

    :param emission_line_texname: Name of the emission line. Default no name
    :type emission_line_texname: str

    :param no_xlabel: Hides the x_label, defaults to False
    :type no_xlabel: bool

    :param epoch_labels: y-axis labels of epochs, defaults to auto.
    :type epoch_labels: list

    :param figsize: Size of the figure (if ax not provided)
    :type figsize: tuple, length 2

    :return: ax
    """
    if ax == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if plotnepochs == -1:
        plotnepochs = np.shape(Model.data)[0]
    
    # Keep track of minimum and maximum values
    ymax, ymin = -1e9, 1e9
    
    # Plot either a range of models or confidence intervals
    if Model.num_samples > 1:
        if lc_style == 'samples':
            # Plot 8 additional samples
            for i in np.arange(3, 25, 3):
                rainbow2 = Model.posterior_sample[index + i,
                           Model.spec_ind[0]:Model.spec_ind[1]]
                lc = np.sum(rainbow2.reshape(Model.data.shape), 1)
                ax.plot(Model.times[:plotnepochs] + times_correct, lc[:plotnepochs], 'c', lw=0.3)
                ymax = max(ymax, max(lc[:plotnepochs]))
                ymin = min(ymin, min(lc[:plotnepochs]))
            # Plot the selected model
            rainbow = Model.posterior_sample[index,
                            Model.spec_ind[0]:Model.spec_ind[1]]
            lc = np.sum(rainbow.reshape(Model.data.shape), 1)
            ax.plot(
                Model.times[:plotnepochs] + times_correct,
                lc[:plotnepochs],
                'r',
                lw=0.8
            )
        elif lc_style == 'confint':
            # Compute the confidence intervals
            tmp = []
            for i in range(0,Model.num_samples,int(Model.num_samples/Model.num_samples)):
                rainbow2 = Model.posterior_sample[i,
                           Model.spec_ind[0]:Model.spec_ind[1]]
                lc = np.sum(rainbow2.reshape(Model.data.shape), 1)
                tmp.append(lc[:plotnepochs])
            tmp = np.array(tmp)
            med,high,low = np.percentile(tmp,(50,50+34.135,50-34.135),axis=0)
            ax.fill_between(Model.times[:plotnepochs] + times_correct, low, high, color='red', alpha=0.2, lw=0)
            ax.plot(Model.times[:plotnepochs] + times_correct, med, 'r', lw=0.6)
            ymax = max(ymax, max(high))
            ymin = min(ymin, min(low))

    # Plot the data
    lc = np.sum(Model.data, 1)
    lc_error = []
    for i in range(len(Model.err)):
        if scale_by_temp:
            lc_error.append(errorsum(Model.err[i] * np.sqrt(Model.temp)))
        else:
            lc_error.append(errorsum(Model.err[i]))
    ax.errorbar(
        Model.times[:plotnepochs] + times_correct,
        lc[:plotnepochs],
        yerr=lc_error[:plotnepochs],
        color='k',
        linewidth=0,
        elinewidth=0.5
    )

    if no_xlabel:
        ax.tick_params(labelbottom='off')
    else:
        if xlabel == '':
            print "Assuming dates are in HJD. If not, please set 'xlabel' parameter."
            if times_correct != 0:
                ax.set_xlabel(r"$\rm HJD %i$" % times_correct)
            else:
                ax.set_xlabel(r"$\rm HJD$")
        else:
            ax.set_xlabel(xlabel)
    ax.set_ylabel('$\\rm Flux$\\\\$\\rm (arbitrary)$')

    if plotstartdate != 0:
        ax.set_xlim(plotstartdate + times_correct, ax.get_xlim()[1])
    if plotenddate != 0:
        ax.set_xlim(ax.get_xlim()[0], plotenddate + times_correct)
    
    xlim,ylim = ax.get_xlim(), ax.get_ylim()
    ax.text(
            0.03,
            0.94,
            emission_line_texname,
            transform=ax.transAxes,
            va='top', ha='left',
            multialignment='left',
            color='black'
        )

def plotContinuumLightCurve(Model, ax=0, index=0, times_correct=0, lc_style='samples',
    plotstartdate=0, plotenddate=0, xlabel='', continuum_texname="${\\rm Continuum}$",
    no_xlabel=False, figsize=(6,4), **kwargs):
    """
    Continuum light curve

    :param Model: Model object for the CARAMEL run
    :type Model: cappuccino.model.Model object

    :param ax: Axes object to plot on. Creates new fig if none provided
    :type ax: matplotlib.axes object

    :param index: Index to draw from posterior sample, defaults to 0
    :type index: int

    :param times_correct: Time correction to be added to dates, e.g., -50000 for HJD - 50,000. Defaults to 0
    :type times_correct: float

    :param lc_style: Plot individual samples ('samples') or confidence intervals ('confint'), defaults to 'samples'
    :type lc_style: str
    
    :param plotstartdate: Start date (xlim[0]) of date axis, in data units (pre-times_correct)
    :type plotstartdate: int

    :param plotenddate: End date (xlim[0]) of date axis, in data units (pre-times_correct)
    :type plotenddate: int

    :param xlabel: x-axis label. Important if data units are not HJD.
    :type xlabel: str

    :param continuum_texname: Name of the emission line. Default to "Continuum"
    :type continuum_texname: str

    :param no_xlabel: Hides the x_label, defaults to False
    :type no_xlabel: bool

    :param epoch_labels: y-axis labels of epochs, defaults to auto.
    :type epoch_labels: list

    :param figsize: Size of the figure (if ax not provided)
    :type figsize: tuple, length 2

    :return: ax
    """
    if ax == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    t1 = Model.cont[:, 0].min()
    t2 = Model.cont[:, 0].max()
    t = np.linspace(t1 - Model.backward_extrap * (t2 - t1),
                    t2 + Model.forward_extrap * (t2 - t1), 1000)

    if plotstartdate != 0:
        startindex = np.argmin(abs(t - plotstartdate))
    else:
        startindex = 0
    if plotenddate != 0:
        endindex = np.argmin(abs(t - plotenddate))
    else:
        endindex = len(t)
    
    ymax, ymin = -1e9, 1e9
    if Model.num_samples > 1:
        if lc_style=='samples':
            for i in np.arange(3, 25, 3):
                y = Model.posterior_sample[index + i, Model.cont_ind:]
                ax.plot(t[startindex:endindex] + times_correct, y[startindex:endindex], 'c', lw=0.3)
                ymax = max(ymax, max(y[startindex:endindex]))
                ymin = min(ymin, min(y[startindex:endindex]))
        
            y = Model.posterior_sample[index, Model.cont_ind:]
            ax.plot(t[startindex:endindex] + times_correct, y[startindex:endindex], 'r', lw=0.8)
        elif lc_style=='confint':
            tmp = []
            for i in range(0,Model.num_samples,int(Model.num_samples/Model.num_samples)):
                y = Model.posterior_sample[i, Model.cont_ind:]
                tmp.append(y)
            tmp = np.array(tmp)
            med,high,low = np.percentile(tmp,(50,50+34.135,50-34.135),axis=0)
            ax.fill_between(t[startindex:endindex] + times_correct, low[startindex:endindex], high[startindex:endindex], color='r', alpha=0.2, lw=0)
            ax.plot(t[startindex:endindex] + times_correct, med[startindex:endindex], 'r', lw=0.6)
            ymax = max(ymax, max(high[startindex:endindex]))
            ymin = min(ymin, min(low[startindex:endindex]))
    
    ax.errorbar(
        Model.cont[:, 0] + times_correct,
        Model.cont[:, 1],
        yerr=Model.cont[:, 2],
        color='k',
        linewidth=0,
        fmt='.',
        markersize=0,
        elinewidth=1.0
    )

    if no_xlabel:
        ax.tick_params(labelbottom='off')
    else:
        if xlabel == '':
            print "Assuming dates are in HJD. If not, please set 'xlabel' parameter."
            if times_correct != 0:
                ax.set_xlabel(r"$\rm HJD %i$" % times_correct)
            else:
                ax.set_xlabel(r"$\rm HJD$")
        else:
            ax.set_xlabel(xlabel)
    ax.set_ylabel('$\\rm Flux$\\\\$\\rm (arbitrary)$')

    if plotstartdate != 0:
        ax.set_xlim(plotstartdate + times_correct, ax.get_xlim()[1])
    if plotenddate != 0:
        ax.set_xlim(ax.get_xlim()[0], plotenddate + times_correct)
    
    xlim,ylim = ax.get_xlim(), ax.get_ylim()
    ax.text(
            0.03,
            0.94,
            continuum_texname,
            transform=ax.transAxes,
            va='top', ha='left',
            multialignment='left',
            color='black'
        )

def plotDisplay(Model, figsize=(6, 8), pad_inches=[0.6, 0.2, 0.1, 0.5, 0.5, 0.4], pad_subplots=0.1,
    savename='', colorbar=False, index=0, epoch=10, times_correct=0, normalized=True,
    scale_by_temp=False, cmap='coolwarm', lc_xlim=[], emission_line_texname="",
    continuum_texname="${\\rm Continuum}$", **kwargs):
    """
    Standard "display plot" used in CARAMEL papers to show the model fits.

    :param Model: Model object for the CARAMEL run
    :type Model: cappuccino.model.Model object

    :param figsize: Dimensions of the figure in inches (width, height)
    :type figsize: tuple, length 2

    :param pad_inches: Padding in inches [left,right,top,bottom,column,row]
    :type pad_inches: tuple or list, length 6

    :param pad_subplots: Padding between line fit and light curve fit panels
    :type pad_subplots: float

    :param savename: Filepath to save figure, defailts to '' (no save)
    :type savename: str

    :param colorbar: Show colorbars for model fits, defaults to False
    :type colorbar: bool

    :param index: Index to draw from posterior sample, defaults to 0
    :type index: int

    :param epoch: Epoch to show for the emission line profile, defaults to 10
    :type epoch: int

    :param times_correct: Time correction to be added to dates, e.g., -50000 for HJD - 50,000. Defaults to 0
    :type times_correct: float

    :param normalized: Whether to show normalized or regular residual, defaults to True
    :type normalized: bool

    :param scale_by_temp: Whether to scale emission line uncertainties by the temperature, defaults to False
    :type scale_by_temp: bool

    :return: ax
    """
    # Set up the empty figure
    ncols, nrows = 1, 1

    Figure = bf.CustomFigure()
    Figure.fixedDimensions(figsize, pad_inches)
    Figure.setRowsCols((nrows, ncols))

    ax = Figure.newSubplot(position=(0, 0), visible=False)
    ax = Figure.makeSubpanels(
        ax,
        dims=(1, 6),
        pad=(0., [0., 0., 0., pad_subplots, 0.])
    )
    ax = ax[0, :]

    if colorbar:
        # Work in progress -- add a colorbar to the display plot
        plotnepochs = np.shape(Model.data)[0]
        data = Model.data[:plotnepochs,:]
        model = Model.posterior_sample[index, Model.spec_ind[0]:Model.spec_ind[1]]
        resid = Model.data[:plotnepochs,:] - model.reshape(Model.data.shape)[:plotnepochs,:]
        if normalized:
            resid /= Model.err[:plotnepochs,:]
        if scale_by_temp:
            resid /= np.sqrt(Model.temp)
        imshow_max_data = max([np.max(abs(data)), np.max(abs(model))])
        imshow_max_resid = np.max(abs(resid))
        imshowlims_data = [0, imshow_max_data]
        imshowlims_resid = [-imshow_max_resid, imshow_max_resid]
    else:
        imshowlims_data = [-9e9,9e9]
        imshowlims_resid = [-9e9,9e9]

    # Create each of the panels individually
    plotData(
        Model,
        ax=ax[0],
        no_xlabel=True,
        cmap=cmap,
        colorbar=False,
        imshowlims=imshowlims_data,
        **kwargs
    )
    plotModel(
        Model,
        ax=ax[1],
        index=index,
        no_xlabel=True,
        cmap=cmap,
        colorbar=False,
        imshowlims=imshowlims_data,
        **kwargs
    )
    plotResidual(
        Model,
        ax=ax[2],
        index=index,
        no_xlabel=True,
        normalized=normalized,
        scale_by_temp=scale_by_temp,
        cmap=cmap,
        colorbar=False,
        imshowlims=imshowlims_resid,
        **kwargs
    )
    plotLineProfile(
        Model,
        ax=ax[3],
        index=index,
        epoch=epoch,
        no_xlabel=False,
        scale_by_temp=scale_by_temp,
        **kwargs
    )
    plotLineLightCurve(
        Model,
        ax=ax[4],
        index=index,
        times_correct=times_correct,
        no_xlabel=True,
        scale_by_temp=scale_by_temp,
        emission_line_texname=emission_line_texname,
        **kwargs
    )
    plotContinuumLightCurve(
        Model,
        ax=ax[5],
        index=index,
        times_correct=times_correct,
        no_xlabel=False,
        continuum_texname=continuum_texname,
        **kwargs
    )

    if colorbar:
        # Work in progress -- add a colorbar to the display plot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=imshowlims_data[0], vmax=imshowlims_data[1])
        )
        sm._A = []
        plt.colorbar(sm, ax=[ax[0],ax[1]],location='right')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=imshowlims_resid[0], vmax=imshowlims_resid[1])
        )
        sm._A = []
        plt.colorbar(sm, ax=[ax[2]])

    ax[0].locator_params(nbins=5, axis='y')
    ax[1].locator_params(nbins=5, axis='y')
    ax[2].locator_params(nbins=5, axis='y')

    if lc_xlim != []:
        ax[4].set_xlim(lc_xlim[0], lc_xlim[1])
        ax[5].set_xlim(lc_xlim[0], lc_xlim[1])
    # Make sure both light curves share the same xlim
    xlims = min(ax[4].get_xlim()[0], ax[5].get_xlim()[0]), max(ax[4].get_xlim()[1], ax[5].get_xlim()[1])
    ax[4].set_xlim(xlims)
    ax[5].set_xlim(xlims)

    if savename != '':
        plt.savefig(savename)
    return ax
