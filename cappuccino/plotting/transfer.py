import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import buildFigures as bf
from cappuccino.utils import constants as const
from cappuccino.utils import util

color_cycle = mpl.rcParams['axes.prop_cycle'].by_key()['color']

class TransferFunction:
    """
    Transfer function plotting object

    :param fp_clouds: Path and filename of the clouds file. If provided, will load.
    :type fp_clouds: str

    :param texname: LaTeX name for labelling the transfer function. Defaults to ''
    :type texname: str

    :param redshift: Redshift of the AGN. Defaults to 0.
    :type redshift: float

    :param tf:
    :type tf:
    """
    def __init__(self, fp_clouds='', texname='', redshift=0., tf=None):
        self.texname = texname
        self.redshift = redshift
        if fp_clouds != '':
            self.loadClouds(fp_clouds)
        try:
            for attr in vars(tf).keys():
                setattr(self, attr, getattr(tf, attr))
        except:
            pass
        self.vr_lags = {}
        self.avg_lag = {}

    def loadClouds(self,fp_clouds):
        """
        Loads in a clouds file and computes the cloud positions, wavelength shift, weights, and delays

        :param fp_clouds: Path and filename of the clouds file.
        :type fp_clouds: str

        :return: None
        """
        clouds = np.loadtxt(fp_clouds)

        # Define clouds:
        x = clouds[:, 0] / (const.c * const.day)
        y = clouds[:, 1] / (const.c * const.day)
        z = clouds[:, 2] / (const.c * const.day)

        self.lam = clouds[:, 6] / (1. + self.redshift)
        self.w = clouds[:, 8]
        self.delays = np.sqrt(x**2 + y**2 + z**2) - x
        self.delays /= (1.0 + self.redshift)
        self.x1, self.y1, self.z1 = x, y, z
        self.vx = -clouds[:,3]/100000

    def setMeasuredLags(self, frame, bins, lags):
        """
        Sets the measured velocity-resolved lags and bins

        :param frame: Whether bins and lags are given in 'observed' or 'rest' frame
        :type frame: str

        :param bins: Edges of the bins (in wavelength)
        :type bins: 2d array (Nbins x 2)

        :param lags: lags and upper/lower uncertainties for each bin
        :type lags: 2d array (Nbins x 3)
        """
        bins = np.array(bins)
        lags = np.array(lags)
        if frame == 'rest':
            self.vr_lags['bins'] = bins
            self.vr_lags['lags'] = lags
        elif frame == 'observed':
            self.vr_lags['bins'] = bins/(1.0 + self.redshift)
            self.vr_lags['lags'] = lags/(1.0 + self.redshift)

    def binClouds(self, tMin=0.0, tMax=40.0, lamMin=-9e9, lamMax=9e9, numLagBins=100,
        numLamBins=100, tMax_avg=0., linecenter=0., fp_spectrum='', zero_center=False):
        """
        Puts the clouds into wavelength bins

        :param tMin: Minimum lag bin edge. Defaults to 0.0.
        :type tMin: float

        :param tMax: Maximum lag bin edge. Defaults to 40.0.
        :type tMax: float

        :param tMax_avg: Maximum lag used when computing average lag. Defaults to same as plotted max lag.
        :type tMax_avg: float

        :param lamMin: Minimum wavelength bin edge.
        :type lamMin: float

        :param lamMax: Maximum wavelength bin edge.
        :type lamMax: float

        :param numLagBins: Number of bins on the lag axis
        :type numLagBins: int

        :param numLamBins: Number of bins on the wavelength axis
        :type numLamBins: int

        :param linecenter: Center of the emission line. Defaults to 0.0
        :type linecenter: float

        :param fp_spectrum: Path to the spectra files. If provided, will used wavelength bins of spectra
        :type fp_spectrum: str

        :param zero_center: Center on the emission line center. Defaults to False.
        :type zero_center: bool
        """

        # Put clouds in bins:
        # tMax: For greater resolution when taking averages :)

        # If spectrum is provided, pull the wavelength values
        if fp_spectrum == '':
            lambda_list = self.lam
        else:
            lambda_list = np.loadtxt(fp_spectrum)[0, :]
            lambda_list /= (1. + self.redshift)

        # If min and max wavelength bins aren't provided, use min/max wavelengths of the clouds
        # in the clouds file or min/max wavelength bins of the spectra.
        if lamMin==-9e9:
            self.lamMin = min(lambda_list)
        else:
            self.lamMin = lamMin
        if lamMax==9e9:
            self.lamMax = max(lambda_list)
        else:
            self.lamMax = lamMax

        if linecenter != 0. and zero_center:
            lamwidth = min([linecenter - self.lamMin, self.lamMax - linecenter])
            self.lamMin = linecenter - lamwidth
            self.lamMax = linecenter + lamwidth
        
        print """Computing 2d transfer function with %i lag bins and %i wavelength bins,
            (lamMin, lamMax) = (%.1f, %.1f) Angstroms,
            (lagMin, lagMax) = (%.0f, %.0f) days""" % (numLagBins, numLamBins,
                self.lamMin, self.lamMax, tMin, tMax)

        self.tMin = tMin
        self.tMax = tMax
        self.numLagBins = numLagBins
        self.numLamBins = numLamBins
        if tMax_avg == 0.:
            self.tMax_avg = tMax
        else:
            self.tMax_avg = tMax_avg

        dlam = (self.lamMax - self.lamMin) / float(self.numLamBins)
        dt = (self.tMax - self.tMin) / float(self.numLagBins)
        
        # Build up the 2d transfer function
        self.img = np.zeros((self.numLagBins, self.numLamBins))
        for i in range(0, len(self.x1)):
            # Lag bin. 0 is top (longest lags)
            ii = int(np.floor((self.tMax - self.delays[i]) / dt))
            # Wavelength bin. 0 is leftmost (shortest) wavelength
            jj = int(np.floor((self.lam[i] - self.lamMin) / dlam))
            if ii >= 0 and ii < self.numLagBins and jj >= 0 and jj < self.numLamBins:
                self.img[ii][jj] += self.w[i]

        # This is used for computing the mean lags at each wavelength. Can be 
        #  different than self.img in case you don't want to integrate/calculate
        #  all the way out to extremely long lags
        dt = (self.tMax_avg - self.tMin) / self.numLagBins
        self.img_avg = np.zeros((self.numLagBins, self.numLamBins))
        for i in range(0, len(self.x1)):
            ii = int(np.floor((self.tMax_avg - self.delays[i]) / dt))
            jj = int(np.floor((self.lam[i] - self.lamMin) / dlam))
            if ii >= 0 and ii < self.numLagBins and jj >= 0 and jj < self.numLamBins:
                self.img_avg[ii][jj] += self.w[i]

    def velocityIntegrated(self):
        """
        Calculate velocity-integrated transfer function
        """
        self.vel_int = []
        for i in range(0, self.numLagBins):
            small_img = self.img[i, :]
            tot_flux = sum(small_img)
            self.vel_int = self.vel_int + [tot_flux]
        self.vel_int.reverse()

    def lagIntegrated(self, tMax_int=0):
        """
        Calculate lag-integrated transfer function
        """
        # Calculate lag-integrated transfer function:
        dt = (self.tMax_avg - self.tMin) / self.numLagBins
        self.lag_int = []
        for i in range(0, self.numLamBins):
            small_img = self.img[:, i]
            if tMax_int != 0:
                tMax_int_index = len(small_img) - int(tMax_int/dt)
                tot_flux = sum(small_img[tMax_int_index:])
            else:
                tot_flux = sum(small_img)
            self.lag_int = self.lag_int + [tot_flux]

    def lagMean(self, lagtype='mean'):
        """
        Calculate mean or median lag spectrum

        :param lagtype: Either 'mean' or 'median'
        :type lagtype: str
        """
        self.avg_lag[lagtype] = []
        times = np.linspace(self.tMin, self.tMax_avg, self.numLagBins)
        self.avg_lams = np.linspace(self.lamMin, self.lamMax, self.numLamBins)
        if lagtype == 'mean':    
            for i in range(0, self.numLamBins):
                small_img = self.img_avg[:, i]
                tot = 0.0
                for m in range(0,np.size(small_img)):
                    tot = tot + times[m] * small_img[m]
                if sum(small_img) == 0:
                    lag = 0
                else:
                    lag = self.tMax_avg - tot/sum(small_img)
                self.avg_lag[lagtype] = self.avg_lag[lagtype] + [lag]
        elif lagtype == 'median':    
            for i in range(0, self.numLamBins):
                small_img = self.img_avg[::-1, i]
                m, tot, condition = 0, 0.0, True
                while condition:
                    if sum(small_img) == 0.0:
                        lag = 0
                        condition = False
                    tot = tot + small_img[m]
                    if tot > 0.5 * sum(small_img):
                        if m == 0:
                            lag = times[0]
                        else:
                            lag = (times[m] + times[m-1])/2.
                        condition = False
                    m += 1
                self.avg_lag[lagtype] = self.avg_lag[lagtype] + [lag]

    def velocityResolved(self, bins=[], lagtype='mean'):
        """
        Computes velocity-resolved lags

        :param bins: Bins for velocity-resolved measurements. In wavelength units
        :type bins: list of length two bins

        :param lagtype: 'mean' or 'median'
        :type lagtype: str
        """
        if bins == []:
            bins = self.vr_lags['bins']

        if lagtype not in self.avg_lag.keys():
            self.lagMean(lagtype=lagtype)
        lams = self.avg_lams

        # Compute the mean lag in the 1d mean or median transfer function
        velocity_lags = []
        for i in range(len(bins)):
            lind, rind = np.argmin(abs(bins[i][0] - lams)), np.argmin(abs(bins[i][1] - lams))
            velocity_lags.append(np.mean(self.avg_lag[lagtype][lind:rind+1]))
            self.vr_lags[lagtype] = velocity_lags
        
    def plot2dTransfer(self, xtype='velocity', linecenter=0., xticks=[], yticks=[], ax=None,
        figsize=(4,5), xlims=[], ylims=[], colorbar=False, **kwargs):
        """
        Makes a 2d transfer function.

        :param xtype: Wavelength or Velocity on the x-axis
        :type xtype: str

        :param linecenter: Emission line center. Required if xtype='velocity'
        :type linecenter: float

        :param xticks: Set the xticks, if provided.
        :type xticks: list

        :param yticks: Set the yticks, if provided.
        :type yticks: list

        :param ax: Axis to plot in. If None, will create new figure
        :type ax: matplotlib axis object

        :param figsize: Size of the axis (if axis passed), or size of the new figure.
        :type figsize: tuple, length 2

        :param xlims: Set the x axis limits
        :type xlims: tuple or list, length 2

        :param ylims: Set the y axis limits
        :type ylims: tuple or list, length 2

        :param colorbar: Set a colorbar above figure. Defaults to False.
        :type colorbar: bool
        """
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=figsize)
    
        if type(linecenter) != list:
            linecenter = [linecenter]
        lcen = np.mean(linecenter)

        # Bin the clouds if not already done
        if not hasattr(self, 'img'):
            binclouds_kwargs = {}
            for key in kwargs.keys():
                if key in ['tMin', 'tMax', 'lamMin', 'lamMax', 'numLagBins', 'numLamBins',
                    'tMax_avg', 'linecenter', 'zero_center']:
                    binclouds_kwargs[key] = kwargs[key]
            self.binClouds(**binclouds_kwargs)

        # Get the indices to plot
        if len(xlims) != 0:
            if xtype == 'velocity':
                xlims = [lim / const.ckm * lcen + lcen for lim in xlims]
            if xlims[0] < self.lamMin or xlims[1] > self.lamMax:
                print "Trying to plot outside of computed x range. Exiting."
                return
            wavelengths = np.linspace(self.lamMin, self.lamMax, self.numLamBins)
            lind, rind = np.argmin(abs(wavelengths - xlims[0])), np.argmin(abs(wavelengths - xlims[1]))
        else:
            xlims = [self.lamMin, self.lamMax]
            lind,rind = 0,self.numLamBins
        if len(ylims) != 0:
            if ylims[0] < self.tMin or ylims[1] > self.tMax:
                print "Trying to plot outside of computed y range. Exiting."
                return
            delays = np.linspace(self.tMin, self.tMax, self.numLagBins)
            bind, tind = np.argmin(abs(delays - ylims[0])), np.argmin(abs(delays - ylims[1]))
        else:
            ylims = [self.tMin, self.tMax]
            bind, tind = 0, self.numLagBins


        # get the axis dimensions
        lstart, bstart, spwidth, spheight = ax.get_position().bounds
        figwidth, figheight = plt.gcf().get_size_inches()
        if xtype == 'velocity':
            xlims_vel = [(lim - lcen)/lcen * const.ckm for lim in xlims]
            extent = [xlims_vel[0], xlims_vel[1], ylims[0], ylims[1]]
            aspects = (xlims_vel[1] - xlims_vel[0]) / (ylims[1] - ylims[0]) * \
                      spheight/spwidth * figheight/figwidth
        elif xtype == 'wavelength':
            extent = [xlims[0], xlims[1], ylims[0], ylims[1]]
            aspects = (xlims[1] - xlims[0]) / (ylims[1] - ylims[0]) * \
                      spheight/spwidth * figheight/figwidth

        vmin = 0.0
        vmax = max(max(self.img[i, j] for j in range(len(self.img[i])))
            for i in range(len(self.img))
        )
        ax.imshow(
            self.img[self.numLagBins-tind:self.numLagBins-bind,lind:rind],
            extent=extent,
            aspect=aspects,
            interpolation=None,
            vmin=vmin,
            vmax=vmax
            )

        if xtype == 'velocity':
            for l in linecenter:
                ax.axvline((l - lcen)/lcen * const.ckm, color='white', ls='dashed', lw=0.5)
        elif xtype == 'wavelength':
            for l in linecenter:
                if l != 0:
                    ax.axvline(l, color='white', ls='dashed', lw=0.5)

        if colorbar:
            lstart, bstart, spwidth, spheight = ax.get_position().bounds
            ax_cmap = plt.gcf().add_axes([
                lstart+0.05*spwidth,        # Left edge
                bstart+1.05*spheight,       # Bottom edge
                0.9*spwidth,                # Width
                0.05*spheight               # Height
            ])
            cbar = mpl.colorbar.ColorbarBase(ax_cmap, cmap='viridis', orientation='horizontal')
            cbar.set_ticks([0,1])
            vmax_str = "%.2e" % vmax
            cbar.ax.set_xticklabels(['${\\rm min}=0$','${\\rm max} = %s \\times 10^{%s%s}$' % (
                vmax_str.split("e")[0], vmax_str.split("e")[1][0], vmax_str.split("e")[1][1:].lstrip("0"))])
            #cbar.ax.set_xticklabels(['{\\rm min}','${\\rm max}$'])
            cbar.ax.xaxis.set_label_position("top")
            cbar.ax.xaxis.tick_top()
            cbar.ax.set_xlabel("$\\Psi(\\lambda,\\tau){\\rm(arbitrary)}$")

        if len(xticks) != 0:
            ax.set_xticks(xticks)
        if len(yticks) != 0:
            ax.set_yticks(yticks)
        ax.set_ylabel('$\\rm Rest~Frame~Delay~(days)$')
        if xtype == 'velocity':
            ax.set_xlabel('$\\rm Velocity~(km/s)$')
        elif xtype == 'wavelength':
            ax.set_xlabel('$\\rm Wavelength~(\\AA)$')

    def plotVelocityIntegrated(self, ax=None, figsize=(6,2), lagticks=[], sideways=False, laglims=[], **kwargs):
        """
        Plot the velocity-integrated transfer function
        
        :param ax: Axis to plot on. If None, will create now plot
        :type ax: matplotlib axis object

        :param figsize: Figure dimensions
        :type figsize: tuple, length 2

        :param lagticks: Set the lag ticks
        :type lagticks: list

        :param sideways: Plot the panel sideways. Default False.
        :type sideways: bool
        """
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=figsize)

        if not hasattr(self,'vel_int'):
            print "Computing the velocity-integrated transfer function"
            self.velocityIntegrated()

        # Velocity-integrated transfer function:
        times = np.linspace(self.tMin, self.tMax, self.numLagBins)
        if laglims != []:
            lind,rind = np.argmin(abs(laglims[0]-times)), np.argmin(abs(laglims[1]-times))
        else:
            lind,rind = 0,len(times)
        if sideways:
            if hasattr(self,'vel_int_std'):
                ax.errorbar(
                    -np.array(self.vel_int[lind:rind+1]),
                    times[lind:rind+1],
                    xerr=-np.array(self.vel_int_std[lind:rind+1]),
                    color='k', linewidth=1
                )
            else:
                ax.plot(-np.array(self.vel_int[lind:rind+1]), times[lind:rind+1], color='k', linewidth=1)
        else:
            if hasattr(self,'vel_int_std'):
                ax.errorbar(
                    times[lind:rind+1],
                    np.array(self.vel_int[lind:rind+1]),
                    yerr=np.array(self.vel_int_std[lind:rind+1]),
                    color='k', linewidth=1
                )
            else:
                ax.plot(times[lind:rind+1], np.array(self.vel_int[lind:rind+1]) , color='k', linewidth=1)

        xlabel = '$\\rm Rest~Frame~Delay~(days)$'
        ylabel = '$\\Psi(\\tau)$'
        if sideways:
            ax.set_xlabel(ylabel)
            ax.set_ylabel(xlabel)
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_ylim([self.tMin, self.tMax])
            ax.set_xlim(ax.get_xlim()[0], 0)
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
            if laglims != []:
                ax.set_ylim(laglims)
            if lagticks != []:
                ax.set_yticks(lagticks)
            ax.set_xticks([])
            ax.set_xticklabels('', visible=False)
        else:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim([self.tMin, self.tMax])
            ax.set_ylim(0,ax.get_ylim()[1])
            if laglims != []:
                ax.set_xlim(laglims)
            if lagticks != []:
                ax.set_xticks(lagticks)
            ax.set_yticks([])
            ax.set_yticklabels('', visible=False)

    def plotLagIntegrated(self, xtype='velocity', linecenter=0., xticks=[], xlims=[], ax=None, figsize=(6,2), **kwargs):
        """
        Plot the lag-integrated transfer function

        :param xtype: Wavelength or velocity on x-axis
        :type xtype: str

        :param linecenter: Emission line center. Required if xtype=velocity
        :type linecenter: float

        :param xticks: Set the x-axis ticks
        :type xticks: list

        :param xlims: x-axis limits
        :type xlims: list or tuple, length 2

        :param ax: Axis to plot on. If None, will create now plot
        :type ax: matplotlib axis object

        :param figsize: Figure dimensions
        :type figsize: tuple, length 2
        """
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=figsize)
        
        if type(linecenter) != list:
            linecenter = [linecenter]
        lcen = np.mean(linecenter)
        wavelengths = np.linspace(self.lamMin, self.lamMax, self.numLamBins)
        
        if xtype == 'wavelength':
            xvals = wavelengths
        elif xtype == 'velocity':
            xvals = [(lam - lcen)/lcen * const.ckm for lam in wavelengths]
        
        if not hasattr(self,'lag_int'):
            print "Computing the lag-integrated transfer function"
            self.lagIntegrated()
        if hasattr(self,'lag_int_std'):
            ax.errorbar(xvals, np.array(self.lag_int), yerr=np.array(self.lag_int_std), color='k', linewidth=1)
        else:
            ax.plot(xvals, np.array(self.lag_int), color='k', linewidth=1)

        if xtype == 'wavelength':
            ax.set_xlabel('$\\rm Rest~Wavelength~(\AA)$')
            ax.set_xlim(self.lamMin, self.lamMax)
            for l in linecenter:
                if l != 0:
                    ax.axvline(l,color='k',ls='dashed',lw=0.5)
        elif xtype == 'velocity':
            ax.set_xlabel('${\\rm Velocity~(km~s}^{-1}{\\rm)}$')
            ax.set_xlim([(self.lamMin - lcen)/lcen * const.ckm,
                        (self.lamMax - lcen)/lcen * const.ckm])
            for l in linecenter:
                ax.axvline((l - lcen)/lcen * const.ckm, color='k', ls='dashed', lw=0.5)

        if len(xticks) != 0:
            ax.set_xticks(xticks)

        if len(xlims) !=  0:
            ax.set_xlim(*xlims)

        ylim = ax.get_ylim()
        ax.set_ylim(0, 1.1 * ylim[1])
        ax.set_yticks([])
        ax.set_yticklabels([])

        ax.set_ylabel('$\\Psi(\\lambda)$')

    def plotLagMean(self, lagtype='mean', xtype='velocity', linecenter=0., xticks=[], yticks=[], xlims=[], measured_lags={}, ax=None, figsize=(6,2), **kwargs):
        """
        Plot the lag-mean or lag-median transfer function

        :param lagtype: Either 'mean' or 'median' lag
        :type lagtype: str

        :param xtype: Wavelength or velocity on x-axis
        :type xtype: str

        :param linecenter: Emission line center. Required if xtype=velocity
        :type linecenter: float

        :param xticks: Set the x-axis ticks
        :type xticks: list

        :param yticks: Set the y-axis ticks
        :type yticks: list

        :param xlims: x-axis limits
        :type xlims: list or tuple, length 2

        :param ax: Axis to plot on. If None, will create now plot
        :type ax: matplotlib axis object

        :param figsize: Figure dimensions
        :type figsize: tuple, length 2
        """
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=figsize)
        
        if type(linecenter) != list:
            linecenter = [linecenter]
        lcen = np.mean(linecenter)
        wavelengths = np.linspace(self.lamMin, self.lamMax, self.numLamBins)
        
        if xtype == 'wavelength':
            xvals = wavelengths
        elif xtype == 'velocity':
            xvals = [(lam - lcen)/lcen * const.ckm for lam in wavelengths]
        
        if lagtype not in self.avg_lag.keys():
            print "Computing the %s lag spectrum" % lagtype
            self.lagMean(lagtype=lagtype)
        if hasattr(self,'avg_lag_std'):
            if lagtype in self.avg_lag_std.keys():
                ax.errorbar(xvals, np.array(self.avg_lag[lagtype]), yerr=np.array(self.avg_lag_std[lagtype]), color='k', linewidth=1)
            else:
                ax.plot(xvals, np.array(self.avg_lag[lagtype]), color='k', linewidth=1, **kwargs)
        else:
            ax.plot(xvals, np.array(self.avg_lag[lagtype]), color='k', linewidth=1, **kwargs)
        
        if xtype == 'wavelength':
            ax.set_xlabel('$\\rm Rest~Wavelength~(\AA)$')
            ax.set_xlim(self.lamMin, self.lamMax)
            for l in linecenter:
                if l != 0:
                    ax.axvline(l,color='k',ls='dashed',lw=0.5)
        elif xtype == 'velocity':
            ax.set_xlabel('${\\rm Velocity~(km~s}^{-1}{\\rm)}$')
            ax.set_xlim([(self.lamMin - lcen)/lcen * const.ckm,
                        (self.lamMax - lcen)/lcen * const.ckm])
            for l in linecenter:
                ax.axvline((l - lcen)/lcen * const.ckm, color='k', ls='dashed', lw=0.5)

        if measured_lags != {}:
            self.setMeasuredLags(**measured_lags)
            if lagtype not in self.vr_lags.keys():
                self.velocityResolved(lagtype=lagtype)

            for i in range(len(self.vr_lags['bins'])):
                if xtype == 'velocity':
                    leftside = (self.vr_lags['bins'][i][0] - lcen)/lcen * const.ckm
                    rightside = (self.vr_lags['bins'][i][1] - lcen)/lcen * const.ckm
                else:
                    leftside = self.vr_lags['bins'][i][0]
                    rightside = self.vr_lags['bins'][i][1]
                x_cen = np.median([leftside, rightside])
                x_err = x_cen - leftside
                y = self.vr_lags[lagtype][i]

                # Modeled velocity-resolved
                if lagtype+'_std' in self.vr_lags.keys():
                    y_err = self.vr_lags[lagtype+'_std'][i]
                    ebar = ax.errorbar(
                        [x_cen], y,
                        xerr=[x_err],
                        yerr=[y_err],
                        color=color_cycle[0],
                        ls='',
                        marker='',
                        lw=1.0,
                        capsize=1,
                        zorder=3
                        )
                else:
                    ebar = ax.errorbar(
                        [x_cen], y,
                        xerr=[x_err],
                        color=color_cycle[0],
                        ls='',
                        marker='',
                        lw=1.0,
                        capsize=1,
                        zorder=3
                        )
                # Measured velocity-resolved
                ebar = ax.errorbar(
                    [x_cen], [self.vr_lags['lags'][i][0]],
                    xerr=[x_err], yerr=[[self.vr_lags['lags'][i][2]],[self.vr_lags['lags'][i][1]]],
                    color=color_cycle[1],
                    ls='',
                    marker='',
                    lw=1.0,
                    capsize=1,
                    zorder=3
                )

        if len(xlims) != 0:
            ax.set_xlim(*xlims)
        ylim = ax.get_ylim()
        ax.set_ylim(0,ylim[1])
        
        if len(yticks) != 0:
            ax.set_yticks(yticks)
        if lagtype == 'mean':
            ax.set_ylabel('${\\rm Mean~Delay}$\n$({\\rm days})$')
        elif lagtype == 'median':
            ax.set_ylabel('${\\rm Median~Delay}$\n$({\\rm days})$')
    
    def plotTransfer(self, xtype='velocity', linecenter=0., lagtype='mean',
        xlims=[], ylims=[], xticks=[], meanlagticks=[], lagticks=[],
        fig=None, ax=None, texname = '', 
        rowheight=[], colwidth=[], figsize=(4,5), pad_inches=[0.05,0.01,0.01,0.05,0.03,0.03], 
        colorbar=False,
        measured_lags={},
        modeled_range=[], subplotlabel='', **kwargs):
        """
        Builds up a standard transfer function plot, including the 2d, lag-integrated, and
        velocity-integrated transfer functions.

        :param fig:
        :type fig: 

        :param ax:
        :type ax: 
        
        :param texname:
        :type texname: 
        
        :param xlims:
        :type xlims: 
        
        :param ylims:
        :type ylims: 
        
        :param yticks:
        :type yticks: 
        
        :param rowheight: Relative heights of each subplot
        :type rowheight: list of floats
        
        :param colheight: Relative widths of each subplot
        :type colheight: list of floats
        
        :param cbar_box:
        :type cbar_box: 
        
        :param lambda_ticklabels:
        :type lambda_ticklabels: 

        :param lag_labels:
        :type lag_labels: 
        
        :param linecenter:
        :type linecenter: 
        
        :param lagtype:
        :type lagtype: 
        
        :param measured_lags: Measured velocity-resolved lags to plot. Dictionary with keys 
            'frame': 'rest' or 'observed'
            'lags': 
        :type measured_lags: dict
        
        :param figsize:
        :type figsize: 
        
        """

        # Bin the clouds if not already done
        if not hasattr(self, 'img'):
            binclouds_kwargs = {}
            for key in kwargs.keys():
                if key in ['tMin', 'tMax', 'lamMin', 'lamMax', 'numLagBins', 'numLamBins',
                    'tMax_avg', 'linecenter', 'zero_center']:
                    binclouds_kwargs[key] = kwargs[key]
            self.binClouds(**binclouds_kwargs)

        if type(lagtype) != list:
            lagtype = [lagtype]

        # Set the relative sizes of the subplots
        if len(lagtype) == 2:
            nrows = 3
            if len(rowheight) == 0:
                rowheight = [4.0,1.0,1.0]
        elif len(lagtype) == 3:
            nrows = 4
            if len(rowheight) == 0:
                rowheight = [4.0,1.0,1.0, 1.0]
        else:
            nrows = 2
            if len(rowheight) == 0:
                rowheight = [3.0,1.0]
        if len(colwidth) == 0:
            colwidth = [4.0,1.0]
        
        # Create the empty figure object
        if fig is None:
            fig = bf.CustomFigure()
            fig.fixedDimensions(figsize, pad_inches)
            fig.setRowsCols((1, 1))
            ax = fig.newSubplot(position=(0, 0), visible=False)

        self.ax = fig.makeSubpanelsUneven(
            ax,
            dims=(2, nrows),
            colwidth=colwidth,
            rowheight=rowheight
        )
        for i in range(len(lagtype)):
            self.ax[1, i+1].set_visible(False)

        if type(linecenter) != list:
            linecenter = [linecenter]
        
        # Find the closest wavelength values to xlims
        if len(xlims) != 0:
            if xtype=='velocity':
                pass
            else:
                wavelengths = np.linspace(self.lamMin, self.lamMax, self.numLamBins)
                lind,rind = np.argmin(abs(wavelengths - xlims[0])), np.argmin(abs(wavelengths - xlims[1]))
                xlims = [wavelengths[lind], wavelengths[rind]]
        
        self.plot2dTransfer(
            ax=self.ax[0, 0],
            xtype=xtype,
            linecenter=linecenter,
            xticks=xticks,
            xlims=xlims,
            ylims=ylims,
            yticks=lagticks,
            figsize=figsize,
            colorbar=colorbar,
        )

        # Plot the bottom subpanels -- integrated and/or mean/median
        for i in range(len(lagtype)):
            if lagtype[i] in ['mean', 'median']:
                self.plotLagMean(
                    ax=self.ax[0, i+1],
                    lagtype=lagtype[i],
                    xtype=xtype,
                    linecenter=linecenter,
                    xticks=xticks,
                    yticks=meanlagticks,
                    xlims=xlims,
                    measured_lags=measured_lags
                )
            elif lagtype[i] in ['int','integrated']:
                self.plotLagIntegrated(
                    ax=self.ax[0, i+1],
                    xtype=xtype,
                    linecenter=linecenter,
                    xticks=xticks,
                    xlims=xlims
                )
        
        # Plot the right subpanel -- velocity integrated
        self.plotVelocityIntegrated(
            ax=self.ax[1, 0],
            yticks=lagticks,
            sideways=True,
            laglims=ylims,
        )
        
        #if modeled_range != []:
        #    for i in range(len(lagtype)):
        #        self.ax[0,i+1].axvspan(xlim[0],modeled_range[0],color='grey', alpha=0.3)
        #        self.ax[0,i+1].axvspan(modeled_range[1],xlim[1],color='grey', alpha=0.3)
        #        self.ax[0,i].set_xticklabels('', visible=False)
        
        xlim = self.ax[0,0].get_xlim()
        ylim = self.ax[0,0].get_ylim()
        self.ax[0, 0].text(
            xlim[0] + 0.02 * (xlim[1] - xlim[0]),
            ylim[0] + 0.98 * (ylim[1] - ylim[0]),
            texname,
            verticalalignment='top',
            horizontalalignment='left',
            multialignment='left',
            color='white'
        )

        if subplotlabel != '':
            self.ax[0, 0].text(
                xlim[0] - 0.2 * (xlim[1] - xlim[0]),
                ylim[0] + 1.2 * (ylim[1] - ylim[0]),
                subplotlabel,
                verticalalignment='top',
                horizontalalignment='left',
                multialignment='left',
                color='black'
            )
        
        for i in range(nrows-1):
            self.ax[0,i].set_xticklabels([])


def plotMultiple(tf_list, redshift=0., dims=[0,0], figsize=[], pad_inches=[], positions=[],linecenter=0., lagtype='', **kwargs):
    """
    Quick function to plot several transfer functions in one figure.

    :param tf_list: List of TransferFunction objects or filepaths to clouds files. If the filepaths
        are provided, the code will create the TransferFunction objects using the redshift argument.
    :type tf_list: list
    
    :param redshift: List of redshifts of the AGNs, only used if filepaths are provided to tf_list.
        If a single float is provided, the code will use that value for all transfer functions.
    :type redshift: float or list of floats

    :param dims: Dimensions (nrows,ncols) of the subplot. By default, will find something close to 
        square.
    :type dims: list, length 2

    :param figsize: Total size of the figure in inches. By default, will use (2.5*ncols, 2.5*nrows)
    :type figsize: list, length 2

    :param pad_inches: Padding between subplots, in inches. Defaults to
        [left,right,top,bottom,column,row] = [0.05, 0.01, 0.01, 0.05, 0.10, 0.06] * label font size
    :type pad_inches: list, length 6

    :param positions: Positions (row,col) in which to plot the transfer functions
    :type positions: list, length(len(tf_list)) * 2


    """
    
    # If filepaths are provided, create the TransferFunction objects
    if type(tf_list[0]) in [str,unicode]:
        if type(redshift) != list:
            redshift = [redshift] * len(tf_list)
        print "Loading clouds files and creating TransferFunction objects:"
        for i in range(len(tf_list)):
            print "Using file %s and redshift = %.4f" % (tf_list[i], redshift[i])
            tf_list[i] = TransferFunction(redshift=redshift[i], fp_clouds=tf_list[i])
    # Set up figure dimensions
    if dims == [0,0]:
        dims = util.findDimensions(len(tf_list))
    nrows, ncols = dims
    if figsize == []:
        figsize = (2.5 * ncols, 2.8 * nrows)
    figwidth, figheight = figsize
    if pad_inches == []:
        fontsize = mpl.rcParams['axes.labelsize']
        if type(fontsize) == str:
            fontsize = util.mplFontScales(fontsize)

        pad_inches = list(np.array([0.05, 0.01, 0.01, 0.05, 0.10, 0.06]) * fontsize) # * mpl.rcParams['axes.labelsize']

    print "Initializing empty %ix%i figure, size %ix%i inches." % (dims[0],dims[1], figwidth, figheight) 
    fig = bf.CustomFigure()
    fig.fixedDimensions((figwidth, figheight), pad_inches)
    fig.setRowsCols((nrows, ncols))

    # Set subplot positions if not defined
    if positions == []:
        for i in range(ncols):
            for j in range(nrows):
                index = ncols * j + i
                if index >= len(tf_list):
                    break
                positions.append([j,i])
    positions = np.array(positions)
    
    for j in range(len(tf_list)):
        row = positions[j,0]
        col = positions[j,1]
        ax = fig.newSubplot(position=(col, row), visible=False)
        
        tf_list[j].plotTransfer(fig=fig, ax=ax, #texname=texnames[agn],
                    #lambda_ticklabels=_lambda_ticklabels,
                    #lag_labels=_lag_labels,
                    lagtype=lagtype,
                    linecenter=linecenter,
                    #xtype=xtype
                    **kwargs
        )
        if col != 0:
            tf_list[j].ax[0,0].set_ylabel('')
        if col != max(positions[:,1]):
            tf_list[j].ax[1,0].set_ylabel('')
        if row != max(positions[:,0]):
            tf_list[j].ax[0,-1].set_xlabel('')


def combineTransfer(tf_list, redshift=0., _type='median', tMin=0.0, tMax=40.0, tMax_avg=0., tMax_int=0.,
    lamMin=-9e9, lamMax=9e9, numLagBins=100, numLamBins=100, linecenter=0., zero_center=False,
    measured_lags={}):
    """
    Combines multiple transfer functions into one, including uncertainties

    :param tf_list: List of TransferFunction objects
    :type tf_list: list
    
    :param redshift: List of TransferFunction objects
    :type redshift: list
    
    :param _type: How to combine the transfer functions, either 'median' or 'mean'.
    :type _type: str

    :param tMin: Minimum lag bin edge. Defaults to 0.0.
    :type tMin: float

    :param tMax: Maximum lag bin edge. Defaults to 40.0.
    :type tMax: float
    
    :param tMax_avg: Maximum lag used when computing average lag. Defaults to same as plotted max lag.
    :type tMax_avg: float
    
    :param tMax_int: Maximum lag used when computing integrated lag. Defaults to same as plotted max lag.
    :type tMax_int: float

    :param lamMin: Minimum wavelength bin edge.
    :type lamMin: float

    :param lamMax: Maximum wavelength bin edge.
    :type lamMax: float

    :param numLagBins: Number of bins on the lag axis
    :type numLagBins: int

    :param numLamBins: Number of bins on the wavelength axis
    :type numLamBins: int

    :param linecenter: Center of the emission line. Defaults to 0.0
    :type linecenter: float

    :param zero_center: Center on the emission line center. Defaults to False.
    :type zero_center: bool
    """
    if type(tf_list[0]) in [str,unicode]:
        print "Loading clouds files and creating TransferFunction objects, using redshift = %.4f" % redshift
        tf_list = [TransferFunction(redshift=redshift, fp_clouds=tf) for tf in tf_list]

    print "Combining %i transfer functions using the %s." % (len(tf_list), _type)
    if lamMin == -9e9:
        lamMin = min([min(tf.lam) for tf in tf_list])
    if lamMax == 9e9:
        lamMax = max([max(tf.lam) for tf in tf_list])
        
    for i in range(len(tf_list)):
        tf = tf_list[i]
        tf.binClouds(
            tMin=tMin, tMax=tMax, tMax_avg=tMax_avg, lamMin=lamMin, lamMax=lamMax,
            numLagBins=numLagBins, numLamBins=numLamBins,
            linecenter=linecenter, zero_center=zero_center
        )
        tf.lagIntegrated(tMax_int=tMax_int)
        tf.lagMean(lagtype='mean')
        tf.lagMean(lagtype='median')
        tf.velocityIntegrated()
        if measured_lags != {}:
            tf.setMeasuredLags(**measured_lags)
            tf.velocityResolved(lagtype='mean')
            tf.velocityResolved(lagtype='median')

    if _type=='median':
        mean_tf = np.median([tf.img for tf in tf_list], axis=0)
        mean_tf_avg = np.median([tf.img_avg for tf in tf_list], axis=0)
    elif _type=='mean':
        mean_tf = np.mean([tf.img for tf in tf_list], axis=0)
        mean_tf_avg = np.mean([tf.img_avg for tf in tf_list], axis=0)
    sigma_tf = np.std([tf.img for tf in tf_list], axis=0)
    sigma_tf_avg = np.std([tf.img_avg for tf in tf_list], axis=0)
    
    # Create the new mean/median TransferFunction object
    tf = TransferFunction(redshift=tf_list[0].redshift)
    tf.lamMin, tf.lamMax, tf.tMin, tf.tMax = tf_list[0].lamMin, tf_list[0].lamMax, tf_list[0].tMin, tf_list[0].tMax
    tf.numLagBins, tf.numLamBins, tf.tMax_avg = tf_list[0].numLagBins, tf_list[0].numLamBins, tf_list[0].tMax_avg
    tf.img, tf.img_avg = mean_tf, mean_tf_avg
    tf.velocityIntegrated()
    tf.lagIntegrated(tMax_int=tMax_int)
    tf.lagMean(lagtype='mean')
    tf.lagMean(lagtype='median')
    if measured_lags != {}:
        tf.setMeasuredLags(**measured_lags)

    tf.vel_int_std = np.std([tf_tmp.vel_int for tf_tmp in tf_list], axis=0)
    tf.lag_int_std = np.std([tf_tmp.lag_int for tf_tmp in tf_list], axis=0)
    tf.avg_lag_std = {
        'mean': np.std([tf_tmp.avg_lag['mean'] for tf_tmp in tf_list], axis=0),
        'median': np.std([tf_tmp.avg_lag['median'] for tf_tmp in tf_list], axis=0)
    }
    if measured_lags != {}:
        tf.vr_lags['mean_std'] = np.std([tf_tmp.vr_lags['mean'] for tf_tmp in tf_list], axis=0)
        tf.vr_lags['median_std'] = np.std([tf_tmp.vr_lags['median'] for tf_tmp in tf_list], axis=0)

    # Create an uncertainty TransferFunction object
    tf_sigma = TransferFunction(redshift=tf_list[0].redshift)
    tf_sigma.lamMin, tf_sigma.lamMax, tf_sigma.tMin, tf_sigma.tMax = lamMin, lamMax, tMin, tMax,
    tf_sigma.numLagBins, tf_sigma.numLamBins, tf_sigma.tMax_avg = numLagBins, numLamBins, tMax_avg
    tf_sigma.img, tf_sigma.img_avg = sigma_tf, sigma_tf_avg
    tf_sigma.velocityIntegrated()
    tf_sigma.lagIntegrated(tMax_int=tMax_int)
    tf_sigma.lagMean(lagtype='mean')
    tf_sigma.lagMean(lagtype='median')

    return tf, tf_sigma


def main(to_test):
    if to_test == 'transferFunctionAll':
        agn_list = ['testagn']
        texnames = {'testagn':'${\\rm Test~AGN}$'}
        fp_clouds = {'testagn':'../tests/data/clouds_test.txt'}
        lambda_ticklabels = {'testagn':[4900, 4950, 5000, 5050, 5100]}
        fp_spectrum = {'testagn':'../tests/data/hbeta_spectra.txt'}
        transferFunctionAll(
            agn_list,
            texnames,
            fp_clouds,
            fp_spectrum=fp_spectrum,
            lambda_ticklabels=lambda_ticklabels,
            lagtype='int'
        )


if __name__ == '__main__':
    import sys

    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.unicode'] = True
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['lines.linewidth'] = 1

    to_test = sys.argv[1]
    main(to_test)

