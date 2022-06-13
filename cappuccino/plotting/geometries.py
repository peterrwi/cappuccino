from pylab import *
import matplotlib as mpl
from matplotlib.pyplot import *
from matplotlib.colors import LinearSegmentedColormap
import buildFigures as bf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = True

# Constants
Msun = 1.9891 * 10 ** 30.  # kg/Msun
c = 299792458.  # m/s
day = 86400.  # sec/day
pi = 3.14159265


class Geometry():
    def __init__(self, fp_clouds = '', texname = ''):
        self.texname = texname
        if fp_clouds != '':
            self.loadClouds(fp_clouds)

    def loadClouds(self,fp_clouds):
        clouds = np.loadtxt(fp_clouds)

        # Define clouds:
        self.x = clouds[:, 0] / (299792458.0 * 86400)
        self.y = clouds[:, 1] / (299792458.0 * 86400)
        self.z = clouds[:, 2] / (299792458.0 * 86400)
        self.vx = clouds[:, 3] / 1000
        self.vy = clouds[:, 4] / 1000
        self.vz = clouds[:, 5] / 1000
        maxvx = max([abs(val) for val in self.vx])
        w = clouds[:, 8]
        self.w = (w * 10000.) ** 2.
        #vx = vx / (2. * maxvx)
        #vx = vx + 0.5
        #_colors = [0] * len(vx)
        #for j in range(len(vx)):
        #    _colors[j] = [vx[j], 0, 1. - vx[j], 1]
        #self.vx = vx
        #self.colors = _colors

    def init3DAxis(self):
        self.ax3D = Axes3D(self.fig3D)
        # light days
        self.ax3D.scatter(self.x, self.y, self.z, s=self.w, marker='o',
                          alpha=0.1)
        #self.ax3D.quiver(
        #    [0], [0], [0],
        #    [1], [0], [0],
        #    length=10.,
        #    color='r', lw=3, alpha = 1
        #)
        self.ax3D.plot([0, 10], [0, 0], [0, 0], color='r', lw=3)
        self.ax3D.plot([10, 9], [0, 0], [0, 0.7], color='r', lw=3)
        self.ax3D.plot([10, 9], [0, 0], [0, -0.7], color='r', lw=3)

        self.ax3D.set_xlabel('$x~({\\rm light~days})$', fontsize=15)
        self.ax3D.set_ylabel('$y~({\\rm light~days})$', fontsize=15)
        self.ax3D.set_zlabel('$z~({\\rm light~days})$', fontsize=15)

        ## arcsec
        #scale = 1. / 1.191e+6 / 0.563 * 1000000. #  1. ld * kpc/ld * arcsec/kpc
        #self.ax3D.scatter(self.x * scale, self.y * scale, self.z * scale,
        #                  s=self.w, marker='o', alpha=0.1)
        #self.ax3D.quiver([0], [0], [0], [1], [0], [0], length=10. * scale,
        #                 color='r', lw=3)
        #self.ax3D.set_xlabel('$x~(\mu{\\rm as})$', fontsize=12)
        #self.ax3D.set_ylabel('$y~(\mu{\\rm as})$', fontsize=12)
        #self.ax3D.set_zlabel('$z~(\mu{\\rm as})$', fontsize=12)

        self.ax3D.set_xlim(-12, 12)
        self.ax3D.set_ylim(-12, 12)
        self.ax3D.set_zlim(-12, 12)
        self.ax3D.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax3D.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax3D.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        return self.fig3D,

    def animate(self,i):
        print i
        self.ax3D.view_init(elev=10., azim=float(i*1. + 20))
        return self.fig3D,

    def runAnimation(self, savename):
        self.fig3D = plt.figure()

        # Animate
        anim = animation.FuncAnimation(
            self.fig3D,
            self.animate,
            init_func=self.init3DAxis,
            frames=3, #360
            interval=500,
            blit=True,
            repeat=True
        )

        anim.save(savename, writer='imagemagick', fps=20)

    def plotGeo(self, ax=[], ax_extent=0, savename='', trim=1, alpha=0.1, sizescale=1.,
        color='blue', figwidth=5, pad_inches=[0.45, 0.1, 0.05, 0.5, 0.0, 0.5],
        axislabelfontsize=14, ticklabelfontsize=12, ticks=[]):

        mpl.rcParams.update({'xtick.labelsize': ticklabelfontsize})
        mpl.rcParams.update({'ytick.labelsize': ticklabelfontsize})
        mpl.rcParams.update({'axes.labelsize': axislabelfontsize})

        ## All the same color
        #in_out = np.median(post[agn].posteriors['inflowoutflow'])
        #_colors = [1. - (in_out + 1.) / 2., 0, (in_out + 1.) / 2., 1.]

        ## w = (w/median(w)*10)**2.
        ## Adjust for different figure sizes
        #w *= _spheightin / 1. * 0.5

        if len(ax) == 0:
            figheight = ((figwidth - pad_inches[0] - pad_inches[1] - pad_inches[4])/2.
                    + pad_inches[2] + pad_inches[3])

            Figure = bf.CustomFigure()
            Figure.fixedDimensions((figwidth, figheight), pad_inches)
            Figure.setRowsCols((1, 2))

            ax0 = Figure.newSubplot((0, 0))
            ax1 = Figure.newSubplot((1, 0))
            ax = [ax0,ax1]
        else:
            ax0,ax1 = ax
        
        if ax_extent == 0:
            # Enclose 95% of the BLR clouds
            ax_extent = max([np.percentile(vals,95)
                             for vals in [self.x,self.y,self.z]])
        min_axis = -ax_extent
        max_axis = ax_extent
        if len(ticks) == 0:
            tick2 = -int(max_axis * 0.40)
            tick1 = 2 * tick2
            tick3 = -tick2
            tick4 = -tick1
        else:
            tick1,tick2,_,tick3,tick4 = ticks

        if color == 'velocity':
            minmax = 0.5*np.max(np.sqrt(self.vx**2+self.vy**2+self.vz**2))
            sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-minmax, vmax=minmax), cmap="coolwarm_r")

        setp(ax0, ylim=(min_axis, max_axis), xlim=(min_axis, max_axis))
        ax0.set_yticks((tick1, tick2, 0, tick3, tick4))
        ax0.set_xticks((tick1, tick2, 0, tick3, tick4))
        if color == 'velocity':
            ax0.scatter(
                self.x[::trim],
                self.z[::trim],
                s=np.array(self.w)[::trim]*sizescale,
                c=sm.to_rgba(self.vx[::trim]),
                alpha=alpha,
                edgecolors='k'
            )
        else:
            ax0.scatter(self.x[::trim], self.z[::trim], marker='o', s=self.w[::trim]*sizescale, alpha=alpha, color=color, edgecolors='k', linewidths=0.2, rasterized=True)
        ax0.set_ylabel("$\\rm z$ $\\rm (light$ $\\rm days)$", labelpad=0.5)
        ax0.set_xlabel("$\\rm x$ $\\rm (light$ $\\rm days)$")

        setp(ax1, ylim=(min_axis, max_axis), xlim=(min_axis, max_axis))
        ax1.set_xticks((tick1, tick2, 0, tick3, tick4))
        ax1.set_yticks((tick1, tick2, 0, tick3, tick4), ())
        ax1.set_yticklabels([],visible=False)
        if color == 'velocity':
            ax1.scatter(
                self.y[::trim],
                self.z[::trim],
                s=np.array(self.w)[::trim]*sizescale,
                c=sm.to_rgba(self.vx[::trim]),
                alpha=alpha
            )
        else:
            ax1.scatter(self.y[::trim], self.z[::trim], marker='o', s=self.w[::trim]*sizescale, alpha=alpha, color=color, edgecolors='k', linewidths=0.2,rasterized=True)

        ax1.set_xticks((tick1, tick2, 0, tick3, tick4))
        ax1.set_xlabel("$\\rm y$ $\\rm (light$ $\\rm days)$")

        ax0.axvline(0, color='k', lw=0.5, ls='dashed')
        ax1.axvline(0, color='k', lw=0.5, ls='dashed')
        ax0.axhline(0, color='k', lw=0.5, ls='dashed')
        ax1.axhline(0, color='k', lw=0.5, ls='dashed')

        if savename != '':
            print "Saving..."
            savefig(savename,dpi=400)
        #plt.show()
        return ax
        #show()


    def plotGeoProjection(self, thetai, ax=[], xmax=0, ymax=0, trim=1, savename='', alpha=0.1, sizescale=1., color='blue', figwidth=3, pad_inches=[0.5, 0.1, 0.05, 0.4, 0.0, 0.5], axislabelfontsize=12, ticklabelfontsize=10):
        mpl.rcParams.update({'xtick.labelsize': ticklabelfontsize})
        mpl.rcParams.update({'ytick.labelsize': ticklabelfontsize})
        mpl.rcParams.update({'axes.labelsize': axislabelfontsize})

        ## w = (w/median(w)*10)**2.
        ## Adjust for different figure sizes
        #w *= _spheightin / 1. * 0.5


        if len(ax) == 0:
            figheight = ((figwidth - pad_inches[0] - pad_inches[1] - pad_inches[4])/1.5
                    + pad_inches[2] + pad_inches[3])

            Figure = bf.CustomFigure()
            Figure.fixedDimensions((figwidth, figheight), pad_inches)
            Figure.setRowsCols((1, 1))

            ax0 = Figure.newSubplot((0, 0))
            ax = [ax0]
        else:
            ax0 = ax[0]

        thetai = thetai * np.pi/180.
        xx,yy,zz,ww = self.x[::trim], self.y[::trim], self.z[::trim], self.w[::trim]
        r,z = np.zeros(len(xx)), np.zeros(len(xx))
        for i in range(len(xx)):
            r[i],z[i] = getRZ(xx[i],yy[i],zz[i],thetai)
        ax0.scatter(r,z,marker='o', s=ww*sizescale, alpha=alpha, color=color,rasterized=True)

        if xmax != 0:
            ax0.set_xlim([0,xmax])
        if ymax != 0:
            ax0.set_ylim([-ymax,ymax])
        if savename != '':
            print "Saving..."
            savefig(savename,dpi=400)
        
        return ax
        #show()

def rotatethetai(x,y,z,thetai):
    xnew = x * np.cos(thetai) - z * np.sin(thetai)
    ynew = y
    znew = x * np.sin(thetai) + z * np.cos(thetai)
    return xnew,ynew,znew

def getRZ(x,y,z,thetai):
    r = np.sqrt(x**2 + y**2 + z**2)
    _,_,z = rotatethetai(x,y,z,thetai)
    return r,z


def singleGeo(fp_clouds,ax_extent=0,savename=''):
    geo = Geometry(fp_clouds=fp_clouds)
    geo.plotGeo(ax_extent,savename)
    pass


def plotGeometriesSingle(agn, savetag=True, temp=0):
    folderlist = code.revmap.params.parInfo.runFilepaths()
    agnlist = [agn]

    for agn in agnlist:
        filepaths[agn] = "../CARAMEL_runs/" + agn + "/" + folderlist[agn][0] + "clouds/"

    cloudslist = code.revmap.params.parInfo.cloudsToUse()

    appendix = ""
    save_folder = "../figures/"
    fig_name = save_folder + "geo" + appendix + agn + "_small"

    clouds = {}
    for agn in agnlist:
        clouds[agn] = np.loadtxt(filepaths[agn] + cloudslist[agn][temp])[0:1000, :]
        # normalize the weights to account for different numbers of clouds
        clouds[agn][:, 6] = clouds[agn][:, 6] * 2

    # Set the figure dimensions
    figwidth = 5
    tl_W = 0.4  # width of tile
    Nagn = len(agnlist)
    figheight = figwidth + (Nagn - 2) * tl_W * figwidth
    tl_H = tl_W * (figwidth / figheight)
    tl_R = 0.015
    tl_T = tl_R * (figwidth / figheight)
    tl_left = 1. - 2. * tl_W - tl_R  # left edge of plot tiles
    tl_bottom = 1. - Nagn * tl_H - tl_T
    font_size = figwidth * 4
    axlabelsize = figwidth * 4

    figure(figsize=(figwidth, figheight))

    text_left = tl_left + 0.02
    text_bottom = tl_bottom + 0.01
    text_dh = tl_H
    matplotlib.rc('font', family='serif', serif='cm10')

    min_axis = -24.
    max_axis = 24.
    tick1 = min_axis + 4.
    tick2 = tick1 / 2.
    tick3 = max_axis - 4.
    tick4 = tick3 / 2.

    counter = 0.
    for agn in agnlist:
        figtext(text_left, text_bottom + text_dh * counter, agn, fontsize=font_size,
                color='r')

        x = clouds[agn][:, 0] / (299792458.0 * 86400)
        y = clouds[agn][:, 1] / (299792458.0 * 86400)
        z = clouds[agn][:, 2] / (299792458.0 * 86400)
        w = clouds[agn][:, 6]
        w = (w * 10000.) ** 2.

        # w = (w/median(w)*10)**2.

        ax = axes([tl_left, tl_bottom + counter * tl_H, tl_W, tl_H])
        setp(ax, ylim=(min_axis, max_axis), xlim=(min_axis, max_axis))
        xticks(size=18)
        yticks(size=18)
        yticks((tick1, tick2, 0, tick3, tick4))
        scatter(x, z, marker='o', s=w, alpha=0.1)
        ylabel("$\\rm z$ $\\rm (light$ $\\rm days)$", size=axlabelsize)
        if counter == 0:
            xticks((tick1, tick2, 0, tick3, tick4))
            xlabel("$\\rm x$ $\\rm (light$ $\\rm days)$", size=axlabelsize)
        else:
            xticks((tick1, tick2, 0, tick3, tick4), ())

        ax = axes([tl_left + tl_W, tl_bottom + counter * tl_H, tl_W, tl_H])
        setp(ax, ylim=(min_axis, max_axis), xlim=(min_axis, max_axis))
        xticks(size=18)
        yticks(size=18)
        xticks((tick1, tick2, 0, tick3, tick4))
        yticks((tick1, tick2, 0, tick3, tick4), ())
        scatter(y, z, marker='o', s=w, alpha=0.1)
        if counter == 0:
            xticks((tick1, tick2, 0, tick3, tick4))
            xlabel("$\\rm y$ $\\rm (light$ $\\rm days)$", size=axlabelsize)
        else:
            xticks((tick1, tick2, 0, tick3, tick4), ())

        counter += 1

    if savetag:
        savefig(fig_name + ".eps")
        savefig(fig_name + ".png")
    show()
    clf()


def plotGeometriesAll(agn_list, filenames, post, savetag=False, temp=0, _colors=True):

    appendix = "_all"
    save_folder = "../figures/"
    fig_name = save_folder + "geo" + appendix

    clouds = {}
    for agn in agn_list:
        clouds[agn] = np.loadtxt(filenames[agn])[0:1000, :]
        # normalize the weights to account for different numbers of clouds
        clouds[agn][:, 6] = clouds[agn][:, 6] * 2

    Nagn = len(agn_list)

    # Set the figure dimensions


    appendix = "4x2"
    nrows = 4
    ncols = 2
    agn_pos = [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1]]
    axes_size = [16, 16, 16, 16, 10, 10, 10]
    figheight = 7.
    """
    appendix = "portrait"
    nrows = 7
    ncols = 1
    agn_pos = [[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0]]
    axes_size = [10, 16, 16, 16, 10, 10, 16]
    figheight = 9.
    """

    _label_fontparams = {'fontsize': 12, 'color': 'red'}
    _axis_fontparams = {'fontsize': 12}
    _tick_fontparams = {'fontsize': 10}

    _lpadin = 0.5
    _rpadin = 0.2
    _bpadin = 0.5
    _tpadin = 0.1
    _wpadin = 0.65
    _hpadin = 0.0

    _spheightin = (figheight - _tpadin - _bpadin - _hpadin * (nrows - 1.)) / nrows
    _spwidthin = 2. * _spheightin
    figwidth = ncols * _spwidthin + (ncols - 1.) * _wpadin + _rpadin + _lpadin

    _lpad = _lpadin / figwidth
    _rpad = _rpadin / figwidth
    _hpad = _hpadin / figheight
    _tpad = _tpadin / figheight
    _bpad = _bpadin / figheight
    _wpad = _wpadin / figwidth

    _spwidth = _spwidthin / figwidth
    _spheight = _spheightin / figheight

    _panelwidth = _spwidth / 2.
    _panelheight = _spheight

    plt.figure(figsize=(figwidth, figheight))

    # matplotlib.rc('font', family='serif', serif='cm10')

    counter = 0.
    for agn in agn_list:
        print agn
        print "\n"

        column = agn_pos[int(counter)][1]
        rowlist = []
        for j in range(len(agn_pos)):
            if agn_pos[j][1] == column:
                rowlist.append(agn_pos[j][0])
        isbottom = agn_pos[int(counter)][0] == max(rowlist)

        l_start = _lpad + agn_pos[int(counter)][1] * (_spwidth + _wpad)
        t_start = _tpad + agn_pos[int(counter)][0] * (_spheight + _hpad)

        figtext(
            l_start + 0.02 / _spwidthin,
            1. - t_start - _spheight + 0.02 / _spheightin,
            post[agn].texname,
            **_label_fontparams)

        x = clouds[agn][:, 0] / (299792458.0 * 86400)
        y = clouds[agn][:, 1] / (299792458.0 * 86400)
        z = clouds[agn][:, 2] / (299792458.0 * 86400)
        vx = -clouds[agn][:, 3] / 100000
        maxvx = max([abs(val) for val in vx])
        w = clouds[agn][:, 6]
        w = (w * 10000.) ** 2.
        vx = vx / (2. * maxvx)
        vx = vx + 0.5
        _colors = [0] * len(vx)
        for i in range(len(vx)):
            _colors[i] = [vx[i], 0, 1. - vx[i], 1]

        # All the same color
        in_out = np.median(post[agn].posteriors['inflowoutflow'])
        _colors = [1. - (in_out + 1.) / 2., 0, (in_out + 1.) / 2., 1.]

        # w = (w/median(w)*10)**2.

        ax = axes((
            l_start,
            1. - t_start - 1. * _panelheight,
            _panelwidth,
            _panelheight)
        )
        min_axis = -axes_size[int(counter)]
        max_axis = axes_size[int(counter)]
        tick1 = -int(max_axis * 0.8)
        tick2 = tick1 / 2.
        tick3 = int(max_axis * 0.8)
        tick4 = tick3 / 2.
        setp(ax, ylim=(min_axis, max_axis), xlim=(min_axis, max_axis))
        yticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
        scatter(x, z, marker='o', s=w, alpha=0.1, color=_colors)
        ylabel("$\\rm z$ $\\rm (light$ $\\rm days)$", labelpad=0.5, **_axis_fontparams)

        if isbottom:
            xticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
            xlabel("$\\rm y$ $\\rm (light$ $\\rm days)$", **_axis_fontparams)
        else:
            xticks((tick1, tick2, 0, tick3, tick4), (), **_tick_fontparams)

        ax = axes((
            l_start + 1. * _panelwidth,
            1. - t_start - 1. * _panelheight,
            _panelwidth,
            _panelheight)
        )
        setp(ax, ylim=(min_axis, max_axis), xlim=(min_axis, max_axis))
        xticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
        yticks((tick1, tick2, 0, tick3, tick4), (), **_tick_fontparams)
        scatter(y, z, marker='o', s=w, alpha=0.1, color=_colors)

        if isbottom:
            xticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
            xlabel("$\\rm y$ $\\rm (light$ $\\rm days)$", **_axis_fontparams)
        else:
            xticks((tick1, tick2, 0, tick3, tick4), (), **_tick_fontparams)

        counter += 1

    if _colors:
        cmap_colors = [(1, 0, 0), (0, 0, 1)]  # R -> G -> B
        n_bins = 100  # Discretizes the interpolation into bins
        cmap_name = 'my_colormap'
        # Create the colormap
        cm = LinearSegmentedColormap.from_list(
            cmap_name, cmap_colors, N=n_bins)

        l_start = _lpad + 1 * (_spwidth + _wpad)
        t_start = _tpad + 3 * (_spheight + _hpad)

        ax_color = axes((
            l_start + 0.1 * _panelwidth,
            1. - t_start - 0.7 * _panelheight,
            _panelwidth * 1.8,
            _panelheight * 0.15)
        )
        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.
        # cmap = mpl.cm.cool
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        cb1 = mpl.colorbar.ColorbarBase(ax_color, cmap=cm,
                                        norm=norm,
                                        orientation='horizontal'
                                        )
        cb1.set_ticks([-1, -0.5, 0, 0.5, 1])
        cb1.set_ticklabels(["${\\rm Inflow}$", "", "", "", "${\\rm Outflow}$"])

    if savetag:
        print "Saving..."
        savefig(fig_name + "_" + appendix + ".pdf")
        savefig(fig_name + "_" + appendix + ".png")
    show()
    clf()


def allVertical2(agn_list, fp_clouds, post, hpad_bool, savename='', temp=0, _colors=True):
    clouds = {}
    for agn in agn_list:
        clouds[agn] = np.loadtxt(fp_clouds[agn])[0:1000, :]
        # normalize the weights to account for different numbers of clouds
        clouds[agn][:, 6] = clouds[agn][:, 6] * 2

    Nagn = len(agn_list)

    # Set the figure dimensions
    nrows = Nagn
    ncols = 1
    figheight = 8.0
    pad_inches = [0.5,0.2,0.1,0.5,0.65,0.5]

    Figure = bf.CustomFigure()
    Figure.fixedHeight(figheight,pad_inches,(nrows,ncols),spaxisratio=0.5)

    axes_size = [16, 16, 16, 16, 10, 10, 10]

    for i in range(Nagn):
        agn = agn_list[i]
        print agn
        print "\n"

        ax = Figure.newSubplot(position=(0, i))
        ax = Figure.makeSubpanels(
            ax,
            dims = (2, 1),
            pad = (0., 0.)
        )

        # Add AGN label
        lstart, bstart, spwidth, spheight = ax[0,0].get_position().bounds
        Figure.fig.figtext(
            lstart + 0.06 * spwidth,
            bstart  + 0.1 * spheight,
            post[agn].texname)

        x = clouds[agn][:, 0] / (299792458.0 * 86400)
        y = clouds[agn][:, 1] / (299792458.0 * 86400)
        z = clouds[agn][:, 2] / (299792458.0 * 86400)
        vx = -clouds[agn][:, 3] / 100000
        maxvx = max([abs(val) for val in vx])
        w = clouds[agn][:, 6]
        w = (w * 10000.) ** 2.
        vx = vx / (2. * maxvx)
        vx = vx + 0.5
        _colors = [0] * len(vx)
        for j in range(len(vx)):
            _colors[j] = [vx[j], 0, 1. - vx[j], 1]

        # All the same color
        in_out = np.median(post[agn].posteriors['inflowoutflow'])
        _colors = [1. - (in_out + 1.) / 2., 0, (in_out + 1.) / 2., 1.]

        # w = (w/median(w)*10)**2.
        # Adjust for different figure sizes
        w *= spheight / 1. * 0.5

        # Edge-on view
        min_axis = -axes_size[i]
        max_axis = axes_size[i]
        tick1 = -int(max_axis * 0.8)
        tick2 = tick1 / 2.
        tick3 = int(max_axis * 0.8)
        tick4 = tick3 / 2.
        ax[0,0].set_xlim(min_axis,max_axis)
        ax[0,0].set_ylim(min_axis,max_axis)
        setp(ax, ylim=(min_axis, max_axis), xlim=(min_axis, max_axis))
        yticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
        scatter(x, z, marker='o', s=w, alpha=0.1, color=_colors)
        ylabel("$\\rm z$ $\\rm (light$ $\\rm days)$", labelpad=0.5)
        if hpad_bool[i] == 1 or i == Nagn - 1:
            xticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
            xlabel("$\\rm y$ $\\rm (light$ $\\rm days)$", **_axis_fontparams)
        else:
            xticks((tick1, tick2, 0, tick3, tick4), (), **_tick_fontparams)

        # Face-on view
        setp(ax, ylim=(min_axis, max_axis), xlim=(min_axis, max_axis))
        xticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
        yticks((tick1, tick2, 0, tick3, tick4), (), **_tick_fontparams)
        scatter(y, z, marker='o', s=w, alpha=0.1, color=_colors)

        if hpad_bool[i] == 1 or i == Nagn - 1:
            xticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
            xlabel("$\\rm y$ $\\rm (light$ $\\rm days)$", **_axis_fontparams)
        else:
            xticks((tick1, tick2, 0, tick3, tick4), (), **_tick_fontparams)


    if _colors:
        cmap_colors = [(1, 0, 0), (0, 0, 1)]  # R -> G -> B
        n_bins = 100  # Discretizes the interpolation into bins
        cmap_name = 'my_colormap'
        # Create the colormap
        cm = LinearSegmentedColormap.from_list(
            cmap_name, cmap_colors, N=n_bins)

        ax - new
        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.
        # cmap = mpl.cm.cool
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        cb1 = mpl.colorbar.ColorbarBase(ax_color, cmap=cm,
                                        norm=norm,
                                        orientation='horizontal'
                                        )
        cb1.set_ticks([-1, -0.5, 0, 0.5, 1])
        cb1.set_ticklabels(["${\\rm Inflow}$", "", "", "", "${\\rm Outflow}$"])
        cb1.ax.tick_params(**_cbar_fontparams)

    if savename != '':
        print "Saving..."
        plt.savefig(savename)
    # show()
    clf()


def allVertical(agn_list, filenames, post, hpad_bool, savename, savetag=True, temp=0, _colors=True):

    clouds = {}
    for agn in agn_list:
        clouds[agn] = np.loadtxt(filenames[agn])[0:1000, :]
        # normalize the weights to account for different numbers of clouds
        clouds[agn][:, 6] = clouds[agn][:, 6] * 2

    Nagn = len(agn_list)

    # Set the figure dimensions

    nrows = Nagn
    ncols = 1
    axes_size = [16, 16, 16, 16, 10, 10, 10]
    figheight = 8.0

    _label_fontparams = {'fontsize': 10, 'color': 'black'}
    _axis_fontparams = {'fontsize': 9}
    _tick_fontparams = {'fontsize': 8}
    _cbar_fontparams = {'labelsize':9}

    _lpadin = 0.5
    _rpadin = 0.2
    _bpadin = 0.5
    _tpadin = 0.1
    _wpadin = 0.65
    _hpadin = 0.5

    _spheightin = (figheight - _tpadin - _bpadin - _hpadin * np.sum(hpad_bool)) / (nrows + 0.5)
    _spwidthin = 2. * _spheightin
    figwidth = ncols * _spwidthin + (ncols - 1.) * _wpadin + _rpadin + _lpadin

    _lpad = _lpadin / figwidth
    _rpad = _rpadin / figwidth
    _hpad = _hpadin / figheight
    _tpad = _tpadin / figheight
    _bpad = _bpadin / figheight
    _wpad = _wpadin / figwidth

    _spwidth = _spwidthin / figwidth
    _spheight = _spheightin / figheight

    _panelwidth = _spwidth / 2.
    _panelheight = _spheight

    plt.figure(figsize=(figwidth, figheight))

    # matplotlib.rc('font', family='serif', serif='cm10')

    for i in range(Nagn):
        agn = agn_list[i]
        print agn
        print "\n"

        figtext(
            _lpad + 0.06 * _panelwidth,
            1. - _tpad - _spheight * (i + 1) - sum(hpad_bool[:i]) * _hpad \
                + 0.1 * _panelheight,
            post[agn].texname,
            **_label_fontparams)

        x = clouds[agn][:, 0] / (299792458.0 * 86400)
        y = clouds[agn][:, 1] / (299792458.0 * 86400)
        z = clouds[agn][:, 2] / (299792458.0 * 86400)
        vx = -clouds[agn][:, 3] / 100000
        maxvx = max([abs(val) for val in vx])
        w = clouds[agn][:, 6]
        w = (w * 10000.) ** 2.
        vx = vx / (2. * maxvx)
        vx = vx + 0.5
        _colors = [0] * len(vx)
        for j in range(len(vx)):
            _colors[j] = [vx[j], 0, 1. - vx[j], 1]

        # All the same color
        in_out = np.median(post[agn].posteriors['inflowoutflow'])
        _colors = [1. - (in_out + 1.) / 2., 0, (in_out + 1.) / 2., 1.]

        # w = (w/median(w)*10)**2.
        # Adjust for different figure sizes
        w *= _spheightin / 1. * 0.5

        ax = axes((
            _lpad,
            1. - _tpad - _spheight * (i + 1) - sum(hpad_bool[:i]) * _hpad,
            _panelwidth,
            _panelheight
        ))
        min_axis = -axes_size[i]
        max_axis = axes_size[i]
        tick1 = -int(max_axis * 0.8)
        tick2 = tick1 / 2.
        tick3 = int(max_axis * 0.8)
        tick4 = tick3 / 2.
        setp(ax, ylim=(min_axis, max_axis), xlim=(min_axis, max_axis))
        yticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
        scatter(x, z, marker='o', s=w, alpha=0.1, color=_colors)
        ylabel("$\\rm z$ $\\rm (light$ $\\rm days)$", labelpad=0.5, **_axis_fontparams)

        if hpad_bool[i] == 1 or i == Nagn - 1:
            xticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
            xlabel("$\\rm x$ $\\rm (light$ $\\rm days)$", **_axis_fontparams)
        else:
            xticks((tick1, tick2, 0, tick3, tick4), (), **_tick_fontparams)

        ax = axes((
            _lpad + _panelwidth,
            1. - _tpad - _spheight * (i + 1) - sum(hpad_bool[:i]) * _hpad,
            _panelwidth,
            _panelheight)
        )
        setp(ax, ylim=(min_axis, max_axis), xlim=(min_axis, max_axis))
        xticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
        yticks((tick1, tick2, 0, tick3, tick4), (), **_tick_fontparams)
        scatter(y, z, marker='o', s=w, alpha=0.1, color=_colors)

        if hpad_bool[i] == 1 or i == Nagn - 1:
            xticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
            xlabel("$\\rm y$ $\\rm (light$ $\\rm days)$", **_axis_fontparams)
        else:
            xticks((tick1, tick2, 0, tick3, tick4), (), **_tick_fontparams)

    if _colors:
        cmap_colors = [(1, 0, 0), (0, 0, 1)]  # R -> G -> B
        n_bins = 100  # Discretizes the interpolation into bins
        cmap_name = 'my_colormap'
        # Create the colormap
        cm = LinearSegmentedColormap.from_list(
            cmap_name, cmap_colors, N=n_bins)

        ax_color = axes((
            _lpad + 0.1 * _panelwidth,
            1. - _tpad - _spheight * Nagn - sum(hpad_bool) * _hpad - 0.7 * _panelheight,
            _panelwidth * 1.8,
            _panelheight * 0.15)
        )
        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.
        # cmap = mpl.cm.cool
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        cb1 = mpl.colorbar.ColorbarBase(ax_color, cmap=cm,
                                        norm=norm,
                                        orientation='horizontal'
                                        )
        cb1.set_ticks([-1, -0.5, 0, 0.5, 1])
        cb1.set_ticklabels(["${\\rm Inflow}$", "", "", "", "${\\rm Outflow}$"])
        cb1.ax.tick_params(**_cbar_fontparams)

    if savetag:
        print "Saving..."
        savefig(savename)
    #show()
    clf()


def singleGeometry(agn, filename, post, savename = '', _colorbar=False):

    clouds = np.loadtxt(filename)[0:1000, :]
    # normalize the weights to account for different numbers of clouds
    clouds[:, 6] = clouds[:, 6] * 2

    nrows = 1
    ncols = 1
    axes_size = 16
    figheight = 2.

    _label_fontparams = {'fontsize': 10, 'color': 'black'}
    _axis_fontparams = {'fontsize': 9}
    _tick_fontparams = {'fontsize': 8}
    _cbar_fontparams = {'labelsize':9}

    _lpadin = 0.5
    _rpadin = 0.2
    _bpadin = 0.5
    _tpadin = 0.1
    _wpadin = 0.65
    _hpadin = 0.5

    _spheightin = (figheight - _tpadin - _bpadin) / (nrows + 0.5)
    _spwidthin = 2. * _spheightin
    figwidth = ncols * _spwidthin + (ncols - 1.) * _wpadin + _rpadin + _lpadin

    _lpad = _lpadin / figwidth
    _rpad = _rpadin / figwidth
    _hpad = _hpadin / figheight
    _tpad = _tpadin / figheight
    _bpad = _bpadin / figheight
    _wpad = _wpadin / figwidth

    _spwidth = _spwidthin / figwidth
    _spheight = _spheightin / figheight

    _panelwidth = _spwidth / 2.
    _panelheight = _spheight

    plt.figure(figsize=(figwidth, figheight))

    x = clouds[:, 0] / (299792458.0 * 86400)
    y = clouds[:, 1] / (299792458.0 * 86400)
    z = clouds[:, 2] / (299792458.0 * 86400)
    vx = -clouds[:, 3] / 100000
    maxvx = max([abs(val) for val in vx])
    w = clouds[:, 6]
    w = (w * 10000.) ** 2.
    vx = vx / (2. * maxvx)
    vx = vx + 0.5

    _colors = [0] * len(vx)
    for j in range(len(vx)):
        _colors[j] = [vx[j], 0, 1. - vx[j], 1]

    # All the same color
    in_out = np.median(post[agn].posteriors['inflowoutflow'])
    _colors = [1. - (in_out + 1.) / 2., 0, (in_out + 1.) / 2., 1.]

    # w = (w/median(w)*10)**2.
    # Adjust for different figure sizes
    w *= _spheightin / 1. * 0.5

    ax = axes((
        _lpad,
        1. - _tpad - _spheight,
        _panelwidth,
        _panelheight
    ))
    min_axis = -axes_size
    max_axis = axes_size
    tick1 = -int(max_axis * 0.8)
    tick2 = tick1 / 2.
    tick3 = int(max_axis * 0.8)
    tick4 = tick3 / 2.
    setp(ax, ylim=(min_axis, max_axis), xlim=(min_axis, max_axis))
    yticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
    scatter(x, z, marker='o', s=w, alpha=0.1, color=_colors)
    ylabel("$\\rm z$ $\\rm (light$ $\\rm days)$", labelpad=0.5, **_axis_fontparams)

    xticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
    xlabel("$\\rm x$ $\\rm (light$ $\\rm days)$", **_axis_fontparams)

    ax = axes((
        _lpad + _panelwidth,
        1. - _tpad - _spheight,
        _panelwidth,
        _panelheight)
    )
    setp(ax, ylim=(min_axis, max_axis), xlim=(min_axis, max_axis))
    xticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
    yticks((tick1, tick2, 0, tick3, tick4), (), **_tick_fontparams)
    scatter(y, z, marker='o', s=w, alpha=0.1, color=_colors)

    xticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
    xlabel("$\\rm y$ $\\rm (light$ $\\rm days)$", **_axis_fontparams)

    if _colorbar:
        cmap_colors = [(1, 0, 0), (0, 0, 1)]  # R -> G -> B
        n_bins = 100  # Discretizes the interpolation into bins
        cmap_name = 'my_colormap'
        # Create the colormap
        cm = LinearSegmentedColormap.from_list(
            cmap_name, cmap_colors, N=n_bins)

        ax_color = axes((
            _lpad + 0.1 * _panelwidth,
            1. - _tpad - _spheight - 0.7 * _panelheight,
            _panelwidth * 1.8,
            _panelheight * 0.15)
        )
        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.
        # cmap = mpl.cm.cool
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        cb1 = mpl.colorbar.ColorbarBase(ax_color, cmap=cm,
                                        norm=norm,
                                        orientation='horizontal'
                                        )
        cb1.set_ticks([-1, -0.5, 0, 0.5, 1])
        cb1.set_ticklabels(["${\\rm Inflow}$", "", "", "", "${\\rm Outflow}$"])
        cb1.ax.tick_params(**_cbar_fontparams)

    if savename != '':
        print "Saving..."
        savefig(savename)
    show()
    clf()


def allLandscape(agn_list, fp_clouds, post, pos, savename='', temp=0, _colors=True):
    clouds = {}
    for agn in agn_list:
        clouds[agn] = np.loadtxt(fp_clouds[agn])[0:1000, :]
        # normalize the weights to account for different numbers of clouds
        clouds[agn][:, 6] = clouds[agn][:, 6] * 2

    Nagn = len(agn_list)

    # Set the figure dimensions
    nrows = int(ceil(float(Nagn)/2.))
    ncols = 2
    figheight = 8.0
    pad_inches = [0.5,0.2,0.1,0.5,0.65,0.5]

    Figure = bf.CustomFigure()
    Figure.fixedHeight(figheight,pad_inches,(nrows,ncols),spaxisratio=0.5)

    axes_size = [16, 16, 16, 16, 10, 10, 10]

    for i in range(Nagn):
        agn = agn_list[i]
        print agn
        print "\n"

        ax = Figure.newSubplot(position=pos[agn])
        ax = Figure.makeSubpanels(
            ax,
            dims = (2, 1),
            pad = (0., 0.)
        )

        # Add AGN label
        lstart, bstart, spwidth, spheight = ax[0,0].get_position().bounds
        #Figure.fig.figtext(
        #    lstart + 0.06 * spwidth,
        #    bstart  + 0.1 * spheight,
        #    post[agn].texname)

        x = clouds[agn][:, 0] / (299792458.0 * 86400)
        y = clouds[agn][:, 1] / (299792458.0 * 86400)
        z = clouds[agn][:, 2] / (299792458.0 * 86400)
        vx = -clouds[agn][:, 3] / 100000
        maxvx = max([abs(val) for val in vx])
        w = clouds[agn][:, 6]
        w = (w * 10000.) ** 2.
        vx = vx / (2. * maxvx)
        vx = vx + 0.5
        _colors = [0] * len(vx)
        for j in range(len(vx)):
            _colors[j] = [vx[j], 0, 1. - vx[j], 1]

        # All the same color
        in_out = np.median(post[agn].posteriors['inflowoutflow'])
        _colors = [1. - (in_out + 1.) / 2., 0, (in_out + 1.) / 2., 1.]

        # w = (w/median(w)*10)**2.
        # Adjust for different figure sizes
        w *= spheight / 1. * 0.5

        # Edge-on view
        min_axis = -axes_size[i]
        max_axis = axes_size[i]
        tick1 = -int(max_axis * 0.8)
        tick2 = tick1 / 2.
        tick3 = int(max_axis * 0.8)
        tick4 = tick3 / 2.
        ax[0,0].set_xlim(min_axis,max_axis)
        ax[0,0].set_ylim(min_axis,max_axis)
        setp(ax, ylim=(min_axis, max_axis), xlim=(min_axis, max_axis))
        yticks((tick1, tick2, 0, tick3, tick4))
        scatter(x, z, marker='o', s=w, alpha=0.1, color=_colors)
        ylabel("$\\rm z$ $\\rm (light$ $\\rm days)$", labelpad=0.5)
        #if hpad_bool[i] == 1 or i == Nagn - 1:
        #    xticks((tick1, tick2, 0, tick3, tick4), **_tick_fontparams)
        #    xlabel("$\\rm y$ $\\rm (light$ $\\rm days)$", **_axis_fontparams)
        #else:
        #    xticks((tick1, tick2, 0, tick3, tick4), (), **_tick_fontparams)

        # Face-on view
        setp(ax, ylim=(min_axis, max_axis), xlim=(min_axis, max_axis))
        xticks((tick1, tick2, 0, tick3, tick4))
        yticks((tick1, tick2, 0, tick3, tick4), ())
        scatter(y, z, marker='o', s=w, alpha=0.1, color=_colors)

        #if hpad_bool[i] == 1 or i == Nagn - 1:
        #    xticks((tick1, tick2, 0, tick3, tick4))
        #    xlabel("$\\rm y$ $\\rm (light$ $\\rm days)$")
        #else:
        #    xticks((tick1, tick2, 0, tick3, tick4), ())


    #if _colors:
    #    cmap_colors = [(1, 0, 0), (0, 0, 1)]  # R -> G -> B
    #    n_bins = 100  # Discretizes the interpolation into bins
    #    cmap_name = 'my_colormap'
    #    # Create the colormap
    #    cm = LinearSegmentedColormap.from_list(
    #        cmap_name, cmap_colors, N=n_bins)
    #
    #    ax - new
    #    # Set the colormap and norm to correspond to the data for which
    #    # the colorbar will be used.
    #    # cmap = mpl.cm.cool
    #    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    #    cb1 = mpl.colorbar.ColorbarBase(ax_color, cmap=cm,
    #                                    norm=norm,
    #                                    orientation='horizontal'
    #                                    )
    #    cb1.set_ticks([-1, -0.5, 0, 0.5, 1])
    #    cb1.set_ticklabels(["${\\rm Inflow}$", "", "", "", "${\\rm Outflow}$"])
    #    cb1.ax.tick_params(**_cbar_fontparams)

    #if savename != '':
    #    print "Saving..."
    #    plt.savefig(savename)
    plt.show()
    clf()


def animateGeo(fp_clouds, savename):
    geo = Geometry(fp_clouds)
    geo.runAnimation(savename)


def main():
    pass


if __name__ == '__main__':
    main()