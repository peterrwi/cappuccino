import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True

class CustomFigure:
    """
    Custom figure class based on matplotlib. This was created to help make more
    precise figures with precise subplot sizes. Matplotlib would add padding 
    seemingly randomly and dependent on the things such as the axis tick label
    locations and font sizes.
    """
    def __init__(self):
        pass

    def fixedDimensions(self,figsize,pad_inches):
        """
        Creates an empty figure object with fixed dimensions.

        :param figsize: Size of the figure in inches (width, height)
        :type figsize: tuple, length 2

        :param pad_inches: Padding, in inches, around the figure border and
            between subplot panels. format=[left,right,top,bottom,column,row]
        :type pad_inches: tuple, length 6

        :return: None
        """
        self.figsize = figsize
        self.pad_inches = pad_inches  # left, right, top, bottom, column, row
        self.calcDimensions()
        self.fig = plt.figure(figsize=figsize)

    def fixedWidth(self,figwidth,pad_inches,dim,spaxisratio=1.):
        """
        Creates a figure object with panels, where the overall figure has a
        fixed width.

        :param figwidth: Width of the figure, in inches
        :type figwidth: float

        :param pad_inches: Padding, in inches, around the figure border and
            between subplot panels. format=[left,right,top,bottom,column,row]
        :type pad_inches: tuple, length 6

        :param dim: Dimensions of the figure, (nrows, ncols)
        :type dim: tuple, length 2

        :param spaxisratio: Axis ratio (height/width) of the subplots, defaults to 1.
        :type spaxisratio: float

        :return: None
        """
        nrows,ncols = dim
        spwidthin = (figwidth - pad_inches[0] - pad_inches[1] - pad_inches[4] \
                * (ncols - 1.)) / ncols
        spheightin = spwidthin * spaxisratio
        figheight = nrows * spheightin + (nrows - 1.) * pad_inches[5] + \
                    pad_inches[2] + pad_inches[3]
        self.figsize = (figwidth,figheight)
        self.pad_inches = pad_inches
        self.calcDimensions()
        self.fig = plt.figure(figsize=self.figsize)
        self.setRowsCols((nrows,ncols))

    def fixedHeight(self,figheight,pad_inches,dim,spaxisratio=1.):
        """
        Creates a figure object with panels, where the overall figure has a
        fixed width.

        :param figheight: Height of the figure, in inches
        :type figheight: float

        :param pad_inches: Padding, in inches, around the figure border and
            between subplot panels. format=[left,right,top,bottom,column,row]
        :type pad_inches: tuple, length 6

        :param dim: Dimensions of the figure, (nrows, ncols)
        :type dim: tuple, length 2

        :param spaxisratio: Axis ratio (height/width) of the subplots, defaults to 1.
        :type spaxisratio: float

        :return: None
        """
        spheightin = (figheight - pad_inches[2] - pad_inches[3] - \
            pad_inches[5] * (nrows - 1.)) / nrows
        spwidthin = spheightin / spaxisratio
        figwidth = ncols * spwidthin + (ncols - 1.) * pad_inches[4] + \
            pad_inches[0] + pad_inches[1]
        self.figsize = (figwidth,figheight)
        self.pad_inches = pad_inches
        self.calcDimensions()
        self.fig = plt.figure(figsize=self.figsize)
        self.setRowsCols((nrows,ncols))

    def calcDimensions(self):
        """
        Converts the inch padding measurements to fractional measurements

        :return: None
        """
        lpadin, rpadin, tpadin, bpadin, wpadin, hpadin = self.pad_inches
        figwidth = self.figsize[0]
        figheight = self.figsize[1]
        lpad = lpadin / figwidth
        rpad = rpadin / figwidth
        tpad = tpadin / figheight
        bpad = bpadin / figheight
        wpad = wpadin / figwidth
        hpad = hpadin / figheight
        self.pads = [lpad,rpad,tpad,bpad,wpad,hpad]

    def setRowsCols(self,dim):
        """
        Sets the number of rows and columns and computes the required dimensions

        :return: None
        """
        self.nrows = dim[0]
        self.ncols = dim[1]
        self.calcSubplotDimensions()

    def calcSubplotDimensions(self):
        """
        Computes the dimensions of the subplots based on the number of rows and
        columns and the padding size.

        :return: None
        """
        spwidth = (1. - self.pads[0] - self.pads[1] - self.pads[4] * (
                    self.ncols - 1.)) / self.ncols
        spheight = (1. - self.pads[2] - self.pads[3] - self.pads[5] * (
                    self.nrows - 1.)) / self.nrows
        self.spsize = (spwidth,spheight)

    def newSubplot(self,position,**kwargs):
        """
        Adds a new subplot at given position = col,row

        :param position: (col, row) position of the subplot
        :type position: tuple, length 2

        :return: ax
        """
        col, row = position
        ax = self.fig.add_axes((
            self.pads[0] + col * (self.spsize[0] + self.pads[4]),
            1. - self.pads[2] - self.spsize[1] - row * (self.spsize[1] + self.pads[5]),
            self.spsize[0],
            self.spsize[1]
        ),
        **kwargs)

        return ax

    def makeSubpanels(self,ax,dims,pad=(0,0),**kwargs):
        """
        Adds subpanels to an existing axes object.

        :param ax: Axes object to split into subpanels
        :type ax: matplotlib.axes object

        :param dims: Dimensions = (ncols, nrows)
        :type dims: tuple, length 2

        :param pad: Padding (wpad, hpad) between subpanels
        :type pad: tuple, length 2

        :return: np.array(ax)
        """
        ncols, nrows = dims
        lstart, bstart, spwidth, spheight = ax.get_position().bounds

        wpad, hpad = pad
        if not isinstance(wpad, list):
            wpad = [wpad] * (ncols - 1)
        if not isinstance(hpad, list):
            hpad = [hpad] * (nrows - 1)
        #wpad, hpad = np.array(wpad) * spwidth, np.array(hpad) * spheight

        panelwidth = (spwidth - np.sum(wpad)) / ncols
        panelheight = (spheight - np.sum(hpad)) / nrows

        ax = np.zeros((ncols, nrows)).tolist()
        for i in range(ncols):
            for j in range(nrows):
                ax[i][j] = plt.axes((
                    lstart + sum(wpad[k] + panelwidth for k in range(i)),
                    bstart + spheight - panelheight - sum(hpad[k] + panelheight
                        for k in range(j)),
                    panelwidth,
                    panelheight),
                    **kwargs)

        return np.array(ax)

    def makeSubpanelsUneven(self,ax,dims,pad=(0,0),colwidth=1,rowheight=1,**kwargs):
        """
        Adds subpanels to an existing axes object, but with uneven spacing
        between each panel and uneven sizes for each panel.

        :param ax: Axes object to split into subpanels
        :type ax: matplotlib.axes object

        :param dims: Dimensions = (ncols, nrows)
        :type dims: tuple, length 2

        :param pad: Padding (wpad, hpad) between subpanels. Each entry of wpad
            and hpad can be a list, defining the spacing between the ncols and
            nrows panels.
        :type pad: tuple, length 2

        :param colwidth: Width of the subpanels. Can be a float (equal widths)
            or a list of floats (for different widths)
        :type colwidth: float or list of floats (length ncols)

        :param rowheight: Height of the subpanels. Can be a float (equal heights)
            or a list of floats (for different heights)
        :type rowheight: float or list of floats (length nrows)

        :return: np.array(ax)
        """
        ncols, nrows = dims
        lstart, bstart, spwidth, spheight = ax.get_position().bounds

        wpad, hpad = pad
        if not isinstance(wpad, list):
            wpad = [wpad] * (ncols - 1)
        if not isinstance(hpad, list):
            hpad = [hpad] * (nrows - 1)

        if not isinstance(rowheight, list):
            rowheight = [rowheight] * (nrows)
        if not isinstance(colwidth, list):
            colwidth = [colwidth] * (ncols)
        rowheight = np.array(rowheight)
        colwidth = np.array(colwidth)

        rowheight = rowheight / sum(rowheight) * (spheight - np.sum(hpad))
        colwidth = colwidth / sum(colwidth) * (spwidth - np.sum(wpad))

        ax = np.zeros((ncols, nrows)).tolist()
        for i in range(ncols):
            for j in range(nrows):
                ax[i][j] = plt.axes((
                    lstart + sum(wpad[k] + colwidth[k] for k in range(i)),
                    bstart + spheight - rowheight[j] - sum(hpad[k] + rowheight[k]
                                                          for k in range(j)),
                    colwidth[i],
                    rowheight[j]),
                    **kwargs)

        return np.array(ax)

    def cornerPlot(self, nvar, spwidth=2, label_all_axes=False):
        """
        Empty corner plot with nvar variables.

        :param nvar: Number of variables
        :type nvar: int

        :param spwidth: Width of the subplots
        :type spwidth: float

        :param label_all_axes: Add labels to all of the subplots, defaults to False
        :type label_all_axes: bool

        :return: ax
        """

        # Plotting values
        nrows,ncols = nvar-1, nvar-1
        figwidth = spwidth * (float(ncols))
        
        _lpadin = 0.35 * spwidth
        _rpadin = 0.025 * spwidth
        _bpadin = 0.25 * spwidth
        _tpadin = 0.025 * spwidth

        if label_all_axes:
            _wpadin, _hpadin = 0.2 * spwidth, 0.2 * spwidth
        else:
            _wpadin, _hpadin = 0.1 * spwidth, 0.1 * spwidth

        _spwidthin = (figwidth - _lpadin - _rpadin - _wpadin * (
                    ncols - 1.)) / ncols
        _spheightin = _spwidthin
        figheight = nrows * _spheightin + (
                    nrows - 1.) * _hpadin + _tpadin + _bpadin

        figsize=(figwidth, figheight)
        pad_inches = [_lpadin,_rpadin,_tpadin,_bpadin,_wpadin,_hpadin]

        self.fixedDimensions(figsize, pad_inches)
        self.setRowsCols((nrows, ncols))

        ax = np.ndarray.tolist(np.zeros((nrows,ncols)))
        for i in range(nrows):
            for j in range(i):
                ax[i][j] = self.newSubplot(position=(j,i))
        return ax