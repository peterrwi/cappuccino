import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cappuccino.utils import parInfo
import pandas as pd
from scipy.stats import gaussian_kde
import scipy.optimize as so

import bokeh.io as bkio
import bokeh.plotting as bkp
import bokeh.layouts as bkl
from bokeh.models import ColumnDataSource, CustomJS, Slider, Button, Div, Span

from bokeh.server.server import Server
reload(parInfo)

def modify_doc(doc):
    panel_width=150
    logify=False
    label_all_axes=False
    if isinstance(logify,bool):
        logify = [logify]*len(params)
    data = {}
    for i in range(len(params)):
        if logify[i]:
            data[params[i]] = np.log10(post[params[i]])
        else:
            data[params[i]] = post[params[i]]

    def callback(attr, old, new):
        indices = source.selected.indices
        for par in params:
            old_data = src_hist[par].data
            edges = np.concatenate([old_data["left"], [old_data["right"][-1]]])
            hist, edges = np.histogram(data[par][indices], density=False, bins=edges)
            new_data = {
                "top": hist,
                "left": edges[:-1],
                "right": edges[1:]
            }

            src_hist[par].data = new_data
            
        medians = {par: [np.median(data[par][indices])] for par in params}
        src_medians.data = medians

    source = ColumnDataSource(data)
    src_hist, src_medians = {}, {}
    for par in params:
        hist, edges = np.histogram(data[par], density=False, bins=20)
        hist_df = pd.DataFrame({
            "top": 0.*hist,
            "left": edges[:-1],
            "right": edges[1:]
        })
        src_hist[par] = ColumnDataSource(hist_df)
    medians = {par: [np.median(data[par])] for par in params}
    src_medians = ColumnDataSource(data = medians)
    
    TOOLS = 'lasso_select, reset'
    Nparam = len(params)
    ax = np.full((Nparam,Nparam), None).tolist()
    #toggle_botton = Button(label='c')
    for row in range(Nparam):
        for col in range(row+1):
            # i is rows (top to bottom), j is columns (left to right)
            vline = Span(location=medians[params[col]][0], dimension='height', line_color='black', line_width=1.5, line_dash='dashed')
            hline = Span(location=medians[params[row]][0], dimension='width', line_color='black', line_width=1.5, line_dash='dashed')
            
            #vline = Span(location=params[col], source=src_medians, dimension='height', line_color='orange', line_width=1.5, line_dash='dashed')
            #hline = Span(location=row, dimension='width', line_color='orange', line_width=1.5, line_dash='dashed')
            if col==0:
                width = int(panel_width*1.25)
            else:
                width = panel_width
            if row==Nparam-1:
                height = int(panel_width*1.25)
            else:
                height = panel_width
            if row == col:
                hist, edges = np.histogram(data[params[col]], density=False, bins=20)
                ax[row][col] = figure(width=width, height=height, tools=TOOLS)
                ax[row][col].quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                       fill_color="navy", line_color="white", alpha=0.5)
                ax[row][col].add_layout(vline)
                
                # Histogram of selected points
                ax[row][col].quad(
                    bottom = 0, top = "top",left = "left", right = "right", source = src_hist[params[col]],
                    fill_color = 'orange', line_color = "white", fill_alpha = 0.5, line_width=0.1)
            else:
                ax[row][col] = figure(width=width, height=height, tools=TOOLS)
                ax[row][col].scatter(
                    params[col],params[row],source=source, size=5,
                    fill_color='blue',line_color='navy', fill_alpha=0.4, line_alpha=0.4,
                    nonselection_fill_color='blue', nonselection_line_color='navy',nonselection_alpha=0.4, nonselection_line_alpha=0.4,
                    selection_fill_color='orange', selection_line_color='orange', selection_alpha=0.5,
                )
                ax[row][col].add_layout(vline)
                ax[row][col].add_layout(hline)
            if col == 0:
                ax[row][col].yaxis.axis_label = params[row]
            else:
                ax[row][col].yaxis.major_label_text_font_size = '0pt'
            if row == Nparam-1:
                ax[row][col].xaxis.axis_label = params[col]
            else:
                ax[row][col].xaxis.major_label_text_font_size = '0pt'
            if label_all_axes:
                ax[row][col].yaxis.axis_label = params[row]
                ax[row][col].xaxis.axis_label = params[col]
    ax_grid = gridplot(ax)
    
    source.selected.on_change('indices', callback)
    
    # add the layout to curdoc
    doc.add_root(ax_grid)

def runBokeh(params,post):
    # Setting num_procs here means we can't touch the IOLoop before now, we must
    # let Server handle that. If you need to explicitly handle IOLoops then you
    # will need to use the lower level BaseServer class.

    server = Server({'/': modify_doc}, num_procs=4)
    server.start()

    print 'Opening Bokeh application on http://localhost:5006/'

    #server.io_loop.add_callback(server.show, "/")
    #server.io_loop.start()

def cornerBokeh(post, params, panel_width=200, logify=False, outfile='', 
    label_all_axes=False):
    bkp.reset_output()
    if outfile != '':
        bkio.output_file(outfile)  # Render to static HTML, or 
    else:
        bkio.output_notebook()  # Render inline in a Jupyter Notebook

    if isinstance(logify,bool):
        logify = [logify]*len(params)
    data = {}
    for i in range(len(params)):
        if logify[i]:
            data[params[i]] = np.log10(post[params[i]])
        else:
            data[params[i]] = post[params[i]]

    medians = {par: [np.median(data[par])] for par in params}
    source = ColumnDataSource(data)
    src_medians = ColumnDataSource(data = medians)
    
    TOOLS = 'lasso_select, reset'
    Nparam = len(params)
    ax = np.full((Nparam,Nparam), None).tolist()
    toggle_botton = Button(label='c')
    source_hist, source_medians = {}, {}

    for row in range(Nparam):
        for col in range(row+1):
            # i is rows (top to bottom), j is columns (left to right)
            vline = Span(location=medians[params[col]][0], dimension='height', line_color='black', line_width=1.5, line_dash='dashed')
            hline = Span(location=medians[params[row]][0], dimension='width', line_color='black', line_width=1.5, line_dash='dashed')
            if row == col:
                hist, edges = np.histogram(data[params[col]], density=True, bins=20)
                hist_df = pd.DataFrame({
                    "top": 0.*hist,
                    "left": edges[:-1],
                    "right": edges[1:]
                })
                source_hist[params[col]] = ColumnDataSource(hist_df)
                ax[row][col] = bkp.figure(width=panel_width, height=panel_width, tools=TOOLS)
                ax[row][col].quad(
                        bottom = 0, top = "top",left = "left", right = "right", source = source_hist[params[col]],
                        fill_color = 'navy', line_color = "white", fill_alpha = 0.5)#, line_width=0.1)
                #ax[row][col].quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                #       fill_color="navy", line_color="white", alpha=0.5)
                ax[row][col].add_layout(vline)
            else:
                ax[row][col] = bkp.figure(width=panel_width, height=panel_width, tools=TOOLS)
                ax[row][col].scatter(params[col],params[row],source=source)
                ax[row][col].add_layout(vline)
                ax[row][col].add_layout(hline)
            if col == 0:
                ax[row][col].yaxis.axis_label = params[row]
            else:
                ax[row][col].yaxis.major_label_text_font_size = '0pt'
            if row == Nparam-1:
                ax[row][col].xaxis.axis_label = params[col]
            else:
                ax[row][col].xaxis.major_label_text_font_size = '0pt'
            if label_all_axes:
                ax[row][col].yaxis.axis_label = params[row]
                ax[row][col].xaxis.axis_label = params[col]
    callback_selected_code = """
        data = source.data;
        indices = source.selected.indices;
        for (var par in data) {
            old_data = src_hist[par].data
            data[key];
        }

    """
    callback_selected = CustomJS(args=dict(source=source, source_hist=source_hist), code=callback_selected_code)
    source.selected.js_on_change('indices', callback_selected)

    ax_grid = bkl.gridplot(ax)
    bkp.show(ax_grid)

def emptyCorner(Ndim, figwidth=0, label_loc='bottom', legend_loc=[]):
    # Plotting values
    if figwidth == 0:
        figwidth = 2. * Ndim - 1
    nrows, ncols = Ndim, Ndim

    _lpadin = 0.7
    _rpadin = 0.08
    _bpadin = 0.6

    if label_loc == 'all':
        _wpadin, _hpadin = 0.4, 0.4
    else:
        _wpadin, _hpadin = 0.2, 0.2
    if label_loc == 'top':
        _tpadin = 0.55
    else:
        _tpadin = 0.05

    _spwidthin = (figwidth - _lpadin - _rpadin - _wpadin * (ncols - 1.)) / ncols
    _spheightin = _spwidthin
    figheight = nrows * _spheightin + (nrows - 1.) * _hpadin + _tpadin + _bpadin

    _lpad = _lpadin / figwidth
    _rpad = _rpadin / figwidth
    _tpad = _tpadin / figheight
    _wpad = _wpadin / figwidth
    _hpad = _hpadin / figheight

    _spwidth = _spwidthin / figwidth
    _spheight = _spheightin / figheight

    plt.figure(figsize=(figheight,figwidth))
    ax = np.ndarray.tolist(np.zeros((Ndim,Ndim)))
    for row in range(Ndim):
        for col in range(row+1):
            ax[row][col] = plt.axes((
                _lpad + (col) * (_spwidth + _wpad),
                1. - _tpad - _spheight - (row - 0) * (_spheight + _hpad),
                _spwidth,
                _spheight
            ))
            ax[row][col].tick_params(direction='in')
    
    if legend_loc!= []:
        ax[legend_loc[0]][legend_loc[1]] = plt.axes((
            _lpad + (legend_loc[1]) * (_spwidth + _wpad),
            1. - _tpad - _spheight - (legend_loc[0] - 0) * (_spheight + _hpad),
            _spwidth,
            _spheight
        ))
    return ax

def findConfInt(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level

def corner(post, params, color='k', ax=None, logify={}, outfile='',
    label_loc='bottom', figwidth=0, truth={}, ax_lims={}, alpha=0.2, contour_sigma=[], contour_alpha=0.6, hist_sigma=[], bins=15):
    _label_params = {'fontsize': 6}

    if ax == None:
        ax = emptyCorner(len(params), figwidth=figwidth, label_loc=label_loc)

    if isinstance(logify, bool):
        logify = {par: logify for par in params}
    else:
        for par in params:
            if par not in logify.keys():
                logify[par] = False

    for par in params:
        if par in truth.keys():
            if logify[par]:
                truth[par] = np.log10(truth[par])

    if hist_sigma == None:
        hist_sigma=[]
    elif (hist_sigma == []) & (contour_sigma != []):
        hist_sigma = contour_sigma
    for row in range(len(params)):
        for col in range(row+1):
            varx = params[col]
            vary = params[row]
            x = post[varx]
            y = post[vary]
            if logify[varx]:
                x = np.log10(x)
            if logify[vary]:
                y = np.log10(y)

            if row == col:
                ax[row][col].hist(x,bins=bins, color=color, density=True, histtype='step')
                if len(hist_sigma) != 0:
                    nsigma_map = {1: 0.6827, 2: 0.9545, 3: 0.9973}
                    ls_map = {1: 'solid', 2: 'dashed', 3: 'dotted'}
                    ax[row][col].axvline(np.median(x), color=color, ls='dashed')
                    hist_sigma.sort()
                    hist_sigma = hist_sigma[::-1]
                    for i in range(len(hist_sigma)):
                        ax[row][col].axvline(
                            np.percentile(x, (50.-nsigma_map[hist_sigma[i]]*100./2.)),
                            ls=ls_map[hist_sigma[i]],
                            lw=0.5,
                            color=color
                        )
                        ax[row][col].axvline(
                            np.percentile(x, (50.+nsigma_map[hist_sigma[i]]*100./2.)),
                            ls=ls_map[hist_sigma[i]],
                            lw=0.5,
                            color=color
                        )
            else:
                #ax[row][col].plot(x,y,'k.',ms=2)
                ax[row][col].scatter(x,y,c=color,s=5,alpha=alpha)

                if len(contour_sigma) != 0:
                    nsigma_map = {nsigma: 1.0 - np.exp(-float(nsigma)**2/2) for nsigma in [1,2,3]}
                    ls_map = {1: 'solid', 2: 'dashed', 3: 'dotted'}

                    contour_sigma.sort()
                    contour_sigma = contour_sigma[::-1]

                    xmin,xmax = ax[row][col].get_xlim()
                    ymin,ymax = ax[row][col].get_ylim()
                    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    values = np.vstack([x, y])
                    kernel = gaussian_kde(values)
                    Z = np.reshape(kernel(positions).T, X.shape)
                    Z /= np.sum(Z)
                    levels = [so.brentq(findConfInt, 0., 1., args=(Z, nsigma_map[nsigma]))
                             for nsigma in contour_sigma]   
                    ax[row][col].contour(X, Y, Z, levels=levels, colors=color, alpha=contour_alpha, linestyles=[ls_map[sigma] for sigma in contour_sigma])

            # Set truth values
            if varx in truth.keys():
                ax[row][col].axvline(truth[varx], ls='dashed', color='red', lw=1)
            if (vary in truth.keys()) & (row != col):
                ax[row][col].axhline(truth[vary], ls='dashed', color='red', lw=1)

            # Setting the ticks and labels
            try:
                xtexname = parInfo.paramTexNames()[params[col]]
            except:
                xtexname = "${\\rm %s}$" % params[col]
            try:
                ytexname = parInfo.paramTexNames()[params[row]]
            except:
                ytexname = "${\\rm %s}$" % params[col]

            if logify[varx]:
                xtexname = "$\\log_{10}(%s)$" % xtexname[1:-1]
            if logify[vary]:
                ytexname = "$\\log_{10}(%s)$" % ytexname[1:-1]

            if col != 0:
                ax[row][col].set_yticklabels([])
            if row != len(params) - 1:
                ax[row][col].set_xticklabels([])

            if col == 0 and row != 0:
                ax[row][col].set_ylabel(ytexname)
            if row == len(params) - 1:
                ax[row][col].set_xlabel(xtexname)

            if label_loc == 'all':
                ax[row][col].set_ylabel(ytexname)
                ax[row][col].set_xlabel(xtexname)

            if label_loc == 'top':
                if row == col:
                    ax[row][col].set_xlabel(xtexname)    
                    ax[row][col].xaxis.set_label_position('top') 

    if ax_lims != None:
        col_lims = []
        for col in range(len(params)):
            varx = params[col]
            if varx in ax_lims.keys():
                col_lims.append(ax_lims[varx])
            else:
                col_lims.append(ax[col][col].get_xlim())

        for row in range(len(params)):
            for col in range(row+1):
                ax[row][col].set_xlim(*col_lims[col])
                if row == col:
                    ax[row][col].set_yticks([])
                else:
                    data = ax[row][col].collections[0].get_offsets()
                    ymin, ymax = min(data[:,1]) - (max(data[:,1]) - min(data[:,1])) * 0.1, \
                        max(data[:,1]) + (max(data[:,1]) - min(data[:,1])) * 0.1
                    ax[row][col].set_ylim(ymin, ymax)

    if outfile != '':
        plt.savefig(outfile)
    #plt.show()

def compare(post_list, params, label_loc='bottom', figwidth=0, color=['g','orange','blue'],  outfile='', alpha=[], label=[], legend_loc=[], **kwargs):
    if legend_loc == [] and label != []:
        legend_loc = [0,len(params)-1]
    ax = emptyCorner(len(params), figwidth=figwidth, label_loc=label_loc, legend_loc=legend_loc)
    if type(alpha) == float:
        alpha = [alpha]*len(post_list)
    elif alpha == []:
        alpha = [0.2]*len(post_list)

    for i in range(len(post_list)):
        corner(post_list[i], params=params, ax=ax, color=color[i], ax_lims=None, alpha=alpha[i], **kwargs)
    col_lims = []
    for col in range(len(params)):
        col_lims.append(ax[col][col].get_xlim())
    for row in range(len(params)):
        for col in range(row+1):
            ax[row][col].set_xlim(*col_lims[col])
            if row == col:
                ax[row][col].set_yticks([])
            else:
                data = ax[row][col].collections[0].get_offsets()
                ymin, ymax = min(data[:,1]) - (max(data[:,1]) - min(data[:,1])) * 0.1, \
                    max(data[:,1]) + (max(data[:,1]) - min(data[:,1])) * 0.1
                ax[row][col].set_ylim(*col_lims[row])

    if label != []:
        for i in range(len(post_list)):
            ax[legend_loc[0]][legend_loc[1]].scatter([], [], color=color[i], label=label[i])
        ax[legend_loc[0]][legend_loc[1]].legend(loc='upper right')#, fontsize=12)
        ax[legend_loc[0]][legend_loc[1]].axis('off')
    #ax[0][1].text(1,1,'test', ha='left', va='top')
    if outfile != '':
        plt.savefig(outfile)
