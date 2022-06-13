from pylab import *
import numpy as np
import sys
import matplotlib.pyplot as plt
from cappuccino.utils import parInfo
from cappuccino.utils import constants as const

def comparison(runs, cut=0.0, params_list=[], truth={}, colors=['dodgerblue','red','green'], **kwargs):
    """
    Runs a modified version of the original convergence.py code. This will
    default to the version of CARAMEL on the master branch. In the event that 
    a different version is being used, the 'params_list' kwarg MUST be supplied.
    If the non-master-branch parameters do not already have entries in the
    param_names, param_scales, and logify dictionaries, these must be added.

    :param savename: Path to the posterior sample file, defaults to 'posterior_sample'
    :type savename: str

    :return: None
    """
    rcParams['text.usetex'] = True
    rcParams['text.latex.unicode'] = True

    all_possible_params = [
        'Mbh', 'mu', 'Beta', 'F',
        'Thetao', 'Thetai',
        'Kappa', 'Xi', 'Gamma',
        'Cadd', 'Cmult', 'epsilon_zero', 'alpha', 'Fellip', 'Fflow',
        'angular_sd_orbiting', 'radial_sd_orbiting', 'angular_sd_flowing', 'radial_sd_flowing',
        'Sigmaturb', 'Thetae', 'narrow_line_flux', 'narrow_line_flux_nii', 'Blurring',
        'Rmean', 'Rmedian', 'Taumean',  'Taumedian',  'NarrowLineCenter'
    ]

    params_list = []
    for par in all_possible_params:
        for i in range(len(runs)):
            if par in runs[i].CARAMEL_params_list:
                params_list.append(par)
                break
    #for i in range(len(runs)):
    #    params_list += runs[i].CARAMEL_params_list
    #params_list = list(set(params_list))
    param_map = []
    for i in range(len(runs)):
        param_map.append({})
        for param in params_list:
            if param in runs[i].CARAMEL_params_list:
                param_map[i][param] = runs[i].CARAMEL_params_list.index(param)

    # Set the params, labels, and units:
    max_param_num = len(params_list)  # max index of params to plot
    rows = 5  # number of rows in subplots
    cols = 6  # number of columns in subplots
    num_boxes = 24  # number of boxes (bins) used in posterior histograms

    param_names = parInfo.paramNamesCARAMELOut()
    param_scales = parInfo.paramScalesCARAMELOut()
    logify = parInfo.logifyCARAMELOut()

    if type(cut) != list:
        cut = [cut] * len(runs)

    samples_list = []
    sample_info_list = []
    for i in range(len(runs)):
        print "Loading samples..."
        samples_list.append(np.load(runs[i].fp_run + 'sample.npy'))
        sample_info_list.append(np.load(runs[i].fp_run + 'sample_info.npy'))

    # convergence plots:
    plt.figure(figsize=(30, 15))
    for j in range(len(samples_list)):
        start = int(cut[j] * samples_list[j].shape[0])
        samples = samples_list[j][start:, :]
        sample_info = sample_info_list[j][start:, :]

        for i in xrange(0, max_param_num):
            par = params_list[i]
            plt.subplot(rows, cols, i + 1)
            if par in param_map[j].keys():
                param = samples[:, param_map[j][par]] * param_scales[par]
                if logify[par]:
                    param = np.log10(param)
                    paramname = "$\\log_{10}($%s$)$" % (param_names[par])
                else:
                    paramname = param_names[par]
                to_plot = []
                for level in range(100):
                    tmp = param[sample_info[:,0]==level]
                    med,hi,lo = np.percentile(tmp, [50,50+34,50-34])
                    to_plot.append([med,hi,lo])
                to_plot = np.array(to_plot)
                #plt.plot(sample_info[:, 0], param, '.b', markersize=3)
                plt.plot(range(100), to_plot[:,0], ls='solid', color=colors[j])
                plt.plot(range(100), to_plot[:,1], ls='dotted', color=colors[j])
                plt.plot(range(100), to_plot[:,2], ls='dotted', color=colors[j])
                plt.fill_between(range(100), to_plot[:,2], to_plot[:,1], color=colors[j], alpha=0.5)
                if par in truth.keys():
                    if logify[par]:
                        plt.axhline(np.log10(truth[par]), color='red')
                    else:
                        plt.axhline(truth[par], color='red')
                plt.ylabel(paramname)
    plt.tight_layout()
    plt.show()
