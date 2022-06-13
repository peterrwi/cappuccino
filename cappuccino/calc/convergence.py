from pylab import *
import numpy as np
import sys
import matplotlib.pyplot as plt
from cappuccino.utils import parInfo
from cappuccino.utils import constants as const

def convergence(cut, cut2=1.0, savename='posterior_sample', params_list=[], truth={}, **kwargs):
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

    # Load the data:
    try:
        samples = atleast_2d(load("sample.npy"))  # load all the samples
        sample_info = atleast_2d(load('sample_info.npy'))
        posterior_samples = atleast_2d(
            load(savename+'.npy'))  # load the posterior samples
    except:
        samples = atleast_2d(loadtxt("sample.txt"))  # load all the samples
        sample_info = atleast_2d(loadtxt('sample_info.txt'))
        posterior_samples = atleast_2d(
            loadtxt(savename+'.txt'))  # load the posterior samples
    
    # This is useful in case you copy files during a run and a new line saves to
    # sample_info.txt after you've copied sample.txt
    if samples.shape[0] != sample_info.shape[0]:
        print('# Size mismatch. Truncating...')
        lowest = np.min([samples.shape[0], sample_info.shape[0]])
        samples = samples[0:lowest, :]
        sample_info = sample_info[0:lowest, :]
    
    start = int(cut * samples.shape[0])
    end = int(cut2 * samples.shape[0])
    half = int((cut2 - cut) * samples.shape[0] / 2)

    samples = samples[start:end, :]
    sample_info = sample_info[start:end, :]

    # Convergence plot for continuum hyperparameters
    names = [r'$\mu$', r'$\log_{10}(\sigma)$', r'$\log_{10}(L)$',
             r'$\alpha$', r'$f_{\rm bad}$', r'$\log_{10}(K_1)$',
             r'$\log_{10}(K_2)$']
    take_log = [False for i in xrange(0, 7)]
    take_log[1] = True
    take_log[2] = True
    take_log[5] = True
    take_log[6] = True
    plt.figure(figsize=(14, 8))
    for i in xrange(0, 7):
        plt.subplot(3, 3, i + 1)
        params = samples[:, -1007 + i:-1000]
        if take_log[i]:
            params = log10(params)
        plt.plot(sample_info[:, 0], params[:, 0], 'b.', markersize=3)
        plt.ylabel(names[i])
    plt.subplot(3, 3, 2)
    plt.title('Continuum Hyperparameters')
    plt.tight_layout()
    plt.savefig("hyperparameters.png")
    plt.show()

    if params_list == []:
        # Default params for the master branch
        params_list = [
            'Mbh', 'mu', 'Beta', 'F',
            'Thetao', 'Thetai',
            'Kappa', 'Xi', 'Gamma',
            'Cadd', 'Cmult', 'Fellip', 'Fflow',
            'angular_sd_orbiting', 'radial_sd_orbiting', 'angular_sd_flowing', 'radial_sd_flowing',
            'Sigmaturb', 'Thetae', 'narrow_line_flux', 'Blurring',
            'Rmean', 'Rmedian', 'Taumean',  'Taumedian',  'NarrowLineCenter', 'Rmax'
        ]

    param_names = parInfo.paramNamesCARAMELOut()
    param_scales = parInfo.paramScalesCARAMELOut()
    logify = parInfo.logifyCARAMELOut()

    # Set the params, labels, and units:
    max_param_num = len(params_list)  # max index of params to plot
    rows = 5  # number of rows in subplots
    cols = 6  # number of columns in subplots
    num_boxes = 24  # number of boxes (bins) used in posterior histograms

    # convergence plots:
    plt.figure(figsize=(30, 15))
    for i in xrange(0, max_param_num):
        par = params_list[i]
        plt.subplot(rows, cols, i + 1)
        param = samples[:, i] * param_scales[par]
        if logify[par]:
            param = np.log10(param)
            paramname = "$\\log_{10}($%s$)$" % (param_names[par])
        else:
            paramname = param_names[par]
        plt.plot(sample_info[:half, 0], param[:half], '.b', markersize=3)
        plt.plot(sample_info[half:, 0], param[half:], '.r', markersize=3)
        if par in truth.keys():
            if logify[par]:
                plt.axhline(np.log10(truth[par]), color='red')
            else:
                plt.axhline(truth[par], color='red')
        plt.ylabel(paramname)
    plt.tight_layout()
    plt.savefig("convergence.png")
    plt.show()

    Mbh = log10(posterior_samples[:, 0] / 1.989E30)
    print('log10(Mbh/Msun) = {a} += {b}'.format(a=Mbh.mean(), b=Mbh.std()))

    opening = posterior_samples[:, 4] * 180. / np.pi
    print('Opening angle (deg) = {a} += {b}'.format(a=opening.mean(),
                                                    b=opening.std()))

    inc = posterior_samples[:, 5] * 180. / np.pi
    print('Inclination angle (deg) = {a} += {b}'.format(a=inc.mean(),
                                                        b=inc.std()))

    rmean = posterior_samples[:, 21] / (299792458. * 86400.)
    print('Mean radius (light days) = {a} += {b}'.format(a=rmean.mean(),
                                                         b=rmean.std()))



    # posterior plots:
    plt.figure(figsize=(30, 15))
    for i in xrange(0, max_param_num):
        par = params_list[i]
        subplot(rows, cols, i + 1)
        param = posterior_samples[:, i] * param_scales[par]
        if logify[par]:
            param = np.log10(param)
            paramname = "$\\log_{10}($%s$)$" % (param_names[par])
        else:
            paramname = param_names[par]

        plt.hist(param, num_boxes)
        #axvline(truth[i], color='r', linewidth=2)
        plt.xlabel(paramname)
    plt.tight_layout()
    plt.savefig("posterior.png")
    plt.show()


def main():
    agn = sys.argv[1]
    template = sys.argv[2]


if __name__ == "__main__":
    main()