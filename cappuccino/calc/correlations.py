import numpy as np
import pidly
import pandas as pd


def runLinMix(linmixin, savename, miniter=5000, maxiter=10000, K=3, idlpath=None):
    """
    Performs a linear regression accounting for errors in x and y as well xy covariance using the
    LINMIX_ERR IDL routine: https://ui.adsabs.harvard.edu/abs/2007ApJ...665.1489K/abstract

    :param linmixin: Data to pass to LINMIX_ERR, structured as follows:
        Row 1: x values
        Row 2: y values
        Row 3: x uncertainties
        Row 4: y uncertainties
        Row 5: xy covariance
    :type linmixin: 2d list

    :param savename: Filename to save the linmix output, defaults to no save.
    :type savename: str

    :param miniter: Minimum number of iterations performed by Gibbs sampler, defaults to 5000
    :type miniter: int

    :param maxiter: Maximum number of iterations performed by Gibbs sampler, defaults to 10000
    :type maxiter: int

    :param K: Number of Gaussians to use in mixture modeling, defaults to 3.
    :type K: int

    :param idlpath: Path to IDL executable
    :type idlpath: str

    :return: linmix chain with keys 'alpha', 'beta', 'sigsqr', 'corr'
    :rtype: dict
    """
    print "Loading IDL..."
    try:
        idl = pidly.IDL(idlpath)
        print "Succesfully loaded."
    except ValueError:
        print "Must provide an IDL path. Exiting."
        return
    except:
        print "Bad IDL path provided. Exiting"
        return

    # Set the linmix_err input parameters
    idl.x = linmixin[0].tolist()
    idl.y = linmixin[1].tolist()
    idl.xsig = linmixin[2].tolist()
    idl.ysig = linmixin[3].tolist()
    idl.xycov = linmixin[4].tolist()
    idl.miniter = miniter  # LINMIX_ERR default: 5000
    idl.maxiter = maxiter  # LINMIX_ERR default: 100000
    idl.k = K
    
    # Perform the fit
    print "Running LinMix"
    idl('LINMIX_ERR, x, y, post, XSIG=xsig, YSIG=ysig, XYCOV=xycov, \
        miniter=miniter, maxiter=maxiter')
    
    # Choose which values you'd like to save from the LINMIX_ERR output.
    # Available keys are ['mu0', 'usqr', 'ximean', 'sigsqr', 'wsqr', 'mu',
    #  'beta', 'corr', 'xisig', 'alpha', 'pi', 'tausqr']
    keys = ['alpha','beta','sigsqr','corr']
    attributes = ['[post.%s]' % key for key in keys]
    idlcommand = 'theta = [%s]' % ','.join(attributes)
    idl(idlcommand)

    lmchain = idl.theta.T

    if savename != "":
        print "Saving results..."
        np.save(savename,lmchain)

    return lmchain


def computeCovariance(posteriors, varx, vary, names=[], campaigns=[], x_logify=False, y_logify=False):
    """
    Takes in a list of posteriors and given two variables, varx and vary, computes the x, y, xsig,
    ysig, and xycov values needed to run LINMIX_ERR.

    :param posteriors: List of posteriors
    :type posteriors: list

    :param varx: x variable name
    :type varx: str

    :param vary: y variable name
    :type vary: str

    :param names: List of AGN names corresponding to posteriors
    :type names: list

    :param campaigns: List of observing campaigns corresponding to posteriors
    :type campaigns: list

    :param x_logify: Whether to compute in the log of x, defaults to False
    :type x_logify: bool

    :param y_logify: Whether to compute in the log of y, defaults to False
    :type y_logify: bool 

    :return: Pandas DataFrame
    """
    print "Computing covariance matrix for %s vs. %s" % (vary, varx)

    Nagn = len(posteriors)
    columnnames = ['name','campaign','x','y','xsig','ysig','xycov']
    df = pd.DataFrame(data={key: [None]*Nagn for key in columnnames}) 

    if len(names) == 0:
        names = [None] * Nagn
    if len(campaigns) == 0:
        campaigns = [None] * Nagn
    for i in range(len(posteriors)):
        df['name'][i] = names[i]
        df['campaign'][i] = campaigns[i]
        post['vary'][i]
        post = posteriors[i]
        x_posterior = post[varx] * 1.
        y_posterior = post[vary] * 1.

        if x_logify:
            x_posterior = np.log10(x_posterior)
        if y_logify:
            y_posterior = np.log10(y_posterior)

        # Calculate the covariance matrix
        cov = np.cov(x_posterior, y_posterior)

        # If there are any NaNs in the covariance matrix, ignore point
        if 1 in [np.isnan(cov[n, m]) for n in [0, 1] for m in [0, 1]]:
            print "Covariance matrix for %s contains NaNs." % (sname)
            if sum(np.isnan(x_posterior[i]) for i in range(len(x_posterior))) > 0:
                print "    NaNs in x_posterior: %s" % (varx)
            if sum(np.isnan(y_posterior[i]) for i in range(len(y_posterior))) > 0:
                print "    NaNs in y_posterior: %s" % (vary)
            print "Ignoring this data point."
        else:
            if cov[0, 1] != cov[1, 0]:
                print "WARNING: Covariance matrix for %s NOT symmetric." % (name)
            df['x'][i] = np.median(x_posterior)
            df['y'][i] = np.median(y_posterior)
            df['xsig'][i] = np.sqrt(cov[0,0])
            df['ysig'][i] = np.sqrt(cov[1,1])
            df['xycov'][i] = cov[0,1]

    return df


def runCorrelations(posteriors, varx, vary, names=[], campaigns=[], x_logify=False, y_logify=False,
    savename='', miniter=5000, maxiter=10000, K=3, idlpath=None):
    """
    Strings together computeCovariance and runLinMix.
    """
    # Compute the covariance matrices
    df = computeCovariance(posteriors, varx, vary, names, campaigns, x_logify, y_logify)
    
    # Set up the LINMIX_ERR input and call runLinMix
    df = df.dropna(subset=['x'])
    linmixin = df.loc[:,['x','y','xsig','ysig','xycov']].values.T
    lmchain = runLinMix(
        linmixin=linmixin,
        savename=savename,
        miniter=miniter,
        maxiter=maxiter,
        K=K,
        idlpath=idlpath
    )
    return lmchain


def saveCorrelations(post, varx, vary, x_logify=False, y_logify=False,
                     usecovariance=False, fp_linmix='', savesuffix='',
                     idlpath=None,**kwargs):

    x,y,cov,campaign,objects = posteriors.extractConfIntsAndCov(post,varx,vary,x_logify,y_logify)

    if usecovariance:
        _savenamecov = 'wcov'
    else:
        _savenamecov = 'nocov'
    fp_linmixout = "%s%s_vs_%s_%s_%s.npy" % (fp_linmix, vary, varx, _savenamecov, savesuffix)

    xsig, ysig, xycov = np.sqrt(cov[:,0,0]), np.sqrt(cov[:,1,1]), cov[:,0,1]
    runLinMix([x[:,0],y[:,0],xsig,ysig,xycov],fp_linmixout,idlpath=idlpath,
              **kwargs)