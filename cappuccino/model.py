# Written by Peter Williams
# 
# This code contains the main cappuccino classes, Model and Posterior, from
# which all CARAMEL postproccessing code can be called. The functions here are
# a mix of the functions provided by Anna Pancoast, modified for a cleaner
# workflow, and functions written by Peter Williams.
#
# Last modified Peter April 19, 2021

import numpy as np
import pandas as pd
import glob
import os
from calc import postprocess, convergence
from plotting import display, posteriors, corner, geometries
from utils import parInfo, util
import utils.constants as const
import json
reload(postprocess)
reload(convergence)
reload(corner)
reload(geometries)
reload(posteriors)

class Model:
    """
    This is the primary cappuccino class that will contain all of the information
    and results for a CARAMEL run. 

    :param fp_run: Path to the directory where the CARAMEL run results are stored.
    :type fp_run: str

    :param cut: Fraction of samples to cut from beginning of sample.txt, defaults to 0.0.
    :type cut: float

    :param temp: Statistical temperature for postprocessing, defaults to 1
    :type temp: int or float

    :param nhyperparams: Number of hyperparameters in the CARAMEL continuum model. This **must** be
        set properly for cappuccino to work. Defaults to the CARAMEL master branch.
    :type nhyperparams: int

    :param CARAMEL_params_list: List of parameters in the CARAMEL sample.txt output file. This
         **must** be set properly for cappuccino to work. Defaults to the CARAMEL master branch.
    :type CARAMEL_params_list: dictionary

    :param forward_extrap: Fraction of lightcurve extrapolated *into the future*. Must match the
        value in Constants.cpp. Defaults to 0.0
    :type forward_extrap: float

    :param backward_extrap: Fraction of lightcurve extrapolated *into the past*. Must match the
        value in Constants.cpp. Defaults to 0.0
    :type backward_extrap: dictionary

    :param redshift: Redshift of the AGN. Necessary for some parameter measurements. Defaults to 0.0.
    :type redshift: float

    :param runname: Name identifier for the CARAMEL run. Defaults to None.
    :type runname: str

    :param agnname: Name identifier for the AGN. Defaults to None.
    :type agnname: str

    :param posterior_sample: Posterior sample array. If included, will automatically create the 
        posterior property.
    :type posterior_sample: numpy array

    :param fp_posterior: Posterior sample filepath. If included, will automatically load and create
        the posterior property.
    :type fp_posterior: str

    :param load_posterior: If posterior_sample and fp_posterior weren't provided, will attempt to 
        find the posterior_sample file based on the standard file structure and load. Defaults to True.
    :type load_posterior: bool

    :param level_cut: If you'd like to specify exact levels rather than using the standard DNest postproccessing
        scripts. First value is the minimum level, second value is the maximum level.
    :type level_cut: list

    :return: None
    """
    def __init__(self, fp_run='', cut=0.0, temp=0., nhyperparams=7, CARAMEL_params_list=[],
        forward_extrap=0.0, backward_extrap=0.0, redshift=0., runname=None, agnname=None,
        posterior_sample=None, fp_posterior='', load_posterior=False, level_cut=[0,0], **kwargs):
        self.cwd = os.getcwd()
        
        self.redshift = redshift
        self.runname = runname
        self.agnname = agnname

        # Info for running postprocess
        self.fp_run = fp_run
        self.cut = cut
        self.temp = temp
        
        # Information about the CARAMEL run.
        self.nhyperparams = nhyperparams
        self.forward_extrap = forward_extrap
        self.backward_extrap = backward_extrap
        if CARAMEL_params_list == []:
            # Default params for the master branch
            self.CARAMEL_params_list = [
                'Mbh', 'mu', 'Beta', 'F',
                'Thetao', 'Thetai',
                'Kappa', 'Xi', 'Gamma',
                'Cadd', 'Cmult', 'Fellip', 'Fflow',
                'angular_sd_orbiting', 'radial_sd_orbiting', 'angular_sd_flowing', 'radial_sd_flowing',
                'Sigmaturb', 'Thetae', 'narrow_line_flux', 'Blurring',
                'Rmean', 'Rmedian', 'Taumean',  'Taumedian',  'NarrowLineCenter', 'Rmax'
            ]
        else:
            self.CARAMEL_params_list = CARAMEL_params_list
        self.nparams = len(self.CARAMEL_params_list)

        # Add any additional properties given in kwargs
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        if 'data_filenames' in kwargs.keys():
            self.loadData(data_filenames=kwargs['data_filenames'])
        else:
            try:
                self.loadData()
            except:
                pass

        # Setting up empty dictionaries and dataframes for the posteriors
        self.posterior = pd.DataFrame()
        self.params = {}
        self.confints = {}

        # Populate the posteriors if posterior_sample is provided
        if posterior_sample is not None:
            self.loadPosterior(posterior_sample=posterior_sample)
        elif fp_posterior != '':
            self.readPosteriorFile(fp_posterior, level_cut=level_cut)
        elif load_posterior:
            self.loadPosterior()
        if 'truth_json' in kwargs.keys():
            with open(kwargs['truth_json']) as f:
                tmp = json.load(f)
            name_map = parInfo.caramelNameToPythonName()
            truth = {}
            for key in tmp.keys():
                truth[name_map[key]] = tmp[key]
            self.truth = truth
        #    fn_posterior = "%sposterior_sample" % self.fp_run
        #    try:
        #        self.loadPosterior()
        #    except:
        #        raise
        #    #if os.path.exists(fn_posterior+'.npy') or os.path.exists(fn_posterior+'.txt'):
        #    #    self.loadPosterior()

    @classmethod
    def fromJSON(cls, fp_json, runname='', **kwargs):
        """
        Loads in the run information from a JSON file.

        :param fp_json: Path to the JSON file.
        :type fp_json: str

        :param runname: If there are multiple values in the JSON file, this is the key for the run
            you would like to load. If not provided, will assume only one run.
        :type runname: str
        
        :return: Model object
        """
        with open(fp_json) as f:
            run_info = json.load(f)
        if runname != '':
            run_info = run_info[runname]
        print "Creating model using the following:"
        for key, value in run_info.items():
            print "  %s: %s" % (key, value)
        run_info.update(kwargs)
        return Model(**run_info)

    def deepCopy(self):
        """
        Makes a deep copy of the Model object.

        :return: new Model object
        """
        import copy        
        return copy.deepcopy(self)

    def loadData(self, data_dir='', data_filenames=None):
        """
        Loads in the data from the CARAMEL input files

        :param data_dir: path to the directory storing the input data files. Defaults to self.fp_run + "Data/"
        :type data_dir: str

        :param data_filenames: filenames of the input files. If None, code will attempt to determine files.
        :type data_filenames: dictionary

        :return: None
        """
        if data_dir == '':
            data_dir = self.fp_run + "Data/"
        if data_filenames == None:
            fp_spectra = glob.glob(data_dir + "*spectra*")[0]
            fp_continuum = glob.glob(data_dir + "*continuum*")[0]
            fp_spec_times = glob.glob(data_dir + "*times*")[0]
        else:
            fp_spectra = data_dir + data_filenames["spectra"]
            fp_continuum = data_dir + data_filenames["continuum"]
            fp_spec_times = data_dir + data_filenames["spectra_times"]

        print "Loading CARAMEL input data files..."
        print "  spectra: %s" % fp_spectra
        print "  continuum: %s" % fp_continuum
        print "  times: %s" % fp_spec_times
        print ""

        self.wavelengths = np.loadtxt(fp_spectra)[0, :]
        self.data = np.loadtxt(fp_spectra)[1::2, :]
        self.err = np.loadtxt(fp_spectra)[2::2, :]
        self.cont = np.loadtxt(fp_continuum)
        self.cont[:, 0] = self.cont[:, 0] / 86400.  # in days
        self.times = np.loadtxt(fp_spec_times)[:, 0] / 86400.  # in days

        self.llim = (
            self.wavelengths[0], self.wavelengths[-1])  # wavelength range
        self.elim = (1, len(self.data))  # observing epoch range

        self.num_epochs = np.size(self.data[:, 0])
        self.num_params_before_rainbow = self.nparams + self.num_epochs

        # Start and end indices of spectra model
        self.spec_ind = [self.num_params_before_rainbow,
                         self.num_params_before_rainbow + self.data.size]
        
        # Start index of continuum model
        self.cont_ind = self.spec_ind[1] + self.nhyperparams

    def loadPosterior(self, fn='', posterior_sample=None):
        """
        Loads in the posterior_sample.npy file and sets the .posterior attribute. If a .txt file, 
        converts to .npy for faster loading in the future.

        :param fn: filepath to the posterior_sample.npy file, defaults to ''
        :type fn: str

        :param posterior_sample: posterior_sample numpy array
        :type posterior_sample: str

        :return: None
        """
        if posterior_sample != None:
            print "Using provided posterior_sample array"
            self.posterior_sample = posterior_sample
        else:
            if fn != '':
                self.fn_posterior = fn
            elif hasattr(self, 'fn_posterior'):
                pass
            else:
                fn_posterior = "%sposterior_sample" % self.fp_run
                if os.path.exists(fn_posterior + '.npy'):
                    self.fn_posterior = "%sposterior_sample.npy" % self.fp_run
                else:
                    self.fn_posterior = "%sposterior_sample.txt" % self.fp_run
            
            # If .txt file, copy to .npy for future use
            if self.fn_posterior[-4:] == '.txt':
                print self.fn_posterior
                print "Loading posterior_sample file: %s" % self.fn_posterior
                print "Converting posterior_sample.txt to posterior_sample.npy..."
                self.posterior_sample = np.loadtxt(self.fn_posterior)
                np.save(self.fn_posterior[:-4] + '.npy', self.posterior_sample)
            else:
                print "Loading posterior_sample file: %s" % self.fn_posterior
                self.posterior_sample = np.load(self.fn_posterior)
        print ""
        self.makePosteriorDataFrame(self.posterior_sample)
        self.num_samples = np.shape(self.posterior_sample)[0]

        print "%s items in posterior sample" % len(self.posterior)
        try:
            print self.posterior[['log10Mbh','Rmin','Rmedian','Taumedian','Thetai','Thetao',
            'Beta','Gamma','Kappa','Xi','Fflow','Fellip']].describe().loc[['mean','50%','std']]
        except:
            pass

    def postprocess(self, temp=-1, cut=-1, savename='posterior_sample', force=False, **kwargs):
        """
        Calls a modified version of the the DNest postprocess code. 

        :param temp: Temperature, defaults to self.temp
        :type temp: int

        :param cut: Fraction of samples to cut from beginning, defaults to self.cut
        :type cut: float

        :param savename: File to save posterior sample, defaults to 'posterior_sample'
        :type savename: str

        :param force: Whether to force-overwrite the current posterior_sample file, defaults to False
        :type force: bool

        :return: None
        """
        os.chdir(self.fp_run)

        if temp == -1:
            temp = self.temp
        if cut == -1:
            cut = self.cut
        
        # Safeguard against over-writing posterior_sample files
        confirm = True
        if os.path.exists(savename+'.npy') or os.path.exists(savename+'.txt'):
            if not force:
                unser_input = raw_input("Are you sure you want to overwrite posterior_sample files? (Y/N)")
                if unser_input.lower() not in ['y','yes']:
                    confirm = False
        if confirm:
            try:
                os.remove(savename+'.npy')
                print "Removing previous posterior_sample.npy file"
            except:
                pass
            try:
                os.remove(savename+'.txt')
                print "Removing previous posterior_sample.txt files"
            except:
                pass
            print "Running postprocess.py with temperature=%i and cut=%.2f..." % (temp, cut)
            postprocess.postprocess(temperature=temp, cut=cut, savename=savename, **kwargs)
            self.posterior_sample = np.load(savename+'.npy')
            self.makePosteriorDataFrame(self.posterior_sample)
            self.num_samples = np.shape(self.posterior_sample)[0]
            self.fn_posterior = "%s%s" % (self.fp_run, savename+'.npy')
        else:
            print "Skipping postprocess.py"
            print ""

        os.chdir(self.cwd)

    def convergence(self, cut=-1, **kwargs):
        """
        Calls calc.convergence. The posterior_sample file must already exist to
        run this code.

        :param cut: Percent of samples to cut from beginning, defaults to self.cut
        :type cut: float

        :return: None
        """
        os.chdir(self.fp_run)

        if cut == -1:
            cut = self.cut
        print "Running convergence.py with cut=%.2f..." % (cut)
        convergence.convergence(cut, params_list=self.CARAMEL_params_list, **kwargs)

        os.chdir(self.cwd)

    def plotDisplay(self, **kwargs):
        """
        Calls cappuccino.plotting.display.plotDisplay()

        :return: None
        """
        display.plotDisplay(self, **kwargs)

    def plotPosterior(self, **kwargs):
        """
        Calls cappuccino.plotting.posteriors.plotPosteriors()

        :return: None
        """
        posteriors.plotPosteriors(self, **kwargs)

    def plotCorner(self, params, interactive=False, **kwargs):
        """
        Calls cappuccino.plotting.corner.cornerPosterior()
        
        :param params: Posterior sample parameters to plot
        :type params: list of strings

        :param interactive: If True, plot using Bokeh
        :type interactive: bool

        :return: None
        """
        if 'truth' not in kwargs:
            if hasattr(self, 'truth'):
                kwargs['truth'] = self.truth
        if interactive:
            corner.cornerBokeh(self.posterior, params, **kwargs)
        else:
            corner.corner(self.posterior, params, **kwargs)

    def plotGeos(self, clouds_num=[], fp_clouds='', print_params=[], **kwargs):
        """
        Reads in the clouds files and calls cappuccino.plotting.geometries.cornerPosterior()

        :return: None
        """
        if fp_clouds == '':
            fp_clouds = self.fp_run + 'clouds/'
        if len(clouds_num) == 0:
            fn_clouds = glob.glob(fp_clouds + 'clouds_*.txt')
        else:
            fn_clouds = [fp_clouds + 'clouds_%i.txt' % num for num in clouds_num]
        for f in fn_clouds:
            start_ind = f.index('clouds_')
            end_ind = f.index('.txt')
            line_number = int(f[start_ind+7:end_ind])
            print "Line %i" % line_number
            for param in print_params:
                print "%s: %f" % (param, self.posterior[param][line_number])
            geo = geometries.Geometry(f)
            geo.plotGeo(**kwargs)

    def readPosteriorFile(self, filepath, fp_sample_info='', level_cut=[0,0]):
        """
        Reads the posterior sample file output from CARAMEL, then calls
        makePosteriorDataFrame to convert to a posterior dictionary.

        :param filepath: Path to the posterior sample file
        :type filepath: str

        :return: None
        """
        # Load the posterior sample file
        print "Loading posterior sample from %s" % filepath
        posterior = np.load(filepath)
        if level_cut != [0,0]:
            print "Using levels between %i and %i" % (level_cut[0], level_cut[1])
            if fp_sample_info == '':
                ind = filepath[::-1].index('/')
                fp_sample_info = filepath[:-ind] + 'sample_info.npy'
            print "Getting sample info from %s" % fp_sample_info
            sample_info = np.load(fp_sample_info)
            #int(cut * sample_info.shape[0])
            posterior = posterior[np.where((sample_info[:,0]>=level_cut[0]) & (sample_info[:,0]<=level_cut[1]))]

        self.posterior_sample = posterior
        self.makePosteriorDataFrame(posterior)

    def readParamFileJSON(self, filepath, json_chain=None):
        """
        Reads a file that contains external parameters for the AGN/BLR being
        studied, such as its LOS velocity, luminosity distance, CCF lag
        measurement, etc.

        :param filepath: Filepath to the .json file to be loaded
        :type filepath: str

        :param json_chain: If there are multiple levels to the json file, 
        :type json_chain: list

        :return: None
        """
        with open(filepath, 'r') as f:
            contents = json.load(f)
        to_load = ["dvrmssigma", "dvmeansigma", "dvmeanfwhm", "Tauccf", "f5100l", "DLMpc", "LEdd", "LBol",
                  "log10L5100","sigmastar","Rmax"]
        for c in json_chain:
            contents = contents[c]
        for p in to_load:
            if key in contents.keys():
                if type(contents[key]) == list:
                    vals = contents[key]
                    self.params[p] = np.array([vals[0], vals[1], vals[2]])
                else:
                    self.params[p] = np.array([contents[key], 0, 0])

    def readParamFile(self, agn, filepath):
        """
        Reads a file that contains external parameters for the AGN/BLR being
        studied, such as its LOS velocity, luminosity distance, CCF lag
        measurement, etc.

        :param filepath: Filepath to the .csv file to be loaded
        :type filepath: str

        :param json_chain: If there are multiple levels to the json file, 
        :type json_chain: list

        :return: None
        """
        df = pd.read_csv(filepath)
        df = df.set_index('param')
        to_load = ["dvrmssigma", "dvmeansigma", "dvmeanfwhm", "Tauccf", "f5100l", "DLMpc", "LEdd", "LBol",
                  "log10L5100","sigmastar","Rmax"]
        for p in to_load:
            self.params[p] = np.array([float(df[agn][p]), float(df[agn][p+"m"]), float(df[agn][p+"p"])])

    def makePosteriorDataFrame(self, posterior):
        """
        Takes a posterior array in CARAMEL output form and puts it into a
        Pandas DataFrame object after applying appropriate scaling and
        conversions for each of the model parameters.

        :param posterior: Posterior sample output from CARAMEL + DNest
        :type filepath: numpy array

        :return: None
        """

        # Get the scaling values for each parameter
        scale = parInfo.paramScalesCARAMELOut()
        
        # Need to add an additional redshift correction for some parameters
        add_redshift_correction = {
            "Rmean": True,
            "Rmedian": True,
            "mu": True,
            "Taumean": True,
            "Taumedian": True,
            "Mbh": True,
        }

        params = {}
        for i in range(len(self.CARAMEL_params_list)):
            par = self.CARAMEL_params_list[i]
            if par in add_redshift_correction.keys():
                redshift_correct = add_redshift_correction[par]
            else:
                redshift_correct = False
            if redshift_correct:
                params[par] = posterior[:, i] * scale[par] / (1. + self.redshift)
            else:
                params[par] = posterior[:, i] * scale[par]
        params["Rmin"] = params["mu"] * params["F"]
        params["Sigmar"] = params["mu"] * params["Beta"] * (1.0 - params["F"])
        params["Sigmar_numeric"] = params["Rmean"] * params["Beta"] * (1.0 - params["F"])
        params["log10Mbh"] = np.log10(params["Mbh"])

        self.paramnames = params.keys()
        for key in params.keys():
            self.addParam(key, params[key])

        self.num_samples = len(posterior)

    def setParam(self, paramname, vals):
        """
        Used to set 'params,' which are values that have been measured, such as
        cross-correlation lag, line width, etc. These are NOT model parameters.

        :param paramname: String defining the param name
        :type paramname: str

        :param vals: [median, minus_uncertainty, plus_uncertainty]
        :type vals: list or numpy.array (length 3)

        :return: None
        """
        self.params[paramname] = np.array(vals)

    def addParam(self, paramname, posterior):
        """
        Adds the posterior sample to the posteriors attribute. Also adds 68%
        confidence intervals to the confints attribute.

        :param paramname: Name of the parameter to be added
        :type paramname: str

        :param posterior: List of posterior samples corresponding to paramname
        :type posterior: list

        :return: None
        """
        self.posterior[paramname] = posterior
        self.confints[paramname] = self.calcConfInt(paramname, 68.27)
    
    def calcConfInt(self, param, percentile=68.27, onesided=False, prior=None):
        """
        Calls utils.util.confInt() to compute the confidence interval for a parameter.

        :param param: Name of the parameter to be added
        :type param: str
        
        :param percentile: Percentile for which to calculate confidence
            interval. Defaults to 68.27.
        :type percentile: float

        :param onesided: Standard confidence interval (False), upper limit
            ('lower'), or lower limit ('upper')
        :type onesided: float

        :param prior: Prior range for the parameter, recommended if 'onesided'
            is not False.
        :type prior: list, length 2

        :return: None
        """
        return util.confInt(self.posterior[param], percentile, onesided, prior)

    def setKeys(self):
        """
        Sets the keys attribute to be the list of posterior keys.

        :return: None
        """
        self.keys = self.posterior.columns()

    def calcVirialProduct(self, veltype=["meanfwhm", "meansigma", "rmssigma", "rmsfwhm"]):
        """
        Calculates the virial product using the Tauccf parameter, and the line
        widths defined in veltype. Saves a posterior sample of the same length
        as the other posteriors. Errors are propagated by assuming Gaussian
        errors in Tauccf and velocities and drawing random samples.

        :param veltype: List of the line with measurement types to use
        :type veltype: list

        :return: None
        """
        print "Calculating virial products, using veltype =", veltype
        if not "Tauccf" in self.params.keys():
            print "-Tauccf must be added to self.params before computing virial product. " +\
                "Add using setParam(). Exiting."
            print ""
            return 

        if type(veltype) != list:
            veltype = [veltype]

        # Virial Product
        tau = self.params["Tauccf"]
        print "-Using lag (days) = %.2f +%.2f/-%.2f" % (tau[0], tau[2], tau[1])
        tau_post = tau[0] + np.mean([tau[1], tau[2]]) * np.random.randn(self.num_samples)
        # Don't allow negative lags
        for i in range(len(tau_post)):
            while tau_post[i] < 0:
                tau_post[i] = tau[0] + np.mean([tau[1], tau[2]]) * np.random.normal()
        
        R = tau_post * const.ckm * const.day
        
        for v in veltype:
            if not "dv" + v in self.params.keys():
                print "-dv%s must be added to self.params before computing virial product. " % v + \
                    "Add using setParam(). Skipping this iteration."
                print ""
                continue 
            dv = self.params["dv" + v]
            print "-Using %s (km/s) = %.2f +%.2f/-%.2f" % (v, dv[0], dv[2], dv[1])
            dv_post = dv[0] + np.mean([dv[1], dv[2]]) * np.random.randn(self.num_samples)
            # Don't allow negative line widths
            for i in range(len(dv_post)):
                while dv_post[i] < 0:
                    dv_post[i] = dv[0] + np.mean([dv[1], dv[2]]) * np.random.normal()
            log10VP = np.log10((R * dv_post ** 2) / const.G)

            self.addParam('log10VP_' + v, log10VP)
        print ""

    def calcF(self, veltype=["meanfwhm", "meansigma", "rmssigma", "rmsfwhm"]):
        """
        Creates an f-factor posterior, using the Mbh posterior distribution, and
        the VP posterior distribution created by calcVirialProduct.

        :param veltype: List of the line with measurement types to use
        :type veltype: list

        :return: None
        """
        print "Calculating scale factors, using veltype =", veltype
        if type(veltype) != list:
            veltype = [veltype]
        log10Mbh = self.posterior['log10Mbh']
        for v in veltype:
            try:
                log10VP = self.posterior['log10VP_' + v]
            except:
                print "-Calculating virial product for %s" % v
                self.calcVirialProduct(v)
                log10VP = self.posterior['log10VP_' + v]
            log10f = log10Mbh - log10VP
            self.addParam('log10f_' + v, log10f)
        print ""

    def calcLog10LWave(self, wave=5100):
        """
        Calculates log10(L_wave), given f_(wave,lambda) and D_L in Mpc, and adds
        the resulting posterior.

        :param wave: Wavelength of luminosity measurement, defaults to 5100
        :type wave: int

        :return: None
        """
        print "Calculating log10L%i from f%il and DLMpc" % (wave, wave)
        z = self.redshift

        if "log10L%i" % wave in self.params.keys():
            user_input = raw_input("-log10L%i already exists in self.params. Re-compute? (Y/N)" % wave)
            if user_input.lower() not in ['y','yes']:
                print "-Transfering self.params to self.posterior and exiting."
                log10Lwave = self.params["log10L%i" % wave]
                log10Lwave_post = log10Lwave[0] + np.mean([log10Lwave[1],log10Lwave[2]]) * np.random.randn(
                    self.num_samples)
                self.addParam("log10L%i" % wave, log10Lwave_post)
                print ""
                return
        
        not_provided = []
        for par in ["f%il" % wave, "DLMpc"]:
            if par not in self.params.keys():
                not_provided.append(par)
        if len(not_provided) != 0:
            print "-%s must be added to self.params before computing. " % (' and '.join(not_provided)) + \
                "Add using setParam(). Exiting." 
            print ""
            return 

        fwavel = self.params["f%il" % wave]  # fwave * lambda
        fwavel_post = fwavel[0] + np.mean([fwavel[1],fwavel[2]]) * np.random.randn(self.num_samples)
        
        dmp = self.params["DLMpc"]  # Megaparsec
        dmp_post = dmp[0] + np.mean([dmp[1],dmp[2]]) * np.random.randn(self.num_samples)
        dcm = dmp_post * 3.086 * 10 ** 24  # centimeters
        fwave = fwavel_post * float(wave) * 10 ** -15  # fwave
        Lwave = fwave * 4 * np.pi * np.power(dcm, 2)  # Luminosity
        log10Lwave = np.log10(np.array(list(Lwave)))

        self.addParam("log10L%i" % wave, log10Lwave)
        print ""

    def calcLog10L5100(self):
        """
        Calls calcLog10LWave for wave=5100

        :return: None
        """
        self.calcLog10LWave(self, wave=5100)

    def calcLBol(self, wave=5100, bol_correction=9.):
        """
        Calculates the bolometric luminosity, given luminosity and a bolometric
        correction, and adds the resulting posterior.

        :param wave: Wavelength of luminosity measurement, defaults to 5100
        :type wave: int

        :param bol_correction: Bolometric correction, defaults to 9.
        :type bol_correction: float

        :return: None
        """
        print "Calculating bolometric luminosity with wavelength = %i Angstroms, bolometric correction = %.1f" % (wave, bol_correction)
        par = "log10L" + str(wave)
        if par not in self.posterior.keys():
            if not par in self.params.keys():
                print "-%s must be be in self.posterior or self.params before computing. " % par +\
                    "Add using setParam(). Exiting."
                print ""
                return -1
            print "-Creating posterior for %s" % par
            logL_post = np.zeros(self.num_samples)
            for i in range(self.num_samples):
                logL_post[i] = self.params[par][0] + np.mean([self.params[par][1],
                    self.params[par][2]]) * np.random.normal()
            self.addParam(par, logL_post)

        Lsun = 3.839 * 10 ** 33.  # erg/s
        Lwave = 10 ** self.posterior[par]
        Lwavesun = Lwave / Lsun  # Solar units
        self.addParam("LBol", Lwavesun * bol_correction)
        print ""
        return

    def convertLwave(self, wave_old, wave_new, bol_correction_old, bol_correction_new):
        """
        Converts log10Lwave_old to log10Lwave_new, using two bolometric corrections. E.g., 
        convert log10L1350 to log10L5100 for comparison with previous modeling results.

        :param wave_old: Wavelength of old luminosity measurement
        :type wave_old: int

        :param wave_new: Wavelength of new luminosity measurement
        :type wave_new: int

        :param bol_correction_old: Bolometric correction of old measurement
        :type bol_correction_old: float

        :param bol_correction_new: Bolometric correction of new measurement
        :type bol_correction_new: float

        :return: None
        """
        par_old = "log10L" + str(wave_old)
        par_new = "log10L" + str(wave_new)

        if par_old not in self.posterior.keys():
            if not par_old in self.params.keys():
                print "-%s must be added to self.params before computing. " % par_old +\
                    "Add using setParam(). Exiting."
                print ""
                return 
            print "-Creating posterior for %s" % par_old
            logL_post = np.zeros(self.num_samples)
            for i in range(self.num_samples):
                logL_post[i] = self.params[par_old][0] + np.mean([self.params[par_old][1],
                    self.params[par_old][2]]) * np.random.normal()
            self.addParam(par_old, logL_post)

        post_new = self.posterior[par_old] + np.log10(bol_correction_old / bol_correction_new)
        self.addParam(par_new, post_new)

    def calcLEdd(self):
        """
        Calculates the Eddington ratio and adds the resulting posterior

        :return: None
        """
        print "Calculating Eddington ratio"
        # LEdd
        self.addParam("LEdd", 3.93 * 10 ** 4 * self.posterior["Mbh"])

        # loverlEdd
        if "LBol" not in self.posterior.keys():
            "-Computing LBol before computing Eddington ratio..."
            return_status = self.calcLBol()
            if return_status == -1:
                "-Couldn't compute Eddington ratio. Exiting."
                return
        
        self.addParam("loverlEdd",
                      self.posterior["LBol"] / self.posterior["LEdd"])
        print ""

    def calcInflowOutflow(self):
        """
        Calculates the inflow-outflow parameter, defined in Williams et al. 2018

        :return: None
        """
        print "Calculating inflow-outflow parameter"
        inflowoutflow = (1. - self.posterior["Fellip"]) * \
                np.where(self.posterior["Fflow"]>0.5,1.,-1.) * \
                np.cos(self.posterior["Thetae"] / 180. * np.pi)
        self.addParam("inflowoutflow", inflowoutflow)
        print ""

    def calcMSini(self):
        """
        Calculates M_BH * sin(i).

        :return: None
        """
        print "Calculating MSini"
        msini = self.posteriors["Mbh"] * np.sin(
            self.posteriors["Thetai"] / 180. * np.pi)
        self.addParam("MbhSinThetai", msini)
        print ""

    def calcRmeanoverRmin(self):
        """
        Calculates Rmean / Rmin

        :return: None
        """
        print "Calculating r_mean/r_min"
        RmeanoverRmin = self.posterior["Rmean"] / self.posterior["Rmin"]
        self.addParam("RmeanoverRmin", RmeanoverRmin)
        print ""

    def calcCovariance(self):
        self.setKeys()
        self.cov = np.cov([self.posterior[key] for key in self.keys])

    def transferParams(self):
        """
        This will add a parameter to the object's params attribute only if 1)
        the loaded value is not NaN and 2) the posterior sample does not already
        exist. This ensures that the published values from previous papers are
        used in case there is a discrepancy between the published values and the
        calculated values.

        :return: None
        """
        for p in self.params.keys():
            if not np.isnan(self.params[p][0]) or p not in self.posterior.keys():
                param = self.params[p]
                posterior = param[0] + param[1] * np.random.randn(self.num_samples)
                self.addParam(p, posterior)


def combinePosteriors(posterior_list, Nsamp, weights=1., seed=0, savename=''):
    """
    Combines the posterior sample files in posterior_list into a new posterior sample
    file of length Nsamp.

    :param posterior_list: List of posterior samples
    :type posterior_list: dict

    :param Nsamp: Length of output combined posterior file.
    :type Nsamp: int

    :param weights: List of weights for combining posterior samples, defaults to equal weights.
    :type weights: list

    :param seed: numpy seed for random numbers, defaults to 0.
    :type seed: int

    :param savename: File to save combined posterior.
    :type savename: str

    :return: Combined posterior sample
    """
    np.random.seed(seed)

    if np.shape(weights) == ():
        weights = np.ones(len(posterior_list)) * weights
    weights = np.array(weights) / np.sum(np.array(weights))

    print "Combining posteriors using weights ", weights

    posterior_combined = [0] * Nsamp
    for i in range(Nsamp):
        # Determine from which posterior to draw from
        rand = np.random.uniform()
        for j in range(len(weights)):
            if rand < sum([w for w in weights[:j + 1]]):
                sample = j
                break
        # Choose a random sample
        rand_index = np.random.randint(0, len(posterior_list[sample]))
        posterior_combined[i] = post[sample][rand_index]

    if savename != '':
        np.save(savename, posterior_combined)
    return posterior_combined


def buildPosterior(fp_params,fp_posterior,agn):
    """
    This function gathers all of the information for a given
    :param fp_params: Filepath to the file with the AGN parameters stored
    :type fp_params: str

    :param fp_posterior: Filepath to the AGN's CARAMEL output file
    :type fp_posterior: str

    :param agn:
    :type agn: str

    :return: Posterior object for the AGN
    """
    # Create the Posterior object
    post = Posterior(name=agn)

    # Read the campaign from data file
    post.campaign = pd.read_csv(fp_params, index_col=0)[agn]["Campaign"]
    post.shortname = pd.read_csv(fp_params,index_col=0)[agn]["shortname"]
    post.texname = pd.read_csv(fp_params,index_col=0)[agn]["texname"]
    post.redshift = float(pd.read_csv(fp_params,index_col=0)[agn]["z"])

    # Load the params
    post.readParamFile(agn, fp_params)

    # Load the posterior sample file
    post.readPosteriorFile(fp_posterior)

    # Calculate various values
    post.calcVirialProduct(veltype=["meanfwhm", "meansigma", "rmssigma"])
    #post.calcVirialProductNOERRORPROP()
    post.calcF(veltype=["meanfwhm", "meansigma", "rmssigma"])
    #post.calcFNOERRORPROP()
    post.calcLog10L5100()
    post.calcLBol()
    post.calcLEdd()
    post.calcInflowOutflow()
    post.calcMSini()
    post.calcRmeanoverRmin()

    # Create posterior samples for any loaded parameters, assuming Gaussian
    # errors when applicable.
    post.transferParams()

    return post


def extractConfIntsAndCov(post, varx, vary, agn_order=[], x_logify=False, y_logify=False):
    print "\nWorking on %s vs. %s" % (vary, varx)
    x = list()
    y = list()
    cov = list()
    campaign = list()
    objectname = list()
    if len(agn_order) == 0:
        agn_order = post.keys()
    for agn in agn_order:
        try:
            # Load in the posterior samples and put into proper form
            x_posterior = post[agn].posteriors[varx]
            y_posterior = post[agn].posteriors[vary]

            # If a constant, copy to array the same size as y_posterior
            if isinstance(x_posterior, float):
                x_posterior = np.array([x_posterior] * len(y_posterior))
            if x_logify:
                x_posterior = np.log10(x_posterior)
            if y_logify:
                y_posterior = np.log10(y_posterior)

            # Calculate the covariance matrix
            cov_tmp = np.cov(x_posterior, y_posterior)

            # If there are any NaNs in the covariance matrix, ignore point
            if 1 in [np.isnan(cov_tmp[n, m]) for n in [0, 1] for m in [0, 1]]:
                print "Covariance matrix for " + agn + " " + vary + ' vs ' + varx + " contains NaNs."
                if sum(np.isnan(x_posterior[i]) for i in range(len(x_posterior))) > 0:
                    print "NaNs in x_posterior: " + varx
                if sum(np.isnan(y_posterior[i]) for i in range(len(y_posterior))) > 0:
                    print "NaNs in y_posterior: " + vary
                print "Ignoring this data point."
            else:
                if cov_tmp[0, 1] != cov_tmp[1, 0]:
                    print "WARNING: Covariance matrix NOT symmetric"
                if varx in post[agn].params.keys():
                    if post[agn].params[varx][1] != 0 and not np.isnan(post[agn].params[varx][1]):
                        tmp = post[agn].params[varx]
                        par = [tmp[0],tmp[2],tmp[1]]
                        if x_logify:
                            par = confIntLogify(par[0],par[1],par[2])
                        x.append([par[0],par[1],par[2]])
                    else:
                        x.append(confInt(x_posterior, 68.2689))
                else:
                    x.append(confInt(x_posterior, 68.2689))
                if vary in post[agn].params.keys():
                    if post[agn].params[vary][1] != 0 and not np.isnan(post[agn].params[vary][1]):
                        tmp = post[agn].params[vary]
                        par = [tmp[0], tmp[2], tmp[1]]
                        if y_logify:
                            par = confIntLogify(par[0],par[1],par[2])
                        y.append([par[0],par[1],par[2]])
                    else:
                        y.append(confInt(y_posterior, 68.2689))
                else:
                    y.append(confInt(y_posterior, 68.2689))
                cov.append(cov_tmp)
                campaign.append(post[agn].campaign)
                objectname.append(post[agn].name)
        except:
            print "Failed on ", agn

    x, y, cov, campaign, objectname = np.array(x), np.array(y), np.array(cov), np.array(campaign), np.array(objectname)
    return x,y,cov,campaign,objectname