file_directory = '/'.join(__file__.split('/')[:-1]) + '/'

import numpy as np
from cappuccino.utils import parInfo, util
#from cappuccino.calc import general
from cappuccino.calc import meanF
reload(parInfo)


def getSigDigs(vals,lim='max'):
    ndigs = []
    for val in vals:
        val_str = str(float(val))
        #dig_start = 0
        print val_str
        if val_str == 'nan':
            ndig = 0
        else:
            for i in range(len(val_str)):
                if val_str[i] == '.':
                    dig_start = i
            for i in range(len(val_str)):
                if val_str[i] not in ['0','.','-']:
                    sig_start = i
                    break
            #print dig_start
            #print sig_start
            if sig_start > dig_start:
                ndig = sig_start - dig_start + 1
            elif sig_start < dig_start - 1:
                ndig = 0
            else:
                ndig = 1
        ndigs.append(ndig)
    if lim == 'min':
        ndig = min(ndigs)
    else:
        ndig = max(ndigs)

    return ndig


def meanFFactor(infile, texprefix, vtype, outfile):
    results = meanF.readMeanF(infile)

    lines = list()
    lines.append('\\newcommand{\\%smeanf%s}{$%.2f \pm %.2f$}\n' % (texprefix,vtype,results[0],results[1]))
    lines.append('\\newcommand{\\%ssigmaf%s}{$%.2f \pm %.2f$}\n' % (texprefix,vtype,results[2],results[3]))
    lines.append('\\newcommand{\\%spredf%s}{$%.2f \pm %.2f$}\n' % (texprefix,vtype,results[4],results[5]))

    with open(outfile,'ab') as f:
        f.writelines(lines)
    print outfile


def CARAMELResults(post,run_key,params,outfile,onesided={},noerrorbars={},ndigits={}, logify={}, scale={}, overwrite=False):
    """
    Save the CARAMEL results as tex commands to import into a tex file.

    :param posteriors: List of posteriors to plot. If Model objects are provided, will extract the
        posteriors, names, and confidence intervals.
    :type posteriors: List of Pandas DataFrames

    :return: ax
    """
    param_tex_names = parInfo.commandTexNames()
    lines = list()
    #ndigits = parInfo.paramDigits()
    post.conf_int = {}
    for param in params:
        if param not in scale.keys():
            scale[param] = 0.0
        if logify.get(param):
            vals = np.log10(post.posterior[param])
        else:
            vals = post.posterior[param]/10**scale[param]
        
        if noerrorbars.get(param):
            p50 = np.percentile(vals,50)
            try:
                ndig = ndigits[param]
            except:
                ndig = getSigDigs([p50])
            paramvalue = '%.*f' % (ndig, p50)
        else:
            conf_int = util.confInt(vals, percentile=68.27, onesided=onesided.get(param), prior=None) # med, med-lo, hi-med
            try:
                ndig = ndigits[param]
            except:
                if not onesided.get(param):
                    ndig = getSigDigs(conf_int[1:], lim='min')
                elif onesided.get(param) in ['lo','upper']:
                    ndig = getSigDigs([conf_int[0] + conf_int[2]])
                elif onesided.get(param) in ['hi', 'lower']:
                    ndig = getSigDigs([conf_int[0] - conf_int[1]])

            if not onesided.get(param):
                #ndig = getSigDigs(conf_int[1:], lim='min')
                paramvalue = '%.*f_{-%.*f}^{+%.*f}' % (
                    ndig, conf_int[0], ndig, conf_int[1], ndig, conf_int[2])
            elif onesided.get(param) in ['lo','upper']:
                #ndig = getSigDigs(conf[0] + conf[2])
                paramvalue = '<%.*f' % (ndig, conf_int[0] + conf_int[2])
            elif onesided.get(param) in ['hi', 'lower']:
                #ndig = getSigDigs(conf[0] - conf[1])
                paramvalue = '>%.*f' % (ndig, conf_int[0] - conf_int[1])

        if scale[param] != 0.0:
            paramvalue = paramvalue[:] + '\\times 10^{%i}' % scale[param]
        lines.append('\\newcommand{\\%s%s}{%s}\n' % (
            run_key, param_tex_names[param], paramvalue))

    lines.append("\n")
    if overwrite:
        with open(outfile,'wb') as f:
            f.writelines(lines)
    else:
        with open(outfile,'ab') as f:
            f.writelines(lines)
    print outfile


def correlations(varsx, varsy, usecovariance=False, fp_linmix='', lmsuffix='',
                 outfile='',ndigs=2):
    # Cast lmsuffix to shape of (len(varsx),len(varsy))
    if np.shape(lmsuffix) == ():
        lmsuffix = np.array([lmsuffix] * len(varsx) * len(varsy))
        lmsuffix = lmsuffix.reshape((len(varsx), len(varsy)))

    lines = list()

    for j in range(len(varsx)):
        varx = varsx[j]
        for l in range(len(varsy)):
            vary = varsy[l]
            print "\n%s vs. %s" % (vary, varx)

            # Load LinMix results to determine correlation parameters
            if usecovariance:
                usecov = 'wcov'
            else:
                usecov = 'nocov'
            loadname = "%s%s_vs_%s_%s_%s.npy" % (
            fp_linmix, vary, varx, usecov, lmsuffix[j][l])

            lmchain = np.load(loadname)
            chain = {
                'alpha': lmchain[:, 0],
                'beta': lmchain[:, 1],
                'sigsqr': lmchain[:, 2],
                'corr': lmchain[:, 3]
            }

            varxname = parInfo.commandTexNames()[varx]
            varyname = parInfo.commandTexNames()[vary]

            alpha_int = general.confInt(chain['alpha'], 68.27)
            beta_int = general.confInt(chain['beta'], 68.27)
            sigsqr_int = general.confInt(chain['sigsqr'], 68.27)

            ndig = ndigs['a'][l][j]
            alphatext = "$%.*f_{-%.*f}^{+%.*f}$" % (ndig, alpha_int[0],
                                                    ndig,alpha_int[1],
                                                    ndig, alpha_int[2])
            ndig = ndigs['b'][l][j]
            betatext = "$%.*f_{-%.*f}^{+%.*f}$" % (ndig, beta_int[0],
                                                   ndig, beta_int[1],
                                                   ndig, beta_int[2])
            ndig = ndigs['s'][l][j]
            scattertext = "$%.*f_{-%.*f}^{+%.*f}$" % (
                ndig, np.sqrt(sigsqr_int[0]),
                ndig, np.sqrt(sigsqr_int[1]),
                ndig, np.sqrt(sigsqr_int[2]))


            lines.append('\\newcommand{\\alpha%svs%s}{%s}\n' % (varyname,
                                                                varxname,
                                                                alphatext))
            lines.append('\\newcommand{\\beta%svs%s}{%s}\n' % (varyname,
                                                                varxname,
                                                                betatext))
            lines.append('\\newcommand{\\scat%svs%s}{%s}\n' % (varyname,
                                                                varxname,
                                                                scattertext))


    with open(outfile,'ab') as f:
        f.writelines(lines)
    print outfile