import numpy as np
from cappuccino.utils import parInfo, util
import pandas as pd
import os
reload(parInfo)
reload(util)


def saveAndCompile(savename, lines):
    with open(savename, 'wb') as f:
        f.writelines('\n'.join(lines))
    outdir = "/".join(savename.split("/")[:-1])
    os.system("pdflatex -interaction=nonstopmode -output-directory %s %s" % (outdir, savename))
    os.system("rm %s.aux" % savename[:-4])
    os.system("rm %s.log" % savename[:-4])
    os.system("open %s.pdf" % savename[:-4])


def buildTable(header,content,cols=""):
    lines = []
    lines.append("\\documentclass[border=6pt]{standalone}")
    lines.append("\\usepackage{amsmath}")
    lines.append("\\usepackage{amssymb}")
    lines.append("")
    lines.append("\\begin{document}")
    lines.append("\\begin{table}")
    lines.append("\\begin{tabular}{%s}" % cols)
    lines.append("\\hline")
    lines.append("\\hline")
    lines.append(header)
    lines.append("\\hline")

    for l in content:
        lines.append(l)

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("\\end{document}")

    return lines


def AGNvsParam(post,agn_list,savename,params,onesided={}, scale={}):
    for agn in agn_list:
        post[agn].conf_int = {}
        for param in params:
            if param not in scale.keys():
                scale[param] = 0.0
            try:
                vals = post[agn].posterior[param]/10**scale[param]
            except:
                vals = None
            if param in post[agn].params.keys():
                if post[agn].params[param][1] != 0:
                    pmed,pup,plo = post[agn].params[param]
                    post[agn].conf_int[param] = np.array([pmed,plo,pup])
                    break
            try:
                conf_int = util.confInt(vals, percentile=68.27, onesided=onesided[agn].get(param), prior=None) # med, med-lo, hi-med
                post[agn].conf_int[param] = conf_int
            except:
                post[agn].conf_int[param] = [np.nan, np.nan, np.nan]

        print "%s:" % agn
        for param in params:
            tmp = post[agn].conf_int[param]
            print "%s: %.3f +%.3f / -%.3f" % (param, tmp[0], tmp[2], tmp[1])

    ndigits = parInfo.paramDigits()
    texnames = parInfo.paramTexNames()
    texunits = parInfo.paramTexUnits()

    cols = "l%s" % ('c'*len(agn_list))

    header = " "
    for agn in agn_list:
        header += " & %s" % post[agn].runname
    header += " \\\\"

    content = []
    for param in params:
        ndig = ndigits[param]
        l = "%s" % texnames[param]
        if texunits[param] != "":
            if scale[param] != 0:
                l += "$ (10^{%i}~$%s)" % (scale[param], texunits[param])
            else:
                l += " (%s)" % texunits[param]
        for agn in agn_list:
            tmp = post[agn].conf_int[param]     
            if np.isnan(tmp[0]):
                l += " & -----"
            else:
                if not onesided[agn].get(param):
                    l += " & $%.*f_{-%.*f}^{+%.*f}$" % (ndig, tmp[0], ndig, tmp[1], ndig, tmp[2])
                elif onesided[agn].get(param) in ['lo','upper']:
                    l += " & $<%.*f$" % (ndig, tmp[0]+tmp[2])
                elif onesided[agn].get(param) in ['hi','lower']:
                    l += " & $>%.*f$" % (ndig, tmp[0]-tmp[1])
        l += " \\\\"
        content.append(l)

    lines = buildTable(header, content, cols)
    print lines
    saveAndCompile(savename, lines)


def ParamvsAGN(post,agn_list,savename,vars):

    for agn in agn_list:
        post[agn].conf_int = {}
        for var in vars:
            p50,p16,p84 = np.percentile(post[agn].posteriors[var],[50,16,84])
            conf_int = np.array([p50,p50-p16,p84-p50])
            post[agn].conf_int[var] = conf_int

        print "%s:" % agn
        for var in vars:
            tmp = post[agn].conf_int[var]
            print "%s: %.3f +%.3f / -%.3f" % (var, tmp[0], tmp[1], tmp[2])

    ndigits = parInfo.paramDigits()
    texnames = parInfo.paramTexNames()

    cols = "l%s" % ('c'*len(vars))

    header = " "
    for var in vars:
        header += " & %s" % texnames[var]
    header += " \\\\"

    content = []
    for agn in agn_list:
        l = "%s" % post[agn].texname
        for var in vars:
            ndig = ndigits[var]
            tmp = post[agn].conf_int[var]
            if np.isnan(tmp[0]):
                l += " & -----"
            else:
                l += " & $%.*f_{-%.*f}^{+%.*f}$" % (ndig, tmp[0], ndig, tmp[1], ndig, tmp[2])
        l += " \\\\"
        content.append(l)

    lines = buildTable(header, content, cols)
    saveAndCompile(savename, lines)


def meanF(results,dvtype,savename):
    header = "Line width & \\langle\\log_{10} f\\rangle & \\sigma(\\log_{10} f) & Pred(\\log_{10} f)\\\\"

    content = []
    for i in range(len(results)):
        d = results[i]
        tmp = "%s & %.2f \\pm %.2f & %.2f \\pm %.2f & %.2f \\pm %.2f \\\\" % \
              (dvtype[i], d[0],d[1], d[2],d[3], d[4],d[5])
        content.append(tmp)

    lines = buildTable(header,content,cols="lccc")

    saveAndCompile(savename, lines)


def correlations(varsx, x_logify=False,datafile='',outfile=''):
    with open(datafile) as f:
        data = f.readlines()

    datadict = {}
    for i in range(len(data)):
        startindex = data[i].find('\\',3)
        endindex = data[i].find('}',startindex)
        key = data[i][startindex+1:endindex]
        startindex = data[i].find('{',endindex)
        endindex = data[i].find('}\\',startindex)
        value = data[i][startindex+1:endindex-1]
        datadict[key] = value

    lines = []

    for j in range(len(varsx)):
        varx = varsx[j]
        xtexname = parInfo.paramTexNames()[varx]
        xtexunits = parInfo.paramTexUnits()[varx]
        if x_logify[j]:
            xtexname = "$\\log_{10}(%s)$" % xtexname[1:-1]
        if xtexunits != "":
            xtexunits = "(%s)" % xtexunits

        varxname = parInfo.commandTexNames()[varx]

        a1 = datadict["alpha%svs%s" % ("logfmeanfwhm", varxname)]
        a2 = datadict["alpha%svs%s" % ("logfmeansigma", varxname)]
        a3 = datadict["alpha%svs%s" % ("logfrmssigma", varxname)]
        b1 = datadict["beta%svs%s" % ("logfmeanfwhm", varxname)]
        b2 = datadict["beta%svs%s" % ("logfmeansigma", varxname)]
        b3 = datadict["beta%svs%s" % ("logfrmssigma", varxname)]
        s1 = datadict["scat%svs%s" % ("logfmeanfwhm", varxname)]
        s2 = datadict["scat%svs%s" % ("logfmeansigma", varxname)]
        s3 = datadict["scat%svs%s" % ("logfrmssigma", varxname)]

        lines.append("%s %s & %s & %s & %s & %s & %s & %s & %s & %s & %s" % (
            xtexname, xtexunits, a1,b1,s1, a2,b2,s2, a3,b3,s3
        ))
        if j != len(varsx) - 1:
            lines[-1] += "\\\\"


    header = [
        "\\begin{deluxetable*}{lccccccccc}",
        "\\tablecaption{Linear regression results}",
        "\\tablewidth{0pt}",
        "\\tablehead{",
        "\\colhead{Parameter} &",
        "\\colhead{} &",
        "\\colhead{$\\log_{10}(f_{{\\rm mean},{\\rm FWHM}})$} &",
        "\\colhead{} &",
        "\\colhead{} &",
        "\\colhead{$\\log_{10}(f_{{\\rm mean},\\sigma})$} &",
        "\\colhead{} &",
        "\\colhead{} &",
        "\\colhead{$\\log_{10}(f_{{\\rm rms},\\sigma})$} &",
        "\\colhead{}\\\\\\hline",
        "\\colhead{$x$} &",
        "\\colhead{$\\alpha$} &",
        "\\colhead{$\\beta$} &",
        "\\colhead{$\\sigma_{\\rm int}$} &",
        "\\colhead{$\\alpha$} &",
        "\\colhead{$\\beta$} &",
        "\\colhead{$\\sigma_{\\rm int}$} &",
        "\\colhead{$\\alpha$} &",
        "\\colhead{$\\beta$} &",
        "\\colhead{$\\sigma_{\\rm int}$}",
        "}",
        "\\startdata"
    ]

    footer = [
        "\\enddata",
        "\\tablecomments{\\correlationtablecomment}.",
        "\\label{tab:table_correlations}",
        "\\end{deluxetable*}"
    ]

    lines = header+lines+footer
    with open(outfile, 'wb') as f:
        f.writelines('\n'.join(lines))


def correlationsRotated(varsx, x_logify=False,datafile='',outfile=''):
    with open(datafile) as f:
        data = f.readlines()

    datadict = {}
    for i in range(len(data)):
        startindex = data[i].find('\\',3)
        endindex = data[i].find('}',startindex)
        key = data[i][startindex+1:endindex]
        startindex = data[i].find('{',endindex)
        endindex = data[i].find('}\\',startindex)
        value = data[i][startindex+1:endindex-1]
        datadict[key] = value

    lines = []

    a = {}
    b = {}
    s = {}
    types = ['meanfwhm', 'meansigma', 'rmssigma']
    textype = {'meanfwhm':'${\\rm mean},{\\rm FWHM}$',
               'meansigma':'${\\rm mean},{\\sigma}$',
              'rmssigma':'${\\rm rms},{\\sigma}$',
               }
    for j in range(len(varsx)):
        varx = varsx[j]
        varxname = parInfo.commandTexNames()[varx]

        a[varx] = {}
        b[varx] = {}
        s[varx] = {}
        for t in types:
            a[varx][t] = datadict["alphalogf%svs%s" % (t, varxname)]
            b[varx][t] = datadict["betalogf%svs%s" % (t, varxname)]
            s[varx][t] = datadict["scatlogf%svs%s" % (t, varxname)]

    for t in types:
        lines.append( " & $\\alpha$ & %s & %s & %s & %s & %s & %s \\\\" %
                     tuple([a[varsx[j]][t] for j in range(len(varsx))]))
        lines.append("%s & $\\beta$ & %s & %s & %s & %s & %s & %s \\\\" %
                     tuple([textype[t]] + [b[varsx[j]][t] for j in range(len(varsx))]))
        lines.append(" & $\\sigma_{\\rm int}$ & %s & %s & %s & %s & %s & %s" %
                     tuple([s[varsx[j]][t] for j in range(len(varsx))]))
        if t != types[-1]:
            lines[-1] += "\\\\\\hline"




    header = [
        "\\begin{deluxetable*}{llcccccc}",
        "\\tablecaption{Linear regression results}",
        "\\tablewidth{0pt}",
        "\\tablehead{",
        "\\colhead{$f$-type} &",
        "\\colhead{} &"
    ]
    for j in range(len(varsx)):
        varx = varsx[j]
        xtexname = parInfo.paramTexNames()[varx]
        xtexunits = parInfo.paramTexUnits()[varx]
        if x_logify[j]:
            xtexname = "$\\log_{10}(%s)$" % xtexname[1:-1]
        if xtexunits != "":
            xtexunits = "(%s)" % xtexunits

        header.append("\\colhead{%s %s}" % (xtexname, xtexunits))
        if j != len(varsx) - 1:
            header[-1] += " &"
    header.append("}")
    header.append("\\startdata")

    footer = [
        "\\enddata",
        "\\tablecomments{\\correlationtablecomment}.",
        "\\label{tab:table_correlations}",
        "\\end{deluxetable*}"
    ]

    lines = header+lines+footer
    with open(outfile, 'wb') as f:
        f.writelines('\n'.join(lines))