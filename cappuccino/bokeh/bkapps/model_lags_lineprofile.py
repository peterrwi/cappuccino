import numpy as np
import pandas as pd

from bokeh.plotting import figure
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, Slider, Button, Div
from bokeh.models import NumeralTickFormatter
from bokeh.io import curdoc


def rotate(x, y, C, S):
    return C*x + S*y, -S*x + C*y

def buildModel(randoms, params):
    rand1, rand2, phi, randorder = randoms['rand1'], randoms['rand2'], randoms['phi'], randoms['randorder']
    beta, mu, F = params['beta'], params['mu'], params['F']
    thetao, thetai, gamma = params['thetao'], params['thetai'], params['gamma']
    kappa, Mbh = params['kappa'], 10**params['logMbh']
    xi = params['xi']
    fellip, fflow = params['fellip'], params['fflow']
    
    Nparticles = len(phi)
    
    alpha = beta**-2.0
    rmin = mu * F
    theta = (mu - rmin) / alpha
    
    ################################################
    # Compute the gamma function
    # Using a shortcut here for computational purposes
    rmax = 500.
    xvals = np.linspace(0,rmax,500)
    
    # Compute the gamma distribution from 0 to rmax, then normalize
    gamma_distr = (xvals - rmin)**(alpha - 1.) * np.exp(-(xvals - rmin) / theta)
    gamma_distr = np.where(~np.isnan(gamma_distr), gamma_distr, 0.0)
    gamma_distr = np.where(gamma_distr > 0.0, gamma_distr, 0.0)
    gamma_distr = gamma_distr / np.sum(gamma_distr)
    
    # Compute the cdf
    cdf = [np.sum(gamma_distr[:i+1]) for i in range(len(gamma_distr))]
    
    # Take sorted distribution of values 0-1 to draw from cdf
    r = np.array([xvals[np.argmin(abs(cdf - val))] for val in np.linspace(0,1,Nparticles)])

    ################################################
    # Determine the per-particle opening angles
    part1 = np.cos(thetao * np.pi / 180.)
    part2 = 1. - np.cos(thetao * np.pi / 180.)
    part3 = np.exp(np.log(rand1) * gamma)
    angle = np.arccos(part1 + part2 * part3)
    
    # Pre-calculate some sines and cosines
    sin1 = np.sin(angle)
    cos1 = np.cos(angle)
    sin2 = np.sin(rand2 * 2.0 * np.pi)
    cos2 = np.cos(rand2 * 2.0 * np.pi)
    sin3 = np.sin(0.5 * np.pi - thetai * np.pi / 180.)
    cos3 = np.cos(0.5 * np.pi - thetai * np.pi / 180.)
    
    sinPhi = np.sin(phi)
    cosPhi = np.cos(phi)

    ################################################
    # Set the positions
    # Turn radial distribution into a disk
    x = r * cosPhi
    y = r * sinPhi
    z = np.zeros(len(r))
    
    # Puff up by opening angle
    x, z = rotate(x, z, cos1, sin1)
    # Restore axi-symmetry
    x, y = rotate(x, y, cos2, sin2)
    # Truth vector to determine if on far side of mid-plane
    far_side = np.where(z < 0.)
    # Rotate by inclination angle (+pi/2)
    x, z = rotate(x, z, cos3, sin3)

    ################################################
    # Compute the velocities
    G = 5.123619161  # ld / Msun * (km/s)^2

    # First put everything on a circular orbit
    vr = np.zeros(Nparticles)
    vphi = np.sqrt(G * Mbh / r)  # * sin(theta)*exp(radial_sd_orbiting*n2[i][j]);
    
    # Next turn (1-fellip) into inflow or outflow
    startindex = int(fellip*Nparticles)
    if fflow > 0:
        vr[randorder[startindex:]] = np.sqrt(2.0 * G * Mbh / r[randorder[startindex:]])
    else:
        vr[randorder[startindex:]] = -np.sqrt(2.0 * G * Mbh / r[randorder[startindex:]])
    vphi[randorder[startindex:]] = np.zeros(Nparticles-startindex)
    
    # Convert to cartesian coordinates
    vx = vr * cosPhi - vphi * sinPhi    
    vy = vr * sinPhi + vphi * cosPhi
    vz = np.zeros(len(r))
    
    # Puff up by opening angle
    vx, vz = rotate(vx, vz, cos1, sin1)
    # Restore axi-symmetry
    vx, vy = rotate(vx, vy, cos2, sin2)
    # Rotate by inclination angle (+pi/2)
    vx, vz = rotate(vx, vz, cos3, sin3)
    
    # Set so that moving away is positive velocity
    vx = -vx
    
    ################################################
    # Compute the weights
    size = 0.5 + kappa * x/r
    size = np.where(np.isnan(size),0.0,size)
    size *= 1.0 * len(size) / np.sum(size)
    size[far_side[0]] *= xi
    
    ################################################
    # Compute the lags
    lags = r - x  # days
    
    return x, y, z, vx, lags, size, xvals, gamma_distr

def continuum(times):
    offset = 2.0
    part1 = 1.5 * np.sin(times/60. * np.pi/2.)
    part2 = 1.0 * np.cos(times/18. * np.pi/2)
    part3 = 0.4 * np.sin(times/4. * np.pi/2)
    part4 = 0.1 * np.sin(times/1. * np.pi/2)
    return offset + part1 + part2 + part3 + part4

def makeIntegratedSpectrum(times, lags):
    integrated = np.zeros(len(times))
    for i in range(len(times)):
        for j in range(len(lags)):
            if np.isnan(lags[j]):
                integrated[i] += 0.0
            else:
                integrated[i] += continuum(times[i]-lags[j])
    integrated = 0.0 + 1.0 * integrated / len(lags)
    return integrated

def makeMeanSpectrum(vx, size, nbins=51, lims=[-10000,10000]):
    bins = np.linspace(lims[0],lims[1],nbins+1, endpoint=True)
    ll = np.zeros(len(bins)-1)
    for j in range(len(bins)-1):
        ll[j] = (bins[j] + bins[j+1])/2.
    spec = np.zeros(len(bins)-1)
    for i in range(len(vx)):
        j = 0
        while j >= 0:
            if j == len(bins)-1:
                j = -1
            elif vx[i] < bins[j+1]:
                spec[j] += 1.0*size[i]
                j = -1
            else:
                j += 1
    win = 2
    tmp = spec.copy()
    for i in range(win,len(spec)-win):
        spec[i] = np.mean(tmp[i-win:i+win+1])
    return ll, spec

def calcFWHM(x,y, _plotting=False):
    peakval = np.max(y)
    peakindex = np.argmax(y)
    minval = 0.0
    if _plotting:
        plt.figure()
        plt.plot(x,y)
    # From the left
    i = 0
    while i >= 0:
        if y[i] > minval + 0.5 * (peakval - minval):
            val1 = (x[i-1] + x[i])/2.
            if _plotting:
                plt.axvline(val1)
            i = -1
        else:
            i += 1
    i = peakindex
    while i >= 0:
        if y[i] < minval + 0.5 * (peakval - minval):
            val2 = (x[i] + x[i+1])/2.
            if _plotting:
                plt.axvline(val2)
            i = -1
        else:
            i -= 1
    left_val = (val1+val2)/2.
    if _plotting:
        plt.axvline(left_val)

    i = peakindex
    while i >= 0:
        if y[i] < minval + 0.5 * (peakval - minval):
            val1 = (x[i-1] + x[i])/2.
            if _plotting:
                plt.axvline(val1)
            i = -1
        else:
            i += 1
    i = len(x)-1
    while i >= 0:
        if y[i] > minval + 0.5 * (peakval - minval):
            val2 = (x[i] + x[i+1])/2.
            if _plotting:
                plt.axvline(val2)
            i = -1
        else:
            i -= 1
    right_val = (val1+val2)/2.
    if _plotting:
        plt.axvline(right_val)    
    return right_val - left_val

def calcLogVP(v,tau):
    return np.log10(v**2 * 3.0e8 * tau / 1.536e9)

def modify_doc(doc):
    # Set the data
    # Model
    Ndata = 1000
    phi = np.random.uniform(0,2.*np.pi,Ndata)
    rand1 = np.random.uniform(0,1,Ndata)
    rand2 = np.random.uniform(0,1,Ndata)
    randorder = np.arange(Ndata)
    np.random.shuffle(randorder)
    randoms = {'rand1':rand1, 'rand2':rand2, 'phi':phi, 'randorder':randorder}

    # Set the starting data
    start = {
        'thetai': 40.,
        'thetao': 30.,
        'kappa': -0.4,
        'beta': 0.85,
        'mu': 19.5,
        'F': 0.29,
        'logMbh': 7.9,
        'gamma': 3.0,
        'xi': 1.0,
        'fellip':1.0,
        'fflow':0.5,
    }
    params = {key: start[key] for key in start.keys()}

    x, y, z, vx, lags, size, gamma_x, gamma_y = buildModel(randoms, params)
    source_model = ColumnDataSource(data=dict(
        x=x, y=y, z=z, size=size, lags=lags, vx=vx,
        rand1=rand1,rand2=rand2,phi=phi,randorder=randorder
    ))

    r = np.sqrt(x**2 + y**2 + z**2)

    Ncont, Nspec, Nwave = 60, 40, 51
    times_cont = np.random.choice(100,Ncont, replace=False)
    cont = continuum(times_cont)
    times_spec = np.random.choice(100,Nspec, replace=False)
    spec_int = makeIntegratedSpectrum(times_spec, lags)
    ll, meanspec = makeMeanSpectrum(vx, size, nbins=Nwave, lims=[-10000,10000])
    
    #Ncont, Nspec, Nwave = np.ones(len(x)) * Ncont, np.ones(len(x)) * Nspec, np.ones(len(x)) * Nwave
    fwhm = calcFWHM(ll,meanspec)
    lag = np.nanmedian(lags)
    vp = calcLogVP(fwhm,lag)
    props = ColumnDataSource(data=dict(fwhm=[fwhm], lag=[lag], vp=[vp]))
    
    source_model = ColumnDataSource(data=dict(
        x=x, y=y, z=z, size=size, lags=lags, vx=vx,
        r=r, rand1=rand1,rand2=rand2,randorder=randorder,phi=phi,
    ))
    source_cont = ColumnDataSource(data=dict(times_cont=times_cont, cont=cont,))
    source_spec = ColumnDataSource(data=dict(spec_int=spec_int, times_spec=times_spec))
    source_meanspec = ColumnDataSource(data=dict(ll=ll, meanspec=meanspec))

    # Create the figures
    TOOLS = "pan,lasso_select,reset,wheel_zoom"
    panel_size = 250
    plot_edge = figure(x_range=(-50, 50), y_range=(-50, 50), plot_width=panel_size, plot_height=panel_size, toolbar_location=None, title="Edge on view (observer at x=inf)")
    plot_face = figure(x_range=(-50, 50), y_range=(-50, 50), plot_width=panel_size, plot_height=panel_size, toolbar_location=None, title="Face on view")
    plot_lc = figure(x_range=(0, 100), y_range=(0,5), plot_width=2*panel_size, plot_height=panel_size, toolbar_location=None, title="Light curves")
    plot_mean = figure(x_range=(-10000,10000), y_range=(0,4.*int(Ndata/(Nwave-1))), plot_width=panel_size, plot_height=panel_size, toolbar_location=None, title="Emission line profile")
    plot_tf = figure(x_range=(-10000, 10000), y_range=(0, 100), plot_width=panel_size, plot_height=panel_size, title="Transfer function", tools=TOOLS, toolbar_location=None)


    # Axis labels
    plot_edge.xaxis.axis_label = "x (light days)"
    plot_edge.yaxis.axis_label = "z (light days)"
    plot_face.xaxis.axis_label = "y (light days)"
    plot_face.yaxis.axis_label = "z (light days)"
    plot_lc.xaxis.axis_label = "Time (days)"
    plot_lc.yaxis.axis_label = "Flux"
    plot_mean.xaxis.axis_label = "Velocity (km/s)"
    plot_mean.yaxis.axis_label = "Flux"
    plot_tf.xaxis.axis_label = "Velocity (km/s)"
    plot_tf.yaxis.axis_label = "Lag (light days)"
    
    # Plot the data
    plot_edge.scatter('x', 'z', size='size', source=source_model)
    plot_face.scatter('y', 'z', size='size', source=source_model)
    plot_lc.scatter('times_cont', 'cont', size=3, source=source_cont, legend_label="Continuum")
    plot_lc.scatter('times_spec', 'spec_int', size=3,  source=source_spec, fill_color='orange', line_color='orange', legend_label="Emission line")
    plot_mean.line('ll', 'meanspec',  source=source_meanspec)
    plot_tf.scatter('vx', 'lags', size='size', source=source_model)
    
    # Some useful measurements on the line
    text_fwhm = Div(text="FWHM = %.0f km/s" % (props.data['fwhm'][0]))
    text_lag = Div(text="Lag = %.1f days" % (props.data['lag'][0]))
    text_vp = Div(text="log10(VP/M_sun) = %.1f" % (props.data['vp'][0]))
    
    plot_lc.legend.location = 'bottom_right'
    plot_lc.legend.label_text_font_size = '8pt'
    plot_lc.legend.margin = 4
    plot_lc.legend.padding = 4
    plot_lc.legend.background_fill_alpha = 0.75

    def callback_sliders(attr, old, new):
        for key in sliders.keys():
            params[key] = sliders[key].value
        x, y, z, vx, lags, size, gamma_x, gamma_y = buildModel(randoms, params)
        ll, meanspec = makeMeanSpectrum(vx, size, nbins=Nwave, lims=[-10000,10000])
        spec_int = makeIntegratedSpectrum(times_spec, lags)

        new_model_data = dict(x=x,y=y,z=z,vx=vx,lags=lags,size=size)
        new_meanspec_data = dict(ll=ll, meanspec=meanspec)
        new_specint_data = dict(times_spec=times_spec, spec_int=spec_int)
        source_model.data = new_model_data
        source_meanspec.data = new_meanspec_data
        source_spec.data = new_specint_data

    def callback_update():
        for key in sliders.keys():
            params[key] = sliders[key].value
        x, y, z, vx, lags, size, gamma_x, gamma_y = buildModel(randoms, params)
        ll, meanspec = makeMeanSpectrum(vx, size, nbins=Nwave, lims=[-10000,10000])
        spec_int = makeIntegratedSpectrum(times_spec, lags)

        new_model_data = dict(x=x,y=y,z=z,vx=vx,lags=lags,size=size)
        new_meanspec_data = dict(ll=ll, meanspec=meanspec)
        new_specint_data = dict(times_spec=times_spec, spec_int=spec_int)
        source_model.data = new_model_data
        source_meanspec.data = new_meanspec_data
        source_spec.data = new_specint_data
    
    def callback_reset():
        for key in sliders.keys():
            sliders[key].value = start[key]
        x, y, z, vx, lags, size, gamma_x, gamma_y = buildModel(randoms, start)
        new_model_data = dict(x=x,y=y,z=z,vx=vx,lags=lags,size=size)
        source_model.data = new_model_data
        
    sliders = {
        "thetai": Slider(start=0.0, end=90., value=start['thetai'], step=1, title="Inclination angle (deg)"),
        "thetao": Slider(start=0.0, end=90., value=start['thetao'], step=1, title="Opening angle (deg)"),
        "gamma": Slider(start=1.0, end=5.0, value=start['gamma'], step=0.1, title="Gamma"),
        "kappa": Slider(start=-0.5, end=0.5, value=start['kappa'], step=0.05, title="Kappa"),
        "xi": Slider(start=0.0, end=1.0, value=start['xi'], step=0.01, title="Slab transparency"),
        "beta": Slider(start=0.1, end=2.0, value=start['beta'], step=0.05, title="Beta"),
        "mu": Slider(start=7.5, end=72.5, value=start['mu'], step=1, title="Mu (light days)"),
        "F": Slider(start=0.125, end=0.825, value=start['F'], step=0.01, title="F"),
        "fellip": Slider(start=0, end=1, value=start['fellip'], step=0.01, title="Circular-like fraction"),
        "fflow": Slider(start=-0.5, end=0.5, value=start['fflow'], step=1, title="Inflow-Outflow"),
        "logMbh": Slider(start=6.5, end=8.5, value=start['logMbh'], step=0.1, title="log10(Mbh/Msun)"),
    }
    update_button = Button(label='Update')
    reset_button = Button(label='Reset')

    #for key in sliders.keys():
    #    sliders[key].on_change("value", callback_sliders)        
    update_button.on_click(callback_update)
    reset_button.on_click(callback_reset)
        
    layout = row(
        column(*[sliders[key] for key in sliders.keys()], update_button, reset_button),
        column(
            row(plot_edge,plot_face),
            row(plot_lc)
        ),
        column(plot_mean, plot_tf,text_fwhm, text_lag, text_vp)
    )
        
    # add the layout to curdoc
    doc.add_root(layout)


modify_doc(curdoc())