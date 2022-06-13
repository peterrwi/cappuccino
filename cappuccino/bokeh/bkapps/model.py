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
    circ_vel = np.sqrt(G * Mbh / r)
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

def modify_doc(doc):
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
    
    x, y, z, vx, lags, size, gamma_x, gamma_y = buildModel(randoms, start)
    source_model = ColumnDataSource(data=dict(
        x=x, y=y, z=z, size=size, lags=lags, vx=vx,
        rand1=rand1,rand2=rand2,phi=phi,randorder=randorder
    ))
    source_gamma = ColumnDataSource(data=dict(gamma_x=gamma_x,gamma_y=gamma_y))

    # Create the figure panels
    TOOLS = "pan,lasso_select,reset,wheel_zoom"
    p_edge = figure(x_range=(-50, 50), y_range=(-50, 50), plot_width=300, plot_height=300, title="Edge on view (observer at x=infinity)", tools=TOOLS, toolbar_location=None)
    p_face = figure(x_range=(-50, 50), y_range=(-50, 50), plot_width=300, plot_height=300, title="Face on view", tools=TOOLS, toolbar_location=None)
    p_tf = figure(x_range=(-15000, 15000), y_range=(0, 100), plot_width=300, plot_height=300, title="Transfer function", tools=TOOLS, toolbar_location=None)
    p_gamma = figure(x_range=(0, 100), plot_width=300, plot_height=300, title="Radial distribution", tools=TOOLS, toolbar_location=None)

    p_edge.xaxis.axis_label = "x (light days)"
    p_edge.yaxis.axis_label = "z (light days)"
    p_face.xaxis.axis_label = "y (light days)"
    p_face.yaxis.axis_label = "z (light days)"
    p_tf.xaxis.axis_label = "Velocity (km/s)"
    p_tf.yaxis.axis_label = "Lag (light days)"
    p_gamma.xaxis.axis_label = "r (light days)"
    p_gamma.yaxis.axis_label = "P(r)"
    p_gamma.yaxis.major_label_text_font_size = '0pt'

    p_edge.scatter('x', 'z', size='size', source=source_model)
    p_face.scatter('y', 'z', size='size', source=source_model)
    p_tf.scatter('vx', 'lags', size='size', source=source_model)
    p_gamma.scatter('gamma_x', 'gamma_y', source=source_gamma)
    
    # Create the slider widgets
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
    reset_button = Button(label='Reset')

    # Set the code to update the data when the sliders are change
    def callback_sliders(attr, old, new):
        params = {key: sliders[key].value for key in sliders.keys()}
        x, y, z, vx, lags, size, gamma_x, gamma_y = buildModel(randoms, params)
        new_model_data = dict(x=x,y=y,z=z,vx=vx,lags=lags,size=size)
        new_gamma_data = dict(gamma_x=gamma_x,gamma_y=gamma_y)
        source_model.data = new_model_data
        source_gamma.data = new_gamma_data
    
    def callback_reset():
        for key in sliders.keys():
            sliders[key].value = start[key]
        x, y, z, vx, lags, size, gamma_x, gamma_y = buildModel(randoms, start)
        new_model_data = dict(x=x,y=y,z=z,vx=vx,lags=lags,size=size)
        new_gamma_data = dict(gamma_x=gamma_x,gamma_y=gamma_y)
        source_model.data = new_model_data
        source_gamma.data = new_gamma_data

    for key in sliders.keys():
        sliders[key].on_change("value", callback_sliders)
    reset_button.on_click(callback_reset)
    # Set the layout with the sliders and plot
    layout = row(
        column(*[sliders[key] for key in sliders.keys()], reset_button),
        gridplot(
            [[p_edge,p_face],
             [p_gamma, p_tf]]
        )
    )

    # add the layout to curdoc
    doc.add_root(layout)

modify_doc(curdoc())