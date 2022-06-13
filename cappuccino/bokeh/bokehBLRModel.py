import numpy as np
import pandas as pd

from bokeh.plotting import show, figure, output_notebook, reset_output, output_file
from bokeh.layouts import gridplot, row, column
from bokeh.models import ColumnDataSource, Span, Slider, Button, CustomJS
from bokeh.server.server import Server

G = 5.123619161  # ld / Msun * (km/s)^2

def drawFromGamma(beta, mu, F, Nclouds=1000, rmax=500., Nsamp=1000):
    alpha = beta**-2.0
    rmin = mu * F
    theta = (mu - rmin) / alpha

    # Compute the gamma function
    gamma_x = np.linspace(0,rmax,Nsamp,endpoint=False)

    # Compute the gamma distribution from 0 to rmax, then normalize
    cdf = np.zeros(Nsamp)
    sumvals = 0.
    for i in range(Nsamp):
        cdf[i] = (gamma_x[i] - rmin)**(alpha-1.) * np.exp(-(gamma_x[i]-rmin) / theta)
        if np.isnan(cdf[i]):
            cdf[i] = 0.
        elif cdf[i] < 0.0:
            cdf[i] = 0.
        sumvals += cdf[i]

    # Compute the cdf
    gamma_y = np.zeros(Nsamp)
    for i in range(Nsamp):
        cdf[i] /= sumvals
        gamma_y[i] = cdf[i]
        if i > 0:
            cdf[i] += cdf[i-1]
    
    # Take sorted distribution of values 0-1 to draw from cdf
    tmp = np.linspace(0,1,Nclouds,endpoint=False)
    j = 0
    r = np.zeros(Nclouds)
    for i in range(Nclouds):
        run_loop = True
        while run_loop:
            if j >= len(cdf):
                r[i] = gamma_x[-1]
                run_loop = False
            elif cdf[j] > tmp[i]:
                r[i] = gamma_x[j]
                run_loop = False
            else:
                j += 1
    return r, gamma_x, gamma_y

def rotate(x, y, C, S):
    if len(np.shape(x)) > 1:
        C = np.atleast_2d(C).T
        S = np.atleast_2d(S).T
    return C*x + S*y, -S*x + C*y

def rotateVecAroundAxis(axis, theta, vec):
    axis = np.array(axis)
    u = axis / np.linalg.norm(axis)
    
    rot_matrix = np.array([
        [
            np.cos(theta) + u[0]**2 * (1.0 - np.cos(theta)),
            u[0] * u[1] * (1.0 - np.cos(theta)) - u[2] * np.sin(theta),
            u[0] * u[2] * (1.0 - np.cos(theta)) + u[1] * np.sin(theta)
        ],
        [
            u[1] * u[0] * (1.0 - np.cos(theta)) + u[2] * np.sin(theta),
            np.cos(theta) + u[1]**2 * (1.0 - np.cos(theta)),
            u[1] * u[2] * (1.0 - np.cos(theta)) - u[0] * np.sin(theta)
        ],
        [
            u[2] * u[0] * (1.0 - np.cos(theta)) - u[1] * np.sin(theta),
            u[2] * u[1] * (1.0 - np.cos(theta)) + u[0] * np.sin(theta),
            np.cos(theta) + u[2]**2 * (1.0 - np.cos(theta))
        ]
    ])
    
    return np.dot(rot_matrix, vec)

def getAngMom0(x):
    if x[2] == 0:
        ang_mom = np.array([0,0,1])
    elif x[2] > 0:
        ang_mom = np.array([-x[0], -x[1], (x[0]**2 + x[1]**2)/x[2]])
        ang_mom = ang_mom / np.linalg.norm(ang_mom)
    else:
        ang_mom = np.array([x[0], x[1], -(x[0]**2 + x[1]**2)/x[2]])
        ang_mom = ang_mom / np.linalg.norm(ang_mom)
    return ang_mom

def drawAngMom(ang_mom_0, thetao, x):
    theta_max = thetao * 1.0
    while True:
        theta = np.random.uniform(-theta_max,theta_max)
        ang_mom = rotateVecAroundAxis(x, theta, ang_mom_0)
        if np.arccos(ang_mom[2]) < thetao:
            return ang_mom, theta
        else:
            theta_max = theta

def getVelocityUnitVector(x, l):
    x, l = np.array(x), np.array(l)
    x = x / np.linalg.norm(x)
    l = l / np.linalg.norm(l)
    return np.cross(l, x)


class BLRModel():
    def __init__(self, Nclouds=1000, Nvpercloud=1, params={}, randoms={}):
        self.Nclouds = Nclouds
        self.Nvpercloud = Nvpercloud

        # Params
        default_params = {
            'beta': 0.4,
            'F': 0.2,
            'mu': 20.0,
            'thetai': 45.0,
            'thetao': 40.0,
            'kappa': -0.45,
            'rmax': 100,
            'logMbh': 7.5,
            'gamma': 1.0,
            'xi': 0.3,
            'fflow': 1.0,
            'fellip': 0.85,
            'ellipseAngle': 30,
            'angular_sd_orbiting': 0.03,
            'radial_sd_orbiting': 0.02,
            'angular_sd_flowing': 0.03,
            'radial_sd_flowing': 0.01,
            'logturbulence': -1.5,
            'Cadd': 0.0,
            'Cmult': 1.0
        }
        for par in default_params.keys():
            if par not in params.keys():
                params[par] = default_params[par]
        self.params = params

        # Randoms
        rand0 = np.random.choice((-1.0,1.0),Nclouds)
        rand1 = np.random.uniform(0,1,Nclouds)
        rand2 = np.random.uniform(-1,1,Nclouds)
        randnorm1 = np.random.normal(0,1,Nclouds)
        randnorm2 = np.random.normal(0,1,Nclouds)
        randnorm3 = np.random.normal(0,1,(Nclouds,Nvpercloud))
        randnorm4 = np.random.normal(0,1,(Nclouds,Nvpercloud))
        randnorm5 = np.random.normal(0,1,(Nclouds,Nvpercloud))
        randorder = np.vstack([np.arange(Nclouds)]*Nvpercloud)
        for i in range(Nvpercloud):
            np.random.shuffle(randorder[i])
        np.random.shuffle(randorder)        
        default_rands = {
            'rand0':rand0, 'rand1':rand1, 'rand2':rand2, 'randorder':randorder,
            'randnorm1': randnorm1, 'randnorm2': randnorm2, 'randnorm3': randnorm3,
            'randnorm4': randnorm4, 'randnorm5': randnorm5,
        }
        for rand in default_rands.keys():
            if rand not in randoms.keys():
                randoms[rand] = default_rands[rand]
        self.randoms = randoms

        self.buildModel(params, randoms, Nclouds=Nclouds, Nvpercloud=Nvpercloud)
    
    def drawFromGamma(self, beta, mu, F, Nclouds=1000, rmax=500., Nsamp=1000):
        alpha = beta**-2.0
        rmin = mu * F
        theta = (mu - rmin) / alpha

        # Compute the gamma function
        gamma_x = np.linspace(0,rmax,Nsamp,endpoint=False)

        # Compute the gamma distribution from 0 to rmax, then normalize
        cdf = np.zeros(Nsamp)
        sumvals = 0.
        for i in range(Nsamp):
            cdf[i] = (gamma_x[i] - rmin)**(alpha-1.) * np.exp(-(gamma_x[i]-rmin) / theta)
            if np.isnan(cdf[i]):
                cdf[i] = 0.
            elif cdf[i] < 0.0:
                cdf[i] = 0.
            sumvals += cdf[i]

        # Compute the cdf
        gamma_y = np.zeros(Nsamp)
        for i in range(Nsamp):
            cdf[i] /= sumvals
            gamma_y[i] = cdf[i]
            if i > 0:
                cdf[i] += cdf[i-1]
        
        # Take sorted distribution of values 0-1 to draw from cdf
        tmp = np.linspace(0,1,Nclouds,endpoint=False)
        j = 0
        r = np.zeros(Nclouds)
        for i in range(Nclouds):
            #j=0
            run_loop = True
            while run_loop:
                if j >= len(cdf):
                    r[i] = gamma_x[-1]
                    run_loop = False
                elif cdf[j] > tmp[i]:
                    r[i] = gamma_x[j]
                    run_loop = False
                else:
                    j += 1
        return r, gamma_x, gamma_y

    def get_vr_vphi(self, Nclouds, Nvpercloud, randoms, params):
        randorder, randnorm1, randnorm2 = randoms['randorder'], randoms['randnorm1'], randoms['randnorm2']
        vr, vphi = np.zeros((Nclouds, Nvpercloud)), np.zeros((Nclouds, Nvpercloud))
        angular_sd_orbiting = params['angular_sd_orbiting']
        radial_sd_orbiting = params['radial_sd_orbiting']
        angular_sd_flowing = params['angular_sd_flowing']
        radial_sd_flowing = params['radial_sd_flowing']
        ellipseAngle = params['ellipseAngle']
        fflow = params['fflow']
        fellip = params['fellip']
        
        startindex = int(fellip*Nclouds)
        _theta = np.empty(Nclouds)

        for j in range(Nvpercloud):
            _circ = randorder[j][:startindex]
            _flow = randorder[j][startindex:]
            _theta[_circ] = 0.5*np.pi + angular_sd_orbiting*randnorm1[_circ]

            vr[_circ,j] = np.sqrt(2.0) * np.cos(_theta[_circ]) * np.exp(radial_sd_orbiting*randnorm2[_circ])    
            vphi[_circ,j] = np.sin(_theta[_circ]) * np.exp(radial_sd_orbiting*randnorm2[_circ])
            
            # Inflow / outflow
            if fflow < 0.5:
                _theta[_flow] = np.pi - ellipseAngle*np.pi/180. + angular_sd_flowing*randnorm1[_flow]
                vr[_flow,j] = np.sqrt(2.0) * np.cos(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow])
                vphi[_flow,j] = np.sin(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow])
            else:
                _theta[_flow] = 0.0 + ellipseAngle*np.pi/180. + angular_sd_flowing*randnorm1[_flow]
                vr[_flow,j] = np.sqrt(2.0) * np.cos(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow])
                vphi[_flow,j] = np.sin(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow])
        
        return vr, vphi

    def getXYZ(self, r, randoms, params):
        # Determine the per-particle opening angles    
        part1 = np.sin(params['thetao']*np.pi/180.)
        part2 = randoms['rand1']**(1.0/params['gamma'])
        angle = np.arcsin(part1 * part2)

        # Pre-calculate some sines and cosines
        sin1 = np.sin(angle)
        cos1 = np.cos(angle)
        sin2 = np.sin(randoms['rand2'] * np.pi)
        cos2 = np.cos(randoms['rand2'] * np.pi)
        sin3 = np.sin(0.5 * np.pi - params['thetai'] * np.pi / 180.)
        cos3 = np.cos(0.5 * np.pi - params['thetai'] * np.pi / 180.)

        # Set the positions
        x = r * randoms['rand0']
        y = np.zeros(len(r))
        z = np.zeros(len(r))

        # Puff up by opening angle into wedge
        x, z = rotate(x, z, cos1, sin1)
        # Rotate into thick disk
        x, y = rotate(x, y, cos2, sin2)

        self.part1, self.part2, self.angle = part1, part2, angle
        self.sin1, self.cos1 = sin1, cos1
        self.sin2, self.cos2 = sin2, cos2
        self.sin3, self.cos3 = sin3, cos3

        return x, y, z

    def getVXYZ(self, r, vr, vphi, pos_unit, v_unit, params, randoms):
        Mbh = 10**params['logMbh']
        vcirc = np.sqrt(G * Mbh / r)
        Nclouds, Nvpercloud = self.Nclouds, self.Nvpercloud
        vx, vy, vz = np.zeros((Nclouds, Nvpercloud)), np.zeros((Nclouds, Nvpercloud)), np.zeros((Nclouds, Nvpercloud))
        for i in range(Nclouds):
            for j in range(Nvpercloud):
                vx[i,j] = (vr[i,j] * pos_unit[i][0] + vphi[i,j] * v_unit[i,j][0]) * vcirc[i]
                vy[i,j] = (vr[i,j] * pos_unit[i][1] + vphi[i,j] * v_unit[i,j][1]) * vcirc[i]
                vz[i,j] = (vr[i,j] * pos_unit[i][2] + vphi[i,j] * v_unit[i,j][2]) * vcirc[i]

        # Rotate by inclination angle (+pi/2)
        #sin3 = np.sin(0.5 * np.pi - params['thetai'] * np.pi / 180.)
        #cos3 = np.cos(0.5 * np.pi - params['thetai'] * np.pi / 180.)
        #vx, vz = rotate(vx, vz, cos3, sin3)

        # Add in turbulence
        #vx += params['turbulence'] * np.atleast_2d(vcirc).T * randoms['randnorm3']
        #vy += params['turbulence'] * np.atleast_2d(vcirc).T * randoms['randnorm4']
        #vz += params['turbulence'] * np.atleast_2d(vcirc).T * randoms['randnorm5']
    
        return vx,vy,vz,vcirc

    def buildModel(self, params, randoms, Nclouds=1000, Nvpercloud=1):
        # Set the parameters and random numbers
        rand0, rand1, rand2, randorder = randoms['rand0'], randoms['rand1'], randoms['rand2'], randoms['randorder']
        randnorm1, randnorm2, randnorm3 = randoms['randnorm1'], randoms['randnorm2'], randoms['randnorm3']
        randnorm4, randnorm5 = randoms['randnorm4'], randoms['randnorm5']

        Mbh = 10**params['logMbh']
        beta, mu, F, rmax = params['beta'], params['mu'], params['F'], params['rmax']
        thetao, thetai = params['thetao'], params['thetai']
        gamma, kappa, xi = params['gamma'], params['kappa'], params['xi']
        fflow, fellip = params['fflow'], params['fellip']
        angular_sd_orbiting, angular_sd_flowing = params['angular_sd_orbiting'], params['angular_sd_flowing']
        radial_sd_orbiting, radial_sd_flowing = params['radial_sd_orbiting'], params['radial_sd_flowing']
        ellipseAngle = params['ellipseAngle']
        turbulence = 10**params['logturbulence']
        
        ###########################################################################################
        ######################################  Geometry  #########################################
        ###########################################################################################
        r, gamma_x, gamma_y = self.drawFromGamma(beta, mu, F, rmax=rmax, Nclouds=Nclouds)
        x, y, z = self.getXYZ(r, randoms, params)
        # Truth vector to determine if on far side of mid-plane
        far_side = np.where(z < 0.)

        ###########################################################################################
        #####################################  Kinematics  ########################################
        ###########################################################################################
        ang_mom_0 = np.zeros((Nclouds,3))
        pos_unit = np.zeros((Nclouds,3))
        for i in range(Nclouds):
            pos_unit[i] = np.array([x[i],y[i],z[i]])
            pos_unit[i] = pos_unit[i] / np.linalg.norm(pos_unit[i])
            ang_mom_0[i] = getAngMom0(pos_unit[i])
        
        vr, vphi = self.get_vr_vphi(Nclouds, Nvpercloud, randoms, params)
        
        theta_rot = np.zeros((Nclouds, Nvpercloud))
        v_unit = np.zeros((Nclouds, Nvpercloud, 3))
        for i in range(Nclouds):
            for j in range(Nvpercloud):
                ang_mom, theta_rot[i,j] = drawAngMom(ang_mom_0[i], thetao * np.pi/180., pos_unit[i])
                v_unit[i,j] = np.cross(ang_mom, pos_unit[i])

        vx, vy, vz, vcirc = self.getVXYZ(r, vr, vphi, pos_unit, v_unit, params, randoms)

        # Rotate by inclination angle (+pi/2)
        sin3 = np.sin(0.5 * np.pi - thetai * np.pi / 180.)
        cos3 = np.cos(0.5 * np.pi - thetai * np.pi / 180.)
        x, z = rotate(x, z, cos3, sin3)
        # Rotate by inclination angle (+pi/2)
        vx, vz = rotate(vx, vz, cos3, sin3)
        #pos_unit[:,0], pos_unit[:,2] = rotate(pos_unit[:,0], pos_unit[:,2], cos3, sin3)
        #v_unit[:,:,0], v_unit[:,:,2] = rotate(v_unit[:,:,0], v_unit[:,:,2], cos3, sin3)

        # Add in turbulence
        vx += turbulence * np.atleast_2d(vcirc).T * randnorm3
        vy += turbulence * np.atleast_2d(vcirc).T * randnorm4
        vz += turbulence * np.atleast_2d(vcirc).T * randnorm5
        
        ###########################################################################################
        ##################################  Weights and Lags  #####################################
        ###########################################################################################

        # Compute the weights
        size = 0.5 + kappa * x/r
        size /= np.sum(size)
        size[far_side] *= xi
        size *= Nclouds*1.5

        # Compute the lags
        lags = r - x

        far_side_tf = []
        for i in range(Nclouds):
            if i in far_side[0]:
                far_side_tf.append(1)
            else:
                far_side_tf.append(0)

        ###########################################################################################
        ##################################  Mean line profile  ####################################
        ###########################################################################################
        
        start = -10000
        end = 10000
        step = 500
        bins = np.arange(start,end+step,step)
        bin_edges = np.arange(start-step/2, end+3.*step/2, step)
        eline = np.zeros(len(bin_edges)-1)
        for i in range(Nclouds):
            for j in range(Nvpercloud):
                if vx[i,j] < bin_edges[0]:
                    break
                if vx[i,j] > bin_edges[-1]:
                    break
                for k in range(len(bins)):
                    if ((vx[i,j] > bin_edges[k]) and (vx[i,j] < bin_edges[k+1])):
                        eline[k] += size[i] / step / Nclouds / Nvpercloud * 2500.

        mod = {
            'x': x, 'y': y, 'z': z,
            'vx': vx, 'vy': vy, 'vz': vz,
            'vr': vr, 'vphi': vphi, 'theta_rot': theta_rot,
            'pos_unit': pos_unit, 'v_unit': v_unit, 'vcirc': vcirc,
            'lags': lags, 'size': size, 'r': r,
            'far_side': far_side_tf
        }
        intermediate = {
            'sin1': self.sin1, 'sin2': self.sin2, 'sin3': self.sin3,
            'cos1': self.cos1, 'cos2': self.cos2, 'cos3': self.cos3,
            'part1': self.part1, 'part2': self.part2, 'angle': self.angle
        }

        mod['x_unit'] = mod['pos_unit'][:,0]
        mod['y_unit'] = mod['pos_unit'][:,1]
        mod['z_unit'] = mod['pos_unit'][:,2]
        mod['vx_unit'] = mod['v_unit'][:,:,0]
        mod['vy_unit'] = mod['v_unit'][:,:,1]
        mod['vz_unit'] = mod['v_unit'][:,:,2]

        self.mod = mod
        self.clouds = {key: mod[key] for key in [
            'x', 'y', 'z', 'r', 
            'size', 'lags', 'far_side',
            'x_unit', 'y_unit', 'z_unit', 'vcirc'
        ]}
        self.vel = {
            'vx': mod['vx'].ravel(), 'vy': mod['vy'].ravel(), 'vz': mod['vz'].ravel(),
            'vx_unit': mod['vx_unit'].ravel(), 'vy_unit': mod['vy_unit'].ravel(), 'vz_unit': mod['vz_unit'].ravel(),
            'lags': np.repeat(mod['lags'],self.Nvpercloud), 'size': np.repeat(mod['size'],self.Nvpercloud),
            'vr': mod['vr'].ravel(), 'vphi': mod['vphi'].ravel(), 'theta_rot': mod['theta_rot'].ravel()
        }
        self.gamma = {'x': gamma_x, 'y': gamma_y}
        self.intermediate = intermediate
        self.eline = {'vel':bins, 'flux':eline}


class BKBLRModel():
    def __init__(self, Nclouds=1000, Nvpercloud=1, randoms={}, start={}, output='notebook', port=5006, notebook_url="http://localhost:8888", **kwargs):
        self.Nclouds = Nclouds
        self.Nvpercloud = Nvpercloud

        # Params
        default_params = {
            'beta': 0.4,
            'F': 0.2,
            'mu': 20.0,
            'thetai': 45.0,
            'thetao': 40.0,
            'kappa': -0.45,
            'rmax': 100,
            'logMbh': 7.5,
            'gamma': 1.0,
            'xi': 0.3,
            'fflow': 1.0,
            'fellip': 0.85,
            'ellipseAngle': 30,
            'angular_sd_orbiting': 0.03,
            'radial_sd_orbiting': 0.02,
            'angular_sd_flowing': 0.03,
            'radial_sd_flowing': 0.01,
            'logturbulence': -1.5,
            'Cadd': 0.0,
            'Cmult': 1.0
        }
        for par in default_params.keys():
            if par not in start.keys():
                start[par] = default_params[par]
        self.start = start

        # Randoms
        rand0 = np.random.choice((-1.0,1.0),Nclouds)
        rand1 = np.random.uniform(0,1,Nclouds)
        rand2 = np.random.uniform(-1,1,Nclouds)
        randnorm1 = np.random.normal(0,1,Nclouds)
        randnorm2 = np.random.normal(0,1,Nclouds)
        randnorm3 = np.random.normal(0,1,(Nclouds,Nvpercloud))
        randnorm4 = np.random.normal(0,1,(Nclouds,Nvpercloud))
        randnorm5 = np.random.normal(0,1,(Nclouds,Nvpercloud))
        randorder = np.vstack([np.arange(Nclouds)]*Nvpercloud)
        for i in range(Nvpercloud):
            np.random.shuffle(randorder[i])
        np.random.shuffle(randorder)        
        default_rands = {
            'rand0':rand0, 'rand1':rand1, 'rand2':rand2, 'randorder':randorder,
            'randnorm1': randnorm1, 'randnorm2': randnorm2, 'randnorm3': randnorm3,
            'randnorm4': randnorm4, 'randnorm5': randnorm5,
        }
        for rand in default_rands.keys():
            if rand not in randoms.keys():
                randoms[rand] = default_rands[rand]
        randoms['randnorm3_flat'] = randoms['randnorm3'].flatten()
        randoms['randnorm4_flat'] = randoms['randnorm4'].flatten()
        randoms['randnorm5_flat'] = randoms['randnorm5'].flatten()
        self.randoms = randoms
        self.output = output

        if output == 'notebook':
            reset_output()
            output_notebook()
            show(self.modify_doc, notebook_url=notebook_url)
        elif output == 'server':
            reset_output()
            server = Server({'/': self.modify_doc}, port=port)
            server.start()
            try:
                server.run_until_shutdown()
            except:
                print("Server already running")
            self.server = server
        else:
            reset_output()
            output_file(output)
            self.modify_doc('')
    
    def callback_geo(self, attr, old, new):
        params = {key: self.start[key] for key in self.start.keys()}
        for key in self.sliders.keys():
            params[key] = self.sliders[key].value

        r, gamma_x, gamma_y = drawFromGamma(params['beta'], params['mu'], params['F'], rmax=params['rmax'], Nclouds=self.Nclouds)
        x, y, z = getPositions(self.Nclouds, r, self.randoms, params)
        # Truth vector to determine if on far side of mid-plane
        far_side = np.where(z < 0.)

        # Rotate by inclination angle (+pi/2)
        sin3 = np.sin(0.5 * np.pi - params['thetai'] * np.pi / 180.)
        cos3 = np.cos(0.5 * np.pi - params['thetai'] * np.pi / 180.)
        x, z = rotate(x, z, cos3, sin3)

        # Compute the weights
        size = 0.5 + params['kappa'] * x/r
        size[far_side] *= params['xi']
        size /= np.sum(size)

        # Compute the lags
        lags = r - x

        new_data = dict(self.source.data)
        new_data_gamma = dict(self.source_gamma.data)
        new_data['x'] = x
        new_data['y'] = y
        new_data['z'] = z
        new_data['lags'] = lags
        new_data['size'] = size*self.Nclouds
        new_data_gamma['x'] = gamma_x
        new_data_gamma['y'] = gamma_y
        self.source.data = new_data
        self.source_gamma.data = new_data_gamma

    def callback_vel_sd(self, attr, old, new):
        params = {key: self.start[key] for key in self.start.keys()}
        for key in self.sliders.keys():
            params[key] = self.sliders[key].value
        vr, vphi = get_vr_vphi(
            Nclouds = self.Nclouds, 
            Nvpercloud = self.Nvpercloud,
            randoms = [self.randoms['randorder'],self.randoms['randnorm1'],self.randoms['randnorm2']],
            params = params
        )
        vx, vy, vz = getVelocities(
            self.Nclouds, self.Nvpercloud,
            self.mod['r'], vr, vphi,
            self.mod['pos_unit'], self.mod['v_unit'],
            params = params,
            randoms = self.randoms
        )

        #Nclouds, Nvpercloud, r, vr, vphi, pos_unit, v_unit, params, randoms

        #vx,vy,vz = get_vxyz(self.Nclouds, self.Nvpercloud, vr, vphi, pos, v, vcirc)

        new_data = dict(self.source_vel.data)
        new_data['vr'] = vr.ravel()
        new_data['vphi'] = vphi.ravel()
        new_data['vx'] = vx.ravel()
        new_data['vy'] = vy.ravel()
        new_data['vz'] = vz.ravel()
        self.source_vel.data = new_data

    def modify_doc(self, doc):
        clouds = {}
        gamma_distr = {}
        r, gamma_x, gamma_y = drawFromGamma(
            self.start['beta'], self.start['mu'], self.start['F'],
            Nclouds=self.Nclouds, rmax=self.start['rmax']
        )
        #clouds['r'] = buildModel(params, randoms, Nclouds=1000, Nvpercloud=1)
        mod = BLRModel(params=self.start, randoms=self.randoms, Nclouds=self.Nclouds, Nvpercloud=self.Nvpercloud)
        self.mod = mod

        self.source = ColumnDataSource(mod.clouds)
        self.source_gamma = ColumnDataSource(mod.gamma)
        self.source_vel = ColumnDataSource(mod.vel)
        self.params = {key: val for key, val in self.start.items()}
        self.params['Nvpercloud'] = self.Nvpercloud
        self.params['Nclouds'] = self.Nclouds
        self.intermediate = mod.intermediate
        self.source_eline = ColumnDataSource(mod.eline)

        plot_edge = figure(x_range=(-50, 50), y_range=(-50, 50), plot_width=300, plot_height=300, toolbar_location=None, output_backend="webgl")
        plot_face = figure(x_range=(-50, 50), y_range=(-50, 50), plot_width=300, plot_height=300, toolbar_location=None, output_backend="webgl")
        plot_tf = figure(x_range=(-10000, 10000), y_range=(0, 100), plot_width=300, plot_height=300, toolbar_location=None, output_backend="webgl")
        plot_gamma = figure(x_range=(0, 100), plot_width=300, plot_height=300, toolbar_location=None, output_backend="webgl")
        plot_vrvphi = figure(x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), plot_width=300, plot_height=300, toolbar_location=None, output_backend="webgl")#x_range=(0, 100), plot_width=300, plot_height=300, toolbar_location=None)
        plot_eline = figure(x_range=(-10000, 10000), y_range=(0, 1), plot_width=300, plot_height=300, toolbar_location=None, output_backend="webgl")#x_range=(0, 100), plot_width=300, plot_height=300, toolbar_location=None)

        plot_edge.xaxis.axis_label = "x (light days)"
        plot_edge.yaxis.axis_label = "z (light days)"
        plot_face.xaxis.axis_label = "y (light days)"
        plot_face.yaxis.axis_label = "z (light days)"
        plot_tf.xaxis.axis_label = "Velocity (km/s)"
        plot_tf.yaxis.axis_label = "Lag (light days)"
        plot_gamma.xaxis.axis_label = "r (light days)"
        plot_gamma.yaxis.axis_label = "P(r)"
        plot_vrvphi.xaxis.axis_label = "vr/vcirc"
        plot_vrvphi.yaxis.axis_label = "vphi/vcirc"
        plot_eline.xaxis.axis_label = "Velocity (km/s)"
        plot_eline.yaxis.axis_label = "Flux (arbitrary)"

        phi = np.linspace(0, 2.0 * np.pi, 100)
        x, y = 1.0 * np.sin(phi), 1.0 * np.cos(phi)
        plot_vrvphi.line(x,y)
        x, y = np.sqrt(2.0) * np.sin(phi), 1.0 * np.cos(phi)
        plot_vrvphi.line(x,y)
        x, y = np.sqrt(2.0) * np.sin(phi), np.sqrt(2.0) * np.cos(phi)
        plot_vrvphi.line(x,y)

        plot_edge.scatter('x', 'z', size='size', source=self.source)
        plot_face.scatter('y', 'z', size='size', source=self.source)
        plot_tf.scatter('vx', 'lags', size='size', source=self.source_vel, alpha=0.5/self.Nvpercloud)
        plot_gamma.scatter('x', 'y', source=self.source_gamma)
        plot_vrvphi.scatter('vr', 'vphi', source=self.source_vel, alpha=0.1*1000/self.Nclouds/self.Nvpercloud)
        plot_eline.line('vel', 'flux', source=self.source_eline)

        slider_width = 150
        self.sliders = {
            'thetai': Slider(start=0.0, end=90., value=self.start['thetai'], step=1, title="Inclination angle (deg)", width=slider_width),
            'thetao': Slider(start=1.0, end=90., value=self.start['thetao'], step=1, title="Opening angle (deg)", width=slider_width),
            'gamma': Slider(start=1.0, end=2.0, value=self.start['gamma'], step=0.1, title="Gamma", width=slider_width),
            'kappa': Slider(start=-0.5, end=0.5, value=self.start['kappa'], step=0.05, title="Kappa", width=slider_width),
            'beta': Slider(start=0.1, end=2.0, value=self.start['beta'], step=0.05, title="Beta", width=slider_width),
            'mu': Slider(start=7.5, end=72.5, value=self.start['mu'], step=1, title="Mu (light days)", width=slider_width),
            'F': Slider(start=0.0, end=0.99, value=self.start['F'], step=0.01, title="F", width=slider_width),
            'xi': Slider(start=0.0, end=1.0, value=self.start['xi'], step=0.01, title="Xi", width=slider_width),
            'logMbh': Slider(start=6.5, end=8.5, value=self.start['logMbh'], step=0.1, title="log10(Mbh/Msun)", width=slider_width),
            'angular_sd_orbiting': Slider(start=0.001, end=0.1, value=self.start['angular_sd_orbiting'], step=0.001, title="Angular sd orbiting", width=slider_width),
            'radial_sd_orbiting': Slider(start=0.001, end=0.1, value=self.start['radial_sd_orbiting'], step=0.001, title="Radial sd orbiting", width=slider_width),
            'angular_sd_flowing': Slider(start=0.001, end=0.1, value=self.start['angular_sd_flowing'], step=0.001, title="Angular sd flowing", width=slider_width),
            'radial_sd_flowing': Slider(start=0.001, end=0.1, value=self.start['radial_sd_flowing'], step=0.001, title="Radial sd flowing", width=slider_width),
            'fflow': Slider(start=0.0, end=1.0, value=self.start['fflow'], step=1.0, title="Fflow", width=slider_width),
            'fellip': Slider(start=0.0, end=1.0, value=self.start['fellip'], step=0.01, title="Fellip", width=slider_width),
            'ellipseAngle': Slider(start=0.0, end=90.0, value=self.start['ellipseAngle'], step=1.0, title="Ellipse angle (deg)", width=slider_width),
            'logturbulence': Slider(start=-3.0, end=-1.0, value=self.start['logturbulence'], step=0.05, title="log10(Turbulence)", width=slider_width),
        }
        reset_button = Button(label='Reset', width=slider_width)

        if self.output in ['notebook', 'server']:
            for key in ['angular_sd_orbiting','angular_sd_flowing','radial_sd_orbiting','radial_sd_flowing','fflow','fellip','ellipseAngle', 'logMbh']:
                self.sliders[key].on_change("value", self.callback_vel_sd)
            for key in ['thetai','thetao','gamma','kappa','beta','mu','F']:
                self.sliders[key].on_change("value", self.callback_geo)
        else:
            with open('callback_vel_sd.js','rb') as f:
                callback_vel_sd_code = f.read().decode('ascii')
            callback_vel_sd = CustomJS(
                args=dict(
                    source_clouds = self.source,
                    source_gamma = self.source_gamma,
                    source_vel = self.source_vel,
                    source_eline = self.source_eline,
                    intermediate = self.intermediate,
                    params = self.params,
                    rands = self.randoms,
                    **self.sliders
                ),
                code=callback_vel_sd_code
            )
            callback_reset_code = '\n'.join(["%s.value = %.2f;" % (key, self.start[key]) for key in self.sliders.keys()])
            callback_reset = CustomJS(
                args={'source':self.source, **self.sliders},
                code=callback_reset_code
            )
            #callback_vel_sd = 
            #callback_geo = 
            for key in self.sliders.keys():
                self.sliders[key].js_on_change("value", callback_vel_sd)
            #for key in ['thetai','thetao','gamma','kappa','beta','mu','F']:
            #    self.sliders[key].js_on_change("value", self.callback_geo)
            reset_button.js_on_click(callback_reset)


        layout = row(
            column(
                [self.sliders[key] for key in self.sliders.keys()][:9]),
            column(
                [self.sliders[key] for key in self.sliders.keys()][9:] + [reset_button]),
            column(
                row(plot_edge, plot_face, plot_eline),
                row(plot_gamma, plot_vrvphi, plot_tf)
                #row(plot_edge,plot_face),
                #row(plot_eline, plot_tf),
                #row(plot_gamma, plot_vrvphi)
            )
        )

        #TOOLS = 'lasso_select, reset'
        
        #hist, edges = np.histogram(clouds['r'], density=False, bins=20)
        #ax_hist = figure(width=200, height=200, tools=TOOLS, title=None)
        #ax_hist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
        #       fill_color="navy", line_color="white", alpha=0.5)
        #ax_scat = figure(width=200, height=200, tools=TOOLS, title=None)
        #ax_scat.scatter(
        #    'x', 'y', source=source_gamma, size=5,
        #    fill_color='blue',line_color='navy', fill_alpha=0.4, line_alpha=0.4,
        #    nonselection_fill_color='blue', nonselection_line_color='navy',nonselection_alpha=0.4, nonselection_line_alpha=0.4,
        #    selection_fill_color='orange', selection_line_color='orange', selection_alpha=0.5,
        #)
        #ax_grid = gridplot([[ax_hist, ax_scat]])
        ##ax_grid = gridplot(ax)
        ##source.selected.on_change('indices', self.callback)
        
        if self.output in ['notebook', 'server']:
            # add the layout to curdoc
            doc.add_root(layout)
        else:
            show(layout)

class BKCorner():
    def __init__(self, trim_factor=1, logify=False, output='notebook', port=5006, notebook_url="http://localhost:8888", **kwargs):
        if output == 'notebook':
            reset_output()
            output_notebook()
            show(self.modify_doc, notebook_url=notebook_url)
        elif output == 'server':
            reset_output()
            server = Server({'/': self.modify_doc}, port=port)
            server.start()
            try:
                server.run_until_shutdown()
            except:
                print("Server already running")
            self.server = server
        else:
            reset_output()
            output_file(output)

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

    def modify_doc(self, doc):
        kwargs = {
            "panel_width": 150,
            "label_all_axes": False,
            "title": False,
        }
        for key in self.kwargs.keys():
            kwargs[key] = self.kwargs[key]

        if isinstance(logify,bool):
            logify = [logify]*len(params)
        data = {}
        for i in range(len(params)):
            if logify[i]:
                data[params[i]] = np.log10(df[params[i]])
            else:
                data[params[i]] = df[params[i]]

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
                vline = Span(location=medians[params[col]][0], dimension='height', line_color='black', line_width=1.5, line_dash='dashed')
                hline = Span(location=medians[params[row]][0], dimension='width', line_color='black', line_width=1.5, line_dash='dashed')

                #vline = Span(location=params[col], source=src_medians, dimension='height', line_color='orange', line_width=1.5, line_dash='dashed')
                #hline = Span(location=row, dimension='width', line_color='orange', line_width=1.5, line_dash='dashed')
                if col==0:
                    width = int(kwargs["panel_width"]*1.25)
                else:
                    width = kwargs["panel_width"]
                if row==Nparam-1:
                    height = int(kwargs["panel_width"]*1.25)
                else:
                    height = kwargs["panel_width"]
                if row == col:
                    hist, edges = np.histogram(data[params[col]], density=False, bins=20)
                    if kwargs['title']:
                        _title = params[col]
                    else:
                        _title = None
                    ax[row][col] = figure(width=width, height=height, tools=TOOLS, title=_title)
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
                if kwargs["label_all_axes"]:
                    ax[row][col].yaxis.axis_label = params[row]
                    ax[row][col].xaxis.axis_label = params[col]
        ax_grid = gridplot(ax)

        source.selected.on_change('indices', callback)

        # add the layout to curdoc
        doc.add_root(ax_grid)