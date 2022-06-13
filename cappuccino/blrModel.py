import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation
import matplotlib as mpl
import pandas as pd
import webbrowser
import os
import json
from scipy.ndimage import gaussian_filter1d

mpl.rcParams['text.usetex'] = True
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

c_km = 299792.458

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

def addText(text, ax, loc=(0.05,0.05), **kwargs):
    llim, rlim = ax.get_xlim()
    blim, tlim = ax.get_ylim()
    ax.text(loc[0]*(rlim-llim)+llim,tlim-loc[1]*(tlim-blim),text,**kwargs)

class Model:
    def __init__(self, params={}, randoms={}, _type='verynew', Nclouds=1000, Nvpercloud=1, planeofsky=False, plotting=False):
        self._type = _type
        if type(params) == str:
            params = self.defaultParams(params)
            self.params = params
        else:
            self.params = params
        self.randoms = randoms
        self.Nclouds = Nclouds
        self.Nvpercloud = Nvpercloud
        self.planeofsky = planeofsky
        self.plotting = plotting

        if params != {}:
            self.buildModel(params=self.params, plotting=self.plotting)

    @classmethod
    def fromCARAMEL(cls, run_dir, fn_clouds, _type='caramel'):
        clouds = np.loadtxt(run_dir + fn_clouds)
        x, y, z = clouds[:,0], clouds[:,1], clouds[:,2]
        vx, vy, vz = clouds[:,3], clouds[:,4], clouds[:,5]
        lams, lags, size = clouds[:,6], clouds[:,7], clouds[:,8]

        x /= c_km * 86400. * 1000.
        y /= c_km * 86400. * 1000.
        z /= c_km * 86400. * 1000.

        vx /= 1000.
        vy /= 1000.
        vz /= 1000.

        lags /= 86400.

        r = np.sqrt(x**2 + y**2 + z**2)
        vtot = np.sqrt(vx**2 + vy**2 + vz**2)

        model = cls(Nclouds=len(x), Nvpercloud=1, _type=_type, planeofsky=False)
        model.x, model.y, model.z = x, y, z
        model.vx, model.vy, model.vz = vx, vy, vz
        model.lams, model.lags, model.size = lams, lags, size
        model.r, model.vtot = r, vtot

        return model

    @staticmethod
    def defaultParams(geo):
        if geo == 'ring':
            params = {'thetai': 40.0, 'thetao': 10.0, 'kappa': 0.0, 'beta': 0.5, 'mu': 10.0, 'F': 0.8, 'rmax': 100,
                'logMbh': 7.5, 'gamma': 1.0, 'xi': 1.0, 'fflow': 1.0, 'fellip': 1.0, 'ellipseAngle': 0.0,
                'angular_sd_orbiting': 0.001, 'radial_sd_orbiting': 0.001, 'angular_sd_flowing': 0.001,
                'radial_sd_flowing': 0.001, 'turbulence': 0.01, 'Cadd': 0.0, 'Cmult': 1.0}
        elif geo == 'ring_outflow':
            params = {'thetai': 40.0, 'thetao': 10.0, 'kappa': 0.0, 'beta': 0.5, 'mu': 10.0, 'F': 0.8, 'rmax': 100,
                'logMbh': 7.5, 'gamma': 1.0, 'xi': 1.0, 'fflow': 1.0, 'fellip': 0.2, 'ellipseAngle': 20.0,
                'angular_sd_orbiting': 0.001, 'radial_sd_orbiting': 0.001, 'angular_sd_flowing': 0.001,
                'radial_sd_flowing': 0.001, 'turbulence': 0.01, 'Cadd': 0.0, 'Cmult': 1.0}
        elif geo == 'flatdisk':
            params = {'thetai': 40.0, 'thetao': 10.0, 'kappa': 0.0, 'beta': 1.2, 'mu': 10.0, 'F': 0.2, 'rmax': 100,
                'logMbh': 7.5, 'gamma': 1.0, 'xi': 1.0, 'fflow': 1.0, 'fellip': 1.0, 'ellipseAngle': 0.0,
                'angular_sd_orbiting': 0.001, 'radial_sd_orbiting': 0.001, 'angular_sd_flowing': 0.001,
                'radial_sd_flowing': 0.001, 'turbulence': 0.01, 'Cadd': 0.0, 'Cmult': 1.0}
        elif geo == 'flatdisk_outflow':
            params = {'thetai': 40.0, 'thetao': 10.0, 'kappa': 0.0, 'beta': 1.2, 'mu': 10.0, 'F': 0.2, 'rmax': 100,
                'logMbh': 7.5, 'gamma': 1.0, 'xi': 1.0, 'fflow': 1.0, 'fellip': 0.2, 'ellipseAngle': 20.0,
                'angular_sd_orbiting': 0.001, 'radial_sd_orbiting': 0.001, 'angular_sd_flowing': 0.001,
                'radial_sd_flowing': 0.001, 'turbulence': 0.01, 'Cadd': 0.0, 'Cmult': 1.0}
        elif geo == 'thickdisk':
            params = {'thetai': 45.0, 'thetao': 40.0, 'kappa': 0.0, 'beta': 0.85, 'mu': 10.0, 'F': 0.3, 'rmax': 100,
                'logMbh': 7.5, 'gamma': 1.0, 'xi': 1.0, 'fflow': 1.0, 'fellip': 1.0, 'ellipseAngle': 30,
                'angular_sd_orbiting': 0.001, 'radial_sd_orbiting': 0.001, 'angular_sd_flowing': 0.001,
                'radial_sd_flowing': 0.001, 'turbulence': 0.01, 'Cadd': 0.0, 'Cmult': 1.0}
        elif geo == 'typical':
            params = {'thetai': 45.0, 'thetao': 40.0, 'kappa': -0.45, 'beta': 1.02, 'mu': 10.0, 'F': 0.4, 'rmax': 100,
                'logMbh': 7.5, 'gamma': 1.0, 'xi': 0.3, 'fflow': 1.0, 'fellip': 0.85, 'ellipseAngle': 30,
                'angular_sd_orbiting': 0.03, 'radial_sd_orbiting': 0.02, 'angular_sd_flowing': 0.03,
                'radial_sd_flowing': 0.01, 'turbulence': 0.05, 'Cadd': 0.0, 'Cmult': 1.0}
        else:
            print "No default set of parameters for '%s'" % geo
        return params

    @staticmethod
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

    @staticmethod
    def drawAngMom(ang_mom_0, thetao, x):
        theta_max = thetao * 1.0
        while True:
            theta = np.random.uniform(-theta_max,theta_max)
            ang_mom = rotateVecAroundAxis(x, theta, ang_mom_0)
            if np.arccos(ang_mom[2]) < thetao:
                return ang_mom
            else:
                theta_max = theta

    @staticmethod
    def getVelocityUnitVector(x, l):
        x, l = np.array(x), np.array(l)
        x = x / np.linalg.norm(x)
        l = l / np.linalg.norm(l)
        return np.cross(l, x)

    @staticmethod
    def drawFromGammaNumpy(beta, mu, F, rmax, Nvals, granularity=4, plotting=False):
        # Change of variables
        alpha = beta**-2.0
        rmin = mu * F
        theta = (mu - rmin) / alpha
        
        r = np.random.gamma(alpha, scale=theta, size=Nvals) + rmin
        ind = np.where(r>rmax)
        counter = 0
        while len(ind[0]) != 0:
            r_new = np.random.gamma(alpha, scale=theta, size=len(ind[0])) + rmin
            r[ind] = r_new
            ind = np.where(r>rmax)
            counter += 1
            if counter > 1000:
                print "Taking too long to draw radii, rmax too small."
                break

        if plotting:
            plt.figure()
            plt.hist(r, bins=50, density=True)
            plt.xlabel('$r~({\\rm light~days})$')
            plt.show()

        return r
    
    @staticmethod
    def drawFromGamma(beta, mu, F, rmax, Nvals, granularity=4, plotting=False):
        # Change of variables
        alpha = beta**-2.0
        rmin = mu * F
        theta = (mu - rmin) / alpha
        
        # Compute the gamma function from 0 to rmax, then normalize
        xvals = np.linspace(0,rmax,granularity*Nvals)
        gamma_distr = (xvals - rmin)**(alpha - 1.) * np.exp(-(xvals - rmin) / theta)
        gamma_distr = np.where(gamma_distr > 0.0, gamma_distr, 0.0)  # remove negatives before rmin
        gamma_distr = gamma_distr / np.sum(gamma_distr)
        cdf = np.array([sum(gamma_distr[:i+1]) for i in range(len(gamma_distr))])
        
        # Take sorted distribution of values 0-1 to draw from cdf
        j = 0
        r = np.zeros(Nvals)
        yvals = np.linspace(0,1,Nvals)
        for i in range(Nvals):
            j=0
            run_loop = True
            while run_loop:
                if j >= len(cdf):
                    r[i] = xvals[-1]
                    run_loop = False
                elif cdf[j] > yvals[i]:
                    r[i] = xvals[j]
                    run_loop = False
                else:
                    j += 1
        
        if plotting:
            plt.figure()
            plt.plot(xvals, gamma_distr/(xvals[1]-xvals[0]))
            plt.hist(r, bins=50, density=True)
            plt.xlabel('$r~({\\rm light~days})$')
            plt.show()

        return r

    @staticmethod
    def drawVrVphi(Nclouds=1000, Nvpercloud=5, params={}, randoms={}):
        if params == {}:
            params = {'angular_sd_orbiting': 0.01, 'angular_sd_flowing': 0.01,
                'radial_sd_orbiting': 0.01, 'radial_sd_flowing': 0.01,
                'fellip': 0.5, 'fflow': 1.0, 'ellipseAngle': 20.0}
        angular_sd_orbiting = params['angular_sd_orbiting']
        angular_sd_flowing = params['angular_sd_flowing']
        radial_sd_orbiting = params['radial_sd_orbiting']
        radial_sd_flowing = params['radial_sd_flowing']
        fellip = params['fellip']
        fflow = params['fflow']
        ellipseAngle = params['ellipseAngle']

        if randoms == {}:
            randoms['randnorm1'] = np.random.normal(0,1,(Nclouds,Nvpercloud))
            randoms['randnorm2'] = np.random.normal(0,1,(Nclouds,Nvpercloud))
            randoms['randorder'] = np.vstack([np.arange(Nclouds)]*Nvpercloud)
            for i in range(Nvpercloud):
                np.random.shuffle(randoms['randorder'][i])
            np.random.shuffle(randoms['randorder'])

        randorder = randoms['randorder']
        randnorm1 = randoms['randnorm1']
        randnorm2 = randoms['randnorm2']

        vr = np.empty((Nclouds,Nvpercloud))
        vphi = np.empty((Nclouds,Nvpercloud))
        _theta = np.empty(Nclouds)
        
        # Circularlike orbits:
        startindex = int(fellip*Nclouds)
                
        for j in range(Nvpercloud):
            _circ = randorder[j][:startindex]
            _flow = randorder[j][startindex:]
            _theta[_circ] = 0.5*np.pi + angular_sd_orbiting*randnorm1[_circ,j]
            vr[_circ,j] = np.sqrt(2.0) * np.cos(_theta[_circ]) * np.exp(radial_sd_orbiting*randnorm2[_circ,j])    
            vphi[_circ,j] = np.sin(_theta[_circ]) * np.exp(radial_sd_orbiting*randnorm2[_circ,j])
            
            # Inflow / outflow
            if fflow < 0.5:
                _theta[_flow] = np.pi - ellipseAngle*np.pi/180. + angular_sd_flowing*randnorm1[_flow,j]
                vr[_flow,j] = np.sqrt(2.0) * np.cos(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow,j])
                vphi[_flow,j] = np.sin(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow,j])
            else:
                _theta[_flow] = 0.0 + ellipseAngle*np.pi/180. + angular_sd_flowing*randnorm1[_flow,j]
                vr[_flow,j] = np.sqrt(2.0) * np.cos(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow,j])
                vphi[_flow,j] = np.sin(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow,j])
        return vr, vphi

    def buildModel(self, params, plotting=False):
        if self._type.lower() == 'old':
            model = self.buildModelOld(params=params, Nclouds=self.Nclouds, Nvpercloud=self.Nvpercloud, randoms=self.randoms,
                planeofsky=self.planeofsky, plotting=plotting)
        elif self._type.lower() == 'new':
            model = self.buildModelNew(params=params, Nclouds=self.Nclouds, Nvpercloud=self.Nvpercloud, randoms=self.randoms,
                planeofsky=self.planeofsky, plotting=plotting)
        elif self._type.lower() == 'verynew':
            model = self.buildModelVeryNew(params=params, Nclouds=self.Nclouds, Nvpercloud=self.Nvpercloud, randoms=self.randoms,
                planeofsky=self.planeofsky, plotting=plotting)
        else:
            print "Invalid _type"

        self.model = model
        self.x, self.y, self.z = model[0], model[1], model[2]
        self.vx, self.vy, self.vz = model[3], model[4], model[5]
        self.lags, self.size = model[6], model[7]

        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.vtot = np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

        if plotting:
            self.showModel(_style='quiver')

    def buildModelOld(self, params, Nclouds=1000, Nvpercloud=1, randoms={}, animate=False, anim_steps=20, mid_pad=5, planeofsky=False, plotting=False):
        # Set the parameters and random numbers
        if randoms != {}:
            rand1, rand2, rand4, rand0, randorder = randoms['rand1'], randoms['rand2'], randoms['rand4'],randoms['rand0'], randoms['randorder']
            Nclouds = len(rand1)
        else:
            rand0 = np.random.uniform(-1,1,Nclouds)
            rand1 = np.random.uniform(0,1,Nclouds)
            rand2 = np.random.uniform(-1,1,Nclouds)
            rand4 = np.linspace(-1,1,Nclouds)
            randnorm1 = np.random.normal(0,1,(Nclouds,Nvpercloud))
            randnorm2 = np.random.normal(0,1,(Nclouds,Nvpercloud))
            randnorm3 = np.random.normal(0,1,(Nclouds,Nvpercloud))
            randnorm4 = np.random.normal(0,1,(Nclouds,Nvpercloud))
            randnorm5 = np.random.normal(0,1,(Nclouds,Nvpercloud))
            randorder = np.vstack([np.arange(Nclouds)]*Nvpercloud)
            for i in range(Nvpercloud):
                np.random.shuffle(randorder[i])
            np.random.shuffle(randorder)
            randoms = {'rand1':rand1, 'rand2':rand2, 'rand4':rand4, 'rand0':rand0, 'randorder':randorder}

        Mbh = 10**params['logMbh']
        beta, mu, F, rmax = params['beta'], params['mu'], params['F'], params['rmax']
        thetao, thetai = params['thetao'], params['thetai']
        gamma, kappa, xi = params['gamma'], params['kappa'], params['xi']
        fflow, fellip = params['fflow'], params['fellip']
        angular_sd_orbiting, angular_sd_flowing = params['angular_sd_orbiting'], params['angular_sd_flowing']
        radial_sd_orbiting, radial_sd_flowing = params['radial_sd_orbiting'], params['radial_sd_flowing']
        ellipseAngle = params['ellipseAngle']
        turbulence = params['turbulence']

        
        ###########################################################################################
        ######################################  Geometry  #########################################
        ###########################################################################################
        
        r = self.drawFromGammaNumpy(beta, mu, F, rmax, Nvals=len(rand1), granularity=4, plotting=plotting)
        
        # Determine the per-particle opening angles    
        part1 = np.sin(thetao*np.pi/180.)
        part2 = rand1**(1.0/gamma)
        angle = np.arcsin(part1 * part2)

        # Pre-calculate some sines and cosines
        sin0 = np.sin(rand0 * np.pi)
        cos0 = np.cos(rand0 * np.pi)
        sin1 = np.sin(angle)
        cos1 = np.cos(angle)
        sin2 = np.sin(rand2 * np.pi)
        cos2 = np.cos(rand2 * np.pi)
        sin3 = np.sin(0.5 * np.pi - thetai * np.pi / 180.)
        cos3 = np.cos(0.5 * np.pi - thetai * np.pi / 180.)
        sin4 = np.sin(rand4 * np.pi)
        cos4 = np.cos(rand4 * np.pi)
        
        if animate:
            Nsteps = anim_steps
            x_anim, y_anim, z_anim = [], [], []
            vx_anim, vy_anim, vz_anim = [], [], []
            size_anim = []
            label_anim = []
            sin0_anim = np.sin(rand0 * np.pi/float(Nsteps))
            cos0_anim = np.cos(rand0 * np.pi/float(Nsteps))
            sin1_anim = np.sin(angle/float(Nsteps))
            cos1_anim = np.cos(angle/float(Nsteps))
            sin2_anim = np.sin(rand2 * np.pi/float(Nsteps))
            cos2_anim = np.cos(rand2 * np.pi/float(Nsteps))
            sin3_anim = np.sin((0.5 * np.pi - thetai * np.pi / 180.)/float(Nsteps))
            cos3_anim = np.cos((0.5 * np.pi - thetai * np.pi / 180.)/float(Nsteps))
            sin4_anim = np.sin(rand4 * np.pi/float(Nsteps))
            cos4_anim = np.cos(rand4 * np.pi/float(Nsteps))

        # Set the positions
        x = r * 1.0
        y = np.zeros(len(r))
        z = np.zeros(len(r))
        if animate:
            x_anim.append(list(x))
            y_anim.append(list(y))
            z_anim.append(list(z))
            size_anim.append([1.0]*len(x))
            label_anim.append('')

        # Rotate into a flat disk
        if animate:
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
            for i in range(Nsteps):
                x_tmp, y_tmp = rotate(x_anim[-1], y_anim[-1], cos0_anim, sin0_anim)
                x_anim.append(x_tmp)
                y_anim.append(y_tmp)
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append('${\\rm Flat~disk}$')
        x, y = rotate(x, y, cos0, sin0)
        
        # Puff up into thick disk
        if animate:
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
            for i in range(Nsteps):
                x_tmp, z_tmp = rotate(x_anim[-1], z_anim[-1], cos1_anim, sin1_anim)
                x_anim.append(x_tmp)
                z_anim.append(z_tmp)
                y_anim.append(y_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append('${\\rm Opening~angle}$')
        x, z = rotate(x, z, cos1, sin1)

        # Restore axi-symmetry
        if animate:
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
            for i in range(Nsteps):
                x_tmp, y_tmp = rotate(x_anim[-1], y_anim[-1], cos2_anim, sin2_anim)
                x_anim.append(x_tmp)
                y_anim.append(y_tmp)
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append('${\\rm Restore~axi-symmetry}$')
        x, y = rotate(x, y, cos2, sin2)
        
        # Truth vector to determine if on far side of mid-plane
        far_side = np.where(z < 0.)
        # Rotate by inclination angle (+pi/2)
        if animate:
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
            for i in range(Nsteps):
                x_tmp, z_tmp = rotate(x_anim[-1], z_anim[-1], cos3_anim, sin3_anim)
                x_anim.append(x_tmp)
                z_anim.append(z_tmp)
                y_anim.append(y_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append('${\\rm Inclination~angle}$')
        x, z = rotate(x, z, cos3, sin3)
        
        # Add in turbulence
        if animate:
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
            for i in range(Nsteps):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append('${\\rm Turbulence}$')

        if planeofsky:
            # Rotate in plane of sky
            if animate:
                for i in range(mid_pad):
                    x_anim.append(x_anim[-1])
                    y_anim.append(y_anim[-1])
                    z_anim.append(z_anim[-1])
                    size_anim.append(size_anim[-1])
                    label_anim.append(label_anim[-1])
                for i in range(Nsteps):
                    y_tmp, z_tmp = rotate(y_anim[-1], z_anim[-1], cos4_anim, sin4_anim)
                    y_anim.append(y_tmp)
                    z_anim.append(z_tmp)
                    x_anim.append(x_anim[-1])
                    size_anim.append(size_anim[-1])
                    label_anim.append('${\\rm Plane~of~sky~rotation}$')
            y, z = rotate(y, z, cos4, sin4)

        
        ###########################################################################################
        #####################################  Kinematics  ########################################
        ###########################################################################################
        # Compute the velocities
        G = 5.123619161  # ld / Msun * (km/s)^2
        
        vr = np.empty(len(r))
        vphi = np.empty(len(r))
        _theta = np.empty(len(r))
        
        # Circularlike orbits:
        startindex = int(fellip*len(r))
        vcirc = np.sqrt(G * Mbh / r)
        vesc = np.sqrt(2.0 * G * Mbh / r)   

        vx, vy, vz = np.zeros((len(x), Nvpercloud)), np.zeros((len(x), Nvpercloud)), np.zeros((len(x), Nvpercloud))
        for j in range(Nvpercloud):
            _circ = randorder[j][:startindex]
            _flow = randorder[j][startindex:]
            _theta[_circ] = 0.5*np.pi + angular_sd_orbiting*randnorm1[_circ,j]
            vr[_circ] = np.sqrt(2.0 * G * Mbh / r[_circ]) * np.cos(_theta[_circ]) * np.exp(radial_sd_orbiting*randnorm2[_circ,j])    
            vphi[_circ] = np.sqrt(G * Mbh / r[_circ]) * np.sin(_theta[_circ]) * np.exp(radial_sd_orbiting*randnorm2[_circ,j])
        
            # Inflow / outflow
            if fflow < 0.5:
                _theta[_flow] = np.pi - ellipseAngle*np.pi/180. + angular_sd_flowing*randnorm1[_flow,j]
                vr[_flow] = np.sqrt(2.0 * G * Mbh / r[_flow]) * np.cos(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow,j])
                vphi[_flow] = np.sqrt(G * Mbh / r[_flow]) * np.sin(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow,j])
            else:
                _theta[_flow] = 0.0 + ellipseAngle*np.pi/180. + angular_sd_flowing*randnorm1[_flow,j]
                vr[_flow] = np.sqrt(2.0 * G * Mbh / r[_flow]) * np.cos(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow,j])
                vphi[_flow] = np.sqrt(G * Mbh / r[_flow]) * np.sin(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow,j])
            
            if (plotting) and (j == 0):
                angles = np.linspace(0,2.*np.pi,360)
                plt.figure(figsize=(10,5))
                ax = plt.subplot2grid((2,4), (0,0), rowspan=2, colspan=2)
                ax_vr = plt.subplot2grid((2,4), (0,2), rowspan=1, colspan=2)
                ax_vphi = plt.subplot2grid((2,4), (1,2), rowspan=1, colspan=2)
                ax.plot(np.sqrt(2.0) * np.cos(angles), np.sqrt(2.0) * np.sin(angles), color='k')
                ax.plot(np.sqrt(2.0) * np.cos(angles), np.sin(angles), color='k', ls='dotted')
                ax.plot(np.cos(angles), np.sin(angles), color='k', ls='dashed')

                ax.scatter(
                    vr[_circ]/vcirc[_circ],
                    vphi[_circ]/vcirc[_circ],
                    alpha = 0.2,
                    color = 'blue'
                )
                ax.scatter(
                    vr[_flow]/vcirc[_flow],
                    vphi[_flow]/vcirc[_flow],
                    alpha = 0.2,
                    color = 'orange'
                )
                
                ax_vr.hist(vr[_circ], color='blue', bins=20, alpha=0.7, label="${\\rm Near-circular}$")
                ax_vphi.hist(vphi[_circ], color='blue', bins=20, alpha=0.7)
                ax_vr.hist(vr[_flow], color='orange', bins=20, alpha=0.7, label="${\\rm Inflow/Outflow}$")
                ax_vphi.hist(vphi[_flow], color='orange', bins=20, alpha=0.7)

                ax.set_xlabel('$v_r/v_{\\rm circ}$')
                ax.set_ylabel('$v_\\phi/v_{\\rm circ}$')
                ax.set_xlim(-1.7, 1.7)
                ax.set_ylim(-1.7, 1.7)
                ax.set_xticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])
                ax.set_yticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])

                ax_vr.legend()
                ax_vr.set_xlabel('$v_r~({\\rm km/s})$')
                ax_vphi.set_xlabel('$v_\\phi~({\\rm km/s})$')
                plt.tight_layout()
                plt.show()
        
            # Convert to cartesian coordinates
            vx[:,j] = vr * 1.0
            vy[:,j] = vphi * 1.0
            vz[:,j] = np.zeros(len(r))
            if (animate) and (j == 0):
                print vx.shape
                vx_anim.append(list(vx[:,j]))
                vy_anim.append(list(vy[:,j]))
                vz_anim.append(list(vz[:,j]))
            # Rotate into a flat disk
            if animate:
                for i in range(mid_pad):
                    vx_anim.append(vx_anim[-1])
                    vy_anim.append(vy_anim[-1])
                    vz_anim.append(vz_anim[-1])
                for i in range(Nsteps):
                    vx_tmp, vy_tmp = rotate(vx_anim[-1], vy_anim[-1], cos0_anim, sin0_anim)
                    vx_anim.append(vx_tmp)
                    vy_anim.append(vy_tmp)
                    vz_anim.append(vz_anim[-1])
            vx[:,j], vy[:,j] = rotate(vx[:,j], vy[:,j], cos0, sin0)
        
            # Puff up into thick disk
            if animate:
                for i in range(mid_pad):
                    vx_anim.append(vx_anim[-1])
                    vy_anim.append(vy_anim[-1])
                    vz_anim.append(vz_anim[-1])
                for i in range(Nsteps):
                    vx_tmp, vz_tmp = rotate(vx_anim[-1], vz_anim[-1], cos1_anim, sin1_anim)
                    vx_anim.append(vx_tmp)
                    vz_anim.append(vz_tmp)
                    vy_anim.append(vy_anim[-1])
            vx[:,j], vz[:,j] = rotate(vx[:,j], vz[:,j], cos1, sin1)
        
            # Restore axi-symmetry
            if animate:
                for i in range(mid_pad):
                    vx_anim.append(vx_anim[-1])
                    vy_anim.append(vy_anim[-1])
                    vz_anim.append(vz_anim[-1])
                for i in range(Nsteps):
                    vx_tmp, vy_tmp = rotate(vx_anim[-1], vy_anim[-1], cos2_anim, sin2_anim)
                    vx_anim.append(vx_tmp)
                    vy_anim.append(vy_tmp)
                    vz_anim.append(vz_anim[-1])
            vx[:,j], vy[:,j] = rotate(vx[:,j], vy[:,j], cos2, sin2)
        
            # Rotate by inclination angle (+pi/2)
            if animate:
                for i in range(mid_pad):
                    vx_anim.append(vx_anim[-1])
                    vy_anim.append(vy_anim[-1])
                    vz_anim.append(vz_anim[-1])
                for i in range(Nsteps):
                    vx_tmp, vz_tmp = rotate(vx_anim[-1], vz_anim[-1], cos3_anim, sin3_anim)
                    vx_anim.append(vx_tmp)
                    vz_anim.append(vz_tmp)
                    vy_anim.append(vy_anim[-1])
            vx[:,j], vz[:,j] = rotate(vx[:,j], vz[:,j], cos3, sin3)
        
        # Add in turbulence
        if animate:
            for i in range(mid_pad):
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
            for i in range(Nsteps):
                vx_anim.append(vx_anim[-1] + turbulence * np.sqrt(G * Mbh / r) * randnorm3[:,0] / float(Nsteps))
                vy_anim.append(vy_anim[-1] + turbulence * np.sqrt(G * Mbh / r) * randnorm4[:,0] / float(Nsteps))
                vz_anim.append(vz_anim[-1] + turbulence * np.sqrt(G * Mbh / r) * randnorm5[:,0] / float(Nsteps))
        vx += turbulence * np.atleast_2d(np.sqrt(G * Mbh / r)).T * randnorm3
        vy += turbulence * np.atleast_2d(np.sqrt(G * Mbh / r)).T * randnorm4
        vz += turbulence * np.atleast_2d(np.sqrt(G * Mbh / r)).T * randnorm5
        
        # Set so that moving away is positive velocity
        #vx = -vx
        #if animate:
            #vx_anim = np.array(vx_anim) * -1.0
            #vx_anim = list(vx_anim)

        if planeofsky:
            # Rotate in plane of sky
            if animate:
                for i in range(mid_pad):
                    vx_anim.append(vx_anim[-1])
                    vy_anim.append(vy_anim[-1])
                    vz_anim.append(vz_anim[-1])
                for i in range(Nsteps):
                    vy_tmp, vz_tmp = rotate(vy_anim[-1], vz_anim[-1], cos4_anim, sin4_anim)
                    vy_anim.append(vy_tmp)
                    vz_anim.append(vz_tmp)
                    vx_anim.append(vx_anim[-1])
            vy, vz = rotate(vy, vz, cos4, sin4)
        
        ###########################################################################################
        ##################################  Weights and Lags  #####################################
        ###########################################################################################
        # Compute the weights
        size = 0.5 + kappa * x/r
        size[far_side] *= xi
        size /= np.sum(size)
        if animate:
            size_step = (size - 1.0) / Nsteps
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
            for i in range(Nsteps):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
                size_anim.append(size_anim[-1] + size_step)
                label_anim.append('$\\xi~{\\rm and}~\\kappa$')

        # Compute the lags
        lags = r - x
        
        if animate:
            return x_anim, y_anim, z_anim, vx_anim, vy_anim, vz_anim, lags, size_anim, label_anim
        else:
            return x, y, z, vx, vy, vz, lags, size

    def buildModelNew(self, params, Nclouds=1000, Nvpercloud=1, randoms={}, animate=False, anim_steps=20, mid_pad=5, planeofsky=False, plotting=False):
        # Set the parameters and random numbers
        if randoms != {}:
            rand1, rand2, rand4, rand0, randorder = randoms['rand1'], randoms['rand2'], randoms['rand4'],randoms['rand0'], randoms['randorder']
            Nclouds = len(rand1)
        else:
            rand0 = np.random.choice((-1.0,1.0),Nclouds)
            rand1 = np.random.uniform(0,1,Nclouds)
            rand2 = np.random.uniform(-1,1,Nclouds)
            rand4 = np.linspace(-1,1,Nclouds)
            randnorm1 = np.random.normal(0,1,Nclouds)
            randnorm2 = np.random.normal(0,1,Nclouds)
            randnorm3 = np.random.normal(0,1,(Nclouds,Nvpercloud))
            randnorm4 = np.random.normal(0,1,(Nclouds,Nvpercloud))
            randnorm5 = np.random.normal(0,1,(Nclouds,Nvpercloud))
            randorder = np.arange(Nclouds)
            np.random.shuffle(randorder)
            randoms = {'rand1':rand1, 'rand2':rand2, 'rand4':rand4, 'rand0':rand0, 'randorder':randorder}

        Mbh = 10**params['logMbh']
        beta, mu, F, rmax = params['beta'], params['mu'], params['F'], params['rmax']
        thetao, thetai = params['thetao'], params['thetai']
        gamma, kappa, xi = params['gamma'], params['kappa'], params['xi']
        fflow, fellip = params['fflow'], params['fellip']
        angular_sd_orbiting, angular_sd_flowing = params['angular_sd_orbiting'], params['angular_sd_flowing']
        radial_sd_orbiting, radial_sd_flowing = params['radial_sd_orbiting'], params['radial_sd_flowing']
        ellipseAngle = params['ellipseAngle']
        turbulence = params['turbulence']

        
        ###########################################################################################
        ######################################  Geometry  #########################################
        ###########################################################################################
        
        r = self.drawFromGammaNumpy(beta, mu, F, rmax, Nvals=len(rand1), granularity=4, plotting=plotting)
        
        # Determine the per-particle opening angles    
        part1 = np.sin(thetao*np.pi/180.)
        part2 = rand1**(1.0/gamma)
        angle = np.arcsin(part1 * part2)

        # Pre-calculate some sines and cosines
        sin1 = np.sin(angle)
        cos1 = np.cos(angle)
        sin2 = np.sin(rand2 * np.pi)
        cos2 = np.cos(rand2 * np.pi)
        sin3 = np.sin(0.5 * np.pi - thetai * np.pi / 180.)
        cos3 = np.cos(0.5 * np.pi - thetai * np.pi / 180.)
        sin4 = np.sin(rand4 * np.pi)
        cos4 = np.cos(rand4 * np.pi)
        
        if animate:
            Nsteps = anim_steps
            x_anim, y_anim, z_anim = [], [], []
            vx_anim, vy_anim, vz_anim = [], [], []
            size_anim = []
            label_anim = []
            sin1_anim = np.sin(angle/float(Nsteps))
            cos1_anim = np.cos(angle/float(Nsteps))
            sin2_anim = np.sin(rand2 * np.pi/float(Nsteps))
            cos2_anim = np.cos(rand2 * np.pi/float(Nsteps))
            sin3_anim = np.sin((0.5 * np.pi - thetai * np.pi / 180.)/float(Nsteps))
            cos3_anim = np.cos((0.5 * np.pi - thetai * np.pi / 180.)/float(Nsteps))
            sin4_anim = np.sin(rand4 * np.pi/float(Nsteps))
            cos4_anim = np.cos(rand4 * np.pi/float(Nsteps))

        # Set the positions
        x = r * rand0
        y = np.zeros(len(r))
        z = np.zeros(len(r))
        if animate:
            x_anim.append(list(x))
            y_anim.append(list(y))
            z_anim.append(list(z))
            size_anim.append([1.0]*len(x))
            label_anim.append('')

        # Puff up by opening angle into wedge
        if animate:
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
            for i in range(Nsteps):
                x_tmp, z_tmp = rotate(x_anim[-1], z_anim[-1], cos1_anim, sin1_anim)
                x_anim.append(x_tmp)
                z_anim.append(z_tmp)
                y_anim.append(y_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append('${\\rm Opening~angle}$')
        x, z = rotate(x, z, cos1, sin1)
        
        # Rotate into thick disk
        if animate:
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
            for i in range(Nsteps):
                x_tmp, y_tmp = rotate(x_anim[-1], y_anim[-1], cos2_anim, sin2_anim)
                x_anim.append(x_tmp)
                y_anim.append(y_tmp)
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append('${\\rm Rotate~into~disk}$')
        x, y = rotate(x, y, cos2, sin2)
        
        # Truth vector to determine if on far side of mid-plane
        far_side = np.where(z < 0.)
        # Rotate by inclination angle (+pi/2)
        if animate:
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
            for i in range(Nsteps):
                x_tmp, z_tmp = rotate(x_anim[-1], z_anim[-1], cos3_anim, sin3_anim)
                x_anim.append(x_tmp)
                z_anim.append(z_tmp)
                y_anim.append(y_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append('${\\rm Inclination~angle}$')
        x, z = rotate(x, z, cos3, sin3)
        
        # Add in turbulence
        if animate:
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
            for i in range(Nsteps):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append('${\\rm Turbulence}$')

        if planeofsky:
            # Rotate in plane of sky
            if animate:
                for i in range(mid_pad):
                    x_anim.append(x_anim[-1])
                    y_anim.append(y_anim[-1])
                    z_anim.append(z_anim[-1])
                    size_anim.append(size_anim[-1])
                    label_anim.append(label_anim[-1])
                for i in range(Nsteps):
                    y_tmp, z_tmp = rotate(y_anim[-1], z_anim[-1], cos4_anim, sin4_anim)
                    y_anim.append(y_tmp)
                    z_anim.append(z_tmp)
                    x_anim.append(x_anim[-1])
                    size_anim.append(size_anim[-1])
                    label_anim.append('${\\rm Plane~of~sky~rotation}$')
            y, z = rotate(y, z, cos4, sin4)

        
        ###########################################################################################
        #####################################  Kinematics  ########################################
        ###########################################################################################
        # Compute the velocities
        G = 5.123619161  # ld / Msun * (km/s)^2
        
        vr = np.empty(len(r))
        vphi = np.empty(len(r))
        _theta = np.empty(len(r))
        
        # Circularlike orbits:
        startindex = int(fellip*len(r))
        vcirc = np.sqrt(G * Mbh / r)
        vesc = np.sqrt(2.0 * G * Mbh / r)   
        
        _circ = randorder[:startindex]
        _flow = randorder[startindex:]
        _theta[_circ] = 0.5*np.pi + angular_sd_orbiting*randnorm1[_circ]
        vr[_circ] = np.sqrt(2.0 * G * Mbh / r[_circ]) * np.cos(_theta[_circ]) * np.exp(radial_sd_orbiting*randnorm2[_circ])    
        vphi[_circ] = np.sqrt(G * Mbh / r[_circ]) * np.sin(_theta[_circ]) * np.exp(radial_sd_orbiting*randnorm2[_circ])
        
        # Inflow / outflow
        if fflow < 0.5:
            _theta[_flow] = np.pi - ellipseAngle*np.pi/180. + angular_sd_flowing*randnorm1[_flow]
            vr[_flow] = np.sqrt(2.0 * G * Mbh / r[_flow]) * np.cos(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow])
            vphi[_flow] = np.sqrt(G * Mbh / r[_flow]) * np.sin(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow])
        else:
            _theta[_flow] = 0.0 + ellipseAngle*np.pi/180. + angular_sd_flowing*randnorm1[_flow]
            vr[_flow] = np.sqrt(2.0 * G * Mbh / r[_flow]) * np.cos(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow])
            vphi[_flow] = np.sqrt(G * Mbh / r[_flow]) * np.sin(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow])
        
        if plotting:
            angles = np.linspace(0,2.*np.pi,360)
            plt.figure(figsize=(10,5))
            ax = plt.subplot2grid((2,4), (0,0), rowspan=2, colspan=2)
            ax_vr = plt.subplot2grid((2,4), (0,2), rowspan=1, colspan=2)
            ax_vphi = plt.subplot2grid((2,4), (1,2), rowspan=1, colspan=2)
            ax.plot(np.sqrt(2.0) * np.cos(angles), np.sqrt(2.0) * np.sin(angles), color='k')
            ax.plot(np.sqrt(2.0) * np.cos(angles), np.sin(angles), color='k', ls='dotted')
            ax.plot(np.cos(angles), np.sin(angles), color='k', ls='dashed')

            ax.scatter(
                vr[_circ]/vcirc[_circ],
                vphi[_circ]/vcirc[_circ],
                alpha = 0.2,
                color = 'blue'
            )
            ax.scatter(
                vr[_flow]/vcirc[_flow],
                vphi[_flow]/vcirc[_flow],
                alpha = 0.2,
                color = 'orange'
            )
            
            ax_vr.hist(vr[_circ], color='blue', bins=20, alpha=0.7, label="${\\rm Near-circular}$")
            ax_vphi.hist(vphi[_circ], color='blue', bins=20, alpha=0.7)
            ax_vr.hist(vr[_flow], color='orange', bins=20, alpha=0.7, label="${\\rm Inflow/Outflow}$")
            ax_vphi.hist(vphi[_flow], color='orange', bins=20, alpha=0.7)

            ax.set_xlabel('$v_r/v_{\\rm circ}$')
            ax.set_ylabel('$v_\\phi/v_{\\rm circ}$')
            ax.set_xlim(-1.7, 1.7)
            ax.set_ylim(-1.7, 1.7)
            ax.set_xticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])
            ax.set_yticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])

            ax_vr.legend()
            ax_vr.set_xlabel('$v_r~({\\rm km/s})$')
            ax_vphi.set_xlabel('$v_\\phi~({\\rm km/s})$')
            plt.tight_layout()
            plt.show()
        
        # Convert to cartesian coordinates
        vx = vr * rand0 
        vy = vphi * rand0
        vz = np.zeros(len(r))
        vx = np.vstack([vx]*Nvpercloud).T
        vy = np.vstack([vy]*Nvpercloud).T
        vz = np.vstack([vz]*Nvpercloud).T
        if animate:
            vx_anim.append(list(vx[:,0]))
            vy_anim.append(list(vy[:,0]))
            vz_anim.append(list(vz[:,0]))

        # Puff up by opening angle into wedge
        if animate:
            for i in range(mid_pad):
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
            for i in range(Nsteps):
                vx_tmp, vz_tmp = rotate(vx_anim[-1], vz_anim[-1], cos1_anim, sin1_anim)
                vx_anim.append(vx_tmp)
                vz_anim.append(vz_tmp)
                vy_anim.append(vy_anim[-1])
        
        vx, vz = rotate(vx, vz, cos1, sin1)
        # Rotate into thick disk
        if animate:
            for i in range(mid_pad):
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
            for i in range(Nsteps):
                vx_tmp, vy_tmp = rotate(vx_anim[-1], vy_anim[-1], cos2_anim, sin2_anim)
                vx_anim.append(vx_tmp)
                vy_anim.append(vy_tmp)
                vz_anim.append(vz_anim[-1])
        vx, vy = rotate(vx, vy, cos2, sin2)
        # Rotate by inclination angle (+pi/2)
        if animate:
            for i in range(mid_pad):
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
            for i in range(Nsteps):
                vx_tmp, vz_tmp = rotate(vx_anim[-1], vz_anim[-1], cos3_anim, sin3_anim)
                vx_anim.append(vx_tmp)
                vz_anim.append(vz_tmp)
                vy_anim.append(vy_anim[-1])
        vx, vz = rotate(vx, vz, cos3, sin3)
        
        # Add in turbulence
        if animate:
            for i in range(mid_pad):
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
            for i in range(Nsteps):
                vx_anim.append(vx_anim[-1] + turbulence * np.sqrt(G * Mbh / r) * randnorm3[:,0] / float(Nsteps))
                vy_anim.append(vy_anim[-1] + turbulence * np.sqrt(G * Mbh / r) * randnorm4[:,0] / float(Nsteps))
                vz_anim.append(vz_anim[-1] + turbulence * np.sqrt(G * Mbh / r) * randnorm5[:,0] / float(Nsteps))
        vx += turbulence * np.atleast_2d(np.sqrt(G * Mbh / r)).T * randnorm3
        vy += turbulence * np.atleast_2d(np.sqrt(G * Mbh / r)).T * randnorm4
        vz += turbulence * np.atleast_2d(np.sqrt(G * Mbh / r)).T * randnorm5
        
        # Set so that moving away is positive velocity
        #vx = -vx
        #if animate:
            #vx_anim = np.array(vx_anim) * -1.0
            #vx_anim = list(vx_anim)

        if planeofsky:
            # Rotate in plane of sky
            if animate:
                for i in range(mid_pad):
                    vx_anim.append(vx_anim[-1])
                    vy_anim.append(vy_anim[-1])
                    vz_anim.append(vz_anim[-1])
                for i in range(Nsteps):
                    vy_tmp, vz_tmp = rotate(vy_anim[-1], vz_anim[-1], cos4_anim, sin4_anim)
                    vy_anim.append(vy_tmp)
                    vz_anim.append(vz_tmp)
                    vx_anim.append(vx_anim[-1])
            vy, vz = rotate(vy, vz, cos4, sin4)
        
        ###########################################################################################
        ##################################  Weights and Lags  #####################################
        ###########################################################################################
        # Compute the weights
        size = 0.5 + kappa * x/r
        size[far_side] *= xi
        size /= np.sum(size)
        if animate:
            size_step = (size - 1.0) / Nsteps
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
            for i in range(Nsteps):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
                size_anim.append(size_anim[-1] + size_step)
                label_anim.append('$\\xi~{\\rm and}~\\kappa$')

        # Compute the lags
        lags = r - x
        
        if animate:
            return x_anim, y_anim, z_anim, vx_anim, vy_anim, vz_anim, lags, size_anim, label_anim
        else:
            return x, y, z, vx, vy, vz, lags, size

    def buildModelVeryNew(self, params, Nclouds=1000, Nvpercloud=1, randoms={}, animate=False, anim_steps=20, mid_pad=5, planeofsky=False, plotting=False):
        # Set the parameters and random numbers
        if randoms != {}:
            rand1, rand2, rand4, rand0, randorder = randoms['rand1'], randoms['rand2'], randoms['rand4'],randoms['rand0'], randoms['randorder']
            Nclouds = len(rand1)
        else:
            rand0 = np.random.choice((-1.0,1.0),Nclouds)
            rand1 = np.random.uniform(0,1,Nclouds)
            rand2 = np.random.uniform(-1,1,Nclouds)
            rand4 = np.linspace(-1,1,Nclouds)
            randnorm1 = np.random.normal(0,1,Nclouds)
            randnorm2 = np.random.normal(0,1,Nclouds)
            randnorm3 = np.random.normal(0,1,(Nclouds,Nvpercloud))
            randnorm4 = np.random.normal(0,1,(Nclouds,Nvpercloud))
            randnorm5 = np.random.normal(0,1,(Nclouds,Nvpercloud))
            randorder = np.vstack([np.arange(Nclouds)]*Nvpercloud)
            for i in range(Nvpercloud):
                np.random.shuffle(randorder[i])
            np.random.shuffle(randorder)
            randoms = {'rand1':rand1, 'rand2':rand2, 'rand4':rand4, 'rand0':rand0, 'randorder':randorder}

        Mbh = 10**params['logMbh']
        beta, mu, F, rmax = params['beta'], params['mu'], params['F'], params['rmax']
        thetao, thetai = params['thetao'], params['thetai']
        gamma, kappa, xi = params['gamma'], params['kappa'], params['xi']
        fflow, fellip = params['fflow'], params['fellip']
        angular_sd_orbiting, angular_sd_flowing = params['angular_sd_orbiting'], params['angular_sd_flowing']
        radial_sd_orbiting, radial_sd_flowing = params['radial_sd_orbiting'], params['radial_sd_flowing']
        ellipseAngle = params['ellipseAngle']
        turbulence = params['turbulence']

        
        ###########################################################################################
        ######################################  Geometry  #########################################
        ###########################################################################################
        
        r = self.drawFromGammaNumpy(beta, mu, F, rmax, Nvals=len(rand1), granularity=4, plotting=plotting)
        
        # Determine the per-particle opening angles    
        part1 = np.sin(thetao*np.pi/180.)
        part2 = rand1**(1.0/gamma)
        angle = np.arcsin(part1 * part2)

        # Pre-calculate some sines and cosines
        sin1 = np.sin(angle)
        cos1 = np.cos(angle)
        sin2 = np.sin(rand2 * np.pi)
        cos2 = np.cos(rand2 * np.pi)
        sin3 = np.sin(0.5 * np.pi - thetai * np.pi / 180.)
        cos3 = np.cos(0.5 * np.pi - thetai * np.pi / 180.)
        sin4 = np.sin(rand4 * np.pi)
        cos4 = np.cos(rand4 * np.pi)

        if animate:
            Nsteps = anim_steps
            x_anim, y_anim, z_anim = [], [], []
            vx_anim, vy_anim, vz_anim = [], [], []
            size_anim = []
            label_anim = []
            sin1_anim = np.sin(angle/float(Nsteps))
            cos1_anim = np.cos(angle/float(Nsteps))
            sin2_anim = np.sin(rand2 * np.pi/float(Nsteps))
            cos2_anim = np.cos(rand2 * np.pi/float(Nsteps))
            sin3_anim = np.sin((0.5 * np.pi - thetai * np.pi / 180.)/float(Nsteps))
            cos3_anim = np.cos((0.5 * np.pi - thetai * np.pi / 180.)/float(Nsteps))
            sin4_anim = np.sin(rand4 * np.pi/float(Nsteps))
            cos4_anim = np.cos(rand4 * np.pi/float(Nsteps))

        # Set the positions
        x = r * rand0
        y = np.zeros(len(r))
        z = np.zeros(len(r))
        if animate:
            x_anim.append(list(x))
            y_anim.append(list(y))
            z_anim.append(list(z))
            size_anim.append([1.0]*len(x))
            label_anim.append('')
            vx_anim.append([0.0]*len(x))
            vy_anim.append([0.0]*len(x))
            vz_anim.append([0.0]*len(x))

        # Puff up by opening angle into wedge
        if animate:
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
            for i in range(Nsteps):
                x_tmp, z_tmp = rotate(x_anim[-1], z_anim[-1], cos1_anim, sin1_anim)
                x_anim.append(x_tmp)
                z_anim.append(z_tmp)
                y_anim.append(y_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append('${\\rm Opening~angle}$')
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
        x, z = rotate(x, z, cos1, sin1)

        # Rotate into thick disk
        if animate:
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
            for i in range(Nsteps):
                x_tmp, y_tmp = rotate(x_anim[-1], y_anim[-1], cos2_anim, sin2_anim)
                x_anim.append(x_tmp)
                y_anim.append(y_tmp)
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append('${\\rm Rotate~into~disk}$')
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
        x, y = rotate(x, y, cos2, sin2)
        
        # Truth vector to determine if on far side of mid-plane
        far_side = np.where(z < 0.)

        
        ###########################################################################################
        #####################################  Kinematics  ########################################
        ###########################################################################################
        # Compute the velocities
        G = 5.123619161  # ld / Msun * (km/s)^2
        
        vr = np.empty(len(r))
        vphi = np.empty(len(r))
        _theta = np.empty(len(r))
        
        # Circularlike orbits:
        startindex = int(fellip*len(r))
        vcirc = np.sqrt(G * Mbh / r)
        vesc = np.sqrt(2.0 * G * Mbh / r)   
        
        ang_mom_0 = np.zeros((len(x),3))
        pos = np.zeros((len(x),3))
        for i in range(len(x)):
            pos[i] = np.array([x[i],y[i],z[i]])
            pos[i] = pos[i] / np.linalg.norm(pos[i])
            ang_mom_0[i] = self.getAngMom0(pos[i])
        
        vx, vy, vz = np.zeros((len(x), Nvpercloud)), np.zeros((len(x), Nvpercloud)), np.zeros((len(x), Nvpercloud))
        for j in range(Nvpercloud):
            _circ = randorder[j][:startindex]
            _flow = randorder[j][startindex:]
            _theta[_circ] = 0.5*np.pi + angular_sd_orbiting*randnorm1[_circ]
            vr[_circ] = np.sqrt(2.0 * G * Mbh / r[_circ]) * np.cos(_theta[_circ]) * np.exp(radial_sd_orbiting*randnorm2[_circ])    
            vphi[_circ] = np.sqrt(G * Mbh / r[_circ]) * np.sin(_theta[_circ]) * np.exp(radial_sd_orbiting*randnorm2[_circ])
            
            # Inflow / outflow
            if fflow < 0.5:
                _theta[_flow] = np.pi - ellipseAngle*np.pi/180. + angular_sd_flowing*randnorm1[_flow]
                vr[_flow] = np.sqrt(2.0 * G * Mbh / r[_flow]) * np.cos(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow])
                vphi[_flow] = np.sqrt(G * Mbh / r[_flow]) * np.sin(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow])
            else:
                _theta[_flow] = 0.0 + ellipseAngle*np.pi/180. + angular_sd_flowing*randnorm1[_flow]
                vr[_flow] = np.sqrt(2.0 * G * Mbh / r[_flow]) * np.cos(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow])
                vphi[_flow] = np.sqrt(G * Mbh / r[_flow]) * np.sin(_theta[_flow]) * np.exp(radial_sd_flowing*randnorm2[_flow])
        
            for i in range(len(x)):
                ang_mom = self.drawAngMom(ang_mom_0[i], thetao * np.pi/180., pos[i])
                v = self.getVelocityUnitVector(pos[i], ang_mom)
                vx[i,j] = vr[i] * pos[i][0] + vphi[i] * v[0]
                vy[i,j] = vr[i] * pos[i][1] + vphi[i] * v[1]
                vz[i,j] = vr[i] * pos[i][2] + vphi[i] * v[2]
            
            if (plotting) and (j == 0):
                angles = np.linspace(0,2.*np.pi,360)
                plt.figure(figsize=(10,5))
                ax = plt.subplot2grid((2,4), (0,0), rowspan=2, colspan=2)
                ax_vr = plt.subplot2grid((2,4), (0,2), rowspan=1, colspan=2)
                ax_vphi = plt.subplot2grid((2,4), (1,2), rowspan=1, colspan=2)
                ax.plot(np.sqrt(2.0) * np.cos(angles), np.sqrt(2.0) * np.sin(angles), color='k')
                ax.plot(np.sqrt(2.0) * np.cos(angles), np.sin(angles), color='k', ls='dotted')
                ax.plot(np.cos(angles), np.sin(angles), color='k', ls='dashed')

                ax.scatter(
                    vr[_circ]/vcirc[_circ],
                    vphi[_circ]/vcirc[_circ],
                    alpha = 0.2,
                    color = 'blue'
                )
                ax.scatter(
                    vr[_flow]/vcirc[_flow],
                    vphi[_flow]/vcirc[_flow],
                    alpha = 0.2,
                    color = 'orange'
                )
                
                ax_vr.hist(vr[_circ], color='blue', bins=20, alpha=0.7, label="${\\rm Near-circular}$")
                ax_vphi.hist(vphi[_circ], color='blue', bins=20, alpha=0.7)
                ax_vr.hist(vr[_flow], color='orange', bins=20, alpha=0.7, label="${\\rm Inflow/Outflow}$")
                ax_vphi.hist(vphi[_flow], color='orange', bins=20, alpha=0.7)

                ax.set_xlabel('$v_r/v_{\\rm circ}$')
                ax.set_ylabel('$v_\\phi/v_{\\rm circ}$')
                ax.set_xlim(-1.7, 1.7)
                ax.set_ylim(-1.7, 1.7)
                ax.set_xticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])
                ax.set_yticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])

                ax_vr.legend()
                ax_vr.set_xlabel('$v_r~({\\rm km/s})$')
                ax_vphi.set_xlabel('$v_\\phi~({\\rm km/s})$')
                plt.tight_layout()
                plt.show()

        # Compute ang_mom_0
        if animate:
            l0_x = ang_mom_0[:,1] * pos[:,2] - ang_mom_0[:,2] * pos[:,1]
            l0_y = ang_mom_0[:,2] * pos[:,0] - ang_mom_0[:,0] * pos[:,2]
            l0_z = ang_mom_0[:,0] * pos[:,1] - ang_mom_0[:,1] * pos[:,0]
            vx_step = l0_x / Nsteps * np.sqrt(vr**2 + vphi**2)
            vy_step = l0_y / Nsteps * np.sqrt(vr**2 + vphi**2)
            vz_step = l0_z / Nsteps * np.sqrt(vr**2 + vphi**2)
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
            for i in range(Nsteps):
                vx_anim.append(vx_anim[-1] + vx_step)
                vy_anim.append(vy_anim[-1] + vy_step)
                vz_anim.append(vz_anim[-1] + vz_step)
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append('${\\rm Compute~}\\vec{L_0}$')
        
        # Draw final angular momentum vectors
        if animate:
            vx_step = (vx[:,0] - vx_anim[-1]) / Nsteps
            vy_step = (vy[:,0] - vy_anim[-1]) / Nsteps
            vz_step = (vz[:,0] - vz_anim[-1]) / Nsteps
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
            for i in range(Nsteps):
                vx_anim.append(vx_anim[-1] + vx_step)
                vy_anim.append(vy_anim[-1] + vy_step)
                vz_anim.append(vz_anim[-1] + vz_step)
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append('${\\rm Draw~}\\vec{L}$')

        # Rotate by inclination angle (+pi/2)
        x, z = rotate(x, z, cos3, sin3)
        # Rotate by inclination angle (+pi/2)
        vx, vz = rotate(vx, vz, cos3, sin3)

        # Rotate by inclination angle (+pi/2)
        if animate:
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
            for i in range(Nsteps):
                x_tmp, z_tmp = rotate(x_anim[-1], z_anim[-1], cos3_anim, sin3_anim)
                x_anim.append(x_tmp)
                z_anim.append(z_tmp)
                y_anim.append(y_anim[-1])
                vx_tmp, vz_tmp = rotate(vx_anim[-1], vz_anim[-1], cos3_anim, sin3_anim)
                vx_anim.append(vx_tmp)
                vz_anim.append(vz_tmp)
                vy_anim.append(vy_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append('${\\rm Inclination~angle}$')
        
        # Add in turbulence
        if animate:
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
            for i in range(Nsteps):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                vx_anim.append(vx_anim[-1] + turbulence * np.sqrt(G * Mbh / r) * randnorm3[:,0] / float(Nsteps))
                vy_anim.append(vy_anim[-1] + turbulence * np.sqrt(G * Mbh / r) * randnorm4[:,0] / float(Nsteps))
                vz_anim.append(vz_anim[-1] + turbulence * np.sqrt(G * Mbh / r) * randnorm5[:,0] / float(Nsteps))
                size_anim.append(size_anim[-1])
                label_anim.append('${\\rm Turbulence}$')

        # Add in turbulence
        vx += turbulence * np.atleast_2d(np.sqrt(G * Mbh / r)).T * randnorm3
        vy += turbulence * np.atleast_2d(np.sqrt(G * Mbh / r)).T * randnorm4
        vz += turbulence * np.atleast_2d(np.sqrt(G * Mbh / r)).T * randnorm5
        
        ###########################################################################################
        ##################################  Weights and Lags  #####################################
        ###########################################################################################
        # Compute the weights
        size = 0.5 + kappa * x/r
        size[far_side] *= xi
        size /= np.sum(size)
        if animate:
            size_step = (size - 1.0) / Nsteps
            for i in range(mid_pad):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
                size_anim.append(size_anim[-1])
                label_anim.append(label_anim[-1])
            for i in range(Nsteps):
                x_anim.append(x_anim[-1])
                y_anim.append(y_anim[-1])
                z_anim.append(z_anim[-1])
                vx_anim.append(vx_anim[-1])
                vy_anim.append(vy_anim[-1])
                vz_anim.append(vz_anim[-1])
                size_anim.append(size_anim[-1] + size_step)
                label_anim.append('$\\xi~{\\rm and}~\\kappa$')

        # Compute the lags
        lags = r - x
        
        if animate:
            return x_anim, y_anim, z_anim, vx_anim, vy_anim, vz_anim, lags, size_anim, label_anim
        else:
            return x, y, z, vx, vy, vz, lags, size

    def showModel(self, Nclouds=1000, trim=1, animate=False, anim_degrees=5, randoms={}, _style='scatter', savename='', elev=30., azim=-60., lim=0, figsize1=(12,10), figsize2=(8,4), show1=False, show_bicone=False):
        x, y, z = self.x[::trim], self.y[::trim], self.z[::trim]
        vx, vy, vz = self.vx[::trim], self.vy[::trim], self.vz[::trim]
        lags, size = self.lags[::trim], self.size[::trim]
        if len(vx.shape) == 1:
            vx, vy, vz = np.atleast_2d(vx).T, np.atleast_2d(vy).T, np.atleast_2d(vz).T
        
        fig = plt.figure(figsize=figsize1)
        ax = fig.gca(projection='3d', elev=elev, azim=azim)
        if lim == 0:
            lim = np.median(np.sqrt(x**2+y**2+z**2))*2.0
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
        ax.set_xlabel('$x~({\\rm ld})$')
        ax.set_ylabel('$y~({\\rm ld})$')
        ax.set_zlabel('$z~({\\rm ld})$')

        if show_bicone != False:
            thetao = show_bicone * np.pi/180.
            rmax = lim * 1.0
            rs = np.linspace(-rmax, rmax, 20)
            phis = np.linspace(0, 2.*np.pi, 30)
            rs, phis = np.meshgrid(rs, phis)
            x_tmp = rs * np.cos(phis)
            y_tmp = rs * np.sin(phis)
            z_tmp = rs * np.tan(thetao)

            # Plot the surface.
            bicone = ax.plot_surface(x_tmp, y_tmp, z_tmp, linewidth=0, antialiased=False, alpha=0.2)

        length = 0.5e-3
        sizescale = 30. * len(size)
        minmax = 0.5*np.max(np.sqrt(vx**2+vy**2+vz**2))
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-minmax, vmax=minmax), cmap="coolwarm_r")
        #sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=np.max(lags)), cmap="Reds")
        scat3D = ax.scatter3D(
            x, y, z,
            s=np.array(size)*sizescale,
            c=sm.to_rgba(vx[:,0]),
            alpha=0.8
        )
        if _style == 'quiver':
            for i in range(len(vx[0])):
                quivers = ax.quiver(
                    x, y, z,
                    vx[:,i]*length, vy[:,i]*length, vz[:,i]*length,
                    length=1.0,
                    arrow_length_ratio=0.02,
                    color=sm.to_rgba(vx[:,i]),
                    alpha=0.5,
                )
        if animate:
            def update(i):
                ax.view_init(elev=30., azim=anim_degrees*i)

            ani = animation.FuncAnimation(fig, update, frames=360/anim_degrees, interval=20, blit=False)
            ani.save(savename, writer='imagemagick', fps=20) 

            url = "file://" + os.path.abspath(savename)
            try:
                browser_path = 'open -a /Applications/Brave\ Browser.app %s'
                webbrowser.get(browser_path).open(url)
            except:
                pass
        else:
            if savename != '':
                plt.savefig(savename)
            plt.show()
            if show1:
                return
            fig,ax = plt.subplots(1,2,figsize=figsize2, sharey=True)
            ax[0].set_xlim([-lim, lim])
            ax[0].set_ylim([-lim, lim])
            ax[1].set_xlim([-lim, lim])
            ax[1].set_ylim([-lim, lim])
            ax[0].set_xlabel('$x~({\\rm ld})$')
            ax[0].set_ylabel('$z~({\\rm ld})$')
            ax[1].set_xlabel('$y~({\\rm ld})$')

            sizescale = 30. * len(size)
            minmax = 0.5*np.max(np.sqrt(vx**2+vy**2+vz**2))
            sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-minmax, vmax=minmax), cmap="coolwarm_r")
            ax[0].scatter(
                x,
                z,
                s=np.array(size)*sizescale,
                c=sm.to_rgba(vx[:,0]),
                alpha=0.8
            )
            ax[1].scatter(
                y,
                z,
                s=np.array(size)*sizescale,
                c=sm.to_rgba(vx[:,0]),
                alpha=0.8
            )
            ax[0].set_aspect(1.0)
            ax[1].set_aspect(1.0)
            plt.tight_layout(w_pad=0.0)
            plt.show()

    def animateBuild(self, params, _type='verynew', Nclouds=1000, anim_steps=30, end_pad=20, mid_pad=20, randoms={}, _style='scatter', savename='animations/buildModel.gif', elev=[20.], azim=[-70.], figsize=()):
        if _type == 'verynew':
            x, y, z, vx, vy, vz, _, size, label = self.buildModelVeryNew(params, Nclouds=Nclouds, randoms=randoms, animate=True, mid_pad=mid_pad, anim_steps=anim_steps)
        elif _type == 'new':
            x, y, z, vx, vy, vz, _, size, label = self.buildModelNew(params, Nclouds=Nclouds, randoms=randoms, animate=True, mid_pad=mid_pad, anim_steps=anim_steps)
        elif _type == 'old':
            x, y, z, vx, vy, vz, _, size, label = self.buildModelOld(params, Nclouds=Nclouds, randoms=randoms, animate=True, mid_pad=mid_pad, anim_steps=anim_steps)

        for j in range(end_pad):
            x.append(x[-1])
            y.append(y[-1])
            z.append(z[-1])
            vx.append(vx[-1])
            vy.append(vy[-1])
            vz.append(vz[-1])
            size.append(size[-1])
            label.append("${\\rm Final~model}$")
        x, y, z = np.array(x), np.array(y), np.array(z)
        vx, vy, vz = np.array(vx), np.array(vy), np.array(vz)
        

        length = 0.5e-3
        sizescale = 30.
        minmax = 0.5*np.max(np.sqrt(vx**2+vy**2+vz**2))
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-minmax, vmax=minmax), cmap="coolwarm")
        scat3D = [0]*len(elev)
        quivers = [0]*len(elev)
        if figsize == ():
            figsize = (6*len(elev),5)
        fig,ax = plt.subplots(1,len(elev), figsize=figsize)
        if len(elev) == 1:
            ax = [ax]
        for i in range(len(elev)):
            ax[i] = plt.subplot(1,len(elev),i+1, projection='3d', elev=elev[i], azim=azim[i])
            lim = np.median(np.sqrt(x**2+y**2+z**2))*2.0
            ax[i].set_xlim([-lim, lim])
            ax[i].set_ylim([-lim, lim])
            ax[i].set_zlim([-lim, lim])
            ax[i].set_xlabel('$x~({\\rm ld})$')
            ax[i].set_ylabel('$y~({\\rm ld})$')
            ax[i].set_zlabel('$z~({\\rm ld})$')
        plt.tight_layout(pad=0)
        for i in range(len(elev)):
            scat3D[i] = ax[i].scatter3D(
                x[0], y[0], z[0],
                s=np.array(size[0])*sizescale,
                c=sm.to_rgba(vx[0]),
                alpha=0.8
            )
            if _style == 'quiver':
                quivers[i] = ax[i].quiver(
                    x[0], y[0], z[0],
                    vx[0]*length, vy[0]*length, vz[0]*length,
                    length=1.0,
                    arrow_length_ratio=0.02,
                    color=sm.to_rgba(vx[0]),
                    alpha=0.5,
                )
        text = ax[0].text2D(0.05, 0.95, "", transform=ax[0].transAxes, size=20)
        def update(i):
            for j in range(len(elev)):
                scat3D[j]._offsets3d = np.array([x[i], y[i], z[i]])
                scat3D[j].set_sizes(np.array(size[i]) * sizescale)
                scat3D[j]._facecolor3d = sm.to_rgba(vx[i])
                scat3D[j]._edgecolor3d = sm.to_rgba(vx[i])
                if _style == 'quiver':
                    segs = np.array([
                        x[i], y[i], z[i],
                        x[i] + vx[i]*length,
                        y[i] + vy[i]*length,
                        z[i] + vz[i]*length
                    ]).reshape(6,-1)
                    new_segs = [[[_x,_y,_z],[_u,_v,_w]] for _x,_y,_z,_u,_v,_w in zip(*segs.tolist())]
                    quivers[j].set_segments(new_segs)
                    quivers[j].set_color(sm.to_rgba(vx[i]))
            text.set_text(label[i])

        ani = animation.FuncAnimation(fig, update, frames=len(x), interval=20, blit=False)
        ani.save(savename, writer='imagemagick', fps=20)    

        url = "file://" + os.path.abspath(savename)
        try:
            browser_path = 'open -a /Applications/Brave\ Browser.app %s'
            webbrowser.get(browser_path).open(url)
        except:
            pass

    def saveClouds(self, save_dir='', lcen=4861.33):
        print "Using lcen = 4861.33 to compute wavelengths"

        # Compute the wavelengths
        self.lams = lcen * (1 + -self.vx/c_km)

        clouds = []
        for i in range(len(self.x)):
            for j in range(len(self.vx[i])):
                clouds.append([
                    self.x[i] * 86400. * c_km * 1000.,
                    self.y[i] * 86400. * c_km * 1000.,
                    self.z[i] * 86400. * c_km * 1000.,
                    self.vx[i,j] * 1000.,
                    self.vy[i,j] * 1000.,
                    self.vz[i,j] * 1000.,
                    self.lams[i,j],
                    self.lags[i] * 86400.,
                    self.size[i]
                ])
        clouds = np.array(clouds)
        np.savetxt(save_dir + 'clouds.txt', clouds)

class MockData:
    def __init__(self, save_dir='', continuum_fnc='default', cont_times=[], spec_times=[], cont_err_frac=0.02, spec_err_frac=0.05, wavelengths=[], model=None, blurring=None, blurring_uncertainty=None, narrow_line_width=None, narrow_line_width_uncertainty=None):
        if save_dir != '':
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        self.save_dir = save_dir
        if cont_times == []:
            cont_times = np.arange(0.,100,1.0)
        if spec_times == []:
            spec_times = np.arange(30,70,1.0)
            #spec_times = [20, 22, 24, 28, 30, 31, 33, 34, 37, 38, 39, 40, 42, 44, 46, 48, 49, 51, 52, 53, 55, 57, 60, 62, 63, 65, 67, 68, 70]
            #spec_times = np.random.choice(np.arange(20,60,0.2), 15, replace=False)
            #spec_times.sort()

        if continuum_fnc == 'default':
            self.continuum_fnc = self.default_continuum_fnc
        else:
            self.continuum_fnc = continuum_fnc
        self.cont_times = cont_times
        self.spec_times = spec_times
        self.cont_err_frac = cont_err_frac
        self.spec_err_frac = spec_err_frac
        self.wavelengths = wavelengths
        self.type = 'python'
        if blurring != None:
            self.blurring = blurring
        if blurring_uncertainty != None:
            self.blurring_uncertainty = blurring_uncertainty
        if narrow_line_width != None:
            self.narrow_line_width = narrow_line_width
        if narrow_line_width_uncertainty != None:
            self.narrow_line_width_uncertainty = narrow_line_width_uncertainty

        if model != None:
            self.model = model

    @classmethod
    def fromCARAMEL(cls, run_dir, fn_output, fn_continuum='', fn_times='', fn_spectra='', fn_clouds='', fn_sigma=''):
        nparams = 27
        nhyperparams = 7
        with open(run_dir + 'Constants.cpp') as f:
            constants = f.readlines()
        consts = {
            'continuumExtrapolationFront': None,
            'continuumExtrapolationBack': None,
            'numVelocitiesPerCloud': None,
            'narrowline_width ': None,
            'narrowline_width_uncertainty': None,
        }
        for line in constants:
            for key in consts.keys():
                if key in line:
                    end_index = line.index(key) + len(key)
                    tmp = line[end_index:]
                    for char in [' ','=',';','\n']:
                        tmp = tmp.replace(char,'')
                    consts[key] = float(tmp)
        
        if fn_times == '':
            times = np.loadtxt(run_dir + 'Data/times.txt') * np.array([1.0/86400.,1.0,1.0])
        if fn_continuum == '':
            cont_data = np.loadtxt(run_dir + 'Data/continuum.txt') * np.array([1.0/86400.,1.0,1.0])
        if fn_spectra == '':
            spec_data = np.loadtxt(run_dir + 'Data/spectra.txt')
        if fn_sigma == '':
            sigma = np.loadtxt(run_dir + 'Data/sigma.txt')

        num_epochs = len(times)
        num_params_before_rainbow = nparams + num_epochs
        
        mod = np.loadtxt(run_dir + fn_output)
        
        num_wave_bins = spec_data.shape[1]
        
        spec = mod[num_params_before_rainbow:num_params_before_rainbow+num_wave_bins*num_epochs].reshape((num_epochs, num_wave_bins))
        cont = mod[num_params_before_rainbow+num_wave_bins*num_epochs+nhyperparams:]
        t_cont_start = cont_data[0,0] - consts['continuumExtrapolationBack'] * (cont_data[-1,0] - cont_data[0,0])
        t_cont_end = cont_data[-1,0] + consts['continuumExtrapolationFront'] * (cont_data[-1,0] - cont_data[0,0])

        dset = cls()
        dset.spec = spec
        dset.err = np.zeros(spec.shape)
        dset.sigma = sigma
        dset.cont = np.vstack([np.linspace(t_cont_start, t_cont_end, 1000), cont, 0.0*cont]).T
        dset.cont_data = cont_data
        dset.wave = spec_data[0]
        dset.type = 'caramel'
        dset.narrow_line_width = consts['narrowline_width ']
        dset.narrow_line_width_uncertainty = consts['narrowline_width_uncertainty']

        if fn_clouds != '':
            dset.model = Model.fromCARAMEL(run_dir, fn_clouds)
            dset.model.Nvpercloud = consts['numVelocitiesPerCloud']

        return dset

    def default_continuum_fnc(self, x):
        offset = 6.0
        A1, A2, A3, A4 = 2.0, 1.4, 0.9, 0.4
        p1, p2, p3, p4 = 60., 25., 14., 4.
        return offset + A1 * np.sin(2. * np.pi * x / p1) + A2 * np.cos(2. * np.pi * x / p2) + A3 * np.sin(2. * np.pi * x / p3) + A4 * np.cos(2. * np.pi * x / p4)

    def makeContinuumData(self, continuum_fnc=None, times=[], err_frac=0.02):
        if err_frac != 0.02:
            self.cont_err_frac == err_frac
        if continuum_fnc == None:
            continuum_fnc = self.continuum_fnc
        if times != []:
            self.cont_times = times
        
        cont = []
        vals_noerr = []
        for t in self.cont_times:
            vals_noerr = continuum_fnc(t)
        err_val = np.median(vals_noerr) * self.cont_err_frac
        for t in self.cont_times:
            val = continuum_fnc(t) + err_val * np.random.normal()
            cont.append([t, val, err_val])
        self.cont = np.array(cont)

    def makeSpectraData(self, wavelengths=[], continuum_fnc=None, times=[], model=None, lcen=4861.33, err_frac=0.05, relativity=True, Cadd=None, Cmult=None, blurring=None, blurring_uncertainty=None, narrow_line_width=None, narrow_line_width_uncertainty=None):
        if wavelengths != []:
            self.wavelengths = wavelengths
        if err_frac != 0.05:
            self.spec_err_frac == err_frac
        if continuum_fnc == None:
            continuum_fnc = self.continuum_fnc
        if model != None:
            self.model = model
        if times != []:
            self.spec_times = times
        x, y, z = self.model.x, self.model.y, self.model.z
        vx, vy, vz = self.model.vx, self.model.vy, self.model.vz
        r = np.sqrt(x**2 + y**2 + z**2)
        if len(vx.shape) == 1:
            vx = np.atleast_2d(vx).T
            vy = np.atleast_2d(vy).T
            vz = np.atleast_2d(vz).T
        lags, size = self.model.lags, self.model.size
        
        if relativity:
            G_msun = 5.12362  # ld * (km/s)^2 / Msun
            Mbh_msun = 10**self.model.params['logMbh']
            Rs = 2.0 * G_msun * Mbh_msun / c_km**2
            # Special relativity radial velocity redshift
            factor1 = np.sqrt((1.0 - vx/c_km) / (1.0 + vx/c_km))
            # General relativity gravitational redshift
            factor2 = 1.0/np.sqrt(1.0 - Rs/r)
            factor2 = np.repeat(np.atleast_2d(factor2),np.shape(factor1)[1],axis=0).T
            wave = factor1 * factor2 * lcen
        else:
            wave = lcen * (1 + -vx/c_km)

        if self.wavelengths == []:
            wave_min, wave_max = np.min(wave), np.max(wave)
            halfwidth = max(lcen - wave_min, wave_max - lcen)
            wavelength_min = np.floor((lcen - 1.2*halfwidth)/10.0) * 10.0
            wavelength_max = np.ceil((lcen + 1.2*halfwidth)/10.0) * 10.0
            self.wavelengths = np.arange(wavelength_min,wavelength_max,2)
        dwave = self.wavelengths[1]-self.wavelengths[0]
        wavelength_bins = np.arange(self.wavelengths[0]-dwave/2., self.wavelengths[-1]+dwave+dwave/2., dwave)
        
        N_wave = len(self.wavelengths)
        N_epochs = len(self.spec_times)

        spec_bins = np.digitize(wave, bins=wavelength_bins) - 1
        spectra = np.zeros((N_epochs, N_wave))
        err = np.zeros((N_epochs, N_wave))

        if Cadd == None:
            if 'Cadd' in self.model.params.keys():
                Cadd = self.model.params['Cadd']
            else:
                Cadd = 0.0
        if Cmult == None:
            if 'Cmult' in self.model.params.keys():
                Cmult = self.model.params['Cmult']
            else:
                Cmult = 1.0

        # TODO: Add as params
        if narrow_line_width == None:
            if not hasattr(self, 'narrow_line_width'):
                self.narrow_line_width = 0.0
        else:
            self.narrow_line_width = narrow_line_width
        if narrow_line_width_uncertainty == None:
            if not hasattr(self, 'narrow_line_width_uncertainty'):
                self.narrow_line_width_uncertainty = 0.0
        else:
            self.narrow_line_width_uncertainty = narrow_line_width_uncertainty

        narrow_blurring = -1
        while narrow_blurring < 0:
            narrow_blurring = self.narrow_line_width + np.random.normal() * self.narrow_line_width_uncertainty

        if not hasattr(self, 'sigma'):
            self.sigma = np.zeros((N_epochs,3))

        C = Cmult / self.model.Nvpercloud
        for i in range(N_epochs):
            cont_meas = continuum_fnc(self.spec_times[i] - lags)
            for j in range(len(spec_bins)):
                try:
                    for k in range(len(spec_bins[j])):
                        spectra[i,spec_bins[j,k]] += (cont_meas[j] + Cadd) * size[j]
                except:
                    pass
            spectra[i] *= C
            
            if blurring != None:
                self.sigma[i,1] = blurring
            if blurring_uncertainty != None:
                self.sigma[i,2] = blurring_uncertainty

            tmp_blurring = max(0.0, self.sigma[i,1] + np.random.normal() * self.sigma[i,2])

            instrumental_resolution = max(0.0,np.sqrt(tmp_blurring**2 - narrow_blurring**2))
            if instrumental_resolution != 0.0:
                spectra[i] = gaussian_filter1d(spectra[i], instrumental_resolution)
            err_max = max(spectra[i] * self.spec_err_frac)
            for j in range(N_wave):
                err[i,j] = err_max
                spectra[i,j] += np.random.normal() * err_max
        
        self.wave = self.wavelengths
        self.spec = spectra
        self.err = err
        self.lcen = lcen

    def showModel(self, **kwargs):
        self.model.showModel(**kwargs)
        pass

    def plotData(self):
        N_epochs = len(self.spec)
        N_wave = len(self.wave)

        fontsize=16
        aspect='auto'
        extent = (self.wave[0], self.wave[-1], N_epochs, 0)

        gridspec = dict(hspace=0.0, height_ratios=[1, 1, 1, 0.6, 1, 1])
        fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(5.5,8), gridspec_kw=gridspec)
        ax[0].imshow(self.spec, aspect=aspect, extent=extent)
        ax[1].errorbar(self.wave, self.spec[0], self.err[0])
        ax[1].errorbar(self.wave, self.spec[5], self.err[5])
        for i in range(len(self.spec)):
            ax[2].plot(self.wave, self.spec[i])
        
        ax[3].set_visible(False)

        if self.type == 'caramel':
            ax[4].plot(self.spec_times, np.sum(self.spec, axis=1))
            ax[5].plot(self.cont[:,0], self.cont[:,1], lw=0.2)
            ax[5].errorbar(self.cont_data[:,0], self.cont_data[:,1], self.cont_data[:,2], ls='')
        else:
            ax[4].errorbar(self.spec_times, np.sum(self.spec, axis=1), np.sqrt(np.sum(self.err**2, axis=1)), ls='')
            ax[5].errorbar(self.cont[:,0], self.cont[:,1], self.cont[:,2], ls='')
        
        xlim = [min(ax[4].get_xlim()[0], ax[5].get_xlim()[0]), max(ax[4].get_xlim()[1], ax[5].get_xlim()[1])]
        ax[4].set_xlim(xlim)
        ax[5].set_xlim(xlim)

        labelfontsize=14
        ax[0].set_xticklabels([], visible=False)
        ax[1].set_xticklabels([], visible=False)
        ax[2].set_xlabel("${\\rm Wavelength~(Ang)}$", fontsize=labelfontsize)
        
        ax[4].set_xticklabels([], visible=False)
        ax[5].set_xlabel("${\\rm Date}$", fontsize=labelfontsize)
        
        ax[0].set_ylabel("${\\rm Epoch}$", fontsize=labelfontsize)
        ax[1].set_ylabel("${\\rm Flux~(arb)}$", fontsize=labelfontsize)
        ax[2].set_ylabel("${\\rm Flux~(arb)}$", fontsize=labelfontsize)
        ax[4].set_ylabel("${\\rm Flux~(arb)}$", fontsize=labelfontsize)
        ax[5].set_ylabel("${\\rm Flux~(arb)}$", fontsize=labelfontsize)

        #plt.tight_layout()
        #plt.savefig('/Users/peterwilliams/Desktop/python_vs_caramel_mockdata.pdf')
        plt.show()

    def saveData(self, save_dir='', suffix='', fn_cont='', fn_spec='', fn_times='', fn_sigma=''):
        if save_dir != '':
            self.save_dir = save_dir
        if (self.save_dir != '') and (not os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)
        if fn_cont == '':
            fn_cont = 'continuum%s.txt' % (suffix)
        if fn_spec == '':
            fn_spec = 'spectra%s.txt' % (suffix)
        if fn_times == '':
            fn_times = 'times%s.txt' % (suffix)
        if fn_sigma == '':
            fn_sigma = 'sigma%s.txt' % (suffix)

        cont_lc_sec = self.cont * np.array([86400., 1.0, 1.0])
        times_sec = np.vstack([self.spec_times, np.zeros(len(self.spec_times)), np.zeros(len(self.spec_times))]).T * np.array([86400., 1.0, 1.0])
        spec_tosave = []
        spec_tosave.append(self.wave)
        for i in range(len(self.spec)):
            spec_tosave.append(self.spec[i])
            spec_tosave.append(self.err[i])
        
        np.savetxt(self.save_dir + fn_cont, cont_lc_sec)
        np.savetxt(self.save_dir + fn_spec, spec_tosave, header="%i" % len(self.wave))
        np.savetxt(self.save_dir + fn_times, times_sec)

        self.sigma[:,0] = times_sec[:,0]
        np.savetxt(self.save_dir + fn_sigma, self.sigma)

        with open(self.save_dir + 'params.json', 'w') as f:
            json.dump(self.model.params, f)
        
        self.model.saveClouds(save_dir=self.save_dir, lcen=self.lcen)


def compareDataSets(dset1, dset2, labels=['',''], figsize=(5.5,8), savename=''):
    N_epochs = len(dset1.spec)
    N_wave = len(dset1.wave)

    fontsize=16
    aspect='auto'
    extent = (dset1.wave[0], dset1.wave[-1], N_epochs, 0)

    gridspec = dict(hspace=0.0, height_ratios=[1, 1, 1, 0.6, 1, 0.6, 1, 1])
    fig, ax = plt.subplots(nrows=8, ncols=1, figsize=figsize, gridspec_kw=gridspec)
    ax[0].imshow(dset1.spec, aspect=aspect, extent=extent)
    ax[1].imshow(dset2.spec, aspect=aspect, extent=extent)
    ax[2].imshow((dset1.spec - dset2.spec)/np.sqrt(dset1.err**2 + dset2.err**2), aspect=aspect, extent=extent)
    
    ax[3].set_visible(False)
    resids = ((dset1.spec - dset2.spec)/np.sqrt(dset1.err**2 + dset2.err**2)).flatten()
    ax[4].hist(resids, bins=40, color='grey')
    ax[4].axvline(np.percentile(resids,50-68.27/2.), ls='dashed', color='blue', label='${68.3\\%}$')
    ax[4].axvline(np.percentile(resids,50+68.27/2.), ls='dashed', color='blue')
    ax[4].axvline(np.percentile(resids,50-95.45/2.), ls='dashed', color='green', label='${95.5\\%}$')
    ax[4].axvline(np.percentile(resids,50+95.45/2.), ls='dashed', color='green')
    ax[4].axvline(np.percentile(resids,50-99.73/2.), ls='dashed', color='orange', label='${99.7\\%}$')
    ax[4].axvline(np.percentile(resids,50+99.73/2.), ls='dashed', color='orange')
    xlim_max = max(np.abs(ax[4].get_xlim()))
    ax[4].set_xlim(-xlim_max, xlim_max)
    ax[4].legend(loc='upper left')

    ax[5].set_visible(False)

    if dset1.type == 'caramel':
        ax[6].plot(dset1.spec_times, np.sum(dset1.spec, axis=1), label=labels[0])
    else:
        ax[6].errorbar(dset1.spec_times, np.sum(dset1.spec, axis=1), np.sqrt(np.sum(dset1.err**2, axis=1)), ls='', label=labels[0])
    if dset2.type == 'caramel':
        ax[6].plot(dset2.spec_times, np.sum(dset2.spec, axis=1), label=labels[1])    
    else:
        ax[6].errorbar(dset2.spec_times, np.sum(dset2.spec, axis=1), np.sqrt(np.sum(dset2.err**2, axis=1)), ls='', label=labels[1])
    ax[6].legend(loc = 'upper left')
    
    if dset1.type == 'caramel':
        ax[7].plot(dset1.cont[:,0], dset1.cont[:,1], lw=0.2, color='k', label=labels[0] + '$~{\\rm (Gaussian~process)}$')
        ax[7].errorbar(dset1.cont_data[:,0], dset1.cont_data[:,1], dset1.cont_data[:,2], ls='', label=labels[0] + '$~{\\rm (data)}$')
    else:
        ax[7].errorbar(dset1.cont[:,0], dset1.cont[:,1], dset1.cont[:,2], ls='', label=labels[0])

    if dset2.type == 'caramel':
        ax[7].plot(dset2.cont[:,0], dset2.cont[:,1], lw=0.2, color='k', label=labels[1] + '$~{\\rm (Gaussian~process)}$')
        ax[7].errorbar(dset2.cont_data[:,0], dset2.cont_data[:,1], dset2.cont_data[:,2], ls='', label=labels[1] + '$~{\\rm (data)}$')
    else:
        ax[7].errorbar(dset2.cont[:,0], dset2.cont[:,1], dset2.cont[:,2], ls='', label=labels[0])
    ax[7].legend(loc = 'upper left')

    xlim = [min(ax[6].get_xlim()[0], ax[7].get_xlim()[0]), max(ax[6].get_xlim()[1], ax[7].get_xlim()[1])]
    ax[6].set_xlim(xlim)
    ax[7].set_xlim(xlim)

    labelfontsize=14

    addText(labels[0], ax[0], loc=(0.02,0.05), color='white', ha='left', va='top', fontsize=labelfontsize*1.1)
    addText(labels[1], ax[1], loc=(0.02,0.05), color='white', ha='left', va='top', fontsize=labelfontsize*1.1)
    addText('${\\rm Normalized~Residual}$', ax[2], loc=(0.02,0.05), color='white', ha='left', va='top', fontsize=labelfontsize*1.1)

    ax[0].set_xticklabels([], visible=False)
    ax[1].set_xticklabels([], visible=False)
    ax[2].set_xlabel("${\\rm Wavelength~(Ang)}$", fontsize=labelfontsize)
    
    ax[4].set_xlabel('${\\rm Normalized~Residual}$', fontsize=labelfontsize)
    
    ax[6].set_xticklabels([], visible=False)
    ax[7].set_xlabel("${\\rm Date}$", fontsize=labelfontsize)
    
    ax[0].set_ylabel("${\\rm Epoch}$", fontsize=labelfontsize)
    ax[1].set_ylabel("${\\rm Epoch}$", fontsize=labelfontsize)
    ax[2].set_ylabel("${\\rm Epoch}$", fontsize=labelfontsize)
    ax[6].set_ylabel("${\\rm Flux~(arb)}$", fontsize=labelfontsize)
    ax[7].set_ylabel("${\\rm Flux~(arb)}$", fontsize=labelfontsize)

    if savename != '':
        plt.savefig(savename)
    plt.show()
