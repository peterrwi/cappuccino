import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def rotate(x, y, C, S):
    return C*x + S*y, -S*x + C*y

def buildModelPlaneOfSky(randoms, params):
    rand1, rand2, rand3, rand4, phi, randorder = randoms['rand1'], randoms['rand2'], randoms['rand3'], randoms['rand4'],randoms['phi'], randoms['randorder']
    beta, mu, F = params['beta'], params['mu'], params['F']
    thetao, thetai, gamma = params['thetao'], params['thetai'], params['gamma']
    kappa, Mbh = params['kappa'], 10**params['logMbh']
    xi = params['xi']
    fflow, fellip = params['fflow'], params['fellip']

    alpha = beta**-2.0
    rmin = mu * F
    theta = (mu - rmin) / alpha

    # Compute the gamma function
    rmax = 500.
    xvals = np.linspace(0,rmax,len(rand1))

    # Compute the gamma distribution from 0 to rmax, then normalize
    #gamma_distr = (xvals - rmin)**(alpha - 1.) * np.exp(-(xvals - rmin) / theta)
    #gamma_distr = np.where(gamma_distr > 0.0, gamma_distr, 0.0)
    #gamma_distr = gamma_distr / np.sum(gamma_distr)
    cdf = np.zeros(len(xvals))
    sumvals = 0.
    for i in range(len(xvals)):
        cdf[i] = (xvals[i] - rmin)**(alpha-1.) * np.exp(-(xvals[i]-rmin) / theta)
        if np.isnan(cdf[i]):
            cdf[i] = 0.
        elif cdf[i] < 0.0:
            cdf[i] = 0.
        sumvals += cdf[i]

    # Compute the cdf
    gamma_distr = np.zeros(len(cdf))
    for i in range(len(xvals)):
        cdf[i] /= sumvals
        gamma_distr[i] = cdf[i]
        if i > 0:
            cdf[i] += cdf[i-1]

    # Take sorted distribution of values 0-1 to draw from cdf
    j = 0
    r = np.zeros(len(rand3))
    for i in range(len(rand3)):
        j=0
        run_loop = True
        while run_loop:
            if j >= len(cdf):
                r[i] = xvals[-1]
                run_loop = False
            elif cdf[j] > rand3[i]:
                r[i] = xvals[j]
                run_loop = False
            else:
                j += 1

    # Determine the per-particle opening angles    
    part1 = np.sin(thetao*np.pi/180.)
    part2 = np.exp(1.0/gamma * np.log(rand1))
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
    
    #sinPhi = np.sin(phi)
    #cosPhi = np.cos(phi)

    ################################################
    # Set the positions
    # Turn radial distribution into a disk
    x = r * phi
    y = np.zeros(len(r))
    z = np.zeros(len(r))

    # Puff up by opening angle
    x, z = rotate(x, z, cos1, sin1)
    # Restore axi-symmetry
    x, y = rotate(x, y, cos2, sin2)
    # Truth vector to determine if on far side of mid-plane
    far_side = np.where(z < 0.)
    # Rotate by inclination angle (+pi/2)
    x, z = rotate(x, z, cos3, sin3)
    # Rotate in plane of sky
    y, z = rotate(y, z, cos4, sin4)
    
    ################################################
    # Compute the velocities
    G = 5.123619161  # ld / Msun * (km/s)^2
    
    # First put everything on a circular orbit
    vr = np.zeros(len(r))
    vphi = np.sqrt(G * Mbh / r)
    
    # Next turn (1-fellip) into inflow or outflow
    startindex = int(fellip*len(r))
    if fflow > 0:
        vr[randorder[startindex:]] = np.sqrt(2.0 * G * Mbh / r[randorder[startindex:]])
    else:
        vr[randorder[startindex:]] = -np.sqrt(2.0 * G * Mbh / r[randorder[startindex:]])
    vphi[randorder[startindex:]] = np.zeros(len(r)-startindex)

    # Convert to cartesian coordinates
    vx = vr * phi 
    vy = vphi * phi
    vz = np.zeros(len(r))

    # Puff up by opening angle
    vx, vz = rotate(vx, vz, cos1, sin1)
    # Restore axi-symmetry
    vx, vy = rotate(vx, vy, cos2, sin2)
    # Rotate by inclination angle (+pi/2)
    vx, vz = rotate(vx, vz, cos3, sin3)
    # Set so that moving away is positive velocity
    vx = -vx
    # Rotate in plane of sky
    vy, vz = rotate(vy, vz, cos4, sin4)
    
    ################################################
    # Compute the weights
    size = 0.5 + kappa * x/r
    size = np.where(np.isnan(size),0.0,size)
    size *= 1.0 * len(size) / np.sum(size)
    size[far_side[0]] *= xi
    
    ################################################
    # Compute the lags
    lags = r - x
    
    return x, y, z, vx, lags, size, xvals, gamma_distr

def main():
    # Set the data
    Ndata = 1000
    phi = np.random.choice((-1.0,1.0),Ndata)
    rand1 = np.random.uniform(-1,1,Ndata)
    rand2 = np.random.uniform(-1,1,Ndata)
    rand3 = np.linspace(0,1,Ndata)
    np.random.shuffle(rand3)
    rand4 = np.linspace(-1,1,Ndata)
    randorder = np.arange(Ndata)
    np.random.shuffle(randorder)
    randoms = {'rand1':rand1, 'rand2':rand2, 'rand3':rand3, 'rand4':rand4, 'phi':phi, 'randorder':randorder}

    start = {
        'thetai': 40.,
        'thetao': 30.,
        'kappa': -0.4,
        'beta': 0.85,
        'mu': 19.5,
        'F': 0.29,
        'logMbh': 7.9,
        'gamma': 1.5,
        'xi': 1.0,
        'fflow': 0.5,
        'fellip': 1.0,
    }

    x,y,z,vx,lags,size, gamma_x, gamma_y = buildModelPlaneOfSky(randoms, start)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, s=4)
    ax.set_xlim(-40,40)
    ax.set_ylim(-40,40)
    ax.set_zlim(-40,40)
    ax.set_xlabel('x (ld)')
    ax.set_ylabel('y (ld)')
    ax.set_zlabel('z (ld)')

    plt.show()

if __name__ == '__main__':
    main()