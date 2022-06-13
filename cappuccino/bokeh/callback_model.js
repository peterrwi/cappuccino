const data = source.data;
const thetai = thetai1.value;
const thetao = thetao1.value;
const gamma = gamma1.value;
const kap = kappa1.value;
const beta = beta1.value;
const mu = mu1.value;
const F = F1.value;
const xi = xi1.value;
const fflow = fflow1.value;
const fellip = fellip1.value;
const Mbh = Math.pow(10, mbh1.value);

const r = data['r'];
const rand1 = data['rand1'];
const rand2 = data['rand2'];
const rand3 = data['rand3'];
const randorder = data['randorder'];
const phi = data['phi'];
const x = data['x'];
const y = data['y'];
const z = data['z'];
const size = data['size'];
const lags = data['lags'];
const vx = data['vx'];
const gamma_x = data['gamma_x'];
const gamma_y = data['gamma_y'];

const alpha = Math.pow(beta,-2.0);
const rmin = mu*F;
const theta = (mu - rmin)/alpha;

const rmax = 500.;

// Draw values from the gamma function
// First create list of x-values
var xvals = [];
for (var i = 0; i < r.length; i++) {
    xvals[i] = i/r.length*rmax;
}
// Compute the gamma function
var cdf = [];
var p1;
for (var i = 0; i < xvals.length; i++) {
    p1 = Math.log(xvals[i] - rmin) * (alpha - 1.0);
    cdf[i] = Math.exp(p1-(xvals[i]-rmin)/theta);
    if (Number.isNaN(cdf[i])) {
        cdf[i] = 0.0;
    } else if (cdf[i] < 0.0) {
        cdf[i] = 0.0;
    }
}
var sumvals = 0.;
for (var i = 0; i < cdf.length; i++) {
    sumvals += cdf[i];
}
// Re-normalize and compute the cumulative density function
for (var i = 0; i < xvals.length; i++) {
    cdf[i] /= sumvals;
    gamma_x[i] = xvals[i];
    gamma_y[i] = cdf[i];
    if (i > 0) {
        cdf[i] += cdf[i-1];
    }
}
// Use random values from 0-1 to draw from CDF
var run_loop;
var j;
for (var i = 0; i < rand3.length; i++) {   // Loop over rand1 values
    j = 0;
    run_loop = 1;
    while (run_loop == 1) {
        if (j >= cdf.length) {             // If we reach end of cdf, store rest as last xvals value
            r[i] = xvals[xvals.length-1];
            run_loop = 0;
        } else if (cdf[j] > rand3[i]) {    // If cdf > val, save x-val
            r[i] = xvals[j];
            run_loop = 0;
        } else {
            j += 1;
        }
    }
}

// Begin assembling
const G = 5.123619161;

function rotate(x, y, C, S) {
    return [C*x + S*y, -S*x + C*y];
}

var cos1, cos2, cos3, sin1, sin2, sin3, sinPhi, cosPhi;
var vy, vz;
var vr = 0;
var vphi = 0;
var part1, part2, part3, angle;
var far_side = [];
cos3 = Math.cos(0.5 * Math.PI - thetai * Math.PI / 180.);
sin3 = Math.sin(0.5 * Math.PI - thetai * Math.PI / 180.);

// First loop over near-circular points
const dynamics_index = Math.floor(r.length * fellip);
for (var index = 0; index < dynamics_index; index++) {
    var i = randorder[index]
    part1 = Math.sin(thetao * Math.PI / 180.);
    part2 = Math.exp(1.0 / gamma * Math.log(rand1[i]));
    angle = Math.asin(part1 * part2);
    
    sin1 = Math.sin(angle);
    cos1 = Math.cos(angle);
    sin2 = Math.sin(rand2[i] * Math.PI);
    cos2 = Math.cos(rand2[i] * Math.PI);
    sinPhi = Math.sin(phi[i]);
    cosPhi = Math.cos(phi[i]);
    
    x[i] = r[i] * phi[i];
    y[i] = 0.;
    z[i] = 0.;
    
    [x[i],z[i]] = rotate(x[i], z[i], cos1, sin1);
    [x[i],y[i]] = rotate(x[i], y[i], cos2, sin2);
    far_side[i] = z[i] < 0.0;
    [x[i],z[i]] = rotate(x[i], z[i], cos3, sin3);
    
    // Compute velocities
    vr = 0;
    vphi = Math.pow(G*Mbh/r[i], 0.5);

    vx[i] = vr * phi[i];
    vy = vphi * phi[i];
    vz = 0;
    
    [vx[i], vz] = rotate(vx[i], vz, cos1, sin1);
    // Rotate to restore axisymmetry
    [vx[i], vy] = rotate(vx[i], vy, cos2, sin2);
    // Inclination
    [vx[i], vz] = rotate(vx[i], vz, cos3, sin3);
    vx[i] = -vx[i];
}
for (var index = dynamics_index; index < r.length; index++) {
    var i = randorder[index]
    part1 = Math.sin(thetao * Math.PI / 180.);
    part2 = Math.exp(1.0 / gamma * Math.log(rand1[i]));
    angle = Math.asin(part1 * part2);
    sin1 = Math.sin(angle);
    cos1 = Math.cos(angle);
    sin2 = Math.sin(rand2[i] * Math.PI);
    cos2 = Math.cos(rand2[i] * Math.PI);
    sinPhi = Math.sin(phi[i]);
    cosPhi = Math.cos(phi[i]);
    
    x[i] = r[i] * phi[i];
    y[i] = 0.;
    z[i] = 0.;
    
    [x[i],z[i]] = rotate(x[i], z[i], cos1, sin1);
    [x[i],y[i]] = rotate(x[i], y[i], cos2, sin2);
    far_side[i] = z[i] < 0.0;
    [x[i],z[i]] = rotate(x[i], z[i], cos3, sin3);
    
    // Compute velocities
    if (fflow > 0) {
        vr = Math.pow(2.0*G*Mbh/r[i], 0.5);
        vphi = 0;
    } else {
        vr = -Math.pow(2.0*G*Mbh/r[i], 0.5);
        vphi = 0;
    }

    vx[i] = vr * phi[i];
    vy = vphi * phi[i];
    vz = 0;
    
    [vx[i], vz] = rotate(vx[i], vz, cos1, sin1);
    // Rotate to restore axisymmetry
    [vx[i], vy] = rotate(vx[i], vy, cos2, sin2);
    // Inclination
    [vx[i], vz] = rotate(vx[i], vz, cos3, sin3);
    vx[i] = -vx[i];
}

var sumweights = 0.;
for (var i = 0; i < r.length; i++) {
    size[i] = 0.5 + kap * x[i]/Math.sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i])
    if (Number.isNaN(size[i])) {
        size[i] = 0.0;
    }
    sumweights += size[i];
}
for (var i = 0; i < r.length; i++) {
    size[i] *= 1.0 * r.length / sumweights;
    if (far_side[i]) {
        size[i] *= xi
    }
}

// Compute lags
for (var i = 0; i < r.length; i++) {
    lags[i] = r[i] - x[i];
}

source.change.emit();