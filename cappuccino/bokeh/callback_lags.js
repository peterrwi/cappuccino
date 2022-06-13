const data = source.data;
const thetai = thetai1.value;
const thetao = thetao1.value;
const mu = mu1.value;
const F = F1.value;
const Mbh = Math.pow(10, mbh1.value);

const gamma = 3.1;
const kap = 0.0;
const beta = 0.4;

const r = data['r'];
const rand1 = data['rand1'];
const rand2 = data['rand2'];
const rand3 = data['rand3'];
const phi = data['phi'];
const x = data['x'];
const y = data['y'];
const z = data['z'];
const size = data['size'];
const lags = data['lags'];
const vx = data['vx'];

const spec = data['spec'];

const times = data['times'];
const ll = data['ll'];
const meanspec = data['meanspec'];

const fwhm = data['fwhm'];
const lag = data['lag'];
const vp = data['vp'];

const alpha = Math.pow(beta,-2.0);
const rmin = mu*F;
const theta = (mu - rmin)/alpha;

const rmax = 500.;

// Draw values from the gamma function
// First create list of x-values
var xvals = [];
for (var i = 0; i < 1000; i++) {
    xvals[i] = i/1000.0*rmax;
}
// Compute the gamma function
var cdf = [];
var p1;
for (var i = 0; i < xvals.length; i++) {
    //cdf[i] = Math.pow(xvals[i] - rmin, alpha-1.0) * Math.exp(-(xvals[i]-rmin)/theta);
    p1 = Math.log(xvals[i] - rmin) * (alpha - 1.0);
    cdf[i] = Math.exp(p1-(xvals[i]-rmin)/theta);
    if (Number.isNaN(cdf[i])) {
        cdf[i] = 0.0;
    } else if (cdf[i] < 0.0) {
        cdf[i] = 0.0
    }
}
sumvals = 0.;
for (var i = 0; i < cdf.length; i++) {
    sumvals += cdf[i];
}
// Re-normalize and compute the cumulative density function
for (var i = 0; i < xvals.length; i++) {
    cdf[i] /= sumvals;
    if (i > 0) {
        cdf[i] += cdf[i-1]
    }
}

// Use random values from 0-1 to draw from CDF
var run_loop = 1;
var j = 0;
for (var i = 0; i < rand1.length; i++) {   // Loop over rand1 values
    run_loop = 1;
    while (run_loop == 1) {
        if (j >= cdf.length) {             // If we reach end of cdf, store rest as last xvals value
            r[i] = xvals[xvals.length-1];
            run_loop = 0;
        } else if (cdf[j] > rand1[i]) {    // If cdf > val, save x-val
            r[i] = xvals[j];
            run_loop = 0;
        } else {
            j += 1;
        }
    }
}

function rotate(x, y, C, S) {
    return [C*x + S*y, -S*x + C*y];
}

const cos3 = Math.cos(0.5 * Math.PI - thetai * Math.PI / 180.);
const sin3 = Math.sin(0.5 * Math.PI - thetai * Math.PI / 180.);
var cos1, cos2, sin1, sin2, sinPhi, cosPhi;
var vr = 0;
var vphi = 0;
const G = 6.673E-11;
var vy, vz;
var part1, part2, part3, angle;

for (var i = 0; i < r.length; i++) {
    part1 = Math.cos(thetao * Math.PI / 180.);
    part2 = 1. - Math.cos(thetao * Math.PI / 180.);
    part3 = Math.exp(Math.log(rand2[i]) * gamma);
    angle = Math.acos(part1 + part2 * part3);
    sin1 = Math.sin(angle);
    cos1 = Math.cos(angle);
    sin2 = Math.sin(rand3[i] * Math.PI);
    cos2 = Math.cos(rand3[i] * Math.PI);
    sinPhi = Math.sin(phi[i]);
    cosPhi = Math.cos(phi[i]);
    
    x[i] = r[i] * cosPhi;
    y[i] = r[i] * sinPhi;
    z[i] = 0.;
    
    [x[i],z[i]] = rotate(x[i], z[i], cos1, sin1);
    [x[i],y[i]] = rotate(x[i], y[i], cos2, sin2);
    [x[i],z[i]] = rotate(x[i], z[i], cos3, sin3);
    
    
    // Compute velocities
    //theta = 0.5*M_PI + angular_sd_orbiting*n1[i][j];
    //vr = sqrt(2.*G*Mbh/r[i]) * cos(theta)*exp(radial_sd_orbiting*n2[i][j]);
    vphi = Math.pow(G*Mbh/r[i], 0.5)  // * sin(theta)*exp(radial_sd_orbiting*n2[i][j]);

    vx[i] = vr*cosPhi - vphi*sinPhi;
    vy = vr*sinPhi + vphi*cosPhi;
    vz = 0;
    
    [vx[i], vz] = rotate(vx[i], vz, cos1, sin1);
    // Rotate to restore axisymmetry
    [vx[i],vy] = rotate(vx[i], vy, cos2, sin2);
    // Inclination
    [vx[i],vz] = rotate(vx[i], vz, cos3, sin3);
    vx[i] *= 299792458. / 1000.;
}

sumweights = 0.;
for (var i = 0; i < r.length; i++) {
    size[i] = 0.5 + kap * x[i]/Math.sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i])  //Math.cos((thetao * randval[i] + thetai) * Math.PI / 180.);
    if (Number.isNaN(size[i])) {
        size[i] = 0.0;
    }
    sumweights += size[i];
}
for (var i = 0; i < r.length; i++) {
    size[i] *= 1.0 * r.length / sumweights;
}

// Compute lags
for (var i = 0; i < r.length; i++) {
    lags[i] = r[i] - x[i];
}


// Compute the emission line light curve
// Continuum equation: cont = 2.0 + sin(time/10. * pi/2)
const Nspec = data['Nspec'][0]
var integrated;
var offset, part1, part2, part3, part4
for (var i = 0; i < Nspec; i++) {
    integrated = 0;
    for (var j = 0; j < lags.length; j++) {
        if (Number.isNaN(lags[j])) {
            integrated += 0.0;
        } else {
            offset = 2.0;
            part1 = 1.5 * Math.sin((times[i] - lags[j])/60. * Math.PI/2.0);
            part2 = 1.0 * Math.cos((times[i] - lags[j])/18. * Math.PI/2.0);
            part3 = 0.4 * Math.sin((times[i] - lags[j])/4. * Math.PI/2.0);
            part4 = 0.1 * Math.cos((times[i] - lags[j])/1. * Math.PI/2.0);
            integrated += offset + part1 + part2 + part3 + part4;
        }
    }
    spec[i] = 0.0 + 2.0 * integrated / lags.length;
}

// Compute the mean spectrum
// Continuum equation: cont = 2.0 + sin(time/10. * pi/2)
const Nwave = data['Nwave'][0]
for (var i = 0; i < Nwave; i++) {
    meanspec[i] = 0;
}
const dl = ll[1]-ll[0];
for (var i = 0; i < vx.length; i++) {
    var j = 0;
    while (j >= 0) {
        if (j == Nwave) {
            j = -1;
        } else if (vx[i] < ll[j] + 0.5 * dl & vx[i] > ll[j] - 0.5 * dl) {
            meanspec[j] += 1;
            j = -1;
        } else {
            j += 1;
        }
    }
}
// Smooth the spectrum
const win=2;
const tmp = meanspec;
var sum;
for (var i=win; i<meanspec.length-win; i++) {
    sum = 0;
    for (var j=i-win; j<i+win+1; j++) {
        sum += tmp[j];
    }
    meanspec[i] = sum / (2.0 * win + 1.0);
}

// Compute the FWHM
var peakindex = 0;
var peakval = 0.;
const minval = 0.0;
const halfval = minval + 0.5 * (peakval - minval);
for (var i = 0; i < Nwave; i++) {
    if (meanspec[i] > peakval) {
        peakval = meanspec[i];
        peakindex = i;
    }
}
var i = 0;
var val1, val2;
while (i >= 0) {
    if (meanspec[i] > minval + 0.5 * (peakval - minval)) {
        val1 = (ll[i-1] + ll[i])/2.;
        i = -1;
    } else {
        i += 1;
    }
}
i = peakindex;
while (i >= 0) {
    if (meanspec[i] < minval + 0.5 * (peakval - minval)) {
        val2 = (ll[i] + ll[i+1])/2.;
        i = -1;
    } else {
        i -= 1;
    }
}
const leftval = (val1 + val2)/2.;

i = peakindex;
while (i >= 0) {
    if (meanspec[i] < minval + 0.5 * (peakval - minval)) {
        val1 = (ll[i-1] + ll[i])/2.;
        i = -1;
    } else {
        i += 1;
    }
}
i = Nwave;
while (i >= 0) {
    if (meanspec[i] > minval + 0.5 * (peakval - minval)) {
        val2 = (ll[i] + ll[i+1])/2.;
        i = -1;
    } else {
        i -= 1;
    }
}
const rightval = (val1 + val2)/2.;
fwhm[0] = rightval-leftval;

var sum = 0.;
var nvals = 0.;
for (var i = 0; i < lags.length; i++) {
    if (Number.isNaN(lags[i])) {
        nvals += 0.;
    } else {
        sum += lags[i];
        nvals += 1.;
    }
}
lag[0] = sum/nvals;

vp[0] = 2.0 * Math.log10(fwhm[0]) + 8.477 + Math.log10(lag[0]) - 9.186;

source.change.emit();
source.properties.data.change.emit();