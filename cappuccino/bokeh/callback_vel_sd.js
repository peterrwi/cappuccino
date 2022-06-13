const data_clouds = source_clouds.data;
const r = data_clouds['r'];
const x = data_clouds['x'];
const y = data_clouds['y'];
const z = data_clouds['z'];
const lags = data_clouds['lags'];
const size = data_clouds['size'];
const far_side = data_clouds['far_side'];
const x_unit = data_clouds['x_unit'];
const y_unit = data_clouds['y_unit'];
const z_unit = data_clouds['z_unit'];
const vcirc = data_clouds['vcirc'];

const data_gamma = source_gamma.data;
const gamma_x = data_gamma['x'];
const gamma_y = data_gamma['y'];

const data_eline = source_eline.data;
const eline_vel = data_eline['vel'];
const eline_flux = data_eline['flux'];

const data_vel = source_vel.data;
const vr = data_vel['vr'];
const vphi = data_vel['vphi'];
const vx = data_vel['vx'];
const vy = data_vel['vy'];
const vz = data_vel['vz'];
const vx_unit = data_vel['vx_unit'];
const vy_unit = data_vel['vy_unit'];
const vz_unit = data_vel['vz_unit'];
const lags_vel = data_vel['lags'];
const size_vel = data_vel['size'];
const theta_rot = data_vel['theta_rot'];

const _thetai = thetai.value;
const _thetao = thetao.value;
const _gamma = gamma.value;
const _kappa = kappa.value;
const _beta = beta.value;
const _mu = mu.value;
const _F = F.value;
const _xi = xi.value;
const _fflow = fflow.value;
const _fellip = fellip.value;
const _ellipseAngle = ellipseAngle.value;
const _angular_sd_orbiting = angular_sd_orbiting.value;
const _radial_sd_orbiting = radial_sd_orbiting.value;
const _angular_sd_flowing = angular_sd_flowing.value;
const _radial_sd_flowing = radial_sd_flowing.value;
const _Mbh = Math.pow(10, logMbh.value);
const _turbulence = Math.pow(10, logturbulence.value);

const randorder = rands['randorder'];
const randnorm1 = rands['randnorm1'];
const randnorm2 = rands['randnorm2'];
const randnorm3 = rands['randnorm3_flat'];
const randnorm4 = rands['randnorm4_flat'];
const randnorm5 = rands['randnorm5_flat'];
const rand0 = rands['rand0'];
const rand1 = rands['rand1'];
const rand2 = rands['rand2'];

const sin1 = intermediate['sin1'];
const cos1 = intermediate['cos1'];
const sin2 = intermediate['sin2'];
const cos2 = intermediate['cos2'];
var sin3 = intermediate['sin3'];
var cos3 = intermediate['cos3'];
var part1 = intermediate['part1'];
const part2 = intermediate['part2'];
const angle = intermediate['angle'];

const Nclouds = params['Nclouds'];
const Nvpercloud = params['Nvpercloud'];

function rotate(x, y, C, S) {
    return [C*x + S*y, -S*x + C*y];
}

function rotateVecAroundAxis(axis, theta, vec) {
    const rotated = [];
    const costheta = Math.cos(theta);
    const sintheta = Math.sin(theta);
    const rot_matrix = [
        [
            costheta + Math.pow(axis[0],2) * (1.0 - costheta),
            axis[0] * axis[1] * (1.0 - costheta) - axis[2] * sintheta,
            axis[0] * axis[2] * (1.0 - costheta) + axis[1] * sintheta
        ],
        [
            axis[1] * axis[0] * (1.0 - costheta) + axis[2] * sintheta,
            costheta + Math.pow(axis[1],2) * (1.0 - costheta),
            axis[1] * axis[2] * (1.0 - costheta) - axis[0] * sintheta
        ],
        [
            axis[2] * axis[0] * (1.0 - costheta) - axis[1] * sintheta,
            axis[2] * axis[1] * (1.0 - costheta) + axis[0] * sintheta,
            costheta + Math.pow(axis[2],2) * (1.0 - costheta)
        ]
    ];
    
    rotated[0] = rot_matrix[0][0] * vec[0] + rot_matrix[0][1] * vec[1] + rot_matrix[0][2] * vec[2];
    rotated[1] = rot_matrix[1][0] * vec[0] + rot_matrix[1][1] * vec[1] + rot_matrix[1][2] * vec[2];
    rotated[2] = rot_matrix[2][0] * vec[0] + rot_matrix[2][1] * vec[1] + rot_matrix[2][2] * vec[2];

    return rotated
}

function drawFromGamma(beta, mu, F, Nclouds, rmax, Nsamp) {
    const alpha = Math.pow(beta, -2.0);
    const rmin = mu * F;
    const theta = (mu - rmin) / alpha;

    // Draw values from the gamma function
    // First create list of x-values
    var gamma_x = [];
    var gamma_y = [];
    for (var i = 0; i < Nsamp; i++) {
        gamma_x[i] = i/Nsamp*rmax;
    }

    // Compute the gamma function
    var cdf = [];
    var p1;
    for (var i = 0; i < Nsamp; i++) {
        p1 = Math.log(gamma_x[i] - rmin) * (alpha - 1.0);
        cdf[i] = Math.exp(p1-(gamma_x[i]-rmin) / theta);
        if (Number.isNaN(cdf[i])) {
            cdf[i] = 0.0;
        } else if (cdf[i] < 0.0) {
            cdf[i] = 0.0;
        } else if (cdf[i] == Infinity) {
            cdf[i] = 0.0;
        }
    }
    var sumvals = 0.;
    for (var i = 0; i < Nsamp; i++) {
        sumvals += cdf[i];
    }
    // Re-normalize and compute the cumulative density function
    for (var i = 0; i < Nsamp; i++) {
        cdf[i] /= sumvals;
        gamma_y[i] = cdf[i];
        if (i > 0) {
            cdf[i] += cdf[i-1];
        }
    }

    // Use sorted values from 0-1 to draw from CDF
    var r = [];
    var run_loop;
    var j;
    for (var i = 0; i < Nclouds; i++) {   // Loop over rand1 values
        j = 0;
        run_loop = 1;
        while (run_loop == 1) {
            if (j >= cdf.length) {             // If we reach end of cdf, store rest as last xvals value
                r[i] = gamma_x[Nsamp-1];
                run_loop = 0;
            } else if (cdf[j] > i/Nclouds) {    // If cdf > val, save x-val
                r[i] = gamma_x[j];
                run_loop = 0;
            } else {
                j += 1;
            }
        }
    }

    return [r, gamma_x, gamma_y];
}

function getAngMom0(_x) {
    var ang_mom = [];
    if (_x[2] == 0) {
        ang_mom = [0,0,1];
    } else if (_x[2] > 0) {
        ang_mom = [-_x[0], -_x[1], (Math.pow(_x[0],2) + Math.pow(_x[1],2))/_x[2]];
    } else {
        ang_mom = [_x[0], _x[1], -(Math.pow(_x[0],2) + Math.pow(_x[1],2))/_x[2]];
    }
    // Normalize
    var squaresum = 0;
    for (var i=0; i<3; i++) {
        squaresum += Math.pow(ang_mom[i],2.0);
    }
    for (var i=0; i<3; i++) {
        ang_mom[i] /= Math.pow(squaresum, 0.5);
    }
    return ang_mom;
}

function updateGamma() {
    const [new_r, new_gamma_x, new_gamma_y] = drawFromGamma(_beta, _mu, _F, Nclouds, params['rmax'], 1000);
    for (var i = 0; i < Nclouds; i++) {
        x[i] = x[i] * new_r[i]/r[i];
        y[i] = y[i] * new_r[i]/r[i];
        z[i] = z[i] * new_r[i]/r[i];
    }
    for (var i = 0; i < Nclouds; i++) {
        for (var j = 0; j < Nvpercloud; j++) {
            vx[i*Nvpercloud+j] = vx[i*Nvpercloud+j] * Math.pow(r[i]/new_r[i], 0.5);
            vy[i*Nvpercloud+j] = vy[i*Nvpercloud+j] * Math.pow(r[i]/new_r[i], 0.5);
            vz[i*Nvpercloud+j] = vz[i*Nvpercloud+j] * Math.pow(r[i]/new_r[i], 0.5);
        }
        vcirc[i] *= Math.pow(r[i]/new_r[i], 0.5);
    }
    for (var i = 0; i < Nclouds; i++) {
        r[i] = new_r[i];
    }
    for (var i = 0; i < Nclouds; i++) {
        lags[i] = r[i] - x[i];
        for (var j = 0; j < Nvpercloud; j++) {
            lags_vel[i*Nvpercloud+j] = lags[i];
        }
    }
    for (var i = 0; i < gamma_x.length; i++) {
        gamma_x[i] = new_gamma_x[i];
        gamma_y[i] = new_gamma_y[i];
    }
}

function updateXYZ() {
    part1 = Math.sin(_thetao * Math.PI / 180.);
    for (var i=0; i<Nclouds; i++) {
        part2[i] = rand1[i]**(1.0/_gamma);
        angle[i] = Math.asin(part1 * part2[i]);
        sin1[i] = Math.sin(angle[i]);
        cos1[i] = Math.cos(angle[i]);
    }

    // Set the positions
    for (var i=0; i<Nclouds; i++) {
        x[i] = r[i] * rand0[i];
        y[i] = 0.0;
        z[i] = 0.0;

        // Puff up by opening angle into wedge
        [x[i], z[i]] = rotate(x[i], z[i], cos1[i], sin1[i]);
        // Rotate into thick disk
        [x[i], y[i]] = rotate(x[i], y[i], cos2[i], sin2[i]);
    }
    // Update position unit vectors
    for (var i=0; i<Nclouds; i++) {
        x_unit[i] = x[i] / r[i];
        y_unit[i] = y[i] / r[i];
        z_unit[i] = z[i] / r[i];
    }
    // Inclination
    for (var i=0; i<Nclouds; i++) {
        [x[i], z[i]] = rotate(x[i], z[i], cos3, sin3);
    }
}

function updateWeights() {
    var sumsize = 0;
    for (var i = 0; i < Nclouds; i++) {
        size[i] = 0.5 + _kappa * x[i]/r[i];
        sumsize += size[i];
        if (far_side[i] == 1) {
            size[i] *= _xi;
        }
    }
    for (var i = 0; i < Nclouds; i++) {
        size[i] /= sumsize;
        size[i] *= Nclouds*1.5;
        for (var j = 0; j < Nvpercloud; j++) {
            size_vel[i*Nvpercloud+j] = size[i];
        }
    }
}

function updateVrVphi() {
    var index1, index2;
    var _theta;
    const startindex = _fellip * Nclouds;
    for (var i = 0; i < Nclouds; i++) {
        for (var j = 0; j < Nvpercloud; j++) {
            index1 = randorder[j][i];
            index2 = i*Nvpercloud+j;
            if (index1 < startindex) {
                _theta = 0.5*Math.PI + _angular_sd_orbiting*randnorm1[i];
                vr[index2] = Math.pow(2.0, 0.5) * Math.cos(_theta) * Math.exp(_radial_sd_orbiting*randnorm2[i]);
                vphi[index2] = Math.sin(_theta) * Math.exp(_radial_sd_orbiting*randnorm2[i]);
            } else {
                if (_fflow < 0.5) {
                    _theta = Math.PI - _ellipseAngle*Math.PI/180. + _angular_sd_flowing*randnorm1[i];
                } else {
                    _theta = 0.0 + _ellipseAngle*Math.PI/180. + _angular_sd_flowing*randnorm1[i];
                }
                vr[index2] = Math.pow(2.0, 0.5) * Math.cos(_theta) * Math.exp(_radial_sd_flowing*randnorm2[i]);
                vphi[index2] = Math.sin(_theta) * Math.exp(_radial_sd_flowing*randnorm2[i]);
            }
        }
    }
}

function updateAngMom() {
    var ang_mom0;
    var ang_mom;
    var index;
    if (params['thetao'] != _thetao) {
        for (var i = 0; i < Nclouds; i++) {
            for (var j = 0; j < Nvpercloud; j++) {
                index = i*Nvpercloud + j;
                theta_rot[index] *= _thetao/params['thetao'];
            }
        }
    }
    for (var i = 0; i < Nclouds; i++) {
        ang_mom0 = getAngMom0([x_unit[i],y_unit[i],z_unit[i]]);
        for (var j = 0; j < Nvpercloud; j++) {
            index = i*Nvpercloud + j;
            ang_mom = rotateVecAroundAxis([x_unit[i],y_unit[i],z_unit[i]], theta_rot[index], ang_mom0);
            vx_unit[index] = ang_mom[1] * z_unit[i] - ang_mom[2] * y_unit[i];
            vy_unit[index] = ang_mom[2] * x_unit[i] - ang_mom[0] * z_unit[i];
            vz_unit[index] = ang_mom[0] * y_unit[i] - ang_mom[1] * x_unit[i];
        }
    }
}

function updateVXYZ() {
    var index;
    for (var i = 0; i < Nclouds; i++) {
        for (var j = 0; j < Nvpercloud; j++) {
            index = i*Nvpercloud + j;
            vx[index] = (vr[index] * x_unit[i] + vphi[index] * vx_unit[index]) * vcirc[i];
            vy[index] = (vr[index] * y_unit[i] + vphi[index] * vy_unit[index]) * vcirc[i];
            vz[index] = (vr[index] * z_unit[i] + vphi[index] * vz_unit[index]) * vcirc[i];
        }
    }
    for (var i = 0; i < Nclouds; i++) {
        for (var j = 0; j < Nvpercloud; j++) {
            index = i*Nvpercloud + j;
            [vx[index],vz[index]] = rotate(vx[index],vz[index],cos3,sin3);
            vx[index] += _turbulence * vcirc[i] * randnorm3[index];
            vy[index] += _turbulence * vcirc[i] * randnorm4[index];
            vz[index] += _turbulence * vcirc[i] * randnorm5[index];
        }
    }       
}

function updateMbh() {
    const scale_fac = Math.pow(_Mbh / Math.pow(10.0, params['logMbh']), 0.5);
    var index;
    for (var i = 0; i < Nclouds; i++) {
        for (var j = 0; j < Nvpercloud; j++) {
            index = i*Nvpercloud + j;
            vx[index] *= scale_fac;
            vy[index] *= scale_fac;
            vz[index] *= scale_fac;
        }
        vcirc[i] *= scale_fac
    }
}

function updateTurbulence() {
    var index;
    for (var i = 0; i < Nclouds; i++) {
        for (var j = 0; j < Nvpercloud; j++) {
            index = i*Nvpercloud + j;
            vx[index] -= Math.pow(10, params['logturbulence']) * vcirc[i] * randnorm3[index];
            vy[index] -= Math.pow(10, params['logturbulence']) * vcirc[i] * randnorm4[index];
            vz[index] -= Math.pow(10, params['logturbulence']) * vcirc[i] * randnorm5[index];
            vx[index] += _turbulence * vcirc[i] * randnorm3[index];
            vy[index] += _turbulence * vcirc[i] * randnorm4[index];
            vz[index] += _turbulence * vcirc[i] * randnorm5[index];
        }
    }
}

function calcMeanProfile() {
    const step = eline_vel[1]-eline_vel[0];
    const bin_edges = [];
    var index
    for (var i=0; i<eline_vel.length+1; i++) {
        bin_edges[i] = eline_vel[0] - step/2.0 + step * i;
    }
    for (var i=0; i<eline_flux.length; i++) {
        eline_flux[i] = 0.0;
    }
    for (var i=0; i<Nclouds; i++) {
        for (var j=0; j<Nvpercloud; j++) {
            index = i*Nvpercloud + j;
            if (vx[index] < bin_edges[0]) {
                break;
            }
            if (vx[i,j] > bin_edges[-1]) {
                break;
            }
            for (var k=0; k<eline_flux.length; k++) {
                if ((vx[index] > bin_edges[k]) && (vx[index] < bin_edges[k+1])) {
                    eline_flux[k] += size[i] / step / Nclouds / Nvpercloud * 2500.;
                    break;
                }
            }
        }
    }
}

sin3 = Math.sin(0.5 * Math.PI - _thetai * Math.PI / 180.);
cos3 = Math.cos(0.5 * Math.PI - _thetai * Math.PI / 180.);

if ((params['mu'] != _mu) || (params['beta'] != _beta) || (params['F'] != _F)) {
    updateGamma();
    calcMeanProfile();
}
if ((params['thetao'] != _thetao) || (params['gamma'] != _gamma) || (params['thetai'] != _thetai)) {
    updateXYZ();
    updateWeights();
    updateAngMom();
    updateVXYZ();
    calcMeanProfile();
}
if ((params['kappa'] != _kappa) || (params['xi'] != _xi)) {
    updateWeights();
    calcMeanProfile();
}

if ((params['angular_sd_orbiting'] != _angular_sd_orbiting) || (params['angular_sd_flowing'] != _angular_sd_flowing) 
    || (params['radial_sd_orbiting'] != _radial_sd_orbiting) || (params['radial_sd_flowing'] != _radial_sd_flowing)
    || (params['fflow'] != _fflow) || (params['ellipseAngle'] != _ellipseAngle)
    || (params['fellip'] != _fellip)) {
    updateVrVphi();
    updateVXYZ();
    calcMeanProfile();
}

if ((params['logMbh'] != Math.log10(_Mbh))) {
    updateMbh();
    calcMeanProfile();
}

if ((params['logturbulence'] != Math.log10(_turbulence))) {
    console.log('Here')
    updateTurbulence();
    calcMeanProfile();
}

// Update params
params['thetai'] = _thetai;
params['thetao'] = _thetao;
params['gamma'] = _gamma;
params['kappa'] = _kappa;
params['beta'] = _beta;
params['mu'] = _mu;
params['F'] = _F;
params['xi'] = _xi;
params['fflow'] = _fflow;
params['fellip'] = _fellip;
params['ellipseAngle'] = _ellipseAngle;
params['angular_sd_orbiting'] = _angular_sd_orbiting;
params['angular_sd_flowing'] = _angular_sd_flowing;
params['radial_sd_orbiting'] = _radial_sd_orbiting;
params['radial_sd_flowing'] = _radial_sd_flowing;
params['logMbh'] = Math.log10(_Mbh);
params['logturbulence'] = Math.log10(_turbulence);

source_clouds.change.emit();
source_gamma.change.emit();
source_vel.change.emit();
source_eline.change.emit();