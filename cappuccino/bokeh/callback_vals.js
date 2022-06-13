const specdata = source.data;
const ll = specdata['ll'];
const meanspec = specdata['meanspec'];
const lags = specdata['lags'];

const data = props.data;
const fwhm = data['fwhm'];
const lag = data['lag'];

// Compute the FWHM
var peakindex = 0;
var peakval = 0.;
const minval = 0.0;
const halfval = minval + 0.5 * (peakval - minval);
for (var i = 0; i < meanspec.length; i++) {
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
i = meanspec.length;
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

props.properties.data.change.emit();
//props.change.emit();

//// Compute the FWHM
//var peakindex = 0;
//var peakval = 0.;
//const minval = 0.0;
//const halfval = minval + 0.5 * (peakval - minval);
//for (var i = 0; i < meanspec.length; i++) {
//    if (meanspec[i] > peakval) {
//        peakval = meanspec[i];
//        peakindex = i;
//    }
//}
//var i = 0;
//var val1, val2, frac;
//while (i >= 0) {
//    if (meanspec[i] > halfval)) {
//        //frac = (halfval - meanspec[i-1]) * (meanspec[i] - meanspec[i-1]);
//        //val1 = ll[i-1] + frac * (ll[i] - ll[i-1]);
//        val1=0;
//        i = -1;
//    } else {
//        i += 1;
//    }
//}
//i = peakindex;
//while (i >= 0) {
//    if (meanspec[i] < halfval) {
//        //frac = (halfval - meanspec[i]) * (meanspec[i+1] - meanspec[i]);
//        //val2 = ll[i] + frac * (ll[i+1] - ll[i]);
//        val2=0;
//        i = -1;
//    } else {
//        i -= 1;
//    }
//}
//const leftval = (val1 + val2)/2.;
//i = peakindex;
//while (i >= 0) {
//    if (meanspec[i] < halfval) {
//        //frac = (halfval - meanspec[i-1]) * (meanspec[i] - meanspec[i-1]);
//        //val1 = ll[i-1] + frac * (ll[i] - ll[i-1]);
//        val1=0;
//        i = -1;
//    } else {
//        i += 1;
//    }
//}
//i = meanspec.length;
//while (i >= 0) {
//    if (meanspec[i] > halfval) {
//        //frac = (halfval - meanspec[i]) * (meanspec[i+1] - meanspec[i]);
//        //val2 = ll[i] + frac * (ll[i+1] - ll[i]);
//        val2=0;
//        i = -1;
//    } else {
//        i -= 1;
//    }
//}
//const rightval = (val1 + val2)/2.;
//fwhm[0] = rightval-leftval;
//props.properties.data.change.emit();
////props.change.emit();