import matplotlib
import numpy as np
import pickle

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True


def log_likelihood(params, xx):
    mu, sigma = params[0], params[1]
    logL = 0.
    for x in xx:
        f = 1. / (sigma * np.sqrt(2. * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        logL += np.log(np.mean(f))
    return logL


def calcFFactorMean(post, dvtype, fp_results):

    np.random.seed(0)
    min_value = 500
    xx = []
    post_nonans = {}
    for key in post.keys():
        if not np.isnan(post[key].posteriors['log10f_'+dvtype][0]):
            post_nonans[key] = post[key]

    post = post_nonans
    for key in post.keys():
        xx.append(post[key].posteriors['log10f_'+dvtype][0:min_value])

    num = 500
    mu = np.linspace(-3., 3., num)
    sigma = np.linspace(0.05, 2., num)
    [mu, sigma] = np.meshgrid(mu, sigma)
    sigma = sigma[::-1, :]  # reverse the order of sigma

    logL = np.zeros(mu.shape)
    for i in xrange(0, logL.shape[0]):
        for j in xrange(0, logL.shape[1]):
            logL[i, j] = log_likelihood([mu[i, j], sigma[i, j]], xx)
        print(i)

    post = np.exp(logL - logL.max())
    post = post / post.sum()

    post_dict = {
        "ylim":sigma.min(),
        "data":post
    }

    filename = fp_results + '_2d_posterior.p'
    with open(filename,'wb') as f:
        pickle.dump(post_dict,f)
    print "2D posterior file: ", filename

    # Compute marginal posterior for mu
    # and predictive distribution for a new f

    # Summarise mu results
    p = post.sum(axis=0)
    p /= np.trapz(p, x=mu[0, :])  # integrate along the given axis using the composite trapezoidal rule

    mu_posterior_mean = np.trapz(p * mu[0, :], x=mu[0, :])
    mu_posterior_sd = np.sqrt(
        np.trapz(p * mu[0, :] ** 2, x=mu[0, :]) - mu_posterior_mean ** 2
    )
    mu_posterior = np.vstack([mu[0, :], p]).T
    np.savetxt(fp_results + '_mu_posterior.txt', mu_posterior)


    # Summarise sigma results
    p = post.sum(axis=1)
    p /= np.trapz(p, x=sigma[:, 0])

    sigma_posterior_mean = np.trapz(p * sigma[:, 0], x=sigma[:, 0])
    sigma_posterior_sd = np.sqrt(
        np.trapz(p * sigma[:, 0] ** 2, x=sigma[:, 0]) - sigma_posterior_mean ** 2
    )
    sigma_posterior = np.vstack([sigma[:, 0], p]).T
    np.savetxt(fp_results + '_sigma_posterior.txt', sigma_posterior)

    # Compute the predictive posterior distribution
    predictive = np.zeros(mu[0, :].shape)
    for i in xrange(0, post.shape[0]):
        for j in xrange(0, post.shape[1]):
            # print i, j
            predictive += post[i, j] / (
                        sigma[i, j] * np.sqrt(2. * np.pi)) * np.exp(
                -0.5 * ((mu[0, :] - mu[i, j]) / sigma[i, j]) ** 2)

    predictive_mean = np.trapz(predictive * mu[0, :], x=mu[0, :])
    predictive_std = np.sqrt(
        np.trapz(predictive * mu[0, :] ** 2, x=mu[0, :]) - predictive_mean ** 2
    )
    predictive_posterior = np.vstack([mu[0, :], predictive]).T
    np.savetxt(fp_results + '_predictive_posterior.txt', predictive_posterior)

    print('Posterior mean of mu = ' + str(mu_posterior_mean))
    print('Posterior sd of mu = ' + str(mu_posterior_sd))
    print('Posterior mean of sigma = ' + str(sigma_posterior_mean))
    print('Posterior sd of sigma = ' + str(sigma_posterior_sd))
    print('Mean value of predictive distribution = ' + str(predictive_mean))
    print('Standard deviation of predictive distribution = ' + str(predictive_std))

    with open(fp_results+'_summary.txt', 'wb') as f:
        f.write('Posterior mean of mu = ' + str(mu_posterior_mean) + '\n')
        f.write('Posterior sd of mu = ' + str(mu_posterior_sd) + '\n')
        f.write('Posterior mean of sigma = ' + str(sigma_posterior_mean) + '\n')
        f.write('Posterior sd of sigma = ' + str(sigma_posterior_sd) + '\n')
        f.write("Mean value of predictive distribution = " + str(predictive_mean) + '\n')
        f.write("Standard deviation of predictive distribution = " + str(predictive_std) + '\n')


def readMeanF(filename):
    with open(filename,'rb') as f:
        results = map(lambda x:float(x.split()[-1]),f.readlines())
    return results


def main():
    pass


if __name__ == '__main__':
    main()