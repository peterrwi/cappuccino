# Copyright (c) 2009-2015 Brendon J. Brewer.
#
# This file is part of DNest3.
#
# DNest3 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DNest3 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DNest3. If not, see <http://www.gnu.org/licenses/>.
#
#
# ################################# NOTE #################################
# This file was edited from its original form by Peter Williams. 
# 
# Lines were added to load '.npy' files when available rather than '.txt'
# files in order to improve the speed. If no '.npy' file is available, the code
# will read the '.txt' file and then save a copy in the form of a '.npy' file
# for future runs.
#
# A "no_output" argument was added to postprocess. This will allow the code to
# run without generating terminal output and is helpful when running as a
# script.
#
# The plt.hold and plt.clf calls were all removed.

import copy
import numpy as np
import matplotlib.pyplot as plt


def logsumexp(values):
    biggest = np.max(values)
    x = values - biggest
    result = np.log(np.sum(np.exp(x))) + biggest
    return result


def logdiffexp(x1, x2):
    biggest = x1
    xx1 = x1 - biggest
    xx2 = x2 - biggest
    result = np.log(np.exp(xx1) - np.exp(xx2)) + biggest
    return result


def postprocess(temperature=1., numResampleLogX=1, plot=True, loaded=[], \
                cut=0., cut2=1.0, save=True, zoom_in=True, compression_bias_min=1.,
                compression_scatter=0., moreSamples=1.,
                compression_assert=None, no_output=False,
                savename='posterior_sample'):
    if len(loaded) == 0:
        try:
            levels_orig = np.atleast_2d(np.load("levels.npy"))
        except:
            try:
                tmp = np.loadtxt("levels.txt")
                np.save("levels.npy",tmp)
                levels_orig = np.atleast_2d(tmp)
            except:
                try:
                    tmp = np.atleast_2d(np.load("sample_info.npy"))
                except:
                    tmp = np.atleast_2d(np.loadtxt("sample_info.txt"))
                levels_tmp = np.hstack([
                    tmp,
                    np.atleast_2d(np.ones(len(tmp))).T,
                    np.atleast_2d(np.ones(len(tmp))).T,
                    np.atleast_2d(np.ones(len(tmp))).T,
                ])
                np.save("levels.npy",levels_tmp)
                levels_orig = np.atleast_2d(levels_tmp)
        try:
            sample_info = np.atleast_2d(np.load("sample_info.npy"))
        except:
            tmp = np.loadtxt("sample_info.txt")
            np.save("sample_info.npy",tmp)
            sample_info = np.atleast_2d(tmp)
        try:
            sample = np.atleast_2d(np.load("sample.npy"))
            # Check if the samples.txt file has been updated
            try:
                num_lines_txt = sum(1 for line in open('sample.txt')) - 1
                if num_lines_txt != len(sample):
                    user_input = raw_input("Looks like you have an updated sample.txt file -- sample.txt has %i rows, sample.npy has %i rows. Re-create sample.npy and sample_info.npy? " % (num_lines_txt, len(sample)))
                    if user_input.lower() in ['yes','y']:
                        print "Loading sample.txt file. This may take a while..."
                        tmp1 = np.loadtxt("sample.txt")
                        tmp2 = np.loadtxt("sample_info.txt")
                        np.save("sample.npy",tmp1)
                        np.save("sample_info.npy",tmp2)
                        sample = np.atleast_2d(tmp1)
                        sample_info = np.atleast_2d(tmp2)
            except:
                print "No sample.txt file. Using sample.npy"
        except:
            print "Loading sample.txt file. This may take a while..."
            tmp = np.loadtxt("sample.txt")
            print "Saving sample.npy file for faster load times in future."
            np.save("sample.npy",tmp)
            sample = np.atleast_2d(tmp)
    # if(sample.shape[0] == 1):
    #	sample = sample.T
    else:
        levels_orig, sample_info, sample = loaded[0], loaded[1], loaded[2]

    # Remove regularisation from levels_orig if we asked for it
    if compression_assert is not None:
        levels_orig[1:, 0] = -np.cumsum(
            compression_assert * np.ones(levels_orig.shape[0] - 1))

    sample = sample[int(cut * sample.shape[0]):int(cut2 * sample.shape[0]), :]
    sample_info = sample_info[int(cut * sample_info.shape[0]):int(cut2 * sample_info.shape[0]), :]

    if sample.shape[0] != sample_info.shape[0]:
        if not no_output:
            print('# Size mismatch. Truncating...')
        lowest = np.min([sample.shape[0], sample_info.shape[0]])
        sample = sample[0:lowest, :]
        sample_info = sample_info[0:lowest, :]

    if plot:
        if numResampleLogX > 1:
            plt.ion()

        plt.figure(1)
        #plt.clf()
        plt.plot(sample_info[:, 0])
        plt.xlabel("Iteration")
        plt.ylabel("Level")
        if numResampleLogX > 1:
            plt.draw()
        plt.savefig('fig1.png')

        plt.figure(2)
        #plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(np.diff(levels_orig[:, 0]))
        plt.ylabel("Compression")
        plt.xlabel("Level")
        xlim = plt.gca().get_xlim()
        plt.axhline(-1., color='r')
        plt.axhline(-np.log(10.), color='g')
        plt.ylim(ymax=0.05)
        if numResampleLogX > 1:
            plt.draw()

        plt.subplot(2, 1, 2)
        good = np.nonzero(levels_orig[:, 4] > 0)[0]
        plt.plot(levels_orig[good, 3] / levels_orig[good, 4])
        plt.xlim(xlim)
        plt.ylim([0., 1.])
        plt.xlabel("Level")
        plt.ylabel("MH Acceptance")
        if numResampleLogX > 1:
            plt.draw()
        plt.savefig('fig2.png')

    # Convert to lists of tuples
    logl_levels = [(levels_orig[i, 1], levels_orig[i, 2]) for i in
                   range(0, levels_orig.shape[0])]  # logl, tiebreaker
    logl_samples = [(sample_info[i, 1], sample_info[i, 2], i) for i in
                    range(0, sample.shape[0])]  # logl, tiebreaker, id
    logx_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    logp_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    logP_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    P_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    logz_estimates = np.zeros((numResampleLogX, 1))
    H_estimates = np.zeros((numResampleLogX, 1))

    # Find sandwiching level for each sample
    sandwich = sample_info[:, 0].copy().astype('int')
    for i in range(0, sample.shape[0]):
        while sandwich[i] < levels_orig.shape[0] - 1 and logl_samples[i] > \
                logl_levels[sandwich[i] + 1]:
            sandwich[i] += 1

    for z in range(0, numResampleLogX):
        # Make a monte carlo perturbation of the level compressions
        levels = levels_orig.copy()
        compressions = -np.diff(levels[:, 0])
        compressions *= compression_bias_min + (
                    1. - compression_bias_min) * np.random.rand()
        compressions *= np.exp(
            compression_scatter * np.random.randn(compressions.size))
        levels[1:, 0] = -compressions
        levels[:, 0] = np.cumsum(levels[:, 0])

        # For each level
        for i in range(0, levels.shape[0]):
            # Find the samples sandwiched by this level
            which = np.nonzero(sandwich == i)[0]
            logl_samples_thisLevel = []  # (logl, tieBreaker, ID)
            for j in range(0, len(which)):
                logl_samples_thisLevel.append(
                    copy.deepcopy(logl_samples[which[j]]))
            logl_samples_thisLevel = sorted(logl_samples_thisLevel)
            N = len(logl_samples_thisLevel)

            # Generate intermediate logx values
            logx_max = levels[i, 0]
            if i == levels.shape[0] - 1:
                logx_min = -1E300
            else:
                logx_min = levels[i + 1, 0]
            Umin = np.exp(logx_min - logx_max)

            if N == 0 or numResampleLogX > 1:
                U = Umin + (1. - Umin) * np.random.rand(len(which))
            else:
                U = Umin + (1. - Umin) * np.linspace(1. / (N + 1),
                                                     1. - 1. / (N + 1), N)
            logx_samples_thisLevel = np.sort(logx_max + np.log(U))[::-1]
            for j in range(0, which.size):
                logx_samples[logl_samples_thisLevel[j][2]][z] = \
                logx_samples_thisLevel[j]
                if j != which.size - 1:
                    left = logx_samples_thisLevel[j + 1]
                elif i == levels.shape[0] - 1:
                    left = -1E300
                else:
                    left = levels[i + 1][0]

                if j != 0:
                    right = logx_samples_thisLevel[j - 1]
                else:
                    right = levels[i][0]

                logp_samples[logl_samples_thisLevel[j][2]][z] = np.log(
                    0.5) + logdiffexp(right, left)
        logl = sample_info[:, 1] / temperature

        logp_samples[:, z] = logp_samples[:, z] - logsumexp(logp_samples[:, z])
        logP_samples[:, z] = logp_samples[:, z] + logl
        logz_estimates[z] = logsumexp(logP_samples[:, z])
        logP_samples[:, z] -= logz_estimates[z]
        P_samples[:, z] = np.exp(logP_samples[:, z])
        H_estimates[z] = -logz_estimates[z] + np.sum(P_samples[:, z] * logl)

        if plot:
            plt.figure(3)
            #plt.clf()
            plt.subplot(2, 1, 1)
            #plt.hold(False)
            plt.plot(logx_samples[:, z], sample_info[:, 1], 'b.',
                     label='Samples')
            #plt.hold(True)
            plt.plot(levels[1:, 0], levels[1:, 1], 'r.', label='Levels')
            plt.legend(numpoints=1, loc='lower left')
            plt.ylabel('log(L)')
            plt.title(
                str(z + 1) + "/" + str(numResampleLogX) + ", log(Z) = " + str(
                    logz_estimates[z][0]))
            # Use all plotted logl values to set ylim
            combined_logl = np.hstack([sample_info[:, 1], levels[1:, 1]])
            combined_logl = np.sort(combined_logl)
            lower = combined_logl[int(0.1 * combined_logl.size)]
            upper = combined_logl[-1]
            diff = upper - lower
            lower -= 0.05 * diff
            upper += 0.05 * diff
            if zoom_in:
                plt.ylim([lower, upper])

            if numResampleLogX > 1:
                plt.draw()
            xlim = plt.gca().get_xlim()

        for k in range(len(P_samples[:,z])):
            if logx_samples[k,z] > -1:
                P_samples[k,z] = 1.e-300
        if plot:
            plt.subplot(2, 1, 2)
            #plt.hold(False)
            plt.plot(logx_samples[:, z], P_samples[:, z], 'b.')
            plt.ylabel('Posterior Weights')
            plt.xlabel('log(X)')
            plt.xlim(xlim)
            if numResampleLogX > 1:
                plt.draw()
            plt.savefig('fig3.png')

    P_samples = np.mean(P_samples, 1)
    P_samples = P_samples / np.sum(P_samples)
    logz_estimate = np.mean(logz_estimates)
    logz_error = np.std(logz_estimates)
    H_estimate = np.mean(H_estimates)
    H_error = np.std(H_estimates)
    ESS = np.exp(-np.sum(P_samples * np.log(P_samples + 1E-300)))

    if not no_output:
        print("log(Z) = " + str(logz_estimate) + " +- " + str(logz_error))
        print("Information = " + str(H_estimate) + " +- " + str(H_error) + " nats.")
        print("Effective sample size = " + str(ESS))

    # Resample to uniform weight
    N = int(moreSamples * ESS)
    posterior_sample = np.zeros((N, sample.shape[1]))
    w = P_samples
    w = w / np.max(w)
    if save:
        np.savetxt('weights.txt', w)  # Save weights
    for i in range(0, N):
        while True:
            which = np.random.randint(sample.shape[0])
            if np.random.rand() <= w[which]:
                break
        posterior_sample[i, :] = sample[which, :]
    if save:
        np.savetxt(savename+'.txt', posterior_sample)
        np.save(savename+'.npy', posterior_sample)

    if plot:
        plt.show()
    #if plot:
    #    if numResampleLogX > 1:
    #        plt.ioff()
    #    plt.hold(True)
    # plt.show()
    # plt.hold(True)

    return [logz_estimate, H_estimate, logx_samples]
