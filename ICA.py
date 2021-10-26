from pydub import AudioSegment
import os

import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np
np.random.seed(0)
sns.set(rc={'figure.figsize': (11.7, 8.27)})

'''Citation for ICA source code:
Author: Cory Maklin
Date: August 22, 2019
Title: Independent Component Analysis (ICA) in Python
Source: https://towardsdatascience.com/independent-component-analysis-ica-in-python-a0ef0db0955e

Range: line 21 to line 146
'''

# (hyerbolic tangent) functions... used in our algorithm which determines
# the new values for the unmixing matrix, w
def g(x):
    return np.tanh(x)


def g_der(x):
    return 1 - g(x) * g(x)


# center the signal
def center(x):

    x = np.array(x)

    mean = x.mean(axis=1, keepdims=True)

    return x - mean


# whiten the signal
def whitening(x):
    # covariance matrix
    cov = np.cov(x)

    # return the eignevalues and eigenvectors of the covariance matrix
    d, E = np.linalg.eigh(cov)

    # create the diagonal matrix
    D = np.diag(d)

    D_inv = np.sqrt(np.linalg.inv(D))

    x_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, x)))

    return x_whiten


# function for updating the de-mixing matrix, w
def calculate_new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - \
        g_der(np.dot(w.T, X)).mean() * w

    w_new /= np.sqrt((w_new ** 2).sum())

    return w_new


# call upon the preprocessing functions
def ica(X, iterations, tolerance=1e-5):
    X = center(X)

    X = whitening(X)

    components_nr = X.shape[0]

    # define the de-mixing matrix, W, as some random set of values
    # to start with
    W = np.zeros((components_nr, components_nr), dtype=X.dtype)

    # continue until we reach convergence
    for i in range(components_nr):

        w = np.random.rand(components_nr)

        for j in range(iterations):

            w_new = calculate_new_w(w, X)

            if i >= 1:
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])

            distance = np.abs(np.abs((w * w_new).sum()) - 1)

            w = w_new

            if distance < tolerance:
                break

        # forming the matrix, W
        W[i, :] = w

    # compute the original signal
    S = np.dot(W, X)

    return S


# plot and compare the original, mixed, and predicted sources
def plot_mixture_sources_predictions(X, S):
    fig = plt.figure()

    plt.subplot(3, 1, 1)
    for x in X:
        # plot 3 different mixtures
        plt.plot(x)
    plt.title("mixtures")

    plt.subplot(3, 1, 3)
    for s in S:
        plt.plot(s)
    plt.title("predicted sources")

    fig.tight_layout()
    plt.show()


# randomly mix the sources
def mix_sources(mixtures, apply_noise=False):

    for i in range(len(mixtures)):

        max_val = np.max(mixtures[i])

        if max_val > 1 or np.min(mixtures[i]) < 1:

            mixtures[i] = mixtures[i] / (max_val / 2) - 0.5

    X = np.c_[[mix for mix in mixtures]]

    if apply_noise:

        X += 0.02 * np.random.normal(size=X.shape)

    return X

# ------------------------------------------------------------------
'''read in pairs of input files (for each pair of input files, both files contain a mixture of the same two sounds, each at different levels)

apply ICA to decompose the mixed signals, and write the original sounds to output
'''

WAVs = os.listdir('input files')

for count, WAV in enumerate(WAVs):
    file1 = ''

    if count % 2 == 1:
        continue

    else:
        for char in WAV:
            if char != "V" and char != ".":
                file1 += char
            else:
                break

    file2 = file1 + "V2"

    sampling_rate, mix1 = wavfile.read('input files/{}.wav'.format(file1))
    sampling_rate, mix2 = wavfile.read('input files/{}.wav'.format(file2))

    if mix1.ndim > 1 or mix2.ndim > 1:
        sound = AudioSegment.from_wav("input files/{}.wav".format(file1))
        sound = sound.set_channels(1)
        sound.export("mix1.wav", format="wav")

        sound = AudioSegment.from_wav("input files/{}.wav".format(file2))
        sound = sound.set_channels(1)
        sound.export("mix2.wav", format="wav")

        sampling_rate, mix1 = wavfile.read('mix1.wav')
        sampling_rate, mix2 = wavfile.read('mix2.wav')

    X = mix_sources([mix1, mix2])

    print("-----------------------------------------------------")
    print("-----------------------------------------------------")

    S = ica(X, iterations=1000)

    plot_mixture_sources_predictions(X, S)

    print("Creating WAVs for file {}".format(file1))
    wavfile.write('output files/out1_{}.wav'.format(file1),
                  sampling_rate, S[0])
    wavfile.write('output files/out2_{}.wav'.format(file1),
                  sampling_rate, S[1])
