
# coding: utf-8

# In[40]:

import os
import wave
import struct
from matplotlib import pyplot as plt
import scipy.fftpack
import numpy as np
import scipy as sp

from scipy.signal import spectrogram
from scipy.signal import welch
from scipy.signal import correlate
from scipy.signal import detrend

import my_config

from scipy.signal import butter, lfilter, freqz

DIR_A = my_config.ROOT_DIR
OUTPUT_DIR_A = './output/set_a'


def draw_graph(input_file, output_file):
    # plot time series
    with wave.open(input_file, 'r') as wav_file:
        # Extract Raw Audio from Wav File
        signal = wav_file.readframes(-1)
        sample_size = wav_file.getsampwidth()

        signal = np.fromstring(signal, 'Int' + str(8 * sample_size))

        # Split the data into channels
        channels = [[] for channel in range(wav_file.getnchannels())]
        for index, datum in enumerate(signal):
            channels[index % len(channels)].append(datum)

        # Get time from indices
        fs = wav_file.getframerate()
        timex = np.linspace(0, len(signal) / len(channels) / fs, num=len(signal) / len(channels))

        # Channels
        for channel in channels:
            plt.plot(timex, channel)

        # get data for channel 0
        N = len(channels[0])
        T = 1.0 / fs # frame rate
        x = np.linspace(0.0, N * T, N)  # x = time data
        y = [v / (2.0 ** (sample_size -1) - 1) for v in channels[0]]

        # plot signal for channel-0
        num_fig = 5
        plt.figure(figsize=(20, 5 * 2))  # 20 inches width, 2 inches height per graph

        plt.subplot(num_fig, 1, 1)
        plt.plot(x, y)
        plt.title(input_file + '\nSignal...')
        plt.xlabel('Time [s]')
        plt.ylabel('Signal')
        plt.grid()

        # plot freq. for channel-0
        fft_bounce = 300 * 10 # 300 Hz fixme
        yf = sp.fft(y)[:fft_bounce]
        xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)[:fft_bounce]

        plt.subplot(num_fig, 1, 2)
        plt.plot(xf, (2.0 / N * np.abs(yf[0:N // 2])))
        plt.title('Frequency')
        plt.xlabel('Freq. [Hz]')
        plt.ylabel('Fourier Coef.')
        plt.grid()

        # plot power spectral density
        f, Pxx = welch(y, fs, nperseg=2 ** 10)

        psd_bounce = 10
        plt.subplot(num_fig, 1, 3)
        plt.plot(f[:psd_bounce], Pxx[:psd_bounce])
        plt.title('PSD')
        plt.xlabel('Freq. [Hz]')
        plt.ylabel('Power')
        plt.grid()

        # get auto correlation fuction
        y_detrend = detrend(y) # if detrended, avg == 0
        Ryy = correlate(y_detrend, y_detrend, mode='same')

        plt.subplot(num_fig, 1, 4)
        n_x = int(3.0 * fs)
        plt.plot(timex[0:n_x], Ryy[0:n_x])
        plt.title('Auto Correlation Function...')
        plt.xlabel('Time Lag [s]')
        plt.ylabel('Auto correlation coef.')
        plt.grid()

        # get spectrogram
        f, t, Sxx = spectrogram(y_detrend, fs=fs, nperseg=2 ** 10, noverlap=2 ** 9)
        plt.subplot(num_fig, 1, 5)
        plt.pcolormesh(t, f, Sxx)
        plt.title('Spectrogram...')
        plt.ylabel('Freq. [Hz]')
        plt.xlabel('Time [s]')
        plt.ylim(0, 200)

        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.5)
        plt.savefig(output_file)
        plt.close()

        print('Saved ', output_file)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def draw_low_path(data, order, fs, cutoff):
    """
    http://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
    :param data: wave data
    :param order: ??
    :param fs: sample rate Hz
    :param cutoff: desired cutoff frequency of the filter, Hz
    :return: None
    """
    # Filter requirements.
    # order = 6
    # fs = 30.0  # sample rate, Hz
    # cutoff = 3.667  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)

    # Plot the frequency response.
    w, h = freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    T = 5.0  # seconds
    n = int(T * fs)  # total number of samples
    t = np.linspace(0, T, n, endpoint=False)
    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    data = np.sin(1.2 * 2 * np.pi * t) + 1.5 * np.cos(9 * 2 * np.pi * t) + 0.5 * np.sin(12.0 * 2 * np.pi * t)

    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(data, cutoff, fs, order)

    plt.subplot(2, 1, 2)
    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.35)
    plt.show()

if __name__ == '__main__':

    # 1. wave to graph
    # if not os.path.exists(OUTPUT_DIR_A):
    #     os.makedirs(OUTPUT_DIR_A)
    #
    # total = len(os.listdir(DIR_A))
    # start = 0
    # end = start + total
    #
    # file_list = sorted(os.listdir(DIR_A)[start:end])
    # for num, filename in enumerate(file_list):
    #     if filename.startswith('normal'):
    #         print('Input file: ', filename)
    #         draw_graph(os.path.join(DIR_A, filename), os.path.join(OUTPUT_DIR_A, filename + '.png'))


    # 2. low path filter
    start = 0
    end = 1000
    file_list = sorted(os.listdir(DIR_A)[start:end])
    for num, filename in enumerate(file_list):
        if filename.startswith('normal'):
            input_file = os.path.join(DIR_A, filename)
            print('Input file: ', filename)
            with wave.open(input_file, 'r') as wav_file:
                # Extract Raw Audio from Wav File
                signal = wav_file.readframes(-1)
                sample_size = wav_file.getsampwidth()

                signal = np.fromstring(signal, 'Int' + str(8 * sample_size))

                # Split the data into channels
                channels = [[] for channel in range(wav_file.getnchannels())]
                for index, datum in enumerate(signal):
                    channels[index % len(channels)].append(datum)

                # Get time from indices
                # fs = wav_file.getframerate()
                # draw_low_path(channels[0], 6, 44100, 40)
                draw_graph(DIR_A, OUTPUT_DIR_A)
            break