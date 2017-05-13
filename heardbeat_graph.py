
# coding: utf-8

# In[40]:

import os
import wave
import struct
from matplotlib import pyplot as plt
import scipy.fftpack
import numpy as np
import scipy as sp

DIR_A = './input/set_a/'
OUTPUT_DIR_A = './output/set_a/'

def get_xy_normal(filename):
    f = wave.open(filename)
    
    print(filename)

    # read frames
    frames = f.readframes(-1)

    # checking the sample width
    print('Sample width: ', f.getsampwidth())

    # samples - unpack byte
    samples = struct.unpack('h' * f.getnframes(), frames)

    # frame rates
    framerate = f.getframerate()
    print('Frame rate: ', framerate)

    # timing information??
    t = [float(i)/framerate for i in range(len(samples))]

    # plotting
#     plt.plot(t, samples)
    
    return t, samples

def get_xy_fft(filename):
    x, y = get_xy_normal(filename)
    # number of sample points
    N = len(x)
    
    # interval
    T = x[1] - x[0]

    # fft
    yf = scipy.fftpack.fft([v / (2.0**15) for v in y])
    
    xf = np.arange(N / 100)
    
    return xf, np.abs(yf[0:len(xf)])
    
def draw_graph(input_file, output_file):
    # plot time series
    with wave.open(input_file, 'r') as wav_file:
        # Extract Raw Audio from Wav File
        signal = wav_file.readframes(-1)
        signal = np.fromstring(signal, 'Int16') # fixme check the sample size

        # Split the data into channels
        channels = [[] for channel in range(wav_file.getnchannels())]
        for index, datum in enumerate(signal):
            channels[index % len(channels)].append(datum)

        # Get time from indices
        fs = wav_file.getframerate()
        Time = np.linspace(0, len(signal) / len(channels) / fs, num=len(signal) / len(channels))

        # Channels
        for channel in channels:
            plt.plot(Time, channel)

    # get data for channel 0
    N = len(channels[0])
    T = 1.0 / fs
    x = np.linspace(0.0, N * T, N)  # x = time data
    y = channels[0]

    # plot signal for channel-0
    num_fig = 5
    plt.figure(figsize=(20, 5 * 2))  # 20 inches width, 2 inches height per graph

    plt.subplot(num_fig, 1, 1)
    plt.plot(x, y)
    plt.title(filename + '\nSignal...')
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
    from scipy.signal import welch
    f, Pxx = welch(y, fs, nperseg=2 ** 10)

    psd_bounce = 10
    plt.subplot(num_fig, 1, 3)
    plt.plot(f[:psd_bounce], Pxx[:psd_bounce])
    plt.title('PSD')
    plt.xlabel('Freq. [Hz]')
    plt.ylabel('Power')
    plt.grid()

    # get auto correlation fuction
    from scipy.signal import correlate
    from scipy.signal import detrend

    y_detrend = detrend(y)
    Ryy = correlate(y_detrend, y_detrend, mode='same')

    plt.subplot(num_fig, 1, 4)
    n_x = int(3.0 * fs)
    plt.plot(Time[0:n_x], Ryy[0:n_x])
    plt.title('Auto Correlation Function...')
    plt.xlabel('Time Lag [s]')
    plt.ylabel('Auto correlation coef.')
    plt.grid()

    # get spectrogram
    from scipy.signal import spectrogram
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


if __name__ == '__main__':
#     graph_normal(NORMAL_FILE)

    if not os.path.exists(OUTPUT_DIR_A):
        os.makedirs(OUTPUT_DIR_A)

    total = len(os.listdir(DIR_A))
    start = 0
    end = start + total 
    # fig = plt.figure(figsize=(20, total * 2)) # 20 inches width, 2 inches height per graph

    filelist = sorted(os.listdir(DIR_A)[start:end])
    for num, filename in enumerate(filelist):
        if filename.startswith('normal'):
            print('Input file: ', filename)
            draw_graph(os.path.join(DIR_A, filename), os.path.join(OUTPUT_DIR_A, filename + '.png'))
#         x, y = get_xy_normal(os.path.join(DIR_A, filename))
#         x, y = get_xy_fft(os.path.join(DIR_A, filename))
#         plt.subplot(total, 1, num + 1)
#         plt.plot(x, y)
#         plt.title(filename)
        
    # plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.35)
    # plt.show()
    
    
    
    
    
    


# In[ ]:




# In[ ]:



