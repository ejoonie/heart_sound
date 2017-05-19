# coding = utf-8

# import modules
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# file to open
path = os.path.join('data_ejoonie', 'wavdata_abnormal_all')
file = 'abnormal_woCHD_001.wav'

# get time series for ch0 and plot
import wave
def TimeSeries(file, i_ch = 0):
    with wave.open(file,'r') as wav_file:
        # Extract Raw Audio from Wav File
        signal = wav_file.readframes(-1)
        signal = np.fromstring(signal, 'Int16')

        # Split the data into channels 
        channels = [[] for channel in range(wav_file.getnchannels())]
        for index, datum in enumerate(signal):
            channels[index%len(channels)].append(datum)

        #Get time from indices
        fs = wav_file.getframerate()
        Time = np.linspace(0, len(signal)/len(channels)/fs, num=len(signal)/len(channels))

        # return
        return fs, Time, channels[i_ch]

fs, t, y = TimeSeries(os.path.join(path, file), i_ch = 0)

plt.figure(1)
plt.plot(t, y)
plt.title('Time series  (Fs = {})'.format(fs))
plt.xlabel('Time [s]')
plt.ylabel('Signal')
plt.grid()

# detrend and plot
from scipy.signal import detrend
y_detrend = detrend(y)

plt.figure(2)
plt.plot(t, y_detrend)
plt.title('Time series  (Fs = {})'.format(fs))
plt.xlabel('Time [s]')
plt.ylabel('Signal-detrend')
plt.grid()

# get auto-correlation and plot
from scipy.signal import correlate, convolve
corr = correlate(y_detrend, y_detrend, mode = 'full')
n_data = np.minimum(len(t), len(corr))

plt.figure(3)
plt.plot(t[0:n_data], corr[0:n_data])
plt.title('Auto-Correlation  (Fs = {})'.format(fs))
plt.xlabel('Time Lag [s]')
plt.ylabel('Auto-Correlation')
plt.grid()

# get-filterred signal and plot
from scipy.signal import butter, lfilter
cutoff = 200
N = 5 # filter oder
Wn = cutoff / (fs * 0.5)
b, a = butter(N, Wn , btype = 'low', analog = False)
y_filtered = lfilter(b, a, y_detrend) # low pass filter

plt.figure(4)
plt.plot(t, y_filtered)
plt.title('Time series  (Fs = {}) (Cutoff Freq. = {})'.format(fs, cutoff))
plt.xlabel('Time [s]')
plt.ylabel('Signal - filtered')
plt.grid()

# get fft and plot
T = 1.0 / fs # time interval
n_sample = len(y_filtered)

freq = np.linspace(0.0, 1.0/(2.0*T), n_sample//2)
yf = sp.fft(y_filtered)

plt.figure(5)
plt.plot(freq, 2.0/n_sample * np.abs(yf[0:n_sample//2]))
plt.title('FFT')
plt.xlabel('Freq. [Hz]')
plt.ylabel('Fourier Coef.')
plt.grid()

# get psd and plot
from scipy.signal import welch
nperseg = fs // 4 # size of sagment to fft
noverlap = nperseg // 100 * 90 # segments overlaped rate 90%
f, Pxx = welch(y_filtered, fs = fs, nperseg= nperseg, noverlap = noverlap, window = sp.signal.hamming(nperseg))

plt.figure(6)
plt.plot(f, Pxx)
plt.title('PSD')
plt.xlabel('Freq. [Hz]')
plt.ylabel('Power')
plt.grid()


# get spectrogram
from scipy.signal import spectrogram
nperseg = fs // 4 # size of sagment to fft
noverlap = nperseg // 100 * 90 # segments overlaped at 90%
f, t, Sxx = spectrogram(y_filtered, fs = fs, nperseg= nperseg, noverlap = noverlap, window = sp.signal.hamming(nperseg))

plt.figure(7)
plt.pcolormesh(t, f, Sxx)
plt.title('Spectrogram')
plt.xlabel('Time [s]')
plt.ylabel('Freq. [Hz]')
plt.grid()

plt.show()
