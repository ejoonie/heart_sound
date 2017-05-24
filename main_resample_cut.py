# coding=utf-8

# fuctions
def PlotWave(file, path = '.'):
    """
    Function to make plt.figure obj. from wav file

    Args:
        file (str) : wav file name
        path (str) : path where the wav file exists

    Returns:
        plt.figure obj. : time series figure        
    """

    import os
    import wave
    import matplotlib.pyplot as plt
    with wave.open(os.path.join(path, file),'r') as wav_file:
        #Extract Raw Audio from Wav File
        signal = wav_file.readframes(-1)
        signal = np.fromstring(signal, 'Int16')

        #Split the data into channels 
        channels = [[] for channel in range(wav_file.getnchannels())]
        for index, datum in enumerate(signal):
            channels[index%len(channels)].append(datum)

        #Get time from indices
        fs = wav_file.getframerate()
        Time = np.linspace(0, len(signal)/len(channels)/fs, num=len(signal)/len(channels))

        #Plot
        fig = plt.figure()
        plt.title('Signal - {}'.format(file))
        for channel in channels:
            plt.plot(Time,channel)
        plt.xlabel('Time [s]')
        plt.grid()

        #return
        return fig

def GetTimeSeries(file, path = '.', channel = 0):
    """
    Function to return time series data from wav file

    Args:
        file (str) : wav file name
        path (str) : path where the wav file exists
        channel (int) : index of channel of wav - 0 or 1 for stereo

    Returns:
        (np.array, np.array) : (1-d time array, 1-d sinal array)
    """
    
    import os
    import wave
    with wave.open(os.path.join(path, file),'r') as wav_file:
        #Extract Raw Audio from Wav File
        signal = wav_file.readframes(-1)
        signal = np.fromstring(signal, 'Int16')

        #Split the data into channels 
        channels = [[] for channel in range(wav_file.getnchannels())]
        for index, datum in enumerate(signal):
            channels[index%len(channels)].append(datum)

        #Get time from indices
        fs = wav_file.getframerate()
        Time = np.linspace(0, len(signal)/len(channels)/fs, num=len(signal)/len(channels))

        # return
        return Time, channels[channel]

def MakeDownSampleWav(file_old, file_new, fs_new, path_old = '.', path_new = '.'):
    """
    Function to make down-sampled wav file

    Args:
        file_old (str) : wav file name to read
        file_new (str) : wav file name to write
        fs_new (int) : sampling frequency for down sampling
        path_old (str) : path where the wav file to read exists
        path_new (str) : path where the new wav file to write in

    Returns:
        None
    """
    
    import os
    import wave
    from scipy.signal import resample
    
    # read old wav file
    with wave.open(os.path.join(path_old, file_old), 'r') as wav_old:
        # info. of old wav
        signal_old = wav_old.readframes(-1)
        signal_old = np.fromstring(signal_old, 'Int16')
        fs_old = wav_old.getframerate()
        n_channel_old = wav_old.getnchannels()
        samplewidth_old = wav_old.getsampwidth()

        # write new wav file
        with wave.open(os.path.join(path_new, file_new), 'w') as wav_new:
            # info. of new wav
            wav_new.setframerate(fs_new)
            wav_new.setnchannels(n_channel_old)
            wav_new.setsampwidth(samplewidth_old)
            wav_new.setnframes(1)
            n_sample_new = round(len(signal_old) * fs_new / fs_old)

            # resample
            signal_new = resample(signal_old, n_sample_new)
            signal_new = signal_new.astype('Int16')

            # write new wav
            wav_new.writeframes(signal_new.copy(order='C'))


def MakeCutWav(file_old, file_new, t_start, t_end, path_old = '.', path_new = '.'):
    """
    Function to make cut wav file
    Args:
        file_old (str) : wav file name to read
        file_new (str) : wav file name to write
        t_start  (float) : time [second] to start to cut
        t_end    (float) : time [second] to finish to cut
        path_old (str) : path where the wav file exists
        path_new (str) : path where the new wav file to write in
    
    Returns:
        None
    """

    import os
    import wave
    
    # read old wav file
    with wave.open(os.path.join(path_old, file_old), 'r') as wav_old:
        # info. of old wav
        signal_old = wav_old.readframes(-1)
        signal_old = np.fromstring(signal_old, 'Int16')
        fs_old = wav_old.getframerate()
        n_channel_old = wav_old.getnchannels()
        samplewidth_old = wav_old.getsampwidth()

        # write new wav file
        with wave.open(os.path.join(path_new, file_new), 'w') as wav_new:
            # new wav file
            wav_new.setframerate(fs_old)
            wav_new.setnchannels(n_channel_old)
            wav_new.setsampwidth(samplewidth_old)
            wav_new.setnframes(1)
            
            # start/end index & length
            fs_ms = fs_old / 1000 # frame per ms
            t_start_ms = int(t_start * 1000) # unit : ms
            t_end_ms = int(t_end * 1000) # unit : ms
            length = int((t_end_ms - t_start_ms) * fs_ms)

            # write new wav
            wav_old.rewind()
            anchor = wav_old.tell()
            wav_old.setpos(anchor + t_start_ms)
            wav_new.writeframes(wav_old.readframes(length))
            

def GetDuration(file, path ='.'):
    """
    Function to make down-sampled wav file

    Args:
        file (str) : wav file name to read
        path (str) : path where the wav file to read exists

    Returns:
        (float) : time length of wav file [sec]
    """
    import os
    import wave

    with wave.open(os.path.join(path, file), 'r') as wav:
        frames = wav.getnframes()
        fs = wav.getframerate()
        duration = frames / float(fs)
        return duration

# iteration
if __name__ == '__main__':
    """Making split wav files"""

    # import module
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import my_config

    DIR = my_config.ROOT_DIR

    # Get list wav file
    list_file = [file for file in os.listdir(DIR) if os.path.splitext(file)[1] == '.wav']  # only wav files

    # Iteration to resampled and cut data
    for i, file in enumerate(list_file):
        # file_old
        print('Old File : {}'.format(file))

        # make resampled file
        file_new = 'resample_' + str(i) +'.wav'
        print(' ... Resampled File : {}'.format(file_new))
        MakeDownSampleWav(file_old = file, file_new = file_new, fs_new = 4000, path_old = DIR, path_new = DIR)

        # split wav file by t_cut
        duration = GetDuration(file, DIR)
        t_cut = 2.5  # cut time duration = 2.5 sec
        n_seg = int(duration / t_cut)
        for j in range(n_seg):
            t_start = j * t_cut
            t_end = t_start + t_cut
            file_new_cut = 'keggle_resample_cut_' + str(i) + '_' + str(j) + '.wav'
            MakeCutWav(file_old = file_new, file_new = file_new_cut, t_start = t_start, t_end = t_end, path_old = DIR, path_new = DIR)
            print(' ... ... Resampled and Cut File : {}'.format(file_new_cut))
