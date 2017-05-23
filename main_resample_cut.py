# coding=utf-8


def plot_wav(file, path ='.'):
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


def get_time_series(file_name, file_dir='.', channel=0):
    """
    Function to return time series data from wav file

    Args:
        file_name (str) : wav file name
        file_dir (str) : path where the wav file exists
        channel (int) : index of channel of wav - 0 or 1 for stereo

    Returns:
        (np.array, np.array) : (1-d time array, 1-d sinal array)
    """
    
    import os
    import wave
    with wave.open(os.path.join(file_dir, file_name), 'r') as wav_file:
        # Extract Raw Audio from Wav File
        signal = wav_file.readframes(-1)
        signal = np.fromstring(signal, 'Int16') # Int16? or Int8?

        # Split the data into channels
        channels = [[] for channel in range(wav_file.getnchannels())]
        for index, datum in enumerate(signal):
            channels[index%len(channels)].append(datum)

        # Get time from indices
        fs = wav_file.getframerate()
        time = np.linspace(0, len(signal)/len(channels)/fs, num=len(signal)/len(channels))

        # return
        return time, channels[channel]


def make_down_sample_wav(file_old, file_new, fs_new, path_old='.', path_new='.'):
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
            wav_new.setnchannels(wav_old.getnchannels())
            wav_new.setsampwidth(samplewidth_old)
            wav_new.setnframes(1)
            n_sample_new = round(len(signal_old) * fs_new / fs_old)

            # resample
            signal_new = resample(signal_old, n_sample_new)
            signal_new = signal_new.astype('Int16')

            # write new wav
            wav_new.writeframes(signal_new.copy(order='C'))


def make_cut_wav(file_old, file_new, t_start, t_end, path_old='.', path_new='.'):
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
            wav_new.setnchannels(wav_old.getnchannels())
            wav_new.setsampwidth(samplewidth_old)
            wav_new.setnframes(1)
            
            # start/end index & length
            fs_ms = fs_old / 1000 # frame per ms
            t_start_ms = t_start * 1000 # unit : ms
            t_end_ms = t_end * 1000 # unit : ms
            length = int((t_end_ms - t_start_ms) * fs_ms)

            # write new wav
            wav_old.rewind()
            anchor = wav_old.tell()
            wav_old.setpos(anchor + t_start_ms)
            wav_new.writeframes(wav_old.readframes(length))
            
# iteration
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from heardbeat_graph import draw_graph
    # Get list data
    import os
    input_dir = os.path.join('.', 'input', 'set_a')
    file_list = [file for file in os.listdir(input_dir) if os.path.splitext(file)[1] == '.wav']  # only wav files

    output_dir = os.path.join('.', 'output', 'set_a')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_original in file_list:
        # file_original
        print('Old File : {}'.format(file_original))
        image_original = os.path.join(output_dir, file_original + '.png')
        draw_graph(os.path.join(input_dir, file_original), image_original)
        # fig = plot_wav(os.path.join(input_dir, file_original))

        # file_new - resampled
        file_resampled = os.path.splitext(file_original)[0] + '_resample' + '.wav'
        print('New File : {}'.format(file_resampled))
        make_down_sample_wav(file_old=file_original, file_new=file_resampled, fs_new=4000, path_old=input_dir, path_new=output_dir)
        image_resampled = os.path.join(output_dir, file_resampled + '.png')
        draw_graph(os.path.join(output_dir, file_resampled), image_resampled)
        # fig = plot_wav(file_resampled)

        # cut file_new
        file_cut = os.path.splitext(file_resampled)[0] + '_cut' + '.wav'
        make_cut_wav(file_old=file_resampled, file_new=file_cut, t_start=0, t_end=2, path_old=output_dir, path_new=output_dir)
        image_cut = os.path.join(output_dir, file_cut + '.png')
        draw_graph(os.path.join(output_dir, file_cut), image_cut)
        # fig = plot_wav(file_cut)

    plt.show()
        




