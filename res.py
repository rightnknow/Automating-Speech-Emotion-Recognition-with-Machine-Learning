import os 
import glob
from pathlib import Path
import re 

audio_root_dir = Path(r'C:\Users\zhanglichuan\Desktop\ECE496\newdata\data')
audio_file_pattern = Path(r'**/*.wav')

def get_emotion_label(filename):
    """
    Each of the 7356 RAVDESS files has a unique filename. The filename consists of a 7-part numerical identifier 
    (e.g., 02-01-06-01-02-01-12.mp4). These identifiers define the stimulus characteristics: 

    Filename identifiers 

    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    Vocal channel (01 = speech, 02 = song).
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
    Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
    
    Here we will only use 'Emotion' as the label for our training
    
    INPUT
        filename
        
    OUTPUT
        emotion label, STARTING FROM 0 AS OPPOSED TO 1
    """
    EMOTION_LABEL_POS = 2 
    return int(re.findall(r"\d+", os.path.basename(filename))[EMOTION_LABEL_POS]) - 1


import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display 
import numpy as np

# Define a function which wil apply a butterworth bandpass filter
from scipy.signal import butter, lfilter


def butter_bandpass_filter(samples, lowcut, highcut, sample_rate, order=5):
    """
    Butterworth's filter
    """
    def butter_bandpass(lowcut, highcut, sample_rate, order=5):
        nyq = 0.5 * sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    b, a = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    y = lfilter(b, a, samples)
    return y

def clean_audio(samples, sample_rate, lowcut=30, highcut=3000):
    """
    return a preprocessed waveform with normalized volumn, bandpass filtered to only
    contain freq range of human speech, and trimmed of starting and trailing silence
    
    INPUT
        samples       1D array containing volumns at different time
        sample_rate
        lowcut        lower bound for the bandpass filter, default to 30Hz
        highcut       higher bound for the bandpass filter, default to 3000Hz
    
    OUTPUT
        filtered      1D array containing preprocessed audio information
    """
    # remove silence at the start and end of 
    trimmed, index = librosa.effects.trim(samples)
    # only keep frequencies common in human speech
    filtered = butter_bandpass_filter(samples, lowcut, highcut, sample_rate, order=5)
    return filtered

def get_melspectrogram(audio_path):
    """
    return a denoised spectrogram of audio clip given path
    
    INPUT
        audio_path    string
    OUTPUT
        spectrogram   2D array, where axis 0 is time and axis 1 is fourier decomposition
                      of waveform at different times
    """
    samples, sample_rate = librosa.load(audio_path)
    samples = clean_audio(samples, sample_rate)
    
    melspectrogram = librosa.feature.melspectrogram(samples, sample_rate) 
    
    # max L-infinity normalized the energy 
    return librosa.util.normalize(melspectrogram)


def get_mfcc(audio_path):
    samples, sample_rate = librosa.load(audio_path)
    samples = clean_audio(samples, sample_rate)
    
    mfcc = librosa.feature.mfcc(samples, sample_rate) 
    
    # max L-infinity normalized the energy 
    return librosa.util.normalize(mfcc)
     
def display_spectrogram(melspectrogram):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(melspectrogram,
                             y_axis='mel', 
                             fmax=8000,
                             x_axis='time')
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('melspectrogram')
    plt.show()
    
def align_and_downsample(spectrogram, max_freq_bins=128, max_frames=150, freq_strides=1, frame_strides=1):
    return spectrogram[:max_freq_bins:freq_strides, :max_frames:frame_strides]

def duplicate_and_stack(layer, dups=3):
    return np.stack((layer for _ in range(dups)), axis=2)


spectrograms = []
labels = []

# takes about 6-8 min on my machine
counter = 0
for audio_file in glob.iglob(str(audio_root_dir / audio_file_pattern), recursive=True):
    labels.append(get_emotion_label(audio_file))
    
    spectrogram = get_mfcc(audio_file).flatten()
    spectrograms.append(spectrogram)
    
    if counter % 100 == 0:
        print('Processing the {}th file: {}'.format(counter, audio_file))
    counter += 1

import pandas as pd
labels_dict = dict(zip(range(8), 
                       ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']))
df = pd.DataFrame(labels, columns=['label'])
df.replace({"label": labels_dict}, inplace=True)
df['label'].value_counts().plot(kind='bar')

'''
spectrogram = get_mfcc(audio_file)
display_spectrogram(spectrogram)
'''

def clean_and_pad_mfcc(array_list):
    max = 0
    for i in array_list:
        thres = i.shape[0]
        if max < thres:
            max = thres

    for i in array_list:
        print (i.shape)
        i = np.pad(i,pad_width=(0,max-i.shape[0]), mode='constant', constant_values = 0 ).flatten()
        print(i.shape)
    return array_list,max

def switch_list_to_ndarray(array_list,max):
    new_array = np.array()
    for i in array_list:
        new_array.append(i)
    return new_array



spectrograms,max_length= clean_and_pad_mfcc(spectrograms)
spectrograms = np.array(spectrograms)
print(type(spectrograms[0]))
print(spectrograms[0].shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(spectrograms, labels, test_size=0.4, random_state=0)


from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=100, batch_size=3)
ipca.fit_transform(X_train)



