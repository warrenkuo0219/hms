import pandas as pd
from utils import *
'''
Code to Retrieve EEG and Spectrogram
    [originated from https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468010]
    For a specific row of train.csv, here is code to retrieve the corresponding EEG and Spectrogram. Note that train.csv
    does not give us the middle timestamp. Instead it gives us the beginning timestamp for each timewindow. The two 
    beginnings are determined from eeg_label_offset_seconds and spectrogram_label_offset_seconds. These are offsets from
     the start of the parquet file which tells us where the eeg and spectrogram time windows begin respectively.
     
Installation Requirements
    pip install pyarrow
    pip install fastparquet 
'''



BASE_PATH = './input/'

GET_ROW = 1000
EEG_PATH = 'train_eegs/'
SPEC_PATH = 'train_spectrograms/'

train = pd.read_csv(f'{BASE_PATH}train.csv')
row = train.iloc[GET_ROW]

eeg = pd.read_parquet(f'{BASE_PATH}{EEG_PATH}{row.eeg_id}.parquet')
eeg_offset = int( row.eeg_label_offset_seconds )
# it looks like eeg is sampled at 200 Hz
eeg = eeg.iloc[eeg_offset*200:(eeg_offset+50)*200]

spec_path = f'{BASE_PATH}{SPEC_PATH}{row.spectrogram_id}.parquet'
plot_spectrogram(spec_path)
spectrogram = pd.read_parquet(spec_path)
spec_offset = int( row.spectrogram_label_offset_seconds )
spectrogram = spectrogram.loc[(spectrogram.time>=spec_offset)
                     &(spectrogram.time<spec_offset+600)]

