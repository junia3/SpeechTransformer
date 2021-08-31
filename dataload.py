from tqdm import tqdm
import os
import torch
import pandas as pd
import torchaudio
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as T
class PhonemeDataset(Dataset):
    def __init__(self, csv_file = '', root_dir = '', args = {}):
        self.phoneme_frame = pd.read_csv(csv_file)
        ##### get label map from given ode #####
        seq_list = list(self.phoneme_frame['sequence'])
        self.seq_listdict = {}
        for idx, row in tqdm(self.phoneme_frame.iterrows()):
            self.seq_listdict[row['audio_filename']] = list(map(int, row['sequence'].split()))
 
        self.root_dir = root_dir
        """
            set mel spectrogram arguments
        """
        if 'sample_rate' in args:
            self.sample_rate = args['sample_rate']
        else:
            self.sample_rate = 16000

        if 'n_fft' in args:
            self.n_fft = args['n_fft']
        else:
            self.n_fft = 512

        if 'win_length' in args:
            self.win_length = args['win_length']
        else:
            self.win_length = int(self.sample_rate * 0.03)

        if 'hop_length' in args:
            self.hop_length = args['hop_length']
        else:
            self.hop_length = int(self.sample_rate * 0.01)

        if 'f_max' in args:
            self.f_max = args['f_max']
        else:
            self.f_max = float(self.sample_rate/2)
        
        if 'n_mels' in args:
            self.n_mels = args['n_mels']
        else:
            self.n_mels = 40
        

        self.feature = T.MelSpectrogram(
            sample_rate = self.sample_rate,
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            f_max = self.f_max,
            n_mels = self.n_mels
        )


    def __len__(self):
        return len(self.phoneme_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        """
            phoneme csv file format looks lik
            original audio / audio / start / end / text
        """
        path = os.path.join(self.root_dir, 'LibriSpeech_wav' ,self.phoneme_frame.iloc[idx, 0])
        x, sr = torchaudio.load(path)
        
        while(x.shape[-1] < self.win_length):
            idx += 1
            path = os.path.join(self.root_dir, 'phoneme_wav' ,self.phoneme_frame.iloc[idx, 1])
            x, sr = torchaudio.load(path)
        x = self.feature(x)
        y = torch.tensor(self.seq_listdict[self.phoneme_frame.iloc[idx, 0]])
        return x, y

    def __getlabel__(self):
        return self.labelmap

