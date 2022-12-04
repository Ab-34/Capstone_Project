from Hyperparameters import Hyperparameters as hp
from torch.utils.data import Dataset, DataLoader
import torch

from utils import *

import os
import unicodedata
import re


class SpeechDataset(Dataset):
    '''
    text: [T_x]
    mel: [T_y/r, n_mels*r]
    mag: [T_y, 1+n_fft/2]
    '''

    def __init__(self,emotion, r=slice(0, None)):
        print('Start loading data')
        # fpaths, texts = get_data(hp.data, r)  # thchs30
        # fpaths, texts = get_keda_data(hp.data, r)  # keda api
        # fpaths, texts = get_thchs30_data(hp.data, r)
        #fpaths, texts = get_blizzard_data(hp.data, r)
        # fpaths, texts = get_LJ_data(hp.data, r)
        self.emotion = emotion
        print(self.emotion)
        self.emotion_array = []
        if emotion is None: 
          fpaths, texts, emotion_array = get_emovdb_data(hp.data, r,None)
          self.emotion_array = emotion_array
          print(self.emotion_array)
        else:
          fpaths, texts= get_emovdb_data(hp.data, r,emotion)


        print('Finish loading data')
        self.fpaths = fpaths
        self.texts = texts

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, idx):
        _, mel, mag = load_spectrograms(self.fpaths[idx])
        mel = torch.from_numpy(mel)
        mag = torch.from_numpy(mag)
        GO_mel = torch.zeros(1, mel.size(1))  # GO frame
        mel = torch.cat([GO_mel, mel], dim=0)
        text = self.texts[idx]
        # return [text,mel,mag]
        if self.emotion_array == []:
          return {'text': text, 'mel': mel, 'mag': mag}
        else:
          return {'text': text, 'mel': mel, 'mag': mag, 'emotion': self.emotion_array[idx]}



def collate_fn(batch):
    '''
    texts: [N, max_T_x]
    mels:  [N, max_T_y/r, n_mels*r]
    mags:  [N, max_T_y, 1+n_fft/2]
    '''

    texts = [d['text'] for d in batch]
    mels = [d['mel'] for d in batch]
    mags = [d['mag'] for d in batch]
    if (len(batch[0])) > 3:
      emotions = [d['emotion'] for d in batch ]
    # emotions = torch.Tensor(emotions)

    texts = pad_sequence(texts)
    mels = pad_sequence(mels)
    mags = pad_sequence(mags)
    # emotions = pad_sequence(emotions)
    
    if (len(batch[0])) > 3:
      return {'text': texts, 'mel': mels, 'mag': mags, 'emotion': emotions}
    else:
      return {'text': texts, 'mel': mels, 'mag': mags}


def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text


def pad_sequence(sequences):
    '''
    pad sequence to same length (max length)
    ------------------
    input:
        sequences --- a list of tensor with variable length
        out --- a tensor with max length
    '''
    lengths = [data.size(0) for data in sequences]
    batch_size = len(sequences)
    max_len = max(lengths)
    trailing_dims = sequences[0].size()[1:]
    out_dims = (batch_size, max_len) + trailing_dims
    dtype = sequences[0].data.type()
    out = torch.zeros(*out_dims).type(dtype)
    for i, data in enumerate(sequences):
        out[i, :lengths[i]] = data

    return out

def get_ESD_data(data_dir, r,specific_emotion):

    wav_paths = []
    texts = []
    emotions = []
    if specific_emotion is not None:
      for file_number in [11,12,13,14,19,20]: # male speakers only
        path = os.path.join(data_dir, '00{file1}/00{file1}.txt'.format(file1 = file_number))
        file_dir = os.path.join(data_dir, '00{file1}/{emotion}'.format(file1 = file_number,emotion=specific_emotion))
        # print(file_dir)
        with open(path, 'r',errors='ignore') as f:
            for line in f.readlines():
                items = line.strip().split('\t')
                file_path = os.path.join(file_dir, items[0] + '.wav')
                # print(items)
                if os.path.isfile(file_path):
                  if items[2]==specific_emotion:
                    wav_paths.append(file_path)
                    text = items[1]
                    text = text_normalize(text) + 'E'
                    text = [hp.char2idx[c] for c in text]
                    text = torch.Tensor(text).type(torch.LongTensor)
                    texts.append(text)
            f.close()
    # print(wav_paths[-5:])
      return wav_paths[r], texts[r]

    else:
      print("NO EMOTION given")
      for file_number in [11,12,13,14,19,20]:
        path = os.path.join(data_dir, '00{file1}/00{file1}.txt'.format(file1 = file_number))
        
        for i,emotion in enumerate(['Angry','Happy','Sad','Neutral','Surprise']):
          file_dir = os.path.join(data_dir, '00{file1}/{emotion}/train'.format(file1 = file_number,emotion=emotion))
          # print(file_dir)
          with open(path, 'r',errors='ignore') as f:
              for line in f.readlines():
                  items = line.strip().split('\t')
                  file_path = os.path.join(file_dir, items[0] + '.wav')
                  # print(items)
                  if os.path.isfile(file_path):
                    # if items[2]==emotion:
                      wav_paths.append(file_path)
                      text = items[1]
                      text = text_normalize(text) + 'E'
                      text = [hp.char2idx[c] for c in text]
                      text = torch.Tensor(text).type(torch.LongTensor)
                      texts.append(text)
                      # emotions.append(emotion)
                      emotions.append(i)
              f.close()
      return wav_paths[r], texts[r], emotions[r]
          



def get_emovdb_data(data_dir, r,specific_emotion):

    wav_paths = []
    texts = []
    emotions = []
    if specific_emotion is not None:
      for file_number in [11,12]: # male speakers only
        path = os.path.join(data_dir, '00{file1}/readme.txt'.format(file1 = file_number))
        file_dir = os.path.join(data_dir, '00{file1}/{emotion}'.format(file1 = file_number,emotion=specific_emotion))
        # print(file_dir)
        with open(path, 'r',errors='ignore') as f:
            for line in f.readlines():
                items = line.strip().split('\t')
                file_path = os.path.join(file_dir, items[0])
                # print(items)
                if os.path.isfile(file_path):
                  if items[2]==specific_emotion:
                    wav_paths.append(file_path)
                    text = items[1]
                    text = text_normalize(text) + 'E'
                    text = [hp.char2idx[c] for c in text]
                    text = torch.Tensor(text).type(torch.LongTensor)
                    texts.append(text)
            f.close()
    # print(wav_paths[-5:])
      return wav_paths[r], texts[r]

    else:
      print("NO EMOTION given")
      for file_number in [11,12]:
        path = os.path.join(data_dir, '00{file1}/readme.txt'.format(file1 = file_number))
        
        for i,emotion in enumerate(['Amused', 'Angry', 'Disgusted', 'Neutral', 'Sleepy']):
          file_dir = os.path.join(data_dir, '00{file1}/{emotion}'.format(file1 = file_number,emotion=emotion))
          # print(file_dir)
          with open(path, 'r',errors='ignore') as f:
              for line in f.readlines():
                  items = line.strip().split('\t')
                  file_path = os.path.join(file_dir, items[0])
                  # print(items)
                  if os.path.isfile(file_path):
                    # if items[2]==emotion:
                      wav_paths.append(file_path)
                      text = items[1]
                      text = text_normalize(text) + 'E'
                      text = [hp.char2idx[c] for c in text]
                      text = torch.Tensor(text).type(torch.LongTensor)
                      texts.append(text)
                      # emotions.append(emotion)
                      emotions.append(i)
              f.close()
      return wav_paths[r], texts[r], emotions[r]
          




def get_eval_data(text, wav_path):
    '''
    get data for eval
    --------------
    input:
        text --- pinyin format sequence
    output:
        text --- [1, T_x]
        mel ---  [1, 1, n_mels]
    '''
    text = text_normalize(text) + 'E'
    text = [hp.char2idx[c] for c in text]
    text = torch.Tensor(text).type(torch.LongTensor)  # [T_x]
    text = text.unsqueeze(0)  # [1, T_x]
    mel = torch.zeros(1, 1, hp.n_mels)  # GO frame [1, 1, n_mels]

    _, ref_mels, _ = load_spectrograms(wav_path)
    ref_mels = torch.from_numpy(ref_mels).unsqueeze(0)

    return text, mel, ref_mels


if __name__ == '__main__':
    dataset = LJDataset()
    loader = DataLoader(dataset=dataset, batch_size=8, collate_fn=collate_fn)

    for batch in loader:
        print(batch['text'][0])
        print(batch['mel'].size())
        print(batch['mag'].size())
        break
