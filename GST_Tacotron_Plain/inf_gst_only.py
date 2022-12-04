from utils import *
from Data import get_eval_data
from Hyperparameters import Hyperparameters as hp
import torch
from scipy.io.wavfile import write
from Network_Test import *

from pypinyin import lazy_pinyin, Style

device = torch.device('cpu')


def synthesis(model, eval_text, emotion, speaker):
    eval_text = _pinyin(eval_text)

    model.eval()

    speakers = {
      '0011' :  {
        'Angry' : ['/content/drive/MyDrive/GST/ESD/0011/Angry/evaluation/0011_000351.wav'],
        'Sad' : ['/content/drive/MyDrive/GST/ESD/0011/Sad/evaluation/0011_001057.wav'],
        'Surprise' : ['/content/drive/MyDrive/GST/ESD/0011/Surprise/evaluation/0011_001420.wav'],
        'Neutral' : ['/content/drive/MyDrive/GST/ESD/0011/Neutral/evaluation/0011_000001.wav'],
        'Happy' : ['/content/drive/MyDrive/GST/ESD/0011/Happy/evaluation/0011_000701.wav']
      },
      '0016' : {
        'Angry' : ['/content/drive/MyDrive/GST/ESD/0016/Angry/evaluation/0016_000351.wav'],
        'Sad' : ['/content/drive/MyDrive/GST/ESD/0016/Sad/evaluation/0016_001057.wav'],
        'Surprise' : ['/content/drive/MyDrive/GST/ESD/0016/Surprise/evaluation/0016_001420.wav'],
        'Neutral' : ['/content/drive/MyDrive/GST/ESD/0016/Neutral/evaluation/0016_000001.wav'],
        'Happy' : ['/content/drive/MyDrive/GST/ESD/0016/Happy/evaluation/0016_000701.wav']
      }
    }

    speaker_dict ={
      'Angry' : '0020_{input_text}_Angry'.format(input_text=eval_text),
      'Sad' : '0012_{input_text}_Sad'.format(input_text=eval_text),
      'Surprise' : '0019_{input_text}_Surprise'.format(input_text=eval_text),
      'Neutral' : '0019_{input_text}_Neutral'.format(input_text=eval_text),
      'Happy' : '0012_{input_text}_Happy'.format(input_text = eval_text)
    }
    wavs = {}
    
    emo_embedding = torch.load('/content/drive/MyDrive/GST/GST-Tacotron/embeddings/{emotion}_mean.pth'.format(emotion = emotion.lower()), map_location=torch.device('cpu'))
    for ref_wav, speaker in zip(speakers[speaker][emotion], speaker_dict[emotion]):
        text, GO, ref_mels = get_eval_data(eval_text, ref_wav)
        text = text.to(device)
        GO = GO.to(device)
        ref_mels = ref_mels.to(device)
        
        mel_hat, mag_hat, attn = model(text, GO, ref_mels,emo_embedding)
        mag_hat = mag_hat.squeeze().detach().cpu().numpy()
        attn = attn.squeeze().detach().cpu().numpy()

        wav_hat = spectrogram2wav(mag_hat)
        wavs[speaker] = wav_hat

    return wavs


def load_model(checkpoint_path):
    model = Tacotron().to(device)
    model.load_state_dict(
        torch.load(
            checkpoint_path, map_location=lambda storage, location: storage))
    return model


def _pinyin(s):
    symbols = '0123456789abcdefghijklmnopqrstuvwxyz '
    s = lazy_pinyin(s, style=Style.TONE2)
    yin = []
    for token in s:
        if token != ' ':
            a = ''
            for c in token:
                if c in symbols:
                    a += c
            yin.append(a)
    a = ''
    s = ' '.join(yin)
    for i in range(len(s)):
        if s[i] == ' ' and i < len(s) - 1 and s[i + 1] == ' ':
            continue
        a += s[i]
    return a


def pred(text,emo,speaker):
    model = load_model('/content/drive/MyDrive/GST/GST-Tacotron/log/train1/state/epoch273.pt')
    

    ''' Input weights for style and emotion separately each time '''
    while(True):

        wavs = synthesis(model, text, emo, speaker)
        path = "/content/drive/MyDrive/GST/GST-Tacotron/kedar/streamtest/t1.wav"
        for k in wavs:
            wav = wavs[k]
            write(path,hp.sr,wav)
        return path
    

