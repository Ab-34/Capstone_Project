from utils import *
from Data_emotion import get_eval_data
from Hyperparameters import Hyperparameters as hp
import torch
from scipy.io.wavfile import write
from NetworkTest import *

from pypinyin import lazy_pinyin, Style

device = torch.device('cpu')

def emo_func(e,speaker):
  if speaker=='Female':
    #print("female")
    emotion_vector = torch.load('correct_embeddings/300/female_only/{}_mean.pth'.format(e),map_location=torch.device('cpu'))
  else:
    #print("male")
    emotion_vector = torch.load('correct_embeddings/300/male_only/{}_mean.pth'.format(e),map_location=torch.device('cpu'))
  return emotion_vector

def synthesis(model, eval_text, emotion, speaker):
    #print('fffffffffffffffff',speaker)
    eval_text = _pinyin(eval_text)

    model.eval()
#     print(emotion)
#     print(eval_text)

    speakers = {
      'Male' :  {
        'Angry' : ['male/Angry.wav'],
        'Sad' : ['male/Sad.wav'],
        'Surprise' : ['male/Surprise.wav'],
        'Neutral' : ['male/Neutral.wav'],
        'Happy' : ['male/Happy.wav']
      },
      'Female' : {
        'Angry' : ['female/Angry.wav'],
        'Sad' : ['female/Sad.wav'],
        'Surprise' : ['female/Surprise.wav'],
        'Neutral' : ['female/Neutral.wav'],
        'Happy' : ['female/Happy.wav']
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
    
    #emo_embedding = torch.load('/content/drive/MyDrive/GST/GST-Tacotron/embeddings/{emotion}_mean.pth'.format(emotion = emotion.lower()), map_location=torch.device('cpu'))
    emo_embedding = emo_func(emotion,speaker)
    #print(speaker)
    for ref_wav, speaker in zip(speakers[speaker][emotion], speaker_dict[emotion]):
        text, GO, ref_mels = get_eval_data(eval_text, ref_wav)
        text = text.to(device)
        GO = GO.to(device)
        ref_mels = ref_mels.to(device)
        mel_hat, mag_hat, attn = model(text, GO, ref_mels,emo_embedding,0.2)
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


def pred(text,emo,speaker,idx):
    model = load_model('epoch300.pt')
    ''' Input weights for style and emotion separately each time '''
    while(True):
        
        wavs = synthesis(model, text, emo, speaker)

        path = "{}.wav".format(idx)
        for k in wavs:
            wav = wavs[k]
            write(path,hp.sr,wav)
        return path
    

