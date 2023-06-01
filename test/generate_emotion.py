from utils import *
from Data_emotion import *
from Hyperparameters import Hyperparameters as hp
import torch
from scipy.io.wavfile import write
from NetworkTest import *

from pypinyin import lazy_pinyin, Style

device = torch.device('cpu')
#emotion_model = Get_emotion()

def emo_func(e,speaker):

  if 'female' in speaker:
    print("female")
    emotion_vector = torch.load('/content/drive/MyDrive/GST-Tacotron-master/correct_embeddings/300/female_only/{}_mean.pth'.format(e),map_location=torch.device('cpu'))
  else:
    print("male")
    emotion_vector = torch.load('/content/drive/MyDrive/GST-Tacotron-master/correct_embeddings/300/male_only/{}_mean.pth'.format(e),map_location=torch.device('cpu'))
  return emotion_vector


def synthesis(model, eval_text, emotion):

    
    eval_text = _pinyin(eval_text)

    model.eval()

    # ref_wavs = [
    #     'ref_wav/nannan.wav', 'ref_wav/xiaofeng.wav', 'ref_wav/donaldduck.wav'
    # ]
    ref_wavs = [
        '/content/drive/MyDrive/GST-Tacotron-master/ref_wav/Female/{}.wav'.format(emotion),
        '/content/drive/MyDrive/GST-Tacotron-master/ref_wav/Male/{}.wav'.format(emotion)
        #'/content/drive/MyDrive/GST/ESD/0011/Angry/test/0011_000371.wav',
        #'/content/drive/MyDrive/GST/ESD/0011/Happy/test/0011_000721.wav',
        #'/content/drive/MyDrive/GST/ESD/0011/Sad/test/0011_001071.wav',
        #'/content/drive/MyDrive/GST/ESD/0011/Surprise/test/0011_001421.wav'
    ]

    speakers = ['0016female{}'.format(emotion), '0011male{}'.format(emotion)]# '0011Happy','0011Sad', '0011Surprise']
    #print('entering')
    
    #torch.load('/content/drive/MyDrive/GST/GST-Tacotron/embeddings/{emotion}_mean.pth'.format(emotion = emotion.lower()), map_location=torch.device('cpu'))
    wavs = {}

    for ref_wav, speaker in zip(ref_wavs, speakers):
        text, GO, ref_mels = get_eval_data(eval_text, ref_wav)
        text = text.to(device)
        GO = GO.to(device)
        ref_mels = ref_mels.to(device)
        emo_embedding = emo_func(emotion,speaker)
        #print(text,GO,ref_mels)
        mel_hat, mag_hat, attn = model(text, GO, ref_mels, emo_embedding, 0.2)
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


if __name__ == '__main__':
    text = 'The sun is so bright today .'
    epoch = 300
    # model = load_model('/content/drive/MyDrive/GST/GST-Tacotron/log/gst_for_all_ESD/state/epoch{}.pt'.format(epoch))
    model = load_model('/content/drive/MyDrive/GST-Tacotron-master/log/train1/state/epoch{}.pt'.format(epoch))

    emotion = ['Happy','Surprise','Angry','Sad','Neutral']

    for e in emotion:
      wavs = synthesis(model, text, e)
      for k in wavs:
          wav = wavs[k]
          write('/content/drive/MyDrive/GST-Tacotron-master/samples/abhijnya/{}/gender_preserve/style_0.2/{}.wav'.format(epoch,k), hp.sr, wav)