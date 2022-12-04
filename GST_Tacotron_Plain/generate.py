from utils import *
from Data import get_eval_data
from Hyperparameters import Hyperparameters as hp
import torch
from scipy.io.wavfile import write
from OG_Network import *

from pypinyin import lazy_pinyin, Style

device = torch.device('cpu')


def synthesis(model, eval_text):
    eval_text = _pinyin(eval_text)

    model.eval()

    # ref_wavs = [
    #     'ref_wav/nannan.wav', 'ref_wav/xiaofeng.wav', 'ref_wav/donaldduck.wav'
    # ]
    # ref_wavs = [
    #     'ref_wav/nannan.wav',
    #     'ref_wav/xiaofeng.wav',
    #     'ref_wav/donaldduck.wav',
    #     'ref_wav/Sejal.wav'
    # ]
    # speakers = ['nannan', 'xiaofeng', 'donaldduck','Sejal']
    ref_wavs =[
        # '/content/drive/MyDrive/GST/ESD/0011/Angry/test/0011_000371.wav',
        # '/content/drive/MyDrive/GST/ESD/0011/Neutral/test/0011_000021.wav',
        # '/content/drive/MyDrive/GST/ESD/0011/Happy/test/0011_000721.wav',
        # '/content/drive/MyDrive/GST/ESD/0011/Sad/test/0011_001071.wav',
        # '/content/drive/MyDrive/GST/ESD/0011/Surprise/test/0011_001421.wav'

        # 'ESD_ref_wav/0011_000370.wav',
        'ESD_ref_wav/0012_000370.wav',
        # 'ESD_ref_wav/0016_000370.wav',
        # 'ESD_ref_wav/0019_000370.wav'
    ]
    # speakers = ['0011_angry', '0011_neutral', '0011_happy','0011_sad','0011_surprise']
    speakers = ['0012_the sun is so bright today_angry_GST for all ESD.']
    wavs = {}

    for ref_wav, speaker in zip(ref_wavs, speakers):
        text, GO, ref_mels = get_eval_data(eval_text, ref_wav)
        text = text.to(device)
        GO = GO.to(device)
        ref_mels = ref_mels.to(device)

        mel_hat, mag_hat, attn = model(text, GO, ref_mels)
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
    text = 'The sun is so bright today. '
    model = load_model('/content/drive/MyDrive/GST/GST-Tacotron/log/gst_for_all_ESD/state/epoch225.pt')
    
    wavs = synthesis(model, text)
    for k in wavs:
        wav = wavs[k]
        write('kedar/model1/{}.wav'.format(k), hp.sr, wav)
