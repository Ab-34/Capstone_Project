from utils import *
from Data_test import get_eval_data, SpeechDataset, collate_fn
from Hyperparameters import Hyperparameters as hp
import torch
from scipy.io.wavfile import write
from Network import *

from torch.utils.data import DataLoader

from pypinyin import lazy_pinyin, Style

device = torch.device('cpu')

# emotion_model = Get_emotion().cuda()

def get_emotion_dict(emotion):

    # train_dataset_anger = SpeechDataset(emotion,r=slice(hp.eval_size, None))

    # train_loader_emotion = DataLoader(dataset=train_dataset_anger, batch_size=hp.batch_size, collate_fn=collate_fn, num_workers=4, shuffle=True)
    # train_loader_emotion = torch.load('/content/drive/MyDrive/GST/GST-Tacotron/train_datasets/train_loader_anger.pth')

    # temp = []
    # for i, batch in enumerate(train_loader_emotion):
    #     mels = batch['mel'].to(device)
        # print(mels.shape)
    #     mels_input = mels[:, :-1, :]  # shift
    #     mels_input = mels_input[:, :, -hp.n_mels:]  # get last frame
    #     ref_mels = mels[:, 1:, :]
    #     temp.append(emotion_model(ref_mels))
    # torch.save(temp,'/content/drive/MyDrive/GST/GST-Tacotron/embeddings/neutral.pth')
    # emotion_vector = torch.mean(torch.cat(temp, dim=0),0)
    # torch.save(emotion_vector,'/content/drive/MyDrive/GST/GST-Tacotron/embeddings/neutral_mean.pth')
    # emotion_vector = torch.load('/content/drive/MyDrive/GST/GST-Tacotron/embeddings/anger_mean.pth',map_location=torch.device('cpu'))
    emotion_vector = torch.load('/content/drive/MyDrive/GST/GST-Tacotron/embeddings/maleonly100/{emotion}_mean.pth'.format(emotion = emotion))
    emotion_vector_rep = torch.mean(torch.cat(emotion_vector, dim = 0),0)

    # emotion_vector = torch.cat(emotion_vector, dim = 0)

    return emotion_vector_rep


def synthesis(model, eval_text, emotion):
    eval_text = _pinyin(eval_text)

    model.eval()

    ref_wavs =[
        # 'ESD_ref_wav/happy/0011_000721.wav',
        # 'ESD_ref_wav/happy/0012_000721.wav',
        # 'ESD_ref_wav/happy/0016_000721.wav',
        # 'ESD_ref_wav/happy/0019_000721.wav'

          'ESD_ref_wav/0012_000370.wav'
        # 'ESD_ref_wav/0011_000370.wav',
        # '/content/drive/MyDrive/GST/ESD/0011/Angry/test/0011_000371.wav',
        # '/content/drive/MyDrive/GST/ESD/0017/Neutral/test/0017_000021.wav',
        # '/content/drive/MyDrive/GST/ESD/0011/Happy/test/0011_000721.wav',
        # '/content/drive/MyDrive/GST/ESD/0011/Sad/test/0011_001071.wav',
        # '/content/drive/MyDrive/GST/ESD/0011/Surprise/test/0011_001421.wav'
        # 'ESD_ref_wav/0016_000370.wav',
        # 'ESD_ref_wav/0019_000370.wav'
    ]
    # speakers = ['0011', '0012', '0016','0019']
    # speakers = ['0011_angry', '0011_neutral', '0011_happy','0011_sad','0011_surprise']
    speakers = ['new-17-0.3']
    wavs = {}

    for ref_wav, speaker in zip(ref_wavs, speakers):
        text, GO, ref_mels = get_eval_data(eval_text, ref_wav)
        text = text.to(device)
        GO = GO.to(device)
        ref_mels = ref_mels.to(device)
        print("generating emo for")
        emotion_vector = get_emotion_dict(emotion)
        emotion_vector = emotion_vector.to(device)
        print("emo generated")
        mel_hat, mag_hat, attn = model(text, GO, ref_mels, emotion_vector)
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
    emotion = 'Angry'
    model = load_model('/content/drive/MyDrive/GST/GST-Tacotron/log/gst_for_all_ESD/state/epoch225.pt')
    # for alpha in range(0,100,5):
    wavs = synthesis(model, text, emotion)
    for k in wavs:
        wav = wavs[k]
        write('Demo/weighted/test/{}.wav'.format(k), hp.sr, wav)
