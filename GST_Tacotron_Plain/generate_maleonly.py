from utils import *
from Data_test import get_eval_data, SpeechDataset, collate_fn
from Hyperparameters import Hyperparameters as hp
import torch
from scipy.io.wavfile import write
from Network import *
import torch.nn as nn
from GST import GST
import numpy as np
import os.path
from os import path


from torch.utils.data import DataLoader

from pypinyin import lazy_pinyin, Style

device = torch.device(hp.device)

# emotion_model = Get_emotion().cuda()

def get_emotion_dict(emotion, gst):

    # train_dataset_anger = SpeechDataset(emotion,r=slice(hp.eval_size, None))

    # train_loader_emotion = DataLoader(dataset=train_dataset_anger, batch_size=hp.batch_size, collate_fn=collate_fn, num_workers=4, shuffle=True)
    # # train_loader_emotion = torch.load('/content/drive/MyDrive/GST/GST-Tacotron/train_datasets/train_loader_anger.pth')
    # train_loader_emotion = torch.load('/content/drive/MyDrive/GST/GST-Tacotron/train_datasets_male/train_loader_{emotion}.pth'.format(emotion = emotion.lower()))

    # temp = []
    # for i, batch in enumerate(train_loader_emotion):
    #     mels = batch['mel'].to(device)
    #     # print(mels.shape)
    #     mels_input = mels[:, :-1, :]  # shift
    #     mels_input = mels_input[:, :, -hp.n_mels:]  # get last frame
    #     ref_mels = mels[:, 1:, :]
    #     with torch.no_grad():
    #             emo = gst(ref_mels).to('cpu')
    #             temp.append(emo)

    # emotion_vector = torch.mean(torch.cat(temp, dim=0),0)
    # torch.save(emotion_vector,'/content/drive/MyDrive/GST/GST-Tacotron/embeddings/maleonly1_1/{emotion}_mean.pth'.format(emotion = emotion))

    emotion_vector = torch.load('/content/drive/MyDrive/GST/GST-Tacotron/embeddings/maleonly1_1/{emotion}_mean.pth'.format(emotion = emotion))

    return emotion_vector


def synthesis(model_enc,model_gst,model_emb,model_dec, eval_text, emotion,input_text,alpha = 0.7,beta = 0.3):
    eval_text = _pinyin(eval_text)

    model_enc.eval()
    model_gst.eval()
    model_emb.eval()
    model_dec.eval()

    ref_wav_dict = {
      'Angry' : ['ESD_ref_wav/Angry/0020_000408.wav'],
      'Sad' : ['ESD_ref_wav/Sad/0012_001398.wav'],
      'Surprise' : ['ESD_ref_wav/Surprise/0019_001453.wav'],
      'Neutral' : ["ESD_ref_wav/Neutral/0019_000052.wav"],
      'Happy' : ['ESD_ref_wav/Happy/0012_000721.wav']
    }

    speaker_dict ={
      'Angry' : '0020_{input_text}_Angry'.format(input_text=input_text),
      'Sad' : '0012_{input_text}_Sad'.format(input_text=input_text),
      'Surprise' : '0019_{input_text}_Surprise'.format(input_text=input_text),
      'Neutral' : '0019_{input_text}_Neutral'.format(input_text=input_text),
      'Happy' : '0012_{input_text}_Happy'.format(input_text = input_text)
    }
    ref_wavs =[
        # 'ESD_ref_wav/happy/0011_000721.wav',
        # 'ESD_ref_wav/happy/0012_000721.wav',
        # 'ESD_ref_wav/happy/0016_000721.wav',
        # 'ESD_ref_wav/happy/0019_000721.wav'

        # 'ESD_ref_wav/0011_000370.wav',
        'ESD_ref_wav/0012_000370.wav'
        # 'ESD_ref_wav/0016_000370.wav',
        # 'ESD_ref_wav/0019_000370.wav'

        # '/content/drive/MyDrive/GST/ESD/0011/Angry/test/0011_000371.wav',
        # '/content/drive/MyDrive/GST/ESD/0011/Neutral/test/0011_000021.wav',
        # '/content/drive/MyDrive/GST/ESD/0011/Happy/test/0011_000721.wav',
        # '/content/drive/MyDrive/GST/ESD/0011/Sad/test/0011_001071.wav',
        # '/content/drive/MyDrive/GST/ESD/0011/Surprise/test/0011_001421.wav'
    ]
    # speakers = ['0011', '0012', '0016','0019']
    speakers = ['0012_{input_text}_{emotion}'.format(input_text=input_text,emotion = emotion)]
    # speakers = ['0012_AtTheRootsOfTheBushOfTheGrass_angry_noVectorAtAll']
    
    # speakers = ['0011_angry_on_angry', '0011_angry_on_neutral', '0011_angry_on_happy','0011_angry_on_sad','0011_angry_on_surprise']

    wavs = {}

    for ref_wav, speaker in zip(ref_wav_dict[emotion], speaker_dict[emotion]):
        print(ref_wav, speaker)
        text, GO, ref_mels = get_eval_data(eval_text, ref_wav)
        text = text.to(device)
        GO = GO.to(device)
        ref_mels = ref_mels.to(device)
        # print("generating emo")
        emotion_vector = get_emotion_dict(emotion, model_gst)
        emotion_vector = emotion_vector.to(device)
        print("{} Emotion vector generated".format(emotion))
        #mel_hat, mag_hat, attn = model(text, GO, ref_mels, emotion_vector)
        embedded = model_emb(text)  # [N, T_x, E]
        memory, encoder_hidden = model_enc(embedded)  # [N, T_x, E]
        style_embed = model_gst(ref_mels)  # [N, 256]
        style_embed = style_embed.expand_as(memory)
        emotion_vector = emotion_vector.expand_as(memory)
        
        # GST Embeddings + Emotion Embeddings
        memory = memory + alpha*style_embed + beta*emotion_vector

        # Without Emotion Embeddings
        # memory = memory + style_embed

        # Without GST Embeddings
        # memory = memory + emotion_vector

        # Without any embeddings
        # memory = memory

        mel_hat, mag_hat, attn = model_dec(GO,memory)
        mag_hat = mag_hat.squeeze().detach().cpu().numpy()
        attn = attn.squeeze().detach().cpu().numpy()

        wav_hat = spectrogram2wav(mag_hat)
        wavs[speaker] = wav_hat
    print(wavs)
    return wavs


def load_model1(checkpoint_path):
    model = Encoder().to(device)
    model.load_state_dict(
        torch.load(
            checkpoint_path, map_location=lambda storage, location: storage))
    return model

def load_model2(checkpoint_path):
    model = GST().to(device)
    model.load_state_dict(
        torch.load(
            checkpoint_path, map_location=lambda storage, location: storage))
    return model

def load_model3(checkpoint_path):
    model = nn.Embedding(len(hp.vocab), hp.E).to(device)
    model.load_state_dict(
        torch.load(
            checkpoint_path, map_location=lambda storage, location: storage))
    return model

def load_model4(checkpoint_path):
    model = Decoder().to(device)
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
    model_enc = load_model1('/content/drive/MyDrive/GST/GST-Tacotron/log/male_only_1,1/state/encoder/epoch200.pt')
    model_gst = load_model2('/content/drive/MyDrive/GST/GST-Tacotron/log/male_only_1,1/state/gst/epoch200.pt')
    model_emb = load_model3('/content/drive/MyDrive/GST/GST-Tacotron/log/male_only_1,1/state/embedding/epoch200.pt')
    model_dec = load_model4('/content/drive/MyDrive/GST/GST-Tacotron/log/male_only_1,1/state/decoder/epoch200.pt')
    emo_list = ["Neutral","Angry","Sad","Happy","Surprise"]
    
    ''' Interate over weights for style and emotion vectors '''
    # print("Using Epoch 200 models for male only")
    # texts = ["Today is going to be nice", "I wish you the best for the future", "I need some food","You are working alot nowadays","We are failing the review", "I kicked them out after an argument"]
    # texts_idx = 0
    # for alpha in [0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]:
    #   if path.exists('/content/drive/MyDrive/GST/GST-Tacotron/Demo/weighted/FindingBestWeights/alpha{}'.format(alpha)) == False:
    #     os.mkdir('/content/drive/MyDrive/GST/GST-Tacotron/Demo/weighted/FindingBestWeights/alpha{}'.format(alpha))
    #   if alpha == 0.00 or alpha == 0.10 or alpha == 0.2 or alpha == 0.3 or alpha == 0.4 or alpha == 0.5:
    #       texts_idx = texts_idx + 1
    #   text = texts[texts_idx]
    #   for beta in np.arange(0.5,2.05,0.05):
    #     print("Alpha : {} || Beta : {}".format(alpha,beta))
    #     for emotion in ["Neutral","Angry","Sad","Happy","Surprise"]:
    #       if path.exists('/content/drive/MyDrive/GST/GST-Tacotron/Demo/weighted/FindingBestWeights/alpha{}/{}'.format(alpha,emotion)) == False:
    #         os.mkdir('/content/drive/MyDrive/GST/GST-Tacotron/Demo/weighted/FindingBestWeights/alpha{}/{}'.format(alpha,emotion))
    #       wavs = synthesis(model_enc, model_gst,model_emb,model_dec, text, emotion, text, alpha, beta)
    #       for k in wavs:
    #           wav = wavs[k]
    #           write('/content/drive/MyDrive/GST/GST-Tacotron/Demo/weighted/FindingBestWeights/alpha{alpha}/{emotion}/weights_{alpha}gst_{beta}emo_{k}_epoch200_actualmaleonly.wav'.format(alpha = alpha,emotion=emotion, beta=beta, k=k), hp.sr, wav)

    ''' Input weights for style and emotion separately each time '''
    while(True):
      text = input("Enter text to generate audio : ")
      emo = int(input("0: Neutral\n1: Angry\n2: Sad\n3: Happy\n4: Surprise\nChoose emotion by entering the number : "))
      alpha = float(input("Enter weight for style vector : "))
      beta = float(input("Enter weight for emotion vector : "))

      emotion = emo_list[emo]
      wavs = synthesis(model_enc, model_gst,model_emb,model_dec, text, emotion, text, alpha, beta)
      for k in wavs:
          wav = wavs[k]
          write('/content/drive/MyDrive/GST/GST-Tacotron/Demo/weighted/test/RefWavsEmotion/weights_{alpha}gst_{beta}{emotion}_{text}_epoch200_actualmaleonly.wav'.format(alpha = alpha,emotion=emotion, beta=beta, text = text), hp.sr, wav)
          # write('/content/drive/MyDrive/Demo/model4/weights_0.7gst_0.3emo_{}_epoch294.wav'.format(k), hp.sr, wav)
      print("WOOHOO\n")

    ''' Fixed weights for style and emotion vectors '''
    # while(True):
    #   text = input("Enter text to generate audio : ")
    #   emo = int(input("0: Neutral\n1: Angry\n2: Sad\n3: Happy\n4: Surprise\nChoose emotion by entering the number : "))
    #   alpha = float(input("Enter weight for style vector : "))
    #   beta = float(input("Enter weight for emotion vector : "))

    #   emotion = emo_list[emo]
    #   wavs = synthesis(model_enc, model_gst,model_emb,model_dec, text, emotion, text, alpha, beta)
    #   for k in wavs:
    #       wav = wavs[k]
    #       write('/content/drive/MyDrive/GST/GST-Tacotron/Demo/weighted/weights_{alpha}gst_{beta}}emo_{k}_epoch101.wav'.format(alpha = alpha, beta=beta, k=k), hp.sr, wav)
    #       # write('/content/drive/MyDrive/Demo/model4/weights_0.7gst_0.3emo_{}_epoch294.wav'.format(k), hp.sr, wav)
    #   print("WOOHOO\n")
