from utils import *
from Data_test import get_eval_data, SpeechDataset, collate_fn
from Hyperparameters import Hyperparameters as hp
import torch
from scipy.io.wavfile import write
from Network import *
import torch.nn as nn
from GST_test import GST, Emotion_embedding


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
    #             emo_lala = gst(ref_mels).to('cpu')
    #             temp.append(emo_lala)

    # emotion_vector = torch.mean(torch.cat(temp, dim=0),0)
    # torch.save(emotion_vector,'/content/drive/MyDrive/GST/GST-Tacotron/embeddings/meansbeforelayers/{emotion}_mean.pth'.format(emotion = emotion))
    

    print(emotion,"load our saved emo embedding")
    emotion_vector = torch.load('/content/drive/MyDrive/GST/GST-Tacotron/embeddings/meansbeforelayers/{emotion}_mean.pth'.format(emotion = emotion))

    return emotion_vector


def synthesis(model_enc,model_gst,model_emb,model_dec,model_emo, eval_text, emotion,input_text):
    eval_text = _pinyin(eval_text)

    model_enc.eval()
    model_gst.eval()
    model_emb.eval()
    model_dec.eval()
    model_emo.eval()

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

    for ref_wav, speaker in zip(ref_wavs, speakers):
        text, GO, ref_mels = get_eval_data(eval_text, ref_wav)
        text = text.to(device)
        GO = GO.to(device)
        ref_mels = ref_mels.to(device)
        emotion_vector = get_emotion_dict(emotion, model_gst)
        emotion_vector = emotion_vector.to(device)
        embedded = model_emb(text)  # [N, T_x, E]
        memory, encoder_hidden = model_enc(embedded)  # [N, T_x, E]
        style_embed = model_gst(ref_mels)  # [N, 256]
        memory = model_emo(style_embed, emotion_vector, memory)

        #style_embed = style_embed.expand_as(memory)
        #emotion_vector = emotion_vector.expand_as(memory)
        #memory = memory + alpha*style_embed + beta*emotion_vector
        mel_hat, mag_hat, attn = model_dec(GO,memory)
        mag_hat = mag_hat.squeeze().detach().cpu().numpy()
        attn = attn.squeeze().detach().cpu().numpy()

        wav_hat = spectrogram2wav(mag_hat)
        wavs[speaker] = wav_hat

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

def load_model5(checkpoint_path):
  model = Emotion_embedding().to(device)
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
    model_enc = load_model1('/content/drive/MyDrive/GST/GST-Tacotron/log/train11/state/encoder/epoch198.pt')
    model_gst = load_model2('/content/drive/MyDrive/GST/GST-Tacotron/log/train11/state/gst/epoch198.pt')
    model_emb = load_model3('/content/drive/MyDrive/GST/GST-Tacotron/log/train11/state/embedding/epoch198.pt')
    model_dec = load_model4('/content/drive/MyDrive/GST/GST-Tacotron/log/train11/state/decoder/epoch198.pt')
    model_emo = load_model5('/content/drive/MyDrive/GST/GST-Tacotron/log/train11/state/emo/epoch198.pt')
    emo_list = ["Neutral","Angry","Sad","Happy","Surprise"]
    

    ''' Input weights for style and emotion separately each time '''
    while(True):
      text = input("Enter text to generate audio : ")
      emo = int(input("0: Neutral\n1: Angry\n2: Sad\n3: Happy\n4: Surprise\nChoose emotion by entering the number : "))
     # alpha = float(input("Enter weight for style vector : "))
     # beta = float(input("Enter weight for emotion vector : "))

      emotion = emo_list[emo]
      wavs = synthesis(model_enc, model_gst,model_emb,model_dec,model_emo, text, emotion, text)
      for k in wavs:
          wav = wavs[k]
          #write('/content/drive/MyDrive/GST/GST-Tacotron/Demo/weighted/test/weights_{alpha}gst_{beta}emo_{k}_epoch198_layeredemo.wav'.format(alpha = alpha, beta=beta, k=k), hp.sr, wav)
          write('/content/drive/MyDrive/GST/GST-Tacotron/Demo/weighted/test/{k}_epoch198_layeredemo.wav'.format( k=k), hp.sr, wav)
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
