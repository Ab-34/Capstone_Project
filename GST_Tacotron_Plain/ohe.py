from Network import Encoder, Decoder
from GST import GST, Emotion_embedding

from Data_test import SpeechDataset, collate_fn, get_eval_data

from Hyperparameters import Hyperparameters as hp
from Loss import TacotronLoss
from pyparsing import java_style_comment
from utils import spectrogram2wav
import torch.nn as nn

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from scipy.io.wavfile import write
from time import time
import matplotlib.pyplot as plt
import os
import sys
# import gc
# import cv2


device = torch.device(hp.device)


def train(log_dir, dataset_size, alpha, start_epoch=0):
    # log directory
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(os.path.join(log_dir, 'state')):
        os.mkdir(os.path.join(log_dir, 'state'))
        os.mkdir(os.path.join(log_dir, 'state','encoder'))
        os.mkdir(os.path.join(log_dir, 'state','decoder'))
        os.mkdir(os.path.join(log_dir, 'state','embedding'))
        os.mkdir(os.path.join(log_dir, 'state','gst'))
        os.mkdir(os.path.join(log_dir, 'state','emo'))
    if not os.path.exists(os.path.join(log_dir, 'wav')):
        os.mkdir(os.path.join(log_dir, 'wav'))
        os.mkdir(os.path.join(log_dir, 'wav','encoder'))
        os.mkdir(os.path.join(log_dir, 'wav','decoder'))
        os.mkdir(os.path.join(log_dir, 'wav','embedding'))
        os.mkdir(os.path.join(log_dir, 'wav','gst'))
        os.mkdir(os.path.join(log_dir, 'wav','emo'))
    if not os.path.exists(os.path.join(log_dir, 'state_opt')):
        os.mkdir(os.path.join(log_dir, 'state_opt'))
        os.mkdir(os.path.join(log_dir, 'state_opt','encoder'))
        os.mkdir(os.path.join(log_dir, 'state_opt','decoder'))
        os.mkdir(os.path.join(log_dir, 'state_opt','embedding'))
        os.mkdir(os.path.join(log_dir, 'state_opt','gst'))
        os.mkdir(os.path.join(log_dir, 'state_opt','emo'))
    if not os.path.exists(os.path.join(log_dir, 'attn')):
        os.mkdir(os.path.join(log_dir, 'attn'))
        os.mkdir(os.path.join(log_dir, 'attn','encoder'))
        os.mkdir(os.path.join(log_dir, 'attn','decoder'))
        os.mkdir(os.path.join(log_dir, 'attn','embedding'))
        os.mkdir(os.path.join(log_dir, 'attn','gst'))
        os.mkdir(os.path.join(log_dir, 'attn','emo'))
    if not os.path.exists(os.path.join(log_dir, 'test_wav')):
        os.mkdir(os.path.join(log_dir, 'test_wav'))
        os.mkdir(os.path.join(log_dir, 'test_wav','encoder'))
        os.mkdir(os.path.join(log_dir, 'test_wav','decoder'))
        os.mkdir(os.path.join(log_dir, 'test_wav','embedding'))
        os.mkdir(os.path.join(log_dir, 'test_wav','gst'))
        os.mkdir(os.path.join(log_dir, 'test_wav','emo'))

    f = open(os.path.join(log_dir, 'log{}.txt'.format(start_epoch)), 'w')

    msg = 'use {}'.format(hp.device)
    print(msg)
    f.write(msg + '\n')

    # load model
    embedding = nn.Embedding(len(hp.vocab), hp.E).to(device)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    gst = GST().to(device)
    emo = Emotion_embedding().to(device)


    if torch.cuda.device_count() > 1:
        embedding = DataParallel(embedding)
        encoder = DataParallel(encoder)
        decoder = DataParallel(decoder)
        gst = DataParallel(gst)
        emo = DataParallel(emo)



    if start_epoch != 0:
        model_path = os.path.join(log_dir, 'state/embedding', 'epoch{}.pt'.format(start_epoch))
        embedding.load_state_dict(torch.load(model_path))

        model_path = os.path.join(log_dir, 'state/encoder', 'epoch{}.pt'.format(start_epoch))
        encoder.load_state_dict(torch.load(model_path))

        model_path = os.path.join(log_dir, 'state/gst', 'epoch{}.pt'.format(start_epoch))
        gst.load_state_dict(torch.load(model_path))

        model_path = os.path.join(log_dir, 'state/decoder', 'epoch{}.pt'.format(start_epoch))
        decoder.load_state_dict(torch.load(model_path))

        model_path = os.path.join(log_dir, 'state/emo', 'epoch{}.pt'.format(start_epoch))
        emo.load_state_dict(torch.load(model_path))

        msg = 'Load model of' + model_path


    else:
        msg = 'New model'
    print(msg)
    f.write(msg + '\n')

    # load optimizer
    optimizer = []
    optimizer.append(optim.Adam(embedding.parameters(), lr=hp.lr))
    optimizer.append(optim.Adam(encoder.parameters(), lr=hp.lr))
    optimizer.append(optim.Adam(gst.parameters(), lr=hp.lr))
    optimizer.append(optim.Adam(decoder.parameters(), lr=hp.lr))
    optimizer.append(optim.Adam(emo.parameters(), lr=hp.lr))


    if start_epoch != 0:
        opt_path = os.path.join(log_dir, 'state_opt/embedding', 'epoch{}.pt'.format(start_epoch))
        optimizer[0].load_state_dict(torch.load(opt_path))
        # encoder
        opt_path = os.path.join(log_dir, 'state_opt/encoder', 'epoch{}.pt'.format(start_epoch))
        optimizer[1].load_state_dict(torch.load(opt_path))
        # gst
        opt_path = os.path.join(log_dir, 'state_opt/gst', 'epoch{}.pt'.format(start_epoch))
        optimizer[2].load_state_dict(torch.load(opt_path))
        # decoder
        opt_path = os.path.join(log_dir, 'state_opt/decoder', 'epoch{}.pt'.format(start_epoch))
        optimizer[3].load_state_dict(torch.load(opt_path))

        opt_path = os.path.join(log_dir, 'state_opt/emo', 'epoch{}.pt'.format(start_epoch))
        optimizer[4].load_state_dict(torch.load(opt_path))

        msg = 'Load optimizer of' + opt_path
    else:
        msg = 'New optimizer'
    print(msg)
    f.write(msg + '\n')

    for i in optimizer:
      for state in i.state.values():
          for k, v in state.items():
              if torch.is_tensor(v):
                  state[k] = v.to(device)

    criterion = TacotronLoss()  # Loss

    train_dataset = torch.load('/content/drive/MyDrive/GST/GST_Tacotron/train_datasets_female/train_dataset_with_emotion_labels.pth')
    train_loader = torch.load('/content/drive/MyDrive/GST/GST_Tacotron/train_datasets_female/train_loader_with_emotion_labels.pth')
    
    num_train_data = len(train_dataset)
    total_step = hp.num_epochs * num_train_data // hp.batch_size
    start_step = start_epoch * num_train_data // hp.batch_size
    step = 0
    global_step = step + start_step
    prev = beg = int(time())
    
    emo_list = ['Amused', 'Angry', 'Disgusted', 'Neutral', 'Sleepy']
    emotion_dict = {
      0:torch.Tensor([1,0,0,0,0]),
      1:torch.Tensor([0,1,0,0,0]),
      2:torch.Tensor([0,0,1,0,0]),
      3:torch.Tensor([0,0,0,1,0]),
      4:torch.Tensor([0,0,0,0,1])
    }
 
    tot_batches = len(train_loader)
    for epoch in range(start_epoch + 1,  hp.num_epochs):
        print("Gathering Emotion Info...")
        
        encoder.train(True)
        gst.train(True)
        embedding.train(True)
        decoder.train(True)
        emo.train(True)
        total_loss = 0;
        # Make dataloader
        for i, batch in enumerate(train_loader):
            step += 1
            global_step += 1

            texts = batch['text'].to(device)
            mels = batch['mel'].to(device)
            mags = batch['mag'].to(device)
            emotion_list = batch['emotion']

            temp = []
            for element in emotion_list:
              temp.append(emotion_dict[int(element)])
            emotion_vector = torch.stack(temp,dim = 0).to(device)
            # print(temp.shape)

            for opt in optimizer:
              opt.zero_grad()

            mels_input = mels[:, :-1, :]  # shift
            mels_input = mels_input[:, :, -hp.n_mels:]  # get last frame
            ref_mels = mels[:, 1:, :]

            # Changed model, running separate models here itself instead of passing through Tacotron
            embedded = embedding(texts)
            memory, encoder_hidden = encoder(embedded)
            style_embed = gst(ref_mels)  # [N, 256]
 
            memory = emo(style_embed, emotion_vector, memory)
            mels_hat, mags_hat, _ = decoder(mels_input, memory)
            

            mel_loss, mag_loss = criterion(mels[:, 1:, :], mels_hat, mags, mags_hat)
            loss = mel_loss + mag_loss
            total_loss = total_loss + loss
            loss.backward()
            

            torch.nn.utils.clip_grad_norm_(embedding.parameters(), 1.,error_if_nonfinite = True)  # clip gradients
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1., error_if_nonfinite = True)  # clip gradients
            torch.nn.utils.clip_grad_norm_(gst.parameters(), 1., error_if_nonfinite  = True)  # clip gradients
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1., error_if_nonfinite  = True)  # clip gradients
            torch.nn.utils.clip_grad_norm_(emo.parameters(), 1., error_if_nonfinite  = True)  # clip gradients



            for opt in optimizer:
              opt.step()
            
            # scheduler.step()

            if global_step in hp.lr_step:
                for opt in optimizer:
                  opt = set_lr(opt, global_step, f)
            

            if (i + 1) % hp.log_per_batch == 0:
                now = int(time())
                use_time = now - prev
                # total_time = hp.num_epoch * (now - beg) * num_train_data // (hp.batch_size * (i + 1) + epoch * num_train_data)
                total_time = total_step * (now - beg) // step
                left_time = total_time - (now - beg)
                left_time_h = left_time // 3600
                left_time_m = left_time // 60 % 60
                msg = 'step: {}/{}, epoch: {}, batch {}, loss: {:.3f}, mel_loss: {:.3f}, mag_loss: {:.3f}, use_time: {}s, left_time: {}h {}m'
                msg = msg.format(global_step, total_step, epoch, i + 1, loss.item(), mel_loss.item(), mag_loss.item(), use_time, left_time_h, left_time_m)

                f.write(msg + '\n')
                # print(msg)

                prev = now


        epoch_wise_loss = total_loss/tot_batches
        print('epoch: {}, train loss: {}'.format(epoch, epoch_wise_loss))
        # save model, optimizer and evaluate
        if epoch % hp.save_per_epoch == 0 and epoch != 0:
            torch.save(embedding.state_dict(), os.path.join(log_dir, 'state/embedding/epoch{}.pt'.format(epoch)))
            torch.save(optimizer[0].state_dict(), os.path.join(log_dir, 'state_opt/embedding/epoch{}.pt'.format(epoch)))

            torch.save(encoder.state_dict(), os.path.join(log_dir, 'state/encoder/epoch{}.pt'.format(epoch)))
            torch.save(optimizer[1].state_dict(), os.path.join(log_dir, 'state_opt/encoder/epoch{}.pt'.format(epoch)))

            torch.save(gst.state_dict(), os.path.join(log_dir, 'state/gst/epoch{}.pt'.format(epoch)))
            torch.save(optimizer[2].state_dict(), os.path.join(log_dir, 'state_opt/gst/epoch{}.pt'.format(epoch)))

            torch.save(decoder.state_dict(), os.path.join(log_dir, 'state/decoder/epoch{}.pt'.format(epoch)))
            torch.save(optimizer[3].state_dict(), os.path.join(log_dir, 'state_opt/decoder/epoch{}.pt'.format(epoch)))
            
            torch.save(emo.state_dict(), os.path.join(log_dir, 'state/emo/epoch{}.pt'.format(epoch)))
            torch.save(optimizer[4].state_dict(), os.path.join(log_dir, 'state_opt/emo/epoch{}.pt'.format(epoch)))
            
            msg = 'save model, optimizer in epoch{}'.format(epoch)
            f.write(msg + '\n')
            print(msg)

            encoder.eval()
            gst.eval()
            embedding.eval()
            decoder.eval()
            emo.eval()

            #for file in os.listdir(hp.ref_wav):
            wavfile = hp.ref_wav
            name, _ = os.path.splitext(hp.ref_wav.split('/')[-1])

            text, mel, ref_mels = get_eval_data(hp.eval_text, wavfile)
            text = text.to(device)
            mel = mel.to(device)
            ref_mels = ref_mels.to(device)

            # Changed model, running separate models here itself instead of passing through Tacotron
            embedded = embedding(text)
            memory, encoder_hidden = encoder(embedded)
            style_embed = gst(ref_mels)  # [N, 256]

            
            memory = emo(style_embed, torch.unsqueeze(emotion_dict[0].to(device), 0), memory)
            mels_hat, mags_hat, attn = decoder(mel, memory)


            mag_hat = mags_hat.squeeze().detach().cpu().numpy()
            attn = attn.squeeze().detach().cpu().numpy()
            plt.imshow(attn.T, cmap='hot', interpolation='nearest')
            plt.xlabel('Decoder Steps')
            plt.ylabel('Encoder Steps')
            fig_path = os.path.join(log_dir, 'attn/epoch{}-{}.png'.format(epoch, name))
            plt.savefig(fig_path, format='png')

            wav = spectrogram2wav(mag_hat)
            write(os.path.join(log_dir, 'wav/epoch{}-{}.wav'.format(epoch, name)), hp.sr, wav)

            msg = 'synthesis eval wav in epoch{} model'.format(epoch)
            print(msg)
            f.write(msg)

    msg = 'Training Finish !!!!'
    f.write(msg + '\n')
    print(msg)

    f.close()


def set_lr(optimizer, step, f):
    if step == 500000:
        msg = 'set lr = 0.0005'
        f.write(msg)
        print(msg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif step == 1000000:
        msg = 'set lr = 0.0003'
        f.write(msg)
        print(msg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    elif step == 2000000:
        msg = 'set lr = 0.0001'
        f.write(msg)
        print(msg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    return optimizer


if __name__ == '__main__':
    
    argv = sys.argv
    log_number = int(argv[1])
    start_epoch = int(argv[3])
    if argv[2].lower() != 'all':
        dataset_size = int(argv[2])
    else:
        dataset_size = None
    train(hp.log_dir.format(log_number), dataset_size, 0.3, start_epoch)



