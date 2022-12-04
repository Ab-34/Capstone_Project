import streamlit as st
from inf import pred
from nrclex import NRCLex
import numpy as np
import pandas as pd
import librosa
import soundfile
from pydub import AudioSegment
import wave
import subprocess
import os
from denoise import AudioDeNoise 



st.title("Deep Learning based Speech Synthesis with Emotion Overlay 🔉")

def stretch( fname,  factor ):
  infile=wave.open( fname, 'rb')
  rate= infile.getframerate()
  channels=infile.getnchannels()
  swidth=infile.getsampwidth()
  nframes= infile.getnframes()
  audio_signal= infile.readframes(nframes)
  outfile = wave.open('stretched_{}.wav'.format(text), 'wb')
  outfile.setnchannels(channels)
  outfile.setsampwidth(swidth)
  outfile.setframerate(rate/factor)
  outfile.writeframes(audio_signal)
  outfile.close()
  

  # audio_file = open("stretched_{}.wav".format(text),'rb') 
  # audio = audio_file.read()
  # st.write("Stretched Audio")
  # st.audio(audio, format='audio/ogg')
  return;

def predict(text,emo):

    if text is not None:
      loud = {
          'Angry' : 8.00,
          'Sad' : -6.00,
          "Neutral" : 0.00,
          'Happy' : 8.00,
          'Surprise' : 10.00
      }

      stretched = {
        'Angry' : 1.10, # might change
        'Sad' : 1.20,
        "Neutral" : 1.00,
        'Happy' : 1.10,
        'Surprise' : 0.85
      }

      st.write("")
      st.write("Just a second ...")
      if emo is not None:
        audio_path =  pred(text,emo)
        original_audio_file = open(audio_path,'rb') 
        audio = original_audio_file.read()
        st.write("Generated Audio")

        # st.audio(audio, format='audio/ogg')
        # Shifts Pitch
        # y, sr = librosa.load(audio_path, sr=16000) # y is a numpy array of the wav file, sr = sample rate
        # new_y = librosa.effects.pitch_shift(y, sr, steps)
        # soundfile.write("./pitchShifted_{}.wav".format(text), new_y, sr)
        # audio_file = open("./pitchShifted_{}.wav".format(text),'rb') 
        # audio = audio_file.read()
        # st.write("Pitch Shifted Audio")
        # st.audio(audio, format='audio/ogg')
        
        # Stretches Audio
        print(os.system("ls"))
        os.system("ffmpeg -i '{}' 'output_{}.wav'".format(audio_path,text))
        stretch("output_{}.wav".format(text), stretched[emo])
        
        
        
        # Audio louder
        # song = AudioSegment.from_wav(audio_path)
        # song = song + loud[emotion] # Song increased by 20 dB 
        # song.export("./audioLouder_{}.mp3".format(text),format="mp3")
        # audio_file = open("./audioLouder_{}.mp3".format(text),'rb') 
        # audio = audio_file.read()
        # st.write("Louder Audio")
        # st.audio(audio, format='audio/ogg')

        # Sad Speed and Loud tuning
        audio_file = AudioSegment.from_wav("stretched_{}.wav".format(text)) 
        audio_file = audio_file + loud[emo]
        audio_file.export("./audioStretchedLouder_{}.wav".format(text),format="wav")
        audio_file = open("./audioStretchedLouder_{}.wav".format(text),'rb') 
        audio = audio_file.read()
        st.write("Stretched and Louder Tuned Audio")
        st.audio(audio, format='audio/ogg')

        # audioDenoiser = AudioDeNoise(inputFile="./audioStretchedLouder_{}.wav".format(text))
        # audioDenoiser.deNoise(outputFile="input_denoised.wav")
        # audio_file = open("input_denoised.wav",'rb') 
        # audio = audio_file.read()
        # st.write("Denoised Stretched and Louder Tuned Audio")
        # st.audio(audio, format='audio/ogg')

        st.download_button(     
            label="Download",
            data=audio,
            file_name="./audio_{}.mp3".format(text),
            mime='audio/ogg')


      else:
        emo = NRCLex(text)
        dict1 = emo.raw_emotion_scores
        # st.write(dict1)
        emotion = {'Anger':0,'Sad':0,"Neutral":0,"Happy":0,"Surprise":0}
        pointer = {'fear': 'Sad', 'anger': 'Anger', 'anticipation': "Surprise", 'trust': "Neutral", 'surprise': "Surprise", 'positive': "Happy", 'negative': 'Anger', 'sadness': 'Sad', 'disgust': 'Anger', 'joy': "Happy"}

        for i in dict1:
          emotion[pointer[i]]+=dict1[i]

        lst = []
        for i in emotion:
          lst.append(emotion[i])
        
        Keymax=None
        if lst.count(lst[0])!=len(lst):
          Keymax = max(zip(emotion.values(), emotion.keys()))[1]
        else:
          Keymax = "Neutral"
        if Keymax is not None:
          st.write("Emotion Detected : " + Keymax)
          audio_path =  pred(text,Keymax)
          original_audio_file = open(audio_path,'rb') 
          audio = original_audio_file.read()
          st.write("Generated Audio")

          # st.audio(audio, format='audio/ogg')
          # Shifts Pitch
          # y, sr = librosa.load(audio_path, sr=16000) # y is a numpy array of the wav file, sr = sample rate
          # new_y = librosa.effects.pitch_shift(y, sr, steps)
          # soundfile.write("./pitchShifted_{}.wav".format(text), new_y, sr)
          # audio_file = open("./pitchShifted_{}.wav".format(text),'rb') 
          # audio = audio_file.read()
          # st.write("Pitch Shifted Audio")
          # st.audio(audio, format='audio/ogg')
          
          # Stretches Audio
          os.system("ffmpeg -i '{}' 'output_{}.wav'".format(audio_path,text))
          stretch("output_{}.wav".format(text), stretched[Keymax])
          

          # Audio louder
          # song = AudioSegment.from_wav(audio_path)
          # song = song + loud[Keymax] # Song increased by 20 dB 
          # song.export("./audioLouder_{}.mp3".format(text),format="mp3")
          # audio_file = open("./audioLouder_{}.mp3".format(text),'rb') 
          # audio = audio_file.read()
          # st.write("Louder Audio")
          # st.write(loud)
          # st.audio(audio, format='audio/ogg')
          
          # Speed and Loud tuning
          audio_file = AudioSegment.from_wav("stretched_{}.wav".format(text)) 
          audio_file = audio_file + loud[Keymax]
          audio_file.export("./audioStretchedLouder_{}.mp3".format(text),format="mp3")
          audio_file = open("./audioStretchedLouder_{}.mp3".format(text),'rb') 
          audio = audio_file.read()
          # st.write("Stretched and Louder Tuned Audio")
          
          st.audio(audio, format='audio/ogg')
          st.download_button(     
            label="Download",
            data=audio,
            file_name="./audio_{}.mp3".format(text),
            mime='audio/ogg')



col1, col2, col3 = st.columns([1, 2, 1])
# data = np.random.randn(10, 1)

with col2: 
  my_form = st.form(key = "form1") 
  text = my_form.text_input("Enter text:","")
  emotion = my_form.selectbox('Enter Emotion:',(None,'Neutral', 'Angry', 'Sad', 'Happy', 'Surprise'))
  # steps = my_form.number_input("Enter number of semitones to shift audio file: ",0,10,0, 1)
  # stretched = my_form.number_input("How much to stretch( >1.0 : Stretch || <1.0 : Shrink)",step=1.,format="%.2f",value=1.0)
  # loud = my_form.number_input("How much louder would you like the audio",step=1.,format="%.2f")
  submit = my_form.form_submit_button(label = "Generate Audio")

with col1:
  st.write("sentences : ")

  texts = ["The sun is so bright today", "You should be happy with the result", "How are you always late ?", "How did they fall from the seat ?"]
  choice = [0,0,0,0]
  for i,t in enumerate(texts):
    choice[i] = st.checkbox(t)
    if choice[i]:
        col2.text = t

with col3:
  st.write("COL 3")


if submit:
    predict(text,emotion)
     
