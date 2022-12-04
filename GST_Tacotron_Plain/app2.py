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


st. set_page_config(page_title="Audio Generator",layout="wide")
st.title("Deep Learning based Speech Synthesis with Emotion Overlay 🔉")


def stretch( fname,  factor,text ):
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
          'Surprise' : 20.00
      }

      stretched = {
        'Angry' : 1.10, 
        'Sad' : 1.20,
        "Neutral" : 1.00,
        'Happy' : 1.10,
        'Surprise' : 0.90
      }
      emojis = {'Neutral':'😶', 'Happy':'😄', 'Sad':'😔','Surprise':'😲','Angry':'😡'}

      # st.write("")
      # st.write("Just a second ...")
      if emo is not None:
        audio_path =  pred(text,emo)
        original_audio_file = open(audio_path,'rb') 
        audio = original_audio_file.read()
        st.write("Generated Audio")
        
        # Stretches Audio
        os.system("ffmpeg -i '{}' 'output_{}.wav'".format(audio_path,text))
        stretch("output_{}.wav".format(text), stretched[emo],text)

        # Sad Speed and Loud tuning
        audio_file = AudioSegment.from_wav("stretched_{}.wav".format(text)) 
        audio_file = audio_file + loud[emo]
        audio_file.export("./audioStretchedLouder_{}.wav".format(text),format="wav")
        audio_file = open("./audioStretchedLouder_{}.wav".format(text),'rb') 
        audio = audio_file.read()
        st.audio(audio, format='audio/ogg')

        st.download_button(     
            label="Download",
            data=audio,
            file_name="./audio_{}.mp3".format(text),
            mime='audio/ogg')

        from scipy.io import wavfile
        import noisereduce as nr
  
        rate, data = wavfile.read("./audioStretchedLouder_{}.wav".format(text))
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
        wavfile.write("mywav_reduced_noise.wav", rate, reduced_noise)
        
        audio_file = open("mywav_reduced_noise.wav".format(text),'rb') 
        audio = audio_file.read()
        st.audio(audio, format='audio/ogg')

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
          st.write("Emotion Detected : " + Keymax,emojis[Keymax])
          audio_path =  pred(text,Keymax)
          original_audio_file = open(audio_path,'rb') 
          audio = original_audio_file.read()
          st.write("Your Generated Audio")
          
          # Stretches Audio
          os.system("ffmpeg -i '{}' 'output_{}.wav'".format(audio_path,text))
          stretch("output_{}.wav".format(text), stretched[Keymax],text)
        
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

          from scipy.io import wavfile
          import noisereduce as nr
    
          rate, data = wavfile.read("./audioStretchedLouder_{}.wav".format(text))
          reduced_noise = nr.reduce_noise(y=data, sr=rate)
          wavfile.write("mywav_reduced_noise.wav", rate, reduced_noise)
          
          audio_file = open("mywav_reduced_noise.wav".format(text),'rb') 
          audio = audio_file.read()
          st.audio(audio, format='audio/ogg')

          
ans = ["",""]
def send_text(t):
  ans[0].append(t)
def send_emo(emo):
  if emotion == 'Detect Emotion':
    ans[1] = None
  else:
    ans[1] = emo[:-2]

def read_text_emo():
  return ans[0],ans[1]


col1, col2, col3 = st.columns([1, 2, 1])

with col1:
  col1_form = st.form(key = "col1_form") 
  text_box = col1_form.text_input("Enter text:","")
  submit_col1 = col1_form.form_submit_button(label = "Submit")
  ans[0] = text_box

  if submit_col1:
    col2.text("Your chosen text is : " + text_box)
  
  st.write("Choose a sentence or enter custom text : ")

  texts = ["The sun is so bright today", "You should be happy with the result", "How are you always late ?", "How did they fall from the seat ?"]
  choice = [0,0,0,0]
  for i,ele in enumerate(texts):
    choice[i] = st.checkbox(ele)
    if choice[i]:
        text = ele
        ans[0] = text
        
  
   
emojis = {'Neutral':'😶', 'Happy':'😄', 'Sad':'😔','Surprise':'😲','Angry':'😡'}

with col3: 
  emotion = st.radio(
    "Which emotion?",
    ('Neutral 😶', 'Happy 😄', 'Sad 😔','Surprise 😲','Angry 😡', 'Detect Emotion'))
  if emotion :
    send_emo(emotion)        

with col2:
    TEXT,EMOTION = read_text_emo()
    if EMOTION is not None:
      col2.text("Your chosen text is : " + TEXT)
      col2.text("Your emotion is : " + EMOTION+emojis[EMOTION])
      submit = col2.button("Generate Audio")
      if submit:
        predict(TEXT,EMOTION)
    else:
      col2.text("Your chosen text is : " + TEXT)
      submit = col2.button("Detect emotion and generate Audio")
      if submit:
        predict(TEXT,EMOTION)

    






     
