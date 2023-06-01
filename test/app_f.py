import streamlit as st
from streamlit_option_menu import option_menu
from inf_gst_only import pred
from nrclex import NRCLex
import numpy as np
import pandas as pd
import librosa
import soundfile
from pydub import AudioSegment
from pydub.silence import split_on_silence
import wave
import subprocess
import os
#from denoise import AudioDeNoise
from moviepy.editor import concatenate_audioclips, AudioFileClip
from scipy.io import wavfile
import noisereduce as nr 
import librosa
import librosa.display

from scipy.io.wavfile import read
import matplotlib.pyplot as plt

from timeit import default_timer as timer

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

def predict(text,emo,speaker,idx):
    print(speaker)
    if text is not None:
      loud = {
          'Angry' : 14.00,
          'Sad' :-2.00,
          "Neutral" : 0.00,
          'Happy' : 8.00,
          'Surprise' : 20.00
      }

      stretched = {
        'Angry' : 1.00, 
        'Sad' : 1.20,
        "Neutral" : 1.00,
        'Happy' : 1.10,
        'Surprise' : 0.90
      }
      emojis = {'Neutral':'😶', 'Happy':'😄', 'Sad':'😔','Surprise':'😲','Angry':'😡'}

      # st.write("")
      # st.write("Just a second ...")
      if emo is not None:
        # To change
        audio_path =  pred(text,emo,speaker,idx)
        print(audio_path)
        original_audio_file = open(audio_path,'rb') 
        audio = original_audio_file.read()
        # st.write("Generated Audio")
        
        # Stretches Audio
        #os.system("ffmpeg -i '{}' 'output_{}.wav' -y".format(audio_path,text))
	
#         print('entering stretch')
#         stretch(audio_path, stretched[emo],text)
#         print('exit')

#         # Sad Speed and Loud tuning
#         audio_file = AudioSegment.from_wav("stretched_{}.wav".format(text)) 
#         audio_file = audio_file + loud[emo]
#         audio_file.export("./audioStretchedLouder_{}.wav".format(text),format="wav")
        # audio_file = open("./audioStretchedLouder_{}.wav".format(text),'rb') 
        # audio = audio_file.read()
        # st.audio(audio, format='audio/ogg')

        # st.download_button(     
        #     label="Download",
        #     data=audio,
        #     file_name="./audio_{}.mp3".format(text),
        #     mime='audio/ogg')

 
  
        #rate, data = wavfile.read("./audioStretchedLouder_{}.wav".format(text))
        #reduced_noise = nr.reduce_noise(y=data, sr=rate)
        # st.write("RATE : ",rate)
        #wavfile.write("mywav_reduced_noise_{}.wav".format(idx), rate, reduced_noise)
        
        # audio_file = open("mywav_reduced_noise_{}.wav".format(idx).format(text),'rb') 
        # audio = audio_file.read()
        # st.audio(audio, format='audio/ogg')
        return audio_path#"audioStretchedLouder_{}.wav".format(text)
	#return audioStretchedLouder_{}.wav".format(text)

      else:
        emo = NRCLex(text)
        dict1 = emo.raw_emotion_scores
        # st.write(dict1)
        emotion = {'Angry':0,'Sad':0,"Neutral":0,"Happy":0,"Surprise":0}
        pointer = {'fear': 'Sad', 'anger': 'Angry', 'anticipation': "Surprise", 'trust': "Neutral", 'surprise': "Surprise", 'positive': "Happy", 'negative': 'Angry', 'sadness': 'Sad', 'disgust': 'Angry', 'joy': "Happy"}

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
          st.session_state.ans[1][idx] = Keymax
          audio_path =  pred(text,Keymax,speaker,idx)
          original_audio_file = open(audio_path,'rb') 
          audio = original_audio_file.read()
          # st.write("Your Generated Audio")
          
          # Stretches Audio
#           os.system("ffmpeg -i '{}' 'output_{}.wav' -y".format(audio_path,text))
#           stretch("output_{}.wav".format(text), stretched[Keymax],text)
        
#           # Speed and Loud tuning
#           audio_file = AudioSegment.from_wav("stretched_{}.wav".format(text)) 
#           audio_file = audio_file + loud[Keymax]
#           audio_file.export("./audioStretchedLouder_{}.wav".format(text),format="wav")
          # audio_file = open("./audioStretchedLouder_{}.mp3".format(text),'rb') 
          # audio = audio_file.read()
          # st.write("Stretched and Louder Tuned Audio")
          
          # st.audio(audio, format='audio/ogg')
          # st.download_button(     
          #   label="Download",
          #   data=audio,
          #   file_name="./audio_{}.mp3".format(text),
          #   mime='audio/ogg')

          #rate, data = wavfile.read("./audioStretchedLouder_{}.wav".format(text))
          #reduced_noise = nr.reduce_noise(y=data, sr=rate)
          #wavfile.write("mywav_reduced_noise_{}.wav".format(idx), rate, reduced_noise)
          
          # audio_file = open("mywav_reduced_noise_{}.wav".format(idx).format(text),'rb') 
          # audio = audio_file.read()
          # st.audio(audio, format='audio/ogg')
          #return "audioStretchedLouder_{}.wav".format(text)
          return audio_path
def plot_wave(y, sr):
    fig, ax = plt.subplots()

    img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)

    return plt.gcf()
def predict_helper(texts_list,emotions_list,speaker_list,speaker = '0011'):
  placeholder = st.empty()
  with placeholder.container():
    audio_paths = []
    i=0
    start_1 = timer()
    # st.session_state.logs.append('Generating audio file')
    # st.write(st.session_state.logs)
    for T,E,S in zip(texts_list,emotions_list,speaker_list):
      audio_paths.append(predict(T,E,S,i))
      i+=1
    end_1 = timer()
    # st.write(audio_paths)
    #logic for concatenating audio
    # st.session_state.logs.append('Denoising audio...')
    # st.write(st.session_state.logs)
    clips = [AudioFileClip(c) for c in audio_paths]
    final_clip = concatenate_audioclips(clips)
    final_clip.write_audiofile("./test_output.wav")
    sound = AudioSegment.from_file("./test_output.wav", format = 'wav') 
    audio_chunks = split_on_silence(sound
                                ,min_silence_len = 1000
                                ,silence_thresh = -80
                                ,keep_silence = 200
                            )
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    st.text("Time taken :\nPrepare audio " + str(round(end_1 - start_1,2)) + " seconds")
    
    # st.session_state.logs.append('Exporting audio...')
    # st.write(st.session_state.logs)
    combined.export("./no_silence.wav", format = "wav")
      
    audio_file = open("./no_silence.wav",'rb') 
    audio = audio_file.read()
    st.write("Final audio output:")
    st.audio(audio, format='audio/ogg')
    st.download_button(     
              label="Download",
              data=audio,
              file_name="Final_Audio.mp3",
              mime='audio/ogg')
              
    btn = st.button("Clear Generated Audio")
  
  if btn:
    placeholder.empty()

# MAIN CODE
selected = option_menu(
            menu_title=None,  # required
            options=["Instructions","App"],  # required
            icons=["book","house"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa", "width": "600px"},
                "icon": {"color": "blue", "font-size": "16px"},
                "nav-link": {
                    "font-size": "16px",
                    "color": "black",
                    "text-align": "center",
                    "margin": "0px",
                    # "--hover-color": "#0e1117",
                },
                "nav-link-selected": {"background-color": "green", "color":"white","width":"300px"},
            }
        )

# tab1, tab2= st.tabs(["Instructions","App"])
if selected == "Instructions":
  st.subheader("1. Choose a text from the given options OR Enter your custom text")
  st.subheader("2. Choose an emotion from the given options OR let us detect the emotion")
  st.subheader("3. Generate Audio and Download your file")

if selected == "App":
  ans = [[],[],[],""]

  if 'ans' not in st.session_state:
    st.session_state.ans = [[],[],[],""]
  if 'flag' not in st.session_state:
    st.session_state.flag = 0
  if 'OUTPUT' not in st.session_state:
    st.session_state.OUTPUT = ""

  # def send_text(t):
  #   ans[0].append(t)
  def send_speaker(speaker):
    st.session_state.ans[2].append(speaker)
    
  def send_emo(emo):
    if emotion == 'Detect Emotion':
      st.session_state.ans[1].append(None)
    else:
      st.session_state.ans[1].append(emo[2:])

  def read_text_emo():
    output = ""
    i = 1
    for TEXT,EMOTION,SPEAKER in zip(st.session_state.ans[0],st.session_state.ans[1],st.session_state.ans[2]):
      if EMOTION is not None:
        output+= str(i) + " " + TEXT + " : " + EMOTION +  " : "+ SPEAKER + "\n"
        i+=1
      else:
        output+= str(i) + " " + TEXT + " : "+ SPEAKER +"\n"
        i+=1
    return st.session_state.ans[0],st.session_state.ans[1],st.session_state.ans[2],output


  col1, col2, col3 = st.columns([1, 2, 1])

  with col1:
    texts = ("The sun is so bright today . ", "You should be happy with the result ", "How are you feeling ?", "How did they fall from the seat ? ")
    choice = st.radio("Choose a sentence : ",texts)
    text_choice = st.button("Choose Text")
    if text_choice:
        if len(st.session_state.ans[0])!=len(st.session_state.ans[1]):
          st.session_state.ans[1].append(None)
        if len(st.session_state.ans[0])!=len(st.session_state.ans[2]):
          st.session_state.ans[2].append('Male')
        
        st.session_state.ans[0].append(choice)
        st.session_state.flag = 1
    st.subheader("OR")
    col1_form = st.form(key = "col1_form") 
    text_box = col1_form.text_input("Enter text:","")
    submit_col1 = col1_form.form_submit_button(label = "Submit")
    if submit_col1:
      if len(st.session_state.ans[0])!=len(st.session_state.ans[1]):
        st.session_state.ans[1].append(None)
      if len(st.session_state.ans[0])!=len(st.session_state.ans[2]):
          st.session_state.ans[2].append('Male')
      st.session_state.ans[0].append(text_box)
      st.session_state.flag = 1


    
    
  emojis = {'Neutral':'😶', 'Happy':'😄', 'Sad':'😔','Surprise':'😲','Angry':'😡'}
  speakers = {
    'Male' : 'Male',
    'Female' : 'Female'
  }
  with col3: 
    # SPEAKER SELECTION
    speaker_choice = st.radio(
      "Which speaker would you like to choose?",
      ('Male','Female'))
    s_choice = st.button("Choose Speaker")
    if s_choice and st.session_state.flag:
      #st.write(speakers[speaker_choice])
      send_speaker(speakers[speaker_choice])
      st.session_state.flag = 1
    #send_speaker(speakers[speaker_choice])
    # EMOTION SELECTION
    emotion = st.radio(
      "Which emotion?",
      ('😶 Neutral', '😄 Happy', '😔 Sad','😲 Surprise','😡 Angry', 'Detect Emotion'))
    emo_choice = st.button("Choose Emotion")
    if emo_choice and st.session_state.flag:
      send_emo(emotion)
      st.session_state.flag = 0 


    

  with col2:
      TEXTS,EMOTIONS,SPEAKERS,st.session_state.OUTPUT = read_text_emo()   
      if len(st.session_state.ans[0]):
          if st.button("Clear the last Text, Emotion pair"):
            st.session_state.ans[1].pop()
            st.session_state.ans[0].pop()
            st.session_state.ans[2].pop()
            st.session_state.OUTPUT = ""
            # if len(st.session_state.ans[0]) == 0:
            #   placeholder_col2.empty()
          st.write("Chosen Text(s)")
          st.write(st.session_state.ans[0])
          st.write("Chosen Emotion(s)")
          st.write(st.session_state.ans[1])
          st.write("Chosen Speaker(s)")
          st.write(st.session_state.ans[2])
            
          st.text(st.session_state.OUTPUT)
          if len(st.session_state.ans[0]) == len(st.session_state.ans[1]) and  len(st.session_state.ans[0]) == len(st.session_state.ans[2]) and len(st.session_state.ans[0])!=0:
            submit = col2.button("Generate Audio")
            if submit:
              print(TEXTS)
              print(EMOTIONS)
              print(SPEAKERS)
              predict_helper(TEXTS,EMOTIONS,SPEAKERS,st.session_state.ans[3])



