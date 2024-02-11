import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
# from win32com.client import Dispatch
from tensorflow.keras.models import load_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np
import pandas as pd
import ktrain
import streamlit.components.v1 as components 

predictor=ktrain.load_predictor("distilbert")



def depression(sentence):
# function to print sentiments
# of the sentence.
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)
 #positive
    if sentiment_dict['compound'] >= 0.05 :
        return 0
 #negative
    elif sentiment_dict['compound'] <= - 0.05 :
        return 1
 #nutral
    else :
        return np.nan

import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

import emoji
nltk.download('words')
words = set(nltk.corpus.words.words())
def clean(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)#Remove http links
    tweet=decontracted(tweet)
    tweet = " ".join(tweet.split())
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI) #Remove Emojis
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    tweet= "".join([w for w in tweet if w not in string.punctuation])
    tweet=re.sub(r'[^a-zA-Z ]+',"",tweet)
    tweet=tweet.lower()
    return tweet

df=pd.read_csv("cleaned_control+dep.csv")

# def speak(text):
#     speak=Dispatch(("SAPI.SpVoice"))
#     speak.Speak(text)

model =  load_model('model.h5')
tokinizer=pickle.load(open('tokinizer.pkl','rb'))
txt="michel is  a good person"
maxlen=30

def main():
    st.title("Depression Detection Application")
    st.write("Build with Streamlit & Python")
    activites=["Classification","About"]
    choices=st.sidebar.selectbox("Select Activities",activites)
    if choices=="Classification":
        st.subheader("Classification")
        msg=st.text_input("Enter a text")
        if st.button("Process"):
            print(msg)
            print(type(msg))
            d=depression(msg)
            x=clean(msg)
            x=[x]
            x=tokinizer.texts_to_sequences(x)
            x=pad_sequences(x,maxlen=maxlen)
            if (model.predict(x)>0.5).astype(int)[0][0]==1 and d==0:
                st.success("You don't have depression")
            elif (model.predict(x)>0.5).astype(int)[0][0]==0 and d==1:
                st.error("You have depression")
            elif (model.predict(x)>0.5).astype(int)[0][0]==1:
                st.error("You have depression")
            else:
                st.success("You don't have depression")
        op=["no depression",'depression']
        g=st.sidebar.select_slider("Select which graph you want to see most common words for",options=op)
        if g=="no depression":
            st.subheader(f"most common words for {g}" )
            x=' '.join(df[df['dep']==0].tweet_cleaned.to_list())
            stopwords=STOPWORDS
            wc=WordCloud(background_color='white',stopwords=stopwords,height=1000,width=1500)
            wc.generate(x)
            fig, ax = plt.subplots()
            plt.imshow(wc,interpolation='bilinear')
            plt.axis("off")
            st.pyplot(fig)
        else:
            st.subheader(f"most common words for {g}" )
            x=' '.join(df[df['dep']==1].tweet_cleaned.to_list())
            stopwords=STOPWORDS
            wc=WordCloud(background_color='white',stopwords=stopwords,height=1000,width=1500)
            wc.generate(x)
            fig, ax = plt.subplots()
            plt.imshow(wc,interpolation='bilinear')
            plt.axis("off")
            st.pyplot(fig)

    else:
        st.markdown('''
            ## ** this project is about classifing user text to: ** 
          #### ** 1)you have Depression **
          #### ** 2)you don't have Depression ** \n\n
        -------------------------------------------------
         It is based on 84000 tweets where almost half of them were scarped  
         from the timeline of twitter users that explicitly stated that they
         were diagnosed with depression and the other half are from the timeline
         of random users. I have used **word2vec** and **LSTM** to create the predictions
         alongside **polarity score**.
         ''')
main()