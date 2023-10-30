#!/usr/bin/env python
# coding: utf-8

from feel_it import EmotionClassifier, SentimentClassifier
import pandas as pd
import streamlit as st
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re
import string
import spacy
from spacy import displacy
# import spacy_streamlit
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.express as px
import requests
import io
#import cairocffi as cairo


pd.set_option('display.max_colwidth', None)

col_7, col_8 = st.columns([1,5])

url_immagine='https://github.com/MarcelloGalimberti/Sentiment/blob/main/Ducati_red_logo.png?raw=true'

with col_7: 
    st.image(url_immagine, width=150)

with col_8:
    
    st.title('Sentiment analysis')
    st.header('Sito: DUCATIMULTISTRADA.it')
st.header('Post dal forum: Nuova Multistrada Rally V4')
st.write('https://multiforum.freeforumzone.com/x/d/11813143/Nuova-Multistrada-rally-V4/discussione.aspx/')
st.header('',divider='red')

# ### Importa stop words da personalizzare ### GITHUB

url_file = 'https://github.com/MarcelloGalimberti/Sentiment/blob/main/Stopwords_2.xlsx?raw=True'
sw = pd.read_excel(url_file)
stop_words_list=sw['Stopwords'].values.tolist()

#stop_words_file = open('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/Scraping/stopwords-it.txt','r')
#stop_words = stop_words_file.read()
#stop_words_list = stop_words.split('\n')
#stop_words_file.close()

@st.cache_data()
def crea_database_post (max_pagine):
    lista_link=[]
    for i in range (1,max_pagine):
        url = f'https://multiforum.freeforumzone.com/x/d/11813143/Nuova-Multistrada-rally-V4/discussione.aspx/{i}'
        lista_link.append(url)
    lista_post=[]
    for link in lista_link:
        pagina = requests.get(link)
        soup = BeautifulSoup(pagina.content, 'html.parser')
        post = soup.find_all(class_=re.compile('^cdivpost cfontmess cfontmess*'))
        for i in range(len(post)):
            messaggio = post[i].text.strip().replace('\n','').replace('\r','')
            lista_post.append(messaggio)
    return lista_post

df_raw = pd.DataFrame(crea_database_post(10),columns=['Post'])

st.dataframe(df_raw, use_container_width=True)
st.subheader (f'{len(df_raw)} post analizzati')

# ### Sentiment and emotion analysis

emotion_classifier = EmotionClassifier()
sentiment_classifier = SentimentClassifier()

df_raw['Predicted_sentiment']=""
df_raw['Predicted_emotion']=""

# funzione per calcolo sentiment
@st.cache_resource()
def calcola_sentiment(df):
    for i in range(len(df)):
        #post = df_raw.loc[i,'Post']
        emozione_predicted = emotion_classifier.predict([df.loc[i,'Post']])
        df.loc[i,'Predicted_emotion'] = emozione_predicted[0]
        sentiment_predicted = sentiment_classifier.predict([df.loc[i,'Post']])
        df.loc[i,'Predicted_sentiment'] = sentiment_predicted[0]
    return df

df_raw = calcola_sentiment(df_raw)

st.header('Sentiment | Emotion predicted ',divider='red')
st.subheader('Sentiment (positive | negative) ed emotion (joy | anger | sadness | fear) sono predetti da AI con affidabilità stimata :red[85%]')
st.dataframe(df_raw,use_container_width=True)

# Statistiche sentiment ed emotion | Tabella di contingenza
tabella_contingenza = pd.pivot_table(df_raw, index = 'Predicted_sentiment',
                                    columns = 'Predicted_emotion',
                                    aggfunc='count', margins=True, margins_name='Total',
                                    fill_value=0)

tabella_contingenza.columns = tabella_contingenza.columns.droplevel(level  =0)

df_chart = pd.DataFrame(df_raw.groupby(['Predicted_sentiment', 'Predicted_emotion']).count())
df_chart.reset_index(inplace=True)

st.subheader('Post: sentiment vs emotion', divider='grey')
fig_tab = px.bar(df_chart, x = 'Post', y = 'Predicted_sentiment', color = 'Predicted_emotion',
                 color_discrete_sequence=["#A30F15", "#FB7858", "#E93529", "#FFEFE8"])
st.plotly_chart(fig_tab, use_container_width=True)

# ### Natural Language Processing | SpaCy
nlp = spacy.load('it_core_news_lg')
spacy_stopwords = spacy.lang.it.stop_words.STOP_WORDS

# Crea stopword set personalizzato
stopwords_set = set(stop_words_list)
stopwords_set.update(["moto", "ducati",'rally','re','cè','😂','😅']) # valore di default per session state

if 'stopwords_set' not in st.session_state:
    st.session_state['stopwords_set'] = stopwords_set

df_nlp = df_raw.copy()

# #### Iterazione NLP per post puliti
df_nlp['Post_Lemmi']=''

# funzione che fa cleansing dei post e restituisce lemmi, elimina: punteggiatura, verbi, ausiliari, stopwords, oov
#@st.cache_resource()
def clean_doc (doc):
    lista_token_puliti = []
    for token in doc:
        if (not token.is_stop 
            and not token.is_punct 
            and not token.pos_ == 'VERB' 
            and not token.pos_ == 'AUX' 
            and not token.is_oov 
            and token.is_ascii
            and not token.is_digit
            and len(token)>1):
            lista_token_puliti.append(token.lemma_.lower().strip())
    return lista_token_puliti

# funzione per creare lemmi
@st.cache_resource()
def crea_post_lemmi (df,my_stopwords_set):
    for i in range (len(df)):
        frase = df.loc[i,'Post']
        doc = nlp(frase)
        complete_filtered_tokens = clean_doc(doc)
        Post_S = ' '.join([item for item in complete_filtered_tokens])
    # fa ulteriore cleansing con Stop Words personalizzate
        filtered_words = []
        for word in Post_S.split():
            if word not in my_stopwords_set:
                filtered_words.append(word)
            stringa_filtered = ' '.join([item for item in filtered_words])        
        df.loc[i,'Post_Lemmi']=stringa_filtered
    return df

df_nlp = crea_post_lemmi(df_nlp,st.session_state.stopwords_set)

st.header('Analisi post', divider='grey')

# st.write(df_nlp) #forse non vale la pena visualizzare - esplicitare LEMMI con spiegone

text = " ".join(review for review in df_nlp.Post_Lemmi)
st.write("Ci sono {} lemmi nella combinazione dei post.".format(len(text)))
st.write('_Lemma: forma di base da cui deriva un intero sistema flessionale nominale, aggettivale e verbale._')

col1, col2 = st.columns([1,2])

# frequenza
from collections import Counter
str = text
arr = Counter(str.split()).most_common(20)
df_frequenze = pd.DataFrame(data = arr,   
                  columns = ['word','frequency']) 

with col1:
    st.subheader('Lemmi più frequenti', divider = 'red')
    fig_bar = px.bar(df_frequenze.sort_values('frequency'), x= 'frequency', y = 'word', orientation = 'h',
            color='frequency',color_continuous_scale=px.colors.sequential.Reds)
    st.plotly_chart(fig_bar, use_container_width=True)

# funzione wordcloud
def wordcloud_chart (text,stopwords_set):
    wordcloud = WordCloud(stopwords=stopwords_set,background_color="white",
                     colormap='Reds').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.figsize=(12,6)
    #plt.title('WordCloud complessivo')
    plt.axis("off")
    return plt

fig_a=wordcloud_chart (text,st.session_state.stopwords_set)

with col2:
    st.subheader('WordCloud dei lemmi', divider = 'red')
    st.pyplot(fig_a)


# funzioni per callback
def aggiorna_stopwords_set():
    new_stopwords_set = set(new_stopwords)
    st.session_state.stopwords_set.update(new_stopwords_set)

def reset_stopwords():
    for key in st.session_state.keys():
        del st.session_state[key]


# Selezione stopwords personalizzate: lenta, non capisco cosa fa rigirare
new_stopwords = st.multiselect('**Seleziona le tue stopword**',df_frequenze['word'],placeholder="Scegliere lemmi da escludere dall'analisi")

pulsante_reset = st.button('Click per eliminare stopword', on_click=aggiorna_stopwords_set)#:
if st.button('Click per ripristinare le stopwords', on_click=reset_stopwords):
    st.write('Stopwords resettate')

#########  Insights

st.header('Analisi del sentiment', divider='red')

col3, col4 = st.columns([1,1])

with col3:
    st.subheader('Sentiment positivo | Lemmi')
    df_nlp_pos = df_nlp[df_nlp['Predicted_sentiment']=='positive']
    text_positive = " ".join(review for review in df_nlp_pos.Post_Lemmi)
    fig_pos = wordcloud_chart (text_positive,st.session_state.stopwords_set)
    st.pyplot(fig_pos)
    arr_pos = Counter(text_positive.split()).most_common(10)
    df_frequenze_pos = pd.DataFrame(data = arr_pos,   
                  columns = ['word','frequency'])
    st.write('Top 10 lemmi positivi', df_frequenze_pos)


with col4:
    st.subheader('Sentiment negativo | Lemmi')
    df_nlp_neg = df_nlp[df_nlp['Predicted_sentiment']=='negative']
    text_negative = " ".join(review for review in df_nlp_neg.Post_Lemmi)
    fig_neg = wordcloud_chart (text_negative,st.session_state.stopwords_set)
    st.pyplot(fig_neg)
    arr_neg = Counter(text_negative.split()).most_common(10)
    df_frequenze_neg = pd.DataFrame(data = arr_neg,   
                  columns = ['word','frequency'])
    st.write('Top 10 lemmi negativi', df_frequenze_neg)

###### Analisi frase

st.header('Relazioni lemmi | sentiment', divider='red')

tipo_sentiment = st.radio('Sceglere se anlizzare sentiment positivo o negativo', ['Positive', 'Negative'])
if not tipo_sentiment:
    st.stop()

if tipo_sentiment == 'Positive':
    df_frequenze_sentiment = df_frequenze_pos
    df_nlp_sentiment = df_nlp_pos
else:
    df_frequenze_sentiment = df_frequenze_neg
    df_nlp_sentiment = df_nlp_neg

lemma_chiave = st.selectbox('Scegli il lemma da analizzare', df_frequenze_sentiment)
if not lemma_chiave:
    st.stop()

#st.write('Il lemma scelto è: ', lemma_chiave) ### aggiungere pulsante per creare grafo

st.subheader(f'Relazione tra lemmi dei post che contengono :red[**{lemma_chiave}**]', divider='gray')

def lemmi_doc (doc):
    lista_lemmi = []
    for token in doc:
        if (not token.is_space
            and token.is_ascii):
            lista_lemmi.append(token.lemma_.lower().strip())
    stringa = ' '.join([item for item in lista_lemmi])
    doc_lemmato = nlp(stringa)
    return doc_lemmato  

def posizione_lemma (doc, lemma_chiave):
    lista_indici = []
    for token in doc:
        if token.text.strip() == lemma_chiave:
            lista_indici.append(token.i)
    return lista_indici

def crea_df_sent (df,lemma_chiave):
    lista_sent = []
    boolean = df['Post_Lemmi'].str.contains(lemma_chiave)
    df = df[boolean].reset_index(drop=True)
    for i in range (len (df)):
        doc = nlp(df.loc[i,'Post'])
        doc = lemmi_doc(doc)
        lista_posizioni = posizione_lemma(doc, lemma_chiave)
        for posizione in lista_posizioni:
            frase = doc[posizione].sent.text
            lista_sent.append(frase)
    df_sent = pd.DataFrame(lista_sent, columns=['Sentence'])
    return df_sent

df_sentence = crea_df_sent(df_nlp_sentiment,lemma_chiave).drop_duplicates()
#st.write(f'Frasi lemmizzate con chiave **{lemma_chiave}**')
#st.dataframe(df_sentence, use_container_width=True)

##### Textnets

import textnets as tn
tn.params.update({'lang':'it', "autodownload": True})
tn.params["seed"] = 42

df_corpus = df_sentence
df_corpus['Label']=df_corpus.index

corpus = tn.Corpus.from_df(df_corpus, doc_col="Sentence", lang="it")

t = tn.Textnet(corpus.tokenized(), min_docs=2)

t.plot(label_nodes=True,          # circular ok, kamada_kawai ok, 
       show_clusters=True,
       node_opacity=0.5,
       scale_nodes_by="birank",
       scale_edges_by="weight",
       #kamada_kawai_layout= True,
       color_clusters=True,
       target = 'plot.png')

st.image('plot.png',width=800)


###### displaCy

frase_render = st.selectbox('Scegli la frase da renderizzare', df_sentence)
if not frase_render:
    st.stop()

doc_dep = nlp(frase_render)

dep_svg = displacy.render(doc_dep, style='dep')#, jupyter=False)

st.image(dep_svg)#, width=400, use_column_width='never')

