#!/usr/bin/env python
# coding: utf-8

from feel_it import EmotionClassifier, SentimentClassifier
import pandas as pd
import streamlit as st
import numpy as np
import umap
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re
import spacy
from matplotlib_inline.config import InlineBackend
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.express as px
import requests
import networkx as nx
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.io as pio
from collections import Counter

# WIDE MODE
st.set_page_config(layout="wide")

col_7, col_8 = st.columns([1, 6])

url_immagine = 'https://github.com/MarcelloGalimberti/Sentiment/blob/main/Ducati_red_logo.png?raw=true'

with col_7:
    st.image(url_immagine, width=150)

with col_8:
    st.title("Natural Language Processing per l'analisi del Sentiment")
    st.header('Sito demo: DUCATIMULTISTRADA.it')
st.header('Post dal forum: Nuova Multistrada Rally V4')
st.write('https://multiforum.freeforumzone.com/x/d/11813143/Nuova-Multistrada-rally-V4/discussione.aspx/')
st.header('Web scraping results', divider='red')

# ### Importa stop words da personalizzare ### GITHUB

url_file = 'https://github.com/MarcelloGalimberti/Sentiment/blob/main/Stopwords_2.xlsx?raw=True'
sw = pd.read_excel(url_file)
stop_words_list = sw['Stopwords'].values.tolist()


@st.cache_data()
def crea_database_post(max_pagine):
    lista_link = []
    for i in range(1, max_pagine):
        url = f'https://multiforum.freeforumzone.com/x/d/11813143/Nuova-Multistrada-rally-V4/discussione.aspx/{i}'
        lista_link.append(url)
    lista_post = []
    for link in lista_link:
        pagina = requests.get(link)
        soup = BeautifulSoup(pagina.content, 'html.parser')
        post = soup.find_all(class_=re.compile('^cdivpost cfontmess cfontmess*'))
        for i in range(len(post)):
            messaggio = post[i].text.strip().replace('\n', '').replace('\r', '')
            lista_post.append(messaggio)
    return lista_post


df_raw = pd.DataFrame(crea_database_post(10), columns=['Post'])

st.dataframe(df_raw, use_container_width=True)
st.subheader(f'{len(df_raw)} post trovati nel forum')

# ### Sentiment and emotion analysis

emotion_classifier = EmotionClassifier()
sentiment_classifier = SentimentClassifier()

df_raw['Predicted_sentiment'] = ""
df_raw['Predicted_emotion'] = ""


# funzione per calcolo sentiment
@st.cache_resource()
def calcola_sentiment(df):
    for i in range(len(df)):
        emozione_predicted = emotion_classifier.predict([df.loc[i, 'Post']])
        df.loc[i, 'Predicted_emotion'] = emozione_predicted[0]
        sentiment_predicted = sentiment_classifier.predict([df.loc[i, 'Post']])
        df.loc[i, 'Predicted_sentiment'] = sentiment_predicted[0]
    return df


df_raw = calcola_sentiment(df_raw)

st.header('Classificazione dei post: sentiment ed emotion', divider='red')
st.subheader(
    'Sentiment (positive | negative) ed emotion (joy | anger | sadness | fear) sono predetti da AI con affidabilità stimata :red[85%]')
st.dataframe(df_raw, use_container_width=True)

# Statistiche sentiment ed emotion | Tabella di contingenza
tabella_contingenza = pd.pivot_table(df_raw, index='Predicted_sentiment',
                                     columns='Predicted_emotion',
                                     aggfunc='count', margins=True, margins_name='Total',
                                     fill_value=0)

tabella_contingenza.columns = tabella_contingenza.columns.droplevel(level=0)

df_chart = pd.DataFrame(df_raw.groupby(['Predicted_sentiment', 'Predicted_emotion']).count())
df_chart.reset_index(inplace=True)

st.subheader('Post: sentiment vs emotion', divider='grey')
fig_tab = px.bar(df_chart, x='Post', y='Predicted_sentiment', color='Predicted_emotion',
                 color_discrete_sequence=["#A30F15", "#FB7858", "#E93529", "#FFEFE8"])
st.plotly_chart(fig_tab, use_container_width=True)


# ### Natural Language Processing | SpaCy

# messa funzione e decorazione
@st.cache_data()
def carica_lingua(lingua):
    nlp = spacy.load(lingua)  # vs lg
    # spacy_stopwords = spacy.lang.it.stop_words.STOP_WORDS # indagare qui
    return nlp


lingua = 'it_core_news_md' # messa small ora medium https://github.com/explosion/spacy-models/releases/download/it_core_news_md-3.7.0/it_core_news_md-3.7.0-py3-none-any.whl

nlp = carica_lingua(lingua)
spacy_stopwords = spacy.lang.it.stop_words.STOP_WORDS  # indagare qui

# Crea stopword set personalizzato
stopwords_set = set(stop_words_list)
stopwords_set.update(["moto", "ducati", 'rally', 're', 'cè', '😂', '😅', 'post', 'autore', 'utente', 'i', 'o', "all'",
                      "sull'",'viene'])  # valore di default per session state

if 'stopwords_set' not in st.session_state:
    st.session_state['stopwords_set'] = stopwords_set

df_nlp = df_raw.copy()

# #### Iterazione NLP per post puliti
df_nlp['Post_Lemmi'] = ''


# funzione che fa cleansing dei post e restituisce lemmi, elimina: punteggiatura, verbi, ausiliari, stopwords, oov
# @st.cache_resource()
def clean_doc(doc):
    lista_token_puliti = []
    for token in doc:
        if (not token.is_stop
                and not token.is_punct
                and not token.pos_ == 'VERB'
                and not token.pos_ == 'AUX'
                and not token.is_oov
                and token.is_ascii
                and not token.is_digit
                and len(token) > 1):
            lista_token_puliti.append(token.lemma_.lower().strip())
    return lista_token_puliti


# funzione per creare lemmi
@st.cache_resource()
def crea_post_lemmi(df, my_stopwords_set):
    for i in range(len(df)):
        frase = df.loc[i, 'Post']
        doc = nlp(frase)
        complete_filtered_tokens = clean_doc(doc)
        Post_S = ' '.join([item for item in complete_filtered_tokens])
        # fa ulteriore cleansing con Stop Words personalizzate
        filtered_words = []
        for word in Post_S.split():
            if word not in my_stopwords_set:
                filtered_words.append(word)
            stringa_filtered = ' '.join([item for item in filtered_words])
        df.loc[i, 'Post_Lemmi'] = stringa_filtered
    return df


df_nlp = crea_post_lemmi(df_nlp, st.session_state.stopwords_set)

st.subheader('Analisi del contenuto dei post', divider='grey')

text = " ".join(review for review in df_nlp.Post_Lemmi)
st.write("Ci sono {} lemmi nella combinazione dei post.".format(len(text)))
st.write('_Lemma: forma di base da cui deriva un intero sistema flessionale nominale, aggettivale e verbale._')

col1, col2 = st.columns([1, 2])

# frequenza

stringa = text
arr = Counter(stringa.split()).most_common(20)
df_frequenze = pd.DataFrame(data=arr,
                            columns=['word', 'frequency'])

with col1:
    st.subheader('Lemmi più frequenti', divider='red')
    fig_bar = px.bar(df_frequenze.sort_values('frequency'), x='frequency', y='word', orientation='h',
                     color='frequency', color_continuous_scale=px.colors.sequential.Reds)
    st.plotly_chart(fig_bar, use_container_width=True)

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
InlineBackend.figure_formats = {'svg'}


# funzione wordcloud
def wordcloud_chart(text, stopwords_set):
    wordcloud = WordCloud(stopwords=stopwords_set, background_color='Black',
                          colormap='Reds', random_state=42, width=800, height=400).generate(text)
    # svg = wordcloud.to_image()
    svg = wordcloud.to_svg()
    plt.figsize = (10, 6)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    return svg


fig_a = wordcloud_chart(text, st.session_state.stopwords_set)

with col2:
    st.subheader('WordCloud dei lemmi', divider='red')
    st.image(fig_a)


# funzioni per callback
def aggiorna_stopwords_set():
    new_stopwords_set = set(new_stopwords)
    st.session_state.stopwords_set.update(new_stopwords_set)
    # aggiungere stopwords al vocabolario spacy


def reset_stopwords():
    for key in st.session_state.keys():
        del st.session_state[key]


# Selezione stopwords personalizzate: lenta, non capisco cosa fa rigirare
new_stopwords = st.multiselect('**Seleziona le tue stopword**', df_frequenze['word'],
                               placeholder="Scegliere lemmi da escludere dall'analisi")

pulsante_reset = st.button('Click per eliminare stopword', on_click=aggiorna_stopwords_set)  #:
if st.button('Click per ripristinare le stopwords', on_click=reset_stopwords):
    st.write('Stopwords resettate')

#########  Insights

st.header('Analisi del sentiment', divider='red')

col3, col4 = st.columns([1, 1])

with col3:
    st.subheader('Sentiment positivo | Lemmi')
    df_nlp_pos = df_nlp[df_nlp['Predicted_sentiment'] == 'positive']
    text_positive = " ".join(review for review in df_nlp_pos.Post_Lemmi)
    fig_pos = wordcloud_chart(text_positive, st.session_state.stopwords_set)
    st.image(fig_pos)  # valutare donut
    arr_pos = Counter(text_positive.split()).most_common(10)
    df_frequenze_pos = pd.DataFrame(data=arr_pos,
                                    columns=['word', 'frequency'])
    st.write('Top 10 lemmi associati a sentiment positivo', df_frequenze_pos)

with col4:
    st.subheader('Sentiment negativo | Lemmi')
    df_nlp_neg = df_nlp[df_nlp['Predicted_sentiment'] == 'negative']
    text_negative = " ".join(review for review in df_nlp_neg.Post_Lemmi)
    fig_neg = wordcloud_chart(text_negative, st.session_state.stopwords_set)
    st.image(fig_neg)
    arr_neg = Counter(text_negative.split()).most_common(10)
    df_frequenze_neg = pd.DataFrame(data=arr_neg,
                                    columns=['word', 'frequency'])
    st.write('Top 10 lemmi associati a sentiment negativo', df_frequenze_neg)


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


def lemmi_doc(doc):
    lista_lemmi = []
    for token in doc:
        if (not token.is_space
                and token.is_ascii):
            lista_lemmi.append(token.lemma_.lower().strip())
    stringa = ' '.join([item for item in lista_lemmi])
    doc_lemmato = nlp(stringa)
    return doc_lemmato


def lemmi_doc_strong(doc):
    lista_lemmi = []
    for token in doc:
        if (not token.is_space
                and token.is_ascii
                and not token.is_stop
                and not token.is_punct
                and not token.is_digit
                and len(token) > 1
                and not token.pos_ == 'VERB'
                and not token.pos_ == 'AUX'
                and not token.is_oov
        ):
            lista_lemmi.append(token.lemma_.lower().strip())
    stringa = ' '.join([item for item in lista_lemmi])
    doc_lemmato = nlp(stringa)
    return doc_lemmato


def lemmi_doc_medium(doc):  # togliere ADP e DET
    lista_lemmi = []
    for token in doc:
        if (not token.is_space
                and token.is_ascii
                # and not token.is_stop
                and not token.is_punct
                # and not token.is_digit
                and len(token) > 1
                and not token.pos_ == 'VERB'
                and not token.pos_ == 'ADP'
                and not token.pos_ == 'DET'
                # and not token.pos_ == 'AUX'
                and not token.is_oov
        ):
            lista_lemmi.append(token.lemma_.lower().strip())
    stringa = ' '.join([item for item in lista_lemmi])
    doc_lemmato = nlp(stringa)
    return doc_lemmato


def posizione_lemma(doc, lemma_chiave):
    lista_indici = []
    for token in doc:
        if token.text.strip() == lemma_chiave:
            lista_indici.append(token.i)
    return lista_indici


# Algo nuovo per sunburst chart

pio.renderers.default = 'notebook'
df_sb = pd.DataFrame(columns=['Chiave', 'Info','Frequenza'])
for i in range (len(df_frequenze_sentiment)):  # top 10 positive o negative
    lemma_chiave = df_frequenze_sentiment.loc[i,'word']
    frequenza = df_frequenze_sentiment.loc[i,'frequency']
    df_subset = df_nlp_sentiment.loc[df_nlp_sentiment['Post_Lemmi'].str.contains(lemma_chiave)].reset_index(drop=True) # dataframe che contiene gli esempi con lemma chiave, equivalente a df_corpus vecchio

    # mettere clausula di uscita se non trova lemma chiave
    for j in range (len (df_subset)):
        doc = nlp(df_subset.loc[j,'Post']) # prende il post originale che contiene il lemma chiave
        lista_sent = []
        for sent in doc.sents: # frazione il post in frasi
            if lemma_chiave in sent.text:
                lista_sent.append(sent) # mette in lista_sent tutte le frasi che contengomo il lemma chiave
        for item in lista_sent: # un item è una frase
            lista_indici =[]
            for k in range (len (item)): # per gli elementi (parole) della frase
                if item[k].text.strip() == lemma_chiave: # se la parola corrisponde al lemma chiave ne cattura l'indice, che sembra essere globale per la frase (item)
                    lista_indici.append(k) 
            for indice in lista_indici:
                token = item[indice] # token del lemma chiave presente nella frase
                lista_children = list(token.children) # poi è da pulire      crea lista dei children
                # qui fare cleaning di lista_children
                lista_info = []
                for token in lista_children:
                    if (not token.is_space
                     and token.is_ascii
                    #and not token.is_stop
                    and not token.is_punct
                    #and not token.is_digit
                    and len(token)>1
                    and not token.pos_ == 'VERB'
                    and not token.pos_ == 'ADP'
                    and not token.pos_ == 'DET'
                    #and not token.pos_ == 'AUX'
                    and not token.is_oov
                     ):
            
                        lista_info.append(token.lemma_.lower().strip())
                
                info = " ".join(child for child in lista_info) # concatena i children in info
                dati = {'Chiave':lemma_chiave,'Info':info,'Frequenza':frequenza}
                df_dati = pd.DataFrame([dati])
                df_sb = pd.concat([df_sb,df_dati])

df_sb.drop_duplicates(inplace=True)
df_sb = df_sb[df_sb['Info'] != '']


fig_sb = px.sunburst(df_sb, path=['Chiave', 'Info'],
                    color_continuous_scale="reds", values='Frequenza', color='Frequenza',width=850, height=850)

st.plotly_chart(fig_sb)


##### Importazione file da Open AI ##############################

st.header('Analisi OpenAI', divider='red')
# st.header('Clustering', divider='grey')

url_AI = 'https://github.com/MarcelloGalimberti/Sentiment/blob/main/df_raw_GPT.xlsx?raw=true'
df_raw_GPT = pd.read_excel(url_AI)


# Processo di GPT


def processa_GPT (GPT_info):
    GPT_info = GPT_info.replace('-','')
    pattern = r'[\n]'
    lista_GPT_info = list(re.split(pattern,GPT_info))
    lista_GPT_info = [x.strip(' ') for x in lista_GPT_info]
    lista_GPT_info = [x for x in lista_GPT_info if x != '']
    return lista_GPT_info


def estrae_df_GPT (df):
    df_sentiment = pd.DataFrame(columns=['Numero_Post','Post','GPT','GPT_info'])
    for i in range (len(df)):
        GPT = df.loc[i,'GPT']
        lista_GPT = processa_GPT(GPT)
        for j in range (len (lista_GPT)):
            bullet = lista_GPT[j]
            dati = {'Numero_Post': i, 'GPT_info':bullet,'Post':df.loc[i,'Post'],'GPT':df.loc[i,'GPT']}
            df_dati = pd.DataFrame([dati])
            df_sentiment = pd.concat([df_sentiment,df_dati])
    df_sentiment.reset_index(inplace=True, drop=True)
    return df_sentiment


df_GPT_info = estrae_df_GPT(df_raw_GPT)
df_GPT_info =  df_GPT_info.drop_duplicates(subset='GPT_info', keep="first")
df_GPT_info.reset_index(drop = True, inplace=True)
st.write(f'Dai post si astraggono {len(df_GPT_info)} commenti')
st.dataframe(df_GPT_info[['Numero_Post','GPT_info']],width=1500)  # fino a qui ha senso

# eliminazione altre stopwords

lista_mie_stopword = ['post','autore','utente','i','o',"all'","sull'",'moto','ducati','rally','v4s','v4','ciò','multi','oswald67'] # deve far parte di preprocessing dopo EDA 
sw1 = list(spacy_stopwords)
sw = sw1 + lista_mie_stopword


# Embedding
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=4, max_df=0.9, ngram_range=(1,2),
                        stop_words=sw)
X = vectorizer.fit_transform(df_GPT_info['GPT_info'])

# Calcolo similarità
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(X) 
sim_array = np.array(similarity_matrix)
np.fill_diagonal(sim_array, 0)

# Grafo
H = nx.from_numpy_array(sim_array)

soglia = st.slider('Soglia', min_value=0.0,max_value=1.0,value=0.8)
if not soglia:
    st.stop()


# Creare funzione per renderlo dinamico
edges_to_kill = []
min_wt = soglia     # this is our cutoff value for a minimum edge-weight 0.8
for n, nbrs in H.adj.items():
    for nbr, eattr in nbrs.items():
        data = eattr['weight']
        if data < min_wt: 
            edges_to_kill.append((n, nbr))             
st.write("\n", len(edges_to_kill) / 2, "edges to kill (of", H.number_of_edges(), "), before de-duplicating")

for u, v in edges_to_kill:
    if H.has_edge(u, v):   # catches (e.g.) those edges where we've removed them using reverse ... (v, u)
        H.remove_edge(u, v)

strong_H = H
st.write('Strong H edges', strong_H.number_of_edges())

strong_H.remove_nodes_from(list(nx.isolates(strong_H)))

from math import sqrt
count = strong_H.number_of_nodes()
equilibrium = 10 / sqrt(count)    # default for this is 1/sqrt(n), but this will 'blow out' the layout for better visibility
pos = nx.fruchterman_reingold_layout(strong_H, k=2, iterations=500)


#plt.rcParams['figure.figsize'] = [16, 12]  # a better aspect ratio for labelled nodes
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
nx.draw(strong_H, pos, font_size=3, node_size=40, edge_color='white', with_labels=False,
        node_color="#C00000", width=0.3, alpha=0.8,connectionstyle='arc3,rad=0.2', arrows=True)
#for p in pos:  # raise positions of the labels, relative to the nodes
#    pos[p][1] -= 0.03
nx.draw_networkx_labels(strong_H, pos, font_size=4, font_color='white')
ax.set_facecolor('black')
fig.patch.set_alpha(0.0)
#fig.set_facecolor('black')
st.pyplot(fig)

st.write('CC = ',nx.number_connected_components(strong_H))

cc = nx.connected_components(strong_H)
lista_cc=[]
for c in cc:
    lista_cc.append(c)

lista_cc_2 = []
for item in lista_cc:
    lista_cc_2.append(list(item))

df_cc = pd.DataFrame(columns = ['cluster_cc','nodo'])
for i in range (len(lista_cc_2)):
    for j in range (len(lista_cc_2[i])):
        df_cc.loc[len(df_cc)] = [i,lista_cc_2[i][j]]
df_cc_info = df_cc.merge(df_GPT_info, how = 'left', left_on='nodo', right_index= True)


df_cluster_info = df_cc_info[['cluster_cc','GPT_info']]
df_sampled = df_cluster_info.groupby("cluster_cc").sample(n=1, random_state=1)
lista_commenti = df_sampled['GPT_info'].tolist()
#st.write('Lista commenti', lista_commenti)



#### scatter cc
mapper_X = umap.UMAP(n_components=2, random_state=1, metric='cosine').fit(X) 
#xycoord_umap = mapper_X.transform(X)

df_scatter_cc = pd.DataFrame(columns=['X','Y','GPT_info'])
df_scatter_cc.X = mapper_X.embedding_[:, 0]
df_scatter_cc.Y = mapper_X.embedding_[:, 1]
df_scatter_cc.GPT_info = df_GPT_info.GPT_info

df_scatter_cc_cluster = df_scatter_cc.merge(df_cc, how = 'left', left_index=True, right_on='nodo')
df_scatter_cc_cluster.reset_index(drop = True, inplace=True)
df_scatter_cc_cluster.fillna(-1, inplace=True)

for i in range (len(df_scatter_cc_cluster)):
    if df_scatter_cc_cluster.loc[i,'cluster_cc']==-1:
        df_scatter_cc_cluster.loc[i,'size']=0.5
    else:
        df_scatter_cc_cluster.loc[i,'size']=len(nx.node_connected_component(strong_H,i))+1


fig_scatter_cc = px.scatter(df_scatter_cc_cluster, color='cluster_cc', x='X',y='Y', template='plotly_dark', color_continuous_scale='Reds',
                         opacity=0.8, width=1000, height=600, hover_data='GPT_info',
                         size = 'size')#, color_continuous_scale='reds')
#fig_scatter_cc.update_traces(marker=dict(size=5),
#                          selector=dict(mode='markers'))
fig_scatter_cc.update_xaxes(visible=False)
fig_scatter_cc.update_yaxes(visible=False)
st.plotly_chart(fig_scatter_cc)


#### Treemap
lista_top_words=[]
for i in range (len (df_cc_info)):
    cluster = df_cc_info.loc[i,'cluster_cc']
    sub_df = df_cc_info[df_cc_info['cluster_cc']==cluster]
    lista_parole = []
    for info in sub_df.GPT_info:
        parole = info.split()
        for parola in parole:
            if parola.lower() in sw:
                pass
            else:
                lista_parole.append(parola)
        word_counter = Counter(lista_parole)
        most_occur = word_counter.most_common(2) # numero di top words
        lista_key_words=[]
        for tuple in most_occur:
            key_word = tuple[0].lower()
            lista_key_words.append(key_word)
    lista_top_words.append(lista_key_words)

df_cc_info['Top_words']=lista_top_words
df_cc_info['Top_words']=df_cc_info['Top_words'].apply(lambda x: ' '.join(x))
df_cc_info['cluster_cc_str']=df_cc_info['cluster_cc'].apply(lambda x: str(x))

df_cc_info['cluster_top_words']=df_cc_info['cluster_cc_str']+'  ('+df_cc_info['Top_words']+')'
df_cc_info['Numero_Post']=df_cc_info['Numero_Post'].astype(int)

fig_tm_cc = px.treemap(df_cc_info, path=[px.Constant('Cluster Treemap'),'cluster_top_words','GPT_info','Post'], color='Numero_Post',
                   color_continuous_scale='reds', template='plotly_dark', width=1500, height=800, hover_data='GPT_info')
fig_tm_cc.update_traces(root_color="lightgrey")
fig_tm_cc.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig_tm_cc.update_traces(marker=dict(cornerradius=5))
st.plotly_chart(fig_tm_cc, use_container_width=True)

### Report 

st.header ('Report finale NLP', divider='red')

st.write('Numero di post esaminati: ', len(df_raw))
st.write('Summary sentiment / emotion')
st.dataframe(tabella_contingenza)
st.write('Top 3 lemmi positivi: ',df_frequenze_pos[0:3])
st.write('Top 3 lemmi negativi: '), df_frequenze_neg[0:3]

st.header ('Report finale GPT', divider='red')
st.write('Commenti estratti dai post: ', len(df_GPT_info))



