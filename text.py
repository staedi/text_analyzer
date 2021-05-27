import nltk
import pandas as pd
import streamlit as st
from string import punctuation
import random
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
data = pd.read_csv('headlines.csv')

st.header('Sentence analyses')
st.write('Enter or paste a text to analyze below')

# sentence = st.text_input(' ')
stop_tokens = nltk.corpus.stopwords.words('english')
# stop_tokens.extend([punc for punc in punctuation])
# stop_tokens = set(stop_tokens)

st.sidebar.header('Choose options below')
sel_dataset = st.sidebar.radio('Which Data to use?',['Manual','Predefined'])
sel_simple = st.sidebar.radio('Remove stopwords?',['Yes','No'])
if sel_dataset == 'Predefined':
    st.sidebar.markdown('Choose a ticker to analze (e.g., VZ)')
    sel_ticker = st.sidebar.selectbox('Ticker',['None']+data['symbol'].unique().tolist())
    sel_number = st.sidebar.slider('Number',1,10,5)
sampled_data = []

if sel_dataset == 'Manual':
    sentence = st.text_input(' ')
    sampled_data = [sentence]
elif sel_ticker != 'None':
    if len(data.loc[data['symbol']==sel_ticker])>sel_number:
        sampled_data = data.loc[data['symbol']==sel_ticker,'headline'].sample(sel_number)
    else:
        sampled_data = data.loc[data['symbol']==sel_ticker,'headline'][:sel_number]

# st.markdown(stop_tokens)

if len(sampled_data)>0:
    st.write(sampled_data)
    for sentence in sampled_data:
        # sentence = data.loc[data['symbol']==sel_ticker,'headline']
        st.write(sentence)
        tokenized = nltk.word_tokenize(sentence)

        if sel_simple == 'Yes':
            tokenized = [token for token in tokenized if token.lower() not in stop_tokens]
            for token_idx in range(len(tokenized)):
                for punc in punctuation:
                    if tokenized[token_idx] == "n't":
                        tokenized[token_idx] = 'not'
                    elif tokenized[token_idx][-2:] == "'s":
                        tokenized[token_idx] = tokenized[token_idx][:-2]
                    elif punc != '.':
                        # if punc == "'":
                        #     tokenized[token_idx] = tokenized[token_idx].replace(punc,'')
                        tokenized[token_idx] = tokenized[token_idx].replace(punc,' ').strip()

        tokenized = [token for token in tokenized if token]
        # tokenized = [token for token in tokenized for stop in stop_tokens if stop not in token]
        st.write(tokenized)
        tagged = nltk.pos_tag(tokenized)
        st.write(pd.DataFrame(tagged))
        st.text(tagged)
        entity = nltk.ne_chunk(tagged)
        st.text(entity)
