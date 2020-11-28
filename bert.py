import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# reading the file with complex words
df=pd.read_csv('../Users_Unique_Annotations_Task_1.csv')

# dropping unused columns and rows
df.drop(df.iloc[:, 11:], inplace = True, axis = 1)
df=df.drop('pid',axis=1)

# creating a list of all complex words
a=[]
for i in df.columns:
    for j in df.index:
        if(df[i][j]!='NaN'):
            a.append(df[i][j])

# creating a dictionary with frequency of each complex word
freq={}
for each in a:
    freq[each]=a.count(each)

# reading the sentences file
df=pd.read_csv('../Sentences.csv')

#drop repeating sentences
j=100
for i in range(1,20):
        df.drop(df.index[j,j+9], inplace=True)
        df.reset_index(inplace=True, drop=True)
        j=j+90

df=df.drop(['group_no','category'],axis=1)

# tokenising the sentences while maintaining the sentence number
tokens=[]
sent=[]
for i in df.index:
    word_list=df['sentence'][i].split(' ')
    senten=[i+1]*len(word_list)
    tokens=tokens+word_list
    sent=sent+senten

# removing blank and newline tokens
for i,j in zip(tokens, sent):
    if i == '' or i == '\n':
        tokens.remove(i)
        sent.remove(j)

# storing the tokens in a dataframe
df1=pd.DataFrame({'tokens':tokens, 'sent':sent})

# defining two classes S-simple words and C-complex words using the freq dictionary
df1['class']='S'
for i in df1.index:
    for j in freq:
        if j == df1['tokens'][i] and freq[j]>=2:
            df1['class'][i]='C'
            break

# aggregating the tokens according to sentence number
agg_func = lambda s: [(w, p) for w, p in zip(df1["tokens"].values.tolist(),df1["class"].values.tolist())]
grouped = df1.groupby("sent").apply(agg_func)
sentences = [s for s in grouped]

# storing labels and tokens in two separate lists
labels = [[s[1] for s in sentence] for sentence in sentences]
sentences = [[word[0] for word in sentence] for sentence in sentences]

# converting the class values to idx and add padding
tag_values = list(set(df1["class"].values))
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}

# using bert pretained multilingual model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(str(word))
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(sentences, labels)
]
