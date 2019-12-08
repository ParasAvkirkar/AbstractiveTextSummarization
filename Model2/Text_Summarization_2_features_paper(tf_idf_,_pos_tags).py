#!/usr/bin/env python
# coding: utf-8

# # **Model 2 seq2seq for text Summurization using seq2seq lib**

# ### Intro
# This is a modification to https://github.com/dongjun-Lee/text-summarization-tensorflow 
# I am builging it in a notebook envronment to be able to easily integrate with colab
# 

# ## Helpers (Googel Drive , Utilits)

# ### Utilits

# https://github.com/dongjun-Lee/text-summarization-tensorflow/blob/master/utils.py

# In[7]:


get_ipython().system('pip install gensim')
get_ipython().system('pip install wget')
  
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# In[1]:


#https://stackoverflow.com/questions/38088652/pandas-convert-categories-to-numbers
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

#ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

def get_pos_tags_dict(words):
    #sent = nltk.word_tokenize(sent)
    #print(sent)
    post_tags_for_words = nltk.pos_tag(words)

    pos_list ={}
    #sent = preprocess(ex)
    for word,pos in post_tags_for_words:
        pos_list[word] = pos
    #print(pos_list)

    import pandas as pd
    df = pd.DataFrame(list(pos_list.items()))
    df.columns = ['word', 'pos']
    df.pos = pd.Categorical(df.pos)
    df['code'] = df.pos.cat.codes
    #print(df)

    pos_list ={}
    for index, row in df.iterrows():
        pos_list[row['word']] = row['code']
#     print(pos_list)
    return pos_list , post_tags_for_words


# In[2]:


from nltk.tokenize import word_tokenize
import re
import collections
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

# default_path = "drive/My Drive/NLP/project/rnn-lstm/data/"
default_path = "/home/sbu_nlp2019/AbstractiveTextSummarization/Model2/dataset/"

train_article_path = default_path + "sumdata/train/train.article.txt"
train_title_path   = default_path + "sumdata/train/train.title.txt"
valid_article_path = default_path + "sumdata/train/valid.article.filter.txt"
valid_title_path   = default_path + "sumdata/train/valid.title.filter.txt"

#valid_article_path = default_path + "sumdata/DUC2003/input.txt"
#valid_title_path   = default_path + "sumdata/DUC2003/task1_ref0.txt"

def clean_str(sentence):
    sentence = re.sub("[#.]+", "#", sentence)
    return sentence


def get_text_list(data_path, toy):
    with open (data_path, "r", encoding="utf-8") as f:
        if not toy:
            return [clean_str(x.strip()) for x in f.readlines()][:200000]
        else:
            return [clean_str(x.strip()) for x in f.readlines()][:50]
        
def build_dict(step, toy=False):
    if step == "train":
        train_article_list = get_text_list(train_article_path, toy)
        train_title_list = get_text_list(train_title_path, toy)

        words = list()
        for sentence in train_article_list + train_title_list:
            for word in word_tokenize(sentence):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open(default_path + "word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    elif step == "valid":
        with open(default_path + "word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

    article_max_len = 50
    summary_max_len = 15

    return word_dict, reversed_dict, article_max_len, summary_max_len


def build_dataset(step, word_dict, article_max_len, summary_max_len, toy=False):
    if step == "train":
        article_list = get_text_list(train_article_path, toy)
        title_list = get_text_list(train_title_path, toy)
    elif step == "valid":
        article_list = get_text_list(valid_article_path, toy)
    else:
        raise NotImplementedError

    x = [word_tokenize(d) for d in article_list]
    x = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in x]
    x = [d[:article_max_len] for d in x]
    x = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in x]
    
    if step == "valid":
        return x
    else:        
        y = [word_tokenize(d) for d in title_list]
        y = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in y]
        y = [d[:(summary_max_len - 1)] for d in y]
        return x, y


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


def get_init_embedding(word_dict , reversed_dict, embedding_size):
    print("Loading Lists...")
    train_article_list = get_text_list(train_article_path, False)
    train_title_list = get_text_list(train_title_path, False)

    print("Loading TF-IDF...")
    tf_idf_list = tf_idf_generate(train_article_list+train_title_list)
    
    print("Loading Pos Tags...")
    pos_list , postags_for_named_entity = get_pos_tags_dict(word_dict.keys())

    #print("Loading Named Entity...")
    #named_entity_recs = named_entity(postags_for_named_entity) 
    
    print("Loading Glove vectors...")

    with open( default_path + "glove/model_glove_300.pkl", 'rb') as handle:
        word_vectors = pickle.load(handle)     
    
    used_words = 0
    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
            if word in tf_idf_list:
                v= tf_idf_list[word]
                rich_feature_array = np.array([v,v,v,v,v,v,v,v,v,v])
                word_vec = np.append(word_vec, rich_feature_array)
            else:
                v=0
                rich_feature_array = np.array([v,v,v,v,v,v,v,v,v,v])
                word_vec = np.append(word_vec, rich_feature_array)

            if word in pos_list:
                v=pos_list[word]
                rich_feature_array_2 = np.array([v,v,v,v,v,v,v,v,v,v])
                word_vec = np.append(word_vec, rich_feature_array_2)
            else:
                v=0
                rich_feature_array_2 = np.array([v,v,v,v,v,v,v,v,v,v])
                word_vec = np.append(word_vec, rich_feature_array_2) 

            #if word in named_entity_recs:
            #  v=named_entity_recs[word]
            #  rich_feature_array_3 = np.array([v,v,v,v,v,v,v,v,v,v])
            #  word_vec = np.append(word_vec, rich_feature_array_3)
            #else:
            #  v=0
            #  rich_feature_array_3 = np.array([v,v,v,v,v,v,v,v,v,v])
            #  word_vec = np.append(word_vec, rich_feature_array_3)  
          
            used_words += 1
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32) #to generate for <padding> and <unk>
        
        
        word_vec_list.append(np.array(word_vec))

    print("words found in glove percentage = " + str((used_words/len(word_vec_list))*100) )
          
    # Assign random vector to <s>, </s> token
    word_vec_list[2] = np.random.normal(0, 1, embedding_size)
    word_vec_list[3] = np.random.normal(0, 1, embedding_size)

    return np.array(word_vec_list)


# In[3]:


# _____TF-IDF libraries_____
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# _____helper Libraries_____
import pickle  # would be used for saving temp files
import csv     # used for accessing the dataset
import timeit  # to measure time of training
import random  # used to get a random number


def tf_idf_generate(sentences):
    #https://stackoverflow.com/questions/30976120/find-the-tf-idf-score-of-specific-words-in-documents-using-sklearn

    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    # our corpus
    data = sentences

    cv = CountVectorizer()

    # convert text data into term-frequency matrix
    data = cv.fit_transform(data)

    tfidf_transformer = TfidfTransformer()

    # convert term-frequency matrix into tf-idf
    tfidf_matrix = tfidf_transformer.fit_transform(data)

    # create dictionary to find a tfidf word each word
    word2tfidf = dict(zip(cv.get_feature_names(), tfidf_transformer.idf_))

    #i = 0
    #for word, score in word2tfidf.items():
    #    print(word, score)
    #    if (i == 10):
    #      break
    #    i+=1  
  
    return word2tfidf


# In[ ]:





# In[4]:


# https://nlpforhackers.io/named-entity-extraction/
from nltk import word_tokenize, pos_tag, ne_chunk


# sentence = "Mark and John are working at Google."


# print (ne_chunk(pos_tag(word_dict.keys())[:5]))
# names = ne_chunk(pos_tag(word_tokenize(sentence)))

# names = ne_chunk(pos_tag(word_tokenize(sentence)))

def named_entity(post_tags_for_words):
    names = ne_chunk(post_tags_for_words)
    names_dict = {}
    for n in names:
        if (len(n) == 1):
            named_entity = str(n).split(' ')[0][1:]
            word = str(n).split(' ')[1].split('/')[0]
            names_dict[word] = named_entity
    print(names_dict)


    import pandas as pd

    df = pd.DataFrame(list(names_dict.items()))
    df.columns = ['word', 'pos']
    df.pos = pd.Categorical(df.pos)
    df['code'] = df.pos.cat.codes
    # print(df)

    names_dict = {}
    for index, row in df.iterrows():
        names_dict[row['word']] = row['code']
    print(names_dict)
    return names_dict


# In[5]:


print("Building dictionary...")
word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("train", False)
print("Loading training dataset...")
train_x, train_y = build_dataset("train", word_dict, article_max_len, summary_max_len, False)


# ### Execute this later

# In[6]:


pos_list , postags_for_named_entity = get_pos_tags_dict(word_dict.keys())


# In[7]:


print("Loading Lists...")
train_article_list = get_text_list(train_article_path, False)
train_title_list = get_text_list(train_title_path, False)

print("Loading TF-IDF...")
tf_idf_list = tf_idf_generate(train_article_list+train_title_list)
tf_idf_list["apple"]


# In[8]:


pos_list["apple"]


# In[9]:


named_entity_recs = named_entity(postags_for_named_entity) 
named_entity_recs


# #### TF-IDF

# ### POS tags

# In[12]:


import nltk
nltk.download('averaged_perceptron_tagger')


# ### Named Entity Reognition

# In[13]:


import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')


# ## Prepare Data (unzip , discover operations)

# 
# https://github.com/dongjun-Lee/text-summarization-tensorflow/blob/master/prep_data.py
# 
# 1.   Word Embedding : Used [Glove pre-trained vectors](https://nlp.stanford.edu/projects/glove/ ) to initialize word embedding.  
# 2.   Dataset :  Dataset is available at [harvardnlp/sent-summary](https://github.com/harvardnlp/sent-summary). Locate the summary.tar.gz file in project root directory.   
# 
# 

# #### unzip

# In[ ]:


import wget
import os
import tarfile
import gzip
import zipfile
import argparse


#parser = argparse.ArgumentParser()
#parser.add_argument("--glove", action="store_true")
#args = parser.parse_args()

# Extract data file
#with tarfile.open(default_path + "sumdata/train/summary.tar.gz", "r:gz") as tar:
#    tar.extractall()

with gzip.open(default_path + "sumdata/train/train.article.txt.gz", "rb") as gz:
    with open(default_path + "sumdata/train/train.article.txt", "wb") as out:
        out.write(gz.read())

with gzip.open(default_path + "sumdata/train/train.title.txt.gz", "rb") as gz:
    with open(default_path + "sumdata/train/train.title.txt", "wb") as out:
        out.write(gz.read())

        
#if args.glove:
#    glove_dir = "glove"
#    glove_url = "https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip"
#
#    if not os.path.exists(glove_dir):
#        os.mkdir(glove_dir)
#
#    # Download glove vector
#    wget.download(glove_url, out=glove_dir)
#
#    # Extract glove file
#    with zipfile.ZipFile(os.path.join("glove", "glove.42B.300d.zip"), "r") as z:
#        z.extractall(glove_dir)


# In[58]:


get_ipython().system('ls')


# In[ ]:


import os
import wget

glove_dir = "glove"
glove_url = "https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip"
#
if not os.path.exists(glove_dir):
    os.mkdir(glove_dir)

# Download glove vector
wget.download(glove_url, out=glove_dir)

# Extract glove file
with zipfile.ZipFile(os.path.join(glove_dir, "glove.42B.300d.zip"), "r") as z:
    z.extractall(glove_dir)


# #### Discover Operations

# In[14]:


train_article_list = get_text_list(train_article_path, False)
train_title_list = get_text_list(train_title_path, False)


# In[15]:


test = tf_idf_generate(train_article_list + train_title_list)


# In[25]:


len(train_article_list)


# In[16]:


test["apple"]


# In[17]:


print(word_dict["apple"])
print(word_dict["cat"])
print(word_dict["dog"])


# In[18]:


print(reversed_dict[9076])
print(reversed_dict[7243])
print(reversed_dict[4206])


# In[19]:


print("article_max_len : " + str(article_max_len))
print("summary_max_len : " + str(summary_max_len))


# In[20]:


print(train_x[0])
for num in train_x[0] :
    print(reversed_dict[num])


# In[21]:


print(train_y[0])
for num in train_y[0] :
  print(reversed_dict[num])


# In[22]:


test_embedding = get_init_embedding(word_dict , reversed_dict, 320)


# In[23]:


#print(reversed_dict[2000])
len(test_embedding[30000])


# In[24]:


pos_list = get_pos_tags_dict(word_dict.keys())
pos_list


# ## Model:
# 
# 

# Encoder-Decoder model with attention mechanism.
# https://github.com/dongjun-Lee/text-summarization-tensorflow/blob/master/model.py
# 
# 1.   Encoder : Used LSTM cell with stack_bidirectional_dynamic_rnn.
# 2.   Decoder : Used LSTM BasicDecoder for training, and BeamSearchDecoder for inference.
# 3.   Attention Mechanism : Used BahdanauAttention with weight normalization.

# In[26]:


import tensorflow as tf
from tensorflow.contrib import rnn
#from utils import get_init_embedding


class Model(object):
    def __init__(self, reversed_dict, article_max_len, summary_max_len, args, forward_only=False):
        self.vocabulary_size = len(reversed_dict)
        self.embedding_size = args.embedding_size
        self.num_hidden = args.num_hidden
        self.num_layers = args.num_layers
        self.learning_rate = args.learning_rate
        self.beam_width = args.beam_width
        if not forward_only:
            self.keep_prob = args.keep_prob
        else:
            self.keep_prob = 1.0
        self.cell = tf.nn.rnn_cell.BasicLSTMCell
        with tf.variable_scope("decoder/projection"):
            self.projection_layer = tf.layers.Dense(self.vocabulary_size, use_bias=False)

        self.batch_size = tf.placeholder(tf.int32, (), name="batch_size")
        self.X = tf.placeholder(tf.int32, [None, article_max_len])
        self.X_len = tf.placeholder(tf.int32, [None])
        self.decoder_input = tf.placeholder(tf.int32, [None, summary_max_len])
        self.decoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_target = tf.placeholder(tf.int32, [None, summary_max_len])
        self.global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("embedding"):
            if not forward_only and args.glove: #training
                #init_embeddings = tf.constant(get_init_embedding(word_dict ,reversed_dict, self.embedding_size), dtype=tf.float32)
                init_embeddings = tf.constant(test_embedding, dtype=tf.float32)
            else: #testing
                init_embeddings = tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0)
            self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.encoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.X), perm=[1, 0, 2])
            self.decoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.decoder_input), perm=[1, 0, 2])

        with tf.name_scope("encoder"):
            fw_cells = [self.cell(self.num_hidden) for _ in range(self.num_layers)]
            bw_cells = [self.cell(self.num_hidden) for _ in range(self.num_layers)]
            # TODO: Dropout
            fw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob) for cell in fw_cells]
            bw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob) for cell in bw_cells]

            encoder_outputs, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells, self.encoder_emb_inp,
                sequence_length=self.X_len, time_major=True, dtype=tf.float32)
            self.encoder_output = tf.concat(encoder_outputs, 2)
            encoder_state_c = tf.concat((encoder_state_fw[0].c, encoder_state_bw[0].c), 1)
            encoder_state_h = tf.concat((encoder_state_fw[0].h, encoder_state_bw[0].h), 1)
            self.encoder_state = rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

        with tf.name_scope("decoder"), tf.variable_scope("decoder") as decoder_scope:
            decoder_cell = self.cell(self.num_hidden * 2)

            if not forward_only: #trainig
                attention_states = tf.transpose(self.encoder_output, [1, 0, 2])
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    self.num_hidden * 2, attention_states, memory_sequence_length=self.X_len, normalize=True)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                                   attention_layer_size=self.num_hidden * 2)
                initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
                initial_state = initial_state.clone(cell_state=self.encoder_state)
                helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_emb_inp, self.decoder_len, time_major=True)
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, scope=decoder_scope)
                self.decoder_output = outputs.rnn_output
                self.logits = tf.transpose(
                    self.projection_layer(self.decoder_output), perm=[1, 0, 2])
                self.logits_reshape = tf.concat(
                    [self.logits, tf.zeros([self.batch_size, summary_max_len - tf.shape(self.logits)[1], self.vocabulary_size])], axis=1)
            else: #testing
                tiled_encoder_output = tf.contrib.seq2seq.tile_batch(
                    tf.transpose(self.encoder_output, perm=[1, 0, 2]), multiplier=self.beam_width)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(self.encoder_state, multiplier=self.beam_width)
                tiled_seq_len = tf.contrib.seq2seq.tile_batch(self.X_len, multiplier=self.beam_width)
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    self.num_hidden * 2, tiled_encoder_output, memory_sequence_length=tiled_seq_len, normalize=True)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                                   attention_layer_size=self.num_hidden * 2)
                initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size * self.beam_width)
                initial_state = initial_state.clone(cell_state=tiled_encoder_final_state)
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=self.embeddings,
                    start_tokens=tf.fill([self.batch_size], tf.constant(2)),
                    end_token=tf.constant(3),
                    initial_state=initial_state,
                    beam_width=self.beam_width,
                    output_layer=self.projection_layer
                )
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, output_time_major=True, maximum_iterations=summary_max_len, scope=decoder_scope)
                self.prediction = tf.transpose(outputs.predicted_ids, perm=[1, 2, 0])

        with tf.name_scope("loss"):
            if not forward_only: #training
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits_reshape, labels=self.decoder_target)
                weights = tf.sequence_mask(self.decoder_len, summary_max_len, dtype=tf.float32)
                self.loss = tf.reduce_sum(crossent * weights / tf.to_float(self.batch_size))
                # TODO: Regularization
                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.update = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)


# ## Train

# https://github.com/dongjun-Lee/text-summarization-tensorflow/blob/master/train.py
# 
# We used sumdata/train/train.article.txt and sumdata/train/train.title.txt for training data. To train the model, use

# In[ ]:


import time
start = time.perf_counter()
import tensorflow as tf
import argparse
import pickle
import os
#from model import Model
#from utils import build_dict, build_dataset, batch_iter

# Uncomment next 2 lines to suppress error and Tensorflow info verbosity. Or change logging levels
# tf.logging.set_verbosity(tf.logging.FATAL)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#def add_arguments(parser):
#    parser.add_argument("--num_hidden", type=int, default=150, help="Network size.")
#    parser.add_argument("--num_layers", type=int, default=2, help="Network depth.")
#    parser.add_argument("--beam_width", type=int, default=10, help="Beam width for beam search decoder.")
#    parser.add_argument("--glove", action="store_true", help="Use glove as initial word embedding.")
#    parser.add_argument("--embedding_size", type=int, default=300, help="Word embedding size.")
#
#    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
#    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
#    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs.")
#    parser.add_argument("--keep_prob", type=float, default=0.8, help="Dropout keep prob.")
#
#    parser.add_argument("--toy", action="store_true", help="Use only 50K samples of data")
#
#    parser.add_argument("--with_model", action="store_true", help="Continue from previously saved model")

class args:
    pass
  
args.num_hidden=150
args.num_layers=2
args.beam_width=10
args.glove="store_true"
args.embedding_size=320

args.learning_rate=1e-3
args.batch_size=64
args.num_epochs=10
args.keep_prob = 0.8

args.toy=False #"store_true"

args.with_model="store_true"


#parser = argparse.ArgumentParser()
#add_arguments(parser)
#args = parser.parse_args()
#with open("args.pickle", "wb") as f:
#    pickle.dump(args, f)

if not os.path.exists(default_path + "saved_model_2"):
    os.mkdir(default_path + "saved_model_2")
else:
    #if args.with_model:
        old_model_checkpoint_path = open(default_path + 'saved_model_2/checkpoint', 'r')
#         old_model_checkpoint_path = "".join([default_path + "saved_model_2/",old_model_checkpoint_path.read().splitlines()[0].split('"')[1] ])
#  abs Change made by paras
        old_model_checkpoint_path = old_model_checkpoint_path.read().splitlines()[0].split('"')[1]


#print("Building dictionary...")
#word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("train", args.toy)
#print("Loading training dataset...")
#train_x, train_y = build_dataset("train", word_dict, article_max_len, summary_max_len, args.toy)

tf.reset_default_graph()

with tf.Session() as sess:
    model = Model(reversed_dict, article_max_len, summary_max_len, args)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    if 'old_model_checkpoint_path' in globals():
        print("Continuing from previous trained model:" , old_model_checkpoint_path , "...")
        saver.restore(sess, old_model_checkpoint_path )

    batches = batch_iter(train_x, train_y, args.batch_size, args.num_epochs)
    num_batches_per_epoch = (len(train_x) - 1) // args.batch_size + 1

    print("\nIteration starts.")
    print("Number of batches per epoch :", num_batches_per_epoch)
    for batch_x, batch_y in batches:
        batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))
        batch_decoder_input = list(map(lambda x: [word_dict["<s>"]] + list(x), batch_y))
        batch_decoder_len = list(map(lambda x: len([y for y in x if y != 0]), batch_decoder_input))
        batch_decoder_output = list(map(lambda x: list(x) + [word_dict["</s>"]], batch_y))

        batch_decoder_input = list(
            map(lambda d: d + (summary_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_input))
        batch_decoder_output = list(
            map(lambda d: d + (summary_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_output))

        train_feed_dict = {
            model.batch_size: len(batch_x),
            model.X: batch_x,
            model.X_len: batch_x_len,
            model.decoder_input: batch_decoder_input,
            model.decoder_len: batch_decoder_len,
            model.decoder_target: batch_decoder_output
        }

        _, step, loss = sess.run([model.update, model.global_step, model.loss], feed_dict=train_feed_dict)

        if step % 1000 == 0:
            print("step {0}: loss = {1}".format(step, loss))

        if step % num_batches_per_epoch == 0:
            hours, rem = divmod(time.perf_counter() - start, 3600)
            minutes, seconds = divmod(rem, 60)
            saver.save(sess, default_path + "saved_model_2/model.ckpt", global_step=step)
            print(" Epoch {0}: Model is saved.".format(step // num_batches_per_epoch),
            "Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds) , "\n")


# last training was 
# ```
# step 47000: loss = 8.033841133117676
# step 48000: loss = 9.481734275817871
# step 49000: loss = 7.188093662261963
# step 50000: loss = 14.354914665222168
# Epoch 16: Model is saved. Elapsed: 02:19:23.39
# ```
# 
# 

# ## Test

# https://github.com/dongjun-Lee/text-summarization-tensorflow/blob/master/test.py
# 
# Generate summary of each article in sumdata/train/valid.article.filter.txt by

# In[ ]:


import tensorflow as tf
import pickle
#from model import Model
#from utils import build_dict, build_dataset, batch_iter


#with open("args.pickle", "rb") as f:
#    args = pickle.load(f)

tf.reset_default_graph()

class args:
    pass
  
args.num_hidden=150
args.num_layers=2
args.beam_width=10
args.glove="store_true"
args.embedding_size=320

args.learning_rate=1e-3
args.batch_size=64
args.num_epochs=10
args.keep_prob = 0.8

args.toy=True

args.with_model="store_true"



print("Loading dictionary...")
word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("valid", args.toy)
print("Loading validation dataset...")
valid_x = build_dataset("valid", word_dict, article_max_len, summary_max_len, args.toy)
valid_x_len = [len([y for y in x if y != 0]) for x in valid_x]
print("Loading article and reference...")
article = get_text_list(valid_article_path, args.toy)
reference = get_text_list(valid_title_path, args.toy)

with tf.Session() as sess:
    print("Loading saved model...")
    model = Model(reversed_dict, article_max_len, summary_max_len, args, forward_only=True)
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(default_path + "saved_model_2/")
    saver.restore(sess, ckpt.model_checkpoint_path)

    batches = batch_iter(valid_x, [0] * len(valid_x), args.batch_size, 1)

    print("Writing summaries to 'result2.txt'...")
    for batch_x, _ in batches:
        batch_x_len = [len([y for y in x if y != 0]) for x in batch_x]

        valid_feed_dict = {
            model.batch_size: len(batch_x),
            model.X: batch_x,
            model.X_len: batch_x_len,
        }

        prediction = sess.run(model.prediction, feed_dict=valid_feed_dict)
        prediction_output = [[reversed_dict[y] for y in x] for x in prediction[:, 0, :]]
        summary_array = []
        with open(default_path + "result2.txt", "wb") as f:
            for line in prediction_output:
                summary = list()
                for word in line:
                    if word == "</s>":
                        break
                    if word not in summary:
                        summary.append(word)
                summary_array.append(" ".join(summary))
                #print(" ".join(summary), file=f)

    print('Summaries have been generated')


# In[ ]:


summary_array = []
with open(default_path + "result2.txt", "wb") as f:
    for line in prediction_output:
        summary = list()
        for word in line:
            if word == "</s>":
                break
            if word not in summary:
                summary.append(word)
        summary_array.append(" ".join(summary))


# In[ ]:


summary_array[:2]


# ## Evaluate & write output

# for comparing (good resources)
# 
# 1.  [thunlp]( https://github.com/thunlp/TensorFlow-Summarization)     works with duc2003
# 2.  [textsum](https://github.com/tensorflow/models/tree/master/research/textsum )   
# 
# 

# In[ ]:


get_ipython().system('pip install sumeval')
get_ipython().system('python -m spacy download en')


# In[ ]:


#https://github.com/chakki-works/sumeval
#https://github.com/Tian312/awesome-text-summarization

from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator

def eval_rouges(refrence_summary,model_summary):
    refrence_summary = "tokyo shares close up #.## percent"
    model_summary = "tokyo stocks close up # percent to fresh record high"

    rouge = RougeCalculator(stopwords=True, lang="en")

    rouge_1 = rouge.rouge_n(
                summary=model_summary,
                references=refrence_summary,
                n=1)

    rouge_2 = rouge.rouge_n(
                summary=model_summary,
                references=[refrence_summary],
                n=2)
    
    rouge_l = rouge.rouge_l(
                summary=model_summary,
                references=[refrence_summary])
    
    # You need spaCy to calculate ROUGE-BE
    
    rouge_be = rouge.rouge_be(
                summary=model_summary,
                references=[refrence_summary])

    bleu = BLEUCalculator()
    bleu_score = bleu.bleu( summary=model_summary,
                        references=[refrence_summary])

    print("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}, ROUGE-BE: {}".format(
        rouge_1, rouge_2, rouge_l, rouge_be
    ).replace(", ", "\n"))
    
    return rouge_1, rouge_2,rouge_l,rouge_be,bleu_score
eval_rouges('a','b')
#rouge_1, rouge_2,rouge_l,rouge_be = eval_rouges( "tokyo shares close up #.## percent",  "tokyo stocks close up # percent to fresh record high")

#print("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}, ROUGE-BE: {}".format(     rouge_1, rouge_2, rouge_l, rouge_be).replace(", ", "\n"))


# In[4]:


#https://pymotw.com/2/xml/etree/ElementTree/create.html

bleu_arr = []
rouge_1_arr  = []
rouge_2_arr  = []
rouge_L_arr  = []
rouge_be_arr = []

from xml.etree import ElementTree
from xml.dom import minidom
from functools import reduce

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
  
from xml.etree.ElementTree import Element, SubElement, Comment

top = Element('ZakSum')

comment = Comment('Generated by Amr Zaki')
top.append(comment)

i=0
for summ in summary_array:
    example = SubElement(top, 'example')
    article_element   = SubElement(example, 'article')
    article_element.text = article[i]

    reference_element = SubElement(example, 'reference')
    reference_element.text = reference[i]

    summary_element   = SubElement(example, 'summary')
    summary_element.text = summ

    rouge_1, rouge_2,rouge_L,rouge_be,bleu_score = eval_rouges(reference[i],summ )

    eval_element = SubElement(example, 'eval')
    bleu_score_element = SubElement(eval_element,'BLEU', {'score':str(bleu_score)})
    ROUGE_1_element  = SubElement(eval_element, 'ROUGE_1' , {'score':str(rouge_1)})
    ROUGE_2_element  = SubElement(eval_element, 'ROUGE_2' , {'score':str(rouge_2)})
    ROUGE_L_element  = SubElement(eval_element, 'ROUGE_l' , {'score':str(rouge_L)})
    ROUGE_be_element  = SubElement(eval_element,'ROUGE_be', {'score':str(rouge_be)})

    bleu_arr.append(bleu_score) 
    rouge_1_arr.append(rouge_1) 
    rouge_2_arr.append(rouge_2) 
    rouge_L_arr.append(rouge_L) 
    rouge_be_arr.append(rouge_be) 

    i+=1

top.set('bleu', str(reduce(lambda x, y: x + y,  bleu_arr) / len(bleu_arr)))
top.set('rouge_1', str(reduce(lambda x, y: x + y,  rouge_1_arr) / len(rouge_1_arr)))
top.set('rouge_2', str(reduce(lambda x, y: x + y,  rouge_2_arr) / len(rouge_2_arr)))
top.set('rouge_L', str(reduce(lambda x, y: x + y,  rouge_L_arr) / len(rouge_L_arr)))
top.set('rouge_be', str(reduce(lambda x, y: x + y, rouge_be_arr) / len(rouge_be_arr)))

with open(default_path + "result_featurerich_15_11_2018_5_28pm.xml", "w+") as f:
    print(prettify(top), file=f)


# In[5]:


len(summary_array)


# In[6]:


sentence = "The Lion Air flight JT 610 had been carrying 189 people, including three children, when it disappeared from radar during a short flight from Jakarta to Pangkal Pinang on the Indonesian island of Bangka, according to Basarnas, Indonesia's national search and rescue agency."


# In[ ]:




