
# Abstractive Summarization

Loading pre-trained GloVe embeddings.
Source of Data: https://nlp.stanford.edu/projects/glove/

Another interesting embedding to look into:
https://github.com/commonsense/conceptnet-numberbatch


```python
import numpy as np
from __future__ import division

filename = 'glove.6B.50d.txt'
def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd
vocab,embd = loadGloVe(filename)

embedding = np.asarray(embd)
embedding = embedding.astype(np.float32)

word_vec_dim = len(embedding[0])
#Pre-trained GloVe embedding
```

    Loaded GloVe!


Here I will define functions for converting words to its vector representations and vice versa. 

### word2vec: 

Converts words to its vector representations.
If the word is not present in the vocabulary, and thus if it doesn't have any vector representation,
the word will be considered as 'unk' (denotes unknown) and the vector representation of unk will be
returned instead. 

### np_nearest_neighbour:

Returns the word vector in the vocabularity that is most similar
to word vector given as an argument. The similarity is evaluated based on the formula of cosine
similarity. 

### vec2word: 

Converts vectors to words. If the vector representation is unknown, and no corresponding word
is known, then it returns the word representation of a known vector representation which is most similar 
to the vector given as argument (the np_nearest_neighbour() function is used for that).



```python
def np_nearest_neighbour(x):
    #returns array in embedding that's most similar (in terms of cosine similarity) to x
        
    xdoty = np.multiply(embedding,x)
    xdoty = np.sum(xdoty,1)
    xlen = np.square(x)
    xlen = np.sum(xlen,0)
    xlen = np.sqrt(xlen)
    ylen = np.square(embedding)
    ylen = np.sum(ylen,1)
    ylen = np.sqrt(ylen)
    xlenylen = np.multiply(xlen,ylen)
    cosine_similarities = np.divide(xdoty,xlenylen)

    return embedding[np.argmax(cosine_similarities)]


def word2vec(word):  # converts a given word into its vector representation
    if word in vocab:
        return embedding[vocab.index(word)]
    else:
        return embedding[vocab.index('unk')]

def vec2word(vec):   # converts a given vector representation into the represented word 
    for x in xrange(0, len(embedding)):
        if np.array_equal(embedding[x],np.asarray(vec)):
            return vocab[x]
    return vec2word(np_nearest_neighbour(np.asarray(vec)))
```

### Loading pre-processed dataset

The Data is preprocessed in [Data_Pre-processing.ipynb](https://github.com/JRC1995/Abstractive-Summarization/blob/master/Data%20Pre-processing.ipynb)

Dataset source: https://www.kaggle.com/snap/amazon-fine-food-reviews


```python
import pickle

with open ('vec_summaries', 'rb') as fp:
    vec_summaries = pickle.load(fp)

with open ('vec_texts', 'rb') as fp:
    vec_texts = pickle.load(fp)
    
```

Here, I am Loading vocab_limit and embd_limit (though I may not ever use embd_limit).
Vocab_limit contains only vocabularies that are present in the dataset and 
some special words representing markers 'eos', '<PAD>' etc.

The network should output the probability distribution over the words in 
vocab_limit. So using limited vocabulary (vocab_limit) will mean requiring
less parameters for calculating the probability distribution.


```python
with open ('vocab_limit', 'rb') as fp:
    vocab_limit = pickle.load(fp)

with open ('embd_limit', 'rb') as fp:
    embd_limit = pickle.load(fp)
    
```

Creating a one hot encoded vector to represent <SOS> which will represent the starting token for the decoder or the initial decoded input.


```python
vocab_limit.append('<SOS>')

SOS_prob_dist = np.zeros((len(vocab_limit)),dtype=np.float32)
SOS_prob_dist[vocab_limit.index('<SOS>')]=1

```

### REMOVING DATA WITH SUMMARIES WHICH ARE TOO LONG

I will not be training the model in batches. I will train the model one sample at a time, because my laptop
will probably not be able to handle batch training (the kernel crashes now and then even with SGD ).

However, if I was training in batches I had to choose a fixed maximum length for output.
Each target output is marked with the word 'eos' at the end. After that each target output can be padded with
'<PAD>' to fit the maximum output length. The network can be taught to produce an output in the form
"word1 word2 eos <PAD> <PAD>". The batch training can be handled better if all target outputs are transformed
to a fixed length. 

But, the fixed length should be less than or equal to the length of the longest target output so as to
not discard any word from any target-output sample.

But there may be a few very long target outputs\summaries (say, 50+) whereas most summaries are near about
length 10. So to fix the length, lots of padding has to be done to most of the summaries just because there
are a few long summaries. 

Better to just remove the data whose summaries are bigger than a specified threshold (MAX_SUMMARY_LEN).
In this cell I will diagnose how many percentage of data will be removed for a given threshold length,
and in the next cell I will remove them.

Note: I am comparing len(summary_vec)-1, instead of len(summary_vec). The reason is that I am ignoring 
the last word vector which is the representation of the 'eos' marker. I will explain why later on this
notebook. 

### REMOVING DATA WITH TEXTS WHOSE LENGTH IS SMALLER THAN THE WINDOW SIZE

In this model I will try to implement <b>local attention</b> with standard encoder-decoder architecture.

Where global attention looks at all the hidden states of the encoder to determine where to attend to,
local attention looks only at the hidden states under the range pt-D to pt+D where D is empirically selected
and pt is a position determined by the program.
The range of pt-D to pt+D can be said to be the window where attention takes place.  Pt is the center of the
window.

I am treating D as a hyperparameter. The window size will be (pt-D)-(pt+D)+1 = 2D+1.

Now, obviously, the window needs to be smaller than or equal to the no. of the encoded hidden states themselves.
We will encode one hidden state for each words in the input text, so size of the hidden states will be equivalent
to the size of the input text.

So we must choose D such that 2D+1 is not bigger than the length of any text in the dataset.

To ensure that, I will first diagnose how many data will be removed for a given D, and in the next cell,
I will remove all input texts whose length is less than 2D+1.

### REMOVING DATA WITH TEXTS(REVIEWS) WHICH ARE TOO LONG

The RNN encoders will encode one word at a time. No. of words in the text data or in other words,
the length of the text size will also be the no. of timesteps for the encoder RNN. To make the training less intensive 
(so that it doesn't burden my laptop too much), I will be removing
all data with whose review size exceeds a given threshold (MAX_TEXT_LEN).



```python
#DIAGNOSIS

count = 0

LEN = 7

for summary in vec_summaries:
    if len(summary)-1>LEN:
        count = count + 1
print "Percentage of dataset with summary length beyond "+str(LEN)+": "+str((count/len(vec_summaries))*100)+"% "

count = 0

D = 10 

window_size = 2*D+1

for text in vec_texts:
    if len(text)<window_size+1:
        count = count + 1
print "Percentage of dataset with text length less that window size: "+str((count/len(vec_texts))*100)+"% "

count = 0

LEN = 80

for text in vec_texts:
    if len(text)>LEN:
        count = count + 1
print "Percentage of dataset with text length more than "+str(LEN)+": "+str((count/len(vec_texts))*100)+"% "
```

    Percentage of dataset with summary length beyond 7: 16.146% 
    Percentage of dataset with text length less that window size: 2.258% 
    Percentage of dataset with text length more than 80: 40.412% 


Here I will start the aformentioned removal process.
vec_summary_reduced and vec_texts_reduced will contain the remaining data after the removal.

<b>Note: an important hyperparameter D is initialized here.</b>

D determines the window size of local attention as mentioned before.


```python
MAX_SUMMARY_LEN = 7
MAX_TEXT_LEN = 80

#D is a major hyperparameters. Windows size for local attention will be 2*D+1
D = 10

window_size = 2*D+1

#REMOVE DATA WHOSE SUMMARIES ARE TOO BIG
#OR WHOSE TEXT LENGTH IS TOO BIG
#OR WHOSE TEXT LENGTH IS SMALLED THAN WINDOW SIZE

vec_summaries_reduced = []
vec_texts_reduced = []

i = 0
for summary in vec_summaries:
    if len(summary)-1<=MAX_SUMMARY_LEN and len(vec_texts[i])>=window_size and len(vec_texts[i])<=MAX_TEXT_LEN:
        vec_summaries_reduced.append(summary)
        vec_texts_reduced.append(vec_texts[i])
    i=i+1
```

Here I will start the aformentioned removal process.
vec_summary_reduced and vec_texts_reduced will contain the remaining data after the removal.

<b>Note: an important hyperparameter D is initialized here.</b>

D determines the window size of local attention as mentioned before.


```python
train_len = int((.7)*len(vec_summaries_reduced))

train_texts = vec_texts_reduced[0:train_len]
train_summaries = vec_summaries_reduced[0:train_len]

val_len = int((.15)*len(vec_summaries_reduced))

val_texts = vec_texts_reduced[train_len:train_len+val_len]
val_summaries = vec_summaries_reduced[train_len:train_len+val_len]

test_texts = vec_texts_reduced[train_len+val_len:len(vec_summaries_reduced)]
test_summaries = vec_summaries_reduced[train_len+val_len:len(vec_summaries_reduced)]
```


```python
print train_len
```

    18293


The function transform_out() will convert the target output sample so that 
it can be in a format which can be used by tensorflow's 
sparse_softmax_cross_entropy_with_logits() for loss calculation.

Think of one hot encoding. This transformation is kind of like that.
All the words in the vocab_limit are like classes in this context.

However, instead of being precisely one hot encoded the output will be transformed
such that it will contain the list of indexes which would have been 'one' if it was one hot encoded.


```python
def transform_out(output_text):
    output_len = len(output_text)
    transformed_output = np.zeros([output_len],dtype=np.int32)
    for i in xrange(0,output_len):
        transformed_output[i] = vocab_limit.index(vec2word(output_text[i]))
    return transformed_output   
```

### Hyperparameters

Here I am simply setting up some of the rest of the hyperparameters.
K, here, is a special hyperparameter. It denotes the no. of previous hidden states
to consider for residual connections. More on that later. 


```python
#Some MORE hyperparameters and other stuffs

hidden_size = 250
learning_rate = 0.001
K = 5
vocab_len = len(vocab_limit)
training_iters = 5 
```

Setting up tensorflow placeholders.
The purpose of the placeholders are pretty much self explanatory from the name.

Note: tf_seq_len, and tf_output_len aren't really necessary. They can be derived 
from tf_text and tf_summary respectively, but I ended up making them anyway.


```python
import tensorflow as tf

#placeholders
tf_text = tf.placeholder(tf.float32, [None,word_vec_dim])
tf_seq_len = tf.placeholder(tf.int32)
tf_summary = tf.placeholder(tf.int32,[None])
tf_output_len = tf.placeholder(tf.int32)
```

### FORWARD AND BACKWARD LSTM WITH RRA

I will be using the encoder-decoder architecture.
For the encoder I will be using a bi-directional LSTM.
Below is the function of the forward encoder (the LSTM in the forward direction
that starts from the first word and encodes a word in the context of previous words),
and then for the backward encoder (the LSTM in the backward direction
that starts from the last word and encodes a word in the context of later words)

The RNN used here, is a standard LSTM with RRA ([Residual Recurrent Attention](https://arxiv.org/abs/1709.03714))

Remember, the hyperparameter K?

The model will compute the weighted sum (weighted based on some trainable parameters
in the attention weight matrix) of the PREVIOUS K hidden states - the weighted sum
is denoted as RRA in this function.

hidden_residuals will contain the last K hidden states.

The RRA will influence the Hidden State calculation in LSTM.

(The attention weight matrix is to be normalized by dividing each elements by the sum of all 
the elements as said in the paper. But, here, I am normalizing it by softmax)

The purpose for this is to created connections between hidden states of different timesteps,
to establish long term dependencies.


```python
def forward_encoder(inp,hidden,cell,
                    wf,uf,bf,
                    wi,ui,bi,
                    wo,uo,bo,
                    wc,uc,bc,
                    Wattention,seq_len,inp_dim):

    Wattention = tf.nn.softmax(Wattention,0)
    hidden_forward = tf.TensorArray(size=seq_len,dtype=tf.float32)
    
    hidden_residuals = tf.TensorArray(size=K,dynamic_size=True,dtype=tf.float32,clear_after_read=False)
    hidden_residuals = hidden_residuals.unstack(tf.zeros([K,hidden_size],dtype=tf.float32))
    
    i=0
    j=K
    
    def cond(i,j,hidden,cell,hidden_forward,hidden_residuals):
        return i < seq_len
    
    def body(i,j,hidden,cell,hidden_forward,hidden_residuals):
        
        x = tf.reshape(inp[i],[1,inp_dim])
        
        hidden_residuals_stack = hidden_residuals.stack()
        
        RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j-K:j],Wattention),0)
        RRA = tf.reshape(RRA,[1,hidden_size])
        
        # LSTM with RRA
        fg = tf.sigmoid( tf.matmul(x,wf) + tf.matmul(hidden,uf) + bf)
        ig = tf.sigmoid( tf.matmul(x,wi) + tf.matmul(hidden,ui) + bi)
        og = tf.sigmoid( tf.matmul(x,wo) + tf.matmul(hidden,uo) + bo)
        cell = tf.multiply(fg,cell) + tf.multiply(ig,tf.sigmoid( tf.matmul(x,wc) + tf.matmul(hidden,uc) + bc))
        hidden = tf.multiply(og,tf.tanh(cell+RRA))
        
        hidden_residuals = tf.cond(tf.equal(j,seq_len-1+K),
                                   lambda: hidden_residuals,
                                   lambda: hidden_residuals.write(j,tf.reshape(hidden,[hidden_size])))

        hidden_forward = hidden_forward.write(i,tf.reshape(hidden,[hidden_size]))
        
        return i+1,j+1,hidden,cell,hidden_forward,hidden_residuals
    
    _,_,_,_,hidden_forward,hidden_residuals = tf.while_loop(cond,body,[i,j,hidden,cell,hidden_forward,hidden_residuals])
    
    hidden_residuals.close().mark_used()
    
    return hidden_forward.stack()
        
```


```python
def backward_encoder(inp,hidden,cell,
                     wf,uf,bf,
                     wi,ui,bi,
                     wo,uo,bo,
                     wc,uc,bc,
                     Wattention,seq_len,inp_dim):
    
    Wattention = tf.nn.softmax(Wattention,0)
    hidden_backward = tf.TensorArray(size=seq_len,dtype=tf.float32)
    
    hidden_residuals = tf.TensorArray(size=K,dynamic_size=True,dtype=tf.float32,clear_after_read=False)
    hidden_residuals = hidden_residuals.unstack(tf.zeros([K,hidden_size],dtype=tf.float32))
    
    i=seq_len-1
    j=K
    
    def cond(i,j,hidden,cell,hidden_backward,hidden_residuals):
        return i > -1
    
    def body(i,j,hidden,cell,hidden_backward,hidden_residuals):
        
        x = tf.reshape(inp[i],[1,inp_dim])
        
        hidden_residuals_stack = hidden_residuals.stack()
        
        RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j-K:j],Wattention),0)
        RRA = tf.reshape(RRA,[1,hidden_size])
        
        # LSTM with RRA
        fg = tf.sigmoid( tf.matmul(x,wf) + tf.matmul(hidden,uf) + bf)
        ig = tf.sigmoid( tf.matmul(x,wi) + tf.matmul(hidden,ui) + bi)
        og = tf.sigmoid( tf.matmul(x,wo) + tf.matmul(hidden,uo) + bo)
        cell = tf.multiply(fg,cell) + tf.multiply(ig,tf.sigmoid( tf.matmul(x,wc) + tf.matmul(hidden,uc) + bc))
        hidden = tf.multiply(og,tf.tanh(cell+RRA))

        hidden_residuals = tf.cond(tf.equal(j,seq_len-1+K),
                                   lambda: hidden_residuals,
                                   lambda: hidden_residuals.write(j,tf.reshape(hidden,[hidden_size])))
        
        hidden_backward = hidden_backward.write(i,tf.reshape(hidden,[hidden_size]))
        
        return i-1,j+1,hidden,cell,hidden_backward,hidden_residuals
    
    _,_,_,_,hidden_backward,hidden_residuals = tf.while_loop(cond,body,[i,j,hidden,cell,hidden_backward,hidden_residuals])

    hidden_residuals.close().mark_used()
    
    return hidden_backward.stack()
        
```

The decoder similarly uses LSTM with RRA


```python
def decoder(x,hidden,cell,
            wf,uf,bf,
            wi,ui,bi,
            wo,uo,bo,
            wc,uc,bc,RRA):
    
    # LSTM with RRA
    fg = tf.sigmoid( tf.matmul(x,wf) + tf.matmul(hidden,uf) + bf)
    ig = tf.sigmoid( tf.matmul(x,wi) + tf.matmul(hidden,ui) + bi)
    og = tf.sigmoid( tf.matmul(x,wo) + tf.matmul(hidden,uo) + bo)
    cell_next = tf.multiply(fg,cell) + tf.multiply(ig,tf.sigmoid( tf.matmul(x,wc) + tf.matmul(hidden,uc) + bc))
    hidden_next = tf.multiply(og,tf.tanh(cell+RRA))
    
    return hidden_next,cell_next
```

### LOCAL ATTENTION:

The cell below includes some major functions for the attention mechanism.

The attention mechanism is usually implemented to compute an attention score 
for each of the encoded hidden state in the context of a particular
decoder hidden state in each timestep - all to determine which encoded hidden
states to attend to for a particular decoder hidden state context.

More specifically, I am here implementing local attention as opposed to global attention.

I already mentioned local attention before. Local attention mechanism involves focusing on
a subset of encoded hidden states, whereas a gloabl attention mechanism invovles focusing on all
the encoded hidden states.

This is the paper on which this implementation is based on:
https://nlp.stanford.edu/pubs/emnlp15_attn.pdf
    
Following the formulas presented in the paper, first, I am computing
the position pt (the center of the window of attention).

pt is simply a position in the sequence.
For a given pt, the model will only consider the hidden state starting from the position
pt-D to the hidden state at the position pt+D. 

To say a hidden state is at position p, I mean to say that the hidden state is the encoded
representation of a word at position p in the sequence.

The paper formulates the equation for calculating pt like this:
pt = sequence_length x sigmoid(..some linear algebras and activations...)

But, I didn't used the sequence_length of the whole text which is tf_seq_len but 'positions' which
is = tf_seq_len-1-2D

if pt = tf_seq_len x sigmoid(tensor)

Then pt will be in the range 0 to tf_seq_len

But, we can't have that. There is no tf_seq_len position. Since the length is tf_seq_len,
the available positions are 0 to (tf_seq_len-1). Which is why I subtracted 1 from it.

Next, we must have the value of pt to be such that it represents the CENTER of the window.
If pt is too close to 0, pt-D will be negative - a non-existent position.
If pt is too close to tf_seq_len, pt+D will be a non-existent position.

So pt can't occupy the first D positions (0 to D-1) and it can't occupy the last D positions
((tf_seq_len-D) to (tf_seq_len-1)) in order to keep pt-D and pt+D as legal positions.
So a total 2D positions should be restricted to pt.

Which is why I further subtracted 2D from tf_seq_len.

Still, after calculating pt = positions x sigmoid(tensor)
where positions = tf_seq_len-(2D+1), 
pt will merely range between 0 to tf_seq_len-(2D+1)

We can't still accept pt to be 0 since pt-D will be negative. But the length of the range 
of integer positions pt can occupy is now perfect.

So at this point, we can simply center pt at the window by adding a D.

After that, pt will range from D to (tf_seq_len-1)-D

Now, it can be checked that pt+D, or pt-D will never become negative or exceed
the total sequence length.

After calculating pt, we can use the formulas presented in the paper to calculate
the G score which signifies the weight (or attention) that should be given to a hidden state.

G scores is calculated for each of hidden states in the local window. This is equivalent to
a(s) used in the paper.

The function returns the G scores and the position pt, so that the model can create the 
context vector. 



```python
def score(hs,ht,Wa,seq_len):
    return tf.reshape(tf.matmul(tf.matmul(hs,Wa),tf.transpose(ht)),[seq_len])

def align(hs,ht,Wp,Vp,Wa,tf_seq_len):
   
    pd = tf.TensorArray(size=(2*D+1),dtype=tf.float32)
    
    positions = tf.cast(tf_seq_len-1-2*D,dtype=tf.float32)
    
    sigmoid_multiplier = tf.nn.sigmoid(tf.matmul(tf.tanh(tf.matmul(ht,Wp)),Vp))
    sigmoid_multiplier = tf.reshape(sigmoid_multiplier,[])
    
    pt_float = positions*sigmoid_multiplier
    
    pt = tf.cast(pt_float,tf.int32)
    pt = pt+D #center to window
    
    sigma = tf.constant(D/2,dtype=tf.float32)
    
    i = 0
    pos = pt - D
    
    def cond(i,pos,pd):
        
        return i < (2*D+1)
                      
    def body(i,pos,pd):
        
        comp_1 = tf.cast(tf.square(pos-pt),tf.float32)
        comp_2 = tf.cast(2*tf.square(sigma),tf.float32)
            
        pd = pd.write(i,tf.exp(-(comp_1/comp_2)))
            
        return i+1,pos+1,pd
                      
    i,pos,pd = tf.while_loop(cond,body,[i,pos,pd])
    
    local_hs = hs[(pt-D):(pt+D+1)]
    
    normalized_scores = tf.nn.softmax(score(local_hs,ht,Wa,2*D+1))
    
    pd=pd.stack()
    
    G = tf.multiply(normalized_scores,pd)
    G = tf.reshape(G,[2*D+1,1])
    
    return G,pt

```

### MODEL DEFINITION

First is the <b>bi-directional encoder</b>.

h_forward is the tensorarray of all the hidden states from the 
forward encoder whereas h_backward is the tensorarray of all the hidden states
from the backward encoder.

The final list of encoder hidden states are usually calculated by combining 
the equivalents of h_forward and h_backward by some means.

There are many means of combining them, like: concatenation, summation, average etc.
    
I will be using concatenation.

hidden_encoder is the final list of encoded hidden state

The first decoder input is the probability distribution with 1 at the index of <SOS>,
in other words, one hot encoded representation of <SOS> - which signifies the start of
decoding.

I am using the first encoded_hidden_state 
as the initial decoder state. The first encoded_hidden_state may have the least 
past context (none actually) but, it will have the most future context.

The next decoder hidden state is generated from the initial decoder input and the initial decoder state.
Next, I start a loop which iterates for output_len times. 

Next the <b>attention function</b> is called, to compute the G score by scoring the encoder hidden states
in term of current decoder hidden step.

The context vector is created by the weight (weighted in terms of G scores) summation
of hidden states in the local attention window.

I used the formulas mentioned here: https://nlp.stanford.edu/pubs/emnlp15_attn.pdf

to calculate the first actual (not the <SOS> one) y (output) from the context vector and decoder hidden state.

Note: y is of the same size as the no. of vocabs in vocab_limit. Y is supposed to be 
a probability distribution. The value of index i of Y denotes the probability for Y 
to be the word that is located in the index i of vacab_limit.

('beam search' is another approach to look into, for not predicting simply the next word,
but the next k words.)

This y will be the input for the <b>decoder LSTM</b>. In the context of this y and 
the current decoder hidden state, the RNN produces the next decoder hidden state. And, the loop
continues for 'output_len' no. of iterations. 

Since I will be training sample to sample, I can dynamically send the output length 
of the current sample, and the decoder loops for the given 'output length' times.

NOTE: I am saving y without softmax in the tensorarray output. Why? Because
I will be using tensorflow cost functions that requires the logits to be without
softmax (the function will internally apply Softmax).


```python
def model(tf_text,tf_seq_len,tf_output_len):
    
    #PARAMETERS
    
    #1.1 FORWARD ENCODER PARAMETERS
    
    initial_hidden_f = tf.zeros([1,hidden_size],dtype=tf.float32)
    cell_f = tf.zeros([1,hidden_size],dtype=tf.float32)
    wf_f = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    uf_f = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    bf_f = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    wi_f = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    ui_f = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    bi_f = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    wo_f = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    uo_f = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    bo_f = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    wc_f = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    uc_f = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    bc_f = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    Wattention_f = tf.Variable(tf.zeros([K,1]),dtype=tf.float32)
                               
    #1.2 BACKWARD ENCODER PARAMETERS
    
    initial_hidden_b = tf.zeros([1,hidden_size],dtype=tf.float32)
    cell_b = tf.zeros([1,hidden_size],dtype=tf.float32)
    wf_b = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    uf_b = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    bf_b = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    wi_b = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    ui_b = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    bi_b = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    wo_b = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    uo_b = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    bo_b = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    wc_b = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    uc_b = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    bc_b = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    Wattention_b = tf.Variable(tf.zeros([K,1]),dtype=tf.float32)
    
    #2 ATTENTION PARAMETERS
    
    Wp = tf.Variable(tf.truncated_normal(shape=[2*hidden_size,50],stddev=0.01))
    Vp = tf.Variable(tf.truncated_normal(shape=[50,1],stddev=0.01))
    Wa = tf.Variable(tf.truncated_normal(shape=[2*hidden_size,2*hidden_size],stddev=0.01))
    Wc = tf.Variable(tf.truncated_normal(shape=[4*hidden_size,2*hidden_size],stddev=0.01))
    
    #3 DECODER PARAMETERS
    
    Ws = tf.Variable(tf.truncated_normal(shape=[2*hidden_size,vocab_len],stddev=0.01))
    
    cell_d = tf.zeros([1,2*hidden_size],dtype=tf.float32)
    wf_d = tf.Variable(tf.truncated_normal(shape=[vocab_len,2*hidden_size],stddev=0.01))
    uf_d = tf.Variable(np.eye(2*hidden_size),dtype=tf.float32)
    bf_d = tf.Variable(tf.zeros([1,2*hidden_size]),dtype=tf.float32)
    wi_d = tf.Variable(tf.truncated_normal(shape=[vocab_len,2*hidden_size],stddev=0.01))
    ui_d = tf.Variable(np.eye(2*hidden_size),dtype=tf.float32)
    bi_d = tf.Variable(tf.zeros([1,2*hidden_size]),dtype=tf.float32)
    wo_d = tf.Variable(tf.truncated_normal(shape=[vocab_len,2*hidden_size],stddev=0.01))
    uo_d = tf.Variable(np.eye(2*hidden_size),dtype=tf.float32)
    bo_d = tf.Variable(tf.zeros([1,2*hidden_size]),dtype=tf.float32)
    wc_d = tf.Variable(tf.truncated_normal(shape=[vocab_len,2*hidden_size],stddev=0.01))
    uc_d = tf.Variable(np.eye(2*hidden_size),dtype=tf.float32)
    bc_d = tf.Variable(tf.zeros([1,2*hidden_size]),dtype=tf.float32)
    
    hidden_residuals_d = tf.TensorArray(size=K,dynamic_size=True,dtype=tf.float32,clear_after_read=False)
    hidden_residuals_d = hidden_residuals_d.unstack(tf.zeros([K,2*hidden_size],dtype=tf.float32))
    
    Wattention_d = tf.Variable(tf.zeros([K,1]),dtype=tf.float32)
    
    output = tf.TensorArray(size=tf_output_len,dtype=tf.float32)
                               
    #BI-DIRECTIONAL LSTM
                               
    hidden_forward = forward_encoder(tf_text,
                                     initial_hidden_f,cell_f,
                                     wf_f,uf_f,bf_f,
                                     wi_f,ui_f,bi_f,
                                     wo_f,uo_f,bo_f,
                                     wc_f,uc_f,bc_f,
                                     Wattention_f,
                                     tf_seq_len,
                                     word_vec_dim)
    
    hidden_backward = backward_encoder(tf_text,
                                     initial_hidden_b,cell_b,
                                     wf_b,uf_b,bf_b,
                                     wi_b,ui_b,bi_b,
                                     wo_b,uo_b,bo_b,
                                     wc_b,uc_b,bc_b,
                                     Wattention_b,
                                     tf_seq_len,
                                     word_vec_dim)
    
    encoded_hidden = tf.concat([hidden_forward,hidden_backward],1)
    
    #ATTENTION MECHANISM AND DECODER
    
    decoded_hidden = encoded_hidden[0]
    decoded_hidden = tf.reshape(decoded_hidden,[1,2*hidden_size])
    Wattention_d_normalized = tf.nn.softmax(Wattention_d)
    
    y = tf.convert_to_tensor(SOS_prob_dist) #inital output <SOS>
    y = tf.reshape(y,[1,vocab_len])
    
    j=K
    
    hidden_residuals_stack = hidden_residuals_d.stack()
    
    RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j-K:j],Wattention_d_normalized),0)
    RRA = tf.reshape(RRA,[1,2*hidden_size])
    
    decoded_hidden_next,cell_d = decoder(y,decoded_hidden,cell_d,
                                  wf_d,uf_d,bf_d,
                                  wi_d,ui_d,bf_d,
                                  wo_d,uo_d,bf_d,
                                  wc_d,uc_d,bc_d,
                                  RRA)
    decoded_hidden = decoded_hidden_next
    
    hidden_residuals_d = hidden_residuals_d.write(j,tf.reshape(decoded_hidden,[2*hidden_size]))
    
    j=j+1
                           
    i=0
    
    def attention_decoder_cond(i,j,decoded_hidden,cell_d,hidden_residuals_d,output):
        return i < tf_output_len
    
    def attention_decoder_body(i,j,decoded_hidden,cell_d,hidden_residuals_d,output):
        
        #LOCAL ATTENTION
        
        G,pt = align(encoded_hidden,decoded_hidden,Wp,Vp,Wa,tf_seq_len)
        local_encoded_hidden = encoded_hidden[pt-D:pt+D+1]
        weighted_encoded_hidden = tf.multiply(local_encoded_hidden,G)
        context_vector = tf.reduce_sum(weighted_encoded_hidden,0)
        context_vector = tf.reshape(context_vector,[1,2*hidden_size])
        
        attended_hidden = tf.tanh(tf.matmul(tf.concat([context_vector,decoded_hidden],1),Wc))
        
        #DECODER
        
        y = tf.matmul(attended_hidden,Ws)
        
        output = output.write(i,tf.reshape(y,[vocab_len]))
        
        y = tf.nn.softmax(y)
        
        hidden_residuals_stack = hidden_residuals_d.stack()
        
        RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j-K:j],Wattention_d_normalized),0)
        RRA = tf.reshape(RRA,[1,2*hidden_size])
        
        decoded_hidden_next,cell_d = decoder(y,decoded_hidden,cell_d,
                                  wf_d,uf_d,bf_d,
                                  wi_d,ui_d,bf_d,
                                  wo_d,uo_d,bf_d,
                                  wc_d,uc_d,bc_d,
                                  RRA)
        
        decoded_hidden = decoded_hidden_next
        
        hidden_residuals_d = tf.cond(tf.equal(j,tf_output_len-1+K+1), #(+1 for <SOS>)
                                   lambda: hidden_residuals_d,
                                   lambda: hidden_residuals_d.write(j,tf.reshape(decoded_hidden,[2*hidden_size])))
        
        return i+1,j+1,decoded_hidden,cell_d,hidden_residuals_d,output
    
    i,j,decoded_hidden,cell_d,hidden_residuals_d,output = tf.while_loop(attention_decoder_cond,
                                            attention_decoder_body,
                                            [i,j,decoded_hidden,cell_d,hidden_residuals_d,output])
    hidden_residuals_d.close().mark_used()
    
    output = output.stack()
    
    return output
```

The model function is initiated here. The output is
computed. Cost function and optimizer are defined.
I am creating a prediction tensorarray which will 
store the index of maximum element of 
the output probability distributions.
From that index I can find the word in vocab_limit
which is represented by it. So the final visible
predictions will be the words that the model decides to
be most probable.


```python
output = model(tf_text,tf_seq_len,tf_output_len)

#OPTIMIZER

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=tf_summary))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#PREDICTION

pred = tf.TensorArray(size=tf_output_len,dtype=tf.int32)

i=0

def cond_pred(i,pred):
    return i<tf_output_len
def body_pred(i,pred):
    pred = pred.write(i,tf.cast(tf.argmax(output[i]),tf.int32))
    return i+1,pred

i,pred = tf.while_loop(cond_pred,body_pred,[i,pred]) 

prediction = pred.stack()
```

### TRAINING

Finally, this is where training takes place.
It's all pretty self explanatory, but one thing to note is that
I am sending "train_summaries[i][0:len(train_summaries[i])-1]"
to the transform_out() function. That is, I am ignoring the last
word from summary. The last word marks the end of the summary.
It's 'eos'. 

I trained it before without dynamically feeding the output_len.
Ideally the network should determine the output_len by itself.

Which is why I defined (in past) a MAX_LEN, and transformed target outputs in
the form "word1 word2 word3....eos <PAD> <PAD>....until max_length"
I created the model output in the same way.

The model would ideally learn in which context and where to put eos.
And then the only the portion before eos can be shown to the user.

After training, the model can even be modified to run until,
the previous output y denotes an eos. 

That way, we can have variable length output, with the length decided
by the model itself, not the user.

But all the padding and eos, makes the model to come in contact with 
pads and eos in most of the target output. The model learns to consider eos and 
pad to be important. Trying to fit to the data, the early model starts to
spam eos and pad in its predicted output.

That necessarily isn't a problem. The model may learn to fare better
later on, but I planned only to check a couple of early iterations, 
and looking at predictions consisting of only eos and pads
isn't too interesting. I wanted to check what kind of words (other than
eos and pads) the model learns to produce in the early iterations. 

Which is why I am doing what I am doing. Ideally, my past implemention
waould be better. 

As I said before, I will run it for only a few early iterations.
So, it's not likely to see any great predicted summaries here.
As can be seen, the summaries seem more influenced by previous 
output sample than the input context in these early iterations.

Some of the texts contains undesirable words like br tags and so
on. So better preprocessing and tokenization may be desirable.

With more layer depth, larger hidden size, mini-batch training,
and other changes, this model may have potential, or may not.

The same arcitechture should be usable for training on translation data.



```python
import string
from __future__ import print_function

init = tf.global_variables_initializer()


with tf.Session() as sess: # Start Tensorflow Session
    
    saver = tf.train.Saver() 
    # Prepares variable for saving the model
    sess.run(init) #initialize all variables
    step = 0   
    loss_list=[]
    acc_list=[]
    val_loss_list=[]
    val_acc_list=[]
    best_val_acc=0
    display_step = 1
    
    while step < training_iters:
        
        total_loss=0
        total_acc=0
        total_val_loss = 0
        total_val_acc = 0
           
        for i in xrange(0,train_len):
            
            train_out = transform_out(train_summaries[i][0:len(train_summaries[i])-1])
            
            if i%display_step==0:
                print("\nIteration: "+str(i))
                print("Training input sequence length: "+str(len(train_texts[i])))
                print("Training target outputs sequence length: "+str(len(train_out)))
            
                print("\nTEXT:")
                flag = 0
                for vec in train_texts[i]:
                    if vec2word(vec) in string.punctuation or flag==0:
                        print(str(vec2word(vec)),end='')
                    else:
                        print((" "+str(vec2word(vec))),end='')
                    flag=1

                print("\n")


            # Run optimization operation (backpropagation)
            _,loss,pred = sess.run([optimizer,cost,prediction],feed_dict={tf_text: train_texts[i], 
                                                    tf_seq_len: len(train_texts[i]), 
                                                    tf_summary: train_out,
                                                    tf_output_len: len(train_out)})
            
         
            if i%display_step==0:
                print("\nPREDICTED SUMMARY:\n")
                flag = 0
                for index in pred:
                    #if int(index)!=vocab_limit.index('eos'):
                    if vocab_limit[int(index)] in string.punctuation or flag==0:
                        print(str(vocab_limit[int(index)]),end='')
                    else:
                        print(" "+str(vocab_limit[int(index)]),end='')
                    flag=1
                print("\n")
                
                print("ACTUAL SUMMARY:\n")
                flag = 0
                for vec in train_summaries[i]:
                    if vec2word(vec)!='eos':
                        if vec2word(vec) in string.punctuation or flag==0:
                            print(str(vec2word(vec)),end='')
                        else:
                            print((" "+str(vec2word(vec))),end='')
                    flag=1

                print("\n")
                print("loss="+str(loss))
            
        step=step+1
    
```

    
    Iteration: 0
    Training input sequence length: 51
    Training target outputs sequence length: 4
    
    TEXT:
    i have bought several of the vitality canned dog food products and have found them all to be of good quality. the product looks more like a stew than a processed meat and it smells better. my labrador is finicky and she appreciates this product better than most.
    
    
    PREDICTED SUMMARY:
    
    15-20 15-20 15-20 effectiveness
    
    ACTUAL SUMMARY:
    
    good quality dog food
    
    loss=10.3909
    
    Iteration: 1
    Training input sequence length: 37
    Training target outputs sequence length: 3
    
    TEXT:
    product arrived labeled as jumbo salted peanuts ... the peanuts were actually small sized unsalted. not sure if this was an error or if the vendor intended to represent the product as `` jumbo ''.
    
    
    PREDICTED SUMMARY:
    
    quality food food
    
    ACTUAL SUMMARY:
    
    not as advertised
    
    loss=10.4148
    
    Iteration: 2
    Training input sequence length: 46
    Training target outputs sequence length: 2
    
    TEXT:
    if you are looking for the secret ingredient in robitussin i believe i have found it. i got this in addition to the root beer extract i ordered( which was good) and made some cherry soda. the flavor is very medicinal.
    
    
    PREDICTED SUMMARY:
    
    quality food
    
    ACTUAL SUMMARY:
    
    cough medicine
    
    loss=10.385
    
    Iteration: 3
    Training input sequence length: 32
    Training target outputs sequence length: 2
    
    TEXT:
    great taffy at a great price. there was a wide assortment of yummy taffy. delivery was very quick. if your a taffy lover, this is a deal.
    
    
    PREDICTED SUMMARY:
    
    quality food
    
    ACTUAL SUMMARY:
    
    great taffy
    
    loss=10.3916
    
    Iteration: 4
    Training input sequence length: 30
    Training target outputs sequence length: 4
    
    TEXT:
    this taffy is so good. it is very soft and chewy. the flavors are amazing. i would definitely recommend you buying it. very satisfying!!
    
    
    PREDICTED SUMMARY:
    
    quality food food food
    
    ACTUAL SUMMARY:
    
    wonderful, tasty taffy
    
    loss=10.2868
    
    Iteration: 5
    Training input sequence length: 29
    Training target outputs sequence length: 2
    
    TEXT:
    right now i 'm mostly just sprouting this so my cats can eat the grass. they love it. i rotate it around with wheatgrass and rye too
    
    
    PREDICTED SUMMARY:
    
    not food
    
    ACTUAL SUMMARY:
    
    yay barley
    
    loss=10.3993
    
    Iteration: 6
    Training input sequence length: 29
    Training target outputs sequence length: 3
    
    TEXT:
    this is a very healthy dog food. good for their digestion. also good for small puppies. my dog eats her required amount at every feeding.
    
    
    PREDICTED SUMMARY:
    
    not food food
    
    ACTUAL SUMMARY:
    
    healthy dog food
    
    loss=9.35136
    
    Iteration: 7
    Training input sequence length: 24
    Training target outputs sequence length: 4
    
    TEXT:
    the strawberry twizzlers are my guilty pleasure- yummy. six pounds will be around for a while with my son and i.
    
    
    PREDICTED SUMMARY:
    
    not food food food
    
    ACTUAL SUMMARY:
    
    strawberry twizzlers- yummy
    
    loss=10.4702
    
    Iteration: 8
    Training input sequence length: 45
    Training target outputs sequence length: 2
    
    TEXT:
    i love eating them and they are good for watching tv and looking at movies! it is not too sweet. i like to transfer them to a zip lock baggie so they stay fresh so i can take my time eating them.
    
    
    PREDICTED SUMMARY:
    
    not food
    
    ACTUAL SUMMARY:
    
    poor taste
    
    loss=10.4183
    
    Iteration: 9
    Training input sequence length: 28
    Training target outputs sequence length: 3
    
    TEXT:
    i am very satisfied with my unk purchase. i shared these with others and we have all enjoyed them. i will definitely be ordering more.
    
    
    PREDICTED SUMMARY:
    
    not food food
    
    ACTUAL SUMMARY:
    
    love it!
    
    loss=10.3971
    
    Iteration: 10
    Training input sequence length: 31
    Training target outputs sequence length: 3
    
    TEXT:
    candy was delivered very fast and was purchased at a reasonable price. i was home bound and unable to get to a store so this was perfect for me.
    
    
    PREDICTED SUMMARY:
    
    food food food
    
    ACTUAL SUMMARY:
    
    home delivered unk
    
    loss=10.3215
    
    Iteration: 11
    Training input sequence length: 52
    Training target outputs sequence length: 2
    
    TEXT:
    my husband is a twizzlers addict. we 've bought these many times from amazon because we 're government employees living overseas and ca n't get them in the country we are assigned to. they 've always been fresh and tasty, packed well and arrive in a timely manner.
    
    
    PREDICTED SUMMARY:
    
    food food
    
    ACTUAL SUMMARY:
    
    always fresh
    
    loss=10.3019
    
    Iteration: 12
    Training input sequence length: 68
    Training target outputs sequence length: 1
    
    TEXT:
    i bought these for my husband who is currently overseas. he loves these, and apparently his staff likes them unk< br/> there are generous amounts of twizzlers in each 16-ounce bag, and this was well worth the price.< a unk '' http: unk ''> twizzlers, strawberry, 16-ounce bags( pack of 6)< unk>
    
    
    PREDICTED SUMMARY:
    
    food
    
    ACTUAL SUMMARY:
    
    twizzlers
    
    loss=9.13059
    
    Iteration: 13
    Training input sequence length: 31
    Training target outputs sequence length: 3
    
    TEXT:
    i can remember buying this candy as a kid and the quality has n't dropped in all these years. still a superb product you wo n't be disappointed with.
    
    
    PREDICTED SUMMARY:
    
    food food food
    
    ACTUAL SUMMARY:
    
    delicious product!
    
    loss=9.87088
    
    Iteration: 14
    Training input sequence length: 21
    Training target outputs sequence length: 1
    
    TEXT:
    i love this candy. after weight watchers i had to cut back but still have a craving for it.
    
    
    PREDICTED SUMMARY:
    
    food
    
    ACTUAL SUMMARY:
    
    twizzlers
    
    loss=8.08563
    
    Iteration: 15
    Training input sequence length: 72
    Training target outputs sequence length: 7
    
    TEXT:
    i have lived out of the us for over 7 yrs now, and i so miss my twizzlers!! when i go back to visit or someone visits me, i always stock up. all i can say is yum!< br/> sell these in mexico and you will have a faithful buyer, more often than i 'm able to buy them right now.
    
    
    PREDICTED SUMMARY:
    
    food food food food food food food
    
    ACTUAL SUMMARY:
    
    please sell these in mexico!!
    
    loss=9.52525
    
    Iteration: 16
    Training input sequence length: 36
    Training target outputs sequence length: 3
    
    TEXT:
    product received is as unk< br/>< br/>< a unk '' http: unk ''> twizzlers, strawberry, 16-ounce bags( pack of 6)< unk>
    
    
    PREDICTED SUMMARY:
    
    food food food
    
    ACTUAL SUMMARY:
    
    twizzlers- strawberry
    
    loss=6.50716
    
    Iteration: 17
    Training input sequence length: 43
    Training target outputs sequence length: 5
    
    TEXT:
    i was so glad amazon carried these batteries. i have a hard time finding them elsewhere because they are such a unique size. i need them for my garage door unk< br/> great deal for the price.
    
    
    PREDICTED SUMMARY:
    
    food food food food food
    
    ACTUAL SUMMARY:
    
    great bargain for the price
    
    loss=9.7148
    
    Iteration: 18
    Training input sequence length: 26
    Training target outputs sequence length: 5
    
    TEXT:
    this offer is a great price and a great taste, thanks amazon for selling this unk< br/>< br/> unk
    
    
    PREDICTED SUMMARY:
    
    food food food food food
    
    ACTUAL SUMMARY:
    
    this is my taste ...
    
    loss=9.77397
    
    Iteration: 19
    Training input sequence length: 60
    Training target outputs sequence length: 7
    
    TEXT:
    for those of us with celiac disease this product is a lifesaver and what could be better than getting it at almost half the price of the grocery or health food store! i love mccann 's instant oatmeal- all flavors!!!< br/>< br/> thanks,< br/> abby
    
    
    PREDICTED SUMMARY:
    
    food food food food food food food
    
    ACTUAL SUMMARY:
    
    love gluten free oatmeal!!!
    
    loss=7.71986
    
    Iteration: 20
    Training input sequence length: 59
    Training target outputs sequence length: 3
    
    TEXT:
    what else do you need to know? oatmeal, instant( make it with a half cup of low-fat milk and add raisins; nuke for 90 seconds). more expensive than kroger store brand oatmeal and maybe a little tastier or better texture or something. it 's still just oatmeal. mmm, convenient!
    
    
    PREDICTED SUMMARY:
    
    great food food
    
    ACTUAL SUMMARY:
    
    it 's oatmeal
    
    loss=9.44353
    
    Iteration: 21
    Training input sequence length: 79
    Training target outputs sequence length: 4
    
    TEXT:
    i ordered this for my wife as it was unk by our daughter. she has this almost every morning and likes all flavors. she 's happy, i 'm happy!!!< br/>< a unk '' http: unk ''> mccann 's instant irish oatmeal, variety pack of regular, apples& cinnamon, and maple& brown sugar, 10-count boxes( pack of 6)< unk>
    
    
    PREDICTED SUMMARY:
    
    great food food food
    
    ACTUAL SUMMARY:
    
    wife 's favorite breakfast
    
    loss=11.2363
    
    Iteration: 22
    Training input sequence length: 38
    Training target outputs sequence length: 1
    
    TEXT:
    i have mccann 's oatmeal every morning and by ordering it from amazon i am able to save almost$ 3.00 per unk< br/> it is a great product. tastes great and very healthy
    
    
    PREDICTED SUMMARY:
    
    great
    
    ACTUAL SUMMARY:
    
    unk
    
    loss=5.83876
    
    Iteration: 23
    Training input sequence length: 41
    Training target outputs sequence length: 3
    
    TEXT:
    mccann 's oatmeal is a good quality choice. our favorite is the apples and cinnamon, but we find that none of these are overly sugary. for a good hot breakfast in 2 minutes, this is excellent.
    
    
    PREDICTED SUMMARY:
    
    great great great
    
    ACTUAL SUMMARY:
    
    good hot breakfast
    
    loss=9.0744
    
    Iteration: 24
    Training input sequence length: 55
    Training target outputs sequence length: 4
    
    TEXT:
    we really like the mccann 's steel cut oats but find we do n't cook it up too unk< br/> this tastes much better to me than the grocery store brands and is just as unk< br/> anything that keeps me eating oatmeal regularly is a good thing.
    
    
    PREDICTED SUMMARY:
    
    twizzlers twizzlers!!
    
    ACTUAL SUMMARY:
    
    great taste and convenience
    
    loss=7.54807
    
    Iteration: 25
    Training input sequence length: 46
    Training target outputs sequence length: 2
    
    TEXT:
    this seems a little more wholesome than some of the supermarket brands, but it is somewhat mushy and does n't have quite as much flavor either. it did n't pass muster with my kids, so i probably wo n't buy it again.
    
    
    PREDICTED SUMMARY:
    
    twizzlers twizzlers
    
    ACTUAL SUMMARY:
    
    hearty oatmeal
    
    loss=10.0591
    
    Iteration: 26
    Training input sequence length: 52
    Training target outputs sequence length: 1
    
    TEXT:
    good oatmeal. i like the apple cinnamon the best. though i would n't follow the directions on the package since it always comes out too soupy for my taste. that could just be me since i like my oatmeal really thick to add some milk on top of.
    
    
    PREDICTED SUMMARY:
    
    twizzlers
    
    ACTUAL SUMMARY:
    
    good
    
    loss=4.35645
    
    Iteration: 27
    Training input sequence length: 25
    Training target outputs sequence length: 1
    
    TEXT:
    the flavors are good. however, i do not see any unk between this and unk oats brand- they are both mushy.
    
    
    PREDICTED SUMMARY:
    
    twizzlers
    
    ACTUAL SUMMARY:
    
    mushy
    
    loss=11.9467
    
    Iteration: 28
    Training input sequence length: 41
    Training target outputs sequence length: 2
    
    TEXT:
    this is the same stuff you can buy at the big box stores. there is nothing healthy about it. it is just carbs and sugars. save your money and get something that at least has some taste.
    
    
    PREDICTED SUMMARY:
    
    twizzlers!
    
    ACTUAL SUMMARY:
    
    same stuff
    
    loss=12.031
    
    Iteration: 29
    Training input sequence length: 25
    Training target outputs sequence length: 4
    
    TEXT:
    this oatmeal is not good. its mushy, soft, i do n't like it. quaker oats is the way to go.
    
    
    PREDICTED SUMMARY:
    
    twizzlers!!!
    
    ACTUAL SUMMARY:
    
    do n't like it
    
    loss=9.48624
    
    Iteration: 30
    Training input sequence length: 37
    Training target outputs sequence length: 3
    
    TEXT:
    we 're used to spicy foods down here in south texas and these are not at all spicy. doubt very much habanero is used at all. could take it up a notch or two.
    
    
    PREDICTED SUMMARY:
    
    twizzlers!!
    
    ACTUAL SUMMARY:
    
    not ass kickin
    
    loss=9.74505
    
    Iteration: 31
    Training input sequence length: 80
    Training target outputs sequence length: 5
    
    TEXT:
    i roast at home with a unk popcorn popper( but i do it outside, of course). these beans( coffee bean direct green mexican altura) seem to be well-suited for this method. the first and second cracks are distinct, and i 've roasted the beans from medium to slightly dark with great results every time. the aroma is strong and persistent. the taste is smooth, velvety, yet lively.
    
    
    PREDICTED SUMMARY:
    
    twizzlers!!!!
    
    ACTUAL SUMMARY:
    
    roasts up a smooth brew
    
    loss=11.588
    
    Iteration: 32
    Training input sequence length: 69
    Training target outputs sequence length: 5
    
    TEXT:
    we roast these in a large cast iron pan on the grill( about 1/3 of the bag at a time). the smell is wonderful and the roasted beans taste delicious too. more importantly, the coffee is smooth; no bitter aftertaste. on numerous occasions, we 've had to send the roasted beans home with friends because they like it so much.
    
    
    PREDICTED SUMMARY:
    
    twizzlers!!!!
    
    ACTUAL SUMMARY:
    
    our guests love it!
    
    loss=6.4716
    
    Iteration: 33
    Training input sequence length: 38
    Training target outputs sequence length: 3
    
    TEXT:
    deal was awesome! arrived before halloween as indicated and was enough to satisfy trick or treaters. i love the quality of this product and it was much less expensive than the local store 's candy.
    
    
    PREDICTED SUMMARY:
    
    twizzlers!!
    
    ACTUAL SUMMARY:
    
    awesome deal!
    
    loss=8.3475
    
    Iteration: 34
    Training input sequence length: 40
    Training target outputs sequence length: 6
    
    TEXT:
    it is chocolate, what can i say. great variety of everything our family loves. with a family of six it goes fast here. perfect variety. kit kat, unk, take five and more.
    
    
    PREDICTED SUMMARY:
    
    twizzlers!!!!!
    
    ACTUAL SUMMARY:
    
    how can you go wrong!
    
    loss=9.67854
    
    Iteration: 35
    Training input sequence length: 26
    Training target outputs sequence length: 3
    
    TEXT:
    halloween is over but, i sent a bag to my daughters class for her share. the chocolate was fresh and enjoyed by many.
    
    
    PREDICTED SUMMARY:
    
    twizzlers!!
    
    ACTUAL SUMMARY:
    
    great deal.
    
    loss=8.14521
    
    Iteration: 36
    Training input sequence length: 38
    Training target outputs sequence length: 6
    
    TEXT:
    watch your prices with this. while the assortment was good, and i did get this on a gold box purchase, the price for this was< br/>$ 3-4 less at target.
    
    
    PREDICTED SUMMARY:
    
    twizzlers!!!!!
    
    ACTUAL SUMMARY:
    
    better price for this at target
    
    loss=8.73441
    
    Iteration: 37
    Training input sequence length: 33
    Training target outputs sequence length: 2
    
    TEXT:
    this bag of candy online is pretty expensive, it should be cheaper in order to compete with grocery stores, other than that, its a good combination of my favorite candy
    
    
    PREDICTED SUMMARY:
    
    twizzlers!
    
    ACTUAL SUMMARY:
    
    pretty expensive
    
    loss=10.8112
    
    Iteration: 38
    Training input sequence length: 64
    Training target outputs sequence length: 4
    
    TEXT:
    this product serves me well as a source of electrolytes during and after a long run or bike unk< br/> i have tried all of the flavors but really do like the grapefruit flavor ... no unk and i actually like the slight unk< br/> i use other hammer products and really like their whole product line.
    
    
    PREDICTED SUMMARY:
    
    good!!!
    
    ACTUAL SUMMARY:
    
    great source of electrolytes
    
    loss=9.28261
    
    Iteration: 39
    Training input sequence length: 36
    Training target outputs sequence length: 4
    
    TEXT:
    this stuff really works for preventing cramping during the middle to latter stages of your rides. pop 1 into each water bottle and you 're set. flavor is fine and goes down easy.
    
    
    PREDICTED SUMMARY:
    
    good!!!
    
    ACTUAL SUMMARY:
    
    great for preventing cramps
    
    loss=7.69772
    
    Iteration: 40
    Training input sequence length: 23
    Training target outputs sequence length: 3
    
    TEXT:
    no tea flavor at all. just whole brunch of unk flavors. it is not returnable. i wasted unk bucks.
    
    
    PREDICTED SUMMARY:
    
    good!!
    
    ACTUAL SUMMARY:
    
    no tea flavor
    
    loss=10.7192
    
    Iteration: 41
    Training input sequence length: 67
    Training target outputs sequence length: 2
    
    TEXT:
    these taste really good. i have been purchasing a different brand and these are very similar in taste and texture. i agree with the other reviewer regarding ordering in the summer. there is no insulating packaging with ice packs so they will melt in warm weather like all chocolate food items. order in cold weather and buy enough to last!!!
    
    
    PREDICTED SUMMARY:
    
    good!
    
    ACTUAL SUMMARY:
    
    taste great
    
    loss=4.2054
    
    Iteration: 42
    Training input sequence length: 28
    Training target outputs sequence length: 5
    
    TEXT:
    the taste was great, but the berries had melted. may order again in winter. if you order in cold weather you should enjoy flavor.
    
    
    PREDICTED SUMMARY:
    
    great!!!!
    
    ACTUAL SUMMARY:
    
    order only in cold weather
    
    loss=9.69668
    
    Iteration: 43
    Training input sequence length: 39
    Training target outputs sequence length: 4
    
    TEXT:
    i know i can not make tea this good. granted, i am not from the south but i know i have never enjoyed tea that was this sweet without being too sweet. it tastes crisp.
    
    
    PREDICTED SUMMARY:
    
    great!!!
    
    ACTUAL SUMMARY:
    
    this is the best
    
    loss=6.75685
    
    Iteration: 44
    Training input sequence length: 41
    Training target outputs sequence length: 2
    
    TEXT:
    this peppermint stick is delicious and fun to eat. my dad got me one for christmas because he remembered me having a similar one when i was a little girl. i 'm 30 now and i love it!
    
    
    PREDICTED SUMMARY:
    
    great great
    
    ACTUAL SUMMARY:
    
    delicious!
    
    loss=4.01642
    
    Iteration: 45
    Training input sequence length: 29
    Training target outputs sequence length: 1
    
    TEXT:
    great gift for all ages! i purchased these giant canes before and the recipients loved them so much, they kept them and would not eat them.
    
    
    PREDICTED SUMMARY:
    
    great
    
    ACTUAL SUMMARY:
    
    great
    
    loss=1.70116
    
    Iteration: 46
    Training input sequence length: 77
    Training target outputs sequence length: 4
    
    TEXT:
    awesome dog food. however, when given to my `` boston '', who has severe reactions to some food ingredients; his itching increased to violent jumping out of bed at night, scratching. as soon as i changed to a different formula, the scratching stopped. so glad natural balance has other choices. i guess you have to try each, until you find what 's best for your pet.
    
    
    PREDICTED SUMMARY:
    
    great great great great
    
    ACTUAL SUMMARY:
    
    increased my dogs itching
    
    loss=10.4727
    
    Iteration: 47
    Training input sequence length: 56
    Training target outputs sequence length: 3
    
    TEXT:
    we have three dogs and all of them love this food! we bought it specifically for one of our dogs who has food allergies and it works great for him, no more hot spots or tummy unk< br/> i love that it ships right to our door with free shipping.
    
    
    PREDICTED SUMMARY:
    
    great great great
    
    ACTUAL SUMMARY:
    
    great food!
    
    loss=3.04694
    
    Iteration: 48
    Training input sequence length: 42
    Training target outputs sequence length: 5
    
    TEXT:
    my unk mix has ibs. our vet recommended a limited ingredient food. this has really helped her symptoms and she likes it. i will always buy it from amazon ... it 's$ 10 cheaper and free shipping!
    
    
    PREDICTED SUMMARY:
    
    great great great great great
    
    ACTUAL SUMMARY:
    
    great for stomach problems!
    
    loss=6.43315
    
    Iteration: 49
    Training input sequence length: 58
    Training target outputs sequence length: 2
    
    TEXT:
    great food! i love the idea of one food for all ages& breeds. t 's a real convenience as well as a really good product. my 3 dogs eat less, have almost no gas, their poop is regular and a perfect consistency, what else can a mom ask for!!
    
    
    PREDICTED SUMMARY:
    
    great great
    
    ACTUAL SUMMARY:
    
    great food
    
    loss=3.02226
    
    Iteration: 50
    Training input sequence length: 24
    Training target outputs sequence length: 3
    
    TEXT:
    this is great dog food, my dog has severs allergies and this brand is the only one that we can feed him.
    
    
    PREDICTED SUMMARY:
    
    great!!
    
    ACTUAL SUMMARY:
    
    great dog food
    
    loss=3.86777
    
    Iteration: 51
    Training input sequence length: 43
    Training target outputs sequence length: 4
    
    TEXT:
    this food is great- all ages dogs. i have a 3 year old and a puppy. they are both so soft and hardly ever get sick. the food is good especially when you have amazon prime shipping:)
    
    
    PREDICTED SUMMARY:
    
    great great great!
    
    ACTUAL SUMMARY:
    
    mmmmm mmmmm good.
    
    loss=9.50443
    
    Iteration: 52
    Training input sequence length: 28
    Training target outputs sequence length: 2
    
    TEXT:
    this is the same food we get at pet store. but it 's delivered to my door! and for the same price or slightly less.
    
    
    PREDICTED SUMMARY:
    
    great!
    
    ACTUAL SUMMARY:
    
    so convenient
    
    loss=13.1772
    
    Iteration: 53
    Training input sequence length: 67
    Training target outputs sequence length: 4
    
    TEXT:
    i 've been very pleased with the natural balance dog food. our dogs have had issues with other dog foods in the past and i had someone recommend natural balance grain free since it is possible they were allergic to grains. since switching i have n't had any issues. it is also helpful that have have different kibble size for unk sized dogs.
    
    
    PREDICTED SUMMARY:
    
    great great great great
    
    ACTUAL SUMMARY:
    
    good healthy dog food
    
    loss=4.78182
    
    Iteration: 54
    Training input sequence length: 43
    Training target outputs sequence length: 1
    
    TEXT:
    i fed this to my golden retriever and he hated it. he would n't eat it, and when he did, it gave him terrible diarrhea. we will not be buying this again. it 's also super expensive.
    
    
    PREDICTED SUMMARY:
    
    great
    
    ACTUAL SUMMARY:
    
    bad
    
    loss=14.4075
    
    Iteration: 55
    Training input sequence length: 24
    Training target outputs sequence length: 2
    
    TEXT:
    arrived slightly thawed. my parents would n't accept it. however, the company was very helpful and issued a full refund.
    
    
    PREDICTED SUMMARY:
    
    great!
    
    ACTUAL SUMMARY:
    
    great support
    
    loss=6.12445
    
    Iteration: 56
    Training input sequence length: 56
    Training target outputs sequence length: 2
    
    TEXT:
    the crust on these tarts are perfect. my husband loves these, but i 'm not so crazy about them. they are just too unk for my taste. i 'll eat the crust and hubby takes my filling. my kids think they 're great, so maybe it 's just me.
    
    
    PREDICTED SUMMARY:
    
    great great
    
    ACTUAL SUMMARY:
    
    tart!
    
    loss=7.88888
    
    Iteration: 57
    Training input sequence length: 39
    Training target outputs sequence length: 3
    
    TEXT:
    these are absolutely unk! my husband and i both love them, however, as another customer put it, they are expensive to ship! the cost of shipping is more than the tartlets themselves are!
    
    
    PREDICTED SUMMARY:
    
    great food!
    
    ACTUAL SUMMARY:
    
    omaha apple tartlets
    
    loss=12.924
    
    Iteration: 58
    Training input sequence length: 37
    Training target outputs sequence length: 3
    
    TEXT:
    what a nice alternative to an apple pie. love the fact there was no slicing and dicing. easy to prepare. i also loved the fact that you can make them fresh whenever needed.
    
    
    PREDICTED SUMMARY:
    
    great food food
    
    ACTUAL SUMMARY:
    
    loved these tartlets
    
    loss=10.2792
    
    Iteration: 59
    Training input sequence length: 58
    Training target outputs sequence length: 2
    
    TEXT:
    i like creme brulee. i loved that these were so easy. just sprinkle on the sugar that came with and broil. they look amazing and taste great. my guess thought i really went out of the way for them when really it took all of 5 minutes. i will be ordering more!
    
    
    PREDICTED SUMMARY:
    
    great food
    
    ACTUAL SUMMARY:
    
    the best
    
    loss=5.38269
    
    Iteration: 60
    Training input sequence length: 63
    Training target outputs sequence length: 2
    
    TEXT:
    i love asparagus. up until very recently, i had never had pickled asparagus. oh my goodness, when a friend introduced me to this exact brand, i could n't believe how great stuff tasted. i loved it so much i bought the six pack. i 've got 2 jars left. gon na need more!!
    
    
    PREDICTED SUMMARY:
    
    great food
    
    ACTUAL SUMMARY:
    
    asparagus bliss
    
    loss=11.9187
    
    Iteration: 61
    Training input sequence length: 33
    Training target outputs sequence length: 5
    
    TEXT:
    i was unk in the flavor and texture of this mix. i usually like most of the low carb things i have tried, but was unk in this specific one.
    
    
    PREDICTED SUMMARY:
    
    great food food food food
    
    ACTUAL SUMMARY:
    
    low carb angel food puffs
    
    loss=9.81204
    
    Iteration: 62
    Training input sequence length: 60
    Training target outputs sequence length: 2
    
    TEXT:
    i have been drinking this tea for a long time now. i used to have to purchase it at a doctor 's office because it was n't available elsewhere. i 'm so glad that i can buy it now from amazon.com. i drink this tea throughout the day like other folks drink coffee. wonderful taste.
    
    
    PREDICTED SUMMARY:
    
    great food
    
    ACTUAL SUMMARY:
    
    delicious tea
    
    loss=4.55072
    
    Iteration: 63
    Training input sequence length: 65
    Training target outputs sequence length: 2
    
    TEXT:
    i love, love this green tea. it is very hard to find in our area and some places on the internet charge a big price and i usually do n't get as many boxes as i did with this merchant. i will definitely order from this seller again!! thanks!! i depend on my green tea fix everyday!
    
    
    PREDICTED SUMMARY:
    
    great food
    
    ACTUAL SUMMARY:
    
    tea review
    
    loss=8.26901
    
    Iteration: 64
    Training input sequence length: 26
    Training target outputs sequence length: 2
    
    TEXT:
    i love this tea. it helps curb my eating during the day. my mom and i have given it all friends to try.
    
    
    PREDICTED SUMMARY:
    
    food.
    
    ACTUAL SUMMARY:
    
    wonderful tea
    
    loss=5.67268
    
    Iteration: 65
    Training input sequence length: 47
    Training target outputs sequence length: 2
    
    TEXT:
    i 'm italian and i lived in italy for years. i used to buy these cookies for my everyday breakfast with an italian espresso. i could n't find them anywhere here in the bay area, so it 's great to have them again.
    
    
    PREDICTED SUMMARY:
    
    food tea
    
    ACTUAL SUMMARY:
    
    great cookies
    
    loss=7.8061
    
    Iteration: 66
    Training input sequence length: 79
    Training target outputs sequence length: 3
    
    TEXT:
    i have done a lot of research to find the best food for my cat, and this is an excellent food. that is also according to my holistic veterinarian. they put probiotics on the kibble as the last step, which is very important to me. the best thing is that my cat loved it immediately and i had to stop mixing it with the old food because she only would eat holistic select.
    
    
    PREDICTED SUMMARY:
    
    delicious tea tea
    
    ACTUAL SUMMARY:
    
    great food.
    
    loss=3.78115
    
    Iteration: 67
    Training input sequence length: 65
    Training target outputs sequence length: 7
    
    TEXT:
    one of my cats is allergic to fish and beef. this formula is one of the few she can eat, and it has much better ingredients than the prescription diets available at the vet. both of my kitties are very active, have soft shiny fur, and neither are fat. dry food reduces tartar buildup on teeth, also.
    
    
    PREDICTED SUMMARY:
    
    food tea tea tea tea tea tea
    
    ACTUAL SUMMARY:
    
    wonderful food- perfect for allergic kitties
    
    loss=8.03118
    
    Iteration: 68
    Training input sequence length: 51
    Training target outputs sequence length: 4
    
    TEXT:
    our cats thrive extremely well on this dry cat food. they definitely have much less hair ball throw ups and their fur is great. they are fit and not over weight. this vendor ships extremely fast. is one of the top amazon suppliers in our book!
    
    
    PREDICTED SUMMARY:
    
    food tea tea tea
    
    ACTUAL SUMMARY:
    
    holistic select cat food
    
    loss=9.35138
    
    Iteration: 69
    Training input sequence length: 45
    Training target outputs sequence length: 3
    
    TEXT:
    i 've been eating ramen noodles since i was a little kid, and i 've never found a better flavor than hot& spicy chicken! it is n't hot at all to a unk like me, but it sure is good!
    
    
    PREDICTED SUMMARY:
    
    food tea tea
    
    ACTUAL SUMMARY:
    
    my favorite ramen
    
    loss=7.70701
    
    Iteration: 70
    Training input sequence length: 54
    Training target outputs sequence length: 3
    
    TEXT:
    i love spicy ramen, but for whatever reasons this thing burns my stomach badly and the burning sensation does n't go away for like 3 hours! not sure if that is healthy or not .... and you can buy this at walmart for$ 0.28, way cheaper than amazon.
    
    
    PREDICTED SUMMARY:
    
    food tea tea
    
    ACTUAL SUMMARY:
    
    it burns!
    
    loss=7.53228
    
    Iteration: 71
    Training input sequence length: 42
    Training target outputs sequence length: 6
    
    TEXT:
    always being a fan of ramen as a quick and easy meal, finding it on amazon for a decent price and having it delivered to your door by the case is an amazing situation for anyone to find themselves in.
    
    
    PREDICTED SUMMARY:
    
    food tea tea tea tea tea
    
    ACTUAL SUMMARY:
    
    amazing to the last bite.
    
    loss=9.26448
    
    Iteration: 72
    Training input sequence length: 56
    Training target outputs sequence length: 3
    
    TEXT:
    i must be a bit of a wuss, because this soup tastes to me how i imagine fire might taste. typically i like spicy food if it has a good flavor. i do n't find this to be the case with this soup. any flavor is killed off by the burn.
    
    
    PREDICTED SUMMARY:
    
    food tea tea
    
    ACTUAL SUMMARY:
    
    not for me
    
    loss=7.93951
    
    Iteration: 73
    Training input sequence length: 50
    Training target outputs sequence length: 3
    
    TEXT:
    i really loved the spicy flavor these had. i found myself liking the broth more than the noodles which is usually the opposite. if you are n't used to the heat this might bother you and if you like hot hot foods this might not be enough.
    
    
    PREDICTED SUMMARY:
    
    food tea tea
    
    ACTUAL SUMMARY:
    
    great spicy flavor
    
    loss=6.68682
    
    Iteration: 74
    Training input sequence length: 78
    Training target outputs sequence length: 5
    
    TEXT:
    got these on sale for roughly 25 cents per cup, which is half the price of my local grocery stores, plus they rarely stock the spicy flavors. these things are a great snack for my office where time is constantly crunched and sometimes you ca n't escape for a real meal. this is one of my favorite flavors of instant lunch and will be back to buy every time it goes on sale.
    
    
    PREDICTED SUMMARY:
    
    food food tea tea tea
    
    ACTUAL SUMMARY:
    
    great value and convenient ramen
    
    loss=7.53814
    
    Iteration: 75
    Training input sequence length: 22
    Training target outputs sequence length: 2
    
    TEXT:
    i have bought allot of different flavors and this happens to be one of my favorites and will be getting more soon
    
    
    PREDICTED SUMMARY:
    
    food tea
    
    ACTUAL SUMMARY:
    
    great flavor
    
    loss=3.15631
    
    Iteration: 76
    Training input sequence length: 74
    Training target outputs sequence length: 5
    
    TEXT:
    the best investment i 've ever made for ginger. it 's unbelievable! it 's fibrous like the real ginger, has that spicy kick to it, but it 's perfect with the sugar- calms it down. it 's very worth the$ 40 for unk of it! i 'll be getting more soon- i use these as a topper for my ginger cupcakes and cookies:)
    
    
    PREDICTED SUMMARY:
    
    food tea tea tea tea
    
    ACTUAL SUMMARY:
    
    tastes awesome& looks beautiful
    
    loss=11.1367
    
    Iteration: 77
    Training input sequence length: 33
    Training target outputs sequence length: 2
    
    TEXT:
    delicious. i can not get australian ginger where i live. this compares favorably to australian ginger i 've purchased in other cities. now i can enjoy it without traveling.
    
    
    PREDICTED SUMMARY:
    
    great tea
    
    ACTUAL SUMMARY:
    
    happy face
    
    loss=13.2036
    
    Iteration: 78
    Training input sequence length: 30
    Training target outputs sequence length: 4
    
    TEXT:
    i keep trying other brands .... cheaper brands. stupid me! this ginger is soooo worth the money. tender, moist and never a let down.
    
    
    PREDICTED SUMMARY:
    
    great tea tea tea
    
    ACTUAL SUMMARY:
    
    simply the best!
    
    loss=6.33492
    
    Iteration: 79
    Training input sequence length: 52
    Training target outputs sequence length: 2
    
    TEXT:
    i bought this for our office to give people something sweet to snack on. because it 's bite size it 's easier for people to grab a couple a pieces rather than an entire licorice stick. my only complaint is that one of the bags broke open in shipping.
    
    
    PREDICTED SUMMARY:
    
    great flavor
    
    ACTUAL SUMMARY:
    
    nice snack
    
    loss=13.0985
    
    Iteration: 80
    Training input sequence length: 59
    Training target outputs sequence length: 2
    
    TEXT:
    twizzlers brand licorice is much better than that other well known unk< br/> if you can get these for$ 2 to$ 2.50 a package with free unk it 's a good unk< br/> the black and cherry have good taste; but the strawberry taste was too delicate and barely there
    
    
    PREDICTED SUMMARY:
    
    great flavor
    
    ACTUAL SUMMARY:
    
    good licorice
    
    loss=8.12259
    
    Iteration: 81
    Training input sequence length: 36
    Training target outputs sequence length: 5
    
    TEXT:
    this is one of the best salsas that i have found in a long time but stay away from the variety pack. the other two that come with it are not worth your money.
    
    
    PREDICTED SUMMARY:
    
    great flavor flavor flavor flavor
    
    ACTUAL SUMMARY:
    
    love the salsa!!
    
    loss=6.54998
    
    Iteration: 82
    Training input sequence length: 44
    Training target outputs sequence length: 2
    
    TEXT:
    these remind me of dog treats i made once using pumpkin and cinnamon. they 're kind of bland and not my favorite back to nature product. but my unk really loves them so that 's where the three stars come from.
    
    
    PREDICTED SUMMARY:
    
    great flavor
    
    ACTUAL SUMMARY:
    
    unk ...
    
    loss=5.97179
    
    Iteration: 83
    Training input sequence length: 39
    Training target outputs sequence length: 2
    
    TEXT:
    this is the best cornmeal. i made regular cornbread and hot water cornbread with this meal and both were outstanding. also fried some oysters with this meal, it gave them a great texture and unk.
    
    
    PREDICTED SUMMARY:
    
    great flavor
    
    ACTUAL SUMMARY:
    
    awesome cornmeal
    
    loss=8.57024
    
    Iteration: 84
    Training input sequence length: 64
    Training target outputs sequence length: 3
    
    TEXT:
    this is a fabulous marinade! i love to use it for chicken, either baked in the oven or on the grill. this has enough flavor& flair, i 've even used it for dinner parties, only to receive rave reviews from my guests!! definitely worth the price! super cheap and super easy! love it!
    
    
    PREDICTED SUMMARY:
    
    great flavor flavor
    
    ACTUAL SUMMARY:
    
    great marinade!
    
    loss=5.96674
    
    Iteration: 85
    Training input sequence length: 29
    Training target outputs sequence length: 2
    
    TEXT:
    works with chicken fish beef or pork. fast easy and makes it taste excellent. plus buying in bulk is more than 50% off from box stores
    
    
    PREDICTED SUMMARY:
    
    great flavor
    
    ACTUAL SUMMARY:
    
    awesome stuff
    
    loss=5.27093
    
    Iteration: 86
    Training input sequence length: 25
    Training target outputs sequence length: 2
    
    TEXT:
    got this for my brother who is on jorge cruise diet and decided to try one for myself. it actually tastes pretty good.
    
    
    PREDICTED SUMMARY:
    
    great flavor
    
    ACTUAL SUMMARY:
    
    tastes good
    
    loss=8.59205
    
    Iteration: 87
    Training input sequence length: 54
    Training target outputs sequence length: 3
    
    TEXT:
    these singles sell for$ 2.50-$ 3.36 at the store for 1 box of 24 singles. i 'm not sure why amazon is selling it for$ 9.99 for a box of 24 singles. hazelnut coffee creamer is my favorite, but truly this is not a good buy.
    
    
    PREDICTED SUMMARY:
    
    great flavor flavor
    
    ACTUAL SUMMARY:
    
    rip off price
    
    loss=10.4594
    
    Iteration: 88
    Training input sequence length: 66
    Training target outputs sequence length: 1
    
    TEXT:
    awesome!!! such a yummy flavor i got it as a healthy alternative to the desserts we normally eat and i am so glad that i did there are so many things you can do with jello desserts and still have them taste good and be good for you. i will


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-19-d577097b223c> in <module>()
         37                 flag = 0
         38                 for vec in train_texts[i]:
    ---> 39                     if vec2word(vec) in string.punctuation or flag==0:
         40                         print(str(vec2word(vec)),end='')
         41                     else:


    <ipython-input-2-66857750bc0f> in vec2word(vec)
         24 def vec2word(vec):   # converts a given vector representation into the represented word
         25     for x in xrange(0, len(embedding)):
    ---> 26         if np.array_equal(embedding[x],np.asarray(vec)):
         27             return vocab[x]
         28     return vec2word(np_nearest_neighbour(np.asarray(vec)))


    /usr/local/lib/python2.7/dist-packages/numpy/core/numeric.pyc in array_equal(a1, a2)
       2602     if a1.shape != a2.shape:
       2603         return False
    -> 2604     return bool(asarray(a1 == a2).all())
       2605 
       2606 


    KeyboardInterrupt: 
