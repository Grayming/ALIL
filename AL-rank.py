# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 11:05:31 2017

@author: lming
"""
import os
import sys
import nltk
from keras.utils import to_categorical
from keras.models import load_model
from tagger import CRFTagger
import utilities
import tensorflow as tf
import numpy as np
import time

max_len = 120
VOCABULARY=20000
EPISODES=1
BUDGET=100
k=5


rootdir=os.path.dirname(os.path.abspath(sys.argv[0]))
trainFile=rootdir+'/datadir/conll2002/esp.train'
devFile=rootdir+'/datadir/conll2002/esp.testb'
testFile=rootdir+'/datadir/conll2002/esp.testa'
embFile=rootdir+'/datadir/twelve.table4.multiCCA.window_5+iter_10+size_40+threads_16.normalized'
policyname='policy.eng.e100.k5.h5'
policy = load_model(policyname)


print("Processing data")
train_x, train_y, train_lens = utilities.load_data2labels(trainFile)
test_x, test_y, test_lens = utilities.load_data2labels(testFile)
dev_x, dev_y, dev_lens = utilities.load_data2labels(devFile)

train_sents = utilities.data2sents(train_x, train_y)
test_sents = utilities.data2sents(test_x, test_y)
dev_sents = utilities.data2sents(dev_x, dev_y)
print(len(train_sents))

# build vocabulary
print("Max document length:", max_len)
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_document_length=max_len, min_frequency=1)
# vocab = vocab_processor.vocabulary_ # start from {"<UNK>":0}
train_idx = list(vocab_processor.fit_transform(train_x))
dev_idx = list(vocab_processor.fit_transform(dev_x))
vocab = vocab_processor.vocabulary_
vocab.freeze()
test_idx = list(vocab_processor.fit_transform(test_x))

# build embeddings
vocab = vocab_processor.vocabulary_
vocab_size = VOCABULARY
w2v = utilities.load_crosslingual_embeddings(embFile, vocab, vocab_size)



start_time = time.time()
allf1list=[]  

diranklist=[]
unranklist=[]
#Training the policy
for tau in range(0, EPISODES):
    f1list=[]
    #Shuffle train_sents, split into train_la and train_pool
    indices = np.arange(len(train_sents))
    np.random.shuffle(indices)
    train_la=[]
    train_pool=[]
    train_la_idx=[]
    train_pool_idx=[]
    for i in range(0,len(train_sents)):
        if(i<10):
            train_la.append(train_sents[indices[i]])
            train_la_idx.append(train_idx[indices[i]])
        else:
            train_pool.append(train_sents[indices[i]])
            train_pool_idx.append(train_idx[indices[i]])
    #Initialise the model
    model = CRFTagger('esp.tagger')
    model.train(train_la)

    coin=np.random.rand(1)
    states=[]
    actions=[]
    for t in range(0, BUDGET):
        print('Episode: '+str(tau+1)+' Budget: '+str(t+1))
        #Random get k sample from train_pool and train_pool_idx
        random_pool, random_pool_idx, queryindices= utilities.randomKSamples(train_pool,train_pool_idx,k)
        
        #get the state and action
        state=utilities.getAllState(random_pool,random_pool_idx, model, w2v, 200)
        tempstates= np.expand_dims(state, axis=0)
        action=policy.predict_classes(tempstates, verbose=0)[0]
        
        theindex=queryindices[action]
        #states.append(state)
        #actions.append(action)
        data_new=train_pool[theindex]
        data_new_idx=train_pool_idx[theindex]
        unscore=utilities.getCRFunRank(train_pool, model, data_new)
        discore=utilities.getCRFdiRand(train_la_idx, train_pool_idx, model, data_new_idx)
        unranklist.append(unscore)
        diranklist.append(discore)
        print("Uncertainty rank:"+str(unscore))
        print("Diversity rank:"+str(discore))
        
        train_la.append(train_pool[theindex])
        train_la_idx.append(train_pool_idx[theindex])
        model.train(train_la)
        #delete the selected data point from the pool
        
        del train_pool[theindex]
        del train_pool_idx[theindex]  
        



print('Uncertainty list: ')
print(unranklist)
print('Diversity rank list:')
print(diranklist)
ww=open('Uncertaintyrank.txt','w')
ww.writelines(str(line)+ "\n" for line in unranklist)
ww.close()
uu=open('Diversityrank.txt','w')
uu.writelines(str(line)+ "\n" for line in diranklist)
uu.close()   
print("Cost:--- %s seconds ---" % (time.time() - start_time))



