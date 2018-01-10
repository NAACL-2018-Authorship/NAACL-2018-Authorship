import json
import numpy as np
import random


class data():
    def __init__(self,filename,ldamodel,dics):
        
        with open(filename,'r') as f:
            d = json.load(f)
        with open(ldamodel,'r') as f:
            d2t = json.load(f)
        it2e, it2c, a2t = dics

        self.pos = 0
        self.size = len(d['labels'])
        self.labels = d['labels']
        self.tlabels = np.zeros((self.size,200),dtype = np.float32)
        self.chars = [] 
        self.topics = np.zeros((self.size,200),dtype = np.float32)
        self.word_vec = []
        for i in range(self.size):
            tmpe = []
            tmpc = []
            self.topics[i] = d2t[str(i)]
            self.tlabels[i] = a2t[str(self.labels[i][0])]
            for it in d['texts'][i]:
 

                tmpe.append(it2e[str(it)])
                tmpc.append(it2c[str(it)])
            self.word_vec.append(tmpe)
            self.chars.append(tmpc)
        
        self.ran = list(range(len(d['labels'])))
        random.shuffle(self.ran)
    def next_batch(self,n_batch,l_word,l_char):
        s = self.pos
        t = self.pos + n_batch
        is_end = False
        t = t if t < self.size else self.size
        if t == self.size:
            is_end = True
        self.pos = t
        batch_size = t - s
        author_label = np.zeros((batch_size,19320),dtype = np.int32)
        profile_label = np.zeros((batch_size,2+3+40),dtype = np.int32)
        chars = np.zeros((batch_size,l_word,l_char,50),dtype = np.int32)
        words = np.zeros((batch_size,l_word,300),dtype = np.float32)
#topics = self.topics[s:t]
        topics = np.zeros((batch_size,200),dtype = np.float32)
        tl = np.zeros((batch_size,200),dtype = np.float32)
#tmp1 = []
#tmp = []
        for i in range(batch_size):
            topics[i] = self.topics[self.ran[s+i]]
            tl[i] = self.tlabels[self.ran[s+i]]
            author_label[i,self.labels[self.ran[s+i]][0]] += 1
            profile_label[i,self.labels[self.ran[s+i]][1]] += 1
            profile_label[i,min(2,self.labels[self.ran[s+i]][2] // 10 - 1) + 2] += 1
#            tmp1.append(self.labels[self.ran[s+i]][2])
#            tmp.append(self.labels[self.ran[s+i]][2] // 10 - 1 + 2)
            profile_label[i,self.labels[self.ran[s+i]][3] + 5] += 1
            for j in range(min(len(self.chars[self.ran[s+i]]),l_word)):
                for k in range(min(len(self.chars[self.ran[s+i]][j]),l_char)):
                    chars[i,j,k,self.chars[self.ran[s+i]][j][k]] = 1
                if len(self.word_vec[self.ran[s+i]][j]) != 300:
                    words[i,j] = self.word_vec[self.ran[s+i]][j][-300:]
                else:
                    #print(i,j)
                    #print(len(self.word_vec[i][j]))
                    words[i,j] = self.word_vec[self.ran[s+i]][j]
#print(tmp1)
#print(tmp)
        return is_end,batch_size,chars,words,topics,author_label,profile_label,tl
