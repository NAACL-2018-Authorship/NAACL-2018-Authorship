import numpy as np
import json

def get_data(F):
    f = open(F,"r")
    dic = json.load(f)
    f.close()
    return dic

class Data():
    def __init__(self,
            data,
            lda,
            author_dic,
            char_dic,
            word_dic,
            word_embedding_dic
            ):
        self.pos = 0
        self.cnt = len(data)
        print(self.cnt)
        author_num = author_dic["author_num"]
        author_dic = author_dic["author2it"]
        word_num = word_dic["word_num"]
        word_dic = word_dic["w2it"]
        self.label = []
        self.chars = []
        self.word_bag = []
        self.word_embedding = []
        for key in data.keys():
            #print("in " + key)
            d = data[key]
            tmpl = [0.] * author_num
            tmpl[author_dic[d["author"]]] += 1.
            #print(tmpl)
            self.label.append(tmpl)
            tmpc = []
            tmpw = [0] * word_num
            tmpwe = []
            
            for word in d["text"]:
                if (word in word_dic.keys()):
                    tmpw[word_dic[word]] += 1
                if (word in word_embedding_dic.keys()):
                    tmpwe.append(word_embedding_dic[word])
                else:
                    tmpwe.append([0.] * 300)
                if word in char_dic.keys():
                    tmpcc = [0.] * len(char_dic)
                    tmpcc[char_dic[word]] += 1.
                    tmpc.append([tmpcc])
                else:
                    tmpqwq = []
                    for ch in word:
                        tmpcc = [0.] * len(char_dic)
                        if (ch in char_dic.keys()):
                            tmpcc[char_dic[ch]] += 1.
                        tmpqwq.append(tmpcc)
                    tmpc.append(tmpqwq)
            self.chars.append(tmpc)
            self.word_bag.append(tmpw)
            self.word_embedding.append(tmpwe)
        self.label = np.array(self.label)
        self.label = np.concatenate((self.label,self.label[0:128]),axis = 0)
        self.word_bag = np.array(self.word_bag)
        self.topics = lda.transform(self.word_bag)
        self.topics = np.concatenate((self.topics,self.topics[0:128]),axis = 0)
        self.max_char = None
        self.max_word = None
        self.chars_array = None
        self.word_embedding_array = None
    def next_batch(self,
            n_batch,
            max_char,
            max_word
            ):
        s = self.pos
        t = self.pos + n_batch
        end = False
        if t >= self.cnt:
            end = True
        self.pos = t % self.cnt
        topics = self.topics[s:t,:]
        labels = self.label[s:t,:]
        if max_char != self.max_char:
            self.max_char = max_char
            chars = []
            for text in self.chars:
                j = 0
                i = 0
                tt = []
                tmp = []
                for word in text:
                    tmp = []
                    for char in word:
                        tmp.append(char)
                        i += 1
                        if i >= max_char:
                            break
                            tt.append(tmp)
                            tmp = []
                            i = 0
                            j += 1
                            if j >= max_word:
                                break
                    if j >= max_word:
                        break
                    if i != 0:
                        while i < max_char:
                            tmp.append([0.] * 67)
                            i += 1
                        tt.append(tmp)
                        tmp = []
                        i = 0
                        j += 1
                        if j >= max_word:
                            break
                while j < max_word:
                    while i < max_char:
                        tmp.append([0.] * 67)
                        i += 1
                    tt.append(tmp)
                    tmp = []
                    i = 0
                    j += 1
                chars.append(tt)
                #print(np.array(t).shape)
            self.chars_array = np.array(chars)
            print(self.chars_array.shape)
            self.chars_array = np.concatenate((self.chars_array,self.chars_array[0:128]),axis = 0)
        #print(self.chars_array.shape)

        char = self.chars_array[s:t,:,:,:]
        if max_word != self.max_word:
            self.max_word = max_word
            words = []
            for text in self.word_embedding:
                i = 0
                tmp = []
                for word in text:
                    tmp.append(word)
                    i += 1
                    if i >= max_word:
                        break
                while i < max_word:
                    tmp.append([0.] * 300)
                    i += 1
                words.append(tmp)
            self.word_embedding_array = np.array(words).reshape((-1,max_word,300,1))
            self.word_embedding_array = np.concatenate((self.word_embedding_array,self.word_embedding_array[0:128]),axis = 0)
        words = self.word_embedding_array[s:t,:,:,:]
        return char,words,topics,labels,end

