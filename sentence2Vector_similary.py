#!/usr/bin/env python
# encoding: utf-8
'''
@author: fangbing
@contact: fangbing@cvte.com
@file: sentence2Vector_similary.py
@time: 2018/8/24 9:07
@desc:
'''
from gensim import matutils
from gensim.models import Word2Vec
import pickle
import scipy
import numpy as np
from gensim import corpora, models
import numpy as np
from sklearn.decomposition import PCA
from typing import List


# ==============词向量求平均===================
def sentenceByWordVectAvg(sentenceList, model, embeddingSize):
    sentenceSet = []
    for sentence in sentenceList:
        # 将所有词向量的woed2vec向量相加到句向量
        sentenceVector = np.zeros(embeddingSize)
        # 计算每个词向量的权重，并将词向量加到句向量
        for word in sentence:
            sentenceVector = np.add(sentenceVector, model[word])
        sentenceVector = np.divide(sentenceVector, len(sentence))
        # 存储句向量
        sentenceSet.append(sentenceVector)
    return sentenceSet


# ===============word2vec词向量+tfidf==================
def sentenceByW2VTfidf(corpus_tfidf, token2id, sentenceList, model, embeddingSize):
    sentenceSet = []
    for i in range(len(sentenceList)):
        # 将所有词向量的woed2vec向量相加到句向量
        sentenceVector = np.zeros(embeddingSize)
        # 计算每个词向量的权重，并将词向量加到句向量
        sentence = sentenceList[i]
        sentence_tfidf = corpus_tfidf[i]
        dict_tfidf = list_dict(sentence_tfidf)
        for word in sentence:
            tifidf_weigth = dict_tfidf.get(str(token2id[word]))
            sentenceVector = np.add(sentenceVector, tifidf_weigth * model[word])
        sentenceVector = np.divide(sentenceVector, len(sentence))
        # 存储句向量
        sentenceSet.append(sentenceVector)
    return sentenceSet


def list_dict(list_data):
    list_data = list(map(lambda x: {str(x[0]): x[1]}, list_data))
    dict_data = {}
    for i in list_data:
        key, = i
        value, = i.values()
        dict_data[key] = value
    return dict_data


# ===============sentence2vec：词向量加权-PCA==================
class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector

    # a sentence, a list of words


class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list

        # return the length of a sentence

    def len(self) -> int:
        return len(self.word_list)

    # convert a list of sentence with word2vec items into a set of sentence vectors


def sentence2vec(wdfs, token2id, sentenceList: List[Sentence], embeddingSize: int, charLen: int, a: float = 1e-3):
    sentenceSet = []
    for sentence in sentenceList:
        sentenceVector = np.zeros(embeddingSize)
        for word in sentence.word_list:
            p = wdfs[token2id[word.text]] / charLen
            a = a / (a + p)
            sentenceVector = np.add(sentenceVector, np.multiply(a, word.vector))
        sentenceVector = np.divide(sentenceVector, sentence.len())
        sentenceSet.append(sentenceVector)
        # caculate the PCA of sentenceSet
    pca = PCA(n_components=embeddingSize)
    pca.fit(np.array(sentenceSet))
    u = pca.components_[0]
    u = np.multiply(u, np.transpose(u))

    # occurs if we have less sentences than embeddings_size
    if len(u) < embeddingSize:
        for i in range(embeddingSize - len(u)):
            u = np.append(u, [0])

            # remove the projections of the average vectors on their first principal component
    # (“common component removal”).
    sentenceVectors = []
    for sentenceVector in sentenceSet:
        sentenceVectors.append(np.subtract(sentenceVector, np.multiply(u, sentenceVector)))
    return sentenceVectors


# 获取训练数据
def gettrainData():
    question_path = r'./shuxueTest/shuxueTrainData.pkl'
    longtextdata1 = pickle.load(open(question_path, 'rb'))
    longtextdata1 = longtextdata1['question_text']
    traind = longtextdata1[:5000]
    traindata = list(map(lambda x: x.split(' '), traind))
    return traindata


def saveIndex(sentence_vecs):
    corpus_len = len(sentence_vecs)
    print(corpus_len)
    index = np.empty(shape=(corpus_len, 200), dtype=np.float32)
    for docno, vector in enumerate(sentence_vecs):
        if isinstance(vector, np.ndarray):
            pass
        elif scipy.sparse.issparse(vector):
            vector = vector.toarray().flatten()
        else:
            vector = matutils.unitvec(matutils.sparse2full(vector, 200))
        index[docno] = vector
    return index


# 计算矩阵与向量余弦相识度
def cosine_Matrix(_matrixA, vecB):
    _matrixA_matrixB = np.dot(_matrixA, vecB.T).T
    _matrixA_norm = np.sqrt(np.multiply(_matrixA, _matrixA).sum(axis=1))
    vecB_norm = np.linalg.norm(vecB)
    return np.divide(_matrixA_matrixB, _matrixA_norm * vecB_norm.transpose())


def trainWordVectAvg():
    traindata = gettrainData()
    dictionary = corpora.Dictionary(traindata)  ##得到词典
    token2id = dictionary.token2id
    charLen = dictionary.num_pos
    corpus = [dictionary.doc2bow(text) for text in traindata]  ##统计每篇文章中每个词出现的次数:[(词编号id,次数number)]
    print('dictionary prepared!')
    tfidf = models.TfidfModel(corpus=corpus, dictionary=dictionary)
    wdfs = tfidf.dfs
    corpus_tfidf = tfidf[corpus]
    model = Word2Vec(traindata, size=200, window=5, min_count=1, workers=4)

    # 词向量求平均得到句向量
    sentence_vecs = sentenceByWordVectAvg(traindata, model, 200)
    # 词向量tfidf加权得到句向量
    sentence_vecs = sentenceByW2VTfidf(corpus_tfidf, token2id, traindata, model, 200)
    # sentence2vec：词向量加权-PCA
    Sentence_list = []
    for td in traindata:
        vecs = []
        for s in td:
            w = Word(s, model[s])
            vecs.append(w)
        sentence = Sentence(vecs)
        Sentence_list.append(sentence)
    sentence_vecs = sentence2vec(wdfs, token2id, Sentence_list, 200, charLen)

    query = sentence_vecs[0]
    print(query)
    index = saveIndex(sentence_vecs)
    query = sentence_vecs[0]
    # 计算相似度
    cosresult = cosine_Matrix(index, query)
    cosresult = cosresult.tolist()
    sort_cosresult = sorted(cosresult)
    print(sort_cosresult)
    for i in sort_cosresult[-8:-1]:
        idx = cosresult.index(i)
        print(i, '===', traindata[idx])
    print(traindata[0])