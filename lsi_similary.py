#!/usr/bin/env python
# encoding: utf-8
'''
@author: fangbing
@contact: fangbing@cvte.com
@file: lsi_similary.py
@time: 2018/8/22 15:19
@desc:相似题推送
'''
from gensim import corpora, models,similarities
import numpy as np
import sys
import pymysql
from scipy.sparse import csr_matrix
import re
import jieba
import operator
import pickle
from sklearn.cluster import AffinityPropagation
from sklearn import metrics

cnx = pymysql.connect(host='****', port=3306, user='fangbing', passwd='FB2018@cvte', db='yitiku2', charset='utf8')
cur = cnx.cursor()
print ('start...')

##加载停用词表；
def load_stop_words(stop_words_path):
    with open(stop_words_path, 'r', encoding='utf-8') as f:
        stop_words = f.readlines()
    f.close()
    for i in range(len(stop_words)):
        stop_words[i] = stop_words[i].replace('\n', '')
    return stop_words

def cleanQuestion(question):
    stop_words = load_stop_words('CNstopwords.txt')
    dr = re.compile(r'<[^>]+>', re.S)
    question = dr.sub('', question)
    newquestion=question.replace('\t', '').replace('\xa0', '').replace('\u3000', '').replace('\ue5e5', '').replace('\ue003', '')
    newquestion=newquestion.replace('□','').replace('{','').replace('}','').replace('（','').replace('）','').replace('[','').replace(']','').replace('(','').replace(')','')
    newquestion=re.sub('[()【】{}“”！，。？、~@#￥%……&*（）]+', '',newquestion)
    cut = jieba.cut(newquestion)
    cut = ' '.join(cut)
    ct = cut.split(' ')
	newct=[]
    for t in ct:  ##去除停用词表中的词；
          if t not in stop_words:
            newct.append(t)
    return newct

##question清洗，切词并去除停用词;
def clean_question():
    select_sql = 'select uid, question_text,question_diff,question_type from t_question limit 1000'
    cur.execute(select_sql)
    questions = cur.fetchall()
    uids = []
    questionText = []
    question_diffs=[]
    question_types=[]
    for q in range(len(questions)):
        uids.append(questions[q][0])
        s = questions[q][1]
        question_diffs.append(questions[q][2])
        question_types.append(questions[q][3])
        ct=cleanQuestion(s)
        questionText.append(ct)
    print ('finish clean_question...')
    return uids, questionText,question_diffs,question_types


def get_lsi_vector(lsi_model_lsi,lsi_model_index,lsi_model_dictionary,questionText):
    word_seg = questionText

    dictionary = corpora.Dictionary(word_seg)  ##得到词典
    corpus = [dictionary.doc2bow(text) for text in word_seg]   ##统计每篇文章中每个词出现的次数:[(词编号id,次数number)]
    pickle.dump(dictionary, open(lsi_model_dictionary, 'wb'))
    
    print('dictionary prepared!')
    ##接下来四行得到lsi向量；
    tfidf = models.TfidfModel(corpus=corpus, dictionary=dictionary)
    corpus_tfidf = tfidf[corpus]
    print('tfidf prepared!')
    lsi_model = models.LsiModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=300)
    corpus_lsi = lsi_model[corpus_tfidf]
    print('lsi prepared!')
    index = similarities.MatrixSimilarity(corpus_lsi)
    print('index prepared!')
    lsi_model.save(lsi_model_lsi)
    index.save(lsi_model_index)
    print('finish train!')

    try:
        information = {'corpus_lsi': list(corpus_lsi)}
        information_path = open('./corpus_lsi.pkl', 'wb')
        pickle.dump(information, information_path)
        information_path.close()
        print('save corlus_lsi success!')
    except Exception as e:
        print(str(e))
        print('save corlus_lsi fail!')

    simid = -np.array(index)
    print(simid.shape)
    print(simid[0])
    print(len(simid))

    p = np.median(simid)  ##重新设置对角线的值,设为中值
    for i in range(len(simid)):
        simid[i][i] = p

    ap = AffinityPropagation(preference=-400, damping=0.6).fit(simid)
    cluster_centers_indices = ap.cluster_centers_indices_  # 预测出的中心点的索引，如[123,23,34]
    print(cluster_centers_indices)

    labels = ap.labels_  # 预测出的每个数据的类别标签,labels是一个NumPy数组
    print(len(labels))
    n_clusters_ = len(cluster_centers_indices)  # 预测聚类中心的个数
    print('预测的聚类中心个数：%d' % n_clusters_)
    print('轮廓系数：%0.3f' % metrics.silhouette_score(simid, labels, metric='sqeuclidean'))

    return list(corpus_lsi)
    

#     similarities.MatrixSimilarity类仅仅适合能将所有的向量都在内存中的情况。
# 例如，如果一个百万文档级的语料库使用该类，可能需要2G内存与256维LSI空间。
# 如果没有足够的内存，你可以使用similarities.Similarity类。该类的操作只需要固定大小的内存，
# 因为他将索引切分为多个文件（称为碎片）存储到硬盘上了。它实际上使用了similarities.MatrixSimilarity
# 和similarities.SparseMatrixSimilarity两个类，因此它也是比较快的，虽然看起来更加复杂了。

def lsi2matrix(corpus_lsi):
    data = []
    rows = []
    cols = []
    line_count = 0
    for line in corpus_lsi:  # lsi_corpus 是之前由gensim生成的lsi向量
        for elem in line:
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
        line_count += 1
    lsi_sparse_matrix = csr_matrix((data,(rows,cols))) # 稀疏向量
    lsi_matrix = lsi_sparse_matrix.toarray()  # 密集向量
    index = similarities.MatrixSimilarity(lsi_matrix)
    return lsi_matrix,index

###保存和加载相关信息；
def save_information(information_save_path, uids,question_diffs,question_types,corpus_lsi):
    information = {'uids':uids,'question_diff':question_diffs,'question_type':question_types,'corpus_lsi':corpus_lsi}
    information_path = open(information_save_path, 'wb')
    pickle.dump(information, information_path)
    information_path.close()

###保存和加载相关信息；
def save_information2(information_save_path, uids,question_diffs,question_types,question_text):
    information = {'uids':uids,'question_diff':question_diffs,'question_type':question_types,'question_text':question_text}
    information_path = open(information_save_path, 'wb')
    pickle.dump(information, information_path)
    information_path.close()
    print('the first question save success!')


def trainData():
    all_questions_save_path = './all_questions.pkl'
    lsi_model_index_savePath='./lsi_index.index'
    lsi_model_lsi_savePath='./lsi_model.lsi'
    lsi_model_dictionary = './lsi_dictionary.pkl'
    ##处理zujuan库中的question信息；
    uids, questionText,question_diffs,question_types = clean_question()
    save_information2('./questions.pkl',uids,question_diffs,question_types,questionText)
    ##lsi模型；
    questions=pickle.load(open('./questions.pkl','rb'))
    questionText=questions['question_text']
    #print(len(questionText))
    corpus_lsi=get_lsi_vector(lsi_model_lsi_savePath,lsi_model_index_savePath,lsi_model_dictionary,questionText)
    try:
      save_information(all_questions_save_path, uids,question_diffs,question_types,corpus_lsi)
      print('save success!!')
    except Exception as e:
      print(str(e))


class test(object):
    def __init__(self):
        self.dictionary = pickle.load(open('./lsi_dictionary.pkl','rb'))
        self.lsi_model=models.LsiModel.load('lsi_model.lsi')
        self.lsi_model_index=similarities.MatrixSimilarity.load('lsi_index.index')

        self.allQuestions=pickle.load(open('./questions.pkl','rb'))
        self.all_uids=self.allQuestions['uids']
        self.all_question_diff=self.allQuestions['question_diff']
        self.all_question_type=self.allQuestions['question_type']
        self.question_text=self.allQuestions['question_text']
        print('init success!!')
    '''
    def request(self,q_uid):
        qid_index = self.all_uids.index(q_uid)
        question_lsi = self.corpus_lsi[qid_index]
        sims = self.lsi_model_index[question_lsi]

        sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])

        idx_set = [i[0] for i in sort_sims[:5]]
        sim_value = [i[1] for i in sort_sims[:5]]
        print('idx_set==', idx_set)
        print('sim_value==', sim_value)
        similarUids=self.getsimiQuestion(idx_set)
        return similarUids
    '''
    def requestByQuestion(self,question):
            query_ques=cleanQuestion(question)
            print('quert==:',query_ques)
            query_bow = self.dictionary.doc2bow(query_ques)
            query_lsi = self.lsi_model[query_bow]
            sims = self.lsi_model_index[query_lsi]

            sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])

            idx_set = [i[0] for i in sort_sims[:10]]
            sim_value = [i[1] for i in sort_sims[:10]]

            print('idx_set==', idx_set)
            print('sim_value==',sim_value)
            results=self.getPatchQuestion(idx_set,sim_value)
            return results

    def getPatchQuestion(self,idx_set,sim_value):
        uidx=[]
        uidx_sim={}
        for idx, sim in zip(idx_set, sim_value):
            uid=self.all_uids[idx]
            uidx.append(uid)
            uidx_sim[uid]=sim

        sql='select uid,question_text from t_question where uuid in %s'%str(tuple(uidx))
        cur.execute(sql)
        rss=cur.fetchall()
        uid_q={}
        for rs in rss:
            uid_q[rs[0]]=uid_q[rs[1]]
        results=[]
        for uidIndex in uid_q.keys():
            results.append(uidIndex+'_'+uid_q[uidIndex]+'_'+str(uidx_sim[uidIndex]))
        return results


    def getsimiQuestion(self,idx_set):
        similarUids = []
        for i in idx_set:
            simi_quid=self.all_uids[i]
            similarUids.append(simi_quid)
        return similarUids


if __name__=='__main__':
    trainData()
    ts=test()
    #q='<div>与北京时间相比，巴黎时间晚7小时，那么北京时间12点时，巴黎时间是（）</div>'
    #results=test.requestByQuestion(q)
    #print(results)
    while True:
         text = input('请输入问题：')
         results=ts.requestByQuestion(text)
         print(results)
