#!/usr/bin/env python
# encoding: utf-8
'''
@author: fangbing
@contact: fangbing@cvte.com
@file: lsi_similary.py
@time: 2018/8/22 15:19
@desc:几种短文本相似度计算
'''
import jieba
from collections import Counter
import difflib

#
def edit_similar(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)
    taglist = np.zeros((len_str1 + 1, len_str2 + 1))
    for a in range(len_str1):
        taglist[a][0] = a
    for a in range(len_str2):
        taglist[0][a] = a
    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            if (str1[i - 1] == str2[j - 1]):
                temp = 0
            else:
                temp = 1
            taglist[i][j] = min(taglist[i - 1][j - 1] + temp, taglist[i][j - 1] + 1, taglist[i - 1][j] + 1)
    return 1 - taglist[len_str1][len_str2] / max(len_str1, len_str2)


def cos_sim(str1, str2):
    co_str1 = (Counter(str1))
    co_str2 = (Counter(str2))
    p_str1 = []
    p_str2 = []
    for temp in set(str1 + str2):
        p_str1.append(co_str1[temp])
        p_str2.append(co_str2[temp])
    p_str1 = np.array(p_str1)
    p_str2 = np.array(p_str2)
    return p_str1.dot(p_str2) / (np.sqrt(p_str1.dot(p_str1)) * np.sqrt(p_str2.dot(p_str2)))


def getdiff(text1, text2):
    # 其中的str1，str2并未分词，是两组字符串
    # 方法一 查找最大相同序列
    result = difflib.SequenceMatcher(None, text1, text2).ratio()
    # 分词
    str1 = jieba.lcut(text1)
    str2 = jieba.lcut(text2)
    # 方法二 余弦相识度  
    cos_result = cos_sim(str1, str2)
    # 方法三 编辑距离
    edit_reslut = edit_similar(text1, text2)
    # result= cos_result * 0.3 + 0.7 * diff_result
    return result


if __name__ == '__main__':
    text1 = '点电荷'
    text2 = '第二章，点电荷的场强'
    getdiff(text1, text2)