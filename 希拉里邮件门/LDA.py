#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import re

from gensim import corpora, models, similarities
import gensim

"""第一步：用正则表达式清洗数据，并去除停用词"""
df = pd.read_csv("HillaryEmails.csv")
# 原邮件数据中有很多Nan的值，直接扔了。
df = df[['Id','ExtractedBodyText']].dropna()

# 用正则表达式清洗数据
def clean_email_text(text):
    text = text.replace('\n'," ")                       # 新行，我们是不需要的
    text = re.sub(r"-", " ", text)                      # 把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text = re.sub(r"\d+/\d+/\d+", "", text)             # 日期，对主体模型没什么意义
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text)  # 时间，没意义
    text = re.sub(r"[\w]+@[\.\w]+", "", text)           # 邮件地址，没意义
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text)    # 网址，没意义
    
    # 以防还有其他除了单词以外的特殊字符（数字）等等，我们把特殊字符过滤掉
    # 只留下字母和空格
    # 再把单个字母去掉，留下单词
    pure_text = ''
    for letter in text:
        if letter.isalpha() or letter==' ':
            pure_text += letter
            
    text = ' '.join(word for word in pure_text.split() if len(word)>1)
    return text

docs = df['ExtractedBodyText']
docs = docs.apply(lambda s: clean_email_text(s))  
# 得到所有邮件的内容
doclist = docs.values
print("一共有",len(doclist),"封邮件。\n")
print("第一封邮件经过清洗后的内容为: \n",doclist[:1],'\n')
 
# 去除停用词，处理成gensim需要的输入格式
stopwords = [word.strip() for word in open('./stopwords.txt','r').readlines()]
weeks = [monday,mon,tuesday,tues,wednesday,wed,thursday,thur,friday,fri,saturday,sat,sunday,sun]
months = [jan,january,feb,february,mar,march,apr,april,may,jun,june,jul,july,aug,august,sept,september,oct,october,nov,november,dec,december]
stoplist = stopwords+weeks+months
texts = [[word for word in doc.lower().split() if word not in stoplist] for doc in doclist]
print("第一封邮件去除停用词并处理成gensim需要的格式为：\n",texts[0],'\n')

"""第二步：构建字典，将文本ID化"""
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
# 将每一篇邮件ID化
print("第一封邮件ID化后的结果为：\n",corpus[0],'\n')

"""第三步：训练LDA模型"""
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
# 第10个主题的单词分布，取权重最高的前10个词
print(lda.print_topic(9, topn=10))
# 所有主题的单词分布
print(lda.print_topics(num_topics=10, num_words=10))


"""第四步：查看某封邮件所属的主题"""
print("第一封邮件的大致内容为：\n",texts[0],'\n')
topic = lda.get_document_topics(corpus[0])
print("第一封邮件的主题分布为：\n",topic,'\n')

# 希拉里发的两条推特
# 给大伙翻译一下这两句：
# 这是选举的一天!数以百万计的美国人投了希拉里的票。加入他们吧，确定你投给谁。
# 希望今天每个人都能度过一个安乐的感恩节，和家人朋友共度美好时光——来自希拉里的问候。
twitter = ["It's Election Day! Millions of Americans have cast their votes for Hillary—join them and confirm where you vote ",
       "Hoping everyone has a safe & Happy Thanksgiving today, & quality time with family & friends. -H"]

text_twitter = [clean_email_text(s) for s in twitter]
text_twitter = [[word for word in text.lower().split() if word not in stoplist] for text in text_twitter]
corpus_twitter = [dictionary.doc2bow(text) for text in text_twitter]
topics_twitter = lda.get_document_topics(corpus_twitter)
print("这两条推特的主题分布分别为：\n",topics_twitter[0] ,'\n',topics_twitter[1])

