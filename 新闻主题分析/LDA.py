#!/usr/bin/python
# -*- coding:utf-8 -*-

import jieba,os,re
from gensim import corpora, models, similarities

# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('./stopwords.txt',encoding='UTF-8').readlines()]
    return stopwords

# 对句子进行中文分词
def seg_depart(sentence):
    sentence_depart = jieba.cut(sentence.strip())
    stopwords = stopwordslist()
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
            outstr += word
            outstr += " "
    return outstr

if not os.path.exists('./cnews.train_jieba.txt'):
    # 给出文档路径
    filename = "./cnews.train.txt"
    outfilename = "./cnews.train_jieba.txt"
    inputs = open(filename, 'r', encoding='UTF-8')
    outputs = open(outfilename, 'w', encoding='UTF-8')

    # 将输出结果写入ouputs.txt中
    for line in inputs:
        line = line.split('\t')[1]
        line = re.sub(r'[^\u4e00-\u9fa5]+','',line)
        line_seg = seg_depart(line.strip())
        outputs.write(line_seg.strip() + '\n')
    
    outputs.close()
    inputs.close()
    print("删除停用词和分词成功！！！")
    
fr = open('./cnews.train_jieba.txt', 'r',encoding='utf-8')
train = []
for line in fr.readlines():
    line = [word.strip() for word in line.split(' ')]
    train.append(line)
        
# 接下来就是模型构建的步骤了，首先构建词频矩阵
dictionary = corpora.Dictionary(train)
corpus = [dictionary.doc2bow(text) for text in train]
lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
topic_list = lda.print_topics(10)
print("10个主题的单词分布为：\n")
for topic in topic_list:
    print(topic)
        
# 用来测试的三条新闻，分别为体验、娱乐和科技新闻    
file_test = "./cnews.test.txt"
news_test = open(file_test, 'r', encoding='UTF-8')
            
test = []
         
# 处理成正确的输入格式       
for line in news_test:
    line = line.split('\t')[1]
    line = re.sub(r'[^\u4e00-\u9fa5]+','',line)
    line_seg = seg_depart(line.strip())
    line_seg = [word.strip() for word in line_seg.split(' ')]
    test.append(line_seg)    
            
# 新闻ID化    
corpus_test = [dictionary.doc2bow(text) for text in test]
topics_test = lda.get_document_topics(corpus_test)  
labels = ['体育','娱乐','科技']
for i in range(3):
    print('这条'+labels[i]+'新闻的主题分布为：\n')
    print(topics_test[i],'\n')
        
fr.close()
news_test.close()