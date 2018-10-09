# -*- coding: UTF-8 -*-
from numpy import *
from functools import reduce

def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],				#切分的词条
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1]   																#类别标签向量，1代表侮辱性词汇，0代表不是
	return postingList,classVec																#返回实验样本切分的词条和类别标签向量

def createVocabList(dataSet):
	vocabSet = set([])  					#创建一个空的不重复列表
	for document in dataSet:
		vocabSet = vocabSet | set(document) #取并集
	return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList)									#创建一个其中所含元素都为0的向量
	for word in inputSet:												#遍历每个词条
		if word in vocabList:											#如果词条存在于词汇表中，则置1
			returnVec[vocabList.index(word)] = 1
		else: print("the word: %s is not in my Vocabulary!" % word)
	return returnVec													#返回文档向量


def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)							#计算训练的文档数目
	numWords = len(trainMatrix[0])							#计算每篇文档的词条数
	pAbusive = sum(trainCategory)/float(numTrainDocs)		#文档属于侮辱类的概率
	p0Num = ones(numWords); p1Num = ones(numWords)	#创建numpy.zeros数组,
	p0Denom = 2.0; p1Denom = 2.0                        	#分母初始化为0.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:							#统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:												#统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = log(p1Num/p1Denom)								#相除  算出来的为 P(w0|0)*P(w1|0)*P(w2|0)*、。。。。。*
	p0Vect = log(p0Num/p0Denom)
	return p0Vect,p1Vect,pAbusive							#返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))									#执行分类并打印分类结果

if __name__ == '__main__':
	testingNB()
