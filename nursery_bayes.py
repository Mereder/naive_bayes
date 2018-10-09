from sklearn.cross_validation import train_test_split
from numpy import *

def loadDataSet(filename):
    with open(filename,'r',encoding='utf-8') as fr:
        dataSet = [test.strip().split(',') for test in fr.readlines()]
    classLabel = []
    for data in dataSet:
        classLabel.append(data[-1])
    return dataSet,classLabel

def createFeatList():
    # 27 个标签
    VecList = [
        'usual', 'pretentious', 'great_pret',
        'proper', 'less_proper', 'improper', 'critical', 'very_crit',
        'complete','completed', 'incomplete', 'foster',
        '1', '2', '3', 'more',
        'convenient', 'less_conv', 'critical',
        'convenient', 'inconv',
        'nonprob', 'slightly_prob', 'problematic',
        'recommended', 'priority', 'not_recom']
    return VecList


def feat2Vec(content, myFeatList):
    # 重点处理相同的  convenient
    returnVec = [0] * len(myFeatList)
    n = len(content)
    for i in range(n):
        word = content[i]
        if  word == 'convenient'and i == 4:
            returnVec[16] = 1
        elif word == 'convenient'and i == 5:
            returnVec[19] = 1
        elif word  not in myFeatList:
            print("the word: %s is not in my Vocabulary!" % word)
        else :
            returnVec[myFeatList.index(word)] = 1
    return returnVec


def trainNaiveBayse(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    #直接赋值就好啦  也可以统计计算
    # 0 not_recom 4320(33.333 %)
    # 丢掉recommend  2(0.015 %)
    # 1 very_recom 328(2.531 %)
    # 2 priority  4266(32.917 %)
    # 3 spec_prior 4044(31.204 %)
    pClass = [0.0,0.0,0.0,0.0]

    pClass[0] = 0.33333; pClass[1] = 0.02531; pClass[2] = 0.32917; pClass[3] = 0.31204
    # pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords);     p1Num = ones(numWords)         # change to ones()
    p2Num = ones(numWords);     p3Num = ones(numWords)
    p0Denom = 2.0;     p1Denom = 2.0                           # change to 2.0
    p2Denom = 2.0;     p3Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 'not_recom':
            p0Num += trainMatrix[i]
            p0Denom += 1
        elif trainCategory[i] == 'very_recom':
            p1Num += 1
            p1Denom += sum(trainMatrix[i])
        elif trainCategory[i] == 'priority':
            p2Num += trainMatrix[i]
            p2Denom += 1
        elif trainCategory[i] == 'spec_prior':
            p3Num += trainMatrix[i]
            p3Denom += 1
    pVect = []
    pVect.append(log(p0Num / p0Denom))
    pVect.append(log(p0Num / p0Denom))
    pVect.append(log(p0Num / p0Denom))
    pVect.append(log(p0Num / p0Denom))

    # p0Vect = log(p0Num / p0Denom)                              # change to log()
    # p1Vect = log(p1Num / p1Denom)
    # p2Vect = log(p0Num / p0Denom)
    # p3Vect = log(p1Num / p1Denom)
    return pVect,pClass


def classifyNB(testVect, pVect, pClass):
    p0 = sum(testVect * pVect[0]) + log(pClass[0])      # element-wise mult
    p1 = sum(testVect * pVect[1]) + log(pClass[1])
    p2 = sum(testVect * pVect[2]) + log(pClass[2])
    p3 = sum(testVect * pVect[3]) + log(pClass[3])

    muchBigger = max(p0,p1,p2,p3)
    print(muchBigger)
    if muchBigger == p0:
        return 'not_recom'
    elif muchBigger == p1:
        return 'very_recom'
    elif muchBigger == p2:
        return 'priority'
    elif muchBigger == p3:
        return 'spec_prior'



def testingNB():
    dataSet,classLabel = loadDataSet('nursery.data')                 # 读取数据集
    # trainSet,trainLabel =\
    trainSet, testSet = train_test_split(dataSet, test_size = 0.1)   # 划分训练集和测试集
    myFeatList = createFeatList()                                    # 创建特征表，主要是将所有特征的属性都展开
    trainSetNoLab = []                                               # 去分类标签的 训练集
    trainSetLab  = []                                                # 训练集的分类标签
    for content in trainSet:
        trainSetNoLab.append(content[:-1])
        trainSetLab.append(content[-1])
    testSetNoLab = []
    testSetLab = []
    for content in testSet:
        testSetNoLab.append(content[:-1])
        testSetLab.append(content[-1])
                                                                     # 对应属性值有 则 为1  没有则为 0
    trainMat = []                                                    # 训练的矩阵，通过featlist 将数据向量化
    for content in trainSetNoLab:
        trainMat.append(feat2Vec(content, myFeatList))               # 将实验样本向量化
    #  0  not_recom    4320   (33.333 %)
    #    recommend       2   ( 0.015 %)
    #  1  very_recom    328   ( 2.531 %)
    #  2  priority     4266   (32.917 %)
    #  3  spec_prior   4044   (31.204 %)
    # 去掉 reconmmed 太少了   0 1 2 3  四个 类
    pVect,pClass = trainNaiveBayse(array(trainMat), array(trainSetLab))  # 训练朴素贝叶斯分类器
    # for vect in pVect:
    #     print(vect)           输出测试
    testnum = len(testSetNoLab)
    correctcnt = 0
    print(testSetNoLab[0])
    print(testSetLab[0])
    for i in range(testnum):
        testEntry = testSetNoLab[i]
        testVect = array(feat2Vec(testEntry,myFeatList))                    # 向量化测试用例
        if classifyNB(testVect,pVect,pClass) == testSetLab[i]:
            correctcnt += 1
    print(correctcnt/testnum * 100)
    # thisDoc = array(feat2Vec(, testEntry))
    # print (testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


if __name__ == '__main__':
    testingNB()
    