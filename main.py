import KNN
import tree
import bayes
import matplotlib
import matplotlib.pyplot as plt
from numpy import *


if __name__ == "__main__":

    # group, labels = KNN.creatDataSet()
    # result = KNN.classify0([0, 0], group, labels, 3)
    # print(result)

    # datingDataMat, datingLabels = KNN.file2matrix('machinelearninginaction/Ch02/datingTestSet2.txt')
    # print(datingDataMat)
    # print(datingLabels)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
    #            15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
    # plt.show()

    # normMat, ranges, minVals = KNN.autoNorm(datingDataMat)

    # KNN.datingClassTest()

    # myDat, labels = tree.CreatDataSet()
    # myDat[0][-1] = 'maybe'

    # # tmp1 = tree.SplitDataSet(myDat, 1, 1)
    # # tmp2 = tree.SplitDataSet(myDat, 1, 0)
    # temp = tree.ChooseBesttoSplit(myDat)
    # # tmp = tree.CalShannonEnt(myDat)
    # # print(tmp1, '\n', tmp2)

    # myTree = tree.createTree(myDat, labels)

    # listofPosts, listClasses = bayes.LoadDataSet()
    # myVocabList = bayes.CreateVocabList(listofPosts)
    # trainMat = []
    # for postinDoc in listofPosts:
    #     trainMat.append(bayes.SetOfWords2Vec(myVocabList, postinDoc))
    # p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
    # testEntry = ['dog', 'my', 'worthless']
    # thisDoc = array(bayes.SetOfWords2Vec(myVocabList, testEntry))
    # print(testEntry, 'classified as: ', bayes.ClassifyNB(thisDoc, p0V, p1V, pAb))
    # testEntry = ['stupid', 'garbage']
    # thisDoc = array(bayes.SetOfWords2Vec(myVocabList, testEntry))
    # print(testEntry, 'classified as: ', bayes.ClassifyNB(thisDoc, p0V, p1V, pAb))

    bayes.SpamTest()






