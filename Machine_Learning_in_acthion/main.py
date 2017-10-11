import matplotlib.pyplot as plt
from Machine_Learning_in_acthion import regression
from Machine_Learning_in_acthion import KNN
from Machine_Learning_in_acthion import kMeans
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
    # print(normMat)
    # print(ranges)
    # print(minVals)

    # KNN.datingClassTest()

    # myDat, labels = tree.CreatDataSet()
    # myDat[0][-1] = 'maybe'
    # print(myDat)
    # # tmp1 = tree.SplitDataSet(myDat, 1, 1)
    # # tmp2 = tree.SplitDataSet(myDat, 1, 0)
    # temp = tree.ChooseBesttoSplit(myDat)
    # # tmp = tree.CalShannonEnt(myDat)
    # # print(tmp1, '\n', tmp2)
    # print(temp)
    # myTree = tree.createTree(myDat, labels)
    # print(myTree)
    # listofPosts, listClasses = bayes.LoadDataSet()
    # myVocabList = bayes.CreateVocabList(listofPosts)
    # trainMat = []
    # for postinDoc in listofPosts:
    #     trainMat.append(bayes.SetOfWords2Vec(myVocabList, postinDoc))
    # p0v, p1v, pAb = bayes.trainNB0(trainMat, listofPosts)

    # bayes.SetOfWords2Vec(myVocabList, listofPosts[0])

    # dataArr, labelMat = logRegres.LoadDataSet()
    # weights = logRegres.GradAscent(dataArr, labelMat)
    # weights = logRegres.StocGradAscent0(array(dataArr), labelMat)
    # logRegres.PlotBestFit(weights)

    # xArr, yArr = regression.LoadDataSet('machinelearninginaction/Ch08/ex0.txt')
    # ws = regression.StandRegres(xArr, yArr)
    # xMat = mat(xArr)
    # yMat = mat(yArr)
    # yHat = xMat * ws
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    # xCopy = xMat.copy()
    # xCopy.sort(0)
    # yHat = xCopy * ws
    # ax.plot(xCopy[:, 1], yHat)
    # plt.show()

    # abX, abY = regression.LoadDataSet('machinelearninginaction/Ch08/abalone.txt')
    # ridgeWeights = regression.RidgeTest(abX, abY)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridgeWeights)
    # plt.show()

    datMat = mat(kMeans.loadDataSet('machinelearninginaction/Ch10/testSet.txt'))
    myCentroids, clustAssing = kMeans.kMeans(datMat, 4)
    print(myCentroids)
    print(clustAssing)














