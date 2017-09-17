import KNN
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

    KNN.datingClassTest()








