import KNN

if __name__ == "__main__":
    group, labels = KNN.creatDataSet()
    result = KNN.classify0([0, 0], group, labels, 3)
    print(result)








