from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

# 交叉验证的作图演示

if __name__ == "__main__":

      lr = linear_model.LinearRegression()
      boston = datasets.load_boston()
      y = boston.target

      # cross_val_predict returns an array of the same size as `y` where each entry
      # is a prediction obtained by cross validation:
      predicted = cross_val_predict(lr, boston.data, y, cv=10)     #采用10折交叉验证 cv=10

      fig, ax = plt.subplots()
      ax.scatter(y, predicted, edgecolors=(0, 0, 0))   #标志值为x轴，预测值为y轴
      ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  #画出标志值的函数图像形式（直线）
      ax.set_xlabel('Measured')
      ax.set_ylabel('Predicted')
      plt.show()


