import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

data_x = np.array([10,14,16,20,25,28,31,34]).reshape(-1,1)
# print('ini datax',data_x)

data_y = [71,54,43,33,24,22,18,13]
# data_y = [77,56,48,39,31,28,20,19]

trans = PolynomialFeatures(degree=2, include_bias=False)
x = trans.fit_transform(data_x)
# print('ini x',x)

model = LinearRegression()
model.fit(x,data_y)
r_square = model.score(x,data_y)
print(f'Score : {r_square}')
print(f'intercept : {model.intercept_}, coef : {model.coef_}')

# print(f'Persamaaan : {round(model.intercept_,2)} + ({round(model.coef_[0],2)} * X)')
#  y_pred = model.intercept_ + model.coef_ * x
jarak = [num for num in range(5,36)]
x_jarak = jarak
# jarak = np.arange(5,40)
jarak = np.array(jarak).reshape(-1,1)
# print('ini jarak',jarak)
# print(jarak.reshape(-1,1))

# print(trans.transform(jarak))
jarak = trans.transform(jarak)
# jarak = jarak.reshape(-1,1)
# for i in jarak:
# y_pred = [model.intercept_+(model.coef_* i) for i in jarak]

# y_pred = [model.predict(i) for i in jarak]
y_pred = model.predict(jarak)

plt.scatter(data_x, data_y,color='r')
plt.plot(x_jarak,y_pred )
plt.title(f'persamaan polynomial orde 2 acc : {round(r_square,5)}%')
plt.xlabel('jarak (cm)')
plt.ylabel('panjang pixel')
plt.savefig('model_plot_persamaan.png')
plt.show()


