import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

dataset = pd.read_csv('melb_data.csv')

X = dataset.iloc[:, :-1].values

imputer = SimpleImputer(missing_values = np.nan, strategy="most_frequent")
imputer.fit(X)
X =imputer.transform(X)

array = [[a[7], a[3]] for a in X]

colors = (0,0,0)
area = np.pi * 3

plt.scatter([x[0] for x in array], [y[1] for y in array], s= area, c = colors, alpha =0.5)
plt.title('Custo de casa')
plt.xlabel('Area construida')
plt.ylabel('Custo da casa')
plt.show()

plt.boxplot([x[0] for x in array])
plt.title('Area construida')
plt.show()
plt.boxplot([y[1] for y in array])
plt.title('Custo da casa')
plt.show()

xmean = np.mean([x[0] for x in array], axis = 0)
xsd = np.std([x[0] for x in array], axis = 0)
ymean = np.mean([y[1] for y in array], axis = 0)
ysd = np.std([y[1] for y in array], axis = 0)

final_array = [x for x in array if (x[0] > xmean - 2 * xsd)]
final_array = [x for x in final_array if (x[0] < xmean + 2 * xsd)]
final_array = [y for y in final_array if (y[1] > ymean - 2 * ysd)]
final_array = [y for y in final_array if (y[1] < ymean + 2 * ysd)]

minMaxScaler = preprocessing.MinMaxScaler()
final_array = minMaxScaler.fit_transform(final_array)

plt.scatter([x[0] for x in final_array], [y[1] for y in final_array], s = area, c = colors, alpha = 0.5)
plt.title('Custo de casa')
plt.xlabel('Area construida')
plt.ylabel('Custo da casa')
plt.show()

plt.boxplot([x[0] for x in final_array])
plt.title('Area construida')
plt.show()
plt.boxplot([y[1] for y in final_array])
plt.title('Custo da casa')
plt.show()

plt.hist([x[0] for x in final_array])
plt.show()
plt.hist([y[1] for y in final_array])
plt.show()
