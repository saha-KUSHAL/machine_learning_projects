import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics  import classification_report

# importing the dataset
data = pd.read_csv('data.csv')

# checking for any NaN values
# sns.heatmap(data.isnull())

# as there is no null values found, no need for 
# data cleanup

x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# analysis of data by visualing via pairplot
# sns.pairplot(data, hue='label')

# splitting 
x_tarin, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

# initializing the model
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_tarin,y_train)

# checking accurecy
y_predict = knn.predict(x_test)
cr = classification_report(y_test, y_predict)
print(cr)

print(knn.predict([[90,42,43,20.87974371,82.00274423,6.502985292,202.9355362]]))

# plt.show()
