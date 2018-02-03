"""
IRIS Dataset

Challenge: To predict the class of the given flower based on its physical characteristics.

"""

import sklearn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib
matplotlib.style.use('ggplot')

def handle_data():
	file = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	col_names = ['Sepal-Length', 'Sepal-Width', 'Petal-Length', 'Petal_Wisdth', 'Class']
	data = pd.read_csv(file, names = col_names)
	print data.shape
	Labels = data.Class
	data.drop('Class', 1, inplace = True)
	
	return data.values, Labels.values 
	
	
from sklearn import model_selection
	
Features, Labels = handle_data()

test_size = 0.20
seed = 10
Features_train, Features_test, Labels_train, Labels_test = model_selection.train_test_split(Features, Labels, test_size=test_size, random_state = seed)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC()
clf.fit(Features_train, Labels_train)
pred = clf.predict(Features_test)

print "ACCURACY:",accuracy_score(pred, Labels_test)
