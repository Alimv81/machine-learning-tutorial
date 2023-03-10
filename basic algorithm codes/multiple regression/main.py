from Utils import *


if __name__ == '__main__':
	valuesNames, featuresNames, objects = MultipleRegression.loadData('datasets/Fish.csv') 
	max_keys = MultipleRegression.optimize(objects)
	slopes = MultipleRegression.fitlines(objects, len(featuresNames))
	MultipleRegression.plot(objects, slopes, featuresNames, valuesNames, max_keys)
