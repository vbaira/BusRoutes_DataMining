from math import radians, cos, sin, asin, sqrt
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator
from collections import Counter
from fastdtw import fastdtw
import numpy as np
import operator
from heapq import nsmallest,nlargest

#Brute force Knn estimator(compatible with scikit api)
class MyKNN(BaseEstimator):

	def __init__(self,k=1):
		self.k = k

	def fit(self,X,y):
		#store the classes seen during fit
		self.classes_ = unique_labels(y)
		self.X_ = X
		self.y_ = y
		#return the classifier
		return self


	def predict(self, X):
		#check is fit had been called
		check_is_fitted(self, ['X_', 'y_'])
		#predict class for each test instance
		predictions=[]
		for test_instance in X:
			k_nearest = self.__get_k_nearest_neighbours(test_instance)
			
			dist,neighb_class = zip(*k_nearest) #unzip

			#vote for the predicted class
			#predicted_class = self.__majority_vote(neighb_class)
			#predicted_class = self.__first(neighb_class)
			predicted_class = self.__vote_dualID(dist,neighb_class)
			#predicted_class = self.__vote_uniform(neighb_class)
			#predicted_class = self.__vote_dualIU(dist,neighb_class)

			predictions.append(predicted_class)
		return predictions


	#return k nearest neighbours in a list of (neighbour_distance,neighbour_class) tuples
	def __get_k_nearest_neighbours(self,test_instance):
		#distances of test instance from all other data points
		test_timestamps,test_lons,test_lats=zip(*test_instance)
		test_coords = zip(test_lons,test_lats)
		distances=[]
		for train_instance in self.X_:
			train_timestamps,train_lons,train_lats=zip(*train_instance)
			train_coords = zip(train_lons,train_lats)
			dist,path = fastdtw(test_coords,train_coords,dist=haversine)
			distances.append( dist )
		#zip the distance from a point with the class of that point 
		zipped = zip(distances,self.y_)
		#return k smallest distances and their corresponding class
		return nsmallest(self.k,zipped)


	#get a list of neighbour classes and return the most common in the list
	def __majority_vote(self,k_neighbours):
		#print "maj_vote"
		count = Counter(k_neighbours)
		return count.most_common()[0][0]


	#return class of closest neighbor
	def __first(self,k_neighbours):
		#print "first"
		return k_neighbours[0]


	#vote with weights
	def __vote_dualID(self,dists,k_neighbours):
		#print "dualID"
		weights=[1]
		for dist in  dists[1:]:
			if dist == dists[0]:
				weights.append[1]
			else:
				a = (dists[self.k-1]-dist) / float(dists[self.k-1]-dists[0])
				b = (dists[self.k-1]+dists[0]) / float(dists[self.k-1]+dist)
				weights.append(a*b)
		counter={}
		for index,key in enumerate(k_neighbours):
			if key in counter:
				counter[key] = counter[key]+weights[index]
			else:
				counter[key] = weights[index]
		return nlargest(1,counter.items(),key=operator.itemgetter(1))[0][0]



	def __vote_uniform(self,k_neighbours):
		#print "Uniform"
		weights=[]
		for i in range(1,self.k+1):
			weights.append(1/float(i))
		counter={}
		for index,key in enumerate(k_neighbours):
			if key in counter:
				counter[key] = counter[key]+weights[index]
			else:
				counter[key] = weights[index]
		return nlargest(1,counter.items(),key=operator.itemgetter(1))[0][0]
	


	def __vote_dualIU(self,dists,k_neighbours):
		#print "dualIU"
		weights=[1]
		for i,dist in enumerate(dists[1:]):
			if dist == dists[0]:
				weights.append[1]
			else:
				a = 1/float(i+2)
				b = (dists[self.k-1]-dist) / float(dists[self.k-1]-dists[0])
				weights.append(a*b)
		counter={}
		for index,key in enumerate(k_neighbours):
			if key in counter:
				counter[key] = counter[key]+weights[index]
			else:
				counter[key] = weights[index]
		return nlargest(1,counter.items(),key=operator.itemgetter(1))[0][0]

		


#harvesine distance
def haversine(x,y):
	# convert decimal degrees to radians
	lon1 = x[0]
	lat1 = x[1]
	lon2 = y[0]
	lat2 = y[1] 
	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
	# haversine formula 
	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
	c = 2 * asin(sqrt(a)) 
	r = 6371 # radius of earth in kilometers.
	return c * r
