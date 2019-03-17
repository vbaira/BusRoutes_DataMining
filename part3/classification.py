import os
import sys
import time
import gmplot
import numpy as np
import pandas as pd
from ast import literal_eval
from my_knn import MyKNN
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold,cross_validate



def main():
	#if dataset is not provided on call terminate
	if len(sys.argv)<3:
		print("usage: python classification.py <train_data_file> <test_data_file> ")
		sys.exit()
	dir_path = os.path.dirname(os.path.abspath(__file__))

	#read train dataset
	train_data = pd.read_csv(sys.argv[1],converters={"Trajectory": literal_eval},index_col='tripId')
	cv_train_data = train_data[0:500]

	#label encoder for categories
	#category_labels ->numeric representation of categories of  the train data
	le = preprocessing.LabelEncoder()
	category_labels=le.fit_transform(train_data["journeyPatternId"])
	cv_category_labels = category_labels[0:500]

	#create knn classifier
	knn_clf = MyKNN(k=5)

	#10-fold cross validation on a subset of the train data(not all because of execution time being too long)
	k_fold = StratifiedKFold(n_splits=10,shuffle=True, random_state=7)
	metrics=['accuracy']
	knn_result=cross_validate(knn_clf,cv_train_data['Trajectory'],cv_category_labels, cv=k_fold,scoring=metrics,return_train_score=False)
	print "My implementation of KNN(brute force):"
	for key,value in knn_result.iteritems():
		print key +" : "+ str(np.round_(np.mean(value),decimals=5))
	print("\n")

	#read test set
	test_data = pd.read_csv(sys.argv[2],sep="\t",converters={"Trajectory": literal_eval})

	#predict jp_id for  records in test set
	knn_clf.fit(train_data['Trajectory'],category_labels)
	test_category_pred = knn_clf.predict(test_data['Trajectory'])

	#store predictions to file
	create_pred_file(test_data,test_category_pred,le,dir_path)



#create testSet_categories.csv file containing the predictions
def create_pred_file(test_data,test_category_pred,le,dir_path):
	temp=[]
	for index,pred_cat in enumerate(le.inverse_transform(test_category_pred)):
		temp.append([index,pred_cat])
	df = pd.DataFrame(temp,columns=['Test_Trip_ID','Predicted_JourneyPatternID'])
	df.to_csv(dir_path+'/output/testSet_JourneyPatternIDs.csv',sep="\t",index=False)



if __name__ == '__main__':
	main()