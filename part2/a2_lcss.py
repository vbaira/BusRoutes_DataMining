import os
import sys
import time
import gmplot
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from fastdtw import fastdtw
from ast import literal_eval
from heapq import nlargest
from selenium import webdriver
from math import radians, cos, sin, asin, sqrt
from a1_nearest_neighbors import haversine


def main():
	#if dataset is not provided on call terminate
	if len(sys.argv)<3:
		print("usage: python a2_lcss.py <train_data_file> <test_data_file> ")
		sys.exit()
	dir_path = os.path.dirname(os.path.abspath(__file__))

	#read train dataset
	train_data = pd.read_csv(sys.argv[1],converters={"Trajectory": literal_eval},index_col='tripId')

	#read test set
	test_data = pd.read_csv(sys.argv[2],sep="\t",converters={"Trajectory": literal_eval})

	#for each of the test trajectories find its 5 nearest neighbors from the train data
	for test_index,test in enumerate(test_data['Trajectory']):
		x_timestamps,x_lons,x_lats=zip(*test)
		x = zip(x_lons,x_lats)
		lcs_distances=[]

		#iterate through test data to find the 5 nearest neighbors
		start = time.time()
		for row in train_data.itertuples():
			trip_id = row[0]
			jp_id = row[1]
			traj = row[2]
			y_imestamps,y_lons,y_lats=zip(*traj)
			y = zip(y_lons,y_lats)
			dist,lcss = lcs(x,y)
			lcs_lons = lcs_lats = []
			if (dist > 0) :
				lcs_lons,lcs_lats = zip(*lcss)
			lcs_distances.append( (dist, trip_id, jp_id, lcs_lons, lcs_lats, y_lons, y_lats) )	
		nn_list=nlargest(5,lcs_distances) 
		stop = time.time()
		elapsed_time = round(stop-start,1)

		#plot test trajectory
		images=[]
		browser = webdriver.Firefox() #open browser
		plot_and_screenshot( (None,None,None,None,None,x_lons,x_lats), browser, test_index, 0, dir_path, images )
		#plot nearest neighbors and create an image for each plot. 
		for index,nn in enumerate(nn_list):
			plot_and_screenshot(nn,browser,test_index,index+1,dir_path,images)
		browser.quit()	#close browser

		#create a grid with the plot images
		create_image_grid(nn_list,test_index+1,dir_path,images,elapsed_time)




def lcs(a, b):
	#(n+1)x(m+1) array.First row and column are set to 0
	row_count = len(a)+1
	column_count = len(b)+1
	#initialize the array
	lcs_dp = [[0 for j in range(column_count)] for i in range(row_count)]
	#fill the array
	for i,x in enumerate(a):
		for j,y in enumerate(b):
			if (haversine(x,y)<=0.2):
				lcs_dp[i+1][j+1] = lcs_dp[i][j] + 1
			else:
				lcs_dp[i+1][j+1] = max(lcs_dp[i+1][j], lcs_dp[i][j+1])
	#find the longest common subsequence and return it
	result = []
	n = len(a)
	m = len(b)
	while n != 0 and m != 0:
		if lcs_dp[n][m] == lcs_dp[n-1][m]:
			n -= 1
		elif lcs_dp[n][m] == lcs_dp[n][m-1]:
			m -= 1
		else:
			result.insert(0,a[n-1])
			n -= 1
			m -= 1
	#return lcs distance and the subsequence itself
	return len(result),result



def plot_and_screenshot(nn,browser,test_index,index,dir_path,images):
	#get coordinates of the neighbor
	neighbor_lons = nn[5]
	neighbor_lats = nn[6]
	#calculate a "center" lon and lat to center  google map
	center_lon = (neighbor_lons[0]+neighbor_lons[len(neighbor_lons)-1])/2.0
	center_lat = (neighbor_lats[0]+neighbor_lats[len(neighbor_lats)-1])/2.0
	#create the plot
	zoom=13
	if (test_index==2):
		zoom=11#reduce zoom for this neighbor for better representation
	gmap=gmplot.GoogleMapPlotter(center_lat,center_lon , zoom)
	gmap.plot(neighbor_lats,neighbor_lons,color="green",edge_width=3)	
	if (index>0):
		#plot longest common subsequence of test record  with the neighbor
		#get lon and lat list
		lcs_lons = nn[3]
		lcs_lats = nn[4]
		gmap.plot(lcs_lats,lcs_lons,color="red",edge_width=3)
	#save plot
	gmap.draw(dir_path+"/a2_output/nn"+str(index)+".html")	
	#open html files in browser and screenshot them
	browser.get("file://"+dir_path+"/a2_output/nn"+str(index)+".html")
	time.sleep(3) #wait for browser to load the file
	image = dir_path+"/a2_output/nn"+str(index)+".png"
	browser.save_screenshot(image)
	images.append(image)



def create_image_grid(nn_list,test_index,dir_path,images,elapsed_time):
	#create new image
	grid_image = Image.new("RGB", (640*3,450*2) , "white")
	for index,image in enumerate(images):
		#get a thumbnail for each plot screenshot and paste it to the new image
		img = Image.open(image)
		img.thumbnail((640,400),Image.ANTIALIAS)
		x = (index % 3)*640
		y = (index / 3)*450
		grid_image.paste(img,(x,y))
		#add text under each plot
		text = ""
		if (index==0) :
			text = "Test Trip "+str(test_index)+"\nDt = "+str(elapsed_time)+" sec"
		else:
			neighbor = nn_list[index-1]
			jp_id = neighbor[2]
			lcs_dist = neighbor[0]
			text = "Neighbor "+str(index)+"\nJP_ID: "+jp_id+" || #Matching Points: "+str(lcs_dist)
		font_type = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf",18)
		draw = ImageDraw.Draw(grid_image)
		draw.text(xy=(x+200,y+400), text=text, fill="black", font=font_type)
	#save the grid image
	grid_image.save(dir_path+"/a2_output/test_trip_"+str(test_index)+".png")



if __name__ == '__main__':
	main()