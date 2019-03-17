import os
import sys
import time
import gmplot
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from fastdtw import fastdtw
from ast import literal_eval
from heapq import nsmallest
from selenium import webdriver
from math import radians, cos, sin, asin, sqrt


def main():
	#if dataset is not provided on call terminate
	if len(sys.argv)<3:
		print("usage: python a1_nearest_neighbors.py <train_data_file> <test_data_file> ")
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
		dtw_distances=[]

		#iterate through test data to find the nearest neighbors
		start = time.time()
		for row in train_data.itertuples():
			trip_id = row[0]
			jp_id = row[1]
			traj = row[2]
			y_imestamps,y_lons,y_lats=zip(*traj)
			y = zip(y_lons,y_lats)
			dist,path = fastdtw(x,y,dist=haversine)
			dtw_distances.append( (dist, trip_id, jp_id, y_lons, y_lats) )	
		nn_list=nsmallest(5,dtw_distances) 
		stop = time.time()
		elapsed_time = round(stop-start,1)

		#plot test trajectory
		images=[]
		browser = webdriver.Firefox() #open browser
		plot_and_screenshot( (None,None,None,x_lons,x_lats), browser, 0, dir_path, images )
		#plot nearest neighbors and create an image of each plot. 
		for index,nn in enumerate(nn_list):
			plot_and_screenshot(nn,browser,index+1,dir_path,images)
		browser.quit()	#close browser

		#create a grid with the plot images
		create_image_grid(nn_list,test_index+1,dir_path,images,elapsed_time)




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
			dtw_dist = neighbor[0]
			text = "Neighbor "+str(index)+"\nJP_ID: "+jp_id+" || DTW: "+str(dtw_dist)+" km"
		font_type = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf",18)
		draw = ImageDraw.Draw(grid_image)
		draw.text(xy=(x+200,y+400), text=text, fill="black", font=font_type)
	#save the grid image
	grid_image.save(dir_path+"/a1_output/test_trip_"+str(test_index)+".png")



def plot_and_screenshot(nn,browser,index,dir_path,images):
	#get lon and lat list
	lons = nn[3]
	lats = nn[4]
	#calculate a "center" lon and lat to center  google map
	center_lon = (lons[0]+lons[len(lons)-1])/2.0
	center_lat = (lats[0]+lats[len(lats)-1])/2.0
	#create the plot
	gmap=gmplot.GoogleMapPlotter(center_lat,center_lon , 13)
	gmap.plot(lats,lons,color="green",edge_width=3)
	gmap.draw(dir_path+"/a1_output/nn"+str(index)+".html")
	#open html files in browser and screenshot them
	browser.get("file://"+dir_path+"/a1_output/nn"+str(index)+".html")
	time.sleep(3) #wait for browser to load the file
	image = dir_path+"/a1_output/nn"+str(index)+".png"
	browser.save_screenshot(image)
	images.append(image)



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



if __name__ == '__main__':
	main()