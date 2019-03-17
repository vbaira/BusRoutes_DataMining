import os
import sys
import time
import gmplot
import numpy as np
import pandas as pd
from PIL import Image
from ast import literal_eval
from selenium import webdriver


def main():
	#if dataset is not provided on call terminate
	if len(sys.argv)<2:
		print("usage: python visualization.py <input_file> ")
		sys.exit()
	dir_path = os.path.dirname(os.path.abspath(__file__))
	
	#read dataset
	train_data = pd.read_csv(sys.argv[1],converters={"Trajectory": literal_eval},index_col='tripId')
	train_data = train_data[0:100]

	#group train data by journeyPatternId to get 5 random unique journeyPatternIds
	unique_jpId = train_data.groupby('journeyPatternId').first()
	choices=[unique_jpId.iloc[0],unique_jpId.iloc[3],unique_jpId.iloc[4],unique_jpId.iloc[7],unique_jpId.iloc[15]]
	
	#for each of these journeyPatterns create plot with gmplot and take a picture of the plot
	browser = webdriver.Firefox()
	images=[]
	for journey in choices:
		timestamps,lons,lats=zip(*journey['Trajectory'])
		gmap=gmplot.GoogleMapPlotter(53.350140,-6.266155, 11)
		gmap.plot(lats,lons,color="green",edge_width=3)
		gmap.draw(dir_path+"/output/"+journey.name+"_route.html")
		#open html files in browser and screenshot them
		browser.get("file://"+dir_path+"/output/"+journey.name+"_route.html")
		time.sleep(3) #wait for browser to load the file
		image=dir_path+"/output/"+journey.name+"_route.png"
		browser.save_screenshot(image)
		images.append(image)
	browser.quit()

	#create a grid picture with the plot images
	grid_image = Image.new("RGB", (640*3,400*2))
	for index,image in enumerate(images):
		img = Image.open(image)
		img.thumbnail((640,400),Image.ANTIALIAS)
		x = (index % 3)*640
		y = (index / 3)*400
		grid_image.paste(img,(x,y))
	grid_image.save(dir_path+"/output/routes.png")

if __name__ == '__main__':
	main()