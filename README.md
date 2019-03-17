# BusRoutes_DataMining
Demonstration of various data mining techniques on Dublin bus routes.Nearest neighbor search using DTW ,Longest common subroute using LCSS,Classification of routes and Visualization of the results.

## Part 1
~~~
python visualization.py <train_data>
~~~
>Selects 5 random  bus routes and visualizes them using **[gmplot library](https://pypi.org/project/gmplot/)**.  
Afterwards it opens the .html files that gmplot made, using Firefox , and takes screenshots of them.This procedure is done using **[selenium library](https://pypi.org/project/selenium/)**.  
Lastly it creates a 3x2 grid of the screenshots using **[Pillow library](https://pypi.org/project/Pillow/)**.  
All the output files are stored in output directory.

![Result](https://github.com/vbaira/BusRoutes_DataMining/blob/master/part1/output/routes.png)

## Part 2
~~~
python a1_nearest_neighbors.py <train_data> <test_data>
~~~
>For each route in test dataset looks for the 5 nearest routes in train dataset.  
Distance between routes is calculated using **[fastdtw library](https://pypi.org/project/fastdtw/)**.  
Results are visualized and stored in a1_output directory.

![Result](https://github.com/vbaira/BusRoutes_DataMining/blob/master/part2/a1_output/test_trip_1.png)

~~~
python a2_lcss.py <train_data> <test_data>
~~~
>For each  route in test dataset looks for the 5 routes with the longest common subroute from the train dataset.  
This is achieved using the dynamic programming LCSS algorithm.  
Results are visualized and stored in a2_output directory.

![Result](https://github.com/vbaira/BusRoutes_DataMining/blob/master/part2/a2_output/test_trip_1.png)

## Part 3
~~~
python classification.py <train_data> <test_data>
~~~
>Creates a KNN classifier which is implemented in **my_knn.py** using the scikit library API.  
Afterwards it uses the said classifier to predict the classes of the routes in test dataset.  
Results are stored in testSet_JourneyPatternIDs.csv file in the output directory.
