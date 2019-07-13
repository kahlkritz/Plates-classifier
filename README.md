# Plates-classifier
You have to create a multilayer neural network with one hidden layer and train it using the gradient decent algorithm.  You have to submit your code and your findings on the completed template.  The case study you have to use is involves categorizing the faults of steel plates.

The file Plates.csv contains attributes of 1900 steel plates. A total of 27 attributes are listed for each plate: X_Minimum, X_Maximum, Y_Minimum, Y_Maximum, Pixels_Areas, X_Perimeter, Y_Perimeter, Sum_of_Luminosity, Minimum_of_Luminosity,   Maximum_of_ Lumino, Length_of_Conveyer, TypeOfSteel_A300, TypeOfSteel_A400, Steel_Plate_Thickness, Edges_Index, Empty_Index, Square_Index, Outside_X_Index, Edges_X_Index, Edges_Y_Index, Outside_Global_Index, LogOfAreas, Log_X_Index, Log_Y_Index, Orientation_Index, Luminosity_Index and SigmoidOfAreas.  Don’t panic, you don’t have to know what each of these are. 

Each steel plate is grouped into one of seven categories of faults: 
1. Pastry
2. Z_Scratch
3. K_Scatch
4. Stains
5. Dirtiness
6. Bumps
7. Other_Faults
Each row in the .csv file contains information of one plate, while the final column indicates the fault number.  You have to train a neural network to classify plates into correct fault categories. Note that producing the actual fault number might not be the best approach here. Make sure you test your neural network on data not used during training. The last 40 rows in the file are given without fault number.  Note the categories of these faults on the template.


You should further investigate how the training time and generalization ability of the network can be improved by making use of at least 3 of the following techniques: Prevention of overfitting, noise injection, appropriate weight initialization, dynamic learning rate, momentum, and network architecture.  

Note that this is real world data, so it is unlikely that you will obtain perfect results. Try to find the lowest error possible. 

The last 40 rows in the file are given without fault number.  Note the categories of these faults on the template.
