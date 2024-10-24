from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import csv

trainingDataFilePath = "Data\\final_dataset_BFP_numbers_only.csv"

userProfileDataPath = "Data\\ProjectData.csv"




#Reading csv for test user profiles
UPdf = pd.read_csv(userProfileDataPath)

#Reading csv for training data
TDdf = pd.read_csv(trainingDataFilePath)

#Preparing Data for training
TDx = TDdf.drop('Exercise Recommendation Plan', axis=1)
TDy = TDdf['Exercise Recommendation Plan']

x_train = TDx.iloc[:4500]
x_test = TDx.iloc[4500:]

y_train = TDy.iloc[:4500]
y_test = TDy.iloc[4500:]


#Preparing data for testing
UPx = UPdf.drop('Exercise Recommendation Plan', axis=1)




#Testing Decision Tree Model
DTModel = DecisionTreeClassifier()

DTModel.fit(x_train, y_train)

DTModelTestPredictions = DTModel.predict(x_test)

DTNumberCorrect = 0
for i in range(len(y_test)):
    if DTModelTestPredictions[i] == y_test.iloc[i]:
        DTNumberCorrect += 1

DTPercentCorrect = DTNumberCorrect/len(y_test)

print("Decision Tree Number Correct" , DTNumberCorrect)
print("Decision Tree Percentage Correct" , DTPercentCorrect)





#Testing Random Forest Model
RFModel = RandomForestClassifier()

RFModel.fit(x_train, y_train)

RFModelTestPredictions = RFModel.predict(x_test)

RFNumberCorrect = 0
for i in range(len(y_test)):
    if RFModelTestPredictions[i] == y_test.iloc[i]:
        RFNumberCorrect += 1

RFPercentCorrect = RFNumberCorrect/len(y_test)

print("Random Forest Number Correct" , RFNumberCorrect)
print("Random Forest Percentage Correct" , RFPercentCorrect)

#Generating exercise plans

DTOutputPlans = DTModel.predict(UPx)

RFOutputPlans = RFModel.predict(UPx)

DTOutputList = DTOutputPlans.tolist()

RFOutputList = DTOutputPlans.tolist()

print(DTOutputList)
print(RFOutputList)

with open('Data\\DTGeneratedPlans.csv', 'w', newline='') as DTFile:
    DTWriter = csv.writer(DTFile)
    DTWriter.writerow(DTOutputList)

with open('Data\\RFGeneratedPlans.csv', 'w', newline='') as RFFile:
    RFWriter = csv.writer(RFFile)
    RFWriter.writerow(RFOutputList)



