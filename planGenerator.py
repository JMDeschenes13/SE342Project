
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import csv

trainingDataFilePath = "Data\\final_dataset_BFP_numbers_only.csv"

userProfileDataPath = "Data\\ProjectData.csv"




#Reading csv for test user profiles
UPdf = pd.read_csv(userProfileDataPath)

#Reading csv for training data
TDdf = pd.read_csv(trainingDataFilePath)


TDCleaneddf = TDdf.drop_duplicates()



#Preparing Data for training
TDx = TDCleaneddf.drop('Exercise Recommendation Plan', axis=1)
TDy = TDCleaneddf['Exercise Recommendation Plan']

# transform the dataset
oversample = SMOTE()
TDx, TDy = oversample.fit_resample(TDx, TDy)



x_train, x_test, y_train, y_test = train_test_split(TDx, TDy, test_size=.1, random_state=12)



#Preparing data for creating plans
UPx = UPdf.drop('Exercise Recommendation Plan', axis=1)




#Testing Decision Tree Model
DTModel = DecisionTreeClassifier()


# parameters for 89%
#DTModel = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=7, min_samples_split=200)


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



#Testing Gaussian Naive Bayes
GNBModel = GaussianNB(var_smoothing=0.000001)

GNBModel.fit(x_train, y_train)

GNBModelTestPredictions = GNBModel.predict(x_test)

GNBNumberCorrect = 0
for i in range(len(y_test)):
    if GNBModelTestPredictions[i] == y_test.iloc[i]:
        GNBNumberCorrect += 1

GNBPercentCorrect = GNBNumberCorrect/len(y_test)

print("Gaussian Naive Bayes Number Correct" , GNBNumberCorrect)
print("Gaussian Naive Bayes Percentage Correct" , GNBPercentCorrect)


#Testing Logistic Regression
scaler = StandardScaler()

LRModel = LogisticRegression(max_iter=5000)


# Had to redo the data so I could scale it for the linear regression
LRTDx = scaler.fit_transform(TDx, TDy)

LRUPx = scaler.fit_transform(UPx)

LRx_train, LRx_test, LRy_train, LRy_test = train_test_split(LRTDx,TDy, test_size=.1, random_state=12)



LRModel.fit(LRx_train, LRy_train)

LRModelTestPredictions = LRModel.predict(LRx_test)

LRNumberCorrect = 0
for i in range(len(LRy_test)):
    if LRModelTestPredictions[i] == LRy_test.iloc[i]:
        LRNumberCorrect += 1

LRPercentCorrect = LRNumberCorrect/len(LRy_test)

print("Logistic Regression Number Correct" , LRNumberCorrect)
print("Logistic Regression Percentage Correct" , LRPercentCorrect)




#Generating exercise plans

DTOutputPlans = DTModel.predict(UPx)

RFOutputPlans = RFModel.predict(UPx)

GNBOutputPlans = GNBModel.predict(UPx)

LROutputPlans = LRModel.predict(LRUPx)

DTOutputList = DTOutputPlans.tolist()
DTOutputList.insert(0, "DT Plans")

RFOutputList = RFOutputPlans.tolist()
RFOutputList.insert(0, "RF Plans")

GNBOutputList = GNBOutputPlans.tolist()
GNBOutputList.insert(0, "GNB Plans")

LROutputList = LROutputPlans.tolist()
LROutputList.insert(0, "LR Plans")

print(DTOutputList)
print(RFOutputList)
print(GNBOutputList)
print(LROutputList)

with open('Data\\AIGeneratedPlans1.csv', 'w', newline='') as AIFile:
    AIWriter = csv.writer(AIFile)
    AIWriter.writerow(DTOutputList)
    AIWriter.writerow(RFOutputList)
    AIWriter.writerow(GNBOutputList)
    AIWriter.writerow(LROutputList)






