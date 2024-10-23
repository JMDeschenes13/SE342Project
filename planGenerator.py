from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

file_path = "Data\\final_dataset_BFP_numbers_only.csv"

#Reading csv
df = pd.read_csv(file_path)

#Preparing Data
x = df.drop('Exercise Recommendation Plan', axis=1)
y = df['Exercise Recommendation Plan']

x_train = x.iloc[:4500]
x_test = x.iloc[4500:]

y_train = y.iloc[:4500]
y_test = y.iloc[4500:]





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