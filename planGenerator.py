from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pandas as pd

file_path = "Data/final_dataset_BFP.csv"

df = pd.read_csv(file_path)

x = df.drop('Exercise Plan Recommendation', axis=1)
y = df['Exercise Plan Recommendation']