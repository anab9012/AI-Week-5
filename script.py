#Importing the necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
import pandas as pd
df=pd.read_csv('hospital_readmissions_30k.csv')
print(df.head())
missing_counts=df.isnull().sum()
print(missing_counts)
#Using the .get_dummies function to encode the columns with 3 categorical values and 2 categorical values
#Storing all the columns i need to encode regardless of their categories into a variable  named columns_encoded
columns_encoded=['gender','diabetes','hypertension','discharge_destination','readmitted_30_days']
#Running the .get_dummies function to the list i just created
df=pd.get_dummies(df,columns=columns_encoded,drop_first=True)
#Seeing the resuling dataset after encoding
print(df.head())
print(df.info())
#Converting the boolean values given by using .get_dummies into integer values by using the .astype(int) function
#Selecting the columns with the boolean data types and storing in a variable names bool_cols
bool_cols=df.select_dtypes(include=['bool']).columns
#Converting these columns into integers
df[bool_cols]=df[bool_cols].astype(int)
#Checking the results
print(df.head())
#Spltting the blood_pressure column into systolic and dilastolic columns
df[['systolic','diastolic']]=df['blood_pressure'].str.split('/',expand=True)
#Converting the new columns into numeric values
df['systolic']=pd.to_numeric(df['systolic'])
df['diastolic']=pd.to_numeric(df['diastolic'])
#Dropping the original blood_pressure column
df=df.drop('blood_pressure', axis=1)
#Checking the results
print(df.head())
print(df.info())

#Splitting the dataset into Target and Features
X=df.drop(columns=['patient_id','readmitted_30_days'],axis=1)
Y=df['readmitted_30_days']
X_test,X_train,Y_test,Y_train=train_test_split(X,Y,test_size=0.2, random_state=42)
#Initializing the standard scaler
scaler= StandardScaler()
