# Step 1: Import required libraries
import pandas as pd
import numpy as np

# Step 2: Load the Titanic dataset
# If you have the dataset file, place it in the same folder and name it 'titanic.csv'
# df = pd.read_csv('titanic.csv')

# For demonstration, let's create a small sample dataframe
data = {
    'PassengerId': [1, 2, 3, 4, 5, 6],
    'Name': ['Braund, Mr. Owen', 'Cumings, Mrs. John', 'Heikkinen, Miss. Laina', 'Allen, Mr. William', 'Moran, Mr. James', 'McCarthy, Mr. Timothy'],
    'Sex': ['male', 'female', 'female', 'male', 'male', None],
    'Age': [22, 38, 26, 35, np.nan, 54],
    'SibSp': [1, 1, 0, 0, 0, 0],
    'Parch': [0, 0, 0, 0, 0, 0],
    'Fare': [7.25, 71.2833, 7.925, 8.05, 8.4583, 51.8625],
    'Embarked': ['S', 'C', 'S', 'S', np.nan, 'S']
}
df = pd.DataFrame(data)

print("✅ Dataset Loaded Successfully!")
print(df.head())

# Step 3: Data Cleaning
print("\n🧹 Data Cleaning Steps...")

# 3.1 - Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Sex'].fillna('male', inplace=True)  # assume male for missing

# 3.2 - Remove duplicates
df.drop_duplicates(inplace=True)

# 3.3 - Check and correct data types if needed
df['Fare'] = df['Fare'].astype(float)

print("✅ Missing values handled and duplicates removed!")

# Step 4: Feature Engineering
print("\n⚙ Feature Engineering Steps...")

# 4.1 - Extract titles from passenger names
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# 4.2 - Create Age categories
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])

# 4.3 - Convert categorical features to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 4.4 - Drop unnecessary columns
df.drop(['Name'], axis=1, inplace=True)

print("✅ Feature engineering completed!")

# Step 5: Display the processed data
print("\n📊 Cleaned and Processed Data:")
print(df.head())

# Step 6: Save processed data to CSV
df.to_csv('processed_data.csv', index=False)
print("\n💾 Processed data saved successfully as 'processed_data.csv'")

print("\n✅ Project Completed Successfully!")