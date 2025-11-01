## 🧠 Overview
This mini project demonstrates *Data Cleaning* and *Feature Engineering* on the *Titanic dataset* using Python’s pandas library.  
The goal is to prepare raw Titanic passenger data for machine learning by handling missing values, correcting data types, and creating meaningful new features.

---

## 🎯 Objectives
- Handle missing values and inconsistent data
- Remove duplicates
- Create new informative features (e.g., passenger title, age group)
- Convert categorical data into numerical form
- Prepare clean data for modeling

---

## ⚙ Implementation Steps

1. *Load the Dataset* — using pandas  
2. *Handle Missing Values* — fill with median or mode  
3. *Remove Duplicates* — eliminate repeated records  
4. *Feature Engineering:*
   - Extract “Title” from passenger names  
   - Categorize “Age” into groups  
   - Convert Sex and Embarked to numeric values  
5. *Save Processed Data* — export clean dataset as processed_data.csv

---

## 📊 Sample Output
| PassengerId | Sex | Age | Fare | Embarked | Title | AgeGroup |
|--------------|-----|-----|------|-----------|--------|----------|
| 1 | 0 | 22 | 7.25 | 0 | Mr | YoungAdult |
| 2 | 1 | 38 | 71.28 | 1 | Mrs | Adult |

---

## 🧮 Libraries Used
- *pandas* – For dataset handling  
- *numpy* – For handling missing values and numeric operations  

---

## 🚀 How to Run
```bash
pip install pandas numpy
python main.py
