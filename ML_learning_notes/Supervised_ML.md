## Supervised ML
### Definition
- Algorithm is trained on a labled dataset. 
- Training data includes both the input features and the corresponding correct output. 
- The model makes a prediction and adjusts its internal paramteters to minimize that error over time. 
- Example: fraud detection ( classify emails as either spam or not spam, based on the email address, subject, and keywords), loan default prediction ( predict whetehr a loan applicant is likely to default on a loan based on their financial history and relevant data). 

### Regression
- source: https://colab.research.google.com/drive/1FfikNXcsL1t77IHjIh0tLhHhvhrnKHir?usp=sharing#scrollTo=s7DEThDhdbDf

#### Import Python libraries
```
# Import the appropriate Python libraries.
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
```
#### Load data
##### Load data from pre-stored csv.file in Github
```
# Load the CSV file into a dataframe.
df_income = pd.read_csv('http://bit.ly/income-file')
```
#### Load data from own computer
```
# Select where data file is located on our local drive.
from google.colab import files
uploaded = files.upload()

# Load the CSV file into a dataframe.
import io
df_income = pd.read_csv(io.BytesIO(uploaded['income.csv']))
```

### Generate Desciptive Stats
- use infor() functin and preview the first few records of the data using head function
```
# List all fields and their data types.
df_income.info()

# Preview the first few records of data.
df_income.head()
```
- After finishing previewing, it is easy to generate basic stats using describe()

```
# Generate descriptive stats.
df_income.describe()
```

- To visualize the data into histogram:
```
# Show the distribution of income.
sns.displot(df_income, x="Income");
```

### Correlation Calculate
- use corr() function
```
# Show the correlation (r) betwen variables.
df_income.corr()
```

### Graph correlation
- use seaborn to generate a heatmap:
```
# Show the relationships on a heatmap.  (We omit the ID column.)
corr = df_income.iloc[:, 1:5].corr()
sns.heatmap(corr, annot=True);
```
### Model building
```
# Sepearate the dependent variable (y) from the independent variables (X).
X = df_income[['Age']] # With one variable.
# X = df_income[['Age', 'Education', 'Gender']] # With all variables.
y = df_income['Income']

# Pull out some of the data (25%) and create a test dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Train the prediction model using the training data.
model = LinearRegression()
model.fit(X_train.values, y_train)

# Display the y-intercept and coefficients of our regression model.
# y = b + mx
print(model.intercept_)
print(model.coef_)

# Predict income for a specific scenario.
new_X = [[25]]
# new_X = [[25, 10, 1]] # With all variables.
model.predict(new_X)

# Make predictions in the test data.
y_pred = model.predict(X_test.values)

# Evaluate the performance of the model (r-squared).
metrics.r2_score(y_test, y_pred)

# Compare the actual vs predicted value in the test data.
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_compare.head(10)
```


