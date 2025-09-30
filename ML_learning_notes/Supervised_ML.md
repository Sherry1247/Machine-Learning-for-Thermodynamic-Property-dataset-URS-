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
