## Supervised ML
### Definition
- Algorithm is trained on a labled dataset. \
- Training data includes both the input features and the corresponding correct output. \
- The model makes a prediction and adjusts its internal paramteters to minimize that error over time. \
- Example: fraud detection ( classify emails as either spam or not spam, based on the email address, subject, and keywords), loan default prediction ( predict whetehr a loan applicant is likely to default on a loan based on their financial history and relevant data). \

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

