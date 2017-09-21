## Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values  ## Divide independent/dependent variables - Only Age and Estimated Salary
y = dataset.iloc[:,4].values

# Splitting the dataset into the Training set and Test set 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling (mean by deafult) - necessary as age compare to salary values range is to wide
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

## Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression      ## Class starts with capital letters, while function has underscore
classifier = LogisticRegression(random_state = 0)        ## create an object for a model
classifier.fit(X_train, y_train)                         ## fit an object into training set       

## Predicting the Test set results
y_pred = classifier.predict(X_test)

## Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix     ## Class starts with capital letters, while function has underscore
cm = confusion_matrix(y_test, y_pred)           ## Object that compares prediction to real values on a test set

## Visualising the Training set results on a graph

from matplotlib.colors import ListedColormap 
X_set, y_set = X_train, y_train      ## Create local variable for convinience

## Prepare the x,y grid 

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), ## age
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)) ## est.salary

## Apply pixel classifier that colours points on green and red                     
## Countour for creating border line                      
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

## Plot the limits of x and y

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

## Loop to plot all the datapoints
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],  ## to create scatterplot with matplotlib
                c = ListedColormap(('red', 'green'))(i), label = j)
    
## Aesthetics:
        
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

## Visualising the Test set results

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
