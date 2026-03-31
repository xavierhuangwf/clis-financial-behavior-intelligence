#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def extract_datetime_features(df):
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'])
    df.loc[:, 'Year'] = df['Datetime'].dt.year
    df.loc[:, 'Month'] = df['Datetime'].dt.month
    df.loc[:, 'Day'] = df['Datetime'].dt.day
    df.loc[:, 'DayOfWeek'] = df['Datetime'].dt.dayofweek
    df.loc[:, 'Hour'] = df['Datetime'].dt.hour
    return df


file_path = 'C:\\Users\\levit\\Downloads\expenses_go_data_with_categories.csv'
data = pd.read_csv(file_path)

# Ensure the data has the required columns
required_columns = ['Date', 'Timestamp', 'Balance', 'Amount', 'Third Party Name', 'Expenditure categories']

X = data[['Date', 'Timestamp', 'Balance', 'Amount', 'Third Party Name']]
y = data['Expenditure categories']

X = extract_datetime_features(X)

numerical_features = ['Balance', 'Amount', 'Year', 'Month', 'Day', 'DayOfWeek', 'Hour']
categorical_features = ['Third Party Name']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))





# In[2]:


feature_names = numerical_features + categorical_features
importances = model.named_steps['classifier'].feature_importances_

print("Feature Importances:")
for feature, importance in zip(feature_names, importances):
    print(f"{feature}: {importance}")


# In[11]:


## save model
import joblib

joblib.dump(model, 'model.pkl')


# In[4]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


title = "Learning Curves (Random Forest)"
cv = 5
plot_learning_curve(model, title, X, y, cv=cv)
plt.show()


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

