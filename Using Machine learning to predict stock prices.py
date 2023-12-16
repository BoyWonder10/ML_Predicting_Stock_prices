#!/usr/bin/env python
# coding: utf-8

# In[31]:


get_ipython().system('pip install yfinance')
import pandas as pd


# In[32]:


import yfinance as yf
import matplotlib.pyplot as plt


# In[33]:


sp500 = yf.Ticker("^GSPC")


# In[34]:


sp500 = sp500.history(period="max")


# In[35]:


sp500


# In[36]:


sp500.index


# In[37]:


sp500.plot.line(y="Close", use_index=True)


# In[38]:


del sp500["Dividends"]
del sp500["Stock Splits"]


# In[39]:


sp500["Tomorrow"] = sp500["Close"].shift(-1)


# In[40]:


sp500


# In[43]:


sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)


# In[44]:


sp500


# In[45]:


sp500 = sp500.loc["1990-01-01":].copy()


# In[47]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume","Open","High","Low"]
model.fit(train[predictors],train["Target"])


# """ The RandomForestClassifier is a machine learning model that belongs to the family of ensemble methods, specifically the Random Forest algorithm. It is implemented in the scikit-learn library, which is a popular machine learning library in Python.
# 
# Here's a brief overview of the RandomForestClassifier:
# 
# 1. Ensemble Learning:
# Ensemble learning involves combining multiple models to create a stronger and more robust model than the individual models. Random Forest is an ensemble learning method that constructs a multitude of decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
# 
# 2. Decision Trees:
# A decision tree is a flowchart-like structure in which each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label (for classification problems) or a numerical value (for regression problems).
# 
# 3. Random Forest:
# The Random Forest algorithm builds multiple decision trees and merges them together. Each tree is constructed using a random subset of the training data and a random subset of the features. This randomness helps to reduce overfitting and improves the generalization ability of the model.
# 
# 4. RandomForestClassifier in scikit-learn:
# In scikit-learn, the RandomForestClassifier is a class that implements the Random Forest algorithm for classification tasks. It has various parameters that allow you to customize the behavior of the algorithm, such as the number of trees in the forest, the maximum depth of each tree, and the number of features to consider when looking for the best split.
# 
# Here's a basic example of how to use RandomForestClassifier:
# 
# python
# Copy code
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris
# from sklearn.metrics import accuracy_score
# 
# # Load the Iris dataset
# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
# 
# # Create a RandomForestClassifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# 
# # Train the model
# rf_classifier.fit(X_train, y_train)
# 
# # Make predictions on the test set
# predictions = rf_classifier.predict(X_test)
# 
# # Evaluate the accuracy
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy: {accuracy}")
# In this example, we use the Iris dataset, split it into training and testing sets, create a RandomForestClassifier, train the model on the training set, make predictions on the test set, and evaluate the accuracy of the model."""

# In[63]:


from sklearn.metrics import precision_score

preds = model.predict(test[predictors])


# In[65]:


preds = pd.Series(preds, index=test.index)


# In[67]:


preds


# In[54]:


precision_score(test["Target"], preds)


# In[55]:


combined=pd.concat([test["Target"], preds], axis=1)


# In[57]:


combined.plot()


# In[70]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index = test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[71]:


def backtest(data,model,predictors,start=2500, step=250):
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)
    


# In[72]:


predictions = backtest(sp500, model, predictors)


# In[74]:


predictions["Predictions"].value_counts()


# In[73]:


precision_score(predictions["Target"],predictions["Predictions"])


# In[75]:


predictions["Target"].value_counts() / predictions.shape[0]


# In[77]:


horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors += [ratio_column, trend_column]
    


# In[82]:


sp500 = sp500.dropna()


# In[83]:


sp500


# In[94]:


model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


# In[95]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index = test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[96]:


predictions = backtest(sp500, model, new_predictors)


# In[97]:


predictions["Predictions"].value_counts()


# In[98]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[ ]:




