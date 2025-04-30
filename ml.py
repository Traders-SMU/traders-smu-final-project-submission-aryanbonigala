import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

spy = yf.download('SPY', start='2019-01-01', end='2024-01-01')

spy.head(25)

spy.shape

spy.isna().sum()

spy.dtypes

"""##feature engineering"""

spy['Return'] = spy['Close'].pct_change() #returns

#n equals the number of days we want to set for our data window

n = 14

# Rolling Volatility
spy['RollingVolatility'] = spy['Return'].rolling(window=n).std()

# Average True Range (simplified)
spy['High-Low'] = spy['High'] - spy['Low']
spy['ATR'] = spy['High-Low'].rolling(window=n).mean()

# Skewness and Kurtosis
spy['RollingSkew'] = spy['Return'].rolling(window=n).apply(skew)
spy['RollingKurtosis'] = spy['Return'].rolling(window=n).apply(kurtosis)

# Rolling Average Volume
spy['AvgVolume'] = spy['Volume'].rolling(window=n).mean()

vol_median = spy['RollingVolatility'].median()
vol_median

spy['Ftr Vol'] = (spy['RollingVolatility'].shift(-1) > vol_median).astype(int) #shift -n so that the label is n days backwards, predicting the volatility for n days after

spy['Ftr Vol'].value_counts()

"""1 = Tomorrow is high volatility

0 = Tomorrow is low volatility
"""

window   = 14    # lookback period for the moving average
num_std  = 2     # number of standard deviations for the bands

# 2.  Compute the middle band (SMA) and rolling std dev
spy['BB_MA']  = spy['Close'].rolling(window).mean()
spy['BB_STD'] = spy['Close'].rolling(window).std()

# 3.  Compute upper and lower bands
spy['BB_upper'] = spy['BB_MA'] + num_std * spy['BB_STD']
spy['BB_lower'] = spy['BB_MA'] - num_std * spy['BB_STD']

# 4.  (Optional) Band width or % bandwidth feature
spy['BB_width']      = spy['BB_upper'] - spy['BB_lower']

spy

spy.isna().sum()

if spy.isna().sum().any():
  spy.dropna(inplace=True)

spy.isna().sum()

spy['Ftr Vol'].value_counts()

spy.head(25)

"""## visualization"""

spy[['ATR', 'RollingVolatility', 'BB_width', 'RollingSkew', 'RollingKurtosis', 'AvgVolume']].plot(subplots=True, figsize=(15, 10))
plt.tight_layout()
plt.show()



"""##Predicting"""

cols = ['Return', 'RollingVolatility', 'ATR', 'RollingSkew',
        'RollingKurtosis', 'AvgVolume','BB_width'] #

X = spy[cols]
y = spy['Ftr Vol']

X_train,X_test, y_train,  y_test = train_test_split(X, y, test_size = .2)

"""###logreg"""

logreg = LogisticRegression(solver = 'liblinear')

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy

scores = cross_val_score(logreg, X, y, cv=5, scoring='accuracy')
print("5-fold CV accuracy:", scores, "mean=", scores.mean())

print(classification_report(y_test, y_pred))

print(y_train.value_counts())      # how many 1’s vs. 0’s in your train set?

print(y_test.value_counts())       # ditto for test set

"""###random forest"""

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced',   # helps if classes are a bit uneven
    n_jobs=-1                  # use all CPU cores
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

rf_tuned = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,            # limit tree height
    min_samples_leaf=5,     # each leaf needs ≥5 samples
    max_features='sqrt',    # consider only √(n_features) at each split
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_tuned.fit(X_train, y_train)
print("Tuned train acc:", rf_tuned.score(X_train, y_train))
print("Tuned test  acc:", rf_tuned.score(X_test,  y_test))

# 3) Set up a time-series split (no shuffling!)
tscv = TimeSeriesSplit(n_splits=5)

# 4) Compute cross-validated accuracies
scores = cross_val_score(
    rf_tuned,
    X,
    y,
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1
)

print("TimeSeriesSplit accuracies:", scores)
print("Mean accuracy         :", scores.mean())

scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print("5-fold CV accuracy:", scores, "mean=", scores.mean())

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

train_acc = rf.score(X_train, y_train)
test_acc  = rf.score(X_test,  y_test)
print(f"Train accuracy: {train_acc:.3f}")
print(f" Test accuracy: {test_acc:.3f}")

rf_oob = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf_oob.fit(X_train, y_train)
print("OOB score:", rf_oob.oob_score_)
