import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer

def func_log(x, a, b, c):
    """Return values from a general log function."""
    return a * np.log(b * x) + c
df=pd.read_csv("../idata/MeanVsStd_JAC.csv")
'''
transformer = FunctionTransformer(np.log, validate=True)
# Data
df=df.sort_values(by=["Mean"])
x_samp=df["Mean"].to_numpy().reshape(-1,1)
y_samp=df["Standard deviation"].to_numpy().reshape(-1,1)
x_trans = transformer.fit_transform(x_samp)             # 1

# Regression
regressor = LinearRegression()
results = regressor.fit(x_trans, y_samp)                # 2
model = results.predict
y_fit = model(x_trans)

# Visualization
plt.scatter(x_samp, y_samp)
plt.plot(x_samp, y_fit, "k--", label="Fit")             # 3
plt.title("Logarithmic Fit")
'''

#print(df.columns)
sns.regplot(x="Mean", y="Standard deviation", data =df)
plt.show()
fit=scipy.stats.linregress(df['Mean'],df['Standard deviation'])
print(fit)

