import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("AutoInsurance.csv")
print(df.head())

df.drop(["State","Response","Coverage","Effective To Date","EmploymentStatus","Location Code","Number of Open Complaints"], axis = 1, inplace = True)
print(df.head())
print(df.columns)
print(df.max)
print(df.info())

labelencoder = LabelEncoder()
df["Customer"] = labelencoder.fit_transform(df[["Customer"]])
print(df.head())
df["Gender"] = labelencoder.fit_transform(df[["Gender"]])
df["Marital Status"] = labelencoder.fit_transform(df[["Marital Status"]])
df["Vehicle Class"] = labelencoder.fit_transform(df[["Vehicle Class"]])
df["Sales Channel"] = labelencoder.fit_transform(df[["Sales Channel"]])
df["Renew Offer Type"] = labelencoder.fit_transform(df[["Renew Offer Type"]])
print(df.head())

one_hot_encoder = OneHotEncoder(sparse_output = False)

encoding_data = one_hot_encoder.fit_transform(df[["Education"]])
encoding_df = pd.DataFrame(encoding_data, columns = one_hot_encoder.get_feature_names_out(["Education"]))
concat = pd.concat([df, encoding_df], axis = 1)
print(concat)

encoding_data2 = one_hot_encoder.fit_transform(df[["Vehicle Size"]])
encoding_df2 = pd.DataFrame(encoding_data2, columns = one_hot_encoder.get_feature_names_out(["Vehicle Size"]))
concat2 = pd.concat([df, encoding_df2], axis = 1)
print(concat2)

x = df[["Customer","Gender","Marital Status","Customer Lifetime Value","Vehicle Class","Sales Channel",
"Renew Offer Type","Income","Monthly Premium Auto","Months Since Policy Inception","Number of Policies"]]
y = df["Total Claim Amount"]

print(x)

print(y)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, test_size = 0.2, random_state = 42)

print(x_train)

print(x_test)

print(y_train)

print(y_test)

standard_scaler = StandardScaler()
x_train_scaler = standard_scaler.fit_transform(x_train)
x_test_scaler = standard_scaler.transform(x_test)

linear_regression = LinearRegression()
linear_regression.fit(x_train_scaler, y_train)
y_prediction = linear_regression.predict(x_test_scaler)
print(y_prediction)

mse = mean_squared_error(y_test,y_prediction)
print("lenear_regression_mse",mse)

model_score = linear_regression.score(x_test_scaler,y_test)
print(model_score)

cross_validation = cross_val_score(linear_regression, x, y, cv = 5, scoring = "neg_mean_squared_error")
print(cross_validation)

r2_score = r2_score(y_test, y_prediction)
print("r2_score",r2_score)

decision_tree_regression = DecisionTreeRegressor()

decision_tree_regression.fit(x_train,y_train)
y_prediction_decision_tree = decision_tree_regression.predict(x_test)
print(y_prediction_decision_tree)

mse2 = mean_squared_error(y_test, y_prediction_decision_tree)
print("decision_tree_regression_mse",mse2)

plt.figure(figsize = (8,10))
plt.scatter(y_test,y_prediction)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()], "r--", lw=2)
plt.title("Total Claim Amoumt")
plt.xlabel("Actual Claim Amount")
plt.ylabel("Predicted Claim Amount")
plt.show()



