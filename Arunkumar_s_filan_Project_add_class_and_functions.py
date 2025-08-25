import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

class Dataset:
    
    def __init__(self,file_path):
        self.data = pd.read_csv(file_path)
        print(self.data)
        print(self.data.describe().info())

    def drop_columns(self,columns):
        self.data.drop(columns, axis = 1, inplace = True)
        print(self.data.head())
        print(self.data.columns)
        print(self.data.max)

    def get_data(self):
        return self.data

class EncodingData:
    def __init__(self,data):
        self.data = data
        self.labelencoder = LabelEncoder()

    def encoding_columns(self, columns):
        for column in columns:
            self.data[column] = self.labelencoder.fit_transform(self.data[column])
            print(self.data.head())

class One_hot_encoding:
    def __init__(self,data):
        self.data = data
        self.one_hot_encoding = OneHotEncoder()

    def one_hot_encoding_columns(self, columns):
        for column in columns:
            self.data[column] = self.one_hot_encoding.fit_transform(self.data[column])
            print(self.data.head())
            print(self.data["Education"])
            print(self.data["Vehicle Size"])
    
class Model:
    def __init__(self,data,x_columns,y_columns):
        self.data = data
        self.x = data[x_columns]
        self.y = data[y_columns]

    def train_and_test_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, train_size = 0.8, test_size = 0.2, random_state = 42)
        print(self.x_train)

        standard_scaler = StandardScaler()
        self.x_train_scaler = standard_scaler.fit_transform(self.x_train)
        self.x_test_scaler = standard_scaler.transform(self.x_test)

        self.linear_regression = LinearRegression()
        self.linear_regression.fit(self.x_train_scaler, self.y_train)
        self.y_predict = self.linear_regression.predict(self.x_test_scaler)
        print(self.y_predict)

        self.mse = mean_squared_error = (self.y_test, self.y_predict)
        print("mean_squared_error",self.mse)

        self.model_score = self.linear_regression.score(self.x_test_scaler, self.y_test)
        print("model_score",self.model_score)

        self.cross_validation = cross_val_score(self.linear_regression, self.x, self.y, cv = 5, scoring = "neg_mean_squared_error")
        print("cross_validation",self.cross_validation)

        self.r2_score = r2_score(self.y_test, self.y_predict)
        print("r2_score",self.r2_score)

        self.decision_tree_regression = DecisionTreeRegressor()

        self.decision_tree_regression.fit(self.x_train,self.y_train)
        self.y_prediction_decision_tree = self.decision_tree_regression.predict(self.x_test)
        print("decision tree",self.y_prediction_decision_tree)

        self.mean_squared_error = (self.y_test, self.y_prediction_decision_tree)
        print("decision_tree_mean_squared_error:",self.mean_squared_error)
        

    def get_xtrain_ytrain(self):
        return self.x_train,self.x_test,self.y_train,self.y_test

    def get_y_prediction(self):
        return self.y_predict
        print("no code error")
class Visualization:
    def __init__(self,x_train,x_test,y_train,y_test,y_predict):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_predict = y_predict

    def view_data(self):
        plt.figure(figsize = (8,10))
        plt.scatter(self.y_test,self.y_predict)
        plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()], "r--") 
        plt.title("Total Claim Amoumt")
        plt.xlabel("Actual Claim Amount")
        plt.ylabel("Predicted Claim Amount")
        plt.show()
    
 
data = Dataset("AutoInsurance.csv")
data.drop_columns(["State","Response","Coverage","Effective To Date","EmploymentStatus","Location Code","Number of Open Complaints"])

encoding = EncodingData(data.get_data())
encoding.encoding_columns(["Customer","Gender","Marital Status","Vehicle Class","Sales Channel","Renew Offer Type"])

one_hot_encoding = One_hot_encoding(data.get_data())
one_hot_encoding.one_hot_encoding_columns([["Education"]])
one_hot_encoding.one_hot_encoding_columns([["Vehicle Size"]])

model = Model(data.get_data(),["Customer","Gender","Marital Status","Customer Lifetime Value","Vehicle Class","Sales Channel","Renew Offer Type","Income","Monthly Premium Auto","Months Since Policy Inception","Number of Policies"],
              "Total Claim Amount")
model.train_and_test_data()
x_train, x_test, y_train, y_test = model.get_xtrain_ytrain()
y_predict = model.get_y_prediction()


view_data = Visualization(x_train,x_test,y_train,y_test,y_predict)
view_data.view_data()
