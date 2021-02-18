import pandas
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
data = pandas.read_csv("iphone_price.csv")

# print(data.head())
# print(data.tall())
# plt.scatter(data["version"], data["price"])
# plt.bar(data["version"], data["price"])
# plt.show()

model = LinearRegression()
model.fit(data[["version"]], data[["price"]])
print(model.predict([[14]]))
print(model.predict([[20]]))