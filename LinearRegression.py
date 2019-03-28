import matplotlib.pyplot as plot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = pd.read_csv('delay.csv')

if data is not None:
    x = data['ArrDelay']    # Also called input value or predictor.
    y = data['DepDelay']    # Also called outcome value or response variable.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/4, random_state= 0)

    x_train = x_train.values.reshape(-1,1)
    x_test = x_test.values.reshape(-1,1)

    linear_reg = LinearRegression()
    linear_reg.fit(x_train, y_train)

    y_predict = linear_reg.predict(x_test)

    plot.scatter(x_train, y_train, color='red')
    plot.plot(x_train, linear_reg.predict(x_train), color='blue')
    plot.title('Flight arrival and departure delay (Training set | Sci-kit)')
    plot.xlabel('Arrival Delay')
    plot.ylabel('Departure Delay')
    plot.show()

    plot.scatter(x_test, y_test, color='red')
    plot.plot(x_test, linear_reg.predict(x_test), color='blue')
    plot.title('Flight arrival and departure delay (Test set | Sci-kit)')
    plot.xlabel('Arrival Delay')
    plot.ylabel('Departure Delay')
    plot.show()


else:
    print("File read failure")