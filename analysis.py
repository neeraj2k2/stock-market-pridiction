# pip install numpy  - lets us perform calculations on our data
# pip install scikit-learn  - lets us build a predictive model
# pip intall matplotlib  - lets us plot the data in graphs - it is a graphical library
from fileinput import filename
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plot

#plot.switch_backend('TkAgg')
#os.listdir(STOCK-)
dates = []
prices = []

# getting data into dates and prices
def getData(filename):
    with open(filename, 'r') as csvfile:
        csvfilereader = csv.reader(csvfile)
        next(csvfilereader)  # it is gone to the 2nd row, we skipped first as it is just 
        for row in csvfilereader:
            # if(int(row[2])>300):
            #     pdates.append([int(row[2]),float(row[6])] )
            #     #prices.append(float(row[6]))
            #     continue
            dates.append(int(row[2]))
            prices.append(float(row[6]))
                
    return

getData('RELIANCE.NS.csv')

# splitting data into xtrain ytrain and x test and y test
traindates = dates[50:]
testdates = dates[:50]
trainprices = prices[50:]
testprices = prices[:50]

traindates = np.reshape(traindates, (len(traindates),1))
testdates = np.reshape(testdates, (len(testdates),1))
trainprices = np.reshape(trainprices, (len(trainprices),1))
prices = np.reshape(prices,(len(prices),1))
dates = np.reshape(dates,(len(dates),1))




# we use regression (SVR)as we want to predict how much the next opening price will be

#finetuning the paramerers
# c_arr = [0.1,1,10,100,1000]
# eps_arr =[1,0.1,0.01,0.001,0.0001,0.00001]
# for c in c_arr:
#     for ep in eps_arr:
#         model = SVR(kernel='rbf', C=c, epsilon=ep)
#         svr = model.fit(train)


# Donot use as this sorts all the dates in random ways and we need dates to be in a specific order 
# X_train, X_test, Y_train, Y_test = train_test_split( prices,dates, test_size=0.3, random_state=1 )
# def predictprice(Y_train,Y_test,X_train):
#     # traindates = np.reshape(traindates, (len(traindates),1))
#     # testdates = np.reshape(testdates, (len(testdates),1))
#     # trainprices = np.reshape(trainprices, (len(trainprices),1))

#     # svr_lin = SVR(kernel ='linear', C=1e3)
#     # svr_poly = SVR(kernel='poly', C=1e3, degree=2)
#     svr_rbf = SVR(kernel='rbf', C=1e3,gamma = 0.1)
#     # svr_lin.fit(dates,prices)
#     # svr_poly.fit(dates,prices)
#     svr_rbf.fit(Y_train,X_train)

#     plot.scatter(Y_train,X_train, color = 'black',label = 'Data')
#     # plot.plot(dates,svr_lin.predict(dates), color = 'green', label = 'Linear model')
#     # plot.plot(dates,svr_poly.predict(dates), color = 'blue', label = 'Polynomial model')
#     plot.plot(Y_test,svr_rbf.predict(Y_test), color = 'red', label = 'Rbf model')
#     plot.xlabel('Data')
#     plot.ylabel('Price')
#     plot.title('Support vector regression')
#     plot.legend()
#     plot.show()

# predictprice(Y_train,Y_test,X_train)


def predictprice(traindates,testdates,trainprices,testprices):
    

    # svr_lin = SVR(kernel ='linear', C=1e3)
    # svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3,gamma = 0.1, epsilon=0.001 )
    # svr_lin.fit(dates,prices)
    # svr_poly.fit(dates,prices)
    svr_rbf.fit(dates,prices)

    plot.plot(traindates,trainprices, color = 'black',label = 'Data')
    # plot.plot(testdates,testprices, color = 'blue',label = 'actual data')

    # plot.plot(dates,svr_lin.predict(dates), color = 'green', label = 'Linear model')
    # plot.plot(dates,svr_poly.predict(dates), color = 'blue', label = 'Polynomial model')
    plot.plot(testdates,svr_rbf.predict(testdates), color = 'red', label = 'Rbf model')
    plot.xlabel('Data')
    plot.ylabel('Price')
    plot.title('Support vector regression')
    plot.legend()
    plot.show()


predictprice(traindates,testdates,trainprices,testprices)

