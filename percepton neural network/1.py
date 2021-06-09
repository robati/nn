import  numpy
import pandas as pd

class Perceptron(object):

    def __init__(self, learningRate=1, bias=1):
        self.bias=bias
        self.weights = numpy.zeros(2+bias)
        self.epochs = 100
        self.learningRate = learningRate

    def getOutPut(self, x):
        return self.activationFunction(self.weights.T.dot(x))# the transpose matrix

    def getErrorvalue(self,d1,y):
        return d1-y

    def activationFunction(self, x):
        if((x >= 0)):
            return 1
        else:
            return 0

    def fit(self, X, d):
        for _ in range(self.epochs):
            # print(self.weights)
            for i in range(d.shape[0]):
                if self.bias == 1:
                    x = numpy.concatenate((X[i], [1]))
                else:
                    x = X[i]
                y = self.getOutPut(x)
                # print(x,y,d[i][0])
                e = self.getErrorvalue(d[i][0] , y)
                self.weights += e* self.learningRate * x

def test():

    path = "HW1_Q1_data.csv"
    files = pd.read_csv(path,usecols=["petal_length","sepal_length"]);
    files= files.iloc[0:, 0:2].values
    Result= pd.read_csv(path,usecols=["flower"]);
    Result= numpy.where(Result == 'Iris-setosa', 0, 1)

    a1=40
    a2=50
    a3=90
    files1=numpy.concatenate((files[0:a1] ,files[a2:a3]))
    files2=numpy.concatenate((files[a1:a2],files[a3:]))
    reseult1=numpy.concatenate((Result[0:a1],Result[a2:a3]))
    reseult2=numpy.concatenate((Result[a1:a2],Result[a3:]))

    X =files1
    d = reseult1
    bias=0
    perceptron = Perceptron(.1,bias)
    perceptron.fit(X, d)
    print(perceptron.weights)

    totalE=0
    for i in range(len(files2)):
        if(bias==1):
            x = numpy.concatenate((files2[i], [1]))
        else:
            x=files2[i]
        y = perceptron.getOutPut(x)
        e = perceptron.getErrorvalue(reseult2[i][0], y)
        # print(reseult2[i][0],y,x)
        if (e != 0):
            totalE+=1
    print(totalE)



def test1():
    X = numpy.array([
        [0.75, 0.75],
        [1, 1],
        [1.25, 1.5],
        [0, 1],
        [0.25, 1.75],
        [1, 2],
        [1.75, 1.75],
        [2, 1],
        [1.75, 0],
        [1,0],
        [0.25,0.25]
    ])
    d = numpy.array([[1],[ 1],[ 1], [0],[0],[0],[0],[0],[0],[0],[0]])
    bias=1
    perceptron = Perceptron(1,bias)
    perceptron.fit(X, d)
    print(perceptron.weights)

    totalE=0
    for i in range(len(X)):
        if(bias==1):
            x = numpy.concatenate((X[i], [1]))
        else:
            x=X[i]
        y = perceptron.getOutPut(x)
        e = perceptron.getErrorvalue(d[i][0], y)
        print(d[i][0],y,x)
        if (e != 0):
            totalE+=1
    print(totalE)

def test2():
        X = numpy.array([
            [0, 2],
            [0, 2.5],
            [0, 3],
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [1, -1],
            [2, -2],
            [3, -3]
        ])
        d = numpy.array([[1], [1], [1], [0], [0], [0], [0], [0], [0], [0]])
        bias = 1
        perceptron = Perceptron(1, bias)
        perceptron.fit(X, d)
        print(perceptron.weights)

        totalE = 0
        for i in range(len(X)):
            if (bias == 1):
                x = numpy.concatenate((X[i], [1]))
            else:
                x = X[i]
            y = perceptron.getOutPut(x)
            e = perceptron.getErrorvalue(d[i][0], y)
            print(d[i][0], y, x)
            if (e != 0):
                totalE += 1
        print(totalE)


if __name__ == '__main__':
    test2()
