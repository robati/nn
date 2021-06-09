import numpy as np
from sklearn.datasets import load_iris



def kmeansClusterGetCenters(X, k, count):

    centroids = X[np.random.choice(range(len(X)), k, replace=False)]

    converged = False
    current_iter = 0

    while (not converged) and (current_iter < count):

        cluster_list = [[] for i in range(len(centroids))]

        for x in X:  # Go through each data point
            distances_list = []
            for c in centroids:
                distances_list.append(getDistance(c, x))
            cluster_list[int(np.argmin(distances_list))].append(x)

        cluster_list = list((filter(None, cluster_list)))

        prev_centroids = centroids.copy()

        centroids = []

        for j in range(len(cluster_list)):
            centroids.append(np.mean(cluster_list[j], axis=0))

        pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids))

        print('K-MEANS: ', int(pattern))

        converged = (pattern == 0)

        current_iter += 1

    return np.array(centroids), [np.std(x) for x in cluster_list]

def getDistance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)

class RBF:

    def __init__(self, X_Train, Y_Train, k):

        self.X = X_Train
        self.y = Y_Train
        self.k = k


    def convertClassLabel(self, x, k):
        arr = np.zeros((len(x), k))
        for i in range(len(x)):
            c = int(x[i])
            arr[i][c] = 1
        return arr

    def getHiddenLayerOutPut(self, x, c, s):
        distance = getDistance(x, c)
        return 1 / np.exp(-distance / s ** 2)

    def getHiddenLayerOutputForX(self, X, centroids, std_list):
        RBF_list = []
        for x in X:
            RBF_list.append([self.getHiddenLayerOutPut(x, c, s) for (c, s) in zip(centroids, std_list)])
        return np.array(RBF_list)

    def fit(self):

        self.centroids, self.std_list = kmeansClusterGetCenters(self.X, self.k, 1000)

        RBF_X = self.getHiddenLayerOutputForX(self.X, self.centroids, self.std_list)

        self.w = np.dot(np.dot(np.linalg.pinv(np.dot(RBF_X.T , RBF_X)) , RBF_X.T), self.convertClassLabel(self.y, self.k))


    def evaluate(self, X_Test, Y_Test):

        RBF_list_tst = self.getHiddenLayerOutputForX(X_Test, self.centroids, self.std_list)
        self.pred_ty = np.dot(RBF_list_tst ,self.w)
        self.pred_ty = np.array([np.argmax(x) for x in self.pred_ty])

        diff = self.pred_ty - Y_Test
        for i in range(len(diff)):
            print(self.pred_ty[i],Y_Test[i])
        print('Accuracy: ',len(np.where(diff == 0)[0]) / len(diff))

def test():
    files, Result = load_iris(return_X_y=True)
    a1=40
    a2=50
    a3=90
    a4=100
    a5=140
    files1=np.concatenate((files[0:a1] ,files[a2:a3],files[a4:a5]))
    files2=np.concatenate((files[a1:a2],files[a3:a4],files[a5:]))

    reseult1=np.concatenate((Result[0:a1] ,Result[a2:a3],Result[a4:a5]))
    reseult2=np.concatenate((Result[a1:a2],Result[a3:a4],Result[a5:]))

    X =files1
    d = reseult1

    CLASSIFIER = RBF(files1, reseult1, k=3)
    CLASSIFIER.fit()
    CLASSIFIER.evaluate(files2, reseult2)


test()

