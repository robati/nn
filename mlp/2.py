import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.


X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

mlp = MLPClassifier(hidden_layer_sizes=(512,512,), max_iter=30,
                    verbose=10, batch_size=4,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)
print("score: %f" % mlp.score(X_test, y_test))

mlp = MLPClassifier(hidden_layer_sizes=(512,512,), max_iter=30,
                    verbose=10, batch_size=4,
                    learning_rate_init=.01)

mlp.fit(X_train, y_train)
print("score: %f" % mlp.score(X_test, y_test))

mlp = MLPClassifier(hidden_layer_sizes=(512,512,), max_iter=30,
                    verbose=10, batch_size=32,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)
print("score: %f" % mlp.score(X_test, y_test))
mlp = MLPClassifier(hidden_layer_sizes=(512,512,), max_iter=30,
                    verbose=10, batch_size=32,
                    learning_rate_init=.01)

mlp.fit(X_train, y_train)
print("score: %f" % mlp.score(X_test, y_test))