from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron
perceptron = Perceptron(n_iter=36, eta0=0.01, random_state=0)
perceptron.fit(X_train_std,y_train)

y_prediction = perceptron.predict(X_test_std)
print('Misclassified samples : %d' % (y_prediction!=y_test).sum())
print(X.shape)
print('Accuracy : %.2f' % (1-((y_prediction!=y_test).sum()/X_test.shape[0])))

from sklearn.metrics import accuracy_score
print('Accuracy : %.2f (built-in)' % accuracy_score(y_test,y_prediction))
