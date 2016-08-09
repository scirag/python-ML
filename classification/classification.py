from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

from sklearn.cross_validation import train_test_split

#test_size is ratio of test set over whole set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron

#n_iter : number of iterations, eta0 : learning rate
perceptron = Perceptron(n_iter=36, eta0=0.01, random_state=0)
perceptron.fit(X_train_std,y_train)

y_prediction = perceptron.predict(X_test_std)
print('Misclassified samples : %d' % (y_prediction!=y_test).sum())
print(X.shape)
print('Metric evaluation : ')
print('Accuracy : %.2f' % (1-((y_prediction!=y_test).sum()/X_test.shape[0])))

from sklearn.metrics import accuracy_score
print('Accuracy : %.2f (built-in)' % accuracy_score(y_test,y_prediction))

from plot_boundary import plot_decision_regions
import numpy as np

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plt = plot_decision_regions(X=X_combined_std, y=y_combined, classifier=perceptron, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('./figures/iris_perceptron_scikit.png', dpi=300)
plt.show()