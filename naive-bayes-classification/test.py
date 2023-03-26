from sklearn.naive_bayes import GaussianNB
import numpy as np

clf = GaussianNB()
data = np.loadtxt("data/train.csv", delimiter=",")

clf.fit(data[:,0].reshape(-1,1), data[:,1])

print(clf.predict_proba([[35]]))