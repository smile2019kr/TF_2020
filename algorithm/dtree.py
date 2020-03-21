from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class DTree:
    def __init__(self):
        pass

    def breast_cancer(self):
        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, stratify=cancer.target, random_state=42
        )
        tree = DecisionTreeClassifier()
        tree.fit(X_train, y_train)
        print('train세트의 정확도: {: .3f}'.format(tree.score(X_train, y_train)))
        print('test세트의 정확도: {: .3f}'.format(tree.score(X_test, y_test)))
        """결과값
        train세트의 정확도:  1.000
        test세트의 정확도:  0.916
        """


    def iris(self):
        np.random.seed(0)
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        print(df.head())

        """결과값
                   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0                5.1               3.5                1.4               0.2
        1                4.9               3.0                1.4               0.2
        2                4.7               3.2                1.3               0.2
        3                4.6               3.1                1.5               0.2
        4                5.0               3.6                1.4               0.2
        """