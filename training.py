import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    df.head()
    train, test = data_split(df, 0.2)
    X_train = train[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].to_numpy()
    X_test = test[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].to_numpy()
    Y_train = train[['Outcome']].to_numpy().reshape(615 ,)
    Y_test = test[['Outcome']].to_numpy().reshape(153 ,)

    clf = LogisticRegression(solver='lbfgs', max_iter=3000).fit(X_train, Y_train)

    ## Open the file
    file = open('model.pkl', 'wb')

    # Dump information to that file
    pickle.dump(clf, file)
    file.close()





