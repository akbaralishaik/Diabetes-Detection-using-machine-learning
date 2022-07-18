import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle



if __name__ == "__main__":
    data = pd.read_csv('data.csv')

    #Replacing all the values containing NaN
    data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN) 

    #replacing all the values containing NaN 
    data["Glucose"].fillna(data["Glucose"].mean(), inplace = True)
    data["BloodPressure"].fillna(data["BloodPressure"].mean(), inplace = True)
    data["SkinThickness"].fillna(data["SkinThickness"].mean(), inplace = True)
    data["Insulin"].fillna(data["Insulin"].mean(), inplace = True)
    data["BMI"].fillna(data["BMI"].mean(), inplace = True)
    
    #Used to scale data between 0 and 1
    sc = MinMaxScaler(feature_range = (0, 1))
    dataset_scaled = sc.fit_transform(data)
    dataset_scaled = pd.DataFrame(dataset_scaled)


    #Selecting only the required attributes
    X = dataset_scaled.iloc[:, [1, 4, 5, 7]].values
    Y = dataset_scaled.iloc[:, 8].values


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = data['Outcome'] )

    #Applying the algorithm as Decision Tree Classifier

    clf = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
    clf.fit(X_train, Y_train)

    ## Open the file
    file = open('model.pkl', 'wb')

    # Dump information to that file
    pickle.dump(clf, file)
    file.close()





