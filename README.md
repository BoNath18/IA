import pandas as pd

def featureEngineering(data):
    dict_features = {}
    
    binary_features = []
    for column in ['Pclass','Sex','SibSp','Parch','Embarked']:
        for value in data[column].unique():
            binary_features.append(column+'_'+str(value))
            data[column+'_'+str(value)] = (data[column] == value).astype(int)
            
    decimal_features = ['Age','Fare']
    
    dict_features['decimal'] = decimal_features
    dict_features['binary'] = binary_features
    
    return data, dict_features
            

def preProcessing(data):
    data = data.fillna(-1)
    data['Sex'] = data['Sex'].astype(str)
    return data
    

train = pd.read_csv('C:\\Users\\Infoeste\\Desktop\\titanic\\train.csv', sep=',')
test = pd.read_csv('C:\\Users\\Infoeste\\Desktop\\titanic\\test.csv', sep=',')
data = train.append(test)

data = preProcessing(data)
data, dict_features = featureEngineering(data)

data[dict_features['binary']+dict_features['decimal']]

X_train = data.loc[data['Survived'] != -1, dict_features['binary']+dict_features['decimal']].values

Y_train = data.loc[data['Survived'] != -1, 'Survived'].values

X_test = data.loc[data['Survived'] == -1,dict_features['binary']+dict_features['decimal']].values

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(min_samples_split = 50, n_estimators=50)

model = model.fit(X_train, Y_train)

pred = model.predict(X_test)

final = pd.DataFrame()

final['PassengerId'] = data.loc[data['Survived'] == -1, 'PassengerId']
final['Survived'] = pred.astype(int)
final.to_csv('C:\\Users\\Infoeste\\Desktop\\titanic\\final.csv', index=False)

    
