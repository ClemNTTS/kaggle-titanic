import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

Y = df['Survived']
df.drop(['Survived'], axis=1, inplace=True)

def preprocess_data(df):

    df.set_index('PassengerId', inplace=True)

    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Pclass'].fillna(df['Pclass'].median(), inplace=True)
    df['SibSp'].fillna(df['SibSp'].median(), inplace=True)
    df['Parch'].fillna(df['Parch'].median(), inplace=True)
    df['Sex'].fillna('unknow', inplace=True)

    df['Cabin'] = df['Cabin'].str.extract('([A-Z])')
    df['Cabin'].fillna('unknow', inplace=True)
    df['Cabin'] = pd.factorize(df['Cabin'])[0]

    df['Name'] = df['Name'].str.split(',').str[1]
    df['Name'] = df['Name'].str.split('.').str[0]
    df['Name'].fillna('unknow', inplace=True)

    df['Sex'] = pd.factorize(df['Sex'])[0]
    df['Name'] = pd.factorize(df['Name'])[0]
    df['Embarked'] = pd.factorize(df['Embarked'])[0]

    scaler = StandardScaler()
    df['Pclass'] = scaler.fit_transform(df[['Pclass']])
    df['Age'] = scaler.fit_transform(df[['Age']])
    df['SibSp'] = scaler.fit_transform(df[['SibSp']])
    df['Parch'] = scaler.fit_transform(df[['Parch']])
    df['Fare'] = scaler.fit_transform(df[['Fare']])
    df['Cabin'] = scaler.fit_transform(df[['Cabin']])
    df['Sex'] = scaler.fit_transform(df[['Sex']])
    df['Embarked'] = scaler.fit_transform(df[['Embarked']])
    df['Name'] = scaler.fit_transform(df[['Name']])

    df['Ticket'].fillna('unknow', inplace=True)
    df['Ticket'] = df['Ticket'].str.split(' ').str[0]
    df['Ticket'] = df['Ticket'].apply(lambda x: "unknown" if x.isdigit() else x)
    df['Ticket'] = pd.factorize(df['Ticket'])[0]
    df['Ticket'] = scaler.fit_transform(df[['Ticket']])
    return df

df = preprocess_data(df)

test_data = preprocess_data(test_data)

X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=1000)

# Comments are for my personnaly best result
param_grid = {
    'n_estimators': [100, 210, 320, 430,540], #100
    'max_depth': [30, 35, 40, 50, 60, 70, 80, 90, 100], #50
    'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], #8
    'min_samples_leaf': [3, 4, 5, 10, 20, 30, 40], #5
    'bootstrap': [True, False] #False
}

random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid,
                                   n_iter=1000, cv=5, verbose=2, n_jobs=-1)


random_search.fit(X_train, y_train)


print("Best Hyperparameters:", random_search.best_params_)

best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)

# Prédictions avec le modèle optimisé
y_pred = best_model.predict(X_test)
print('Optimized Accuracy:', accuracy_score(y_test, y_pred))


y_pred = best_model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))


y_pred_probs = best_model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_probs)

# Choisir le seuil basé sur le meilleur F1-score
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_idx = np.argmax(f1_scores)
best_threshold = thresholds[optimal_idx]
print(f'Best threshold: {best_threshold}')

# Prédire sur les données de test
test_pred_probs = best_model.predict_proba(test_data)[:, 1]
test_predictions = (test_pred_probs > 0.5).astype(int)

submission_df = pd.DataFrame({
    'PassengerId': test_data.index,
    'Survived': test_predictions.flatten()
})

print(submission_df.head())

submission_df.to_csv('submission.csv', index=False)