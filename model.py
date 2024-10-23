import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, accuracy_score
from sklearn.ensemble import RandomForestClassifier  # Importation de RandomForestClassifier

def extract_ticket_number(ticket):
    ticket_parts = ticket.split(" ")
    last_part = ticket_parts[-1]
    if last_part.isdigit():
        return int(last_part)
    else:
        return np.nan

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y = train_data['Survived']

# Clean data
train_data['ticket_number'] = train_data['Ticket'].apply(extract_ticket_number)
train_data['ticket_items'] = train_data['Ticket'].apply(lambda x: ' '.join(x.split(" ")[0:-1]))

test_data['ticket_number'] = test_data['Ticket'].apply(extract_ticket_number)
test_data['ticket_items'] = test_data['Ticket'].apply(lambda x: ' '.join(x.split(" ")[0:-1]))

train_data['Cabin_ABC'] = train_data['Cabin'].str.extract(r'([A-Za-z]+)', expand=False)
train_data['Cabin_Num'] = train_data['Cabin'].str.extract(r'(\d+)', expand=False).astype(float)

test_data['Cabin_ABC'] = test_data['Cabin'].str.extract(r'([A-Za-z]+)', expand=False)
test_data['Cabin_Num'] = test_data['Cabin'].str.extract(r'(\d+)', expand=False).astype(float)

train_data['Cabin_Num'] = train_data['Cabin_Num'].fillna(train_data['Cabin_Num'].median())
test_data['Cabin_Num'] = test_data['Cabin_Num'].fillna(test_data['Cabin_Num'].median())

train_data = train_data.drop(['Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].mean())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())

train_data['ticket_number'] = train_data['ticket_number'].fillna(train_data['ticket_number'].median())
test_data['ticket_number'] = test_data['ticket_number'].fillna(test_data['ticket_number'].median())

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

# Set PassengerId as index for both datasets
train_data = train_data.set_index('PassengerId')
test_data = test_data.set_index('PassengerId')

# Combine train and test data
combined_data = pd.concat([train_data.drop('Survived', axis=1), test_data], axis=0)

# Encode categorical variables
combined_data = pd.get_dummies(combined_data, columns=['Name', 'Sex', 'ticket_items', 'Cabin_ABC', 'Embarked'], drop_first=True)

# Scale 'Age' and 'Fare' columns
scaler = StandardScaler()
combined_data[['Age', 'Fare', 'ticket_number', 'Cabin_Num', 'FamilySize']] = scaler.fit_transform(combined_data[['Age', 'Fare', 'ticket_number', 'Cabin_Num', 'FamilySize']])

# Sépare les données d'entraînement et de test à nouveau
train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(train_data, y, test_size=0.2, shuffle=True, stratify=y)

# Créer le modèle de forêt aléatoire
model = RandomForestClassifier(n_estimators=500)  # n_estimators définit le nombre d'arbres dans la forêt

# Entraîner le modèle
model.fit(x_train, y_train)

# Faire des prédictions
y_pred_probs = model.predict_proba(x_test)[:, 1]  # Récupérer les probabilités de classe positive
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_probs)

# Choisir le seuil basé sur le meilleur F1-score
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_idx = np.argmax(f1_scores)
best_threshold = thresholds[optimal_idx]
print(f'Best threshold: {best_threshold}')

# Évaluer le modèle
y_pred = (y_pred_probs > best_threshold).astype(int)  # Utiliser le meilleur seuil
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Prédire sur les données de test
test_pred_probs = model.predict_proba(test_data)[:, 1]
test_predictions = (test_pred_probs > best_threshold).astype(int)

submission_df = pd.DataFrame({
    'PassengerId': test_data.index,
    'Survived': test_predictions.flatten()
})

print(submission_df.head())

submission_df.to_csv('submission.csv', index=False)
