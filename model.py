import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import precision_recall_curve
from keras.layers import LeakyReLU

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y = train_data['Survived']

# Clean data
train_data = train_data.drop(['Ticket', 'Cabin', 'Embarked'], axis=1)
test_data = test_data.drop(['Ticket', 'Cabin', 'Embarked'], axis=1)

# Fill missing values in 'Age' column with mean age
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].mean())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())

# Set PassengerId as index for both datasets
train_data = train_data.set_index('PassengerId')
test_data = test_data.set_index('PassengerId')

# Combine train and test data
combined_data = pd.concat([train_data.drop('Survived', axis=1), test_data], axis=0)

# Encode categorical variables
combined_data = pd.get_dummies(combined_data, columns=['Name', 'Sex'], drop_first=True)

# Scale 'Age' and 'Fare' columns
scaler = StandardScaler()
combined_data[['Age', 'Fare']] = scaler.fit_transform(combined_data[['Age', 'Fare']])

# Sépare les données d'entraînement et de test à nouveau
train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):]



# Create and train the model
model = Sequential()
model.add(Input(shape=(train_data.shape[1],)))
model.add(Dense(32, activation=LeakyReLU(alpha=0.01)))
model.add(Dropout(0.2))
model.add(Dense(16, activation=LeakyReLU(alpha=0.01)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)


x_train, x_test, y_train, y_test = train_test_split(train_data, y, test_size=0.2, shuffle=True, stratify=y)

model.fit(x_train, y_train, epochs=2000, batch_size=32, callbacks=[early_stopping, reduce_lr], validation_split=0.1, verbose=1)

y_pred_probs = model.predict(x_test).flatten()
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_probs)

# Choose threshold based on highest F1-score or balance between precision and recall
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_idx = np.argmax(f1_scores)
best_threshold = thresholds[optimal_idx]
print(f'Best threshold: {best_threshold}')

model.evaluate(x_test, y_test)

predictions = model.predict(test_data)
predictions = (predictions > 0.5).astype(int)
submission_df = pd.DataFrame({
    'PassengerId': test_data.index,
    'Survived': predictions.flatten()
})

print(submission_df.head())

submission_df.to_csv('submission.csv', index=False)
