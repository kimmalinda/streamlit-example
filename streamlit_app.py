import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.title('Classifying Iris Flowers')
st.markdown('Toy model to play to classify iris flowers into setosa, versicolor, virginica')
st.header('Plant Features')
col1, col2 = st.columns(2)
with col1:
  st.text('Sepal characteristics')
  sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
  sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)
with col2:
  st.text('Pepal characteristics')
  petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
  petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

r = st.button('Predict type of Iris')
def predict(data):
  clf = joblib.load('rf_model.sav')
  return clf.predict(data)

if r:
  result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
  st.text(result[0])

# Read original dataset
iris_df = pd.read_csv('iris.csv')
iris_df.sample(frac=1)
# selecting features and target data
X = iris_df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = iris_df[['variety']]
# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)
# train the classifier on the training data
clf.fit(X_train, y_train)
# predict on the test set
y_pred = clf.predict(X_test)
# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}') #Accuracy: 0.91

joblib.dump(clf, 'rf_model.sav')




