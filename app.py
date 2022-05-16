from asyncore import write
from re import X
import streamlit as st
import pandas as pd
from sklearn import datasets
import numpy as np
from catboost import CatBoostClassifier
# Inputs
age = st.sidebar.slider('Age', 0, 100,1)
nwith = st.sidebar.slider('Number of people with you abroad', 0, 10,0)
embarked = st.sidebar.radio('Choose port of port of embarkation', ['Cherbourg', 'Queenstown','Southampton'])
gender = st.sidebar.radio('Gender', ['Female', 'Male'])
pclass  = st.sidebar.radio('Passenger Class',['First Class','Second Class','Third Class'])

if embarked == 'Cherbourg':
    Q = 0
    S = 0
elif embarked == 'Queenstown':
    Q = 1
    S = 0 
else: 
    Q=0
    S=1
if pclass == 'First Class':
    pclass = 1
elif pclass == 'Second Class':
    pclass = 2
else: 
    pclass = 3 

gender = 1 if gender =="Male" else 0

data = {'pclass' : pclass,
        'sex': gender,
        'age': age,
        'Q': Q,
        'S': S,
        'nwith': nwith}



df = pd.DataFrame(data, index = [0])


st.write('## Would You Have Survived the Titanic Disaster?')
st.image('titanic.jpg',width=500)
st.write('### The application  predict your  probability of survival ')

# Model
data = pd.read_csv('df_final.csv')
X = data.drop('survived',axis=1)
y = data['survived']


model = CatBoostClassifier(depth=4,early_stopping_rounds=3,iterations=5,
                    l2_leaf_reg=5,learning_rate=0.75,random_state=606).fit(X,y)
pred = model.predict(df)
probability = round(np.max(model.predict_proba(df)) * 100)
prediction = 'die' if pred[0] ==0 else 'survive'
st.write(f'### You would {prediction} with probability of {probability} % ')



