# Deployment of Telecom Churn project using Streamlit app

import streamlit as st
import pandas as pd
import dill

st.header("Telecommunication Customer Churn Prediction Model")
st.write("Required Action: Please complete every field below.")

# taking user inputs
states = ['KS', 'OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI', 'IA', 'MT', 'NY', 'ID', 'VT', 'VA', 'TX', 'FL', 
          'CO', 'AZ', 'SC', 'NE', 'WY', 'HI', 'IL', 'NH', 'GA', 'AK', 'MD', 'AR', 'WI', 'OR', 'MI', 'DE', 'UT', 'CA', 'MN', 
          'SD', 'NC', 'WA', 'NM', 'NV', 'DC', 'KY', 'ME', 'MS', 'TN', 'PA', 'CT', 'ND']

state_id = st.selectbox("Please select State", options=states)
area_code = st.selectbox("Enter Area Code", options=['area_code_415', 'area_code_408', 'area_code_510'])
account_length = st.number_input("Enter Account Length", min_value=0)
voice_plan = st.selectbox("Is Voice Plan active?", options=['yes', 'no'])
voice_messages = st.slider("Select number of Voice Messages", 0.0, 55.0)
intl_plan = st.selectbox("Is International Plan active?", options=['yes', 'no'])
intl_mins = st.slider("Total minutes spent on international calls", 0.0, 25.0)
intl_calls = st.slider("Total number of international calls made", 0, 25)
intl_charge = st.number_input("Total billed amount for international calls", 0.0, 6.0)
day_mins = st.number_input("Total minutes spent on calls during the day time", 0.0, 400.0)
day_calls = st.number_input("Total number of calls made during the day time", 0, 200)
day_charge = st.number_input("Total billed amount for day calls", 0.0, 60.0)
eve_mins = st.number_input("Total minutes spent on calls during the evening time", 0.0, 400.0)
eve_calls = st.number_input("Total number of calls made during the evening time", 0, 200)
eve_charge = st.number_input("Total billed amount for evening calls", 0.0, 40.0)
night_mins = st.number_input("Total minutes spent on calls during the night time", 0.0, 400.0)
night_calls = st.number_input("Total number of calls made during the night time", 0, 200)
night_charge = st.number_input("Total billed amount for night calls", 0.0, 20.0)
customer_calls = st.slider("Number of customer calls", 0, 10)

# encoding
state_freq={'WV': 0.032185064118682424, 'MN': 0.024893135529293436, 'TX': 0.024641689715866232, 'VA': 0.024641689715866232, 'AL': 0.024641689715866232, 'OR': 0.02388735227558461, 'NY': 0.02388735227558461, 'OH': 0.023133014835302994, 'UT': 0.022881569021875787, 'NJ': 0.022881569021875787, 'ID': 0.022378677395021373, 'WY': 0.02187578576816696, 'WA': 0.02137289414131255, 'MD': 0.02137289414131255, 'MA': 0.02137289414131255, 'ME': 0.020870002514458134, 'KY': 0.020870002514458134, 'WI': 0.020870002514458134, 'CT': 0.020618556701030927, 'MT': 0.02036711088760372, 'MO': 0.02036711088760372, 'MI': 0.020115665074176513, 'RI': 0.01986421926074931, 'VT': 0.019612773447322103, 'NM': 0.019361327633894896, 'NC': 0.018858436007040482, 'CO': 0.018606990193613275, 'TN': 0.018606990193613275, 'MS': 0.018606990193613275, 'KS': 0.01835554438018607, 'IN': 0.01835554438018607, 'FL': 0.018104098566758865, 'NH': 0.017852652753331658, 'NE': 0.017852652753331658, 'OK': 0.017852652753331658, 'GA': 0.017349761126477244, 'AR': 0.017349761126477244, 'LA': 0.017349761126477244, 'DC': 0.017098315313050037, 'PA': 0.017098315313050037, 'IL': 0.017098315313050037, 'ND': 0.017098315313050037, 'AZ': 0.017098315313050037, 'DE': 0.01684686949962283, 'NV': 0.01684686949962283, 'HI': 0.016595423686195626, 'SD': 0.016595423686195626, 'SC': 0.016595423686195626, 'AK': 0.014332411365350767, 'IA': 0.012823736484787528, 'CA': 0.009806386723661051}

state = state_freq.get(state_id)

# dataframe
df = pd.DataFrame([[state, area_code, account_length, voice_plan, voice_messages, intl_plan, intl_mins, intl_calls, 
                    intl_charge, day_mins, day_calls, day_charge, eve_mins, eve_calls, eve_charge, night_mins, 
                    night_calls, night_charge, customer_calls]],
                  columns=['state_freq', 'area.code', 'account.length', 'voice.plan',
                           'voice.messages', 'intl.plan', 'intl.mins', 'intl.calls', 'intl.charge',
                           'day.mins', 'day.calls', 'day.charge', 'eve.mins', 'eve.calls',
                           'eve.charge', 'night.mins', 'night.calls', 'night.charge',
                           'customer.calls'])

# feature engineering
df['total.mins'] = df[['intl.mins','day.mins','eve.mins','night.mins']].sum(axis=1)
df['intl_ratio'] = df['intl.mins'] / df['total.mins']
df['day_ratio'] = df['day.mins'] / df['total.mins']
df['eve_ratio'] = df['eve.mins'] / df['total.mins']
df['night_ratio'] = df['night.mins'] / df['total.mins']

for col, charge_col in [('day.mins','day.charge'), ('night.mins','night.charge'),
                        ('intl.mins','intl.charge'), ('eve.mins','eve.charge')]:
    df[f'{charge_col}_per_min'] = df[charge_col] / df[col].replace(0, 1)

# load transformer & model
with open('transformer.pkl', 'rb') as file:
    transformer = dill.load(file)

with open('best_model.pkl', 'rb') as file:
    model =dill.load(file)

# transform & predict
df_transformed = transformer.transform(df)
prediction = model.predict(df_transformed)

# show result
if st.button("Predict"):
    if prediction[0] == 1:
        st.error("Customer likely to CHURN")
    else:
        st.success("Customer likely to STAY")
