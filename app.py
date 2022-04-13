import streamlit as st
import pandas as pd
import numpy as np

def submit_form():
    CSV_row = '\n' + username_input + ',' + comment_input
    with open('comments.csv','a') as fd:
        fd.write(CSV_row)


st.title('Counter Example')
if 'count' not in st.session_state:
    st.session_state.count = 0

increment = st.button('Increment')
if increment:
    st.session_state.count += 1

st.write('Count = ', st.session_state.count)

## comments
username_input = st.text_input("username", "user")
comment_input = st.text_input("write a comment", "comment")
st.button(label="Submit", help=None, on_click=submit_form, args=None, kwargs=None)


df = pd.read_csv("comments.csv")

st.dataframe(df)  # Same as st.write(df)
#st.line_chart(df['fare_amount'])