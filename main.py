import streamlit as st
import pandas as pd
import numpy as np

# Your Streamlit code here
'''
st.write("Hello world")
data  = pd.read_csv("Book1.csv")

st.write(data)
'''
chartData = pd.DataFrame(np.random.randn(20,3),columns = ["a","b","c"])

st.bar_chart(chartData)
st.line_chart(chartData)
st.link_button("Profile",url ='/Page1.py')