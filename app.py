import streamlit as st
import requests

LAMBDA_URL = "https://fwr3aqiugoebpn56twuf3c47le0opxap.lambda-url.eu-north-1.on.aws/"

st.title("AWS Lambda Streamlit App")

if st.button("Call Lambda Function"):
    response = requests.get(LAMBDA_URL).json()
    st.json(response)
