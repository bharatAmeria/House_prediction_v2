import streamlit as st
import requests

LAMBDA_URL = "https://fwr3aqiugoebpn56twuf3c47le0opxap.lambda-url.eu-north-1.on.aws/"

response = requests.get(LAMBDA_URL)
print(response.status_code)  # Check if it's 200 OK
print(response.text)  # See the actual response content

try:
    data = response.json()  # Attempt to parse JSON
    print(data)
except requests.exceptions.JSONDecodeError as e:
    print(f"JSON decode error: {e}")

st.set_page_config(
    page_title="Gurgaon Real Estate Analytics App",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")