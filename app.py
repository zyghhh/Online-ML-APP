import streamlit as st
from multipages import MultiPage
from page import (
    ensemble_learning,
    home,
    machine_learning,
    online_machine_learning,
    contrast_experiment,
)

st.set_page_config(page_title="ML", page_icon=":tiger:", layout="wide")
st.title("在线机器学习应用平台")

app = MultiPage()
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Online-ML")

# add applications
app.add("Home", home.app)
app.add("ML", machine_learning.app)
app.add("Online ML", online_machine_learning.app)
app.add("Ensemble learning", ensemble_learning.app)
app.add("Contrast experiment", contrast_experiment.app)
# app.add('Online SVR')
# Run application
if __name__ == "__main__":
    app.run()
