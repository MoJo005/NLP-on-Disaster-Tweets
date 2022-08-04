import streamlit as st
from main import main_function
import time


def main():

    st.title('Disaster Tweets Classification')
    text= st.text_area('Type your Tweet here')

    if st.button('Predict'):
        with st.spinner('BERT is predicting, please wait'):
            time.sleep(6)
        # p=st.text('Predicting!!')
        prediction=main_function(text)
        if prediction[0][0]==1:
            st.subheader("This Tweet is related to a Real Disaster.")
        else:
            st.subheader('This Tweet is not related to a Real Disaster.')


if __name__=='__main__':
    main()