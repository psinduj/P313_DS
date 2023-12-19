import streamlit as st
import pickle

# Load model
model = pickle.load(open('sentiment_analysis.pkl', 'rb'))

# Create title
st.title('Hotel Review Sentiment')
review = st.text_input('Enter your review')
submit = st.button('Predict')

if submit:
    prediction = model.predict([review])[0]

    # Emoji representation based on sentiment
    if prediction == 'Positive':
        emoji = '😃'
    elif prediction == 'Negative':
        emoji = '😞'
    else:
        emoji = '😐'

    st.write(f'Prediction: {prediction} {emoji}')






