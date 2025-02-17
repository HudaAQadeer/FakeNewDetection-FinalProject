import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load model and tokenizer
MODEL_PATH = "distilbert-fake-news.pt"  # Update with your saved model path
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Streamlit UI
st.title("Fake News Detector")
st.write("Enter a news statement below and the model will predict whether it is fake or real news.")

user_input = st.text_area("Enter a news statement:")

if st.button("Check"):
    if user_input:
        # Tokenize input
        inputs = tokenizer(user_input, padding=True, truncation=True, max_length=128, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)

            #putting in the probablity marker
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            fake_prob = probabilities[0][0].item() * 100 # this converts it to a percentage 
            real_prob = probabilities[0][1].item() * 100 

            # this determines the labels
            label = "🛑 **Fake News**" if fake_prob > real_prob else "✅ **Real News**"

            # displaying the results 

            st.subheader(f"Prediction: {label}")
            st.write(f"Fake News Probability: {fake_prob:.2f}%**")
            st.write(f"Real News Probability: {real_prob:.2f}%**")
            
        
        # Get prediction
        #prediction = torch.argmax(outputs.logits, dim=1).item()
        #label = "Fake News" if prediction == 0 else "Real News"
        
        st.progress(int(fake_prob))
    else:
        st.warning("Please enter a news statement.")
