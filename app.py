import streamlit as st
import torch
import validators
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import requests
from bs4 import BeautifulSoup

# Load model and tokenizer
MODEL_PATH = "distilbert-fake-news.pt"  # Update with your model path
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Sidebar navigation using buttons instead of radio buttons
st.sidebar.title("Navigation")
home_button = st.sidebar.button("Home")
about_button = st.sidebar.button("About")
instructions_button = st.sidebar.button("Instructions")

# Set the default page based on button clicks
if home_button:
    page = "Home"
elif about_button:
    page = "About"
elif instructions_button:
    page = "Instructions"
else:
    page = "Home"  # Default page if no button is clicked yet

if page == "Home":
    st.title("📰 Fake News Detector")
    st.write("Select whether you're entering a news statement or a URL, and the model will predict if it's fake or real.")

    # Dropdown selector
    input_type = st.selectbox("Choose input type:", ["Text", "URL"])

    # Text area or URL input based on selection
    if input_type == "Text":
        user_input = st.text_area("Enter a news statement:")
    else:
        user_input = st.text_input("Enter a news article URL:")

    # Function to extract article text
    def extract_text_from_url(url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.text for p in paragraphs])
            return article_text
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching URL: {e}")
            return None

    if st.button("Check"):
        if user_input:
            if input_type == "URL":
                if validators.url(user_input):
                    st.info("Extracting article content from the URL...")
                    article_text = extract_text_from_url(user_input)
                    
                    if article_text:
                        user_input = article_text  # Replace input with extracted text
                    else:
                        st.error("Failed to extract text from the provided URL. Please enter a valid news link.")
                        st.stop()
                else:
                    st.error("Invalid URL. Please enter a valid news link.")
                    st.stop()

            # Tokenize input
            inputs = tokenizer(user_input, padding=True, truncation=True, max_length=128, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)

                # Probability scores
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                fake_prob = probabilities[0][0].item() * 100
                real_prob = probabilities[0][1].item() * 100 

                # Label determination
                label = "🛑 **Fake News**" if fake_prob > real_prob else "✅ **Real News**"

                # Display results
                st.subheader(f"Prediction: {label}")
                st.write(f"**Fake News Probability:** {fake_prob:.2f}%")
                st.write(f"**Real News Probability:** {real_prob:.2f}%")

                # Feedback section
                sentiment_mapping = ["one", "two", "three", "four", "five"]
                selected = st.slider("Rate the prediction:", 1, 5)
                if selected is not None:
                    st.markdown(f"You selected {sentiment_mapping[selected-1]} star(s).")
        
        else:
            st.warning("Please enter a news statement or URL.")

elif page == "About":
    st.title("ℹ️ About")
    st.write("""This Fake News Detection System is developed as a final project for the course, aiming to identify and classify news as real or fake using a DistilBERT model. 
The goal is to leverage natural language processing (NLP) and machine learning techniques to analyze news articles and provide accurate predictions.

The model is trained to process textual data, distinguishing between legitimate news and misinformation based on patterns it has learned from labeled datasets. 
Users can enter a news statement or a URL, and the system will analyze the content to determine its authenticity.

This project highlights the importance of AI in combating misinformation and demonstrates how transformer-based models like DistilBERT can be utilized in real-world applications to improve information credibility.""")


elif page == "Instructions":
    st.title("📖 Instructions")
    st.write("1. Select whether you're entering text or a URL.")
    st.write("2. Input the news statement or paste a news article URL.")
    st.write("3. Click the 'Check' button to analyze it.")
    st.write("4. View the prediction and rate the accuracy of the results.")
