import streamlit as st
import torch
import validators
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import requests
from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download


# Model configuration
MODEL_PATH = "distilbert_fake_news.pt"  # path of the model file in local storage
REPO_ID = "HudaAQadeer/fake-news-detector"  # Hugging Face repository ID
revision = "main"

# Load the tokenizer and model from Hugging Face or local storage
try:
    # Try loading tokenizer and model from Hugging Face Hub
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(REPO_ID)
    model = DistilBertForSequenceClassification.from_pretrained(REPO_ID)

    # Load the model state dict manually 
    model_state_dict = hf_hub_download(repo_id=REPO_ID, filename=MODEL_PATH, repo_type="model", revision=revision)
    model.load_state_dict(torch.load(model_state_dict, map_location=torch.device("cpu")))

except Exception as e:
    print(f"Error loading from Hugging Face: {e}")
    
    # If there's an issue with downloading or accessing the model from Hugging Face, loads it from local storage
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # If there is a manually downloaded model file, loads it here
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))

# Set model to evaluation mode (no gradients)
model.eval()

#Streamlit app UI

# sidebar navigation using buttons instead of radio buttons
st.sidebar.title("Navigation")
home_button = st.sidebar.button("Home")  # home button
about_button = st.sidebar.button("About")  # about button
instructions_button = st.sidebar.button("Instructions")  # instructions button

# set the default page based on button clicks
if home_button:
    page = "Home"
elif about_button:
    page = "About"
elif instructions_button:
    page = "Instructions"
else:
    page = "Home"  # default page if no button is clicked yet

if page == "Home":
    st.title("📰 Fake News Detector")
    st.write("Select whether you're entering a news statement or a URL, and the model will predict if it's fake or real.")

    # dropdown selector for input type (Text or URL)
    input_type = st.selectbox("Choose input type:", ["Text", "URL"])

    # text area or URL input based on selection
    if input_type == "Text":
        user_input = st.text_area("Enter a news statement:")
    else:
        user_input = st.text_input("Enter a news article URL:")

    # function to extract article text from URL
    def extract_text_from_url(url):
        try:
            response = requests.get(url)  # request the URL content
            response.raise_for_status()  # check for errors
            soup = BeautifulSoup(response.content, 'html.parser')  # parse HTML content
            paragraphs = soup.find_all('p')  # find all paragraphs
            article_text = ' '.join([p.text for p in paragraphs])  # extract text from paragraphs
            return article_text
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching URL: {e}")  # show error message
            return None

    if st.button("Check"):  # when button is clicked
        if user_input:
            if input_type == "URL":  # if input is URL
                if validators.url(user_input):  # check if URL is valid
                    st.info("Extracting article content from the URL...")
                    article_text = extract_text_from_url(user_input)
                    
                    if article_text:
                        user_input = article_text  # replace input with extracted text
                    else:
                        st.error("Failed to extract text from the provided URL. Please enter a valid news link.")
                        st.stop()  # stop execution if extraction fails
                else:
                    st.error("Invalid URL. Please enter a valid news link.")
                    st.stop()  # stop execution if URL is invalid

            # tokenize the input text
            inputs = distilbert_tokenizer(user_input, padding=True, truncation=True, max_length=128, return_tensors="pt")

            
            with torch.no_grad():  # no need for gradient calculations
                outputs = model(**inputs)  # forward pass through the model

                # probability scores for fake and real news
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                fake_prob = probabilities[0][0].item() * 100
                real_prob = probabilities[0][1].item() * 100 

                # determine the label based on probabilities
                label = "🛑 **Fake News**" if fake_prob > real_prob else "✅ **Real News**"

                # display the results
                st.subheader(f"Prediction: {label}")
                st.write(f"**Fake News Probability:** {fake_prob:.2f}%")
                st.write(f"**Real News Probability:** {real_prob:.2f}%")

                # feedback section with rating slider
                sentiment_mapping = ["one", "two", "three", "four", "five"]
                selected = st.slider("Rate the prediction:", 1, 5)
                if selected is not None:
                    st.markdown(f"You selected {sentiment_mapping[selected-1]} star(s).")
        
        else:
            st.warning("Please enter a news statement or URL.")  # warning if no input is given

elif page == "About":
    st.title("ℹ️ About")
    st.write("""  
This Fake News Detection System is developed as a final project for the course, aiming to identify and classify news as real or fake using a DistilBERT model.  
The goal is to leverage natural language processing (NLP) and machine learning techniques to analyze news articles and provide accurate predictions.  

The model is trained to process textual data, distinguishing between legitimate news and misinformation based on patterns it has learned from labeled datasets.  
Users can enter a news statement or a URL, and the system will analyze the content to determine its authenticity.  

This project highlights the importance of AI in combating misinformation and demonstrates how transformer-based models like DistilBERT can be utilized in real-world applications to improve information credibility.  
""")

elif page == "Instructions":
    st.title("📖 Instructions")
    st.write("1. Select whether you're entering text or a URL.")  # instruction step 1
    st.write("2. Input the news statement or paste a news article URL.")  # instruction step 2
    st.write("3. Click the 'Check' button to analyze it.")  # instruction step 3
    st.write("4. View the prediction and rate the accuracy of the results.")  # instruction step 4
