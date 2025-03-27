Fake News Detection System

This repository contains the code for the Fake News Detection System, developed as a final project for the course. The system leverages a DistilBERT model to classify news statements as real or fake using natural language processing (NLP) techniques.

Features

Classifies news as real or fake using a trained DistilBERT model.

Supports both text input and URL-based news extraction.

Provides probability scores for each classification.

Includes a user feedback system to rate predictions.



1. Installation & Setup
   Clone the Repository Clone the repository to your local machine:

    
       git clone https://github.com/HudaAQadeer/FakeNewsDetection-FinalProject.git
       cd FakeNewsDetection-FinalProject


2. Create and Activate a Virtual Environment

   
   Create a virtual environment and activate it:

   For macOS/Linux:
        
        python3 -m venv fnvenv
        source fnvenv/bin/activate

   For Windows:
        
        python -m venv fnvenv
        fnvenv\Scripts\activate

   

4. Install Dependencies
   Install the required dependencies using pip:

        pip install -r requirements.txt


5. Download the Model

    The model file (distilbert_fake_news.pt) will be automatically downloaded from Hugging Face upon running the app, as the model is stored publicly on the Hugging Face Hub. You do not need to manually download the model.

6. Run the Streamlit Application

    After setting everything up, run the application using Streamlit:

        streamlit run app.py

This will start the app and open it in your browser, where you can interact with the Fake News Detection System.

Project Structure

FakeNewsDetection-FinalProject/

│-- app.py                  # Main Streamlit app

│-- requirements.txt        # Required dependencies

│-- FakeNewsDetection.ipynb # Jupyter Notebook for model training/testing

│-- UnitTests.py            # Unit tests for the project

│-- train.tsv               # Training dataset

│-- valid.tsv               # Validation dataset

│-- test.tsv                # Test dataset

│-- .gitignore              # Ignore unnecessary files

│-- README.md               # Project documentation

│-- upload_model.py         # Model upload script


Notes

Ensure your Python environment is set up correctly before running the application.

If Streamlit doesn't run correctly, try reinstalling it:

    pip install streamlit --upgrade

The model is trained on a limited and small dataset, so accuracy may vary.

License
This project is for educational purposes only.