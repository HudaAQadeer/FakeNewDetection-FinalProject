from huggingface_hub import HfApi

# Initializing Hugging Face API client
api = HfApi()

# Uploading model file
api.upload_file(
    path_or_fileobj="distilbert_fake_news.pt",  # model file
    path_in_repo="distilbert_fake_news.pt",     # keeping the same filename
    repo_id="HudaAQadeer/fake-news-detector",  #hugging face repo
    repo_type="model"                           # specifying that it's a model
)
