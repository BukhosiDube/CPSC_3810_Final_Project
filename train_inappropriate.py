import kagglehub

# Download latest version
path = kagglehub.dataset_download("muhammadatef/english-profanity-words-dataset")

print("Path to dataset files:", path)
