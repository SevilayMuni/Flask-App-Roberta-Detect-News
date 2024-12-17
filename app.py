from flask import Flask, render_template, request
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import re


# Load model and tokenizer (optimized pipeline)
model_name = "SevilayG/Roberta-detect-unreliable-news"
classifier = pipeline("text-classification", model = model_name, tokenizer = model_name)

# Preprocessing function
def preprocess_input(title: str, text: str) -> str:
    """
    Preprocesses the input title and text by cleaning and combining them.
    
    Args:
        title (str): News title provided by the user.
        text (str): News body text provided by the user.

    Returns:
        str: Preprocessed and combined text.
    """
    def clean_text(input_text):
        # Remove URLs
        input_text = re.sub(r'http\S+|www.\S+', '', input_text)
        # Remove special characters
        input_text = re.sub(r'[^A-Za-z0-9\s]', '', input_text)
        # Collapse multiple spaces
        input_text = re.sub(r'\s+', ' ', input_text)
        return input_text.strip()

    cleaned_title = clean_text(title)
    cleaned_text = clean_text(text)
    return f"{cleaned_title} {cleaned_text}"

# Define label mapping
LABEL_MAPPING = {
    "LABEL_0": "✅ Reliable News",
    "LABEL_1": "❌ Unreliable News"}

# Initialize Flask app
app = Flask(__name__, template_folder = 'template')

# Routes
@app.route("/", methods=["GET"])
def home():
    """
    Renders the homepage with input forms for the user.
    """
    return render_template("index.html")

@app.route("/", methods=["POST"])
def predict_news():
    """
    Handles user input, processes the input, generates predictions, and displays the results.
    """
    # Retrieve input from form
    title = request.form.get("title")
    text = request.form.get("text")
    
    # Preprocess input
    combined_text = preprocess_input(title, text)
    
    # Generate prediction
    result = classifier(combined_text)[0]  # The pipeline returns a list of dicts
    raw_label = result["label"]  # Raw label from the model, e.g., "LABEL_0"
    label = LABEL_MAPPING.get(raw_label, "unknown") 
    confidence = result["score"]
    
    # Render the output
    return render_template(
        "index.html", 
        prediction_text=f"The news is classified as '{label}' with Confidence {confidence:.2f}.",
        original_text=f"Title: {title}\n\nText: {text}")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)