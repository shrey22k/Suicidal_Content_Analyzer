import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- Download NLTK data (only need to run this once) ---
print("Downloading NLTK stopwords...")
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
print("Download complete.")


# --- Text Preprocessing Function ---
def preprocess_text(text):
    """Cleans and preprocesses a single text string."""
    # 1. Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    
    # 2. Tokenize the text
    words = text.split()
    
    # 3. Remove stopwords and apply stemming
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
    
    # 4. Join the words back into a single string
    return " ".join(words)


# --- Main Training Logic ---
if __name__ == '__main__':
    print("Starting the model training process...")

    # 1. Load the dataset
    print("Loading dataset: Suicide_Detection.csv...")
    try:
        df = pd.read_csv('Suicide_Detection.csv')
    except FileNotFoundError:
        print("\nERROR: 'Suicide_Detection.csv' not found.")
        print("Please download it from Kaggle and place it in the same folder as this script.")
        exit()
        
    # For performance, let's use a smaller, balanced sample of the data.
    # The full dataset is very large and can be slow to train on a personal computer.
    print("Balancing and sampling the dataset for efficient training...")
    suicide_df = df[df['class'] == 'suicide'].sample(n=25000, random_state=42)
    non_suicide_df = df[df['class'] == 'non-suicide'].sample(n=25000, random_state=42)
    df_sample = pd.concat([suicide_df, non_suicide_df])
    
    print(f"Using a sample of {len(df_sample)} entries.")

    # 2. Preprocess the text data
    print("Preprocessing text data... (This may take a few minutes)")
    df_sample['processed_text'] = df_sample['text'].apply(preprocess_text)
    print("Preprocessing complete.")

    # 3. Define features (X) and target (y)
    X = df_sample['processed_text']
    y = df_sample['class']

    # 4. Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Vectorize the text data using TF-IDF
    print("Vectorizing text data with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("Vectorization complete.")

    # 6. Train a Logistic Regression model
    print("Training the Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    print("Model training complete.")

    # 7. Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")

    # 8. Save the trained model and the vectorizer
    print("Saving the model and vectorizer to disk...")
    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    print("\nTraining complete! 'model.joblib' and 'vectorizer.joblib' have been saved.")
