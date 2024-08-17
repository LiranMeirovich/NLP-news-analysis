import csv
#from instagram_rivate_api import Client, ClientCompatPatch
import datetime

# Part 1 - data scraping
#def get_instagram_posts_descriptions(username, num_posts, user, password):
#    # Authenticate with the Instagram API
#    api = Client(user, password)
#
#    # Get the user ID
#    user_info = api.username_info(username)
#    user_id = user_info['user']['pk']

#    # Get user feed
#    posts = []
#    results = api.user_feed(user_id)
#    posts.extend(results.get('items', []))

#    # Keep fetching more posts until we have enough or there are no more posts
#    next_max_id = results.get('next_max_id')
#    while len(posts) < num_posts and next_max_id:
#        results = api.user_feed(user_id, max_id=next_max_id)
#        posts.extend(results.get('items', []))
#        next_max_id = results.get('next_max_id')

#    # Extract descriptions and timestamps, then save to CSV
#    with open(f'{username}_posts.csv', 'w', newline='', encoding='utf-8') as csvfile:
#        fieldnames = ['description', 'timestamp']
#        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#        writer.writeheader()

#        for post in posts[:num_posts]:
#            description = post['caption']['text'] if 'caption' in post and post['caption'] else ''
#            timestamp = datetime.datetime.fromtimestamp(post['taken_at']).strftime('%Y-%m-%d %H:%M:%S')
#            writer.writerow({'description': description, 'timestamp': timestamp})

username = 'guardian'
num_posts = 2500
user = ''
password = ''
#get_instagram_posts_descriptions(username, num_posts, user, password)

import pandas as pd
import re


def clean_description(text):
    # Remove all non-alphabetical and non-numerical symbols
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return cleaned_text


def clean_csv(file_name):
    # Read the input CSV file
    df = pd.read_csv(file_name)

    # Check if the 'description' column exists
    if 'description' not in df.columns:
        print(f"'description' column not found in {file_name}")
        return

    # Apply cleaning to the 'description' column
    df['cleaned_description'] = df['description'].apply(clean_description)

    # Save the results to a new CSV file
    output_file_name = 'cleaned_' + file_name.split('/')[-1]
    df.to_csv(output_file_name, index=False)
    print(f"Cleaned data saved to {output_file_name}")


# Part 2 - Tokenization, Lemmatization & Stop words removal

import pandas as pd
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import csv

# Download NLTK data files (if not already installed)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Stop words removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stopwords]

    return ' '.join(tokens)


def preprocess_csv(file_name):
    # Read the input CSV file
    df = pd.read_csv(file_name)

    # Check if the 'description' column exists
    if 'cleaned_description' not in df.columns:
        print(f"'cleaned_description' column not found in {file_name}")
        return

    # Apply preprocessing to the 'description' column
    df['processed_description'] = df['cleaned_description'].apply(preprocess_text)

    # Save the results to a new CSV file
    output_file_name = 'processed_' + file_name
    df.to_csv(output_file_name, index=False)
    print(f"Preprocessed data saved to {output_file_name}")


# Part 3 - Tokenization, Lemmatization & Stop words removal
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf(df):
    stop_words = list(set(stopwords.words('english')))

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(df['processed_description'].dropna())

    # Get feature names and scores
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    tfidf_scores = dict(zip(feature_names, scores))

    return tfidf_scores


# Function to plot bar chart
def plot_bar_chart(tfidf_scores):
    sorted_scores = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    words, scores = zip(*sorted_scores)

    plt.figure(figsize=(10, 8))
    plt.barh(words, scores, color='skyblue')
    plt.xlabel('TF-IDF Score')
    plt.title('Top 20 Words by TF-IDF Score')
    plt.gca().invert_yaxis()
    plt.show()


# Function to plot word cloud
def plot_word_cloud(tfidf_scores):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tfidf_scores)

    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of TF-IDF Scores')
    plt.show()


def displayPlot(file_name):
    # Read the preprocessed CSV file
    df = pd.read_csv(file_name)

    if 'processed_description' not in df.columns:
        print(f"'processed_description' column not found in {file_name}")
        return

    # Compute TF-IDF
    tfidf_scores = compute_tfidf(df)

    # Plot bar chart
    plot_bar_chart(tfidf_scores)

    # Plot word cloud
    plot_word_cloud(tfidf_scores)


# Example usage
file_name = 'processed_cleaned_guardian_posts.csv'
#displayPlot(file_name)
file_name = 'processed_cleaned_theeconomist_posts.csv'
#displayPlot(file_name)

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from collections import Counter

nltk.download('punkt')


# Function to tokenize text
def tokenize_text(text):
    return word_tokenize(text)


# Function to train Word2Vec and find most common words
def train_word2vec(df):
    # Tokenize the processed descriptions
    sentences = df['processed_description'].dropna().apply(tokenize_text).tolist()

    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    return model, sentences


# Function to find the most common words
def find_most_common_words(sentences):
    # Flatten the list of sentences to count word frequencies
    all_words = [word for sentence in sentences for word in sentence]
    word_freq = Counter(all_words)

    return word_freq


# Main function
def frequentWords(file_name):
    # Read the preprocessed CSV file
    df = pd.read_csv(file_name)

    if 'processed_description' not in df.columns:
        print(f"'processed_description' column not found in {file_name}")
        return

    # Train Word2Vec model
    model, sentences = train_word2vec(df)

    # Find most common words
    word_freq = find_most_common_words(sentences)

    # Print the most common words
    for word, freq in word_freq.most_common(20):
        print(f'{word}: {freq}')


# Example usage
file_name = 'processed_cleaned_guardian_posts.csv'
#frequentWords(file_name)
file_name = 'processed_cleaned_theeconomist_posts.csv'
#frequentWords(file_name)


# Part 5


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')


# Function to preprocess text: tokenize and remove stop words
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(tokens)


# Function to build and train an autoencoder
def build_and_train_autoencoder(X, encoding_dim):
    input_dim = X.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder.fit(X, X, epochs=50, batch_size=256, shuffle=True, validation_split=0.2, verbose=0)

    encoder = Model(inputs=input_layer, outputs=encoded)
    return encoder


# Function to identify the most important words using the trained autoencoder
def identify_important_words(encoder, vectorizer, feature_names):
    encoded_weights = encoder.get_weights()[0]
    word_importance = np.sum(encoded_weights, axis=1)
    important_words = sorted(zip(feature_names, word_importance), key=lambda x: x[1], reverse=True)
    return important_words


# Main function
def main(file_name):
    # Read the preprocessed CSV file
    df = pd.read_csv(file_name)

    if 'processed_description' not in df.columns:
        print(f"'processed_description' column not found in {file_name}")
        return

    # Preprocess the descriptions
    df['processed_description'] = df['processed_description'].apply(preprocess_text)

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_description'].dropna())
    X = X.toarray()
    feature_names = vectorizer.get_feature_names_out()

    # Build and train the autoencoder
    encoding_dim = 100  # Dimension of the encoded representation
    encoder = build_and_train_autoencoder(X, encoding_dim)

    # Identify the most important words
    important_words = identify_important_words(encoder, vectorizer, feature_names)

    # Print the top 20 most
    print("The 20 most important in " + file_name + "words are:")
    for word, importance in important_words[:20]:
        print(f"{word}: {importance}")

# Example usage
file_name = 'processed_cleaned_theeconomist_posts.csv'
#main(file_name)
file_name = 'processed_cleaned_guardian_posts.csv'
#main(file_name)

# Part 7
import pandas as pd
import random
import spacy

# Load the spaCy model for NER
nlp = spacy.load('en_core_web_sm')


# Function to run NER on a given text
def run_ner(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# Function to randomly choose 10 statements and run NER on each
def ner_on_random_statements(file_name):
    # Read the CSV file
    df = pd.read_csv(file_name)

    if 'cleaned_description' not in df.columns:
        print(f"'cleaned_description' column not found in {file_name}")
        return

    # Drop any NaN values in the 'processed_description' column
    descriptions = df['cleaned_description'].dropna().tolist()

    # Randomly choose 10 statements
    random_statements = random.sample(descriptions, 10)

    # Run NER on each statement and display the results
    for i, statement in enumerate(random_statements):
        print(f"Original Statement {i + 1}: {statement}")
        ner_result = run_ner(statement)
        print("NER Result:")
        for ent_text, ent_label in ner_result:
            print(f" - {ent_text}: {ent_label}")
        print("-" * 80)


# Example usage
file_name = 'processed_cleaned_guardian_posts.csv'
#ner_on_random_statements(file_name)
file_name = 'processed_cleaned_theeconomist_posts.csv'
#ner_on_random_statements(file_name)

#Part 8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Function to perform EDA
def perform_eda(file_name):
    # Read the CSV file
    df = pd.read_csv(file_name)

    if 'timestamp' not in df.columns:
        print(f"'timestamp' column not found in {file_name}")
        return

    # Convert the 'timestamp' column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract date and hour from the timestamp
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.to_period('M')  # Aggregate by month
    df['week'] = df['timestamp'].dt.to_period('W')  # Aggregate by week

    # Number of posts per day
    posts_per_day = df.groupby('date').size()
    plt.figure(figsize=(12, 6))
    posts_per_day.plot()
    plt.title('Number of Posts per Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Posts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Number of posts per hour
    posts_per_hour = df.groupby('hour').size()
    plt.figure(figsize=(12, 6))
    posts_per_hour.plot(kind='bar')
    plt.title('Number of Posts per Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Posts')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Posts distribution by month
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='month', palette='viridis')
    plt.title('Posts Distribution by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Posts')
    plt.xticks(rotation=45)
    plt.show()

    # Posts distribution by week
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='week', palette='viridis')
    plt.title('Posts Distribution by Week')
    plt.xlabel('Week')
    plt.ylabel('Number of Posts')
    plt.xticks(rotation=45)
    plt.show()

# Example usage
file_name = 'processed_cleaned_theeconomist_posts.csv'
#perform_eda(file_name)
file_name = 'processed_cleaned_guardian_posts.csv'
#perform_eda(file_name)

# Part 9

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')


# Function to perform sentiment analysis
def sentiment_analysis(df, column):
    sia = SentimentIntensityAnalyzer()
    sentiments = df[column].apply(lambda x: sia.polarity_scores(x)['compound'])
    return sentiments


# Function to categorize sentiments
def categorize_sentiments(sentiments):
    return sentiments.apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))


# Function to perform and compare sentiment analysis for two files
def compare_sentiment_analysis(file1, file2):
    # Read the CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Check for 'cleaned_description' column in both files
    if 'cleaned_description' not in df1.columns or 'cleaned_description' not in df2.columns:
        print("'cleaned_description' column not found in one of the files")
        return

    # Perform sentiment analysis
    sentiments1 = sentiment_analysis(df1, 'cleaned_description')
    sentiments2 = sentiment_analysis(df2, 'cleaned_description')

    # Categorize sentiments
    df1['sentiment'] = categorize_sentiments(sentiments1)
    df2['sentiment'] = categorize_sentiments(sentiments2)

    # Plot sentiment distributions
    plt.figure(figsize=(12, 6))
    sns.countplot(x='sentiment', data=df1, palette='viridis')
    plt.title(f'Sentiment Distribution for {file1}')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.countplot(x='sentiment', data=df2, palette='viridis')
    plt.title(f'Sentiment Distribution for {file2}')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

    # Compare sentiment trends
    df1['date'] = pd.to_datetime(df1['timestamp']).dt.date
    df2['date'] = pd.to_datetime(df2['timestamp']).dt.date

    sentiment_trends1 = df1.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
    sentiment_trends2 = df2.groupby(['date', 'sentiment']).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 6))
    sentiment_trends1.plot(kind='line', ax=plt.gca())
    plt.title(f'Sentiment Trends for {file1}')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sentiment_trends2.plot(kind='line', ax=plt.gca())
    plt.title(f'Sentiment Trends for {file2}')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Example usage
file1 = 'processed_cleaned_guardian_posts.csv'
file2 = 'processed_cleaned_theeconomist_posts.csv'
#compare_sentiment_analysis(file1, file2)

# Part 10
import pandas as pd
import random
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

LANGUAGE = "english"


# Function to perform KL-Sum summarization on a given text
def summarize_text(text):
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = KLSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = summarizer(parser.document, 1)  # Summarize to 1 sentence

    return ' '.join([str(sentence) for sentence in summary])


# Function to perform summarization and save to a new column
def perform_summarization(file_name):
    # Read the CSV file
    df = pd.read_csv(file_name)

    if 'cleaned_description' not in df.columns:
        print(f"'cleaned_description' column not found in {file_name}")
        return

    # Apply summarization to the 'cleaned_description' column
    df['summarized_description'] = df['cleaned_description'].apply(summarize_text)

    # Save the updated DataFrame to a new CSV file
    output_file_name = 'summarized_' + file_name.split('/')[-1]
    df.to_csv(output_file_name, index=False)
    print(f"Summarized data saved to {output_file_name}")

    return df


# Function to randomly choose 10 statements and show the summarized compared to the original
def show_random_statements(df, file_name):
    random_statements = df.sample(n=10)
    print(f"\nRandomly chosen statements from {file_name}:")
    for i, row in random_statements.iterrows():
        original = row['cleaned_description']
        summarized = row['summarized_description']
        print(f"\nOriginal Statement {i + 1}: {original}")
        print(f"Summarized Statement {i + 1}: {summarized}")


# Function to perform summarization on two files
def summarize_two_files(file1, file2):
    df1 = perform_summarization(file1)
    df2 = perform_summarization(file2)

    if df1 is not None:
        show_random_statements(df1, file1)
    if df2 is not None:
        show_random_statements(df2, file2)

#summarize_two_files(file1, file2)

# Part 11
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Function to train RNN and generate text
def train_and_generate_text(file_name, num_statements=10):
    # Read the CSV file
    df = pd.read_csv(file_name)

    if 'cleaned_description' not in df.columns:
        print(f"'cleaned_description' column not found in {file_name}")
        return

    # Assume cleaned_description column is a list of strings, each string being a message or a line of text
    cleaned_descriptions = df['cleaned_description'].dropna().tolist()

    # Tokenize the text
    test_texts = cleaned_descriptions[:3]  # Take the first 3 messages as test seeds
    cleaned_descriptions = cleaned_descriptions[3:]
    tokenizer = Tokenizer(num_words=5000)  # Limit to top 5000 words
    tokenizer.fit_on_texts(cleaned_descriptions)
    total_words = min(5000, len(tokenizer.word_index) + 1)

    # Create input sequences and output words
    input_sequences = []
    for line in cleaned_descriptions:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    # Pad sequences
    max_sequence_len = max([len(seq) for seq in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # Create predictors and label
    X, y = input_sequences[:, :-1], input_sequences[:, -1]

    # Convert y to one-hot encoding
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    # Define and compile the RNN model
    rnn_model = Sequential([
        Input(shape=(max_sequence_len - 1,)),
        Embedding(total_words, 16),  # Reduce embedding dimension
        SimpleRNN(32, activation='tanh'),  # Reduce RNN units
        Dense(total_words, activation='softmax')
    ])

    rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the RNN model
    rnn_model.fit(X, y, epochs=10, validation_split=0.1, batch_size=32, verbose=1)  # Reduce epochs and set batch size

    # Function to generate text
    def generate_text(seed_text, next_words, model, max_sequence_len):
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
            predicted = model.predict(token_list, verbose=0)
            predicted = np.argmax(predicted, axis=-1)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
        return seed_text

    # Generate 10 statements
    generated_statements = []
    for i in range(num_statements):
        seed = ' '.join(test_texts[i % len(test_texts)].split()[:3])  # Use first 3 words of each message as seed
        generated_statement = generate_text(seed, 5, rnn_model, max_sequence_len)
        generated_statements.append(generated_statement)
        print(f"Generated Statement {i + 1}: {generated_statement}")

    return generated_statements


file_name = 'processed_cleaned_theeconomist_posts.csv'
#generated_statements = train_and_generate_text(file_name, 10)

# Part 12
import os
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, \
    DataCollatorForLanguageModeling

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"


# Function to fine-tune GPT-2 and generate text
def fine_tune_gpt2(file_name, num_statements, model_dir="./gpt2_finetuned"):
    # Read the CSV file
    df = pd.read_csv(file_name)

    if 'cleaned_description' not in df.columns:
        print(f"'cleaned_description' column not found in {file_name}")
        return

    # Use a subset of the dataset for faster training
    cleaned_descriptions = df['cleaned_description'].dropna().tolist()[:5000]

    # Save cleaned descriptions to a text file for training
    with open("train_data.txt", "w", encoding="utf-8") as f:
        for line in cleaned_descriptions:
            f.write(line + "\n")

    # Load pre-trained distilGPT-2 model and tokenizer
    model_name = 'distilgpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Check if the model directory exists and contains the necessary files
    if os.path.exists(model_dir) and os.path.isfile(os.path.join(model_dir, 'config.json')):
        model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        print(f"Loaded fine-tuned model from {model_dir}")
    else:
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

        # Create TextDataset with a smaller block size for faster training
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path="train_data.txt",
            block_size=128  # Adjust block size
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Training arguments with mixed precision training
        training_args = TrainingArguments(
            output_dir=model_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,  # Increase epochs
            per_device_train_batch_size=8,  # Adjust batch size
            save_steps=10_000,
            save_total_limit=2,
            fp16=True,  # Enable mixed precision training
            learning_rate=5e-5  # Adjust learning rate
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        # Fine-tune the model
        trainer.train()

        # Save the model and tokenizer
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Saved fine-tuned model to {model_dir}")

    # Function to generate text with top-k and top-p sampling
    def generate_text(prompt, num_sequences):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=50,  # Generate up to 50 new tokens
            num_return_sequences=num_sequences,
            do_sample=True,  # Enable sampling
            top_k=50,  # Top-K sampling
            top_p=0.95,  # Top-P (nucleus) sampling
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        generated_outputs = [text[len(prompt):].strip() for text in generated_texts]
        return generated_outputs

    # Generate statements
    seed_texts = cleaned_descriptions[:3]  # Use first 3 cleaned descriptions as seeds
    generated_statements = []
    for seed in seed_texts:
        generated_texts = generate_text(seed, num_statements // len(seed_texts))
        generated_statements.extend(generated_texts)

    for i, statement in enumerate(generated_statements):
        print(f"Generated Statement {i + 1}: {statement}")

    return generated_statements




file_name = 'processed_cleaned_guardian_posts.csv'
num_statements = 10
generated_statements = fine_tune_gpt2(file_name, num_statements)
