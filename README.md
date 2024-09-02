# NLP Final Project

This repository contains the final project for the NLP course. The project explores various Natural Language Processing techniques, including tokenization, lemmatization, Tf-IDF, Word2Vec, and more.

There are 3 main code files for this project, we have decided to seperate them for ease of reading:
- Data_Collection.ipynb - this contains the code for collecting data from Instagram posts
- NLP_final_project.ipynb - the courses main tasks which is mainly different NLP techniques
- Sentimental Analysis.ipynb - in this part we've done some extra work analyzing the sentimental analysis of different news outlets regarding to 3 subjects and comparing between them.

The other files are mainly data files that's been cleaned and tokenized for ease of use by the program.

## Project Structure - NLP_final_project

- **Installation of Packages:** The project begins with the installation of the necessary Python packages for NLP.
  
- **Part 2 - Tokenization, Lemmatization & Stop Words Removal:**
  - This section covers the process of tokenizing and lemmatizing the text data.
  - Stop words are removed to focus on the most significant words in the dataset.
  - The processed text is saved into a new file.
  
- **Part 3 - Tf-IDF:**
  - Implementation of Term Frequency-Inverse Document Frequency (Tf-IDF) to convert the text data into a matrix of TF-IDF features.
  
- **Part 4 - Word2Vec:**
  - Utilizes the Word2Vec model to create word embeddings from the text data, capturing the semantic meaning of words.
  
- **Part 5 - Autoencoder:**
  - This section involves the use of an autoencoder, a type of neural network, for dimensionality reduction and feature extraction from the text data.
  
- **Part 6 - Comparison of Results:**
  - Compares the results between TF-IDF, Word2Vec, and Autoencoder models.
    
- **Part 7 - Named Entity Recognition (NER) and Categorization of Text:**
  - This section involves the use of NER to identify and categorize entities within the text data.
  
- **Part 8 - Exploratory Data Analysis (EDA):**
  - Performs EDA on the text data to uncover patterns and insights before further processing.
  
- **Part 9 - Sentiment Analysis:**
  - Implements sentiment analysis to determine the sentiment polarity (positive, negative, neutral) of the text data.
  
- **Part 10 - Summarization:**
  - This part focuses on generating concise summaries of the text data using NLP techniques.
  
- **Part 11 - RNN for Text Generation:**
  - Utilizes Recurrent Neural Networks (RNN) to generate text based on the patterns learned from the input data.
  
- **Part 12 - GPT for Text Generation:**
  - Implements GPT (Generative Pre-trained Transformer) for text generation, producing more grammatically correct and contextually coherent sentences compared to RNN.
  
- **Part 13 - Comparison:**
  - Compares the text generated by RNN and GPT.

## Sentimental Analysis: Israel-Hamas, Russia-Ukraine, and Armenia-Azerbaijan Wars

The additional Jupyter Notebook, `Sentimental Analysis.ipynb`, focuses on the sentiment analysis of media coverage during three significant conflicts: the Israel-Hamas war, the Russia-Ukraine war, and the Armenia-Azerbaijan war. This analysis aims to uncover the sentiment trends in different news outlets before and after key events in these conflicts.

This notebook provides a deeper understanding of how media sentiment can vary across different outlets and events, offering insights into potential biases and the impact of conflict on news reporting.


## How to Run

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <repository_name>
