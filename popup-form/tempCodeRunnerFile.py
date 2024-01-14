import pandas as pd
import nltk
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import pos_tag

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'form_data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the DataFrame
print(df)

df['visit_frequency'] = df['visit_frequency'].astype('category')

# Aggregate data
grouped_data = df.groupby('visit_frequency').agg({
    'comments': lambda x: ' '.join(x),
    'nps_rating': 'mean'
}).reset_index()

# Display aggregated data
print(grouped_data)

def clean_text(text):
    # Remove special characters, digits, and extra spaces
    text = re.sub(r'[^A-Za-z\s]', '', str(text))
    text = re.sub(r'[^\w\s]', '', text)
    return text

def analyze_sentiment(comment):
    sid = SentimentIntensityAnalyzer()
    sentiment_score = sid.polarity_scores(comment)['compound']

    # Classify sentiment based on the compound score
    if sentiment_score >= 0.05:
        return 'Positive'
    elif sentiment_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['clean_comments'] = df['comments'].apply(clean_text)
# Apply sentiment analysis to the DataFrame
df['sentiment'] = df['comments'].apply(analyze_sentiment)

# Function to analyze common words, removing stop words
def analyze_common_words(comments):
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for comment in comments for word in word_tokenize(comment) if word.lower() not in stop_words]
    # Part-of-speech tagging
    tagged_tokens = pos_tag(tokens)
    # Filter out verbs
    tokens_no_nouns_verbs = [word for word, pos in tagged_tokens if pos not in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] or analyze_sentiment(word) not in ['Positive', 'Negative']]
    freq_dist = FreqDist(tokens_no_nouns_verbs)
    filtered_terms = [(term, freq) for term, freq in freq_dist.items() if freq >= 2]
    return filtered_terms # Adjust the number of top words as needed

# Analyze common words by visit frequency group

common_words_by_group = df.groupby('visit_frequency')['comments'].apply(analyze_common_words)

# Display results
print("Sentiment Analysis Results:")
print(df[['comments', 'sentiment']])

def is_special_character(s):
    # Use the regex pattern to check if the string contains only special characters
    pattern = re.compile('^[^A-Za-z0-9]+$')
    return bool(pattern.match(s))

print("\nAreas that need to be looked into for different groups of people visiting:")
for group, common_words in common_words_by_group.items():
    print(f"\nVisit Frequency: {group}")
    for term, frequency in common_words:
        if(frequency >= 2 and is_special_character(term)!=True):
            print(term)
    #print(common_words)

#for term, frequency in common_words:
   # print(f"Word: {term}, Frequency: {frequency}")

