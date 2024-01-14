import pandas as pd
import panel as pn
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

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

file_path = 'form_data.csv'

df = pd.read_csv(file_path)

print(df)

def clean_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', str(text))
    text = re.sub(r'[^\w\s]', '', text)
    return text

def analyze_sentiment(comment):
    sid = SentimentIntensityAnalyzer()
    sentiment_score = sid.polarity_scores(comment)['compound']

    if sentiment_score >= 0.05:
        return 'Positive'
    elif sentiment_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['clean_comments'] = df['comments'].apply(clean_text)
df['sentiment'] = df['comments'].apply(analyze_sentiment)

def analyze_common_words(comments):
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for comment in comments for word in word_tokenize(comment) if word.lower() not in stop_words]
 
    tagged_tokens = pos_tag(tokens)
    tokens_no_nouns_verbs = [word for word, pos in tagged_tokens if pos not in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] or analyze_sentiment(word) not in ['Positive', 'Negative']]
    freq_dist = FreqDist(tokens_no_nouns_verbs)
    filtered_terms = [(term, freq) for term, freq in freq_dist.items() if freq >= 2]
    return filtered_terms 

common_words_by_group = df.groupby('visit_frequency')['comments'].apply(analyze_common_words)

print("Sentiment Analysis Results:")
print(df[['comments', 'sentiment']])

df['visit_frequency'] = df['visit_frequency'].astype('category')

grouped_data = df.groupby('visit_frequency').agg({
    'comments': lambda x: ' '.join(x),
    'nps_rating': 'mean'
}).reset_index()

print(grouped_data)

def is_special_character(s):
    pattern = re.compile('^[^A-Za-z0-9]+$')
    return bool(pattern.match(s))

print("\nAreas that need to be looked into for different groups of people visiting:")
for group, common_words in common_words_by_group.items():
    print(f"\nVisit Frequency: {group}")
    for term, frequency in common_words:
        if(frequency >= 2 and is_special_character(term)!=True):
            print(term)


df = pd.read_csv('form_data.csv')

def calculate_nps_category(nps_rating):
    if nps_rating >= 9:
        return 'Promoter'
    elif 7 <= nps_rating <= 8:
        return 'Passive'
    else:
        return 'Detractor'

df['nps_category'] = df['nps_rating'].apply(calculate_nps_category)

# Panel Dashboard
pn.extension()

def nps_distribution_plot(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='nps_rating', kde=True, bins=10, hue='nps_category', multiple='stack')
    plt.title('NPS Distribution')
    plt.xlabel('NPS Rating')
    plt.ylabel('Count')
    plt.legend(title='NPS Category')
    return pn.pane.Matplotlib(plt.gcf())

colors = {'Promoter': '#65c2a5', 'Passive': '#8da0cb', 'Detractor': '#e789c5'}

def nps_pie_chart(df):
    pie_chart_data = df['nps_category'].value_counts().reset_index()
    pie_chart_data.columns = ['nps_category', 'count']
    fig = px.pie(pie_chart_data, names='nps_category', values='count', title='NPS Distribution Pie Chart',color='nps_category', color_discrete_map=colors)
    return pn.pane.Plotly(fig)


def nps_comments_table(df):
    return pn.widgets.DataFrame(df[['name', 'nps_rating', 'nps_category', 'comments']])

def nps_vs_visit_frequency(df):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='visit_frequency', y='nps_rating', ci=None, palette='viridis')
    plt.title('Average NPS Rating vs Visit Frequency')
    plt.xlabel('Visit Frequency')
    plt.ylabel('Average NPS Rating')
    return pn.pane.Matplotlib(plt.gcf())

grouped_data_pane = pn.pane.DataFrame(grouped_data)


dashboard = pn.Column(
    pn.Row("# Net Promoter Score Analysis Dashboard For NPS survey"),
    pn.Row(nps_distribution_plot(df)),
    pn.Row(nps_pie_chart(df), nps_comments_table(df)),
    pn.Row(nps_vs_visit_frequency(grouped_data))
)

dashboard.servable()
pn.serve(dashboard)

