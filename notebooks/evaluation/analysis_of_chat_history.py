import glob
import inspect
import os
import re
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from regex import D
from wordcloud import STOPWORDS, WordCloud


class ChatHistoryAnalysis():
    def __init__(self) -> None:
        root = Path(__file__).parents[2]
        self.evaluation_dir = root / "notebooks/evaluation"
        results_dir = f'{self.evaluation_dir}/results'
        self.visualisation_dir = f'{results_dir}/visualisations/'
        self.table_dir = f'{results_dir}/table/'
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(self.visualisation_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)

        # IMPORTANT - For this to work you must save your chat history CSV dump in notebooks/evaluation/data/chat_histories 
        file_path = self.latest_chat_history_file()
        self.chat_logs = pd.read_csv(file_path)

        # Select specific columns and converting to readable timestamp
        self.chat_logs = self.chat_logs[['created_at', 'users', 'chat_history', 'text', 'role']]
        self.chat_logs['created_at'] = pd.to_datetime(self.chat_logs['created_at'])

        self.ai_responses = self.chat_logs[self.chat_logs['role'] == 'ai']
        self.user_responses = self.chat_logs[self.chat_logs['role'] == 'user']

        self.chat_logs['tokens'] = self.chat_logs['text'].apply(self.preprocess_text)
        self.ai_responses['tokens'] = self.ai_responses['text'].apply(self.preprocess_text)
        self.user_responses['tokens'] = self.user_responses['text'].apply(self.preprocess_text)

    def latest_chat_history_file(self):
        chat_history_folder = glob.glob(f'{self.evaluation_dir}/data/chat_histories/*')
        latest_file = max(chat_history_folder, key=os.path.getctime)
        return latest_file

    def preprocess_text(self, text):
        tokens = text.split()
        tokens = [word.lower() for word in tokens if word.isalpha()]
        return tokens
    
    def table_dataframe(self, data, file_name, sort_value=None, index=None):
        if index:
            table_dataframe = pd.DataFrame(data=data, index=index)
        elif sort_value:
            table_dataframe = pd.DataFrame(data=data).sort_values(sort_value, ascending=False)
        else:
            table_dataframe = pd.DataFrame(data=data)
        table_dataframe.to_csv(f'{self.table_dir}{file_name}.csv', index=False)
    
    def plot_bar_graph(self, title, x, y, x_label, y_label, file_name):
        plt.figure(figsize=(10, 5))
        sns.barplot(x=x, y=y, palette='viridis')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        barplot_path = os.path.join(self.visualisation_dir, file_name)
        plt.savefig(barplot_path)

    # 1) Who uses Redbox the most?
    def user_frequency_analysis(self):
        user_counts = self.chat_logs['users'].value_counts()

        user_name = [user_email.split('@')[0].replace('.', ' ').title() for user_email in user_counts.index]

        wrapped_user_name = ['\n'.join(textwrap.wrap(name, width=10)) for name in user_name]

        self.table_dataframe(data={'Name': user_name, 'Email': user_counts.index, 'Number of times used': user_counts.values}, index=user_name, file_name='top_users')

        # Barplot
        plt.figure(figsize=(10, 5))
        sns.barplot(x=wrapped_user_name, y=user_counts.values, palette='viridis')

        plt.xticks(ha='right', size=9)
        plt.title('Unique Users by Total Number of Messages')
        plt.xlabel('Users')
        plt.ylabel('Total No. of Messages')
        top_users_path = os.path.join(self.visualisation_dir, 'top_users.png')
        plt.savefig(top_users_path)

    # 2) Redbox traffic analysis
    def redbox_traffic_analysis(self):
        # think about GA integration?
        self.chat_logs['date'] = self.chat_logs['created_at'].dt.date
        date_counts = self.chat_logs['date'].value_counts().sort_index()

        # Table
        self.table_dataframe(data={'Date': date_counts.index, 'Usage': date_counts.values}, file_name='usage_of_redbox_ai_over_time')
        
        # Line graph
        plt.figure(figsize=(10, 5))
        date_counts.plot(kind='line')
        plt.title('Usage of Redbox AI Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Conversations')
        conversation_frequency_path = os.path.join(self.visualisation_dir, 'usage_of_redbox_ai_over_time.png')
        plt.savefig(conversation_frequency_path)

    # 3) Which words are used the most frequently by USERS?
    def user_word_frequency_analysis(self):
        all_tokens = [token for tokens in self.user_responses['tokens'] for token in tokens]

        # far too many stopwords and wordcloud has a lovely constant attached to resolve this
        stopwords_removed_from_all_tokens = [word for word in all_tokens if word not in STOPWORDS]

        word_freq = Counter(stopwords_removed_from_all_tokens)

        most_common_words = word_freq.most_common(20)  #TODO - determine how many common words we want and the right vis. for this
        words, counts = zip(*most_common_words)

        self.table_dataframe(data={'Word': list(word_freq.keys()), 'Frequency': list(word_freq.values())}, file_name='user_most_frequent_words_table', sort_value='Frequency')

        self.plot_bar_graph('Top 20 Most Frequent Words', list(counts), list(words), 'Frequency', 'Words', 'user_most_frequent_words_barplot.png')

        # Wordcloud - TODO - assess value
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Frequent Words')
        wordcloud_path = os.path.join(self.visualisation_dir, 'user_most_frequent_words.png')
        plt.savefig(wordcloud_path)

    # 4) Which words are used the most frequently by AI?
    def ai_word_frequency_analysis(self):
        ai_word_freq = Counter([token for tokens in self.ai_responses['tokens'] for token in tokens if token not in STOPWORDS])
        most_common_words = ai_word_freq.most_common(20)  #TODO - determine how many common words we want and the right vis. for this
        words, counts = zip(*most_common_words)

        self.table_dataframe(data={'Word': list(ai_word_freq.keys()), 'Frequency': list(ai_word_freq.values())}, file_name='ai_most_frequent_words_table', sort_value='Frequency')
        self.plot_bar_graph('Top 20 Most Frequent Words', list(counts), list(words), 'Frequency', 'Words', 'ai_most_frequent_words_barplot.png')
        
        # Wordcloud - TODO - assess value
        ai_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ai_word_freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(ai_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Frequent Words in AI Responses')
        ai_wordcloud_path = os.path.join(self.visualisation_dir, 'ai_most_frequent_words.png')
        plt.savefig(ai_wordcloud_path)

    # 5) Is there a clear pattern behind AI responses?
    def ai_response_pattern_analysis(self):
        def clean_text(text): # remove symbols from responses
            return re.sub('[!@#$*]', '', text).strip()

        self.ai_responses['clean_text'] = self.ai_responses['text'].apply(clean_text)
        ai_response_patterns = self.ai_responses['clean_text'].apply(lambda x: ' '.join(x.split()[:2])).value_counts().head(10)
        
        # Table
        words, counts = zip(*self.ai_responses['clean_text'].apply(lambda x: ' '.join(x.split()[:2])).value_counts())
        self.table_dataframe(data={'Word': words, 'Frequency': counts}, file_name='common_ai_patterns', sort_value=None)
        self.plot_bar_graph('Common Patterns in AI Responses', ai_response_patterns.values, ai_response_patterns.index, 'Frequency', 'Patterns', 'common_ai_patterns.png')


def main():
    chat_history_analysis = ChatHistoryAnalysis()
    attrs = (getattr(chat_history_analysis, name) for name in dir(chat_history_analysis))
    methods = filter(inspect.ismethod, attrs)
    for method in methods:
        try:
            method()
        except TypeError:
            pass


if __name__ == "__main__":
    main()
