import inspect
import os
import re
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import STOPWORDS, WordCloud


class ChatHistoryAnalysis():
    def __init__(self) -> None:
        root = Path(__file__).parents[2]
        evaluation_dir = root / "notebooks/evaluation"
        results_dir = f'{evaluation_dir}/results'
        self.visualisation_dir = f'{results_dir}/visualisations/'
        self.table_dir = f'{results_dir}/table/'
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(self.visualisation_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)

        # data load
        file_path = f'{evaluation_dir}/data/chat_histories/chat_history_17_07_2024.csv' # TODO - implement function to fetch chat_history with specified date/latest date
        self.chat_logs = pd.read_csv(file_path)

        # slimming it to specific columns and converting to timestamp that can be read properly
        self.chat_logs = self.chat_logs[['created_at', 'users', 'chat_history', 'text', 'role']]
        self.chat_logs['created_at'] = pd.to_datetime(self.chat_logs['created_at'])

        self.ai_responses = self.chat_logs[self.chat_logs['role'] == 'ai']
        self.user_responses = self.chat_logs[self.chat_logs['role'] == 'user']

        self.chat_logs['tokens'] = self.chat_logs['text'].apply(self.preprocess_text)
        self.ai_responses['tokens'] = self.ai_responses['text'].apply(self.preprocess_text)
        self.user_responses['tokens'] = self.user_responses['text'].apply(self.preprocess_text)


    def preprocess_text(self, text):
        tokens = text.split()
        tokens = [word.lower() for word in tokens if word.isalpha()]
        return tokens

    # 1) Who uses Redbox the most?
    def user_frequency_analysis(self):
        user_counts = self.chat_logs['users'].value_counts()

        user_name = [user_email.split('@')[0].replace('.', ' ').title() for user_email in user_counts.index]

        wrapped_user_name = ['\n'.join(textwrap.wrap(name, width=10)) for name in user_name]

        # Table
        table_data = {'Name': user_name, 'Email': user_counts.index, 'Number of times used': user_counts.values}
        table_dataframe = pd.DataFrame(data=table_data, index=user_name)
        table_dataframe.to_csv(f'{self.table_dir}top_users.csv', index=False)

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
        table_data = {'Date': date_counts.index, 'Usage': date_counts.values}
        table_dataframe = pd.DataFrame(data=table_data)
        table_dataframe.to_csv(f'{self.table_dir}usage_of_redbox_ai_over_time.csv', index=False)

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

        # Table
        table_data = {'Word': list(word_freq.keys()), 'Frequency': list(word_freq.values())}
        table_dataframe = pd.DataFrame(data=table_data).sort_values('Frequency', ascending=False)
        table_dataframe.to_csv(f'{self.table_dir}user_most_frequent_words_table.csv', index=False)

        # Barplot
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(counts), y=list(words), palette='viridis')
        plt.title('Top 20 Most Frequent Words')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        barplot_path = os.path.join(self.visualisation_dir, 'user_most_frequent_words_barplot.png')
        plt.savefig(barplot_path)

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

        # Table
        table_data = {'Word': list(ai_word_freq.keys()), 'Frequency': list(ai_word_freq.values())}
        table_dataframe = pd.DataFrame(data=table_data).sort_values('Frequency', ascending=False)
        table_dataframe.to_csv(f'{self.table_dir}ai_most_frequent_words_table.csv', index=False)

        # Barplot
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(counts), y=list(words), palette='viridis')
        plt.title('Top 20 Most Frequent Words')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        barplot_path = os.path.join(self.visualisation_dir, 'ai_most_frequent_words_barplot.png')
        plt.savefig(barplot_path)
        
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
        def clean_text(text): # was including asterisks giving useless info to the graph I'm still not entirely convinced on the benefit of this analysis
            return re.sub('[!@#$*]', '', text).strip()

        self.ai_responses['clean_text'] = self.ai_responses['text'].apply(clean_text)
        ai_response_patterns = self.ai_responses['clean_text'].apply(lambda x: ' '.join(x.split()[:2])).value_counts().head(10)
        
        # Table
        words, counts = zip(*self.ai_responses['clean_text'].apply(lambda x: ' '.join(x.split()[:2])).value_counts())
        table_data = {'Word': words, 'Frequency': counts}
        table_dataframe = pd.DataFrame(data=table_data)
        table_dataframe.to_csv(f'{self.table_dir}common_ai_patterns.csv', index=False)

        # Barplot
        plt.figure(figsize=(10, 5))
        sns.barplot(x=ai_response_patterns.values, y=ai_response_patterns.index, palette='magma')
        plt.title('Common Patterns in AI Responses')
        plt.xlabel('Frequency')
        plt.ylabel('Patterns')
        ai_patterns_path = os.path.join(self.visualisation_dir, 'common_ai_patterns.png')
        plt.savefig(ai_patterns_path)


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
