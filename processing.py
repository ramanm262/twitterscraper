import pandas as pd
import tqdm
import nltk
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import datetime
from nltk.stem.snowball import EnglishStemmer
nltk.download('stopwords')


def preprocess_stream(stream_file, punctuation_dict, additional_stopwords=[], stem_words=True, most_common=0,
                      save_wordcloud=True):
    stream_output = pd.read_csv(stream_file, delimiter=',', index_col=0, encoding='utf-8')

    tt = TweetTokenizer()

    tweets_list = []

    for tweet in tqdm.tqdm(stream_output.Text, desc='Tokenizing tweets'):
        tweet = tweet.lower()
        for char in punctuation_dict:
            tweet = tweet.replace(char, punctuation_dict[char])
        tweets_list.append(tt.tokenize(tweet))

    printable = set(string.printable)
    for tweet in tqdm.trange(len(tweets_list), desc='Filtering non-readable chars...'):
        for token in range(len(tweets_list[tweet])):
            for char in tweets_list[tweet][token]:
                if char not in printable:
                    tweets_list[tweet][token] = tweets_list[tweet][token].replace(char, '')

    print("Trimming stopwords")
    vocab_size = 0
    for tweet in tweets_list:
        vocab_size += len(tweet)
    print(f"Vocabulary size before trimming: {vocab_size}")
    stop_words = set(stopwords.words())
    for word in additional_stopwords:
        stop_words.add(word)  # This is done iteratively because we don't know that the additional stopwords are unique
    for tweet in range(len(tweets_list)):
        tweets_list[tweet] = [token for token in tweets_list[tweet] if token not in stop_words]
        tweets_list[tweet] = [token for token in tweets_list[tweet] if 'https://' not in token]
        tweets_list[tweet] = [token for token in tweets_list[tweet] if 'http://' not in token]
        tweets_list[tweet] = [token for token in tweets_list[tweet] if 't.co/' not in token]
        tweets_list[tweet] = [token for token in tweets_list[tweet] if 'bit.ly/' not in token]
        tweets_list[tweet] = [token for token in tweets_list[tweet] if not token.isdigit()]
        tweets_list[tweet] = [token for token in tweets_list[tweet] if len(token) > 1]
        tweets_list[tweet] = [token for token in tweets_list[tweet] if token[0] != "@"]

    vocab_size = 0
    for tweet in tweets_list:
        vocab_size += len(tweet)
    print(f"Vocabulary size after trimming: {vocab_size}")

    if stem_words:
        stemmer = EnglishStemmer()
        for tweet in tqdm.trange(len(tweets_list), desc="Stemming words"):
            tweets_list[tweet] = [stemmer.stem(token) for token in tweets_list[tweet]]

    tokens = [token for tweet in tweets_list for token in tweet]

    freq_dist = nltk.FreqDist(tokens)
    if most_common > 0:
        print("Most frequent tokens:")
        print(sorted(freq_dist, key=freq_dist.__getitem__, reverse=True)[:most_common])

    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(freq_dist)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud)
    plt.axis("off")
    if save_wordcloud:
        now = str(datetime.datetime.now()).replace(':', '').replace(' ', '_')[:9]
        plt.savefig(f"twitter_wordcloud_{now}.png")
        print(f"Saved wordcloud to twitter_wordcloud_{now}.png")
    else:
        plt.show()

    return tweets_list
