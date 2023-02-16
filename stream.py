import tweepy
from tweepy import Stream
from tweepy.streaming import Stream
import numpy as np
import time

with open("resources/twitter_auth.txt") as twitter_auth:
    passwords = twitter_auth.readlines()

consumer_key = passwords[0]
consumer_secret = passwords[1]
bearer_token = passwords[2]
access_token = passwords[3]
access_token_secret = passwords[4]

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

class Listener(Stream):
    def __init__(self, output_file):
        super(Listener, self).__init__(consumer_key, consumer_secret, access_token, access_token_secret)
        self.output_file = output_file

    def on_status(self, status):
        if hasattr(status, "retweeted_status"):  # Check if Retweet
            try:
                print(status.retweeted_status.extended_tweet["full_text"].replace('\n', ' '), file=self.output_file)
            except AttributeError:
                print(status.retweeted_status.text.replace('\n', ' '), file=self.output_file)
        else:
            try:
                print(status.extended_tweet["full_text"].replace('\n', ' '), file=self.output_file)
            except AttributeError:
                print(status.text.replace('\n', ' '), file=self.output_file)

    def on_error(self, status_code):
        print(status_code)
        return False


output = open('resources/stream_output.txt', 'w', encoding='utf8')
stream = Listener(output_file=output)

stream_duration = 1.
print(f"Started streaming for {stream_duration} seconds.")
stream.sample(languages=['en'])
time.sleep(stream_duration)
stream.disconnect()
output.close()
print("Finished streaming.")

with open('resources/stream_output.txt', 'w', encoding='utf8') as file:
    stream_size = file.read()
print(f"Colllected {stream_size} bytes of tweets.")
