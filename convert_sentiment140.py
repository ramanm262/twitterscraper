import pandas as pd

tweets = pd.read_csv("resources/sentiment140_data/training.1600000.processed.noemoticon.csv",
                     names=["polarity", "id", "date", "query", "user", "text"], nrows=232719)
tweets.rename(columns={"text": "Text"}, inplace=True)
tweets.to_csv("resources/stream_output.txt")
