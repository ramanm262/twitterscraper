# My Twitterscraper

### Background

A small personal project that I thought would be fun to share. This project originally was inspired by the paper "Twitter mood predicts the stock market" (https://doi.org/10.1016/j.jocs.2010.12.007). I didn't want to try and make money off a Twitter sentiment analyzer, because that would be too much work. Besides, there are people whose jobs are to take advantage of sentiment for financial gain, and their professional data products are much better than my side project could ever hope to be.

So instead, I decided it would be fun to have a "finger on the pulse of the world", and make my efforts no more ambitious than that. This project is a stable and mostly-complete version of what I planned to make for the time being.

## Use instructions

Clone this repository for use in your favorite environment.

### Run `stream.py` to download tweets.

You may need to adjust some parameters in this file depending on your preferences, directory structure, etc. Before running, create a text file in the `resources/` directory named `twitter_auth.txt`. This file will contain your API access passwords. It should be structured as the following:

```
{your consumer key here}
{your consumer secret here}
{your bearer token here}
{your access token here}
{your access secret here}
```

And it should go without saying that you will need to be connected to the Internet to use this script.

### Run `convert_sentiment140.py` to prepare Sentiment140 data

If you want to use benchmark Twitter data from Sentiment140 (http://help.sentiment140.com/for-students) instead of data you scraped with `stream.py`, create a directory named `sentiment140_data/` and put the testing data and training data inside it. The example training data in this repository is a subset of the training set `training.1600000.processed.noemoticon.csv`. Then, run the aforementioned script. It will create/overwrite the file `stream_output.txt` in your `resources/` directory.

### Run `main.py` for everything else

You may wish to adjust some parameters in the script. If I haven't forgotten to update this readme since the last edit to `main.py`, this program will:

* Clean and tokenize the Twitter data
* Show you the most common words used in all the tweets (in the command line and in fancy wordcloud form)
* Create Word2Vec embeddings for the tokens and the tweets they come from
* Automatically determine the optimal number of k-means clusters of the data based on silhouette coefficients 
* Cluster the tweets in an unsupervised fashion with k-means
* Show you the most representative words and tweets from each cluster

## An angry complaint

As of February 9, 2023, Twitter is forcing everyone to pay for API access. Yeah, you can download a limited number of tweets per month, but the amount allowed is not actually very much! Especially considering that there are advanced data products that use millions of tweets per day for their analysis, or that there are developers who want to test their code more than a few times each month, the amount of data Twitter now has available is useless in many applications. I hope my code still words with the new API, if you decide you want to support it. At least it worked for the free one.

Yes, Twitter has the right to charge for what is technically "its" data. But it ought not to be considered as such.
