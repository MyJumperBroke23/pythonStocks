from textblob import TextBlob


def sentiment(text: str):
    return sum([i.sentiment.polarity for i in TextBlob(text).sentences])
