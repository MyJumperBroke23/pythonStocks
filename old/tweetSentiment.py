from textblob import TextBlob


def sentiment(text: str):
    return sum([i.sentiment.polarity for i in TextBlob(text).sentences])


def query(text, api):
    results = api.search(q=text, count=1)
    for i in results:
        print(i.id)
        print(i.text, sentiment(i.text))
        favorites = i.favorite_count
        try:
            j = i.retweeted_status.favorite_count
            favorites += j
        except AttributeError:
            print("not a retweet")
        print(favorites)
