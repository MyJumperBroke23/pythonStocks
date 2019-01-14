import math
import random
import tweepy
from stockPrices import *
from tweetSentiment import query

# Key dictionary
keys = {}
# Grabbing keys
keyfile = open('keys.secret', 'r').read().splitlines()
for line in keyfile:
    pair = line.split(':')
    keys[pair[0]] = pair[1]
# Init twitter bot
auth = tweepy.OAuthHandler(keys["apikey"], keys["apisecret"])
auth.set_access_token(keys["accesstoken"], keys["accesssecret"])
api = tweepy.API(auth)
# api.update_status("OK1")
searchTerm = 'APPL'
max_tweets = 1000
