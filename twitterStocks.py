import math
import random
import tweepy
#Key dictionary
keys={}
#Grabbing keys
keyfile = open('keys.secret','r').read().splitlines()
for line in keyfile:
	pair = line.split(':')
	keys[pair[0]]=pair[1]
#Init twitter bot
auth = tweepy.OAuthHandler(keys["apikey"],keys["apisecret"])
auth.set_access_token(keys["accesstoken"],keys["accesssecret"])
api=tweepy.API(auth)
api.update_status("OK")