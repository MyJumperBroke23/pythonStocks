import requests
url = "https://www.nasdaq.com/symbol/aapl/historical"
r = requests.get(url)
sourceCode = str(r.content)

def getOpen(date):
    dateIndex = sourceCode.find(date)
    #Stock price is located 131 characters after the date for each date
    return(sourceCode[dateIndex + 131:dateIndex + 137])

def getClose(date):
    dateIndex = sourceCode.find(date)
    return(sourceCode[dateIndex + 512:dateIndex + 518])