from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

tweet = "@kazu this is crazy  💕💕😍  https://t.co/abcdefg"
print('\n', tweet, '\n')

#preprocess tweet
tweetwords = []
for word in tweet.split(' '):   
    if word.startswith('@') and len(word) > 1 :
        word = '@user'

    elif word.startswith('http'):
        word = 'http'
    tweetwords.append(word)

print(tweetwords,'\n')

tweet_proc = " ".join(tweetwords)
print(tweet_proc)

#load model and tokenizer   
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

Labels = ['Negative', 'Neutral', 'Positive']

#sentiment analysis
encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
print(encoded_tweet)

# output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
output = model(**encoded_tweet)

print(output)

scores = output[0][0].detach().numpy()
scores = softmax(scores)
print(scores)

for i in range(len(scores)):
    l = Labels[i]
    s = scores[i]
    print(l, s)
