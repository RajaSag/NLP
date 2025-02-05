import string
import nltk
import numpy as np
import re
from nltk.corpus import twitter_samples, stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

# nltk.download('twitter_samples')
# nltk.download('stopwords')
# Load datasets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# Split data into training and testing sets
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4400:]
train_neg = all_negative_tweets[:2400]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

def process_tweet(tweet):
    tokenizer=TweetTokenizer(preserve_case=False,strip_handles=True,reduce_len=True)
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    tweet=re.sub(url_pattern, "", tweet)
    tweet_tokens=tokenizer.tokenize(tweet)
    # print('tokenized string',tweet_tokens)

    stopwords_english=stopwords.words('english')
    # print(stopwords_english)
    # print(string.punctuation)
    tweets_clean=[]
    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation:
            tweets_clean.append(word)
    # print(tweets_clean)
    stemmer=PorterStemmer()
    tweets_stem=[]
    for word in tweets_clean:
        stem_word = stemmer.stem(word)
        tweets_stem.append(stem_word)
    return tweets_stem

def build_freq(tweets,labels):
    freq = {}
    labels=np.squeeze(labels).tolist()
    for tweet,label in zip(tweets,labels):
        for word in process_tweet(tweet):
            freq[(word,label)]=freq.get((word,label),0)+1
    return freq

freq=build_freq(train_x,train_y)

# Feature extraction
def extract_features(tweet, freq):
    word_l = process_tweet(tweet)
    x = np.zeros(3)
    x[0] = 1
    for word in word_l:
        x[1] += freq.get((word, 1), 0)
        x[2] += freq.get((word, 0), 0)
    return x

def gradientDescent(x, y, theta, alpha, num_iters):
    m = x.shape[0]

    for i in range(num_iters):
        z = np.dot(x, theta)
        h = 1 / (1 + np.exp(-z))
        J = -(1 / m) * np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
        gradient = (1 / m) * np.dot(x.T, (h - y))
        theta -= alpha * gradient
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {J:.6f}")
    return J, theta

X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freq)
J, theta = gradientDescent(X,train_y, np.zeros((3, 1)), 1e-4, 1500)

def predict_tweet(tweet, freq, theta):
    x = extract_features(tweet, freq)
    z = np.dot(x, theta)
    y_pred = 1 / (1 + np.exp(-z))
    return y_pred

# for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great',"it might be a good day but, it's not"]:
#     print( '%s -> %f' % (tweet, predict_tweet(tweet, freq, theta)))

def test_logistic_regression(test_x, test_y, freq, theta, predict_tweet=predict_tweet):
    y_hat=[]
    for tweet in test_x:
        y_pred=predict_tweet(tweet,freq,theta)
        if y_pred>0.5:
            y_hat.append(1.0)
        else:
            y_hat.append(0.0)
    y_hat=np.array(y_hat)
    test_y=test_y.flatten()
    accuracy=np.mean(y_hat==test_y)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

accuracy= test_logistic_regression(test_x, test_y, freq, theta)


def train_naive_bayes(freqs, train_x, train_y):
    loglikelihood = {}
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    N_pos = N_neg = 0
    for pair in freqs.keys():
        if pair[1] > 0:
            N_pos += freqs[pair]
        else:
            N_neg += freqs[pair]
    D_pos = np.count_nonzero(train_y == 1)
    D_neg = np.count_nonzero(train_y == 0)
    logprior = np.log(D_pos) - np.log(D_neg)
    for word in vocab:
        freq_pos = freqs.get((word, 1), 0)
        freq_neg = freqs.get((word, 0), 0)
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)
        loglikelihood[word] = np.log(p_w_pos) - np.log(p_w_neg)
    return logprior, loglikelihood

logprior, loglikelihood = train_naive_bayes(freq, train_x, train_y)

def naive_bayes_predict(tweet, logprior, loglikelihood):
    word_l = process_tweet(tweet)
    p = logprior
    for word in word_l:
        if word in loglikelihood:
            p += loglikelihood[word]
    return p

def test_naive_bayes(test_x, test_y, logprior, loglikelihood, naive_bayes_predict=naive_bayes_predict):
    y_hats = []
    for tweet in test_x:
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0
        y_hats.append(y_hat_i)
    y_hats=np.array(y_hats)
    test_y = test_y.flatten()
    accuracy = np.mean(y_hats==test_y)
    return accuracy

print("Naive Bayes accuracy = %0.4f" %(test_naive_bayes(test_x, test_y, logprior, loglikelihood)))