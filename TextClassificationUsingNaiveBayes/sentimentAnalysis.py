#Importing necessary libraires

import dictdiffer as dictdiffer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split


#Read File and Split Data operations here
file = open('all_sentiment_shuffled.txt', 'r', encoding='utf-8')

lines = file.readlines()

category = []
sentiment = []
document_id = []
token = []

for line in lines:
    line = line.split(" ", 3)
    category.append(line[0])
    sentiment.append(line[1])
    document_id.append(line[2])
    token.append(line[3])

X_train, X_test, y_train, y_test = train_test_split(token, sentiment, test_size=0.2, shuffle=False)

posRevs = []
negRevs = []

for i in range(len(y_train)):
    if y_train[i] == 'neg':
        negRevs.append(X_train[i])
    else:
        posRevs.append(X_train[i])

np.array(posRevs)
np.array(negRevs)

#You can change ngram_range and or make stop_words None here for watching accuracy under different situations
posBow = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words='english', ngram_range=(1,1))
posBows_ = posBow.fit_transform(posRevs)
posCtr = posBows_.toarray().sum(axis=0)
posDict = dict(zip(posBow.get_feature_names(), posCtr))

#You can change ngram_range and or make stop_words None here for watching accuracy under different situations
negBow = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words='english', ngram_range=(1,1))
negBows_ = negBow.fit_transform(negRevs)
negCtr = negBows_.toarray().sum(axis=0)
negDict = dict(zip(negBow.get_feature_names(), negCtr))


pos_set = posBow.get_feature_names()
neg_set = negBow.get_feature_names()

pos_word_counter = posBows_.sum()
neg_word_counter = negBows_.sum()

p_pos = np.log10(len(posRevs) / len(X_train))
p_neg = np.log10(len(negRevs) / len(X_train))

unique_words = set(pos_set + neg_set)
unique_words_ctr = len(unique_words)

classifier_result = []

# Naive Bayes Classifier Implementation
for k in X_test:
    prop_pos = 0
    prop_neg = 0
    testBow = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words='english', ngram_range=(1,1))
    line_list = []
    line_list.append(k)
    testBows_ = testBow.fit_transform(line_list)
    testCtr = testBows_.toarray().sum(axis=0)
    testDict = dict(zip(testBow.get_feature_names(), testCtr))

    for item, value in testBow.vocabulary_.items():
        prop_pos += testBows_.toarray()[0, value] * np.log10(
            (posDict.get(item, 0) + 1) / (pos_word_counter + unique_words_ctr))        #Laplace smoothing applied here
        prop_neg += testBows_.toarray()[0, value] * np.log10(
            (negDict.get(item, 0) + 1) / (neg_word_counter + unique_words_ctr))        #Laplace smoothing applied here

    prop_pos += (p_pos)
    prop_neg += (p_neg)

    if prop_pos > prop_neg:
        classifier_result.append('pos')
    elif prop_pos < prop_neg:
        classifier_result.append('neg')

correct_list = 0

for i in range(len(y_test)):
    if y_test[i] == classifier_result[i]:
        correct_list += 1

    acc = (correct_list / len(y_test)) * 100

print("Accuracy : " + str(round(acc)))

#Modul Analysis TFIDF Part

posTfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
posTfidf_ = posTfidf.fit_transform(posRevs)
pos_counter = posTfidf_.toarray().sum(axis=0)
positiveD = dict(zip(posTfidf.get_feature_names(),  pos_counter))
sorted_positive= sorted(positiveD.items(), key=lambda x:x[1], reverse= True)
print("\n")
print("List the 10 words whose presence most strongly predicts that the review is positive.")
for i in range(len(sorted_positive)):
    if i < 10:
        print(sorted_positive[i])


negTfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
negTfidf_ = negTfidf.fit_transform(negRevs)
neg_counter = negTfidf_.toarray().sum(axis=0)
negativeD = dict(zip(negTfidf.get_feature_names(), neg_counter))
sorted_negative = sorted(negativeD.items(), key=lambda  x:x[1], reverse=True)
print("\n")
print("List the 10 words whose presence most strongly predicts that the review is negative.")
for i in range(len(sorted_negative)):
    if i < 10:
        print(sorted_negative[i])
print("\n")
print("List the 10 words whose absence most strongly predicts that the review is positive.")

positiveDifference = list(set(sorted_positive)-set(sorted_negative))
for i in range(10):
    print(positiveDifference[i])

print("\n")
print("List the 10 words whose absence most strongly predicts that the review is negative.")

negativeDifference = list(set(sorted_negative)-set(sorted_positive))
for i in range(10):
    print(negativeDifference[i])

