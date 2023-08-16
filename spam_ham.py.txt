import numpy as np
import pandas as pd
import os 
import re
import striprtf.striprtf import rtf_to_text
df = pd.read_csv('spam.csv')
# renaming the cols
df.rename(columns={'type':'target','text':'text'},inplace=True)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
# remove duplicates
df = df.drop_duplicates(keep='first')
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

# import string library function 
import string 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
  
ps = PorterStemmer()

def transform_text(text):
    x = text
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)




df['transformed_text'] = df['text'].apply(transform_text)
train_spam = []
train_ham = []

for ind in df.index:
    if df['target'][ind] == 1:
        train_spam.append(df['text'][ind])
    
    elif df['target'][ind] == 0:
        train_ham.append(df['text'][ind])
        
        
        
# make a vocabulary of unique words that occur in known spam emails

vocab_words_spam = []

for sentence in train_spam:
    sentence_as_list = sentence.split()
    for word in sentence_as_list:
        vocab_words_spam.append(word) 
        
vocab_unique_words_spam = list(dict.fromkeys(vocab_words_spam))

dict_spamicity = {}
for w in vocab_unique_words_spam:
    emails_with_w = 0     # counter
    for sentence in train_spam:
        if w in sentence:
            emails_with_w+=1
            
    total_spam = len(train_spam)
    spamicity = (emails_with_w+1)/(total_spam)
    dict_spamicity[w.lower()] = spamicity
    


vocab_words_ham = []

for sentence in train_ham:
    sentence_as_list = sentence.split()
    for word in sentence_as_list:
        vocab_words_ham.append(word)
        
vocab_unique_words_ham = list(dict.fromkeys(vocab_words_ham))

dict_hamicity = {}
for w in vocab_unique_words_ham:
    emails_with_w = 0     # counter
    for sentence in train_ham:
        if w in sentence:
            emails_with_w+=1
            
    total_ham = len(train_ham)
    Hamicity = (emails_with_w+1)/(total_ham)       # Smoothin applied
    dict_hamicity[w.lower()] = Hamicity
        
        
        
prob_spam = len(train_spam) / (len(train_spam)+(len(train_ham)))
prob_ham = len(train_ham) / (len(train_spam)+(len(train_ham)))

# tests = ['Dear Sir, PRML quiz was the first exam we have taken among all the other courses here at IIT Madras. We were not really sure about what kind of questions to expect either.I have received a lot of requests from my classmates to request you to increase the number of quizzes to 6 and make it best 4 out of 6. I have raised this issue in the CR meeting and they have told me to ask you. We would be really grateful if you could consider it. Thanks and regards']

# tests = ['Dear Beneficiary, The United Nations Compensation Commission (UNCC) has approved to pay you a compensation amount of US$1,500,000 (One Million, Five Hundred Thousand United State Dollars) due to losses and damages suffered as to delayed foreign contract payment of individuals, firms, contractors, inheritance, next-of-kin, super hurricane Sandy and lottery beneficiaries that originated from Africa, Europe, Americas, Asia including the Middle East. Your approved Compensation package has been deposited in the "Security Vault of SunWay Finance & Security company USA" waiting for delivery. For identification and swift delivery of your compensation package, you are advice to contact Diplomat Ellis Gammon of SunWay Finance & Security company and re-confirm your delivery details: call Tel: +1 321 586 1802, E-mail: ellisgammon8@gmail.com']

file = os.listdir('test/')
tests = []

for email in file:
    with open('test/'+mail) as infile:
        content = infile.read()
        text = rtf_to_text(content)
    tests.append(text)
  

    #     for i in range(len(tests)):
#     tests[i] = [''.join(c for c in s if c not in string.punctuation) for s in tests[i]]  #Remove punctuation

distinct_words_as_sentences_test = []
x = []
for sentence in tests:
    x.append(sentence)
    sentence_as_list = sentence.split()
    senten = []
    for word in sentence_as_list:
        senten.append(word)
    distinct_words_as_sentences_test.append(senten)
        
# print(distinct_words_as_sentences_test)

reduced_sentences_test = []
for sentence in distinct_words_as_sentences_test:
    words_ = []
    for word in sentence:
        if word in vocab_unique_words_spam:
            words_.append(word)
        elif word in vocab_unique_words_ham:
            words_.append(word)
        
    reduced_sentences_test.append(words_)
# print(reduced_sentences_test)



for j in reduced_sentences_test:
    Bayes(j)


def mult(probs,j) :        # function to multiply all word probs together 
    total_prob = j
    for i in probs: 
         total_prob = total_prob * i  
    return total_prob

def Bayes(email):
    probs = []
    for word in email:
        p_spam = prob_spam
        try:
            pr_WS = dict_spamicity[word.lower]
        except KeyError:
            pr_WS = (1/(total_spam))
            
        p_ham = prob_ham
        try:
            pr_WH = dict_hamicity[word.lower]
        except KeyError:
            pr_WH = (1/(total_ham))  # Apply smoothing for word not seen in ham training data, but seen in spam training

        
        prob_word_spam = (pr_WS*p_spam)/((pr_WS*p_spam)+(pr_WH*p_ham))

        probs.append(prob_word_spam)
        
    final_classification = mult(probs,1)
    
    if final_classification >= 0.5:
        print('+1')
    else:
        print('0')

