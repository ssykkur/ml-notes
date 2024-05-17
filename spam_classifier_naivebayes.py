import numpy as np
import pandas as pd 
from nltk.corpus import stopwords
from nltk import word_tokenize
import string 
from performance_measure import *

dataframe_emails = pd.read_csv('emails.csv')

#Takes dataframe containing text and its classification
#returns: 1array with text. 2array with classification
def preprocess_emails(df):
    #shuffling the dataset
    df = df.sample(frac=1, ignore_index=True, random_state=42)
    #removes the 'subject' string which is the first 9 chars of an email 
    #and convert to np array
    X = df.text.apply(lambda x: x[9:]).to_numpy()
    #convert column with classifiers to np array
    Y = df.spam.to_numpy()

    return X, Y  


#preprocesses X(text) by removing stopwords and punctuation
def preprocess_text(X):
    #set with stopwords and punctuation
    stop = set(stopwords.words('english') + list(string.punctuation))

    if isinstance(X, str):
        X = np.array([X])

    X_preprocessed = []

    #create an array without stopwords from an email
    #and add it to 'x preprocessed'
    for i, email in enumerate(X):    
        email = np.array([i.lower() for i in word_tokenize(email) if i.lower() not in stop])
        X_preprocessed.append(email)
    if len(X) == 1:
        return X_preprocessed[0]
    
    return X_preprocessed


#calcuate the frequency of each word appearing in spam(1) 
#or not spam (0)
#returns dict with key:word, value:frequency for (1) and (0)
def get_word_frequency(X, Y):
    word_dict = {}

    num_emails = len(X)

    for i in range(num_emails):
        email = X[i]    #get email 
        cls = Y[i]      #get its class (0,1) 
        email = set(email)      #remove duplicates

        for word in email:
            #initialize at count 1 to avoid 0 in probability 
            if word not in word_dict.keys():
                word_dict[word] = {'spam':1, 'ham': 1}
            
            if cls == 0:
                word_dict[word]['ham'] += 1
            if cls == 1:
                word_dict[word]['spam'] += 1

    return word_dict

def prob_word_given_class(word, cls, word_frequency, class_frequency):
    amount_word_and_class = word_frequency[word][cls]
    p_word_given_class = amount_word_and_class/class_frequency[cls]

def prob_email_given_class(treated_email, cls, word_frequency, class_frequency):
    prob = 1

    for word in treated_email:
        if word in word_frequency.keys():
            prob += np.log(prob_word_given_class(word, cls,word_frequency, class_frequency))
    
    return prob


def naive_bayes(treated_email, word_frequency, class_frequency, return_likelihood = False):
    prob_email_given_spam = prob_email_given_class(treated_email, 'spam', word_frequency, class_frequency)
    prob_email_given_ham = prob_email_given_class(treated_email, 'ham', word_frequency, class_frequency)
    p_spam = class_frequency['spam']/(class_frequency['ham'] + class_frequency['spam'])
    p_ham = class_frequency['ham']/(class_frequency['ham'] + class_frequency['spam'])
    spam_likelihood = prob_email_given_spam * np.log(p_spam)
    ham_likelyhood = prob_email_given_ham * np.log(p_ham)

    if return_likelihood:
        return (spam_likelyhood, ham_likelyhood)
    elif spam_likelyhood >= ham_likelyhood:
        return 1
    else:
        return 0




if __name__ == '__main__':  
    X, Y = preprocess_emails(dataframe_emails) 
    #testing
    X_treated = preprocess_text(X)
    #email_index = 989
    #print(f"Email before preprocessing: {X[email_index]}")
    #print(f"Email after preprocessing: {X_treated[email_index]}")

    #splitting into train/test, no need for CV
    TRAIN_SIZE = int(0.80*len(X_treated)) # 80% of emails used for training
    X_train = X_treated[:TRAIN_SIZE]
    Y_train = Y[:TRAIN_SIZE]
    X_test = X_treated[TRAIN_SIZE:]
    Y_test = Y[TRAIN_SIZE:]

    #test the proportions of spam after splitting (24% in the original dataset)
    #print(f'Proportion of spam in "train": {sum(Y_train == 1)/len(Y_train)}')
    #print(f'Proportion of spam in "Test" {sum(Y_test==1)/len(Y_test)}')

    word_frequency = get_word_frequency(X_train,Y_train)
    
    class_frequency = {'ham': sum(Y_train == 0), 'spam': sum(Y_train == 1)}
    #print(class_frequency)
    proportion_spam = class_frequency['spam']/(class_frequency['ham'] + class_frequency['spam'])
    email = "Please meet me in 2 hours in the main building. I have an important task for you."
    treated_email = preprocess_text(email)
    

