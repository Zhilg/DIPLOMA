import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

LIMIT = 1000

def make_spam_index(train_dir): 
    """Создаёт частотный словарь термов с коэффициентами спамности""" 
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)] 
    all_words = [] 
    spam_count = {} 
    ham_count = {} 
    ham = 0 
    spam = 0

    for i, mail in enumerate(emails): 
        with open(mail) as m: 
          # print((mail, "spam" )if mail.find('/spm') != -1 or mail.find('spam') != -1 else "ham")
          for line in m: 

                words = line.strip('.,!?&#@').split() 
                for word in words: 
                    all_words.append(word) 

                    if mail.find('/spm') != -1 or mail.find('spam') != -1: 
                        spam_count[word] = spam_count.get(word, 0) + 1 
                        spam += 1
                    else: 
                        ham_count[word] = ham_count.get(word, 0) + 1 
                        ham += 1

                      
    print(ham, spam)
    spam_index = {} 
    for word in set(all_words): 
        spam_freq = spam_count.get(word, 0) 
        ham_freq = ham_count.get(word, 0) 
        if spam_freq + ham_freq > 0: 
            spam_index[word] = np.log((spam_freq + 1) / (ham_freq + 1)) 

    print(spam_index.__len__())
    return spam_index
  
def extract_weighted_features(mail_dir, dictionary): 
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)] 
    
    features_matrix = np.zeros((len(files), len(dictionary)), dtype=np.int32) 
    word_to_id = {} 
    for idx, word in enumerate(dictionary): 
        word_to_id[word] = idx 
 
    for docID, fil in enumerate(files): 
        word_counts = {} 
        with open(fil) as fi: 
            for i, line in enumerate(fi): 
                if i == 2: 
                    words = line.strip('.,!?&#@').split() 
                    for word in words: 
                        word_counts[word] = word_counts.get(word, 0) + 1 
 
        for word, count in word_counts.items(): 
            if word in word_to_id: 
                features_matrix[docID, word_to_id[word]] = count 
  
    return features_matrix  
# Create a dictionary of words with its frequency




# Prepare feature vectors per training mail and its labels

# Training SVM and Naive bayes classifier and its variants


# Test the unseen mails for Spam






