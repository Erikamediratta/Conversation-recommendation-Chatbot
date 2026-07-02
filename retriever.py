import json
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

with open("catalog.json") as f:
    CATALOG=json.load(f)

# opens the catalog file(56 documents) and loads all 56 assessments into memory as python list

def clean(text):
    text=re.sub(r'[^\w\s]',' ', text.lower())
    words=[]
    for w in text.split():
        if(len(w)>1):
            words.append(w)
    return words
        
        
#returns clean text re.sub remove punctuations, converts raw text into a list of clean words

def make_doc(item):
    return " ".join([item["name"],item["description"]," ".join(item["keywords"])," ".join(item["test_type"]), " ".join(item["job_levels"])])


#combine all useful fields of one assessment into one big string

all_tokens=[]
for item in CATALOG:
    all_tokens.append(clean(make_doc(item)))

#doing the above steps for all 56 assessments

VOCAB=set()
for token in all_tokens:
    for item in token:
        VOCAB.add(item)

N=len(CATALOG)

#collecting every word in all 56 assessments in the big set VOCAB

doc_freq={}
for token in all_tokens:
    for word in VOCAB:
        if word in set(token):
            if word not in doc_freq:
                doc_freq[word]=1
            else:
                doc_freq[word]=doc_freq[word]+1

#set(token) remove duplicates, also this is done to calculate
#doc_freq = {"java": 4, "test": 50, "python": 3, "the": 56, ...}

#IDF means inverse document frequency,
#which is how rare a word is across all documents
#java appears 4 in out of 56 documents->very rare, very useful
#the appears in 55 out of 56 documents ->very common->less useful
#IDF= log(total docs/ doc containing this word)

IDF={}
for word in VOCAB:
    IDF[word]=math.log((N+1)/(doc_freq[word]+1))

    #this +1 is to avoid dividing by zero

#tf is term frequency
#first doc=. java appears 3 times out of 10 words so tf=3/10=0.3

MATRIX=np.zeros((N,len(VOCAB)),dtype=np.float64)
# creates a matrix with 56 rows( N) and 530 columns(sum of all words in all documents)

VOCAB=list(VOCAB)
for doc_idx, token in enumerate(all_tokens):
    freq={}
    for word in token:
        if word not in freq:
            freq[word]=1
        else:
            freq[word]=freq[word]+1

    #compute TF-IDF
    for word in freq:
        TF=freq[word]/len(token)
        TF_IDF=TF*IDF[word]
        MATRIX[doc_idx, VOCAB.index(word)] = TF_IDF


# Now when then the user searches , do the same tf-idf for the query itself 
#Now the query is also a row of numbers

#Compare the query row with all 56 assessments in row at once, 
#this gives a similarity score between 0.0 and 1.0 for each assessment

#this is cosine similarity-> measures how similar two rows are

def search_query(query,top_k=10):
    token_query=clean(query)

    freq={}
    for word in token_query:
        if word not in freq:
            freq[word]=1
        else:
            freq[word]=freq[word]+1

    #create tf-idf vector for the query
    query_vector=np.zeros(len(VOCAB),dtype=np.float32)
    for word in freq:
        if word in IDF:
            tf=freq[word]/len(token_query)
            tf_idf=tf*IDF[word]
            query_vector[VOCAB.index(word)]=tf_idf
    

    scores=cosine_similarity(query_vector.reshape(1,-1),MATRIX)[0]

    #cosine similarity expects a 2d array so reshape(1,-1) converts [1,2] to [[1,2]]
    #MATRIX has suppose shape(56,530) so cosine similarity compares every query with all 56 documents at once,
    # and [0] means reconverting to 1 d array which was done as 2d in reshape

    top_indices = np.argsort(scores)[::-1][:top_k]

    #np.argsort(scores) gives indices that would sort the array in ascending order
    #[::-1] reverses it  (descending order)
    #[:top_k] takes only first top_k indices
    
    results = []
    for i in top_indices:
        item = CATALOG[i].copy()   # make a copy
        item["score"] = float(scores[i])
        results.append(item)

    return results




