from nltk import bigrams, ngrams, trigrams 
from collections import Counter
import nltk,re
import numpy as np
from nltk.corpus import brown
from nltk.tokenize import RegexpTokenizer
import sys

fp=open(sys.argv[1])
abc=fp.read()
SSS=abc.split('\n')
print('completed reading input file')

tokenizer = RegexpTokenizer(r'\w+')
s=brown.sents()
#len(s)
print("brown corpus imported")



#pre-processing
print("pre-processing started")
S=[[re.sub(r'[^a-z ]+','',j.lower()) for j in i] for i in s] #case-folding and removing numerals

sentences=[" ".join(i) for i in S]
print("pre-processing finished")

#unigrams
print("creating unigram model")
def unigram_model(sent):
    unigrams=[]
    for elem in sent:
        unigrams.append(None)
        unigrams.extend(elem.split())
        unigrams.append(None)
    unigram_counts=dict(Counter(unigrams))
    unigram_total=len(unigrams)
    for word in unigram_counts:
        unigram_counts[word]/=(unigram_total+0.0)
    return unigram_counts
unigram_counts=unigram_model(sentences)
print("finished : creating unigram model","length of unigram model",len(unigram_counts))

#bigrams
print("creating bigram model")
def bigram_model(sentences):
    model={}
    bigrams=[]
    for sent in sentences:
         for w1,w2 in ngrams(sent.split(),2, pad_left=True,pad_right=True):
            bigrams.append((w1,w2))
            if w1 not in model:
                model[w1]={}
            if w2 not in model[w1]:
                model[w1][w2]=0
            model[w1][w2]+=1
    for w1 in model:
        tot_count=float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2]/=tot_count
    
    return model,dict(Counter(bigrams))

bigram_counts,bigrams= bigram_model(sentences)
print("finished : creating bigram model")

#trigram

print("creating trigram model")

def trigram_model(sentences):
    model={}
    trigrams=[]
    for sent in sentences:
         for w1,w2,w3 in ngrams(sent.split(),3, pad_left=True,pad_right=True):
            trigrams.append((w1,w2,w3))
            if (w1,w2) not in model:
                model[(w1,w2)]={}
            if w3 not in model[(w1,w2)]:
                model[(w1,w2)][w3]=0
            model[(w1,w2)][w3]+=1
    for (w1,w2) in model:
        tot_count=float(sum(model[(w1,w2)].values()))
        for w3 in model[(w1,w2)]:
            model[(w1,w2)][w3]/=tot_count
     
    return model,dict(Counter(trigrams))

def trigram_true_model(sentences):
    model={}
    for sent in sentences:
        for w1,w2,w3 in ngrams(sent.split(),3,pad_left=True,pad_right=True):
            if w1 not in model:
                model[w1]={}
            if w2 not in model[w1]:
                model[w1][w2]={}
            if w3 not in model[w1][w2]:
                model[w1][w2][w3]=0
            model[w1][w2][w3]+=1
    return model

trigram_counts,trigrams= trigram_model(sentences)
#trigram_true_counts= trigram_true_model(sentences)
#print(len(trigram_counts),len(trigram_true_counts))
print("finished : creating trigram model")

#No smoothing : calculating log-likelihood and perplexity
print('No smoothing : log-likelihood and perplexity')
for t in SSS:
    #calculating log-likelihood using unigram model 
    p=unigram_counts[None]**2
    log_LE=0
    for w in t.split():
        if w in unigram_counts:
            p*=unigram_counts[w]
            log_LE+=np.log(unigram_counts[w])
        else:
            p*=0
            log_LE=-float('inf')
    print(t,', unigram log-likelihood :',log_LE,', unigram perplexity :',np.power(p,(np.divide(-1.0,len(t.split())))))
    #calculating log-likelihood using bigram model 
    p=1
    log_LE=0
    for w1,w2 in ngrams(t.split(),2,pad_left=True,pad_right=True):
        if w1 in bigram_counts:
            if w2 in bigram_counts[w1]:
                p*=bigram_counts[w1][w2]
                log_LE+=np.log(bigram_counts[w1][w2])
        else:
            p*=0
            log_LE=-float('inf')
    print(t,', bigram log-likelihood :',log_LE,', bigram perplexity :',np.power(p,(np.divide(-1.0,len(t.split())))))
    #calculating log-likelihood using bigram model	
    p=1
    log_LE=0
    for w1,w2,w3 in ngrams(t.split(),3,pad_left=True,pad_right=True):
        if (w1,w2) in trigram_counts:
            if w3 in trigram_counts[(w1,w2)]:
                p*=trigram_counts[(w1,w2)][w3]
                log_LE+=np.log(trigram_counts[(w1,w2)][w3])
        else:
            p*=0
            log_LE=-float('inf')
    print(t,', trigram log-likelihood :',log_LE,', trigram perplexity :',np.power(p,(np.divide(-1.0,len(t.split())))))

#unigram with model with laplace smoothing
v=len(unigram_counts)
def unigram_model_laplace(sent,k):
    unigrams=[]
    for elem in sent:
        unigrams.extend(elem.split())
        unigrams.append(None)
    unigram_counts=dict(Counter(unigrams))
    unigram_total=len(unigrams)
    for word in unigram_counts:
        unigram_counts[word]=(unigram_counts[word]+k)/(unigram_total+k*v)
    return unigram_counts


#bigram model with laplace smoothing
def bigram_model_laplace(sentences,k):
    model={}
    for sent in sentences:
         for w1,w2 in ngrams(sent.split(),2, pad_left=True,pad_right=True):
            if w1 not in model:
                model[w1]={}
            if w2 not in model[w1]:
                model[w1][w2]=0
            model[w1][w2]+=1
    for w1 in model:
        tot_count=float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2]=(model[w1][w2]+k)/(tot_count+k*v)
        model[w1]['0']=(k/(k*v+tot_count))
    return model

#trigram model with laplace smoothing 
def trigram_model_laplace(sentences,k):
    model={}
    for sent in sentences:
         for w1,w2,w3 in ngrams(sent.split(),3, pad_left=True,pad_right=True):
            if (w1,w2) not in model:
                model[(w1,w2)]={}
            if w3 not in model[(w1,w2)]:
                model[(w1,w2)][w3]=0
            model[(w1,w2)][w3]+=1
    for (w1,w2) in model:
        tot_count=float(sum(model[(w1,w2)].values()))
        for w3 in model[(w1,w2)]:
            model[(w1,w2)][w3]=(model[(w1,w2)][w3]+k)/(tot_count+k*v)
        model[(w1,w2)]['0']=(k/(k*v+tot_count))
    return model

k_list=[1.0,1e-1,1e-2,1e-3,1e-4]
#print(k_list)
#Laplace smoothing : calculating log-likelihood and perplexity

for k_para in k_list:
    print('Laplace smoothing k ='+str(k_para) +': log-likelihood and perplexity')
    print('k',k_para)
    unigram_counts=unigram_model_laplace(sentences,k_para)
    bigram_counts=bigram_model_laplace(sentences,k_para)
    trigram_counts=trigram_model_laplace(sentences,k_para)
    for t in SSS:
        #calculating log-likelihood using unigram model 
        p=unigram_counts[None]**2
        log_LE=0
        for w in t.split():
            if w in unigram_counts:
                p*=unigram_counts[w]
                log_LE+=np.log(unigram_counts[w])
            else:
                p*=0
                log_LE=-float('inf')
        print(t,', unigram log-likelihood :',log_LE,', unigram perplexity :',np.power(p,(np.divide(-1.0,len(t.split())))))
        #calculating log-likelihood using bigram model 
        p=1
        log_LE=0
        for w1,w2 in ngrams(t.split(),2,pad_left=True,pad_right=True):
            if w1 in bigram_counts:
                if w2 in bigram_counts[w1]:
                    p*=bigram_counts[w1][w2]
                    log_LE+=np.log(bigram_counts[w1][w2])
                else:
                    p*=bigram_counts[w1]['0']
                    log_LE+=np.log(bigram_counts[w1]['0'])
            else:
                p*=0
                log_LE=-float('inf')
        print(t,', bigram log-likelihood :',log_LE,', bigram perplexity :',np.power(p,(np.divide(-1.0,len(t.split())))))
        #calculating log-likelihood using bigram model 
        p=1
        log_LE=0
        for w1,w2,w3 in ngrams(t.split(),3,pad_left=True,pad_right=True):
            if (w1,w2) in trigram_counts:
                if w3 in trigram_counts[(w1,w2)]:
                    p*=trigram_counts[(w1,w2)][w3]
                    log_LE+=np.log(trigram_counts[(w1,w2)][w3])
                else:
                    p*=trigram_counts[(w1,w2)]['0']
                    log_LE+=np.log(trigram_counts[(w1,w2)]['0'])
            else:
                p*=0
                log_LE=-float('inf')
        print(t,', trigram log-likelihood :',log_LE,', trigram perplexity :',np.power(p,(np.divide(-1.0,len(t.split())))))

v=len(unigram_counts)
#bigram model with Good Turing smoothing
def bigram_model_goodturing():
    model={}
    count_numbers=dict(Counter(bigrams.values()))
    # calculating effective counts
    eff_count_0=count_numbers[1]/(v*v-len(bigrams))
    eff_count_sum=eff_count_0+sum(bigrams.values())
    for i in bigrams:
         model[i]=bigrams[i]/eff_count_sum
    model['0']=eff_count_0/eff_count_sum
    return model

#trigram model with Good Turing smoothing 
def trigram_model_goodturing():
    model={}
    count_numbers=dict(Counter(trigrams.values()))
    # calculating effective counts
    eff_count_0=count_numbers[1]/(v*v-len(trigrams))
    eff_count_sum=eff_count_0+sum(trigrams.values())
    for i in trigrams:
         model[i]=trigrams[i]/eff_count_sum
    model['0']=eff_count_0/eff_count_sum
    return model

bigram_counts=bigram_model_goodturing()
trigram_counts=trigram_model_goodturing()
print('good turing models created')

print('Using Good Turing Smoothing')
for t in SSS:
    #calculating log-likelihood using bigram model 
    p=1
    log_LE=0
    for w1,w2 in ngrams(t.split(),2,pad_left=True,pad_right=True):
        if (w1,w2) in bigram_counts:
            p*=bigram_counts[(w1,w2)]
            log_LE+=np.log(bigram_counts[(w1,w2)])
        else:
            p*=bigram_counts['0']
            log_LE+=np.log(bigram_counts['0'])
    print(t,', bigram log-likelihood :',log_LE,', bigram perplexity :',np.power(p,(np.divide(-1.0,len(t.split())))))
    p=1
    log_LE=0
    for w1,w2,w3 in ngrams(t.split(),3,pad_left=True,pad_right=True):
        if (w1,w2,w3) in trigram_counts:
            p*=trigram_counts[(w1,w2,w3)]
            log_LE+=np.log(trigram_counts[(w1,w2,w3)])
        else:
            p*=bigram_counts['0']
            log_LE+=np.log(bigram_counts['0'])
    print(t,', trigram log-likelihood :',log_LE,', trigram perplexity :',np.power(p,(np.divide(-1.0,len(t.split())))))

#interpolation method for the bigram model
unigram_counts=unigram_model(sentences)
bigram_counts,bigrams= bigram_model(sentences)
lamda_list=[0.2,0.5,0.8]
print('using interpolation method for bigram model')
for lamda in lamda_list:
    print('choosing interpolation parameter lambda =',lamda)
    for t in SSS:
        #calculating log-likelihood using bigram model 
        p=unigram_counts[None]
        log_LE=np.log(unigram_counts[None])
        for w1,w2 in ngrams(t.split(),2,pad_left=True,pad_right=True):
            if w1 in bigram_counts:
                if w2 in bigram_counts[w1]:
                    p*=(lamda*bigram_counts[w1][w2]+(1-lamda)*unigram_counts[w2])
                    log_LE+=np.log(bigram_counts[w1][w2]*unigram_counts[w2])
            else:
                p*=0
                log_LE=-float('inf')
        print(t,', bigram log-likelihood :',log_LE,', bigram perplexity :',np.power(p,(np.divide(-1.0,len(t.split())))))
