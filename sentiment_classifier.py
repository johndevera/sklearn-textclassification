
# coding: utf-8

# In[26]:

import numpy as np
import itertools
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, make_scorer, accuracy_score
from sklearn.metrics import classification_report

from scipy.sparse import csr_matrix
from sklearn.cross_validation import cross_val_score, cross_val_predict, train_test_split
from sklearn.grid_search import GridSearchCV
    
def pre_process():
    classifiedFile = 'classified_tweets.txt'
    corpusFile = 'corpus.txt'
    stopFile = 'stop_words.txt'
    unclassifiedFile = 'unclassified_tweets.txt'
    fileList = [classifiedFile, corpusFile, stopFile, unclassifiedFile]

    fileIndex = {classifiedFile:0,
                  corpusFile:1,
                  stopFile:2,
                  unclassifiedFile:3}
    
    fileLength = {classifiedFile:0,
                  corpusFile:0,
                  stopFile:0,
                  unclassifiedFile:0}
    
    for fileName in fileList: ##get file lengths
        with open(fileName) as inputfile:
            count = 0
            for line in inputfile:
                count = count + 1
            fileLength[fileName] = count
            
    ##make a data file with the number of files (4)
    classifiedData = []
    corpusData = []
    stopData = []
    unclassifiedData = []
    dataList = [classifiedData, corpusData, stopData, unclassifiedData]
    
    ## Put all the text files data into 1 list consisting of 4 lists
    i=0
    for fileName in fileList: ##get file lengths
        with open(fileName, encoding='utf8') as inputfile:
            for line in inputfile:
                dataList[i].append(line)
            i = i + 1
    
    stop_dict = {}
    for i in range(0, fileLength[stopFile]):
        x = dataList[2][i].strip('\n')
        current_dict = {x: x}
        stop_dict.update(current_dict)       

    return fileIndex, fileLength, dataList, stop_dict

fileIndex, fileLength, dataList, stop_dict = pre_process()

def clean_data(tw):
    """
    Initailly change every symbol to its lowercase representative.
    Cleans the data, removing the symbols seen in "symbolList"
    All of those symbols are common unicode symbols, as seen in the keyboard, and are typical in tweets

    """
    #remove these symbols from the tweet tw'
    symbolList = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', 
                  '-', '_', '+', '=', '{', '}', '[', ']', '|', ':',
                  ';', '"', '<', '>', ',', '.', '?', '/', ',', "'",
                  '~', '`', '*']
    goodList = []
    tw_lower = tw.lower() #set all tweets to lower case
    for i in range(0, len(tw)):
        if tw_lower[i] not in symbolList:
            goodList.append(tw_lower[i])
    cleanString = ''.join(goodList)
    return cleanString #output type string


def tokenize_unigram(tw):
    """
    Initially not used for remove_stop_words, but it was found to be much more efficient to perform this way, instead
    of by comparing each stop word and each character in the tweet, which could have complexity of O(n^2)
    There is a tendancy for certain words at the output to have a unicode-utf8 encoding problem as seen with text like
    \aoe\ac3\xe4 etc.
    """    
    
    partWord = [] #create a list containing the partial word in a tweet
    tokenList = []
    currentWord = [] #create a list containing the fully spelled word in a tweet
    for t in range(0, len(tw)):
        if tw[t] == " ": #indicate when at a blank space in the tweet
            if len(partWord) != 0:
                currentWord = ''.join(partWord)
                partWord = [] #clear the partWord
                tokenList.append(currentWord.replace('\xc2\xa0', ' '))
        else:
            partWord.append(tw[t])
    #partWord.remove('\n')
    #partWord.remove('\r')
    tokenList.append(''.join((partWord)))
    #tokenList.encode('ascii', 'replace')
    return tokenList #output type list

def bag_of_words(tw): #input is a string
    """
    The works off using the tokenized words from previous sections. 
    Data comes in as a normal string but is immediately tokenized.
    When taking text from a text file, it comes in as a list of tokens
    """
    bag_dict = {}
    if type(tw) == str:
        tokenTweet = tokenize_unigram(tw) #input is a string. Convert to list
    else:
        tokenTweet = tw #input is a list. Use as list.
    for i in range(0, len(tokenTweet)):
        amount = tokenTweet.count(tokenTweet[i])
        current_dict = {tokenTweet[i] : amount}
        bag_dict.update(current_dict)
       
    return bag_dict #output type dictionary

def getSentimentDictionary(sentimentFile):
    sent_dict = {}
    for i in range(0, fileLength[sentimentFile]):
        sentiment_word, sentiment_score = dataList[fileIndex[sentimentFile]][i].split("\t")
        sentiment_score = sentiment_score.strip('\n')
        current_dict = {sentiment_word: sentiment_score}
        sent_dict.update(current_dict)

    return sent_dict

sentimentDictionary = getSentimentDictionary('corpus.txt')

def tweet_score(tw):
    token_tweet = tokenize_unigram(tw) #tw is a string or a list and x is a dictionary
    bag_of_words_tweets = bag_of_words(tw)
    current_score = float(0)
    no_corpus_words = 0
    for i in range(0, len(token_tweet)):
        word_score = sentimentDictionary.get(token_tweet[i])
        if word_score == None:
            word_score = 0.0
            no_corpus_words = no_corpus_words + 1
        word_score = float(word_score)
        current_score = current_score + (word_score/5) # divide by 5 to normalize values -5 to 5 to be between -1 and 1
    if no_corpus_words == len(token_tweet):
        score = current_score/len(token_tweet)
    else:
        score = current_score/(len(token_tweet)-no_corpus_words) #divided by the number of words in the whole tweet
    if score < 0:
        #pred_score = 0
        return 0
    if score > 0:
        #pred_score = 1
        return 1
    #if score == 0: #cannot classify tweet, therefore assign value of -1
    else:
        #pred_score = -1
        return -1
    #sentiment = pred_score
        
    #return sentiment #float
    
def party(tw): #input is a string
    """
    The party words chosen are based on a few criteria in order to catch as many variations
    Some examples include: 
    1) liberal and liberals, which is both plural and possessive ('s with the apostrophe removed)
    2) party leader first name
    3) party leader last name, which is both plural and possessive, simialr to point 1
    4) party leader firstlast name, as seen with some hashtags like #justintrudeau
    5) liberalparty and conservativeparty are common hashtags, but ndpparty is not. Just try saying it you'll know.
    5) omitted words like "party" because it is often used with liberal or conservative, which is already used
    
    This was scored based on the number of token keywords used for each party.
    The one with the most keywords used was the assigned party
    To handle the edge case of a tie, the current highest number was compared to other party's numbers.
    If there was a tie, or if there was zero use of any political keywords in the tweet, the output was "other"
    
    """

    if dict == type(tw):
        current_dict = tw
    else:
        tw.lower()
        tw = clean_data(tw)
        current_dict = bag_of_words(tw) #takes in string and converts to bag of words

    liberal_words = ['liberal', 'liberals', 'liberalparty', 'justin', 'trudeau', 'trudeaus', 'justintrudeau']
    conservative_words = ['conservative', 'conservatives', 'conservativeparty', 'stephen', 'harper', 'harpers', 'stephenharper']
    ndp_words = ['ndp', 'ndps', 'tom', 'thomas' 'mulcair', 'mulcairs', 'tommulcair', 'thomasmulcair']
    party_words = [liberal_words, conservative_words, ndp_words]
    
    liberal_count = 0
    conservative_count = 0
    ndp_count = 0
    party_count = [liberal_count, conservative_count, ndp_count]
    
    #votes = [liberal_count, conservative_count, ndp_count]
    political_parties = ['Liberals', 'Conservatives', 'NDP']
    
    for p in range(0, len(party_count)):
        for w in range(0, len(party_words[p])):
            y = party_words[p][w]
            x = current_dict.get(y) 
            if x == None:
                x = 0
            party_count[p] = party_count[p] + x
            #print("X:",x, party_count[p])
    currentMax = -1 #Used for retaining the count of the party with most counts
    maxName = "" # Used for assigning the name of the current highest count party
    
    #Thi
    for i in range(0, len(political_parties)):
        
        #if currentMax == 0:
        #    maxName = "Other"
        #    print(i, party_count[i], maxName)

        if party_count[i] > currentMax:
            currentMax = party_count[i]
            maxName = political_parties[i]
            #political = 1
            #print(i, party_count[i], currentMax, maxName)
        elif party_count[i] == currentMax: #created a 4th option. Should be political, if TIED use of party words
            maxName = "Tie"
            #political = 1
            #print(i, party_count[i], currentMax, maxName)    
    if currentMax == 0:
        maxName = "Other"
        #political = 0
    
    return maxName  #,political


# In[11]:

def k_cross(clf, data, labels, k, pr=False):
    accuracy = np.mean(cross_val_score(clf, X, y, cv=k, scoring='accuracy'))
    precision = np.mean(cross_val_score(clf, X, y, cv=k, scoring='precision'))
    recall = np.mean(cross_val_score(clf, X, y, cv=k, scoring='recall'))
    f1 = np.mean(cross_val_score(clf, X, y, cv=k, scoring='f1'))
    if pr==True:
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
    scores = cross_val_score(clf, X, y, cv=k)    
    print("ZAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    info = {"accuracy" : accuracy,
            "precision" : precision,
            "recall" : recall,
            "f1" : f1}
    return info

def performance(actual, predicted, info=False):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    P = 0
    N = 0
    
    for i in range(len(predicted)): 

        if actual[i]==predicted[i]==1: 
            TP += 1
        if predicted[i]==1 and actual[i]!=predicted[i]: 
            FP += 1 
        if actual[i]==predicted[i]==0: 
            TN += 1
        if predicted[i]==0 and actual[i]!=predicted[i]: 
            FN += 1
        if actual[i] == 1:
            P += 1
        else:
            N += 1
    if info==True:
        info ={"TP": TP,
               "FP": FP,
               "TN": TN,
               "FN": FN,
               "P" : P,
               "N" : N}
        return info
    return TP, FP, TN, FN, P, N

def confusion_mat(y, pred, pr=False):
    TP, FP, TN, FN, P, N = performance(y, pred)
    TP = float(TP)
    FP = float(FP)
    TN = float(TN)
    FN = float(FN)
    P = float(P)
    N = float(N)
    
    Total = P + N
    Cond_pos = TP +FN
    Cond_neg = FP + TN
    Pred_pos = TP + FP
    Pred_neg = FN + TN
    Accuracy = (TP+TN)/(P+N)
    Precision = TP/(TP+FP)
    Recall  = TP/P
    TPR = Recall
    F1 = (2*TP)/(2*TP+FP+FN)
    
    TNR = TN/N
    NPV = TN/(TN+FN)
    FNR = FN/P
    FPR = FP/N
    FDR = FP/(FP+TP)
    FOR = FN/(FN+TN)
    LR_pos = TPR/FPR
    LR_neg = FNR/TNR
    #DOR = LR_pos/LR_neg
    if pr == True:
        print("Total Population:", Total)
        print("Condition Positive:", Cond_pos)
        print("Condition Negative:", Cond_neg)
        print("Predicted Positive:", Pred_pos)
        print("Predicted Negative:", Pred_neg)
        print("True Positive:", TP)
        print("False Positive:", FP)
        print("False Negative:", FN)
        print("True Negative:", TN)
        print("Accuracy:", Accuracy)
        print("Precision:", Precision)
        print("Recall:", Recall)
        print("F1:", F1)
        print("FDR:", FDR)
        print("NPV:", NPV)
        print("FOR:", FOR)
        print("FPR:", FPR)
        print("FNR:", FNR)
        print("TNR:", TNR)
        print("LR+:", LR_pos)
        print("LR-:", LR_neg)
        #print("DOR:", DOR)
    info = {"accuracy" : Accuracy,
            "precision" : Precision,
            "recall" : Recall,
            "f1" : F1}
    return info
        
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.show()
    return plt

# Compute confusion matrix
def show_confusion(y, pred, class_names):
    cnf_matrix = confusion_matrix(y, pred)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

def prep(data, y=None):
    stopset = set(dataList[2])
    vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset, norm='l2')
    X = vectorizer.fit_transform(data)
    if y == None:
        return X
    else:
        selector = SelectPercentile(f_classif, percentile=10)
        X = selector.fit_transform(X, y)
        return X    


# In[12]:

size = 100000
  
data = [0]*(size*2+len(dataList[3]))
base = [0]*(size*2+len(dataList[3]))

y = np.zeros(len(data))
for i in range(size, len(data)):
    y[i] = 1

for i in range(len(data)):
    if i >= 2*size:
        data[i] = dataList[3][i-2*size]
        base[i] = tweet_score(data[i-2*size])
    else:
        data[i] = dataList[0][i][2:]
        base[i] = tweet_score(data[i])
    
political = [0]*len(dataList[3])
partyName = [0]*len(dataList[3])
unclassified = [0]*len(dataList[3])
for i in range(0, len(dataList[3])):
    partyName[i] = party(dataList[3][i])
    unclassified[i] = dataList[3][i]    


# In[13]:

X = prep(data, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)  

data_train = X[0:200000]
label_train = y[0:200000]
data_test = X[200000:]


# In[14]:

nb = naive_bayes.MultinomialNB()
logReg = LogisticRegression(penalty='l2', C=1)


# In[15]:

k=5
info_nb = k_cross(nb, data_train, label_train, k, True)
info_log = k_cross(logReg, data_train, label_train, k, True)

info_base = confusion_mat(y, base, True)

pred_nb = cross_val_predict(nb, data_train, label_train, k)
pred_logReg = cross_val_predict(logReg, data_train, label_train, k)

_ = confusion_mat(y, pred_nb, True)
_ = confusion_mat(y, pred_logReg, True)

met_nb = performance(label_train, pred_nb, True)
met_logReg = performance(label_train, pred_logReg, True)
#met_base = performance(X, base, True)


# In[16]:

# Choose the type of classifier. 
#clf = RandomForestClassifier()
clf = logReg
# Choose some parameter combinations to try

parameters = {'C':[0.5]}
# make_scorer returns a callable object that scores an estimator’s output.
#We are using accuracy_score for comparing different parameter combinations. 
acc_scorer = make_scorer(accuracy_score)

# Run the grid search for the Random Forest classifier
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(data_train, label_train)

# Set our classifier, clf, to the have the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the selected classifier to the training data
clf.fit(X, y)
_ = k_cross(clf, data_train, label_train, k, True)


# In[17]:

U = prep(unclassified)
sentiment = clf.predict(data_test)


# In[18]:

myParties = set(partyName)
partyList = []
count = 0
countList = []
for e in myParties:
    partyList.append(e)
    countList.append(count)
    count = count + 1
    
binList0 = [0]*count
binList1 = [0]*count

count = 0
for e in myParties:
    #print(e)
    #print(partyName[0])
    for i in range(0, len(sentiment)):
        if partyName[i] == e:
            if sentiment[i] == 0:
                binList0[count] = binList0[count] + 1
            else:
                binList1[count] = binList1[count] + 1
    count = count + 1    
            


# In[19]:

n_groups = 5
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4
error_config = {'ecolor': '0.3'}

lib1 = plt.bar(index,
                 (binList1[2], 0, 0, 0,0),
                 bar_width,
                 alpha=opacity,
                 color = 'r',
                 error_kw=error_config,
                 label=partyList[2])

con1 = plt.bar(index,
                 (0, binList1[4], 0, 0,0),
                 bar_width,
                 alpha=opacity,
                 color = 'b',
                 error_kw=error_config,
                 label=partyList[4])
ndp1 = plt.bar(index,
                 (0, 0, binList1[0], 0,0),
                 bar_width,
                 alpha=opacity,
                 color = 'y',
                 error_kw=error_config,
                 label=partyList[0])
tie1 = plt.bar(index,
                 (0, 0, 0, binList1[3],0),
                 bar_width,
                 alpha=opacity,
                 color = 'g',
                 error_kw=error_config,
                 label=partyList[3])
other = plt.bar(index,(0, 0, 0, 0, binList1[1]),bar_width, alpha=opacity, color = 'k',error_kw=error_config,label=partyList[1])


#plt.xlabel('Classifiers')
plt.ylabel("Positive Political Tweets")
plt.title("Positive Political Tweets")
plt.xticks(index + bar_width / 2, (partyList[2], partyList[4], partyList[0], partyList[3], partyList[1]))
plt.legend(loc=2)
plt.tight_layout()    
plt.show()



# In[20]:

n_groups = 5
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4
error_config = {'ecolor': '0.3'}

lib0 = plt.bar(index,
                 (binList0[2], 0, 0, 0,0),
                 bar_width,
                 alpha=opacity,
                 color = 'r',
                 error_kw=error_config,
                 label=partyList[2])

con0 = plt.bar(index,
                 (0, binList0[4], 0, 0,0),
                 bar_width,
                 alpha=opacity,
                 color = 'b',
                 error_kw=error_config,
                 label=partyList[4])
ndp0 = plt.bar(index,
                 (0, 0, binList0[0], 0,0),
                 bar_width,
                 alpha=opacity,
                 color = 'y',
                 error_kw=error_config,
                 label=partyList[0])
tie0 = plt.bar(index,
                 (0, 0, 0, binList0[3],0),
                 bar_width,
                 alpha=opacity,
                 color = 'g',
                 error_kw=error_config,
                 label=partyList[3])
other = plt.bar(index,(0, 0, 0, 0, binList0[1]),bar_width, alpha=opacity,color = 'k',error_kw=error_config,label=partyList[1])


#plt.xlabel('Classifiers')
plt.ylabel("Negative Political Tweets")
plt.title("Negative Political Tweets")
plt.xticks(index + bar_width / 2, (partyList[2], partyList[4], partyList[0], partyList[3], partyList[1]))
plt.legend(loc=2)
plt.tight_layout()    
plt.show()



# In[21]:

metrics = ["accuracy", "precision", "recall", "f1"]
classifiers = ["NaiveBayes", "LogisticRegression", "Base"]
for i in range(0, len(metrics)):
    typ = metrics[i]
    yplot = [info_nb[typ], info_log[typ], info_base[typ]]
    #xplot = classifiers
    #plt.ylabel(typ)
    #plt.xlabel("Classifiers")
    #plt.plot(xplot, yplot)
    
    n_groups = 3
    
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    bar_nb = plt.bar(index,
                     (yplot[0], 0, 0),
                     bar_width,
                     alpha=opacity,
                     color = 'r',
                     error_kw=error_config,
                     label=classifiers[0])

    bar_logReg = plt.bar(index,
                     (0, yplot[1], 0),
                     bar_width,
                     alpha=opacity,
                     color = 'b',
                     error_kw=error_config,
                     label=classifiers[1])
    bar_base = plt.bar(index,
                     (0, 0, yplot[2]),
                     bar_width,
                     alpha=opacity,
                     color = 'g',
                     error_kw=error_config,
                     label=classifiers[2])
    

    #plt.xlabel('Classifiers')
    plt.ylabel(metrics[i])
    plt.title('Scores by ' + str(metrics[i]))
    plt.xticks(index + bar_width / 2, (classifiers[0], classifiers[1], classifiers[2]))
    plt.legend(loc=4)
    plt.tight_layout()    
    plt.show()
    


# In[30]:

show_confusion(y[0:200000], pred_nb, (0,1))
show_confusion(y[0:200000], pred_logReg, (0,1))
show_confusion(y, base,(-1,0,1))


# In[31]:

nb.fit(X,y)
logReg.fit(X,y)
probaNB = nb.predict_proba(X)
probaLogReg = logReg.predict_proba(X)
yy=y


# In[32]:

nb.fit(X_train,y_train)
logReg.fit(X_train,y_train)
probaNB = nb.predict_proba(X_test)
probaLogReg = logReg.predict_proba(X_test)
yy=y_test


# In[33]:

probaNB1 = [0]*len(probaNB)
probaLogReg1 = [0]*len(probaLogReg)
for i in range(0, len(probaNB)):
    probaNB1[i] = probaNB[i][1]
    probaLogReg1[i] = probaLogReg[i][1]


# In[34]:

# Compute fpr, tpr, thresholds and roc auc
fprNB, tprNB, thresholdsNB = roc_curve(yy, probaNB1)
roc_aucNB = auc(yy, probaNB1, True)

fprLogReg, tprLogReg, thresholdsLogReg = roc_curve(yy, probaLogReg1)
roc_aucLogReg = auc(yy, probaLogReg1, True)


# Plot ROC curve
plt.plot(fprNB, tprNB, label='NB ROC curve (area = %0.3f)' % roc_aucNB)
plt.plot(fprLogReg, tprLogReg, label='LogReg ROC curve (area = %0.3f)' % roc_aucLogReg)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")


# In[35]:

print(roc_auc_score(y_test, nb.predict_proba(X_test)[:,1]))
print(roc_auc_score(y_test, logReg.predict_proba(X_test)[:,1]))

