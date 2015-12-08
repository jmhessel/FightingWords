import numpy as np
from sklearn.feature_extraction.text import CountVectorizer as CV
import string
exclude = set(string.punctuation)

def basicSanitize(inString):
    '''Returns a very roughly sanitized version of the input string.'''
    returnString = ' '.join(inString.encode('ascii', 'ignore').strip().split())
    returnString = ''.join(ch for ch in returnString if ch not in exclude)
    returnString = returnString.lower()
    returnString = ' '.join(returnString.split())
    return returnString

def bayesCompareLanguage(l1, l2, ngram = 1, prior=.01, cv = None):
    '''
    Arguments:
    - l1, l2; a list of strings from each language sample
    - ngram; an int describing up to what n gram you want to consider (1 is unigrams,
    2 is bigrams + unigrams, etc). Ignored if a custom CountVectorizer is passed.
    - prior; either a float describing a uniform prior, or a vector describing a prior
    over vocabulary items. If you're using a predefined vocabulary, make sure to specify that
    when you make your CountVectorizer object.
    - cv; a sklearn.feature_extraction.text.CountVectorizer object, if desired.

    Returns:
    - A list of length |Vocab| where each entry is a (n-gram, zscore) tuple.'''
    if cv is None and type(prior) is not float:
        print "If using a non-uniform prior:"
        print "Please also pass a count vectorizer with the vocabulary parameter set."
        quit()
    l1 = [basicSanitize(l) for l in l1]
    l2 = [basicSanitize(l) for l in l2]
    if cv is None:
        cv = CV(decode_error = 'ignore', min_df = 10, max_df = .5, ngram_range=(1,ngram),
                binary = False,
                max_features = 15000)
    countsMat = cv.fit_transform(l1+l2).toarray()
    # Now sum over languages...
    vocabSize = len(cv.vocabulary_)
    print "Vocab size is {}".format(vocabSize)
    if type(prior) is float:
        priors = np.array([prior for i in range(vocabSize)])
    else:
        priors = prior
    zScores = np.empty(priors.shape[0])
    countMatrix = np.empty([2, vocabSize], dtype=np.float32)
    countMatrix[0, :] = np.sum(countsMat[:len(l1), :], axis = 0)
    countMatrix[1, :] = np.sum(countsMat[len(l1):, :], axis = 0)
    a0 = np.sum(priors)
    n1 = 1.*np.sum(countMatrix[0,:])
    n2 = 1.*np.sum(countMatrix[1,:])
    print "Comparing language..."
    for i in range(vocabSize):
        #compute delta
        term1 = np.log((countMatrix[0,i] + priors[i])/(n1 + a0 - countMatrix[0,i] - priors[i]))
        term2 = np.log((countMatrix[1,i] + priors[i])/(n2 + a0 - countMatrix[1,i] - priors[i]))        
        delta = term1 - term2
        #compute variance on delta
        var = 1./(countMatrix[0,i] + priors[i]) + 1./(countMatrix[1,i] + priors[i])
        #store final score
        zScores[i] = delta/np.sqrt(var)
    indexToTerm = {v:k for k,v in cv.vocabulary_.iteritems()}
    sortedIndices = np.argsort(zScores)
    returnList = []
    for i in sortedIndices:
        returnList.append((indexToTerm[i], zScores[i]))
    return returnList
