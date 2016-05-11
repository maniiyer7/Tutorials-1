

import nltk

dir(nltk)
dir(nltk.corpus)
dir(nltk.utilities)
from nltk import utilities

from nltk import corpus
from nltk.corpus import gutenberg

print gutenberg.fileids()
nltk.download('gutenberg')

from nltk import FreqDist
fd = FreqDist()
for word in gutenberg.words('austen-persuasion.txt'):
    fd.inc(word)

dir(fd)
fd.keys()

###############################################################################
########################### PLOT FREQUENCY VS RANK ############################
###############################################################################
from nltk.corpus import gutenberg
from nltk import FreqDist

# For plotting, we need matplotlib (get it from the NLTK download page)
import matplotlib
import matplotlib.pyplot as plt

# Project gutenberg spans multiple books. To see the list of the books and iterate through them, use gutenberg.fileids()
print(gutenberg.fileids())

# Count each token in each text of the Gutenberg collection
fd = FreqDist(brown.words())

# # The following code is deprecated
for text in gutenberg.fileids():
    for word in gutenberg.words(text):
        fd[word] += 1
        # fd.inc(word)  # deprecated. superseded by the line above

# Initialize two empty lists which will hold our ranks and frequencies
ranks = []
freqs = []

# Generate a (rank, frequency) point for each counted token and
# and append to the respective lists, Note that the iteration
# over fd is automatically sorted.
for rank, word in enumerate(fd):
    ranks.append(rank+1)
    freqs.append(fd.freq(word))
word

# Plot rank vs frequency on a log-log plot
plt.loglog(ranks, freqs)
plt.ylabel('frequency(f)', fontsize=14, fontweight='bold')
plt.xlabel('rank(r)', fontsize=14, fontweight='bold')
plt.grid(True)
plt.show()
plt.close()
###############################################################################

###############################################################################
############################## PREDICTING WORDS ###############################
###############################################################################
### PREDICTING WORDS
from nltk.corpus import gutenberg
from nltk import ConditionalFreqDist
from random import choice
# Create distribution object
cfd = ConditionalFreqDist()
# For each token, count current word given previous word
prev_word = None

for word in gutenberg.words('austen-persuasion.txt'):
    cfd[prev_word][word] += 1
    prev_word = word

# Start predicting at the given word, say ’therefore’
word = 'therefore'
i = 1
# Find all words that can possibly follow the current word
# and choose one at random
while i < 20:
    print word,
    lwords = cfd[word].keys()
    follower = choice(lwords)
    word = follower
    i += 1
###############################################################################

###############################################################################
###################### DISCOVERING PART-OF-SPEECH TAG #########################
###############################################################################
# http://www.nltk.org/book/ch05.html
# tagging is the second step in the typical NLP pipeline, following tokenization.
# sequence labeling, n-gram models, backoff, and evaluation
from nltk.corpus import brown
from nltk import FreqDist, ConditionalFreqDist


### What is the most frequent tag?
### Which word has the most number of distinct tags?
fd = FreqDist()
cfd = ConditionalFreqDist()

# for each tagged sentence in the corpus, get the (token, tag) pair and update
# both count(tag) and count(tag given token)
for sentence in brown.tagged_sents():
    for (token, tag) in sentence:
        fd[tag] += 1
        cfd[token][tag] += 1

# Find the most frequent tag
fd.max()

# Initialize a list to hold (numtags,word) tuple
wordbins = []

# Append each tuple (number of unique tags for token, token) to list
for token in cfd.conditions():
    wordbins.append((cfd[token].B(), token))

# sort tuples by number of unique tags (highest first)
wordbins.sort(reverse=True)
print wordbins[0] # token with max. no. of tags is ...


### What is the ratio of masculine to feminine pronouns?
male = ['he','his','him','himself']  # masculine pronouns
female = ['she','hers','her','herself']  # feminine pronouns
n_male, n_female = 0, 0  # initialize counters

# total number of masculine samples
for m in male:
    n_male += cfd[m].N()

# total number of feminine samples
for f in female:
    n_female += cfd[f].N()

print float(n_male)/n_female  # calculate required ratio


### How many words are ambiguous, in the sense that they appear with at least two tags?
n_ambiguous = 0
for (ntags, token) in wordbins:
    if ntags > 1:
        n_ambiguous += 1

n_ambiguous # number of tokens with more than a single POS tag
###############################################################################

###############################################################################
############################## WORD ASSOCIATION ###############################
###############################################################################
from nltk.corpus import brown, stopwords
from nltk import ConditionalFreqDist
cfd = ConditionalFreqDist()

# get a list of all English stop words
# stop words are very common words that are not helpful in text recognition.
stopwords_list = stopwords.words('english')

# define a function that returns true if the input tag is some form of noun
def is_noun(tag):
    return tag.lower() in ['nn','nns','nn$','nn-tl','nn+bez',\
                            'nn+hvz', 'nns$','np','np$','np+bez','nps',\
                            'nps$','nr','np-tl','nrs','nr$']

# count nouns that occur within a window of size 5 ahead of other nouns
for sentence in brown.tagged_sents():
    for (index, tagtuple) in enumerate(sentence):
        (token, tag) = tagtuple
        token = token.lower()
        if token not in stopwords_list and is_noun(tag):
            window = sentence[index+1:index+5]
            for (window_token, window_tag) in window:
                window_token = window_token.lower()
                if window_token not in stopwords_list and is_noun(window_tag):
                    cfd[token][window_token] +=1

# Associating
print cfd['left'].max()

print cfd['life'].max()

print cfd['man'].max()

print cfd['woman'].max()

print cfd['boy'].max()

print cfd['girl'].max()

print cfd['male'].max()

print cfd['ball'].max()

print cfd['doctor'].max()

print cfd['road'].max()
###############################################################################

###############################################################################
############################# FIND COLLOCATIONS ###############################
###############################################################################
# Generate a list of word tuples (bigram) to examine collocations
from nltk import bigrams
bigram_tuples = list(bigrams(brown.words()))
len(bigram_tuples)

# http://www.nltk.org/howto/collocations.html
# collocations are essentially just frequent bigrams,
# except that we want to pay more attention to the cases that involve rare words.
# In particular, we want to find bigrams that occur more often than we would expect
# based on the frequency of the individual words. The collocations() function does this for us.
from nltk import collocations
from nltk import BigramCollocationFinder
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder = BigramCollocationFinder.from_words(brown.words())
finder.nbest(bigram_measures.pmi, 10)

# apply filters to collocations, such as ignoring all bigrams which occur less than three times in the corpus
finder.apply_freq_filter(3)
finder.nbest(bigram_measures.pmi, 10)

# find collocations among tagged words
finder = BigramCollocationFinder.from_words(brown.tagged_words('ca01', tagset='universal'))
finder.nbest(bigram_measures.pmi, 5)

# tags alone
finder = BigramCollocationFinder.from_words(t for w, t in  brown.tagged_words('ca01', tagset='universal'))
finder.nbest(bigram_measures.pmi, 5)

# Spanning intervening words
finder = BigramCollocationFinder.from_words(brown.words(), window_size=20)


###############################################################################
### NLTK Book
import nltk
from nltk import FreqDist
# Donwload all books (http://www.nltk.org/data.html)
# NOTE: if this does not work, run this code in Python from the Terminal (not from inside IDE)
# nltk.download()

# Import a text and examine its words
from nltk.corpus import brown
brown.words()

# Find the frequency of each word in a text
fd = FreqDist(brown.words())

# Find the most frequent words in a text:
# http://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
import operator
max(fd.iteritems(), key=operator.itemgetter(1))
sorted(fd.iteritems(), key=operator.itemgetter(1), reverse=True)[:10]
# Or use the wrapper function
fd.most_common(10)

# plot the most frequent words
fd.plot(10)
fd.plot(10, cumulative=True)

# See the words with lowest frequency (these words are called hapaxes)
fd.hapaxes()

# Count all the words
len(text1)
# count unique words
len(set(text1))
# count unique words, irrespective of word case
len(set(w.lower() for w in text1))


# Find the words that are more than 15 characters long
words = set(brown.words())
long_words = [w for w in words if len(w) > 15]

# Words that are more frequent than 7 times and are more than 7 characters long
rare_and_long = sorted(w for w in set(brown.words()) if len(w) > 7 and fd[w] > 7)


### SUMMARY
# Other functions and attributes of frequency distribution object
fd = FreqDist(brown.words())
fd['County']  # count of a specific word
fd.freq('County')  # frequency of a specific word
fd.N()  # total number of samples
fd.most_common(10)
for sample in fd:
    print sample
fd.max()
fd.tabulate()
fd.plot()
fd1 |= fd2  # update fd1 with counts from fd2
fd1 < fd2  # test if samples in fd1 occur less frequenctly than in fd2


### IMPORTING TEXT
# NLTK comes with a collection of texts to get started. To import a specific text:
from nltk.book import text1
from nltk.book import sent7

### USING CONDITIONALS
# select words based on their length
[w for w in sent7 if len(w) < 4]
# select words based on other attributes
w.startswith('t')  # same as w[0]=='t'
w.endswith('t')  # same as w[-1]=='t'
't' in w
w.islower()  # test if w contains cased characters and all are lowercase
w.isupper()  # test if w contains cased characters and all are uppercase
w.isalpha()  # test if w is non-empty and all characters in w are alphabetic
w.isalnum()  # test if w is non-empty and all characters in we are alphanumeric
w.isdigit()  # test if w is non-empty and all characters in w are digits
w.istitle()  # test if w contains cased characters and is titlecased (ie, all words in w have initial capitals)

sorted(fd)  # sorts based on the keys (the words), not the counts. If you want to sort by counts:
sorted(fd.iteritems(), key=operator.itemgetter(1))  # To display highest-frequency first: reverse=True
sorted(w for w in set(fd) if w.endswith('ableness'))
sorted(w for w in set(fd) if (w.islower() and len(w) > 5) or ('I' in w))


#######################################
### OTHER STATS ABOUT THE DOCUMENT
# Find the distribution of length of words in a document
fdist = FreqDist(len(w) for w in brown.words())

fdist.most_common(5)
sorted(fdist.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]


### READING
# http://textminingonline.com/dive-into-nltk-part-i-getting-started-with-nltk
# https://www.packtpub.com/books/content/python-text-processing-nltk-storing-frequency-distributions-redis
# https://blogs.princeton.edu/etc/files/2014/03/Text-Analysis-with-NLTK-Cheatsheet.pdf
# http://www.nltk.org/py-modindex.html
# http://www.nltk.org/howto/collocations.html
# http://www.nltk.org/book/ch05.html

