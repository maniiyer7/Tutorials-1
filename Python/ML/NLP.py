

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

#######################################
from nltk.corpus import gutenberg
from nltk import FreqDist

# For plotting, we need matplotlib (get it from the NLTK download page)
import matplotlib
import matplotlib.pyplot as plt

# Count each token in each text of the Gutenberg collection
fd = FreqDist()
for text in gutenberg.fileids():
    for word in gutenberg.words(text):
        fd.inc(word)

# Initialize two empty lists which will hold our ranks and frequencies
ranks = []
freqs = []
# Generate a (rank, frequency) point for each counted token and
# and append to the respective lists, Note that the iteration
# over fd is automatically sorted.

for rank, word in enumerate(fd):
    ranks.append(rank+1)
    freqs.append(fd[word])

# Plot rank vs frequency on a log-log plot
plt.loglog(ranks, freqs)
plt.xlabel('frequency(f)', fontsize=14, fontweight='bold')
plt.ylabel('rank(r)', fontsize=14, fontweight='bold')
plt.grid(True)
plt.show()

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

# Find the words that are more than 15 characters long
words = set(brown.words())
long_words = [w for w in words if len(w) > 15]

# Words that are more frequent than 7 times and are more than 7 characters long
rare_and_long = sorted(w for w in set(brown.words()) if len(w) > 7 and fd[w] > 7)


### COLLOCATIONS
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
finder = BigramCollocationFinder.from_words(brown.words('english-web.txt'), window_size=20)
