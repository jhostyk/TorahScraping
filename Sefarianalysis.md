# Torah scraping

### Looking for hidden patterns.
### and not so hidden patterns.
### 2020-4-21
### Joe Hostyk and Alex Zaloum


```
import csv
from IPython.display import display, clear_output
import os
import itertools
from collections import defaultdict, Counter
import numpy as np

import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

# import spacy
import re
import unicodedata

from functools import partial
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, hstack
```


```
### UMAP:
import umap
import umap.plot
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib notebook
from bokeh.plotting import show, save, output_notebook, output_file
from bokeh.resources import INLINE
output_notebook(resources=INLINE)
import sklearn.cluster as cluster

```



<div class="bk-root">
    <a href="https://bokeh.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
    <span id="1003">Loading BokehJS ...</span>
</div>





```
GMATRIA = {"א": 1, "ב": 2, "ג": 3, "ד": 4, "ה": 5, "ו": 6, "ז": 7, "ח": 8, "ט": 9, "י": 10, "כ": 20, "ך": 20, "ל": 30, "מ": 40, "ם": 40, "נ": 50, "ן": 50, "ס": 60, "ע": 70, "פ": 80, "ף": 80, "צ": 90, "ץ": 90, "ק": 100, "ר": 200, "ש": 300, "ת": 400}
ALEPH_BEIS = GMATRIA.keys()
```

### Load texts


```
def seferNameFromPath(seferPath):
    """
    Parse out the name. This keeps changing based on folder structure.

    Args:
        seferPath (str): Full path to the sefer.

    Returns:
        The name itself.
    """
    
    return seferPath.split("/")[-3]


def getFilePaths(folder, additionalSearch = "", hebrewOrEnglish = "Hebrew"):
    """
    Recursively go through a folder and get all file names, for processing later.

    Args:
        folder (str): Full path to the folder.
        additionalSearch (str): Optional extra term in the file name. Useful for some
            Sefaria names. For gmara, "merged.txt" seemed to be the best Hebrew version.
            For Tanach, "Text Only.txt" was a non-nekudot, good version.
        hebrewOrEnglish (str): We saved a Hebrew and English version for each sefer. The
            user can specify which to load. Defaults to Hebrew.

    Returns:
        filePaths (list of strings): 
    """

#     print ("Getting file paths...")

    filePaths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if ".txt" in file and hebrewOrEnglish in root and additionalSearch in file:
                filePaths.append(os.path.join(root, file))
    return filePaths


### Previous folder format:
def getAllTanachPaths():
    """
    Get all the links at once.

    Returns:
        tanachPaths (dict): {"Torah": [paths to Torah sfarim], "Neviyim": [...], "Ktuvim": [...]}
    """   
    
    tanachPaths = dict()
    
    sections = ["Torah", "Neviyim", "Ktuvim"]
    for section in sections:
        tanachPaths[section] = getFilePaths("Texts/{}".format(section), additionalSearch = "Text Only")

    return tanachPaths

def getPathsForSections(sections):
    """
    Using the Sefaria-defined sections (e.g. "Tanakh", "Apocrypha"), get the paths
    to "merged.txt" in every subfolder.

    Returns:
        seferFilePaths (list): Full paths to the files.
    """
    
    seferFilePaths = []
    for section in sections:
        
        seferFilePaths += getFilePaths("Texts/{}".format(section))
    
    return seferFilePaths
        

getTanachPaths = partial(getPathsForSections, sections = ["Tanakh/Torah", "Tanakh/Prophets", "Tanakh/Writings"])

sdarim = ["Kodashim", "Moed", "Nashim", "Nezikin", "Tahorot", "Zeraim"]
sdarimPaths = ["Talmud/Bavli/Seder " + seder for seder in sdarim]
getTalmudPaths = partial(getPathsForSections, sections = sdarimPaths)

getAllPaths = partial(getPathsForSections, sections = ["."])
    
```


```
print ("We have {} Sefaria sfarim.".format(len(getAllPaths())))
```

    We have 83 Sefaria sfarim.


### Intro helper functions.


```
def makeNgramsDictionaryFromSefer(seferPath, sizeOfNgram, byPerek = False, keep_nekudot = False):
    """
    Read through a file and get counts of all the n-grams in it.

    Args:
        seferPath (str): Full path to the sefer file.
        sizeOfNgram (int): the length of the phrases (n-grams) to return
        byPerek (bool): If false (default), return counts of the words for the whole sefer.
            If true, return counts of words per perek.

    Returns:
        seferWords (Counter, or dict of Counters): If not byPerek, Counter of words to counts.
            If byPerek, dictionary of perek names to Counters of words to counts.
    """ 
    
    seferName = seferNameFromPath(seferPath)
    if not byPerek:
        seferNgrams = Counter()
    if byPerek:
        seferNgrams = defaultdict(Counter)
    problematicWords = set()
    with open(seferPath, "r") as psukim:
        
        ### Skip first 9 lines - metadata
        for i in range(9):
            next(psukim)
            
        for pasuk in psukim:

            ### Some quick cleaning:
            cleanedPasuk = pasuk.strip().replace("־", " ").replace("[", "").replace("]", "").replace("׃", "")
            cleanedPasuk = cleanedPasuk.replace("(פ)", "").replace("(ס)", "") ### This is for now; change later if splitting on these.
            
            ### Deal with nekudot: https://gist.github.com/yakovsh/345a71d841871cc3d375#gistcomment-1672181
            if not keep_nekudot:
                normalized = unicodedata.normalize('NFKD', cleanedPasuk)
                cleanedPasuk = "".join([c for c in normalized if not unicodedata.combining(c)])
            
            splitPasuk = cleanedPasuk.split(" ")

            # Skip the non-text lines.
            if len(splitPasuk) == 1:
                continue
            if splitPasuk[0] in ["Chapter", "Section", "Seif", "Siman", "Part", "Daf"]: ## "Paragraph"?
                
                if byPerek:
                    
                    perekNumber = splitPasuk[1]
                    perekName = seferName + " " + perekNumber
                    
                continue

                
            ### Delete spaces:
            if [""] in splitPasuk:
                splitPasuk.remove([""])
            for wordIndex in range(len(splitPasuk) - sizeOfNgram + 1):
                
#                 if word == "":
#                     continue
                try:
                    
                    ngram = " ".join(splitPasuk[wordIndex:wordIndex + sizeOfNgram])
                    if not byPerek:
                        seferNgrams[ngram] += 1
                    if byPerek:
                        seferNgrams[perekName][ngram] += 1
                except KeyError as e:
                    problematicWords.add(ngram)
                              
    return seferNgrams

makeWordDictionaryFromSefer = partial(makeNgramsDictionaryFromSefer, sizeOfNgram = 1)

def makeLetterDictionaryFromSefer(filePath, allLetters):
    
    ### Skip first 9 lines - metadata
    for i in range(9):
        next(psukim)
            
    with open(filePath, "r") as psukim:
        for pasuk in psukim:

            cleanedPasuk = pasuk.strip().replace("־", " ")
            splitPasuk = cleanedPasuk.split(" ")

            # Skip the non-text lines.
            if len(splitPasuk) == 1 or splitPasuk[0] == "Chapter":
                continue

            for word in splitPasuk:
                try:
                    for letter in word:
                        allLetters[letter] += 1
                except KeyError as e:
                    problematicWords.add(word)
    return allLetters  

def getAllFiles(filePaths):
    
    allWords = {}
    allLetters = Counter()
    allNgrams = Counter()
    
    for filePath in filePaths:
        
        seferName = seferNameFromPath(filePath)
        print(seferName)
#         allWords[seferName] = makeDictionaryFromSefer(filename)
#         allLetters = makeLetterDictionaryFromSefer(filename, allLetters)
#         allNgrams = makeNgramsDictionaryFromSefer(filePath, allNgrams, sizeOfNgram = 2)

#         raise
    return allNgrams
        

```


```
def getNgramCountsBySefer(seferPaths, sizeOfNgram, byPerek = False, removeExclusiveNgrams = True, keep_nekudot = False):
    """
    Get counts of all words in a list of sfarim.

    Args:
        seferPaths (list of strings): Full paths to the sfarim to analyze.
        byPerek (bool): If false (default), return counts of the words by sefer.
            If true, return counts of words per perek.
        removeExclusiveNgrams (bool): Remove any ngrams, that only shows up in one
            of the selected sections.

    Returns:
        sfarimToWords (Counter): If not byPerek, Counter of words to counts.
            If byPerek, dictionary of perek names to Counters of words to counts.
        sfarimToSizeOfSefer (dict): 
    """   
    
    sfarimToWords = {} ### Dict of dicts
    sfarimToSizeOfSefer = dict() ### dict of {seferName: int}
    
    
    ### Keep track of total counts, to remove later:
    for seferPath in seferPaths:
        

        seferName = seferNameFromPath(seferPath)
        wordsInSefer = makeNgramsDictionaryFromSefer(seferPath = seferPath, sizeOfNgram = sizeOfNgram, byPerek = byPerek, keep_nekudot = keep_nekudot)
#         wordsInSefer = makeWordDictionaryFromSefer(seferPath = seferPath, byPerek = byPerek)

        if not byPerek:
            sfarimToWords[seferName] = wordsInSefer
            sfarimToSizeOfSefer[seferName] = sum(wordsInSefer.values())
            

        ### If running by perek, have a dict of Counters. Need to extract.
        if byPerek:
#           sfarimToWords.update(wordsInSefer)
            for perekName, perekCounts in wordsInSefer.items():

                sfarimToWords[perekName] = perekCounts
                sfarimToSizeOfSefer[perekName] = sum(perekCounts.values())
    
    if not removeExclusiveNgrams:
        
        return sfarimToWords, sfarimToSizeOfSefer
    
    ### Otherwise, by default, reemove ngrams that only show up in one of the sections.

    wordsToNumberOfSfarimTheyAppearIn = Counter()

    ## Find how many sfarim each word appears in.
    for seferName, seferToWords in sfarimToWords.items():

        for word in seferToWords:

            wordsToNumberOfSfarimTheyAppearIn[word] += 1


    ### Remove the most infrequent:
    wordsToRemove = set()
    for word, numberOfSfarimThatWordAppearsIn in wordsToNumberOfSfarimTheyAppearIn.items():

        if numberOfSfarimThatWordAppearsIn == 1: # < len(seferPaths) / 10.0: ### magic parameter to play around with:


            wordsToRemove.add(word)

    print ("Removing {} ngrams that only appear in one section.".format(len(wordsToRemove)))

    ### Deleting from original dict took too long, so we create a new one, and delete the entire old one.
    cleanedSfarimToWords = {}

    for seferName, seferToWords in sfarimToWords.items():

        cleanedSfarimToWords[seferName] = Counter({word: counts for word, counts in seferToWords.items() if word not in wordsToRemove})

    sfarimToWords.clear()

                
    return cleanedSfarimToWords, sfarimToSizeOfSefer

getWordCountsBySefer = partial(getNgramCountsBySefer, sizeOfNgram = 1)

getTanachWordCountsBySefer = partial(getWordCountsBySefer, seferPaths = getTanachPaths())
getTalmudWordCountsBySefer = partial(getWordCountsBySefer, seferPaths = getTalmudPaths())
getAllWordCountsBySefer = partial(getWordCountsBySefer, seferPaths = getAllPaths())

getTanachNgramCountsBySefer = partial(getNgramCountsBySefer, seferPaths = getTanachPaths())
getTalmudNgramCountsBySefer = partial(getNgramCountsBySefer, seferPaths = getTalmudPaths())
getAllNgramCountsBySefer = partial(getNgramCountsBySefer, seferPaths = getAllPaths())

```


```
folder = "./texts/Torah"
filePaths = getFilePaths(folder, additionalSearch = "Text Only")
# allWords = getAllFiles(filenames)
# allLetters = getAllFiles(filenames)
# allNgrams = getAllFiles(filenames)
```


```
# print ("There are {} total bigrams in Vayikra and {} words.".format(sizeOfSeferInBigrams["Leviticus 1"], sizeOfSeferInWords["Leviticus 1"]))
# print ("There are {} unique bigrams and {} unique words.".format(len(bigramCounts["Leviticus 1"]), len(wordCounts["Leviticus 1"])))

```

### Check which pairs never show up in the Torah


```
allPossiblePairs = list(itertools.product(ALEPH_BEIS, ALEPH_BEIS))
allPossiblePairs = ["".join(pair) for pair in allPossiblePairs]
# allPossiblePairs
```


```
print([pair for pair in allPossiblePairs if pair not in allNgrams])
```

    ['בף', 'גט', 'גכ', 'גס', 'גצ', 'גץ', 'גק', 'דז', 'דט', 'דס', 'דצ', 'דץ', 'הף', 'הץ', 'זט', 'זס', 'זף', 'זצ', 'זץ', 'זש', 'חא', 'חע', 'טג', 'טז', 'טכ', 'טס', 'טצ', 'טץ', 'טק', 'כץ', 'ךא', 'ךב', 'ךג', 'ךד', 'ךה', 'ךו', 'ךז', 'ךח', 'ךט', 'ךי', 'ךכ', 'ךך', 'ךל', 'ךמ', 'ךם', 'ךנ', 'ךן', 'ךס', 'ךע', 'ךפ', 'ךף', 'ךצ', 'ךץ', 'ךק', 'ךר', 'ךש', 'ךת', 'מף', 'םא', 'םב', 'םג', 'םד', 'םה', 'םו', 'םז', 'םח', 'םט', 'םי', 'םכ', 'םך', 'םל', 'םמ', 'םם', 'םנ', 'םן', 'םס', 'םע', 'םפ', 'םף', 'םצ', 'םץ', 'םק', 'םר', 'םש', 'םת', 'ןא', 'ןב', 'ןג', 'ןד', 'ןה', 'ןו', 'ןז', 'ןח', 'ןט', 'ןי', 'ןכ', 'ןך', 'ןל', 'ןמ', 'ןם', 'ןנ', 'ןן', 'ןס', 'ןע', 'ןפ', 'ןף', 'ןצ', 'ןץ', 'ןק', 'ןר', 'ןש', 'ןת', 'סז', 'סט', 'סצ', 'סץ', 'סש', 'עח', 'עע', 'עף', 'פב', 'פפ', 'ףא', 'ףב', 'ףג', 'ףד', 'ףה', 'ףו', 'ףז', 'ףח', 'ףט', 'ףי', 'ףכ', 'ףך', 'ףל', 'ףמ', 'ףם', 'ףנ', 'ףן', 'ףס', 'ףע', 'ףפ', 'ףף', 'ףצ', 'ףץ', 'ףק', 'ףר', 'ףש', 'ףת', 'צז', 'צס', 'צש', 'ץא', 'ץב', 'ץג', 'ץד', 'ץה', 'ץו', 'ץז', 'ץח', 'ץט', 'ץי', 'ץכ', 'ץך', 'ץל', 'ץמ', 'ץם', 'ץנ', 'ץן', 'ץס', 'ץע', 'ץפ', 'ץף', 'ץצ', 'ץץ', 'ץק', 'ץר', 'ץש', 'ץת', 'קג', 'קז', 'שצ', 'שץ']


### What are the most common pairs in the Torah?


```
allNgrams.most_common(5)
```




    [('את', 5543), ('וי', 4352), ('אל', 4090), ('ים', 3970), ('יה', 3734)]




```
### Most common:
allNgrams.most_common(575)

### Least common:
allNgrams.most_common(575)[-30:]
```




    [('טד', 5),
     ('סס', 5),
     ('קא', 4),
     ('כט', 4),
     ('קף', 4),
     ('זם', 4),
     ('חף', 3),
     ('א\u200d', 3),
     ('\u200dש', 3),
     ('זך', 3),
     ('שף', 3),
     ('עס', 3),
     ('זח', 3),
     ('קק', 3),
     ('אט', 3),
     ('פמ', 3),
     ('נץ', 2),
     ('תץ', 2),
     ('לץ', 2),
     ('פף', 2),
     ('זפ', 2),
     ('צץ', 2),
     ('טט', 1),
     ('אץ', 1),
     ('סן', 1),
     ('אא', 1),
     ('צט', 1),
     ('קכ', 1),
     ('הך', 1),
     ('זג', 1)]




```
# sortedLetters = {k: v for k, v in sorted(allLetters.items(), key=lambda item: item[1])}

# plt.hist(list(sortedLetters.keys()), list(sortedLetters.values()))
```

### Heatmap


```
### Todo: make a heatmap of 24x24 letters

```

### Get gmatrias:


```
def gmatrifyAword(word):
    """
    Get the gmatria for one word. Doesn't catch punctuation/errors.

    Args:
        word (str)

    Returns:
        The gmatria (int)

    """

    return sum([GMATRIA[letter] for letter in word])


def calculateGmatriaForAsefer(seferPath):
    """
    Get the gmatria for every word in a text.

    Args:
        seferPath (str): Path to a file from Sefaria.

    """
#     print ("Gmatrifying {}...".format(seferPath))

    gmatriasToWords = defaultdict(set)
    problematicWords = set()
    with open(seferPath, "r") as psukim:
        
        ### Skip first nine lines of file - metadata.
        for i in range(9):
            next(psukim)
        for pasuk in psukim:

            cleanedPasuk = pasuk.strip().replace("־", " ").replace("\u200d", " ").replace("[", "").replace("]", "")
            splitPasuk = cleanedPasuk.split(" ")

            # Skip the non-text lines.
            if len(splitPasuk) == 1 or splitPasuk[0] == "Chapter":
                continue

            for word in splitPasuk:
                try:
                    gmatria = gmatrifyAword(word)
                    gmatriasToWords[gmatria].add(word)
                except KeyError as e:
                    print (splitPasuk)
                    problematicWords.add(word)
    
    
    return gmatriasToWords

    
def writeWordsAndTheirGmatria(gmatriasToWords, outputFilePath):
    """
    Write out our results.

    Args:
        gmatriasToWords (defaultdict(set)): {Gmatria1 (int): {word1withThatGmatria, word2, ...}, ...}
        outputFilePath (str): Path to write to.
    """

    writer = csv.writer(open(outputFilePath, "w"), delimiter = "\t")
    writer.writerow(["Gmatria", "Word"])
    for gmatria, wordsWithThatGmatria in sorted(gmatriasToWords.items()):

        writer.writerow([gmatria, " | ".join(wordsWithThatGmatria)])
        
    
### todo clean this up - either calculating, or reading.
def getAllGmatrias():
    
    gmatriasToWordsForTanach = defaultdict(dict)

    tanachPaths = getAllTanachPaths()    

    
    ### Sections are Torah, Neviyim, Ktuvim
    for section, seferPaths in tanachPaths.items():
        
        gmatriasToWordsForSefer = defaultdict(dict)

        
        for seferPath in seferPaths:
            seferName = seferPath.split("/")[2]
            

            pathToGmatriaFile = "Gmatria/{}GmatriaByNumber.tsv".format(seferName)

            if os.path.exists(pathToGmatriaFile):
#                 print ("Gmatria file exists. Reading...")
                reader = csv.reader(open(pathToGmatriaFile), delimiter = "\t")
                header = next(reader)
                for line in reader:
                    line = dict(zip(header, line))
                    word = int(line["Gmatria"])
                    shifts = set(line["Word"].split(" | "))
                    gmatriasToWords[word] = shifts
#                 print ("Finished reading.")
            else:
#                 print ("Gmatria file does not exist. Creating...")
                gmatriasToWords = calculateGmatriaForAsefer(seferPath)
                writeWordsAndTheirGmatria(gmatriasToWords, outputFilePath = pathToGmatriaFile)


            
    return gmatriasToWords
```


```
getAllGmatrias()
```


```
def analyzeGematria(gmatriasToWords):
    """
    Analysis.

    Args:
        gmatriasToWords (defaultdict(set)): {Gmatria1 (int): {word1withThatGmatria, word2, ...}, ...}
        originalFilename (str): The text from which the Gmatria dict was made.
    """

    print ("The largest gmatrias:")

    for gmatria, wordsWithThatGmatria in gmatriasToWords.items():

        if len(wordsWithThatGmatria) > 20:

            print ("Gmatria {} has words: {}".format(gmatria, wordsWithThatGmatria))
```


```
gmatriasToWords = getAllGmatrias()
gmatriasToWords[220]
```

    Gmatria file exists. Reading...
    Finished reading.





    {'הארדי', 'וטהר', 'וידר', 'וירד', 'ורוח', 'טהור', 'יבחר', 'צפים', 'רוחו'}



## Unique words


```
sfarimToWords, sfarimToSizeOfSefer = getTanachWordCountsBySefer(byPerek = False)
```

    Removing 24369 ngrams that only appear in one section.



```
sfarimToWords["Genesis"].most_common(5)
```




    [('ויאמר', 1735), ('את', 1334), ('אשר', 1071), ('אלהים', 755), ('אל', 690)]




```
### Write out most common per sefer

amountToWrite = 20
outputFilePath = "MostCommonWordsBySefer.tsv"
header = ["Sefer"] + [str(i) for i in range(1, amountToWrite + 1)]
writer = csv.writer(open(outputFilePath, "w"), delimiter = "\t")
writer.writerow(header)

for sefer, counts in sfarimToWords.items():
    
    topCounts = counts.most_common(amountToWrite)
    topWords = [word for word, count in topCounts]
    writer.writerow([sefer] + topWords)

```


```
### Get counts per word, across all of Tanach

wordsToTanachCounts = Counter()
wordsToNumberOfSfarimTheyAppearIn = Counter()

for seferName, seferToWords in sfarimToWords.items():
        
    for word, wordCounts in seferToWords.items():
        
        wordsToTanachCounts[word] += wordCounts
        wordsToNumberOfSfarimTheyAppearIn[word] += 1

print ("There are {} total words in Tanach.".format(sum(wordsToTanachCounts.values())))
```

    There are 306881 total words in Tanach.


#### Quick Torah comparison


```
torahSfarim = set(["Breishit", "Shmot", "Vayikra", "Bamidbar", "Dvarim"])
for sefer in torahSfarim:

    seferWords = set(sfarimToWords[sefer])
    print ("{} has {} words.".format(sefer, len(seferWords)))

combos = itertools.combinations(torahSfarim, 2)
for sefer1, sefer2 in combos:

    sefer1words = set(sfarimToWords[sefer1])
    sefer2words = set(sfarimToWords[sefer2])

    print ("{} and {} share {} words.".format(sefer1, sefer2, len(sefer1words.intersection(sefer2words))))
#     print ("{} has {} unique words.".format(sefer1, len(sefer1words.difference(sefer2words))))
#     print ("{} has {} unique words.".format(sefer2, len(sefer2words.difference(sefer1words))))

```

    Breishit has 5007 words.
    Bamidbar has 3847 words.
    Vayikra has 2710 words.
    Dvarim has 4089 words.
    Shmot has 4171 words.
    Breishit and Bamidbar share 1323 words.
    Breishit and Vayikra share 860 words.
    Breishit and Dvarim share 1313 words.
    Breishit and Shmot share 1482 words.
    Bamidbar and Vayikra share 973 words.
    Bamidbar and Dvarim share 1201 words.
    Bamidbar and Shmot share 1412 words.
    Vayikra and Dvarim share 858 words.
    Vayikra and Shmot share 1007 words.
    Dvarim and Shmot share 1302 words.


Scenarios for our scoring system:

__Starting:__

1: aaaab -> ab<br>
2: abbb -> ab<br>
3: aaaac -> ac

totals: a: 9, b: 4, c: 1

compare 1 and 3:<br>
1/9 * 4 = 16/36

compare 1 and 2:<br>
1/9 + 1/4 = 13/36

compare 2 and 3:<br>
1/9 = 4/36


compare 1, 2, and 3:<br>
1/9 = 4/36

__Rare words?__

a is super common: 2500<br>
b is semi common: 500<br>
c is rare: 2<br>

1: a * 750, b * 249, c * 1<br>
2: a* 999, c * 1<br>
3: a * 750, b * 250


compare 1 and 2:<br>
750/2500 + 1/2

compare 1 and 3:<br>
750/2500, 250/500


```
### This counts each word once.

def compareSfarimNaive(listOfSfarim):
    
    unionWords = set()
    intersectionWords = set(sfarimToWords[listOfSfarim[0]])
    
    for sefer in listOfSfarim:
        
        seferWords = set(sfarimToWords[sefer])
        
        unionWords |= seferWords
        intersectionWords &= seferWords
        
    total = len(unionWords)
    shared = len(intersectionWords)
    sharedPercentage = float(shared) / total * 100
    print ("There are {} unique words across the {} sfarim.".format(total, len(listOfSfarim)))
    print ("They share {} of them ({:.2f} percent).".format(shared, sharedPercentage))
    
    ### Weighted shared percentage:
    
    individualSharedPercentages = {}
    
    for sefer in listOfSfarim:
        
        seferWords = set(sfarimToWords[sefer])
        sefersSharedWordsWithAllOtherSfarim = seferWords.intersection(intersectionWords)
        
        sefersSharedPercentage = float(len(sefersSharedWordsWithAllOtherSfarim)) / len(seferWords)
        
        individualSharedPercentages[sefer] = sefersSharedPercentage
    
    meanSharedPercentage = 100 * sum(individualSharedPercentages.values())/len(individualSharedPercentages)
    print ("The average shared percentage is {:.2f}.".format(meanSharedPercentage))
    
    for sefer in listOfSfarim:
        
        print ("{}: {:.2f}".format(sefer, 100*individualSharedPercentages[sefer]))
        
            
```


```
### This uses the Alex Joe weighting system (ie tf=idf?)

def compareSfarimScoring(listOfSfarim):
    
    unionWords = set()
    intersectionWords = set(sfarimToWords[listOfSfarim[0]])
    numberOfSfarimInTanach = len(sfarimToWords)
    numberOfSfarimInThisSet = len(listOfSfarim)
    

    for sefer in listOfSfarim:
        
        seferWords = set(sfarimToWords[sefer])
        
        intersectionWords &= seferWords
        
    ### Calculate score:
    score = 0
    for word in intersectionWords:
        
        numberOfSfarimThisWordAppearsIn = wordsToNumberOfSfarimTheyAppearIn[word]
        
        tanachOccurences = float(wordsToTanachCounts[word])
        
        
        amountShared = min([sfarimToWords[sefer][word] / np.log(sfarimToSizeOfSefer[sefer]) for sefer in listOfSfarim])
        
        ### Maybe weight/ multiply the idf by a scaling factor?
        idf = np.log(numberOfSfarimInTanach / numberOfSfarimThisWordAppearsIn)
        
#         score += amountShared/tanachOccurences
#         score += amountShared/numberOfSfarimThisWordAppearsIn
        score += amountShared * idf
        

    ### Normalize:
    
    ## Divide by the total words across all the sfarim, to penalize larger sfarim.
    totalWords = sum([sfarimToSizeOfSefer[sefer] for sefer in listOfSfarim])

    
    ### Multiply by number of sfarim so that the perfect score can be 1.
    score = score * numberOfSfarimInThisSet / totalWords
    
    return score
```


```
compareSfarimScoring(["Bamidbar", "Dvarim"])
```




    0.09219429204471698




```
### todo: 
### pinwheel visualization - pick the sfarim you want, shows score in middle
### Change weightings to bring in ngrams: right now we've done 1-word comparisons. add in 
### weightings of n-grams of 2, then n-grams of 3.
### Is this just a plagiarism detector? Let's code next time and see.
```


```
### All comparisons
sfarimNames = sfarimToWords.keys()
combos = itertools.combinations(sfarimNames, 2)
scores = defaultdict(dict)
for sefer1, sefer2 in combos:
    
#     print (sefer1, sefer2)
    score = compareSfarimScoring([sefer1, sefer2])
    
    scores[(sefer1, sefer2)] = score
    scores[(sefer2, sefer1)] = score   
```


```
### Highest scores:
topScores = sorted(scores.items(), key = lambda sefer: sefer[1], reverse = True)
print("Top scores:", topScores[:10:2])
print("Bottom scores:", topScores[-10::2])

```

    Top scores: [(('I Kings', 'II Chronicles'), 0.18812657649592282), (('Ezra', 'Nehemiah'), 0.18387993217610032), (('II Kings', 'II Chronicles'), 0.15923858097448412), (('I Samuel', 'II Samuel'), 0.15779185246722202), (('I Kings', 'II Kings'), 0.15640112036033604)]
    Bottom scores: [(('Obadiah', 'II Chronicles'), 0.00609234547648764), (('Shmot', 'Obadiah'), 0.005512926970922008), (('Bamidbar', 'Obadiah'), 0.0054090621022245145), (('Breishit', 'Obadiah'), 0.005329106693265538), (('Vayikra', 'Obadiah'), 0.004725239042564009)]



```
outputFileName = "sfarimComparisons.tsv"
with open(outputFileName, "w") as out:
    
    out.write("Sefer1\tSefer2\tScore\n")
    skipDuplicate = True ### Because we did symmetrical tuples)
    for (sefer1, sefer2), score in sorted(scores.items(), key = lambda sefer: sefer[1], reverse = True):
        
        if skipDuplicate:
            
            out.write("{}\t{}\t{}\n".format(sefer1, sefer2, score))
        skipDuplicate = not skipDuplicate
```

#### Histogram of scores


```
plt.figure()
plt.hist(scores.values(), bins = len(scores))
plt.xlabel("Score")
plt.ylabel("Number of Sfarim With This Score")
```




    Text(0, 0.5, 'Number of Sfarim With This Score')




    
![png](output_43_1.png)
    


#### Confirm no correlation with sefer size


```
### Get sizes per pair
sfarimPairsToSizes = dict()
for (sefer1, sefer2) in scores:
    
    sizeOfPair = sfarimToSizeOfSefer[sefer1] + sfarimToSizeOfSefer[sefer2]
    sfarimPairsToSizes[(sefer1, sefer2)] = sizeOfPair

```


```
assert(len(scores) == len(sfarimPairsToSizes))
```


```
sortedPairs = sorted(scores)
scoresSortedBySeferPair = [score for sfarimPair, score in sorted(scores.items())]
sizesSortedBySeferPair = [score for sfarimPair, score in sorted(sfarimPairsToSizes.items())]
```


```
plt.figure()
plt.scatter(sizesSortedBySeferPair, scoresSortedBySeferPair)
plt.xlabel("Size of Sefer-Pair")
plt.ylabel("Score")
plt.title("Correlation of Score and Sfarim Size")
plt.show()
plt.savefig("Normalized_ModelScoreSizeCorrelation.png")
x = True
```


    
![png](output_48_0.png)
    



    <Figure size 432x288 with 0 Axes>


# UMAP


```
def makeNgramMatrix(sfarimToWords, sfarimToSizeOfSefer, interactive = True):
    """
    Prep the UMAP input.

    Args:
         (str)

    Returns:
         (int)

    """    
    ### Get all words:
    allWordsInSelection = set()
    sfarim = sorted(sfarimToWords)
    for sefer, wordsCounter in sfarimToWords.items():

        seferWords = set(wordsCounter)

        allWordsInSelection |= seferWords

    allWordsInSelection = sorted(allWordsInSelection) # Turn into list and sort.
    numberWordsInSelection = len(allWordsInSelection)
    numberSfarim = len(sfarimToWords)

    print ("In the {} selected sections, there are {} unique words.".format(numberSfarim, numberWordsInSelection))
#     wordsMatrix = np.zeros((numberSfarim, numberWordsInSelection))

    ### Use sparse matrix:
    wordsMatrix = lil_matrix((numberSfarim, numberWordsInSelection))
    
    ### Fill in matrix:
    for seferNumber, (sefer, seferWordCounts) in enumerate(sorted(sfarimToWords.items())):

        ### raw frequency:
        counts = [seferWordCounts[word] for word in allWordsInSelection]

        ### special scoring:
        #     counts = [seferWordCounts[word] * np.lo for word in allWordsInSelection]

        wordsMatrix[seferNumber] = counts        

    wordsMatrix = csc_matrix(wordsMatrix)
    return wordsMatrix

def getUMAPembedding(wordsMatrix):
    """
    Actually running UMAP.
    Args:
        sfarimToWords ():

    Returns:
        The gmatria (int)

    """       
    embedding = umap.UMAP(n_components=2, metric='hellinger').fit(wordsMatrix)
    return embedding
#     print (embedding.embedding_.shape)

    ### Instead of perek names, just do Sefer names.
    ### This breaks if not doing byPerek.

def plotUMAP(embedding, sfarim, byPerek = False, interactive = True):
    """
    Plotting for UMAP.
    Args:
        sfarimToWords ():

    Returns:
        The gmatria (int)

    """   

    ### Pull out the sefer name - e.g. "Amos" from "Amos 1"
    if byPerek and not interactive:
        sfarim = [" ".join(sefer.split(" ")[:-1]) for sefer in sfarim]
        
        
    labels = pd.DataFrame(sfarim, columns=["Sefer"])
    print(labels)
    
    if interactive:
        f = umap.plot.interactive(embedding, labels=labels["Sefer"], hover_data=labels, point_size=10)
        show(f)
    else:
        f = umap.plot.points(embedding, labels=labels["Sefer"])
        
        # umap.plot.points(embedding, theme='fire')
        # umap.plot.connectivity(embedding, edge_bundling='hammer', show_points=True)
        

def runUMAP(numNgramsToRun, byPerek = False, interactive = True, removeExclusiveNgrams = False):
    """
    Handler function for UMAP.
    Args:
        sfarimToWords ():

    Returns:
        The gmatria (int)

    """

    createdMatrix = False
    for sizeOfNgram in range(1, numNgramsToRun + 1):

        print ("ngram size:", sizeOfNgram)
        sfarimToWords, sfarimToSizeOfSefer = getTanachNgramCountsBySefer(sizeOfNgram = sizeOfNgram, byPerek = byPerek, removeExclusiveNgrams = removeExclusiveNgrams)
        ngramMatrix = makeNgramMatrix(sfarimToWords, sfarimToSizeOfSefer)

        if not createdMatrix:
            fullMatrix = ngramMatrix
            createdMatrix = True

        if createdMatrix:
            fullMatrix = hstack([fullMatrix, ngramMatrix])
    
    print ("made matrix; now embedding")       
    sfarim = sorted(sfarimToSizeOfSefer.keys())
    print (sfarim)
    embedding = getUMAPembedding(fullMatrix)
    plotUMAP(embedding, sfarim, byPerek, interactive)
    
    
```


```
byPerek = True
sizeOfNgram = 2
interactive = True
# runUMAP(sfarimToWords, sfarimToSizeOfSefer, byPerek, interactive)

createdMatrix = False
numNgramsToRun = 1
for sizeOfNgram in range(1, numNgramsToRun + 1):
        
    print ("ngram size:", sizeOfNgram)
    sfarimToWords, sfarimToSizeOfSefer = getTanachNgramCountsBySefer(sizeOfNgram = sizeOfNgram, byPerek = byPerek, removeExclusiveNgrams = False)
    ngramMatrix = makeNgramMatrix(sfarimToWords, sfarimToSizeOfSefer)
    
    if not createdMatrix:
        fullMatrix = ngramMatrix
        createdMatrix = True
        
    if createdMatrix:
        fullMatrix = hstack([fullMatrix, ngramMatrix])
print ("made matrix; now embedding")       
embedding = getUMAPembedding(fullMatrix)
sfarim = sorted(sfarimToSizeOfSefer.keys())

```

    ngram size: 1
    In the 929 selected sections, there are 40000 words.
    made matrix; now embedding



```
runUMAP(numNgramsToRun = 1, byPerek = False, interactive = True, removeExclusiveNgrams = True)

```

    ngram size: 1
    Removing 24369 ngrams that only appear in one section.
    In the 39 selected sections, there are 15631 unique words.
    made matrix; now embedding
    ['Amos', 'Daniel', 'Deuteronomy', 'Ecclesiastes', 'Esther', 'Exodus', 'Ezekiel', 'Ezra', 'Genesis', 'Habakkuk', 'Haggai', 'Hosea', 'I Chronicles', 'I Kings', 'I Samuel', 'II Chronicles', 'II Kings', 'II Samuel', 'Isaiah', 'Jeremiah', 'Job', 'Joel', 'Jonah', 'Joshua', 'Judges', 'Lamentations', 'Leviticus', 'Malachi', 'Micah', 'Nahum', 'Nehemiah', 'Numbers', 'Obadiah', 'Proverbs', 'Psalms', 'Ruth', 'Song of Songs', 'Zechariah', 'Zephaniah']
                Sefer
    0            Amos
    1          Daniel
    2     Deuteronomy
    3    Ecclesiastes
    4          Esther
    5          Exodus
    6         Ezekiel
    7            Ezra
    8         Genesis
    9        Habakkuk
    10         Haggai
    11          Hosea
    12   I Chronicles
    13        I Kings
    14       I Samuel
    15  II Chronicles
    16       II Kings
    17      II Samuel
    18         Isaiah
    19       Jeremiah
    20            Job
    21           Joel
    22          Jonah
    23         Joshua
    24         Judges
    25   Lamentations
    26      Leviticus
    27        Malachi
    28          Micah
    29          Nahum
    30       Nehemiah
    31        Numbers
    32        Obadiah
    33       Proverbs
    34         Psalms
    35           Ruth
    36  Song of Songs
    37      Zechariah
    38      Zephaniah









<div class="bk-root" id="cfe95fad-53aa-4ae2-92cd-9bca30196d53" data-root-id="2620"></div>






```
plotUMAP(embedding, sfarim, byPerek, interactive = True)
```








<div class="bk-root" id="60a9868b-294e-4d8c-b960-8df089e663e3" data-root-id="1465"></div>






```
sfarimToWords, sfarimToSizeOfSefer = getAllWordCountsBySefer()#byPerek = True)
runUMAP(sfarimToWords, sfarimToSizozeOfSefer)#, interactive = False)
```


```
# paths = getPathsForSections(sections = ["Apocrypha", "Tanakh"])
# sfarimToWords, sfarimToSizeOfSefer = getWordCountsBySefer(paths, byPerek = False)
# runUMAP(sfarimToWords, sfarimToSizeOfSefer)#, interactive = False)
```

### Motzaot HaPeh
מוצאות הפה

The 5 Organs of Articulation.

Tnuyot Haotiyot
תנועות האותיות

(From [here](https://www.inner.org/gematria/5origins.php).)

Todo: vowels/non-matres lectiones


```
## Define the motzaot categories
motzaotToLetters = {}
motzaotToLetters["throat"] = {"א","ח", "ה", "ע"}
motzaotToLetters["palate"] = {"כ","י", "ג", "ק", "ך"}
motzaotToLetters["tongue"] = {"ר" ,"ס", "ש", "ז", "צ", "ץ"}
motzaotToLetters["teeth"] = {"נ" ,"ל", "ט", "ד", "ת", "ן"}
motzaotToLetters["lips"] = {"פ" ,"מ", "ו", "ב", "ם", "ף"}

lettersToMotzaot = {}
for organ, letters in motzaotToLetters.items():
    
    for letter in letters:
        lettersToMotzaot[letter] = organ
        
# motzaotToKey = {"throat": "t", "palate": "p", "tongue": "g", "teeth": "e", "lips": "l"}
```


```
def convertWordToMotzaot(word):
    
#     convertedWord = [motzaotToKey[lettersToMotzaot[letter]] for letter in word]
    convertedWord = [lettersToMotzaot[letter] for letter in word]
    ### Change from list to string:
    convertedWord = "".join(convertedWord)
    return convertedWord
```


```
def makeDictionariesFromSefer(filename, allWords, allLetters):
    
    problematicWords = set()
    
    with open(filename, "r") as psukim:
        for pasuk in psukim:

            cleanedPasuk = pasuk.strip().replace("־", " ").replace("[", "").replace("]", "")
            splitPasuk = cleanedPasuk.split(" ")

            # Skip the non-text lines.
            if len(splitPasuk) == 1 or splitPasuk[0] == "Chapter":
                continue

            for word in splitPasuk:
                try:
                    ### Change to motzaot:
                    convertedWord = convertWordToMotzaot(word)
                    allWords[convertedWord] += 1
                    
                    for letter in convertedWord.split("_"):
                        allLetters[letter] += 1                    
                except KeyError as e:
                    problematicWords.add(word)
    return allWords, allLetters


def makeNgramsDictionaryFromSefer(filename, allNgrams, sizeOfNgram):
    
    problematicWords = set()
    
    with open(filename, "r") as psukim:
        for pasuk in psukim:

            cleanedPasuk = pasuk.strip().replace("־", " ").replace("[", "").replace("]", "")
            splitPasuk = cleanedPasuk.split(" ")

            # Skip the non-text lines.
            if len(splitPasuk) == 1 or splitPasuk[0] == "Chapter":
                continue

            for word in splitPasuk:
                try:
                    
                    convertedWord = convertWordToMotzaot(word).split("_")
                    
                    for letterIndex in range(len(convertedWord) - sizeOfNgram + 1):
                        
    
                        ngram = "_".join(convertedWord[letterIndex:letterIndex + sizeOfNgram])
                        allNgrams[ngram] += 1
                except KeyError as e:
                    problematicWords.add(word)
    return allNgrams  

def runOnWholeTorah(seferFilenames):
    
    allWords = Counter()
    allLetters = Counter()
    allNgrams = Counter()
    
    for seferFilename in seferFilenames:
        
        seferName = seferFilename.replace(".txt", "").split("/")[-1]
        print(seferName)
        allWords, allLetters = makeDictionariesFromSefer(seferFilename, allWords, allLetters)
        allNgrams = makeNgramsDictionaryFromSefer(seferFilename, allNgrams, sizeOfNgram = 3)

    return allWords, allLetters, allNgrams

     
```


```
folder = "./texts/Torah"
seferFilenames = getFilePaths(folder)
allWords, allLetters, allNgrams = runOnWholeTorah(seferFilenames)

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-11-65d37c394bb4> in <module>
          1 folder = "./texts/Torah"
    ----> 2 seferFilenames = getFilePaths(folder)
          3 allWords, allLetters, allNgrams = runOnWholeTorah(seferFilenames)


    NameError: name 'getFilePaths' is not defined



```
allLetters.most_common(5)
```




    [('lips', 76863),
     ('throat', 73615),
     ('teeth', 62542),
     ('palate', 50390),
     ('tongue', 41753)]




```
allNgrams.most_common()[-10:]
```




    [('tongue_tongue_teeth', 274),
     ('lips_lips_lips', 256),
     ('tongue_teeth_teeth', 249),
     ('tongue_teeth_tongue', 244),
     ('teeth_tongue_tongue', 211),
     ('teeth_teeth_tongue', 154),
     ('throat_throat_throat', 146),
     ('palate_throat_throat', 125),
     ('palate_palate_palate', 78),
     ('tongue_tongue_tongue', 69)]



### Basic histogram/breakdowns:
    
#### Letters, pairs of letters


```
allWords
for 
```


```

```

# Talmud


```
## Load:
folder = "texts/Talmud"
masechetFilenames = getFilePaths(folder, hebrewOrEnglish = "English")
print ("There are {} masechtot.".format(len(masechetFilenames)))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-12-dba5de21f9c5> in <module>
          1 ## Load:
          2 folder = "texts/Talmud"
    ----> 3 masechetFilenames = getFilePaths(folder, hebrewOrEnglish = "English")
          4 print ("There are {} masechtot.".format(len(masechetFilenames)))


    NameError: name 'getFilePaths' is not defined



```
# yikes
def masechetFromFilename(masechetFilename):
    
    splitName = masechetFilename.split("/")
    nextIndex = splitName.index("English")
    masechetName = splitName[nextIndex - 1]
    
    return masechetName
```

Ran the spaCy code below to get all people in the Talmud. That had lots of false positives (e.g. "Leviticus 8:11", "chews ginger"), so took all lines with "Rav", "Rabb", and "The" in them, to get most of the true hits.
(Missing people like "Berurya" or "Ḥoni HaMe’aggel" with this, though.)


```
rabbis = set([person.strip() for person in open("RavRabbThe.txt", "r")])
print("Working with {} rabbis from the Talmud.".format(len(rabbis)))

```

    Working with 1666 rabbis from the Talmud.



```
nlp = spacy.load('en_core_web_sm')
filename = "texts/Talmud/Berakhot/English/merged.txt"
masechetName = masechetFromFilename(filename)
print ("\r{}\r".format(masechetName))
reader = open(filename, "r")
lineNumber = 0
for line in reader:

    ### Skip the header info, and the whitespace/daf numbers
    lineNumber += 1
    if lineNumber < 21 or len(line) < 13:
        continue
    line = re.sub('<[^<]+?>', '', line) # ayy https://stackoverflow.com/a/4869782

    ### Incredible https://spacy.io/usage/linguistic-features#named-entities
    doc = nlp(line)            
    print (line)
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
#     for chunk in doc.noun_chunks:
#         print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
            
#     print (doc.ents)
    raise
```

    Berakhot
    The beginning of tractate Berakhot, the first tractate in the first of the six orders of Mishna, opens with a discussion of the recitation of Shema, as the recitation of Shema encompasses an acceptance of the yoke of Heaven and of the mitzvot, and as such, forms the basis for all subsequent teachings. The Mishna opens with the laws regarding the appropriate time to recite Shema: MISHNA: From when, that is, from what time, does one recite Shema in the evening? From the time when the priests enter to partake of their teruma. Until when does the time for the recitation of the evening Shema extend? Until the end of the first watch. The term used in the Torah (Deuteronomy 6:7) to indicate the time for the recitation of the evening Shema is beshokhbekha, when you lie down, which refers to the time in which individuals go to sleep. Therefore, the time for the recitation of Shema is the first portion of the night, when individuals typically prepare for sleep. That is the statement of Rabbi Eliezer.
    
    The the DET DT det Xxx True True
    beginning beginning NOUN NN nsubj xxxx True False
    of of ADP IN prep xx True True
    tractate tractate NOUN NN compound xxxx True False
    Berakhot Berakhot PROPN NNP pobj Xxxxx True False
    , , PUNCT , punct , False False
    the the DET DT det xxx True True
    first first ADJ JJ amod xxxx True True
    tractate tractate NOUN NN appos xxxx True False
    in in ADP IN prep xx True True
    the the DET DT det xxx True True
    first first ADJ JJ pobj xxxx True True
    of of ADP IN prep xx True True
    the the DET DT det xxx True True
    six six NUM CD nummod xxx True True
    orders order NOUN NNS pobj xxxx True False
    of of ADP IN prep xx True True
    Mishna Mishna PROPN NNPS pobj Xxxxx True False
    , , PUNCT , punct , False False
    opens open VERB VBZ ROOT xxxx True False
    with with ADP IN prep xxxx True True
    a a DET DT det x True True
    discussion discussion NOUN NN pobj xxxx True False
    of of ADP IN prep xx True True
    the the DET DT det xxx True True
    recitation recitation NOUN NN pobj xxxx True False
    of of ADP IN prep xx True True
    Shema Shema PROPN NNP pobj Xxxxx True False
    , , PUNCT , punct , False False
    as as SCONJ IN mark xx True True
    the the DET DT det xxx True True
    recitation recitation NOUN NN nsubj xxxx True False
    of of ADP IN prep xx True True
    Shema Shema PROPN NNP pobj Xxxxx True False
    encompasses encompass VERB VBZ advcl xxxx True False
    an an DET DT det xx True True
    acceptance acceptance NOUN NN dobj xxxx True False
    of of ADP IN prep xx True True
    the the DET DT det xxx True True
    yoke yoke NOUN NN pobj xxxx True False
    of of ADP IN prep xx True True
    Heaven Heaven PROPN NNP pobj Xxxxx True False
    and and CCONJ CC cc xxx True True
    of of ADP IN conj xx True True
    the the DET DT det xxx True True
    mitzvot mitzvot NOUN NN pobj xxxx True False
    , , PUNCT , punct , False False
    and and CCONJ CC cc xxx True True
    as as SCONJ IN prep xx True True
    such such ADJ JJ amod xxxx True True
    , , PUNCT , punct , False False
    forms form VERB VBZ conj xxxx True False
    the the DET DT det xxx True True
    basis basis NOUN NN dobj xxxx True False
    for for ADP IN prep xxx True True
    all all DET DT det xxx True True
    subsequent subsequent ADJ JJ amod xxxx True False
    teachings teaching NOUN NNS pobj xxxx True False
    . . PUNCT . punct . False False
    The the DET DT det Xxx True True
    Mishna Mishna PROPN NNPS nsubj Xxxxx True False
    opens open VERB VBZ ROOT xxxx True False
    with with ADP IN prep xxxx True True
    the the DET DT det xxx True True
    laws law NOUN NNS pobj xxxx True False
    regarding regard VERB VBG prep xxxx True True
    the the DET DT det xxx True True
    appropriate appropriate ADJ JJ amod xxxx True False
    time time NOUN NN pobj xxxx True False
    to to PART TO aux xx True True
    recite recite VERB VB relcl xxxx True False
    Shema Shema PROPN NNP dobj Xxxxx True False
    : : PUNCT : punct : False False
    MISHNA MISHNA PROPN NNP ROOT XXXX True False
    : : PUNCT : punct : False False
    From from ADP IN prep Xxxx True True
    when when ADV WRB pobj xxxx True True
    , , PUNCT , punct , False False
    that that ADV RB advmod xxxx True True
    is is ADV RB advmod xx True True
    , , PUNCT , punct , False False
    from from ADP IN prep xxxx True True
    what what PRON WP det xxxx True True
    time time NOUN NN pcomp xxxx True False
    , , PUNCT , punct , False False
    does do AUX VBZ aux xxxx True True
    one one NUM CD nsubj xxx True True
    recite recite VERB VB ROOT xxxx True False
    Shema Shema PROPN NNP dobj Xxxxx True False
    in in ADP IN prep xx True True
    the the DET DT det xxx True True
    evening evening NOUN NN pobj xxxx True False
    ? ? PUNCT . punct ? False False
    From from ADP IN ROOT Xxxx True True
    the the DET DT det xxx True True
    time time NOUN NN pobj xxxx True False
    when when ADV WRB advmod xxxx True True
    the the DET DT det xxx True True
    priests priest NOUN NNS nsubj xxxx True False
    enter enter VERB VBP relcl xxxx True False
    to to PART TO prep xx True True
    partake partake VERB VB pobj xxxx True False
    of of ADP IN prep xx True True
    their -PRON- DET PRP$ poss xxxx True True
    teruma teruma NOUN NN pobj xxxx True False
    . . PUNCT . punct . False False
    Until until ADP IN ROOT Xxxxx True True
    when when ADV WRB pcomp xxxx True True
    does do AUX VBZ pobj xxxx True True
    the the DET DT det xxx True True
    time time NOUN NN nsubj xxxx True False
    for for ADP IN prep xxx True True
    the the DET DT det xxx True True
    recitation recitation NOUN NN pobj xxxx True False
    of of ADP IN prep xx True True
    the the DET DT det xxx True True
    evening evening NOUN NN compound xxxx True False
    Shema Shema PROPN NNP compound Xxxxx True False
    extend extend VERB VB pobj xxxx True False
    ? ? PUNCT . punct ? False False
    Until until ADP IN ROOT Xxxxx True True
    the the DET DT det xxx True True
    end end NOUN NN pobj xxx True False
    of of ADP IN prep xx True True
    the the DET DT det xxx True True
    first first ADJ JJ amod xxxx True True
    watch watch NOUN NN pobj xxxx True False
    . . PUNCT . punct . False False
    The the DET DT det Xxx True True
    term term NOUN NN nsubj xxxx True False
    used use VERB VBN acl xxxx True True
    in in ADP IN prep xx True True
    the the DET DT det xxx True True
    Torah Torah PROPN NNP pobj Xxxxx True False
    ( ( PUNCT -LRB- punct ( False False
    Deuteronomy Deuteronomy PROPN NNP appos Xxxxx True False
    6:7 6:7 NUM CD nummod d:d False False
    ) ) PUNCT -RRB- punct ) False False
    to to PART TO aux xx True True
    indicate indicate VERB VB relcl xxxx True False
    the the DET DT det xxx True True
    time time NOUN NN dobj xxxx True False
    for for ADP IN prep xxx True True
    the the DET DT det xxx True True
    recitation recitation NOUN NN pobj xxxx True False
    of of ADP IN prep xx True True
    the the DET DT det xxx True True
    evening evening NOUN NN compound xxxx True False
    Shema Shema PROPN NNP pobj Xxxxx True False
    is be AUX VBZ ROOT xx True True
    beshokhbekha beshokhbekha PROPN NNP attr xxxx True False
    , , PUNCT , punct , False False
    when when ADV WRB advmod xxxx True True
    you -PRON- PRON PRP nsubj xxx True True
    lie lie VERB VBP advcl xxx True False
    down down ADV RB prt xxxx True True
    , , PUNCT , punct , False False
    which which DET WDT nsubj xxxx True True
    refers refer VERB VBZ advcl xxxx True False
    to to ADP IN prep xx True True
    the the DET DT det xxx True True
    time time NOUN NN pobj xxxx True False
    in in ADP IN prep xx True True
    which which DET WDT pobj xxxx True True
    individuals individual NOUN NNS nsubj xxxx True False
    go go VERB VBP relcl xx True True
    to to ADP IN prep xx True True
    sleep sleep VERB VB pobj xxxx True False
    . . PUNCT . punct . False False
    Therefore therefore ADV RB advmod Xxxxx True True
    , , PUNCT , punct , False False
    the the DET DT det xxx True True
    time time NOUN NN nsubj xxxx True False
    for for ADP IN prep xxx True True
    the the DET DT det xxx True True
    recitation recitation NOUN NN pobj xxxx True False
    of of ADP IN prep xx True True
    Shema Shema PROPN NNP pobj Xxxxx True False
    is be AUX VBZ ROOT xx True True
    the the DET DT det xxx True True
    first first ADJ JJ amod xxxx True True
    portion portion NOUN NN attr xxxx True False
    of of ADP IN prep xx True True
    the the DET DT det xxx True True
    night night NOUN NN pobj xxxx True False
    , , PUNCT , punct , False False
    when when ADV WRB advmod xxxx True True
    individuals individual NOUN NNS nsubj xxxx True False
    typically typically ADV RB advmod xxxx True False
    prepare prepare VERB VBP relcl xxxx True False
    for for ADP IN prep xxx True True
    sleep sleep NOUN NN pobj xxxx True False
    . . PUNCT . punct . False False
    That that DET DT nsubj Xxxx True True
    is be AUX VBZ ROOT xx True True
    the the DET DT det xxx True True
    statement statement NOUN NN attr xxxx True False
    of of ADP IN prep xx True True
    Rabbi Rabbi PROPN NNP compound Xxxxx True False
    Eliezer Eliezer PROPN NNP pobj Xxxxx True False
    . . PUNCT . punct . False False
    
     
     SPACE _SP  
     False False
    first 40 45 ORDINAL
    first 62 67 ORDINAL
    six 75 78 CARDINAL
    Mishna 89 95 GPE
    Shema 142 147 PERSON
    Shema 170 175 GPE
    Mishna 307 313 PERSON
    Shema 375 380 PERSON
    MISHNA 382 388 PERSON
    Shema 442 447 PERSON
    the evening 451 462 TIME
    evening 580 587 TIME
    Shema 588 593 PERSON
    first 623 628 ORDINAL
    Torah 657 662 GPE
    Shema 736 741 GPE
    Shema 879 884 GPE
    first 892 897 ORDINAL
    the night 909 918 TIME
    Rabbi Eliezer 991 1004 PERSON



    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-16-7bf32f9aa521> in <module>
         24 
         25 #     print (doc.ents)
    ---> 26     raise
    

    RuntimeError: No active exception to reraise



```

```


```
nlp = spacy.load('en_core_web_sm')
lineNumber = 0
people = set()
rabbisToMasechet = defaultdict(Counter)
masechetToRabbis = defaultdict(Counter)

for filename in masechetFilenames:
    
    masechetName = masechetFromFilename(filename)
    print ("\r{}\r".format(masechetName))
    with open(filename, "r") as reader:
        for line in reader:

            ### Skip the header info, and the whitespace/daf numbers
            lineNumber += 1
            if lineNumber < 21 or len(line) < 13:
                continue
            line = re.sub('<[^<]+?>', '', line) # ayy https://stackoverflow.com/a/4869782

            ### Incredible https://spacy.io/usage/linguistic-features#named-entities
            doc = nlp(line)            
            for entity in doc.ents:

                ### First pass: get all the people (lots of false positives):
    #             if entity.label_ == "PERSON" and entity.text in rabbis:
    #                 people.add(entity)

                ### Next round: use that to make an edited list:
                if entity.label_ == "PERSON" and entity.text in rabbis:
                    rabbisToMasechet[entity.text][masechetName] += 1
                    masechetToRabbis[masechetName][entity.text] += 1
```

    Niddah
    Chagigah
    Yoma
    Rosh Hashanah
    Moed Katan
    Beitzah
    Sukkah
    Megillah
    Taanit
    Pesachim
    Shabbat
    Eruvin
    Berakhot
    Horayot
    Avodah Zarah
    Bava Batra
    Shevuot
    Sanhedrin
    Bava Metzia
    Bava Kamma
    Makkot
    Sotah
    Kiddushin
    Nazir
    Yevamot
    Gittin
    Ketubot
    Nedarim
    Zevachim
    Meilah
    Bekhorot
    Temurah
    Keritot
    Menachot
    Arakhin
    Chullin
    Tamid



```
### First round: save all the spaCy results.

# peopleToText = {name.text for name in people}
# with open("talmudPeople.txt", "w") as out:
#     out.write("\n".join(peopleToText))
```


```
### Second round: save the more accurate counts.

sortedRabbis = sorted(rabbis)
with open("RabbiCounts.tsv", "w") as out:
    
    masechtot = sorted(masechetToRabbis)
    out.write("\t{}\n".format("\t".join(masechtot)))
    
    for rabbi in sortedRabbis:
        
        rabbiCounts = rabbisToMasechet[rabbi]
        sortedCounts = [str(rabbiCounts[masechet]) for masechet in masechtot]
        
        out.write("{r}\t{c}\n".format(r = rabbi, c = "\t".join(sortedCounts)))
```

### Exact Matching (without spaCy)

Alternative (quicker) way that doesn't use spaCy, so it doesn't do true entity recognition and just searches for any matches. Runs instantly, but counts "Rav" a bunch because it counts "Rav x" for Rav as well.


```
# rabbisToMasechet = defaultdict(Counter)
# masechetToRabbis = defaultdict(Counter)

# for filename in masechetFilenames:
        
#     masechetName = masechetFromFilename(filename)
#     print ("\r{}\r".format(masechetName))
#     with open(filename, "r") as masechet:
        
#         text = masechet.read()
#         for rabbi in rabbis:
#             appearances = text.count(rabbi)
            
#             if appearances != 0:
#                 rabbisToMasechet[rabbi][masechetName] += appearances
#                 masechetToRabbis[masechetName][rabbi] += appearances
                            
```


```
### Big to print
# print(rabbisToMasechet)
# print(masechetToRabbis)
```

Exploratory:


```
def formattedCounts(counts):
    
    formatted = "\t" + "; ".join(["{}: {}".format(value, count) for value, count in counts])
    return formatted
    
rabbi = "Rabbi Yehuda"
masechet = "Berakhot"
### Most common:

print("{}'s most common:\n".format(rabbi), formattedCounts(rabbisToMasechet[rabbi].most_common(5)))
print("{}'s most common:\n".format(masechet), formattedCounts(masechetToRabbis[masechet].most_common(5)))
print()

### Least common:

print("{}'s least common:\n".format(rabbi), formattedCounts(rabbisToMasechet[rabbi].most_common()[-5:]))
print("{}'s least common:\n".format(masechet), formattedCounts(masechetToRabbis[masechet].most_common()[-10:]))
```

    Rabbi Yehuda's most common:
     	Shabbat: 446; Pesachim: 409; Chullin: 406; Eruvin: 395; Menachot: 322
    Berakhot's most common:
     	Rabbi Yoḥanan: 292; Rabbi Yehuda: 222; Rav: 219; Rava: 122; Rabbi Yosei: 110
    
    Rabbi Yehuda's least common:
     	Horayot: 48; Rosh Hashanah: 40; Chagigah: 37; Meilah: 16; Tamid: 2
    Berakhot's least common:
     	Rav Sama: 1; Rabbi Yosei bar Yehuda: 1; Rabbi Parnakh: 1; Rav Huna bar Berekhya: 1; Rabbi Elazar HaKappar: 1; Rabbi Yosei ben Keifar: 1; Zekharya ben Kevutal: 1; Zekharya ben: 1; the Sages of the Mishna: 1; Rav Yehuda bar Zevida: 1


Full counts across the whole Talmud:


```
rabbisFullCounts = Counter()

for rabbi, masechetCounts in rabbisToMasechet.items():
    
    masechetTotalCounts = sum(masechetCounts.values())
    rabbisFullCounts[rabbi] += masechetTotalCounts
```


```
rabbisFullCounts.most_common(30)
```




    [('Rabbi Yoḥanan', 6473),
     ('Rabbi Yehuda', 6335),
     ('Rav', 5819),
     ('Rabbi Shimon', 3915),
     ('Rabbi Meir', 3756),
     ('Rabbi Yosei', 3589),
     ('Rabbis', 3552),
     ('Rava', 3507),
     ('Rabbi Eliezer', 3123),
     ('Rabbi Yehuda HaNasi', 3079),
     ('Rabbi Akiva', 2926),
     ('Rabbi Elazar', 2712),
     ('Rav Huna', 2481),
     ('Rav Yehuda', 2432),
     ('Rav Ashi', 2241),
     ('Rav Pappa', 2027),
     ('Rav Yosef', 1965),
     ('Rav Ḥisda', 1742),
     ('Rabba', 1664),
     ('Rabbi Yishmael', 1583),
     ('Rabbi Ḥiyya', 1421),
     ('Rabbi Yehoshua', 1344),
     ('Rabbi Ḥanina', 1171),
     ('Rabbi Zeira', 1148),
     ('Rav Sheshet', 1029),
     ('Rabban Gamliel', 848),
     ('Rav Kahana', 764),
     ('Shimon ben Gamliel', 717),
     ('Ravina', 699),
     ('Rabbi Yehoshua ben Levi', 685)]



## Word Themes
### 2021-1-20

Hebrew verbs from [here](https://github.com/NLPH/NLPH_Resources/blob/master/linguistic_resources/word_lists/hebrew_verbs_eran_tomer/TheVerbIndex.csv) and dictionary from [here](https://github.com/gregarkhipov/milon/blob/gh-pages/dict-he-en.json)


```
import csv
import json
import unicodedata
```


```
### load roots:
roots = set()
rootsFile = "hebrewVerbs.csv"
reader = csv.DictReader(open(rootsFile, encoding='utf-8-sig'))
for line in reader:
    if "base_form" not in line:
        print (line)
    roots.add(line["base_form"])
```


```
print ("We have {} roots.".format(len(roots)))
```

    We have 3530 roots.



```
### load dictionary:

hebrewToEnglishDictionary = defaultdict(set)#{}
json_file = "hebrewEnglishDictionary.json"
with open(json_file) as f:
     json_obj = json.load(f)
for word in json_obj:
    hebrew = word["translated"]#.encode('UTF-8')
    english = word["translation"]
    ### Deal with nekudot: https://gist.github.com/yakovsh/345a71d841871cc3d375#gistcomment-1672181
    noNekkudot = unicodedata.normalize('NFKD', hebrew)
    noNekkudot = "".join([c for c in noNekkudot if not unicodedata.combining(c)])
    hebrewToEnglishDictionary[noNekkudot].add(english[0])
```


```
threeLetterWordsWithHeh = {word for word in hebrewToEnglishDictionary if len(word) == 3 and word[1] == "ז" and word[2] == "ר"}
print ("There are {} 3-letter words with hehs in the middle.".format(len(threeLetterWordsWithHeh)))

```

    There are 0 3-letter words with hehs in the middle.



```
for word in threeLetterWordsWithHeh:
    print (word, hebrewToEnglishDictionary[word])
    
```

    שזר ['to weave', 'to bundle', 'to arrange', 'to integrate']
    בזר ['to decentralize', 'to disperse']
    גזר ['(grammar) to derive']
    אזר ['(Biblical) to gird']
    עזר ['aid', 'assistance', 'aide', 'assistant', 'עזרים - aids', 'accessories']
    פזר ['biblical cantillation symbol']
    חזר ['to court', 'to woo']
    נזר ['to deprive oneself (of something)']



```
# convertWordToMotzaot
# for word, translation in hebrewToEnglishDictionary.items():
#     try:
#         if convertWordToMotzaot(word) == "tonguelipsteeth":
#             print (word, translation)
#     except KeyError as e:
#         continue
```


```
nlp = spacy.load("en_core_web_md")
tokens = nlp("dog cat banana weave disperse")

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
```

    dog True 7.0336733 False
    cat True 6.6808186 False
    banana True 6.700014 False
    weave True 6.4481254 False
    disperse True 6.4651346 False



```
# doc1 = nlp("I like salty fries and hamburgers.")
# doc2 = nlp("Fast food tastes very good.")

# Similarity of two documents
# print(doc1, "<->", doc2, doc1.similarity(doc2))
# Similarity of tokens and spans
french_fries = nlp("desk")
burgers =nlp("furniture")
print(french_fries, "<->", burgers, french_fries.similarity(burgers))
```

    desk <-> furniture 0.5216913205865407


## Words and Roots
#### Querying Sefaria API


```
import requests
import time
import csv
from IPython.display import display, clear_output
import spacy
import statistics
```


```
sfarimToWords, sfarimToSizeOfSefer = getTanachWordCountsBySefer(byPerek = False, keep_nekudot = True)
```

    Removing 86762 ngrams that only appear in one section.



```
# hebrewToEnglishDictionary
have = 0
nope = 0
for sefer, words in sfarimToWords.items():
    print (sefer, len(words))
    for word, count in words.items():#.most_common(20):
        if word in hebrewToEnglishDictionary:
            print (word, hebrewToEnglishDictionary[word])
            have += 1
            raise
        else:
            print ("No translation for", word)
            nope += 1
    raise
    print ("In {}, translations for {} words but missing {}.".format(sefer, have, nope))

```

    Numbers 4504



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-11-20cf883be7d3> in <module>
          5     print (sefer, len(words))
          6     for word, count in words.items():#.most_common(20):
    ----> 7         if word in hebrewToEnglishDictionary:
          8             print (word, hebrewToEnglishDictionary[word])
          9             have += 1


    NameError: name 'hebrewToEnglishDictionary' is not defined



```
response = requests.get("http://www.sefaria.org/api/words/בְּרֵאשִׁ֖ית")
print(response)
```

    <Response [200]>



```
response.json()
```




    [{'headword': 'רֵאשִׁית',
      'parent_lexicon': 'BDB Augmented Strong',
      'content': {'morphology': 'n-f',
       'senses': [{'definition': 'first,  beginning,  best,  chief',
         'senses': [{'definition': 'beginning'},
          {'definition': 'first'},
          {'definition': 'chief'},
          {'definition': 'choice part'}]}]},
      'strong_number': '7225',
      'transliteration': 'rêʼshîyth',
      'pronunciation': "ray-sheeth'",
      'language_code': 'heb',
      'parent_lexicon_details': {'name': 'BDB Augmented Strong',
       'language': 'heb.biblical',
       'to_language': 'eng',
       'text_categories': ['Tanakh, Torah',
        'Tanakh, Prophets',
        'Tanakh, Writings'],
       'source': 'Open Scriptures on GitHub',
       'source_url': 'https://github.com/openscriptures/strongs',
       'attribution': 'Based on the work of Larry Pierce at the Online Bible',
       'attribution_url': 'http://www.blueletterbible.org/contact/contact_gen.cfm'}},
     {'headword': 'בְּרֵאשִׁית',
      'parent_lexicon': 'Jastrow Dictionary',
      'rid': 'B01162',
      'refs': ['Targum Jonathan on Isaiah 41:4',
       'Bereishit Rabbah 3:5',
       'Bereishit Rabbah 3:6',
       'Bereishit Rabbah 3:7',
       'Bereishit Rabbah 3:8',
       'Bereishit Rabbah 3:9',
       'Ketubot 8b:8',
       'Eichah Rabbah 3:2',
       'Eichah Rabbah 3:8',
       'Eichah Rabbah 3:13',
       'Eichah Rabbah 3:14',
       'Eichah Rabbah 3:22',
       'Bereishit Rabbah 3:5',
       'Bereishit Rabbah 3:6',
       'Bereishit Rabbah 3:7',
       'Bereishit Rabbah 3:8',
       'Bereishit Rabbah 3:9'],
      'content': {'senses': [{'definition': ' (b. h.) <i>in the beginning</i>, as a cosmological term (ref. to <a class="refLink" href="/Genesis.1.1" data-ref="Genesis 1:1">Gen. I, 1</a>) <i>creation, primeval period, Nature, Universe</i>. <a class="refLink" href="/Targum_Jonathan_on_Isaiah.28.29" data-ref="Targum Jonathan on Isaiah 28:29">Targ. Is. XXVIII, 29</a>; a. e.—מִבְּ׳ <i>from the beginning</i>. <a class="refLink" href="/Targum_Jonathan_on_Isaiah.41.4" data-ref="Targum Jonathan on Isaiah 41:4">Ib. XLI, 4</a>.—<span dir="rtl">מעשה ב׳</span> a) <i>creation</i>. <a class="refLink" href="/Bereishit_Rabbah.3.9" data-ref="Bereishit Rabbah 3:9">Gen. R. s. 3</a>; a. fr.—b) <i>cosmogony</i>, contrad. to <span dir="rtl">מעשה מרכבה</span> theosophy, <a class="refLink" href="/Mishnah_Chagigah.2.1" data-ref="Mishnah Chagigah 2:1">Ḥag. II, 1</a>; a. fr.—Y. Shebi. I, beg. 33ᵃ <span dir="rtl">שבת ב׳</span> the Sabbath commemorative of creation, i.e. the regular weekly Sabbath, contrad. to Holy Days. [In later Hebr. <span dir="rtl">שבת ב׳</span> the Sabbath on which the first section of the Pentateuch is read.]—<a class="refLink" href="/Mishnah_Berakhot.9.2" data-ref="Mishnah Berakhot 9:2">Ber. IX, 2</a> <span dir="rtl"><span dir="rtl">ברוך עושה</span> (<a dir="rtl" class="refLink" href="/Jastrow,_מַעֲשֶׂה.1" data-ref="Jastrow, מַעֲשֶׂה 1">מעשה</a>) <a dir="rtl" class="refLink" href="/Jastrow,_ב׳.1" data-ref="Jastrow, ב׳ 1">ב׳</a></span> praised be the Author of creation—a formula of benediction for awe-inspiring natural phenomena; v. ib. a. Y. ib. 13ᶜ bot.—<span dir="rtl">ששת ימי ב׳</span> from the six days of creation. <a class="refLink" href="/Ketubot.8b.8" data-ref="Ketubot 8b:8">Keth. 8ᵇ</a> <span dir="rtl">נתיב הוא מש׳ וכ׳</span> this is the way (the lot of humanity) since the world existed.—<a class="refLink" href="/Tosefta_Ma\'asrot.3.14" data-ref="Tosefta Ma\'asrot 3:14">Tosef. Maasr. III, 14</a>: a. fr.—Y. Taan. II, 65ᵃ bot. <span dir="rtl">מי ב׳</span>; <a class="refLink" href="/Eichah_Rabbah.3.22" data-ref="Eichah Rabbah 3:22">Lam. R. to III, 40</a> <span dir="rtl">מימי ב׳</span> primeval waters, <i>Ocean</i> &c. (v. <a class="refLink" href="/Genesis.1.9" data-ref="Genesis 1:9">Gen. I, 9</a> sq.).—ספר ב׳ <i>The Book of Genesis</i>. <a class="refLink" href="/Bereishit_Rabbah.3.9" data-ref="Bereishit Rabbah 3:9">Gen. R. s. 3</a>; a. e.—ב׳ רבה <i>B’reshith Rabbah</i> (Gen. R.), name of the first book of the Midrash Rabbah.'}]},
      'plural_form': [],
      'alt_headwords': [],
      'quotes': [[None, 'בְּרֵאשִׁית רבה', None]],
      'prev_hw': 'בְּרָאִי',
      'next_hw': 'ברבוי',
      'parent_lexicon_details': {'name': 'Jastrow Dictionary',
       'language': 'heb.talmudic',
       'to_language': 'eng',
       'text_categories': [],
       'source': 'Jastrow Dictionary',
       'attribution': 'Rabbi Marcus Jastrow',
       'index_title': 'Jastrow',
       'version_title': 'London, Luzac, 1903',
       'version_lang': 'en'}},
     {'headword': 'בְּרֵאשִׁית ᴵ',
      'parent_lexicon': 'Klein Dictionary',
      'content': {'morphology': 'adv.',
       'senses': [{'definition': 'Genesis — name of the first book of the Pentateuch.',
         'number': '1',
         'language_code': 'PBH'},
        {'definition': 'In the beginning.', 'number': '2'}]},
      'rid': 'B00952',
      'refs': ['Klein Dictionary, רֵאשִׁית 1',
       'Klein Dictionary, בְּ◌ 1',
       'Klein Dictionary, רֵאשִׁית 1',
       'Klein Dictionary, בְּ◌ 1'],
      'prev_hw': 'ברא ᴵᴵᴵ',
      'next_hw': 'בְּרֵאשִׁית ᴵᴵ',
      'notes': '[Formed from <a dir="rtl" class="refLink" href="/Klein Dictionary,_רֵאשִׁית.1" data-ref="Klein Dictionary, רֵאשִׁית 1">רֵאשִׁית</a> (= beginning) with pref. <a dir="rtl" class="refLink" href="/Klein Dictionary,_בְּ◌.1" data-ref="Klein Dictionary, בְּ◌ 1">בְּ◌</a>.]',
      'parent_lexicon_details': {'name': 'Klein Dictionary',
       'language': 'heb.modern',
       'to_language': 'eng',
       'text_categories': [],
       'source': 'Klein Dictionary',
       'attribution': 'Ezra Klein',
       'index_title': 'Klein Dictionary',
       'version_title': 'Carta Jerusalem; 1st edition, 1987',
       'version_lang': 'en'}},
     {'headword': 'בְּרֵאשִׁית ᴵᴵ',
      'parent_lexicon': 'Klein Dictionary',
      'content': {'morphology': 'm.n.',
       'senses': [{'definition': 'primeval period, creation.'}]},
      'rid': 'B00953',
      'language_code': 'NH',
      'refs': ['Klein Dictionary, בְּרֵאשִׁית ᴵ 1',
       'Klein Dictionary, בְּרֵאשִׁית ᴵ 1'],
      'prev_hw': 'בְּרֵאשִׁית ᴵ',
      'next_hw': 'בְּרֵאשִׁיתִי',
      'notes': '[Nominalization of the adv. <a dir="rtl" class="refLink" href="/Klein Dictionary,_בְּרֵאשִׁית ᴵ.1" data-ref="Klein Dictionary, בְּרֵאשִׁית ᴵ 1">בְּרֵאשִׁית</a> (= in the beginning).]',
      'parent_lexicon_details': {'name': 'Klein Dictionary',
       'language': 'heb.modern',
       'to_language': 'eng',
       'text_categories': [],
       'source': 'Klein Dictionary',
       'attribution': 'Ezra Klein',
       'index_title': 'Klein Dictionary',
       'version_title': 'Carta Jerusalem; 1st edition, 1987',
       'version_lang': 'en'}}]




```
for result in response.json():
#     print ()
    print (result["content"]["senses"][0]["definition"])
    
```

    to go in,  enter,  come,  go,  come in



```
result["content"]["senses"]
```




    [{'definition': 'father of an individual'},
     {'definition': 'of God as father of his people'},
     {'definition': 'head or founder of a household,  group,  family,  or clan'},
     {'definition': 'ancestor',
      'senses': [{'definition': 'grandfather,  forefathers — of person'},
       {'definition': 'of people'}]},
     {'definition': 'originator or patron of a class,  profession,  or art'},
     {'definition': 'of producer,  generator (fig.)'},
     {'definition': 'of benevolence and protection (fig.)'},
     {'definition': 'term of respect and honour'},
     {'definition': 'ruler or chief (spec.)'}]




```
# https://github.com/Emmeth/tools/blob/master/scripts/remove_nikkud.py
def strip_trop_not_nekudot(hebrew_text):
    return re.sub(re.compile(r'[\u0591-\u05AF]'), "", hebrew_text)

def strip_nekudot(hebrew_text):
    normalized = unicodedata.normalize('NFKD', hebrew_text)
    cleanedPasuk = "".join([c for c in normalized if not unicodedata.combining(c)])
    return cleanedPasuk
```


```
all_words = Counter()
for sefer, words_counter in sfarimToWords.items():
    all_words += words_counter
# Strip trop:
all_words = set([strip_trop_not_nekudot(word) for word in all_words if word])
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-47-0bda5d709965> in <module>
          1 all_words = Counter()
    ----> 2 for sefer, words_counter in sfarimToWords.items():
          3     all_words += words_counter
          4 # Strip trop:
          5 all_words = set([strip_trop_not_nekudot(word) for word in all_words if word])


    NameError: name 'sfarimToWords' is not defined



```
verbs_filename = "tanach_verbs_roots_and_translations.tsv"
non_verbs_filename = "tanach_non_verbs_roots_and_translations.tsv"
no_translation_filename = "tanach_no_sefaria_translation.tsv"
verb_writer = csv.writer(open(verbs_filename, "w"), delimiter = "\t")
non_verb_writer = csv.writer(open(non_verbs_filename, "w"), delimiter = "\t")
no_translation_writer = csv.writer(open(no_translation_filename, "w"), delimiter = "\t")
verb_writer.writerow(["Word", "Shoresh", "Morphology", "First Translation", "All Translations"])
non_verb_writer.writerow(["Word", "Shoresh", "Morphology", "First Translation", "All Translations"])
no_translation_writer.writerow(["Word"])
words = ["וּבָ֥אוּ", "הַמִּזְבֵּֽחַ", "וַיְדַבֵּ֖ר"]
# for word in words:
for word_number, word in enumerate(all_words):
    clear_output(wait=True)
    print(word_number, word)
    response = requests.get("http://www.sefaria.org/api/words/{}".format(word))
    if not response.json(): # No translation found
        no_translation_writer.writerow([word])
        continue
    result = response.json()[0] # Take the first source.
    shoresh = result["headword"]
    try:
        first_translation = result["content"]["senses"][0]["definition"]
    except IndexError as e:
        first_translation = "No translation"
    all_translations = str(result["content"]["senses"])
    try:
        morphology = result["content"]["morphology"]
    except KeyError as e:
        morphology = "No morphology found"
    is_verb = morphology == "v"
    if is_verb:
        verb_writer.writerow([word, shoresh, morphology, first_translation, all_translations])
    else:
        non_verb_writer.writerow([word, shoresh, morphology, first_translation, all_translations])
    time.sleep(1)

```

    8835 וַיְבִאֻהוּ



    ---------------------------------------------------------------------------

    JSONDecodeError                           Traceback (most recent call last)

    <ipython-input-24-a85a21ce4c20> in <module>
         14     print(word_number, word)
         15     response = requests.get("http://www.sefaria.org/api/words/{}".format(word))
    ---> 16     if not response.json(): # No translation found
         17         no_translation_writer.writerow([word])
         18         continue


    ~/opt/anaconda3/lib/python3.8/site-packages/requests/models.py in json(self, **kwargs)
        896                     # used.
        897                     pass
    --> 898         return complexjson.loads(self.text, **kwargs)
        899 
        900     @property


    ~/opt/anaconda3/lib/python3.8/json/__init__.py in loads(s, cls, object_hook, parse_float, parse_int, parse_constant, object_pairs_hook, **kw)
        355             parse_int is None and parse_float is None and
        356             parse_constant is None and object_pairs_hook is None and not kw):
    --> 357         return _default_decoder.decode(s)
        358     if cls is None:
        359         cls = JSONDecoder


    ~/opt/anaconda3/lib/python3.8/json/decoder.py in decode(self, s, _w)
        335 
        336         """
    --> 337         obj, end = self.raw_decode(s, idx=_w(s, 0).end())
        338         end = _w(s, end).end()
        339         if end != len(s):


    ~/opt/anaconda3/lib/python3.8/json/decoder.py in raw_decode(self, s, idx)
        353             obj, end = self.scan_once(s, idx)
        354         except StopIteration as err:
    --> 355             raise JSONDecodeError("Expecting value", s, err.value) from None
        356         return obj, end


    JSONDecodeError: Expecting value: line 1 column 1 (char 0)



```
# todo: get first mapping of all translations
verbs_filename = "tanach_verbs_roots_and_translations.tsv"
reader = csv.DictReader(open(verbs_filename), delimiter = "\t")
def get_first_definition(translation):
    # The translation has multiple synonyms, but starts with "to ..."
    return translation.split(",")[0].replace("to ", "")
shorashim_to_translations = {line["Shoresh"]: get_first_definition(line["First Translation"]) for line in reader}

```


```
shorashim_to_translations
two_letter_roots_to_words = defaultdict(set)
for shoresh in shorashim_to_translations:
    no_nekudot = strip_nekudot(shoresh)
    two_letter_root = no_nekudot[:2]
    two_letter_roots_to_words[two_letter_root].add(shoresh)

```


```
two_letter_roots_to_words
```




    defaultdict(set,
                {'עב': {'עָבַד', 'עָבַר'},
                 'של': {'שָׁלַח',
                  'שָׁלַךְ',
                  'שָׁלַם',
                  'שָׁלַף',
                  'שָׁלַשׁ',
                  'שָׁלָה'},
                 'ענ': {'עֲנָה', 'עָנָה'},
                 'נת': {'נָתַךְ', 'נָתַן', 'נָתַץ'},
                 'בר': {'בָּרַח', 'בָרַךְ'},
                 'שו': {'שָׁוַע', 'שׁוּב', 'שׁוּחַ', 'שׁוּר', 'שׂוּם'},
                 'נג': {'נָגַד', 'נָגַח', 'נָגַע', 'נָגַף', 'נָגַשׁ', 'נָגַשׂ'},
                 'שפ': {'שָׁפֵל', 'שָׁפַט', 'שָׁפַךְ'},
                 'חפ': {'חָפֵץ', 'חָפַר', 'חָפַשׂ'},
                 'רא': {'רָאָה'},
                 'דב': {'דָּבַק', 'דָבַר'},
                 'כל': {'כָּלַם', 'כָּלָא', 'כָּלָה'},
                 'פל': {'פָּלַל', 'פָּלָא'},
                 'שכ': {'שָׁכַב',
                  'שָׁכַח',
                  'שָׁכַם',
                  'שָׁכַן',
                  'שָׁכָה',
                  'שָׂכַל',
                  'שָׂכַר'},
                 'קב': {'קָבַץ', 'קָבַר'},
                 'בנ': {'בָּנָה'},
                 'או': {'אָוָה', 'אוֹר'},
                 'חנ': {'חָנֵף', 'חָנַן', 'חָנָה'},
                 'שת': {'שָׁתַן', 'שָׁתָה'},
                 'פצ': {'פָּצַר', 'פָּצָה'},
                 'עש': {'עָשַׁק', 'עָשָׂה'},
                 'צו': {'צָוָה', 'צוּץ', 'צוּר'},
                 'יצ': {'יָצַב', 'יָצַק', 'יָצַר', 'יָצַת', 'יָצָא'},
                 'דש': {'דָּשֵׁן'},
                 'נב': {'נָבַט', 'נָבָא'},
                 'עמ': {'עָמַד', 'עָמַס'},
                 'טו': {'טוֹב'},
                 'בו': {'בּוֹא', 'בּוּז', 'בּוּשׁ'},
                 'שח': {'שָׁחַט', 'שָׁחַר', 'שָׁחַת', 'שָׁחָה', 'שָׂחַק'},
                 'לב': {'לָבַט', 'לָבַשׁ'},
                 'חל': {'חָלַל', 'חָלַם', 'חָלַץ', 'חָלַק', 'חָלָה'},
                 'אס': {'אָסַף', 'אָסַר'},
                 'עי': {'עָיֵף'},
                 'לק': {'לָקַח', 'לָקַט'},
                 'מל': {'מָלֵא', 'מָלַט', 'מָלַךְ'},
                 'לח': {'לָחַם'},
                 'פר': {'פָּרַח', 'פָּרַס', 'פָּרַשׂ', 'פָּרָה', 'פָּרָשׁ'},
                 'כס': {'כָּסָה'},
                 'נט': {'נָטַע', 'נָטַשׁ', 'נָטָה'},
                 'הל': {'הָלַךְ', 'הָלַל'},
                 'נד': {'נָדַד', 'נָדַח', 'נָדַר'},
                 'דח': {'דָּחַף'},
                 'קר': {'קָרַב', 'קָרַע', 'קָרָא'},
                 'נו': {'נוּד', 'נוּחַ', 'נוּן', 'נוּס', 'נוּעַ', 'נוּף'},
                 'אה': {'אָהַב'},
                 'יש': {'יָשֵׁן', 'יָשַׁב', 'יָשַׁע', 'יָשַׁר'},
                 'עו': {'עָוַר',
                  'עָוַת',
                  'עָוָה',
                  'עוּד',
                  'עוּל',
                  'עוּף',
                  'עוּץ',
                  'עוּר'},
                 'רג': {'רָגַז', 'רָגַל', 'רָגַם'},
                 'אב': {'אָבַד', 'אָבַל', 'אָבָה'},
                 'זכ': {'זָכַר', 'זָכָה'},
                 'רו': {'רוּחַ', 'רוּם', 'רוּעַ', 'רוּץ', 'רוּק', 'רוּשׁ'},
                 'רע': {'רָעַע', 'רָעָה'},
                 'אכ': {'אָכַל'},
                 'כנ': {'כָּנַע'},
                 'נכ': {'נָכַר', 'נָכָה'},
                 'בק': {'בָּקַע', 'בָּקַר', 'בָּקַשׁ'},
                 'בע': {'בָּעַר', 'בָּעַת'},
                 'סו': {'סוּף', 'סוּר', 'סוּת'},
                 'יד': {'יָדַד', 'יָדַע', 'יָדָה'},
                 'גנ': {'גָּנַב'},
                 'ספ': {'סָפַד', 'סָפַן', 'סָפַר', 'סָפָה'},
                 'אר': {'אָרַב', 'אָרַג', 'אָרַח', 'אָרַךְ'},
                 'קנ': {'קָנָא', 'קָנָה'},
                 'יר': {'יָרֵא', 'יָרַד', 'יָרַע', 'יָרַשׁ', 'יָרָה', 'יָרָק'},
                 'דר': {'דָּרַךְ', 'דָּרַשׁ'},
                 'על': {'עָלַל', 'עָלַם', 'עָלָה'},
                 'נפ': {'נְפַק', 'נָפַל'},
                 'חו': {'חוּל'},
                 'פע': {'פָּעַל'},
                 'יל': {'יָלַד', 'יָלַךְ', 'יָלַל'},
                 'פק': {'פָּקַד', 'פָּקַח'},
                 'רח': {'רָחַם', 'רָחַץ', 'רָחַק'},
                 'כת': {'כָּתַב', 'כָּתַת'},
                 'יא': {'יָאַל', 'יָאַשׁ'},
                 'בכ': {'בָּכָה'},
                 'אמ': {'אָמַן', 'אָמַץ', 'אָמַר'},
                 'שמ': {'שָׁמֵם', 'שָׁמַד', 'שָׁמַע', 'שָׁמַר', 'שָׂמַח'},
                 'עכ': {'עָכַר'},
                 'אש': {'אָשַׁם'},
                 'מו': {'מוֹט', 'מוּג', 'מוּל', 'מוּשׁ', 'מוּת'},
                 'שנ': {'שְׁנָא', 'שָׁנָה', 'שָׂנֵא'},
                 'יה': {'יְהַב', 'יָהַב'},
                 'חי': {'חָיָה'},
                 'הר': {'הָרַג'},
                 'חד': {'חָדַד', 'חָדַל'},
                 'שא': {'שָׁאַל', 'שָׁאַר'},
                 'הי': {'הָיָה'},
                 'פנ': {'פָּנָה'},
                 'זר': {'זָרַח', 'זָרַע', 'זָרַק', 'זָרָה'},
                 'בי': {'בִּין'},
                 'עז': {'עָזַב', 'עָזַר'},
                 'שב': {'שָׁבַע', 'שָׁבַר', 'שָׁבַת', 'שָׁבָה', 'שָׂבַע'},
                 'צע': {'צָעַק'},
                 'מצ': {'מָצָא'},
                 'זו': {'זוּב', 'זוּר'},
                 'קד': {'קָדַד', 'קָדַם', 'קָדַשׁ'},
                 'צב': {'צָבָא'},
                 'רב': {'רֶבַע', 'רָבַב', 'רָבָה'},
                 'נש': {'נָשַׁל', 'נָשַׁק', 'נָשַׂג', 'נָשָׁה', 'נָשָׂא'},
                 'הס': {'הָסָה'},
                 'יב': {'יָבֵשׁ'},
                 'חז': {'חָזַק', 'חָזָה'},
                 'בצ': {'בָּצַע', 'בָּצַר'},
                 'בש': {'בָּשַׂר'},
                 'מא': {'מָאֵן', 'מָאַס'},
                 'כב': {'כָּבַד', 'כָּבַס', 'כָּבָה'},
                 'כע': {'כַּעַס'},
                 'אל': {'אָלָה'},
                 'חק': {'חָקַר'},
                 'גר': {'גָּרַשׁ', 'גָּרָה'},
                 'טה': {'טָהֵר'},
                 'בח': {'בָּחַן', 'בָּחַר'},
                 'זנ': {'זָנָה'},
                 'גל': {'גָּלַח', 'גָּלַל', 'גָּלָה'},
                 'הד': {'הָדַר'},
                 'קצ': {'קָצַף', 'קָצַר'},
                 'קו': {'קָוָה', 'קוּם', 'קוּץ'},
                 'קט': {'קָטַר'},
                 'גו': {'גּוּב', 'גּוּר'},
                 'פו': {'פּוּץ'},
                 'סב': {'סָבַב'},
                 'יס': {'יָסַד', 'יָסַף'},
                 'יע': {'יַעַל', 'יָעַד', 'יָעַץ'},
                 'טמ': {'טָמֵא'},
                 'בה': {'בָּהַל'},
                 'גז': {'גָּזַל', 'גָּזַר'},
                 'עצ': {'עָצַם', 'עָצָר'},
                 'קש': {'קָשַׁב', 'קָשַׁר', 'קָשָׁה'},
                 'חצ': {'חָצַר'},
                 'יט': {'יָטַב'},
                 'ינ': {'יָנַח', 'יָנַק'},
                 'נק': {'נָקַב', 'נָקַם', 'נָקָה'},
                 'זק': {'זָקֵן'},
                 'זב': {'זָבַח'},
                 'אצ': {'אָצַר'},
                 'קל': {'קָלַל'},
                 'שק': {'שָׁקַט', 'שָׁקַף', 'שָׁקָה'},
                 'שר': {'שָׁרַץ', 'שָׁרַת', 'שָׂרַף', 'שָׂרָה'},
                 'נס': {'נָסַע', 'נָסָה'},
                 'סג': {'סָגַר'},
                 'בט': {'בָּטַח'},
                 'יכ': {'יָכַח', 'יָכֹל'},
                 'הו': {'הָוָא'},
                 'כר': {'כָּרַע', 'כָּרַת'},
                 'כו': {'כּוּל', 'כּוּן'},
                 'רד': {'רָדַם', 'רָדַף'},
                 'נז': {'נָזַל'},
                 'זע': {'זָעַם', 'זָעַק'},
                 'רצ': {'רָצַח', 'רָצַץ', 'רָצָה'},
                 'לו': {'לָוָה', 'לוּז', 'לוּן'},
                 'חר': {'חָרַב', 'חָרַד', 'חָרַף', 'חָרַשׁ', 'חָרָה'},
                 'סת': {'סָתַר'},
                 'קה': {'קָהַל', 'קָהָה'},
                 'פג': {'פָּגַע', 'פָּגַשׁ'},
                 'נצ': {'נָצַב', 'נָצַל', 'נָצַר', 'נָצָה'},
                 'מש': {'מָשַׁח', 'מָשַׁךְ', 'מָשַׁל'},
                 'שי': {'שִׁיר', 'שִׁית', 'שִׂיחַ'},
                 'מע': {'מָעַט', 'מָעַל'},
                 'רכ': {'רָכַב'},
                 'חכ': {'חָכַם'},
                 'סמ': {'סָמַךְ'},
                 'גי': {'גִּיל'},
                 'הפ': {'הָפַךְ'},
                 'גד': {'גָּדַל', 'גָּדַע', 'גָּדַף'},
                 'פש': {'פָּשַׁט', 'פָּשַׁע'},
                 'עט': {'עָטָה'},
                 'שג': {'שָׁגַג', 'שָׁגַל'},
                 'סכ': {'סָכַל'},
                 'דע': {'דָּעַךְ'},
                 'חש': {'חָשַׁב', 'חָשַׂךְ', 'חָשָׁה'},
                 'ער': {'עָרֵל', 'עָרַב', 'עָרַךְ', 'עָרַץ'},
                 'חט': {'חָטָא'},
                 'בג': {'בָּגַד'},
                 'לכ': {'לָכַד'},
                 'נח': {'נָחַל', 'נָחַם'},
                 'גב': {'גָּבַהּ', 'גָּבַר'},
                 'נא': {'נָאַף', 'נָאַץ', 'נָאָה'},
                 'טב': {'טָבַל', 'טָבַע'},
                 'תע': {'תָּעָה'},
                 'שס': {'שָׁסַע'},
                 'שד': {'שָׁדַד'},
                 'תפ': {'תָּפַשׂ'},
                 'מנ': {'מָנָה'},
                 'פת': {'פָּתַח', 'פָּתָה'},
                 'בד': {'בָּדַד', 'בָּדַל'},
                 'כפ': {'כָּפַר'},
                 'צר': {'צָרַע', 'צָרַף'},
                 'בז': {'בָּזַז', 'בָּזָה'},
                 'גא': {'גָּאַל'},
                 'מר': {'מָרַד', 'מָרַר', 'מָרָה'},
                 'מכ': {'מָכַר'},
                 'זמ': {'זִמָּה', 'זָמַר'},
                 'ני': {'נִיר'},
                 'יח': {'יָחַשׂ'},
                 'טר': {'טָרַף'},
                 'צל': {'צָלַח'},
                 'מד': {'מָדַד'},
                 'לע': {'לָעַג'},
                 'רי': {'רִיב'},
                 'סל': {'סֶלָה', 'סָלַח', 'סָלַף'},
                 'פד': {'פָּדָה'},
                 'רמ': {'רָמַס', 'רָמַשׂ', 'רָמָה'},
                 'תו': {'תּוּר'},
                 'חת': {'חָתַת'},
                 'כז': {'כָּזַב'},
                 'לא': {'לָאָה'},
                 'מח': {'מָחָא', 'מָחָה'},
                 'חס': {'חָסֵר', 'חָסַד', 'חָסָה'},
                 'די': {'דִּין'},
                 'תק': {'תָּקַע'},
                 'המ': {'הָמָה'},
                 'מה': {'מָהַר'},
                 'חג': {'חָגַג', 'חָגַר'},
                 'ית': {'יָתַר'},
                 'כח': {'כָּחַד'},
                 'מט': {'מָטַר'},
                 'חב': {'חָבַל', 'חָבָא'},
                 'מס': {'מָסַךְ'},
                 'רפ': {'רָפָא'},
                 'רנ': {'רָנַן'},
                 'צפ': {'צָפַן', 'צָפָה'},
                 'בל': {'בָּלַל'},
                 'אז': {'אָזַן'},
                 'סח': {'סָחַר'},
                 'עד': {'עֲדַף', 'עָדָה'},
                 'תמ': {'תָּמַךְ', 'תָּמַם'},
                 'הג': {'הָגָה'},
                 'אפ': {'אָפַף'},
                 'עק': {'עָקַר'},
                 'דמ': {'דָּמַם'},
                 'שע': {'שָׁעַן'},
                 'אח': {'אָחַז'},
                 'גמ': {'גָּמַל'},
                 'בא': {'בָּאַשׁ'},
                 'צד': {'צָדַק'},
                 'נה': {'נָהַג'},
                 'כה': {'כָּהַן'},
                 'קפ': {'קָפַץ'},
                 'למ': {'לָמַד'},
                 'חמ': {'חָמַד'},
                 'רש': {'רָשַׁע'},
                 'רק': {'רָקַח'},
                 'רט': {'רָטָשׁ'}})




```
# todo: start testing similarities. Take two words that should be similar, and see how spacy ranks them.
nlp = spacy.load("en_core_web_md")

REMOVE_TOKENS = ["to", "or", ",", "be", "by"]
def tokenize(definition):
    
    for token in REMOVE_TOKENS:
        definition = definition.replace(token, "")
    return definition

def get_similarity(word1, word2):
    
    word1_object = nlp(tokenize(word1))
    word2_object = nlp(tokenize(word2))
    return word1_object.similarity(word2_object)

# todo: based on that, start calculating average similarity value of words that have one letter
# different in their shoresh.
# If hard to get all averages, get random sample and average those.
```


```
synonym = Synonyms('apple')
synonym_results = synonym.find_synonyms()
print(synonym_results)
# for word1, word2 in itertools.combinations(synonym_results, 2):
#     score = get_similarity(word1, word2)
#     print(score)
```

    ['aquamarine', 'beryl', 'bice', 'blue-green', 'chartreuse', 'crab apple', 'crabapple', 'dessert apple', 'eating apple', 'edible fruit', 'false fruit', 'fir', 'forest', 'grass', 'greenish-blue', 'jade', 'kelly', 'lime', 'malachite', 'malus pumila', 'moss', 'olive', 'orchard apple tree', 'pea', 'peacock', 'pine', 'pome', 'sage', 'sap', 'sea', 'spinach', 'verdigris', 'vert', 'viridian', 'willow']



```
word1 = "take"#"to pick up,  gather,  glean,  gather up"
word2 = "grab"#"to take,  get,  fetch,  lay hold of,  seize,  receive,  acquire,  buy, bring,  marry,  take a wife,  snatch,  take away"
# word2 = "to consecrate,  sanctify,  prepare, dedicate,  be hallowed,  be holy, be sanctified,  be separate"
get_similarity(word1, word2)

```




    0.5514558266362236




```
two_letter_roots_to_scores = defaultdict(list)
for two_letter_root, shorashim in two_letter_roots_to_words.items():
    if len(shorashim) == 1:
        continue
    for word1, word2 in itertools.combinations(shorashim, 2):
        translation1 = shorashim_to_translations[word1]
        translation2 = shorashim_to_translations[word2]
        score = get_similarity(translation1, translation2)
        two_letter_roots_to_scores[two_letter_root].append(score)
    
```

    <ipython-input-27-f7b2459078f0>:15: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.
      return word1_object.similarity(word2_object)



```
two_letter_roots_to_average_score = {root: statistics.mean(scores) for root, scores in two_letter_roots_to_scores.items()}

```


```
count = 0
for root, score in sorted(two_letter_roots_to_average_score.items(), key = lambda pair: pair[1], reverse = True):
    count += 1
    if count > 20:
        break
    print (root, score)
    for word in two_letter_roots_to_words[root]:
        print ("\t{}\t{}\t".format(word, shorashim_to_translations[word]))

```

    ענ 1.0
    	עָנָה	answer	
    	עֲנָה	answer	
    יה 1.0
    	יָהַב	give	
    	יְהַב	give	
    מא 0.7200636065106191
    	מָאֵן	 refuse	
    	מָאַס	reject	
    לק 0.7095485683091118
    	לָקַח	take	
    	לָקַט	pick up	
    רב 0.6828418706373731
    	רָבַב	be or become many	
    	רָבָה	be or become great	
    	רֶבַע	lie down	
    נפ 0.6172813757843766
    	נָפַל	fall	
    	נְפַק	go or come out	
    יכ 0.5893764367136163
    	יָכֹל	prevail	
    	יָכַח	prove	
    גז 0.5422816939728718
    	גָּזַל	tear away	
    	גָּזַר	cut	
    עד 0.5190573390265029
    	עָדָה	pass on	
    	עֲדַף	remain over	
    עז 0.5126417059283648
    	עָזַב	leave	
    	עָזַר	help	
    של 0.5027044747648135
    	שָׁלַף	draw out or off	
    	שָׁלָה	be at rest	
    	שָׁלַךְ	throw	
    	שָׁלַשׁ	do a third time	
    	שָׁלַח	send	
    	שָׁלַם	be in a covenant of peace	
    או 0.499841880389464
    	אָוָה	desire	
    	אוֹר	be or become light	
    בק 0.498435745313506
    	בָּקַר	seek	
    	בָּקַע	split	
    	בָּקַשׁ	seek	
    אב 0.47434552314260137
    	אָבָה	be willing	
    	אָבַד	perish	
    	אָבַל	mourn	
    צפ 0.4717226173633646
    	צָפָה	look out or about	
    	צָפַן	hide	
    חס 0.45102471191648835
    	חָסַד	be good	
    	חָסֵר	lack	
    	חָסָה	 seek refuge	
    שת 0.4503002789897179
    	שָׁתַן	 urinate	
    	שָׁתָה	drink	
    קו 0.44345501471717247
    	קוּם	rise	
    	קוּץ	 spend the summer	
    	קָוָה	wait	
    יא 0.4415539140533182
    	יָאַשׁ	despair	
    	יָאַל	be foolish	
    סו 0.44003811939581156
    	סוּף	cease	
    	סוּר	turn aside	
    	סוּת	incite	


check if correlation or causation:
Compare all 491 shorashim. Get average score across all of them.
(Plot the distribution)

Get average.
See how many two-letter comparisons are greater than the average?



```
scores_to_pairs = defaultdict(list)
for pair_number, (word1, word2) in enumerate(itertools.combinations(shorashim_to_translations, 2)):
    clear_output(wait=True)
    print(pair_number)
    translation1 = shorashim_to_translations[word1]
    translation2 = shorashim_to_translations[word2]
    score = get_similarity(translation1, translation2)
    scores_to_pairs[score].append((word1, word2))
    
```

    120294



```
count = 0
for score, pairs in sorted(scores_to_pairs.items(), reverse = True):
    count += 1
    if count > 20:
        break
    print (score, pairs)
```

    0.8440238314315596 [('עָבַר', 'בָּרַח')]
    0.827620461005549 [('עָבַר', 'עָדָה')]
    0.7528102754914819 [('עָבַר', 'עָלָה')]
    0.7483312090573144 [('עָבַר', 'יָרַד')]
    0.7452117768064583 [('עָבַר', 'יָצָא'), ('עָבַר', 'דָּעַךְ')]
    0.7418433700186148 [('עָנָה', 'שָׁאַל')]
    0.7304947917738495 [('עָבַר', 'יָתַר')]
    0.7299956931980619 [('עָבַר', 'סָחַר')]
    0.7241135124017588 [('עָבַר', 'נְפַק')]
    0.7215981824839721 [('עָבַר', 'שָׁלַף')]
    0.7203039770970062 [('עָבַר', 'עֲדַף')]
    0.7086994074870323 [('עָבַר', 'נִיר')]
    0.7055390354313348 [('עָבַר', 'נָסַע')]
    0.6998415387869558 [('עָבַר', 'גָּרַשׁ')]
    0.6963531043488504 [('עָבַר', 'לָקַט')]
    0.6909210983105276 [('עָבַר', 'נָתַץ')]
    0.6897323417364254 [('עָבַר', 'נָחַל')]
    0.6893990413874783 [('עָבַר', 'שָׁלַשׁ')]
    0.6874290207071883 [('עָבַר', 'צָפָה')]
    0.6835254133915473 [('עָבַר', 'נָטָה')]



```
random_average = float(sum(scores_to_pairs.keys())) / len(scores_to_pairs)
two_letter_average = float(sum(two_letter_roots_to_average_score.values())) / len(two_letter_roots_to_average_score)
print ("Random average:", random_average)
print ("Two letter average:", two_letter_average)

```

    Random average: 0.3323675684089061
    Two letter average: 0.3034117620257923



```
# todo plotting: is there a package? If not, redo Similarity Map where we take the 
# vector for each word, make a big matrix, run UMAP, and then plot the words.


# embedding = umap.UMAP(n_components=2, metric='hellinger').fit(wordsMatrix)
# f = umap.plot.interactive(embedding, labels=labels["Sefer"], hover_data=labels, point_size=10)
# show(f)
```


```
# todo: be careful for shorashim with same letters but different nekudot.
# although e.g. ענ has both 'עֲנָה' and 'עָנָה', which we do want to treat the same.
```


```
# todo: find the best/primary translation for each shoresh
```


```
# Todo: make Tanach embedding
# https://towardsdatascience.com/creating-word-embeddings-coding-the-word2vec-algorithm-in-python-using-deep-learning-b337d0ba17a8


# First, load Sefaria results.

# no_translation_filename = "tanach_no_sefaria_translation.tsv"

def get_sefaria_results(filename):
    
    words_to_shorashim = {}
    reader = csv.DictReader(open(filename), delimiter = "\t")
    for line in reader:
        words_to_shorashim[line["Word"]] = line["Shoresh"]
    return words_to_shorashim

words_to_shorashim = {}
words_to_shorashim.update(get_sefaria_results("tanach_verbs_roots_and_translations.tsv"))
words_to_shorashim.update(get_sefaria_results("tanach_non_verbs_roots_and_translations.tsv"))


```


```
# Todo: make stopword list
# HEBREW_STOP_WORDS
```


```
def sliding_window(seferPath, window_size):
    """
    Read through a file and get words and their contexts.

    Args:
        seferPath (str): Full path to the sefer file.
        window_size (int): how many words to *each* side that we're looking at.
            (E.g. window_size = 2 will be two words left and two words rightt.)

    Returns:
        context_pairs (list of tuples): focus words and context words.
    """ 
    
    seferName = seferNameFromPath(seferPath)
    context_pairs = []
    with open(seferPath, "r") as psukim:
        
        ### Skip first 9 lines - metadata
        for i in range(9):
            next(psukim)
            
        for pasuk in psukim:

            ### Some quick cleaning:
            cleanedPasuk = pasuk.strip().replace("־", " ").replace("[", "").replace("]", "").replace("׃", "")
            cleanedPasuk = cleanedPasuk.replace("(פ)", "").replace("(ס)", "") ### This is for now; change later if splitting on these.
            
            splitPasuk = cleanedPasuk.split(" ")

            # Skip the non-text lines.
            if len(splitPasuk) == 1:
                continue
            if splitPasuk[0] in ["Chapter", "Section", "Seif", "Siman", "Part", "Daf"]: ## "Paragraph"?                
                continue
                
            ### Delete spaces:
            if [""] in splitPasuk:
                splitPasuk.remove([""])
                
            splitPasuk = [strip_trop_not_nekudot(word) for word in splitPasuk]
            pasuk_as_shorashim = [words_to_shorashim[word] if word in words_to_shorashim else word for word in splitPasuk]
            for focus_index in range(len(pasuk_as_shorashim)):
                for context_index in range(max(0, focus_index - window_size), min(len(pasuk_as_shorashim), focus_index + window_size)):
                    if context_index == focus_index: # skip over the word itself
                        continue
                    context_pairs.append([pasuk_as_shorashim[focus_index], pasuk_as_shorashim[context_index]])
    return context_pairs

```


```
tanach_paths = getTanachPaths()
bamidbar = tanach_paths[0]
context_pairs = sliding_window(seferPath = bamidbar, window_size = 2)
```


```
# One-hot encoding: make a key of all words going to an int
unique_words = set(itertools.chain.from_iterable(context_pairs))
encoding_dict = {word: index for index, word in enumerate(unique_words)}
```


```
X = np.zeros((len(context_pairs), len(unique_words)))
Y = np.zeros((len(context_pairs), len(unique_words)))
for row_number, (focus_word, context_word) in enumerate(context_pairs):
    focus_index = encoding_dict[focus_word]
    context_index = encoding_dict[context_word]
    X[row_number][focus_index] = 1
    Y[row_number][context_index] = 1    
```


```
from keras.models import Input, Model
from keras.layers import Dense

# Defining the size of the embedding
embed_size = 2

# Defining the neural network
inp = Input(shape=(X.shape[1],))
x = Dense(units=embed_size, activation='linear')(inp)
x = Dense(units=Y.shape[1], activation='softmax')(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

# Optimizing the network weights
model.fit(
    x=X, 
    y=Y, 
    batch_size=256,
    epochs=1000
    )

# Obtaining the weights from the neural network. 
# These are the so called word embeddings

# The input layer 
weights = model.get_weights()[0]
```

    Epoch 1/1000
    176/176 [==============================] - 6s 28ms/step - loss: 8.0749
    Epoch 2/1000
    176/176 [==============================] - 4s 23ms/step - loss: 7.6757
    Epoch 3/1000
    176/176 [==============================] - 4s 24ms/step - loss: 7.1443
    Epoch 4/1000
    176/176 [==============================] - 4s 21ms/step - loss: 6.7422
    Epoch 5/1000
    176/176 [==============================] - 4s 23ms/step - loss: 6.5722
    Epoch 6/1000
    176/176 [==============================] - 5s 30ms/step - loss: 6.5240
    Epoch 7/1000
    176/176 [==============================] - 4s 25ms/step - loss: 6.5063
    Epoch 8/1000
    176/176 [==============================] - 5s 29ms/step - loss: 6.4954
    Epoch 9/1000
    176/176 [==============================] - 4s 23ms/step - loss: 6.4876
    Epoch 10/1000
    176/176 [==============================] - 4s 23ms/step - loss: 6.4816
    Epoch 11/1000
    176/176 [==============================] - 4s 22ms/step - loss: 6.4767
    Epoch 12/1000
    176/176 [==============================] - 4s 21ms/step - loss: 6.4725
    Epoch 13/1000
    176/176 [==============================] - 3s 20ms/step - loss: 6.4689
    Epoch 14/1000
    176/176 [==============================] - 4s 22ms/step - loss: 6.4657
    Epoch 15/1000
    176/176 [==============================] - 4s 24ms/step - loss: 6.4628
    Epoch 16/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.4601
    Epoch 17/1000
    176/176 [==============================] - 4s 21ms/step - loss: 6.4576
    Epoch 18/1000
    176/176 [==============================] - 4s 20ms/step - loss: 6.4552
    Epoch 19/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.4529
    Epoch 20/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.4507
    Epoch 21/1000
    176/176 [==============================] - 3s 20ms/step - loss: 6.4486
    Epoch 22/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.4464
    Epoch 23/1000
    176/176 [==============================] - 4s 21ms/step - loss: 6.4444: 0s - loss
    Epoch 24/1000
    176/176 [==============================] - 5s 26ms/step - loss: 6.4422
    Epoch 25/1000
    176/176 [==============================] - 4s 22ms/step - loss: 6.4401
    Epoch 26/1000
    176/176 [==============================] - 4s 22ms/step - loss: 6.4378
    Epoch 27/1000
    176/176 [==============================] - 3s 20ms/step - loss: 6.4354
    Epoch 28/1000
    176/176 [==============================] - 4s 20ms/step - loss: 6.4331
    Epoch 29/1000
    176/176 [==============================] - 4s 20ms/step - loss: 6.4303
    Epoch 30/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.4275
    Epoch 31/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.4244
    Epoch 32/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.4209
    Epoch 33/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.4171
    Epoch 34/1000
    176/176 [==============================] - 4s 23ms/step - loss: 6.4128
    Epoch 35/1000
    176/176 [==============================] - 4s 22ms/step - loss: 6.4080
    Epoch 36/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.4027
    Epoch 37/1000
    176/176 [==============================] - 4s 20ms/step - loss: 6.3968
    Epoch 38/1000
    176/176 [==============================] - 4s 23ms/step - loss: 6.3905
    Epoch 39/1000
    176/176 [==============================] - 4s 23ms/step - loss: 6.3837
    Epoch 40/1000
    176/176 [==============================] - 4s 21ms/step - loss: 6.3767
    Epoch 41/1000
    176/176 [==============================] - 4s 22ms/step - loss: 6.3695
    Epoch 42/1000
    176/176 [==============================] - 5s 26ms/step - loss: 6.3624
    Epoch 43/1000
    176/176 [==============================] - 4s 22ms/step - loss: 6.3553
    Epoch 44/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.3481
    Epoch 45/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.3410
    Epoch 46/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.3338
    Epoch 47/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.3265
    Epoch 48/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.3192
    Epoch 49/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.3118
    Epoch 50/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.3043
    Epoch 51/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.2965
    Epoch 52/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.2888
    Epoch 53/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.2809
    Epoch 54/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.2729
    Epoch 55/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.2648
    Epoch 56/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.2565
    Epoch 57/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.2483
    Epoch 58/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.2399
    Epoch 59/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.2317
    Epoch 60/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.2233
    Epoch 61/1000
    176/176 [==============================] - 3s 20ms/step - loss: 6.2150
    Epoch 62/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.2069
    Epoch 63/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.1988
    Epoch 64/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.1908
    Epoch 65/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.1829
    Epoch 66/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.1752
    Epoch 67/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.1676
    Epoch 68/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.1601
    Epoch 69/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.1530
    Epoch 70/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.1458
    Epoch 71/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.1388
    Epoch 72/1000
    176/176 [==============================] - 3s 20ms/step - loss: 6.1320
    Epoch 73/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.1254
    Epoch 74/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.1189
    Epoch 75/1000
    176/176 [==============================] - 3s 17ms/step - loss: 6.1126
    Epoch 76/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.1064
    Epoch 77/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.1004
    Epoch 78/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.0944
    Epoch 79/1000
    176/176 [==============================] - 3s 20ms/step - loss: 6.0888
    Epoch 80/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.0832
    Epoch 81/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.0777
    Epoch 82/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.0723
    Epoch 83/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.0671
    Epoch 84/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.0620
    Epoch 85/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.0570
    Epoch 86/1000
    176/176 [==============================] - 4s 21ms/step - loss: 6.0522: 0s -
    Epoch 87/1000
    176/176 [==============================] - 3s 17ms/step - loss: 6.0475
    Epoch 88/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.0429
    Epoch 89/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.0383
    Epoch 90/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.0339
    Epoch 91/1000
    176/176 [==============================] - 4s 21ms/step - loss: 6.0295
    Epoch 92/1000
    176/176 [==============================] - 3s 17ms/step - loss: 6.0254
    Epoch 93/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.0213
    Epoch 94/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.0172
    Epoch 95/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.0134
    Epoch 96/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.0094
    Epoch 97/1000
    176/176 [==============================] - 3s 19ms/step - loss: 6.0057
    Epoch 98/1000
    176/176 [==============================] - 3s 18ms/step - loss: 6.0020
    Epoch 99/1000
    176/176 [==============================] - 3s 19ms/step - loss: 5.9984
    Epoch 100/1000
    176/176 [==============================] - 3s 18ms/step - loss: 5.9949
    Epoch 101/1000
    176/176 [==============================] - 3s 17ms/step - loss: 5.9915
    Epoch 102/1000
    176/176 [==============================] - 3s 18ms/step - loss: 5.9881
    Epoch 103/1000
    176/176 [==============================] - 3s 20ms/step - loss: 5.9848
    Epoch 104/1000
    176/176 [==============================] - 3s 17ms/step - loss: 5.9815
    Epoch 105/1000
    176/176 [==============================] - 3s 15ms/step - loss: 5.9784
    Epoch 106/1000
    176/176 [==============================] - 3s 14ms/step - loss: 5.9753
    Epoch 107/1000
    176/176 [==============================] - 3s 15ms/step - loss: 5.9723
    Epoch 108/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9693
    Epoch 109/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9664
    Epoch 110/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.9635
    Epoch 111/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.9607
    Epoch 112/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9580
    Epoch 113/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9553
    Epoch 114/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9526
    Epoch 115/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9501
    Epoch 116/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9475
    Epoch 117/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9450
    Epoch 118/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.9426
    Epoch 119/1000
    176/176 [==============================] - 3s 15ms/step - loss: 5.9402
    Epoch 120/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9378
    Epoch 121/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9355
    Epoch 122/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.9333
    Epoch 123/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9310
    Epoch 124/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9289
    Epoch 125/1000
    176/176 [==============================] - 3s 14ms/step - loss: 5.9267
    Epoch 126/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9245
    Epoch 127/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.9225
    Epoch 128/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9204
    Epoch 129/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.9184
    Epoch 130/1000
    176/176 [==============================] - 3s 14ms/step - loss: 5.9164
    Epoch 131/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9145
    Epoch 132/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9126
    Epoch 133/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9107
    Epoch 134/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.9088
    Epoch 135/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.9070
    Epoch 136/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9052
    Epoch 137/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9034
    Epoch 138/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.9017
    Epoch 139/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.9000
    Epoch 140/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.8983
    Epoch 141/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.8967
    Epoch 142/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.8949
    Epoch 143/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.8934
    Epoch 144/1000
    176/176 [==============================] - 3s 15ms/step - loss: 5.8918
    Epoch 145/1000
    176/176 [==============================] - 3s 14ms/step - loss: 5.8901
    Epoch 146/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.8886
    Epoch 147/1000
    176/176 [==============================] - 3s 14ms/step - loss: 5.8871
    Epoch 148/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.8856
    Epoch 149/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.8841
    Epoch 150/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.8827
    Epoch 151/1000
    176/176 [==============================] - 3s 14ms/step - loss: 5.8812
    Epoch 152/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.8798
    Epoch 153/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.8784
    Epoch 154/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.8770
    Epoch 155/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.8756
    Epoch 156/1000
    176/176 [==============================] - 3s 15ms/step - loss: 5.8742
    Epoch 157/1000
    176/176 [==============================] - 3s 14ms/step - loss: 5.8729
    Epoch 158/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8716
    Epoch 159/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.8703
    Epoch 160/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.8690
    Epoch 161/1000
    176/176 [==============================] - 3s 16ms/step - loss: 5.8678
    Epoch 162/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8664
    Epoch 163/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.8652
    Epoch 164/1000
    176/176 [==============================] - 3s 17ms/step - loss: 5.8640
    Epoch 165/1000
    176/176 [==============================] - 3s 15ms/step - loss: 5.8627
    Epoch 166/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.8616
    Epoch 167/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.8604
    Epoch 168/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8593
    Epoch 169/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8581
    Epoch 170/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8570
    Epoch 171/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8559
    Epoch 172/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8547
    Epoch 173/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8536
    Epoch 174/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8525
    Epoch 175/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8514
    Epoch 176/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.8504
    Epoch 177/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.8494
    Epoch 178/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8484
    Epoch 179/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.8473
    Epoch 180/1000
    176/176 [==============================] - 3s 15ms/step - loss: 5.8462
    Epoch 181/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.8453
    Epoch 182/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8443
    Epoch 183/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.8434
    Epoch 184/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.8423
    Epoch 185/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.8413
    Epoch 186/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8404
    Epoch 187/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.8394
    Epoch 188/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8385
    Epoch 189/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.8376
    Epoch 190/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.8367
    Epoch 191/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.8358
    Epoch 192/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8348
    Epoch 193/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.8339
    Epoch 194/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.8330
    Epoch 195/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8321
    Epoch 196/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8313
    Epoch 197/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8305
    Epoch 198/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8296
    Epoch 199/1000
    176/176 [==============================] - 3s 16ms/step - loss: 5.8288
    Epoch 200/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8279
    Epoch 201/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8271
    Epoch 202/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8262
    Epoch 203/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8254
    Epoch 204/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8246
    Epoch 205/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.8238
    Epoch 206/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8230
    Epoch 207/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.8223
    Epoch 208/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.8214
    Epoch 209/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8207
    Epoch 210/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.8199
    Epoch 211/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8191
    Epoch 212/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8183
    Epoch 213/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8175
    Epoch 214/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8169
    Epoch 215/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8161
    Epoch 216/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.8154
    Epoch 217/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.8147
    Epoch 218/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.8140
    Epoch 219/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.8131
    Epoch 220/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8125
    Epoch 221/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.8117
    Epoch 222/1000
    176/176 [==============================] - 3s 15ms/step - loss: 5.8111
    Epoch 223/1000
    176/176 [==============================] - 3s 14ms/step - loss: 5.8104
    Epoch 224/1000
    176/176 [==============================] - 3s 16ms/step - loss: 5.8097
    Epoch 225/1000
    176/176 [==============================] - 3s 15ms/step - loss: 5.8090
    Epoch 226/1000
    176/176 [==============================] - 3s 14ms/step - loss: 5.8083
    Epoch 227/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8076
    Epoch 228/1000
    176/176 [==============================] - 3s 17ms/step - loss: 5.8069
    Epoch 229/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8063
    Epoch 230/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8057
    Epoch 231/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.8049
    Epoch 232/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.8044
    Epoch 233/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8036
    Epoch 234/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8030
    Epoch 235/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.8024
    Epoch 236/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.8018
    Epoch 237/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.8011
    Epoch 238/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.8005
    Epoch 239/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7998
    Epoch 240/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7993
    Epoch 241/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7986
    Epoch 242/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7980
    Epoch 243/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.7974
    Epoch 244/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7968
    Epoch 245/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7962
    Epoch 246/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7955
    Epoch 247/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7950
    Epoch 248/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.7944
    Epoch 249/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.7938
    Epoch 250/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.7933
    Epoch 251/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.7926
    Epoch 252/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.7920
    Epoch 253/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.7915
    Epoch 254/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7909
    Epoch 255/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7903
    Epoch 256/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7898
    Epoch 257/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7893
    Epoch 258/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7887
    Epoch 259/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7882
    Epoch 260/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7876
    Epoch 261/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7871
    Epoch 262/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7865
    Epoch 263/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7860
    Epoch 264/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7855
    Epoch 265/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7849
    Epoch 266/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7845
    Epoch 267/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7839
    Epoch 268/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7834
    Epoch 269/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7829
    Epoch 270/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7824
    Epoch 271/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7819
    Epoch 272/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7814
    Epoch 273/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7809
    Epoch 274/1000
    176/176 [==============================] - 3s 17ms/step - loss: 5.7804
    Epoch 275/1000
    176/176 [==============================] - 4s 21ms/step - loss: 5.7800
    Epoch 276/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7795
    Epoch 277/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7790
    Epoch 278/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.7785
    Epoch 279/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7780
    Epoch 280/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.7776
    Epoch 281/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7771
    Epoch 282/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.7766
    Epoch 283/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.7763
    Epoch 284/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7757
    Epoch 285/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7753
    Epoch 286/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7748
    Epoch 287/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7744
    Epoch 288/1000
    176/176 [==============================] - 3s 15ms/step - loss: 5.7739
    Epoch 289/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7736
    Epoch 290/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7731
    Epoch 291/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7728
    Epoch 292/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7722
    Epoch 293/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7718
    Epoch 294/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7714
    Epoch 295/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7710
    Epoch 296/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7706
    Epoch 297/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7702
    Epoch 298/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7697
    Epoch 299/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7693
    Epoch 300/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7689
    Epoch 301/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7685
    Epoch 302/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7681
    Epoch 303/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7677
    Epoch 304/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7673
    Epoch 305/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7669
    Epoch 306/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7665
    Epoch 307/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7662
    Epoch 308/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7657
    Epoch 309/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7654
    Epoch 310/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7650
    Epoch 311/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7646
    Epoch 312/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7642
    Epoch 313/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7639
    Epoch 314/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7636
    Epoch 315/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7631
    Epoch 316/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7628
    Epoch 317/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7625
    Epoch 318/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7621
    Epoch 319/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7618
    Epoch 320/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7614
    Epoch 321/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7610
    Epoch 322/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7607
    Epoch 323/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7603
    Epoch 324/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7600
    Epoch 325/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7597
    Epoch 326/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7593
    Epoch 327/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7589
    Epoch 328/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7586
    Epoch 329/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7583
    Epoch 330/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7580
    Epoch 331/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7576
    Epoch 332/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7573
    Epoch 333/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7570
    Epoch 334/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7567
    Epoch 335/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7564
    Epoch 336/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7560
    Epoch 337/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7557
    Epoch 338/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7553
    Epoch 339/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7550
    Epoch 340/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7547
    Epoch 341/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7544
    Epoch 342/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7541
    Epoch 343/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7538
    Epoch 344/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7535
    Epoch 345/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7532
    Epoch 346/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7530
    Epoch 347/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7527
    Epoch 348/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7523
    Epoch 349/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7521
    Epoch 350/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7517
    Epoch 351/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7515
    Epoch 352/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7512
    Epoch 353/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7508
    Epoch 354/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7506
    Epoch 355/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7503
    Epoch 356/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7500
    Epoch 357/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7497
    Epoch 358/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7495
    Epoch 359/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7491
    Epoch 360/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7489
    Epoch 361/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7486
    Epoch 362/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7483
    Epoch 363/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7480
    Epoch 364/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7478
    Epoch 365/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7475
    Epoch 366/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7473
    Epoch 367/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7470
    Epoch 368/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7467
    Epoch 369/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7465
    Epoch 370/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7462
    Epoch 371/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7459
    Epoch 372/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7457
    Epoch 373/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7454
    Epoch 374/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7451
    Epoch 375/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7450
    Epoch 376/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7446
    Epoch 377/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7443
    Epoch 378/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7441
    Epoch 379/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7439
    Epoch 380/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7437
    Epoch 381/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7434
    Epoch 382/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7431
    Epoch 383/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7429
    Epoch 384/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7427
    Epoch 385/1000
    176/176 [==============================] - 4s 21ms/step - loss: 5.7424
    Epoch 386/1000
    176/176 [==============================] - 4s 22ms/step - loss: 5.7422
    Epoch 387/1000
    176/176 [==============================] - 5s 26ms/step - loss: 5.7420
    Epoch 388/1000
    176/176 [==============================] - 3s 15ms/step - loss: 5.7417
    Epoch 389/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7415
    Epoch 390/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7412
    Epoch 391/1000
    176/176 [==============================] - 3s 15ms/step - loss: 5.7410
    Epoch 392/1000
    176/176 [==============================] - 3s 15ms/step - loss: 5.7407
    Epoch 393/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7406
    Epoch 394/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7403
    Epoch 395/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7401
    Epoch 396/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7399
    Epoch 397/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7396
    Epoch 398/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7393
    Epoch 399/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7392
    Epoch 400/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7389
    Epoch 401/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7387
    Epoch 402/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7386
    Epoch 403/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7382
    Epoch 404/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7381
    Epoch 405/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7378
    Epoch 406/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7376
    Epoch 407/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7375
    Epoch 408/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7372
    Epoch 409/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7370
    Epoch 410/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7368
    Epoch 411/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7365
    Epoch 412/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7363
    Epoch 413/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7362
    Epoch 414/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7359
    Epoch 415/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7357
    Epoch 416/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7355
    Epoch 417/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7353
    Epoch 418/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7351
    Epoch 419/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7349
    Epoch 420/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7347
    Epoch 421/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7345
    Epoch 422/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7343
    Epoch 423/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7341
    Epoch 424/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7338
    Epoch 425/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7338
    Epoch 426/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7335
    Epoch 427/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7333
    Epoch 428/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7331
    Epoch 429/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7329
    Epoch 430/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7327
    Epoch 431/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7325
    Epoch 432/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7323
    Epoch 433/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7321
    Epoch 434/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7320
    Epoch 435/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7318
    Epoch 436/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7316
    Epoch 437/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7313
    Epoch 438/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7312
    Epoch 439/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7310
    Epoch 440/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7308
    Epoch 441/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7306
    Epoch 442/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7304
    Epoch 443/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7303
    Epoch 444/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7301
    Epoch 445/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7299
    Epoch 446/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7297
    Epoch 447/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7296
    Epoch 448/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7294
    Epoch 449/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7291
    Epoch 450/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7291
    Epoch 451/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7288
    Epoch 452/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7286
    Epoch 453/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7285
    Epoch 454/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7283
    Epoch 455/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7282
    Epoch 456/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7280
    Epoch 457/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7278
    Epoch 458/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7276
    Epoch 459/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7275
    Epoch 460/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7272
    Epoch 461/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7271
    Epoch 462/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7270
    Epoch 463/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7268
    Epoch 464/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7266
    Epoch 465/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7265
    Epoch 466/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7263
    Epoch 467/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7262
    Epoch 468/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7260
    Epoch 469/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7258
    Epoch 470/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7256
    Epoch 471/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7255
    Epoch 472/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7253
    Epoch 473/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7252
    Epoch 474/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7250
    Epoch 475/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7248
    Epoch 476/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7247
    Epoch 477/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7246
    Epoch 478/1000
    176/176 [==============================] - 3s 15ms/step - loss: 5.7244
    Epoch 479/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7242
    Epoch 480/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7241
    Epoch 481/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7239
    Epoch 482/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7238
    Epoch 483/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7236
    Epoch 484/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7235
    Epoch 485/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7233
    Epoch 486/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7232
    Epoch 487/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7230
    Epoch 488/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7228
    Epoch 489/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7227
    Epoch 490/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7226
    Epoch 491/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7224
    Epoch 492/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7223
    Epoch 493/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7221
    Epoch 494/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7220
    Epoch 495/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7218
    Epoch 496/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7217
    Epoch 497/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7216
    Epoch 498/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7214
    Epoch 499/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7213
    Epoch 500/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7212
    Epoch 501/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7210
    Epoch 502/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7209
    Epoch 503/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7207
    Epoch 504/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7206
    Epoch 505/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7204
    Epoch 506/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7203
    Epoch 507/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7201
    Epoch 508/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7200
    Epoch 509/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7199
    Epoch 510/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7197
    Epoch 511/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7196
    Epoch 512/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7195
    Epoch 513/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7193
    Epoch 514/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7192
    Epoch 515/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7191
    Epoch 516/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7190
    Epoch 517/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7188
    Epoch 518/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7187
    Epoch 519/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7185
    Epoch 520/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7184
    Epoch 521/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7183
    Epoch 522/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7181
    Epoch 523/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7180
    Epoch 524/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7179
    Epoch 525/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7179
    Epoch 526/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7176
    Epoch 527/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7176
    Epoch 528/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7174
    Epoch 529/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7172
    Epoch 530/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7171
    Epoch 531/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7170
    Epoch 532/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7168
    Epoch 533/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7167
    Epoch 534/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7166
    Epoch 535/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7165
    Epoch 536/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7164
    Epoch 537/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7163
    Epoch 538/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7162
    Epoch 539/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7160
    Epoch 540/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7159
    Epoch 541/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7158
    Epoch 542/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7157
    Epoch 543/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7155
    Epoch 544/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7154
    Epoch 545/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7153
    Epoch 546/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7152
    Epoch 547/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7151
    Epoch 548/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7149
    Epoch 549/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7148
    Epoch 550/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7147
    Epoch 551/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7146
    Epoch 552/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7145
    Epoch 553/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7144
    Epoch 554/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7143
    Epoch 555/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7142
    Epoch 556/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7140
    Epoch 557/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7139
    Epoch 558/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7138
    Epoch 559/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7137
    Epoch 560/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7136
    Epoch 561/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7134
    Epoch 562/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7134
    Epoch 563/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7132
    Epoch 564/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7132
    Epoch 565/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7130
    Epoch 566/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7128
    Epoch 567/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7128
    Epoch 568/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7126
    Epoch 569/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7127
    Epoch 570/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7125
    Epoch 571/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7124
    Epoch 572/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7123
    Epoch 573/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7122
    Epoch 574/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7120
    Epoch 575/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7119
    Epoch 576/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7119
    Epoch 577/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7118
    Epoch 578/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7117
    Epoch 579/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7115
    Epoch 580/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7114
    Epoch 581/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7114
    Epoch 582/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7112
    Epoch 583/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7112
    Epoch 584/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7109
    Epoch 585/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7109
    Epoch 586/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7109
    Epoch 587/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7107
    Epoch 588/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7106
    Epoch 589/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7105
    Epoch 590/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7104
    Epoch 591/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7104
    Epoch 592/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7102
    Epoch 593/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7101
    Epoch 594/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7100
    Epoch 595/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7100
    Epoch 596/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7099
    Epoch 597/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7097
    Epoch 598/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7096
    Epoch 599/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7096
    Epoch 600/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7095
    Epoch 601/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7094
    Epoch 602/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7093
    Epoch 603/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7092
    Epoch 604/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7091
    Epoch 605/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7089
    Epoch 606/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7089
    Epoch 607/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7088
    Epoch 608/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7088
    Epoch 609/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7086
    Epoch 610/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7085
    Epoch 611/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7083
    Epoch 612/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7084
    Epoch 613/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7082
    Epoch 614/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7081
    Epoch 615/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7081
    Epoch 616/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7080
    Epoch 617/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7079
    Epoch 618/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7078
    Epoch 619/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7077
    Epoch 620/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7076
    Epoch 621/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7075
    Epoch 622/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7074
    Epoch 623/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7073
    Epoch 624/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7072
    Epoch 625/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7071
    Epoch 626/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7071
    Epoch 627/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7070
    Epoch 628/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7069
    Epoch 629/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7068
    Epoch 630/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7067
    Epoch 631/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7066
    Epoch 632/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7065
    Epoch 633/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7064
    Epoch 634/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7064
    Epoch 635/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7063
    Epoch 636/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7062
    Epoch 637/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7061
    Epoch 638/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7060
    Epoch 639/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7060
    Epoch 640/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7060
    Epoch 641/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7058
    Epoch 642/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7057
    Epoch 643/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7057
    Epoch 644/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7055
    Epoch 645/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7055
    Epoch 646/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7054
    Epoch 647/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7053
    Epoch 648/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7052
    Epoch 649/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.7051
    Epoch 650/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7051
    Epoch 651/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7050
    Epoch 652/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7049
    Epoch 653/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7048
    Epoch 654/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7048
    Epoch 655/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7047
    Epoch 656/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7046
    Epoch 657/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7045
    Epoch 658/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7044
    Epoch 659/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7043
    Epoch 660/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7043
    Epoch 661/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7042
    Epoch 662/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7041
    Epoch 663/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7040
    Epoch 664/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7039
    Epoch 665/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7039
    Epoch 666/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7038
    Epoch 667/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7038
    Epoch 668/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7037
    Epoch 669/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7036
    Epoch 670/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7036
    Epoch 671/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7035
    Epoch 672/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7034
    Epoch 673/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7033
    Epoch 674/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7032
    Epoch 675/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7032
    Epoch 676/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7031
    Epoch 677/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7030
    Epoch 678/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7029
    Epoch 679/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7029
    Epoch 680/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7028
    Epoch 681/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7027
    Epoch 682/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7027
    Epoch 683/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7026
    Epoch 684/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7025
    Epoch 685/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7025
    Epoch 686/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7024
    Epoch 687/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7022
    Epoch 688/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7022
    Epoch 689/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7021
    Epoch 690/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7021
    Epoch 691/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7020
    Epoch 692/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7020
    Epoch 693/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7018
    Epoch 694/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7018
    Epoch 695/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7017
    Epoch 696/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7017
    Epoch 697/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7016
    Epoch 698/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7015
    Epoch 699/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7015
    Epoch 700/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7014
    Epoch 701/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7013
    Epoch 702/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7012
    Epoch 703/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.7012
    Epoch 704/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7011
    Epoch 705/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.7011
    Epoch 706/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7010
    Epoch 707/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7009
    Epoch 708/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7008
    Epoch 709/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7007
    Epoch 710/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7008
    Epoch 711/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7006
    Epoch 712/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7006
    Epoch 713/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7005
    Epoch 714/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7005
    Epoch 715/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7003
    Epoch 716/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.7003
    Epoch 717/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7003
    Epoch 718/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7002
    Epoch 719/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7001
    Epoch 720/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.7000
    Epoch 721/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.7000
    Epoch 722/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6999
    Epoch 723/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6999
    Epoch 724/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6997
    Epoch 725/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6997
    Epoch 726/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.6997
    Epoch 727/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6996
    Epoch 728/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6995
    Epoch 729/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6994
    Epoch 730/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6995
    Epoch 731/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6994
    Epoch 732/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6993
    Epoch 733/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6992
    Epoch 734/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6992
    Epoch 735/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6991
    Epoch 736/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6990
    Epoch 737/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6990
    Epoch 738/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6989
    Epoch 739/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6988
    Epoch 740/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6988
    Epoch 741/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6987
    Epoch 742/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.6986
    Epoch 743/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6986
    Epoch 744/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6985
    Epoch 745/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6985
    Epoch 746/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6985
    Epoch 747/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6983
    Epoch 748/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6983
    Epoch 749/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6983
    Epoch 750/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6982
    Epoch 751/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6981
    Epoch 752/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6981
    Epoch 753/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6980
    Epoch 754/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6979
    Epoch 755/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6978
    Epoch 756/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6978
    Epoch 757/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6978
    Epoch 758/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6977
    Epoch 759/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6977
    Epoch 760/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6977
    Epoch 761/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6976
    Epoch 762/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6974
    Epoch 763/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6974
    Epoch 764/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6973
    Epoch 765/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6973
    Epoch 766/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6972
    Epoch 767/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6971
    Epoch 768/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6972
    Epoch 769/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6970
    Epoch 770/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6970
    Epoch 771/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6969
    Epoch 772/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6968
    Epoch 773/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6968
    Epoch 774/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.6968
    Epoch 775/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6968
    Epoch 776/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6967
    Epoch 777/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.6966
    Epoch 778/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6966
    Epoch 779/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6965
    Epoch 780/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6964
    Epoch 781/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6965
    Epoch 782/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6964
    Epoch 783/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6962
    Epoch 784/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.6962
    Epoch 785/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6962
    Epoch 786/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6960
    Epoch 787/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6961
    Epoch 788/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6960
    Epoch 789/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6959
    Epoch 790/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6959
    Epoch 791/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6958
    Epoch 792/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.6958
    Epoch 793/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.6957
    Epoch 794/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.6957
    Epoch 795/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6956
    Epoch 796/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6956
    Epoch 797/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6956
    Epoch 798/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6955
    Epoch 799/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6954
    Epoch 800/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6953
    Epoch 801/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6953
    Epoch 802/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6952
    Epoch 803/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6952
    Epoch 804/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6951
    Epoch 805/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6952
    Epoch 806/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6950
    Epoch 807/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.6950
    Epoch 808/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6949
    Epoch 809/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6949
    Epoch 810/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.6948
    Epoch 811/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6948
    Epoch 812/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6947
    Epoch 813/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6947
    Epoch 814/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6947
    Epoch 815/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6946
    Epoch 816/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6945
    Epoch 817/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6945
    Epoch 818/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.6945
    Epoch 819/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6944
    Epoch 820/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6943
    Epoch 821/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6943
    Epoch 822/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.6943
    Epoch 823/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6942
    Epoch 824/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6941
    Epoch 825/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6941
    Epoch 826/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6940
    Epoch 827/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6939
    Epoch 828/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6939
    Epoch 829/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.6939
    Epoch 830/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.6938
    Epoch 831/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6938
    Epoch 832/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6937
    Epoch 833/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.6936
    Epoch 834/1000
    176/176 [==============================] - 1s 7ms/step - loss: 5.6937
    Epoch 835/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.6936
    Epoch 836/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6936
    Epoch 837/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6935
    Epoch 838/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6934
    Epoch 839/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6934
    Epoch 840/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6933
    Epoch 841/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6932
    Epoch 842/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6932
    Epoch 843/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6932
    Epoch 844/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6931
    Epoch 845/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6931
    Epoch 846/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6930
    Epoch 847/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6930
    Epoch 848/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6929
    Epoch 849/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6929
    Epoch 850/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6929
    Epoch 851/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6928
    Epoch 852/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6927
    Epoch 853/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6927
    Epoch 854/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6926
    Epoch 855/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6926
    Epoch 856/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.6926
    Epoch 857/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.6925
    Epoch 858/1000
    176/176 [==============================] - 1s 8ms/step - loss: 5.6924
    Epoch 859/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6924
    Epoch 860/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6923
    Epoch 861/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.6924
    Epoch 862/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6922
    Epoch 863/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.6922
    Epoch 864/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6923
    Epoch 865/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.6921
    Epoch 866/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.6921
    Epoch 867/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.6920
    Epoch 868/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6920
    Epoch 869/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6920
    Epoch 870/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6918
    Epoch 871/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6919
    Epoch 872/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6918
    Epoch 873/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.6918
    Epoch 874/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6917
    Epoch 875/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6917
    Epoch 876/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.6916
    Epoch 877/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6915
    Epoch 878/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.6916
    Epoch 879/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6915
    Epoch 880/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6914
    Epoch 881/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6914
    Epoch 882/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6914
    Epoch 883/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6913
    Epoch 884/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6912
    Epoch 885/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6912
    Epoch 886/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6912
    Epoch 887/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6911
    Epoch 888/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6911
    Epoch 889/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6910
    Epoch 890/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6909
    Epoch 891/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6909
    Epoch 892/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6909
    Epoch 893/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6909
    Epoch 894/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6908
    Epoch 895/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6908
    Epoch 896/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6908
    Epoch 897/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6906
    Epoch 898/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.6907
    Epoch 899/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6906
    Epoch 900/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6905
    Epoch 901/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6905
    Epoch 902/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6905
    Epoch 903/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6904
    Epoch 904/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6904
    Epoch 905/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6903
    Epoch 906/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.6903
    Epoch 907/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6902
    Epoch 908/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.6903
    Epoch 909/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6901
    Epoch 910/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6901
    Epoch 911/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6900
    Epoch 912/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6901
    Epoch 913/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6900
    Epoch 914/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6900
    Epoch 915/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6899
    Epoch 916/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6898
    Epoch 917/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6898
    Epoch 918/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6897
    Epoch 919/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6897
    Epoch 920/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6897
    Epoch 921/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6896
    Epoch 922/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6896
    Epoch 923/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.6895
    Epoch 924/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.6896
    Epoch 925/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6895
    Epoch 926/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6894
    Epoch 927/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6894
    Epoch 928/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6893
    Epoch 929/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6892
    Epoch 930/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6892
    Epoch 931/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6893
    Epoch 932/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6892
    Epoch 933/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.6891
    Epoch 934/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6891
    Epoch 935/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6891
    Epoch 936/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6890
    Epoch 937/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6890
    Epoch 938/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6889
    Epoch 939/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6889
    Epoch 940/1000
    176/176 [==============================] - 2s 14ms/step - loss: 5.6888
    Epoch 941/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6888
    Epoch 942/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6888
    Epoch 943/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6887
    Epoch 944/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6887
    Epoch 945/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6887
    Epoch 946/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6886
    Epoch 947/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6885
    Epoch 948/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6885
    Epoch 949/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6884
    Epoch 950/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6885
    Epoch 951/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6884
    Epoch 952/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6883
    Epoch 953/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6882
    Epoch 954/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6883
    Epoch 955/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6883
    Epoch 956/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6882
    Epoch 957/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6882
    Epoch 958/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6881
    Epoch 959/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6881
    Epoch 960/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6880
    Epoch 961/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6881
    Epoch 962/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6880
    Epoch 963/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6878
    Epoch 964/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6879
    Epoch 965/1000
    176/176 [==============================] - 2s 11ms/step - loss: 5.6879
    Epoch 966/1000
    176/176 [==============================] - 3s 17ms/step - loss: 5.6877
    Epoch 967/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6877
    Epoch 968/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6877
    Epoch 969/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6877
    Epoch 970/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6876
    Epoch 971/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6876
    Epoch 972/1000
    176/176 [==============================] - 3s 15ms/step - loss: 5.6875
    Epoch 973/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6876
    Epoch 974/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6876
    Epoch 975/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6874
    Epoch 976/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6874
    Epoch 977/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6875
    Epoch 978/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6874
    Epoch 979/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6873
    Epoch 980/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6872
    Epoch 981/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6872
    Epoch 982/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6871
    Epoch 983/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6871
    Epoch 984/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6871
    Epoch 985/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6871
    Epoch 986/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6870
    Epoch 987/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6870
    Epoch 988/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6869
    Epoch 989/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6870
    Epoch 990/1000
    176/176 [==============================] - 2s 9ms/step - loss: 5.6868
    Epoch 991/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6868
    Epoch 992/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6868
    Epoch 993/1000
    176/176 [==============================] - 2s 10ms/step - loss: 5.6868
    Epoch 994/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6867
    Epoch 995/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6867
    Epoch 996/1000
    176/176 [==============================] - 2s 13ms/step - loss: 5.6866
    Epoch 997/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6866
    Epoch 998/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6865
    Epoch 999/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6865
    Epoch 1000/1000
    176/176 [==============================] - 2s 12ms/step - loss: 5.6864



```
# Creating a dictionary to store the embeddings in. The key is a unique word and 
# the value is the numeric vector
embedding_dict = {word: weights[encoding_dict[word]] for word in unique_words}
writer = csv.writer(open("2021-12-23_bamidbar_embedding_window_size_2.tsv", "w"), delimiter = "\t")
for word, coords in embedding_dict.items():
    writer.writerow([word, coords[0], coords[1]])
```


```
import matplotlib.pyplot as plt
plt.figure()#figsize=(10, 10))
for word, coord in embedding_dict.items():
    plt.scatter(coord[0], coord[1])
    plt.annotate(word, (coord[0], coord[1]))
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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QCgBLFEpj6m8svPmzbNHH33Urr766jwuEqzPPvus3Xnnnc47HGxr1661XXfd1bZu3eoEsRezwdcKXp+fe+659sILL9hrr71mp512Wthx2HvvvW3mzJk2ZcoU69Chgzu+qOzN7733np1wwgnOc/3TTz9ZjRo18q4hMbty5UrbtGlT2OteeumlNnz4cHvwwQedx7osGwK4LI9emL4jgMvx4HJrEIAABCAAAQhAAALFIhBKACtkuFGjRvb33387r+m0adPytS3xuP/++7tQ6LFjx4a8bufOnW3y5MkujLpp06bumHACWCHKhx12mJ1//vlOYBdln3/+uR1xxBHWpUsX59H1Fq58Uf/+/e3pp582ie3nn3/enZabm2tVq1Z1/ZwzZ06R15W4V4i37M8//yzT4c+6BwRwsaZN2TgJAVw2xoleQgACEIAABCAAAQjEh8CKDVvtzamLbPKC1fbP1myrUTnDWlbbbLed3i1vD7C/skKeFWJ877332uWXX27Lly83hQlrf6zCiLXvtWXLljZ79uyQnT300EPtm2++cYLS7xUOJYC1x1hCuW7duvb777/bXnvt5cKtFX5dlPkw7VGjRtnJJ58ckQCWwL7iiivst99+c8f7cyXiDz/8cFdnWPWGi7JHHnnErr32WpOQVnh4WTcEcFkfwSL6jwAux4PLrUEAAhCAAAQgAAEIFEpgS1aO3fHBDBv9w5+WlZOb77jsdcttyfDzbec929qc6VOtSsXt+1kPOeQQty9X4cXazyuP6YIFC5wIllWoUMF5QrVXNpRFKoAlRm+77TZTsivtrZWo1rlfffVVofejfbc6TtcP3oNbmAf4u+++s65du5rCpm+55Rbr16+fS7alPcpXXnmlrV692u1vVth3YSbPuK4rDhLrrVq1KvNPHQK4zA9h4TeAAC7Hg8utQQACEIAABCAAAQiEJCDxe/bzU5zXN5R5AVy5yd52/OBn7cXzOjoRXKdOHbcfVnt4//rrL/vxxx9dyLHP6iwB3KRJExcGXBIBrORZCmfec8898wSwxLeEbGF2zTXX2JAhQ1wSLXljA03nyWN99913u/2+3o466ijn0ZbHWh5p7WseOHCg+7hTp07Om9uuXbsin6L333/fZao+5phjXGKw8mAI4PIwioXcAwK4HA8utwYBCEAAAhCAAAQgEJLATe9Mt9enhPbS6oRAAbxTvwfttI672AVtqjmR2L59e5s6dWrIdmMlgNW4PMwy7wEuSgBv3LjRCW9law61B1dljHzSLd/xnJwcV75J5/k9vtr3e+SRRzrxfddddzmvcDhTmLTCpXWO/l0eDAFcHkYRAVyOR5FbgwAEIAABCKQWAV8DVfsStT+xLJkSKG3evNllA8YSQ+DvDVvswPu/LBD2HNibYAFcMb2C3dBqjV14Tr8i97kmSgAPGzbMlVDSz9ChQyMCu2LFCttxxx0LJMxatmyZS/KlxFYqwaTEXYWZQrXlUNtnn33sl19+iei6ZeEgBHBZGKVi9hEPcDHBcRoEIAABCEAAAgkjUJYFsGrAKhxViY+wxBAY+uUce/i/oZNU+R4FC2D9vtXC9+3zN0a4/bEaR5n+vWrVqrwbUfmfwBBoiUKFGHtTpmXtzw2XBEvHR+MB1h5e7b+dNWuW248biS1cuNDtXT7wwANdjeFA86WRmjdv7koj1apVK2STl1xyiav9WxZfRhXFCAEcyRNURo9BAJfRgaPbEIAABCAAgRQm4AWwMt5+8MEHZWrxrT2jqquqfaNYYgic+dxkGzdnZZEXDyWAt75/h/01c6oTphKcMu3RlegMtEABrLJFqg8cbLESwNqHrDlw0UUXuT24b7zxhruUnjElw1J26sIsVKmnwGN9Xd8zzjjDXnnllQLN+NJHKpWkpF9VqlRJzIDG4aoI4DhATZYmEcDJMhL0AwIQgAAEIACBSAmUZQEc6T1yXPwInDhsgv20aG3UF2i7S20b0//AqM+L9oRo1uc+q7S/hjzSsuDs1NH2IdWPRwCX4ycgmglWjjFwaxCAAAQgAAEIJDGBYE+VF8BFdblNmzaufItMNVsHDRpkffv2tTfffNOVa7nppptM9U/XrFljDRs2tD59+rgMuTVq1Mhr9oUXXnBCQh6uPfbYwy6++GJTyGeg+WMCf5eenm477LCD7bvvvi5U1ofL+mPkmVuyZEmB7stbp7qvSrI0YMAA69GjR75jgkO/3333XTvxxBPzjqlUqZI1bdrUJTFSJl+Fr4brq/asKhGSPJnio9I3wcmSkvjRKFbXIvEAh2q4a8v69vL5nYp1zcCTlEE6sH6w/8z/XpmZZb7mr54/eZVDmUKX5e31tv/++7t/BmenDnWufwYLazuSG1WZJP0oOZiSden51U9ZNwRwWR/BIvqPAC7Hg8utQQACEIAABMoJgWABPHHiRNPPf//7X5d5VuJVJVtk2m/5xRdfWGDYppJOaa/junXr3D7Jgw8+2JQA6JRTTrHddtvNlW75/vvvneBUm94U6ioRojBTeda00JfgPfvss/OO8QL4sMMOs549e7rfqy6qkl1pH6VEgcSsytN4057JRYsWOWGufilxkUz9VDjt6NGjLTMz015//XXXR2/BAlhtSwR727p1qwvPffvtt01i+JtvvjEviHRMqL4q66/uS8zE4LjjjnP9Ls8WyR7gUPd/3ZF72GXdWpQYjR+HwPrBgeMTfIFw5Y+Cjy9MYAcfJ++xrKjSSuFu1j+Teo70MsXXLg53XrJ/jgBO9hEqQf8QwCWAx6kQgAAEIAABCJQKAS+A5RmVSPNJe6LxBKucyz333JPXX62B7rjjDuvdu7dJOKrWqcSjvLUSKIGWlpbmvMAqNSPBPG3aNOc1DSVaJALkAZMgl/d48ODBLuGRxI68s77P9913n/NCy/MqUe5NwrVBgwa2dOlSt4dTQtlbYfcb7MXt2rWrSTTJs61re4+uF17yhsvbHWgSMEcffbR9+umnzjPerVu3UhnbRFwkkizQwf1SFuiJNx5mDWpWLnGXNaahPLT+91dccYW7xpNPPun+W79+fTvooIMivm5hAju4AZ/4Kpq2g9vQs6sf1RbWyxg9z/op64YALusjWET/EcDleHC5NQhAAAIQgEA5IVCYAPaeYH+bs2fPdgmxVGJImXADPcG//vqrC0mW6b/z5s1zHleJPXnCfJh0IDIJSwnSevXquTYnTZrkPlY4tETK5Zdf7s6VB817gOX9VRmZ1157zXmBJWblbX755ZediJGAlnkBLOH9ww8/uJBpiXTvxVUyI4lSeaeV3EgWKIADPc6hvLgSv2IxZswYJ05kRQlgfa4+nnXWWU6033nnneXk6Ql9G+HqAAefpTrA9520/fmJt5V0fV6YwI53v8tT+wjg8jSaQfdS0glWjtFwaxCAAAQgAAEIJIDAP//847xjEr0KA5YArVy5sp166qlub2ygBziwewr71B5WeTBHjRrlQjFnzpxpU6ZMsQ4dOjgxWa1aNduyZYsLn9Zn559/vvPSPv/88y7s+OSTT3Z7LbU30odVK9OtQom9t0xtbNq0ye2VlYdO3l6VwQn2qiqUWntxvd1///15IdeBAlje7AkTJuQrnaPPda7a2GmnnZwXWF7cQAEczosr77JE9g033GC6diQCeOzYsXb44Yc7LspeXJ5tS1aOnf38FJu8YHXY2+y0W1178byOVqViethjY3EA6/NYUCxZGwjgkvFzX5L6AlOK8qlTp7ov9JycHLdZXGE311xzTb6EC4GXe+mll1wxaxWZ1h8AFaLW28EuXbqUsFfbT2eCxQQjjUAAAhCAAAQgUEICWVlZbr+tElfJixpoEqGPP/54XqKdUHVLJWIl3HbZZRdXl1SJhLRekriUqX2FMUvAKvGVQoP32msv6969u0n4yYursN/WrVvb9OnTC5RW6tevn7366qtWu3ZtU3/kRdbaTN5bWbAg9e35+5B41d7hYA9wYQLYX0/ney9uOAGsY70X9/TTT3de6PPOO8+ee+45141wHmDfZ11b7ZR3kwi+44PfbPQPiy0rJ7fA7SrsuU+7pnZbr71LTfyyPk+Opw4BXMJxCKz/JcGpumHr1693bx/1tlJx8nqzuOOOO+a7koSxEiboy1q14vTGUl/QeoP51ltv5cv6V9wuIoCLS47zIAABCEAAAhCIFQGJU9UZXbhwYcgmvQBu1KiRCy8OFLb+BAk9Jaq6+eabnQfZe4Ll1ZVJWLdt29btjZXQC06s5cWfhO0vv/xSQACrPYnq6tWru9qqcmYouZXfSxssgJWMS0mxFEattZs80tpfHCyAFRYt77LuTft+vQUKYO/FVViysljLMRLKA6xzvRdX96n1ZTQeYM+gsLqvsRrvZGtnxYatNur7xfbd/FX2z9Zsq1E5wzo3r2d92zeNyZ7faO+X9Xm0xGJ/PAK4hEzlxf3uu+/s6quvdvtRvOkLXHs6fvrpJzvttNPcWzpv2o+ivR3ac6L9Jv48/Vt7TSSKlUyhTp06JeodE6xE+DgZAhCAAAQgAIEYEHj//fedaC3MggVw0xZNbc7vc6xy+vaEREpOpdDl7OxsF+6r/bnyBM+fP99UkkgmL+gFF1xg1157rd1w5w024usRNujoQdZg7wZ2wpATrNbiWvbIBY+4kGNlfZYX+aKLLsrrkkSn1k1VqlRx+3WVOVkC00flBQvSd955x0X6yUusUO5gAayEXDpHHmit++SZ1n2EEsDei6t2tF5U2HZhAtiLWF+WRv3wpZIi9QCnmgCOwSMc0yZYn8cUZ7EaQwAXC1tkJ0nQ6otTe1vkFdaXpEzC+OOPP3YeYL1dDLSrrrrKnnjiCXv44YddjbeSGBOsJPQ4FwIQgAAEIACBkhJQNJzWO8Fhz4HtFhDATZtaj6E97Kmjn7IqGVVs2LBhrpSQfrT1TJ5gZXi+9dZb85rp37+/Pf3009bnrj42d9e5tmn5Jpt93Wyr1rKaNR/U3P6Z+Y/98cAfecdLoMqb7E0lh+SQ8Gs1iWAJ5Vq1ajnhev3119sDDzyQd/yjjz7q1mmhPMA6VyHd8gifeeaZLtxYkYCqDRtKABfHiyvhryzSkWaB1nVT1QNc0mc41uezPo810ejbQwBHzyziM/QlrVAamcJeFP6iUGftL1EWwMWLF7svr0AbN26cq18XbU2wUJ1igkU8VBwIAQhAAAIQgEAcCHz77bfOA1qUhRLAHQd2tH077mu3HXCb216mUiyKqtP6SJ7gP//8062nvHXs1NGmTplqrR5qZZUaVLItS7bY3EFzrWL9ilbvsHqW+Xemrf7q34RIRxx5hF115VUu3Fi2cuVKJ3CVjEoh20oWpRrEys8yefJkl+BKmZu9yYsaGN3nPcAHHHCAK5GkNZ683hK/EsHyYKvPoQRwYV5cCejAOsDyEH/11Vcu+7XCtNWvcHWAA7krM/bw4cPz1VCOw5DTZBgCrM8T/4gggOM4Bj4lv75M9QZUnmC/R0Vp85U+P9j0lrFGjRou/FnhNyUxJlhJ6HEuBCAAAQhAAAIlJaCtYgpVjlYAH9P/GPuuyXc2qM4gO7nXyS56TnVsvSdYSUS9ab9ulepVLLdSru315F7u15vmbbL5d4W+bka9DMtenW2HHrK9xFGgeY+uwomV1Eph0io3pEzTyjitLNa6H0X46bpybMgkgFVeSCJZIc3a5qaSQyq/JG9tsAD20YAqZySxHKqWr7Ji+/BmXUPHqGasttkdd9xx9t577+Xruw+BDjdmhECHIxTfz1mfx5dvJK0jgCOhVMxjLrzwQpdmvlevXqb9LzL99/jjj3eJGgrbDyPxq7ehCpv2hdiL0wUmWHGocQ4EIAABCEAAArEioHVQoOczEiHctGlTO/7S4+3rxl/blW2vtAtbX+hO857gWbNm5cu7snLzSusxuoetnLjSstdku2NXfb7KslZn5YVA63ezBs6yrFVZtuu1u9qixxdZ2rY0t4+3TZs2LmmWhK5M4lTrtbS0NFfjVyJVglahysr78vbbb+fdhsKoFU5dmPnQal8HWF5ceZ1VlkmCVvuM4+HFlfdY4l4/SsylsHH9Dks8AdbniR8DBHCcxkB7fI899lgXoqLySPpylSlcRm/eQqX4911RWLRq1Pmw6XBd9BMp+DiFuuy+++42Y8aMcE3wOQQgAAEIQAACEIg5gUg8wP6iPhRaAlge4PGNxtsBjQ6wEUeMcOHIqpohcfrhhx/m6+eI6SPsyZ+etPn3zLdNczbl+6zSjpVst5t3s4q1K+YJ4P977v9s69KtVm9SPftz2p8uIk97fxWF16xZMxdirPWTN58VOrBhJeHSPt8TTjihSGYKYQ724mqdp3vRvuLmzZvnOz9WXlw5WRTWrSzUKtcpka0+F8e8iC9q7VqcdlP1HARw4kceARyHMVDxdX1JqA7dY489Zkps5U015pT6Xl9I2u8byhQmI/GLAI7D4NAkBCAAAQhAAAKlRiCSPcChOvNrnV9tVu1Z1rpBa3v16FeL7O9F/73IJi2blO+YrDVZ7v8XD1/s/tv8pub5BHCF9Ap54lqfe5Enb68yQcsjLFO4seoPq4axTEmxrrvuOhfurLBn1e4NZwqrDg6BDneOT1il/kiUa79vtF5cbb9T4i2dLxFfXFP/ZQjg4hLMfx4CODYcS9IKArgk9EKcqzAffUEsWrTIVOv3kUceyXcUIdAxBk5zEIAABCAAAQgkLYFIskAHd36bbbOPd/nYtqZvzSdSC7vJMz4+w6avmJ7v4/n3bd//27B3Q/ff6q22JyUNNInrJzs/6XKuqKyQzHtO69at67JEK4GX9vrKJHaVzVr1eu+66y5XN7hdu3Yhu6Wax16AKqGWyi+99dZb7niVuwxn8t6++OKLzmurrXHafywHSjReXO9NVmlNieDimpw5KveJAC4uwfznIYBjw7EkrSCAS0Iv6Fx9WXXt2tVlKlQNN9Wk82/N/KGRJsFSZkN5kEtiTLCS0ONcCEAAAhCAAARiQcDXAc61XKtg272Jocx/Pr/mfPup/k/ukMA9wIWdF8oDvHH29pq7yggtq1inYoHTFV7deGJjV1IpNzc33+cSfSqz1KdPH9OWMnmyg+3JJ5+0yy+/PGS3JDglgoNNWZwVAi2BrZ9wVhIvrpwxEvQKt1Yd4uKavOHaR4wALi7B/OexPo8Nx5K0ggAuCb2Ac/UFpWLn33//vZ100kk2atSovOLsgZfYvHmze5MXrgySSiEpMUJJjAlWEnqcCwEIQAACEIBALAiorNDIl0ba0sVLwza3osoKG99wvG1L22YZaRn2eZ/PrX7V+kWe5/cAhzrIe4IVAh1sEtddK213XATv5VUuFglUeWvHjx/v9tMqw/Obb75pY8aMcft6BwwY4GochzLtu1Ud4YceeshUFUTt3HLLLS6rtCqBRBo+7b242jYnJ0ugCA0OrZaHWp5phU336NHDJecKLLep5Kra26x7USKuvfbay/7zn//YU089ZbNnz3brViVpVTvyWnvzAjjc4KlM1X777RfusJT/nPV54h8BBHAMxkBiVmEweqt35JFHusyBvpB6qOaVxl97SfSlqS/PQNN+4SeeeMIefPBBt8ekJMYEKwk9zoUABCAAAQhAIFYEJILvefEe2/bnNkuztLxmtddVL/yP7XWs1Tmkjk2rO82JX1mfVn1cHeBw5rNAZ2/bngE60LwnWCHQv57zq2XUybA9h+wZkbhWciw5LnbddVfXpGro6kfRfEputeeee7qfokylkFQLOHAPcKTnql3vxd1jjz1cFuyiBPDEiROdp1rJT5VzRl7oX375xapX/zf8W1vzrr32WpfRWkJc/953331dWSUlIVPWboV8KxGYai7LJJxHjBgR8jYVPn7fffe5awTXZg43bqn6OevzxI88AriEY6AviZNPPtm9DdSbuU8//TRsmIm+VPRmTin1J02alJfKX//u1q2bqxes/RqRhMYU1X0mWAkHl9MhAAEIQAACEIgZgS3ZW+yyjy+zdQvWWYPNDSwjN8MmjZ1kP4z9wXY9e1er2a1m3rXaN2xvw3sMt8rplSO6/u0Tb7e35/xbnijUSYECOBJx7UsJ+YRYEXUk6KBYtKEmQ4UhF5VcS3t2FcYtD6+8zt7WrVvnxLjKbCp6UY4bebV9kiyfrFVRjWPHjg17yw888IDdeOON1r9/f+dJTjbTCwx583XPkey9Lo3+sz4vDcpFXwMBXMIxUAp878VVOIyyA4ayhx9+2BUv96ZzdK72ZEgMqy6cUvwruYLCp1WXrqTGBCspQc6HAAQgAAEIQCCWBCSCH5j6gL07912Tx3b5mOW24r0V1vjcxlb3kLrOM3tCixPsxo43Rix+1T+1e+kXl9r3y78vtLteAPd7rV9E4tonxNIe2kRbtAJYybsOO+wwlzhLHuhA075m1TJWHeLFixe7kGmZMmGrXrG8zSoJ9c8//7hjCjM5gVTGSW0ojLxVq1aJxlTg+j6btqI0Dz300KToH+vzxA8DAriEY6B9EkqeEM5CZeDTvo6hQ4eayibpC6Zz585uf4hKJMXCmGCxoEgbEIAABCAAAQjEmoDClsfMGWPDHxpuP77yo3Ud0NXOPPdMO7HliWH3/BbWl2BxHXycBHDNBjVtxbIVUYnrWN97JO2t2LDV3py6yCYvWG3/bM022/C3vXvjSdax8wE2edJE10RRHmAJUu3xleh75ZVXnAfU2/3332833XSTS4712Wef5f1e7Wlv8nvvvefCvOU5bdy4caHdlYiWmNbWvo8++iiS2yr1Y7TfWXu4tbYOdESVekcCLsj6PJH0t18bAZz4MYhbD5hgcUNLwxCAAAQgAAEIFINAsCczEkdCmzZtnCCT+TwqSizl1znyQKo80MUXX+z26HpxPfj0wbZ8xnI79aVTrXub7nZRm39DgQvrura0KSFWcRNGqV2FPauEkSL7lEzKlxHy11SdYWVV7tWrlw0cONB23HHHvO5sycqxw/tdbhNGDc/fRdXizc21CpWqWc/zr7d3Hr/FqlbKKLS+sK9r3KFDBxdtKE+oN4lclXIKzjfj9yZfcsklbl+22lDSrMJMe4S151gJvxTNWBrmPbpnnHGGE/alacWp5xyqf6zPS3PUQl8LAZz4MYhbD5hgcUNLwxCAAAQgAAEIFINAsABW4ib9hDJ5J5U3JVDsXHnllabyQ/I+quqGTB4+5WEJzqy8//77uwzM2geqDMfajhbKsrOzXQSetqEFhvIWJ2FUKAGsUGqFJHtTf1Q15MMPP3Tid/LkyU7AS/ye/fwU++zlJ23dhNet2h4HWaVG28OKt21eb+snj7YKGZUsNzvTWnTrY+tnfGtXXHGF63uwKa9Mly5dLC0tzYnZwOhCCXx5eQsLC5bXWOfMmTMnrz5ycPvTp083vZjYZ599XKKtcBacSVp7jpULp3379u7FhV4GeFNotRJzKULy448/dolmNd4S7Lruli1bXEWVYcOG2amnnlrg0sFCVXyUETvYdFyNGjVcLh5tPdTLCOXhKcwQwOFGuex8jgAuO2MVdU8RwFEj4wQIQAACBQhQA5OHAgKxIxDNfFLWY4mgKVOmmDyZMolfiWAJmptvvtn97vnnn7fzzz/fnnvuOTvvvPPyOqvw3dq1a7uSP0WZkkBJSAWH8hY3YVSwB7iwa6siiK55wAEHuJcAN70z3V6fstjWjn/VCeC6Pa+wai062Z9D+1nF+s0sa+UfVqnxXpabtdmyVvzhmpXQ1RY6mff6SsBK5AZXGimKgTJd+2RfO+20ky1fvjyiQX/mmWecJzZYMHsGgY1IjCsRlcKxO3bsaPKEv/POO6ZM0qqAIjHvzYdXn3LKKU6c6r/Kat2pUydXjkrVVpQ/R9fWC5JACxaqEyZMcJyefvppmz9/vkvaJfGtFx7KXK3wbf3+mGOOcS8lCjMEcESPRJk4CAFcJoapeJ1EABePG2dBAAIQCCQQzZj6ljgAACAASURBVIIdchCAQNEEIp1PCh/WHlV5MSVgvHnRqPJCCjOW+aRO8kTKIynT3k+V+Tn99NNdSaCiTKWFJEBDhfIWljBK7clTqWsEJ4yKVACrDd2j7nXs+O/s4o9XWVZObj4BXLPNkbZoSF/Lzc0xy9pqlZvsbVV2a2vrxm+/p+9/+d3a7bOH+7f3yuqe5WFeuHChqVSnQpp9KLMSWyl3jRJfqQSSvOfyBgeK/9atWzuvrkolBe6bVZZnjZ+Etf6txK8SkD179ixUACsDtTysnteyZcucN1fnycMuL76uJ0Grusk+G7VPsLVixQrn7ZVAF6epU6e6iikKudb/674k/AOtMKFamGdbJbr0EkGeeoVY+/JPwc8MArj8fLshgMvPWBa4EwRwOR5cbg0CECg1ApEu2EutQ1wIAmWEQPbKlbZ29GjbNGWqbdu40dKqV7eVuze3drfckq+ebfDtqDxP3759XWlJVcZQuUlvCstVtmGFN48cOdJ9Lm+wQnzHjRuXd5y8escee2wBz2LwtSR62rVr5/YTK1w62ApLGOWPa9u2bYGEUdEI4LvvvtsGDx5spwy4w76r3M41G+gBlgBe9sJVlrl8nvtMArj6vofb6k+fUCofe/y/M+zKHnu5z3yFEZXR1L5feTovv/xy51mVh1UmpgopPvvss03JWBU6LhF8wQUXuOzPMu8pDQ6BFmO9jLj++utdOPKgQYNM/Q8lLD0DhXj7lxE6TjzllVf9ZIlx9VV7uAPNe4l1DxL1sh9++MGNud8DHG4KyPuvJF4yL1xbtGiRJ9TlOdaLAL2AEBPd82WXXWbak65Q+lBjqHbCmd9DXtRxrM/DUYz/5wjg+DNO2BWYYAlDz4UhAIFyRAABXI4Gk1spFQLbtmyx5ffca2vHjDHLzs53zSVZmdZj/nzr0LSpfTdnjqWF2HPpQ5q1L1ahqdq/6037dSWQ9F9vCqeV8AjMWCzPo8JzJaC0n7QwkxB86aWXbMSIEXbhhRcWOKywhFH+QC/+AhNGRSOAVadXHtrWx19k6/Y8zjUbLIBXvPeAbfp9u7iXAK6210G25osRlla9jp3++Cf28vmd3GdNmzZ1nlWt/+Rl3Xnnnd2PvKnyBsuTqpcF2hMrD6686xLw1atXd17sYCtMAIuzPLNqs1GjRkUK4MP6HWbrMtbZkulLbNmPy2zHXXa0ab9Os0EDBrnQdQnkv//+211anmv1R3ukNW4Kj1aYshKG6TkQY33mM1dLwM+YMcMJV4VH63PxlCnMWixkoQSw2pSnV4L+0UcfdRmz9WLAJ1LzY6g96ConJYt2D3lhzxzr81L5GiryIgjgxI9B3HrABIsbWhqGAARSiEBw8pbCbv2nn35yNTTl2dBet6VLl7r/v++++9wCW+XwtOdN+/3uvfde53UKNC3StNDXglILMR0TXLYjlAdCi1eFByoMUR4fhWQGmjK7qk59oKl/8sIo9FB7Js8888wUGlFuNZ4EJH4XX3iRbZo6NeRlvADev2pVe6d3H2v6zAhLq1Il37F6JuXdlYfu1ltvLdCOPHnz5s1zIk/i1ofY+gPXrl3r5oTEnwRSYSYRJ9FYs2ZNW7RokRPWwVachFES0trX+vjjj7s+FGXeA9zm1Gtt7a7b69QGC+A1375k6yeNsh269rMqTfa2VZ8Ntew1S61K8w7Wc+AQG9P/QHeePOVismbNGvfdI1P2bCXZkrALtnPPPdd5gWWR1Mn1HuDg0p7BHmCVo+p8XGeb9sk0a3ZdM6vxfzXcNebfO982zd5kLW5uYY2XNrZvX/jW7cVdtWqV+/zZZ591e7m9ySutfd3aC65x9GWa5KWVBYvUSy+91GUBl4UTwHqGJOD3228/J6o1VnoRoTb17CnEXvWTxU4vWIqywvaQF3YO6/MicZbKhwjgUsGcmIswwRLDnatCAALli4DKocg7FMq0yJXAlQiVt0GiUuGGqvGukDot2vRveVoUaqeFo8SwQvsUeqlQTm/yLmghpj2OStiiRaUWpYGmBaAWtoFZXxVGqH1xypYqr4baVT+8ySsiERxo8qrI66LfS0ioJqgENwaBkhJYNvhWW/vWW4U2EyiAX9llV6vdt681uvOOvOPliZR4kYfXz6ngxvSyR15AvXSSgAk2eRJff/119xMqS7A/3otPH8obqtMSyHqZJVEtoRxsocJ/lZxJe1ojMb8H+KjBI+23zO0vr4IF8D+/fGGrPn7s3+YqpJnlbrNaB5xix51/td3fc2cnevVyTV5z7emVqR9aCwbvo9ZnCkvWvSnrsbym8nKGq5MbiQCuW29Hu+iTi23ssE9t7YS1+QTwX6P+spUfr7QmFzaxzL8zbcV7K1wItL5HZfLYS/R689+leqnnv6/0glE/skBPu7JJ+5cN/hkqzAMc6K1XCL0SbImbXoL40HA/LmojsIZyqDEtag95qONZn0cyM+J7DAI4vnwT2joTLKH4uTgEIJACBB544AHnde3fv78LKZT5PYoSo/IwaBGnOqDefAiivFzybgSbFv7ai6eQP52vhZm3opKwyFsmz4iErARtJCbBLS+wrqP9csHe40ja4BgIeALZK1bYnG7dC4Q9BxIKFsCWkWEtv/7KMurXd4eptI1CWvWjl0ehzAuj0aNHuwzB3iRUr776arcvWGV13n///UIHR/NMYbMSVj6UN/hgeYgVous9kKEai6RkUGGd8Am9OnfuZIde1ste/3W7mA8WwFv+nGHLX73BKjVqaXW6X2CZfy+wNZ8PdwL43nvutrfuuMDtbQ22hx56yK677jqX8EoJpgJN3xMS/tdcc43JcyqRF8oDHnhOYQJYESsNGuxop3S90l5aO8J+23Gi/fnMnwUE8PIxy53obXxuY8taneX+XbVWVdu8frO7TLAA9vuT9TJBybyCLVAAT5s2zSX10nexnqFQHmCNt0KeJXQl/vVCUsdrD7A3CWDvuVc2bH2fF2Xh9pCHOpf1eeK/MxHAiR+DuPWACRY3tDQMAQhAwO2ra968uUvgElg7VGjkjVDSFglKfR5YW1LeDoX96Vx5X0OZvFYKq/v2229dkhpvRQlgCWZlRJUnR/vWIjUJeAn5YDER6fkcBwFPYOXw4bbisceLBFJAAJtZgwEDrP4l20N0JTY1n2bNmlUgtNk3LIGihE8SudpKIKGqrMUSN5qXKoujl0tF1XRVKZ3TTjstX43h4I4HJ4zS55pbCiv2VljCqMC2QtUBlvdanloJ7FdePcnWbZ1g1317h+XkZuQJYF8HODdzk62b+Ial1ahrtdqfYJl/zXF7gut0OcVmffqijf/iY+e51NYGvYhTRIpMbcs7KUbKCu1N4l8v6JSR+bXXXnMe0JKEQG/6Z7N99p+ZNvePRfbK/rfbtrSciAVweo10y/knx3UtUAArSkXRLvLm5ubmup9g8wJY0QAS8RK0enb0AjGUANbzIPGrNjdt2uRe/imCRhE+Kq0lHvKWa6uK6jbrO1jJ2IqycHvIQ53L+jzx35kI4MSPQdx6wASLG1oahgAEyjEBLYi0YNXeXx/KKM+BMpDWqLF9L5vM16kMrh2qz7SXTGGFSsTjk7IEIlMyGi3C1H4oKyyJTlECeMuWLc6Do/BA1U4NNHlHFFatfgWb97gpBHvgwIHleGS5tXgTWHTe+bZx4sQiLxNKAFfv0sV2ef45V9ZGoafh6rEq3F/HyCpWrOgSJWnPr/brau9mmzZtwt6qD1vV9gG9sAplgQmj5CnUHNM1fBKmwHOCE0YFfqYIEHmmvUmI6TtFWaqvvPJcmzuvt+XmZtmLM06xb5ccmCeAw91El94X2t2XneZCgfWiTYItMGO2PNu6TvB2Ch/2q2M1/5UFWt7d4oZAf/XK7/bb+KX2Q5P/2tRdPnLdDvYA52zOsUVDF9nGGRutVvtapv/XvytUqmC5mdvFrfqj/bb63pQ3X4mpfL1fCWDVg9ZLQW+6Xwl7Rb8oWZl/boK/J/VCQG3ppaR/Qeijd3ztaF9fWm3LO6zP9YKkKItkD3mo81mfh3uy4/85Ajj+jBN2BSZYwtBzYQhAoAwSUC1IhSTKu6MFdbDJu6IMpdp/qEW3MohqMRaqdqj3CgSXcPFtalGqxak8MYEZbv3nxRHAakv9UtvyhHiTx0PiQAt+LXSDTYlnlLSnsIRDZXAo6XKCCPxxyqm2edq0qK9etU0ba/bmG1Gf50+QwJOpRE48zZfgicRbGmk/FvzxlM2f/6g7PDOnoj324yU2a832urmyv1670f03e8Mqy1m7zKrv093SazawmplLreq6v2zu3DmuHq+SPwULtsIEsA9llsdcHs9ILVQI9MZ1W+2lmyfatpxc+3CvYfZn7VmuOS+A6xxaxyo3rGzZ/2Tbyo9WRnQpfX+pb/369XMvHfVSUgLYl6QKbkTfy/Ko6wXK4Ycfnpf12e8B1t5tnd+pUyeX9EsC98gjj3Tf3X6Przz7qi0s03/1kiNc2aNI9pCHumHW5xE9BnE9CAEcV7yJbZwJllj+XB0CECg7BCR+VRdSC8ZwJjGp0Dl5jfbZZx8XehlsSswjr2twtlR/nBfAum5GRkaB80sigNU/ea+9hSvj5AWwPCg+u2o4BnwOgVAEIvEAhzrPe4CLS9W/2JFAi8S0993XiA08Xi+QNH8KMyWOitRb6ttQTePly5e7F1O6pqI0FJrs7aefzrbVa/59MSUR/Mask2z8kk4uHFr7f9eNf822LPz3xUJ6xQxrUK+ydep0uHsRp3wCPutzYN+DBbCEpDzVimbR95e839FYKAH8/cd/2OT357tmxuwzxJbX3P7d4wVwuPbTq6dbzsYcK+y7UCWXFJocvBfX1wr+7bffXKKvwr6LC7u+ztG5isLRuMv04lMvU1RPWvuntae4MItkD3lR19ZnRWUoD8eNz0tGAAFcMn5JfTYCOKmHh85BAAJJREDJchT2HKlNnDjReRuCk7bofIVJyuOgMGdf3zK4XQRwpKQ5riwRiGQPcKj7CdwDXJz79WI2XLZe37b35AZfK/jlUfDnXjhHkjDKnyuPo8oN6WWYxJV+fOkhHTP1+z62fv1PBW573daaNn5JZ/t9TUubPuYDWzLpG7v6mYvtoCbf2Q6VN1itWm2tQ/vRReIKFsCBfdF3UCzs/cd/ssUz17imAj3A+n8J2zXfbv9MtuHXDS7sWUmw6h5S1/1u1sBZlrUqq1ABrO9RZeAONgl/jaPfbhLqu9ifE1w+Sb/XHmi9HAwW3oqYUQi99ghr24heFoSySPaQF8aX9XksnryStYEALhm/pD6bCZbUw0PnIACBJCEgD82QIUNChj2H6qIWRtrTp7BDhc8pYUqgaZ+aQu1C7Q32xyGAk2Tw6UZMCUSSBbrABYOyQBenQ9GGQHtPbvC1VJ5Me5ALs+KEQEtQ6eWa2pX3WPtsA0VVsAc41LVffHG1vfzSWvtibPO8j+vWOcjatn2xSFzBAjiwL7rXSE3tyOOqH+/N9pErox/43pYvWO+a8nuAN0zfYAsfXeiE7tKRSwtcJhoBrEzMuqZyKaiWsTKAd+/e3SU4U810eaXVL4VA61h9J4txYL6GUAK4qO9gX9dXybS0b9gnFQu8kUj2kCOAI33CSv84BHDpMy+1KyKASw01F4IABMowAe3jVcbPSE2L2LFjx9qZZ57pavoGm/biKSNpYEixEmYF7stVCRLV7fTeBy1MtV/Ym0/u4ve0+d8XlQTL7wEmBDrSkeS4eBAIVwc4+JrBdYCL06doQ6CLcw2dU5wQ6HDXCtwDXNixd9+13L7+emM+Abx784HWrFn/IpsPFsCB4dihtl4U1phPpKUXDd6b7T3IgR7gTRXXuyzQ62etsz8e+MOa3dDMauz1b+LAUO1npGXY530+t/pVt5fBKsyUFEsvFe+880477rjj3HYN5S3wprwL+p08uyNHjnSJwQK/N5Npiwfr83CzIv6fI4DjzzhhV2CCJQw9F4YABMoQAYlYlRCKxLRH7IknnjDvNVbpjGBTQintq/3oo4/cgk377vr06RMyc6wXwIEJWALbCyeAFcJXu3Zt94MAjmQEOSbeBLZt2WKLL7zINkWwv7Rahw7W9NlnLK1y5Xh3K2nb37p1hU2Y2NVlgY7UKlSoaAceON4qVypaNMYqBFrJouRV1Y/3ZnsPcuAeYPX/m+Zv2HeLPrXFTy22ppc1tR067FDkbfVp1ccGdRiUL29B4AktWrQIeb7KHenHm+qlKwO++ioPe3D99EABrFJ0+ims7UjGITAcXm3JIg3BZ30eCeH4HoMAji/fhLbOBEsofi4OAQiUEQISqz5baDRd3nnnne2CCy4Ie0o89t35i4YK7SusQ9qPvH799lDFQKtbt67pB4NArAhIBC+/9z5b+847ZtnZBZvNyLDaJ51kDQfdnNLi14OZ+fvNtnTpmxHjb9z4VNtrz3vCHr9u3Tq3XUPeWnlIixsCXdSFArNA67jstEx7Of1u+2HohLAe4PYN29vwHsNt2eJlznMbykLV/y2sP3oJqJeCDRs2dHkYCjPvPY6m7eC2AsPhffLASLOQsz4P++jG/QAEcNwRJ+4CTLDEsefKEIBA2SEQjQc48K6aN2/uao6Gs+IuOiNJuPPuu+86r4d+wpnPLB18XDKFBoa7Bz4vWwSyV660taPftk1Tpti2jRstrXp1q9axo9Xu09sy6hftvSxbd1qy3ubkbLGfp51na9dODttQ7dqdbL82Iy09PXm85r4OsO+8RPCEZmNsVoPJti0tp8A9pVu6ndjqRLux441WOb2ySzglz20oU93lSM1nvA8OgQ4+33uPo2k7uI3AcHjviY40Cznr80hHNH7HIYDjxzbhLTPBEj4EdAACECgDBKLdA+xvSYlYDj744LB3WNx9d5Ek3Jk7d67z3kbiwVXoosS4RLv6pKQyKscSqYAOe6McAIFySiCaeVZcBBLBs+fcZcuWvR0yHFphz40a9bZWLW9NKvGr+83OzLEPnpxmS+fkz9asPcG/7/idLa0117LSt1rFnMq2R8V97aZ+l1rDWv+Wgious+DzvJAODoGOVfuxaof1eaxIFr8dBHDx2SX9mUywpB8iOggBCCQBgWizQKvLaWlpds011+TLNFrYrRQ3BDqShDvRhED7/mkP81VXXWUTJkywLl26JMEI0AUIJDeB4syz4t7R1syVtmzpKFuzZrJl52y0jPTqVqdOJ2vUuG/YPb/FvWYszpMIHvfWHPt94jLblpNboMm09Aq2Z5dG1rVvS8uomB6LS5bZNlifJ37oEMCJH4O49YAJFje0NAwBCJQzAtHWAW7Xrp316tUrIgrFDYGOpPFoQqB9e8UV5JH0h2MgUB4JFGeelUcOkdzTpvWZ9tuEpbZ09hrL3JJjlaqkW+NWdWzvAxtbtVqVImmi3B/D+jzxQ4wATvwYxK0HTLC4oaVhCECgnBFQNuZXXnnFlDU1nKnMUL9+/axixYrhDuVzCJRbAn6/peqh+jJEsbpZRWWolNirr75qzzzzTNhkc0rwpj3zmpsYBJKdAOvzxI8QAjjxYxC3HjDB4oaWhiEAgXJIQCL4008/tZ9++slU7ijYFPbctm1b69mzJ+K3HI4/txQdgXgKYB+loB4VJoAlepXtt3379qb1jvbMq0933XWX3XrrrTZp0iTr3Lmzuynt4W3ZsqWpjq0E+z33FMygrDDnGjVquON69+5tAwcOtMpB5ZlCJZJLT093e/D13XDJJZfYiSeeGB1Ijk45AqzPEz/kCODEj0HcesAEixtaGoYABMoxAdXtVcIoLaYzMzOtUqVKroyIEqtogYxBIJYEfEmWSDydsbxuSdsKJYAVQVG1alWXXK0kpm0Dxx13nE2bNq1ID/AOO+xgKmUj8as980cccYTzFj/33HO2YMECN29l06dPtzZt2ti5555r559/vhPHTz/9tKv/feONN1q9evXcSy+VQ1P9bv3+mGOOsQ8//DDfbXgBfNFFFzmhLMvJybFly5bZ22+/7c5/5JFHXH4ADAKFEWB9nvhnAwGc+DGIWw+YYHFDS8MQgAAEIACBmBAoTwJYglNeVnlwS2qRcNFefL2sWrJkiTVu3NhdUpnZJ06c6EKi/TYFldg58sgj7YEHHrDrr7/eHad+fvPNNzZnzhxr0aJFXncVCSLPsdqVsD7kkEPyPvMC+PPPP7fDDz883y2uXbvWZVTXC7S//vqLl2UlfQDK8fmszxM/uAjgxI9B3HrABIsbWhqGAAQgAAEIxIRAJEIvJheKcSPeAxyuWW0pkDjs1q2b2z7wySefFDgllDfZcymq/dq1a7u2v/zySzvzzDOdEA62jIwMq1KlihOm9913n/P4BgrgK664wp588sk8T7MSXgWGMSsCpGnTpk5AS9i+8847FiiAA8O1/bUVTl2zZk0niPv27WtXXnkl2ybCPSgp9Dnr88QPNgI48WMQtx4wweKGloYhAAEIQAACMSFQVgXw+vXrbcSIESEZrF692onN6tWru7BgbR2oX7++bd261fSZwqQDLZQAlhdXP962bc22zMUbLHv1Fvvqlwn27e+Tba9mrWzmH7Pdvl+1O3r0aFu8eLG71g033OBOlSf42WefdTWwJUxff/11O+WUU/I8wMECWPuF77//fhdGLU/wscceazNmzHAhzgqTzs7ODimADzvsMHe9sWPH2qmnnurCr7/44gv7/vvvXTj3e++9F5PnhUbKPgHW54kfQwRw4scgbj1ggsUNLQ1DAAIQgAAEikUgWOxF4unU/tWff/7ZXe+EE07IJ6bkadTf+9NOO81lTg7MTu7DfHWeT/LUqlUr18bVV1/tBGqgHX300c5D++uvv7o2ZRKUu+yyi1188cU2fPjwvMMPOuggV0ta96Psy4FCftWqVc7T2r17d+edDbZatWrl847Kc7vbbruZ2hw3bly+w3OzcmztB/Nt4w/Lzf5XX7b7s2fanFUL7aouZ9vjE1/MO173rhDmwMzUEsCNGjWynXbayYU7q6/a4+vZBAtgNSbh2qNHD1PI88iRI1378mR36NDB7fkVI3mzZd4DPGjQIJO3+Y477nCCW3uNtT9ZTJVcTxzkBccgwPo88c8AAjjxYxC3HjDB4oaWhiEAAQhAAALFIhAsgIM9nYGNfvbZZ06MnXHGGa5Ml0yeRAk5b2vWrHEeyalTp7r9qvq3F8FvvvmmE7AyiTEJU7UpMd2pUydXvkiizZtCdRUOLG/nSSed5H6tY7p27Wq33XabE7nelBROQlkCU5mQvQD+z3/+47Is67r33nuv3XTTTSbvqDIlv/XWW07k6ifQO/rggw86Qaz+a9+tN4nfFc/PsMwF6/J+9+2CqXbGqIHWvsk+dtOhl1jvVy+3No32tLvOuMGantLa2nZol08AS8Ced955pmvo+uKkMTj77LPdHuCiBLCOCdzPrJcHYq97FY/CBHBgQrOXX37ZzjrrLBs8eLDdeeedxXpmOKl8EWB9nvjxRAAnfgzi1gMmWNzQ0jAEIAABCECgWASiKR+0995728yZM23KlCnO+1iUyeOqJE8SnBKehZmE8PHHH28ffPCBjRkzxnmDvUn8SgRLwN58883u188//7zzZiok+Ly+x5r9+KLZHxOs8ZWfWO3qley3128z2/8su/3hYc77qVJA8hTL83nyySe7zMvyjvbv39+aNGliEpGzZs3K5x3V8TpP5ygLs7c178yxjVP+yncr546+0b6YN9GePv4O67hza2v31Im2944t7LNzn7dVzbJtv0u75wlgeZblPZfI//3339019FJAWaCvvfZa58EeMGCAPfbYY3l7gOUpVwi1skoHC2C9CNALAYVQv/HGGxEJYIVEK2GWGMozjEGA9XninwEEcOLHIG49KGsTrKzug4rbANIwBCAAAQiUOwKRCmB5clXWp0uXLk6ohbONGze6ZE3yxi5durTIpEvyAiuEVyWD5K30ptBeiVB5LF98cXtocZ8+fZxH+JcnTrd91nxmti3LVm7aZg0e+sdO3zfDXj2pmllaRbt95u52xxtTnMCdPXu2KfOyBKgXwHfffbftt99+rrSRQpAV8uy9o7qG9u9efvnlzgMty9mQacvun5IX9qzfLVy71A4ecbo1qtnAJlz8hqWnpdteQ3paruXa71d/Zos3/GVdhvV1Alj7ghXCLE+0RK/EeL9+/ezVV191XmZ5hbUvWC8O1DfvtS1KACssWp7rwGReoUKgAz3AupZCn3Vt3S8GgbK2Pi+PI4YALo+j+r97KmsTDAFcjh9Gbg0CEIBAChKQiNs49S/bumCd5W7NsQqV0+2vauut9ekH5QvTDYWmV69erg7tqFGjnHiLxORplMdRXmOFFIeyv//+232uPbASdBKq3hTeKwGr8GaFDuva8gYf1LKOjTs9J++4j2Zn2bGvb7YnelaxKzpVcr+//estdsc3me7f++yzj/3yyy/59sdKZMqrrORYw4YNc/uVvXdUXloJY3mbJZhl679cZOv/uzDf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7E8dYNeO5qlfAoXuqJJhfecKMEstudcCzrAO6EIiIAIiIAIiIAIiEGIEND933iESwM77wLYeaIDZhjYkGk6KT8TRj/+HuF2nfPYne4UCKNqrOqKyZfF5bLAOOHbsGLga3KRJExQpUiRYzaodERABERABERABEQhbApqfO+86CWDnfWBbDzTAbEMbMg1TBMfM2YnYXw8DiUmp+xUdhTx1iqPgrZUyVPyGDCB1RAREQAREQAREQARCiIDm5847QwLYeR/Y1gMNMNvQhlzD3BMcu+4QLu48haSLiYjKEY0cFQsgT90SPvf8htzNqEMiIAIiIAIiIAIikEkJaH7uvGMlgJ33gW090ACzDa0aFgEREAEREAEREAEREIGACWh+HjCyoJ8gARx0pKHToAZY6PhCPREBERABERABERABERABzc+dfwYkgJ33gW090ACzDa0aFgEREAEREAEREAEREIGACWh+HjCyoJ8gARx0pKHToAZY6PhCPREBERABERABERABERABzc+dfwYkgJ33gW090ACzDa0aFgEREAEREAEREAEREIGACWh+HjCyoJ8gARx0pKHToAZY6PhCPREBERABERABERABERABzc+dfwYkgJ33gW090ACzDa0aFgEREAEREAEREAEREIGACWh+HjCyoJ8gARx0pKHToAZY6PhCPREBERABERABERABERABzc+dfwYkgJ33gW090ACzDa0aFgEREAEREAEREAEREIGACWh+HjCyoJ8gAXyFSLdt24ZvvvkGCxcuxF9//YXDhw+jUKFCaNy4MZ544gk0bdo0zSvs378fgwYNwvz583HixAmUK1cO99xzD1588UXkzJnzCnsGaIBdMUI1IAIiIAIiIAIiIAIiIAJBI6D5edBQprshCeB0o7t8YpkyZXDgwAHkz58fDRo0MOJ369at2LJlC6KiojBq1Cg8/vjjqa6yY8cONGrUCEePHkX16tVRrVo1rFu3Djt37jS/X7JkCXLkyHFFvdMAuyJ8OlkEREAEREAEREAEREAEgkpA8/Og4kxXYxLA6cL2z0lt2rRBz549ceeddyJ79uzJH4wfPx79+vVDdHQ0Nm3aZASuqzVv3hw//fQTHn30UYwZM8Z8lJCQgC5dumDWrFlmZXjIkCFX1DsNsCvCp5NFQAREQAREQAREQAREIKgEND8PKs50NSYBnC5s/p3U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8mTz8tzVeN7YsWPBcjzcEtWwYUMTIcYXgbLgEvA3rwtX3l1LJrIX8lNwfZFZW5MAzqye1X2JgAiIgAiIgAiIgAiIgAiIgAikICABrAdCBERABERABERABERABERABEQgIghIAEeEm3WTIiACIiACIiACIiACIiACIiACEsB6BkRABERABERABERABERABERABCKCgARwRLhZNykCIiACIiACIiACIiACIiACIiABrGdABERABERABERABERABERABEQgIghIAEeEm3WTIiACIiACIiACIiACIiACIiACEsB6BkRABERABERABERABERABERABCKCgARwRLhZNykCIiACIiACIiACIiACIiACIiABrGdABERABERABERABERABERABEQgIghIAEeEm3WTIiACIiACIiACIiACIiACIiACEsB6BkRABERABERABERABERABERABCKCgARwRLhZNykCIiACIiACIiACIiACIiACIiABrGdABERABERABERABERABERABEQgIghIAEeEm3WTIiACIiACIiACIiACIiACIiACEsB6BkRABERABERABERABERABERABCKCgARwRLhZNykCIiACIiACIiACIiACIiACIiABrGdABERABERABERABERABERABEQgIghIAEeEm3WTIiACIiACIiACIiACIiACIiACEsB6BkRABERABERABERABERABERABCKCgARwRLhZNykCIiACIiACIiACIiACIiACIiABrGdABERABERABERABERABERABEQgIghIAEeEm3WTIiACIiACIiACIiACIiACIiACEsB6BkRABERABERABERABERABERABCKCgARwRLhZNykCIiACIiACIiACIiACIiACIiABrGdABERABERABERABERABERABEQgIghIAEeEm3WTIiACIiACIiACIiACIiACIiACEsB6BkRABERABERABERABERABERABCKCgARwRLhZNykCIiACIiACIiACIiACIiACIiABrGdABERABERABERABERABERABEQgIghIAEeEm3WTIiACIiACIiACIiACIiACIiACEsB6BkRABERABERABERABERABERABCKCgARwRLhZNykCIiACIiACIiACIiACIiACIiABrGdABERABERABERABERABERABEQgIghIAEeEm3WTIiACIiACIiACIiACIiACIiACEsB6BkRABERABERABERABERABERABCKCgARwRLhZNykCIiACIiACIiACIiACIiACIiABrGdABERABERABERABERABERABEQgIghIAEeEm3WTIiACIiACIiACIiACIiACIiACEsB6BkRABERABERABERABERABERABCKCgARwRLhZNykCIiACIiACIiACIiACIiACIiABrGdABERABERABERABERABERABEQgIghIAEeEm3WTIiACIiACIiACIiACIiACIiACEsB6BkRABERABERABERABERABERABCKCgARwRLhZNykCIiACIiACIiACIiACIiACIiABrGdABERABERABERABERABERABEQgIghIAEeEm3WTIiACIiACIiACIiACIiACIiACEsB6BkRABERABERABERABERABERABCKCgARwRLhZNykCIiACIiACIiACIiACIiACIvB/fmScut98ozMAAAAASUVORK5CYII=" width="640">



```
import plotly.express as px

# https://stackoverflow.com/a/46078114
df = pd.DataFrame(embedding_dict).T.rename_axis('Word').add_prefix('Coord').reset_index()

fig = px.scatter(df, x="Coord0", y="Coord1", text="Word")

# fig.update_traces(textposition='top center')

# fig.update_layout(
#     height=800,
#     title_text='GDP and Life Expectancy (Americas, 2007)'
# )
config = dict({'scrollZoom': True})
fig.show(config=config)

```


<div>                            <div id="7f418e2e-7a72-49ae-9499-ab0325968431" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("7f418e2e-7a72-49ae-9499-ab0325968431")) {                    Plotly.newPlot(                        "7f418e2e-7a72-49ae-9499-ab0325968431",                        [{"hovertemplate": "Coord0=%{x}<br>Coord1=%{y}<br>Word=%{text}<extra></extra>", "legendgroup": "", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "markers+text", "name": "", "showlegend": false, "text": ["", "\u05d5\u05b7\u05ea\u05bc\u05b5\u05e9\u05c1\u05b0\u05ea\u05bc\u05b0", "\u05de\u05b5\u05e8\u05b5\u05d0\u05e9\u05c1\u05b4\u05d9\u05ea", "\u05ea\u05b0\u05d1\u05b8\u05e8\u05b2\u05db\u05d5\u05bc", "\u05d0\u05b1\u05d5\u05b4\u05d9", "\u05d4\u05b7\u05d6\u05bc\u05b9\u05d0\u05ea", "\u05d0\u05d5\u05b9", "\u05e9\u05b0\u05c2\u05e8\u05b5\u05e4\u05b8\u05d4", "\u05ea\u05b7\u05bc\u05d7\u05b7\u05ea", "\u05d5\u05b7\u05d4\u05b2\u05e8\u05b5\u05de\u05b9\u05ea\u05b8", "\u05e8\u05b9\u05d0\u05e9\u05c1\u05d5\u05b9", "\u05de\u05b8\u05d0\u05b5\u05df", "\u05d4\u05b7\u05e1\u05bc\u05b7\u05dc", "\u05e0\u05b6\u05e1\u05b6\u05da\u05b0", "\u05db\u05b0\u05bc\u05e0\u05b7\u05e2\u05b7\u05df", "\u05d6\u05b8\u05d4\u05b8\u05d1", "\u05d9\u05b8\u05d1\u05b4\u05d9\u05d0\u05d5\u05bc", "\u05d1\u05bc\u05b0\u05ea\u05d5\u05b9\u05db\u05b8\u05bd\u05dd", "\u05d1\u05bc\u05b0\u05d0\u05b6\u05e6\u05b0\u05d1\u05bc\u05b8\u05e2\u05d5\u05b9", "\u05d1\u05bc\u05b0\u05e6\u05b5\u05d0\u05ea\u05d5\u05b9", "\u05d5\u05bc\u05dc\u05b0\u05d0\u05b4\u05de\u05bc\u05d5\u05b9", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05bd\u05d7\u05b7\u05e8", "\u05e8\u05b8\u05d2\u05b7\u05dd", "\u05dc\u05b0\u05d8\u05bd\u05b7\u05d4\u05b2\u05e8\u05b8\u05dd", "\u05dc\u05b0\u05d1\u05b8\u05d3\u05b8\u05d3", "\u05dc\u05b5\u05bd\u05d0\u05dc\u05b9\u05d4\u05b5\u05d9\u05db\u05b6\u05bd\u05dd", "\u05d7\u05b5\u05de\u05b8\u05d4", "\u05d4\u05b7\u05de\u05bc\u05b4\u05e9\u05c1\u05b0\u05db\u05bc\u05b8\u05bd\u05df", "\u05dc\u05b4\u05e1\u05b0\u05e4\u05bc\u05d5\u05b9\u05ea", "\u05e0\u05b8\u05d1\u05b8\u05d0", "\u05d7\u05b9\u05e7", "\u05dc\u05b0\u05d3\u05b4\u05d2\u05b0\u05dc\u05b5\u05d9\u05d4\u05b6\u05bd\u05dd", "\u05e8\u05bd\u05d5\u05bc\u05d7\u05b7", "\u05d6\u05b6\u05bd\u05d4", "\u05d5\u05b0\u05d0\u05b5\u05bd\u05dc\u05b0\u05db\u05b8\u05d4", "\u05d4\u05b7\u05e6\u05bc\u05b9\u05e8\u05b5\u05e8", "\u05dc\u05b7\u05d2\u05bc\u05bb\u05dc\u05b0\u05d2\u05bc\u05b9\u05dc\u05b6\u05ea", "\u05d5\u05b4\u05d9\u05d1\u05b5\u05e9\u05c1\u05b4\u05d9\u05dd", "\u05ea\u05b0\u05e8\u05d5\u05bc\u05de\u05b8\u05d4", "\u05e9\u05c1\u05d5\u05bc\u05ea\u05b6\u05dc\u05b7\u05d7", "\u05d4\u05b7\u05d2\u05bc\u05b0\u05d1\u05bb\u05dc", "\u05d4\u05b5\u05e0\u05bc\u05b8\u05d4", "\u05d0\u05b2\u05d1\u05d5\u05b9\u05ea\u05b8\u05bd\u05dd", "\u05d5\u05b0\u05d4\u05b4\u05bd\u05ea\u05b0\u05d9\u05b7\u05e6\u05bc\u05b0\u05d1\u05d5\u05bc", "\u05d9\u05b4\u05e4\u05b0\u05e8\u05b0\u05e9\u05c2\u05d5\u05bc", "\u05d0\u05b5\u05d9\u05d1\u05b8\u05d4", "\u05ea\u05b5\u05bc\u05d9\u05de\u05b8\u05df", "\u05e9\u05b8\u05c1\u05e0\u05b4\u05d9", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05d5\u05bc\u05e4\u05b8\u05de\u05b4\u05d9", "\u05d4\u05b7\u05d9\u05bc\u05b8\u05de\u05b4\u05d9\u05e0\u05b4\u05d9", "\u05d0\u05b2\u05d7\u05b4\u05d9\u05d4\u05d5\u05bc\u05d3", "\u05d2\u05b7\u05bc\u05d9\u05b0\u05d0", "\u05de\u05b7\u05e2\u05b7\u05dc", "\u05dc\u05b8\u05bd\u05d0\u05b1\u05de\u05b9\u05e8\u05b4\u05bd\u05d9", "\u05de\u05b8\u05d7\u05b8\u05e8", "\u05d0\u05b8\u05e8\u05bd\u05d5\u05bc\u05e8", "\u05d4\u05b7\u05e4\u05bc\u05b0\u05d3\u05bb\u05d9\u05b4\u05dd", "\u05d4\u05b2\u05e8\u05b5\u05e2\u05b9\u05ea\u05b4\u05d9", "\u05d9\u05b0\u05d1\u05b5\u05e9\u05c1\u05b8\u05d4", "\u05db\u05b0\u05bc\u05e8\u05d5\u05bc\u05d1", "\u05ea\u05b8\u05bc\u05e7\u05b7\u05e2", "\u05dc\u05b4\u05e0\u05b0\u05d3\u05b8\u05e8\u05b6\u05d9\u05d4\u05b8", "\u05de\u05b4\u05e9\u05b0\u05c1\u05e4\u05b8\u05bc\u05d8", "\u05db\u05bc\u05b0\u05ea\u05d5\u05b9\u05e2\u05b2\u05e4\u05b9\u05ea", "\u05d5\u05b0\u05e0\u05b4\u05d2\u05b0\u05e8\u05b0\u05e2\u05b8\u05d4", "\u05ea\u05b4\u05e9\u05c2\u05b0\u05ea\u05bc\u05b8\u05e8\u05b5\u05e8", "\u05dc\u05b7\u05e9\u05c1\u05bc\u05b8\u05e0\u05b8\u05d4", "\u05dc\u05b0\u05e6\u05b4\u05d9\u05e6\u05b4\u05ea", "\u05e2\u05b7\u05de\u05bc\u05b4\u05d9\u05d4\u05bd\u05d5\u05bc\u05d3", "\u05d4\u05b2\u05e8\u05b8\u05e4\u05b6\u05d4", "\u05de\u05d5\u05b9\u05ea", "\u05de\u05b4\u05de\u05bc\u05bb\u05dc\u05b4\u05bd\u05d9", "\u05de\u05b4\u05db\u05b0\u05e1\u05b6\u05d4", "\u05d2\u05bc\u05b8\u05d3\u05b5\u05e8", "\u05e2\u05b6\u05dc\u05b0\u05d9\u05d5\u05b9\u05df", "\u05e2\u05b6\u05e9\u05b6\u05c2\u05e8", "\u05de\u05b8\u05e9\u05c1\u05b7\u05d7", "\u05d5\u05b0\u05d3\u05b8\u05ea\u05b8\u05df", "\u05d4\u05b7\u05de\u05bc\u05b8\u05d0\u05d5\u05b9\u05e8", "\u05de\u05b6\u05dc\u05b7\u05d7", "\u05d5\u05bc\u05e8\u05b0\u05e7\u05b4\u05d9\u05e7\u05b5\u05d9", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05b9\u05d2\u05b6\u05d2\u05b6\u05ea", "\u05e7\u05b0\u05e0\u05b8\u05ea", "\u05d4\u05b7\u05bd\u05de\u05bc\u05b6\u05d7\u05b1\u05e6\u05b8\u05d4", "\u05d5\u05bc\u05de\u05b8\u05e1\u05b7\u05da\u05b0", "\u05d5\u05bc\u05e6\u05b0\u05d1\u05b8\u05d0\u05d5\u05b9", "\u05d4\u05b7\u05e7\u05bc\u05b4\u05e9\u05c1\u05bc\u05bb\u05d0\u05b4\u05d9\u05dd", "\u05d5\u05b7\u05d0\u05b2\u05d2\u05b8\u05e8\u05b0\u05e9\u05c1\u05b6\u05e0\u05bc\u05d5\u05bc", "\u05d4\u05b4\u05db\u05bc\u05b8\u05d4\u05d5\u05bc", "\u05d5\u05bc\u05e6\u05b0\u05dc\u05b8\u05e4\u05b0\u05d7\u05b8\u05d3", "\u05de\u05b4\u05e7\u05b0\u05e0\u05b5\u05d4\u05b6\u05dd", "\u05d4\u05b4\u05e0\u05bc\u05b8\u05e7\u05b4\u05d9", "\u05e9\u05c2\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05d5\u05b4\u05bd\u05d9\u05e9\u05c1\u05b7\u05dc\u05bc\u05b0\u05d7\u05d5\u05bc", "\u05e2\u05b2\u05e9\u05b4\u05c2\u05d9\u05e8\u05b4\u05d9", "\u05d9\u05b8\u05e1\u05b8\u05bd\u05e4\u05d5\u05bc", "\u05d5\u05b0\u05ea\u05b7\u05dc\u05b0\u05de\u05b7\u05d9", "\u05d5\u05b0\u05bd\u05d4\u05b7\u05e2\u05b2\u05de\u05b7\u05d3\u05b0\u05ea\u05bc\u05b8", "\u05e4\u05b8\u05bc\u05e0\u05b8\u05d4", "\u05dc\u05b7\u05d0\u05b2\u05e8\u05d5\u05b9\u05d3", "\u05e9\u05c1\u05b8\u05de\u05b8\u05e8\u05d5\u05bc", "\u05dc\u05b0\u05ea\u05b4\u05e9\u05c1\u05b0\u05e2\u05b7\u05ea", "\u05d1\u05bc\u05b4\u05d2\u05b0\u05d3\u05b5\u05d9\u05d4\u05b6\u05bd\u05dd", "\u05d4\u05b7\u05e0\u05bc\u05b8\u05e9\u05c1\u05d5\u05bc\u05da\u05b0", "\u05d1\u05bc\u05b0\u05de\u05b4\u05e6\u05b0\u05e8\u05b7\u05d9\u05b4\u05dd", "\u05e9\u05c1\u05b0\u05de\u05d5\u05bc\u05d0\u05b5\u05dc", "\u05de\u05b5\u05bd\u05d0\u05b7\u05e8\u05b0\u05e0\u05b9\u05df", "\u05db\u05b0\u05d1\u05b8\u05e9\u05c2\u05b4\u05d9\u05dd", "\u05d5\u05b0\u05d9\u05b4\u05de\u05bc\u05b8\u05dc\u05b5\u05d0", "\u05e9\u05c1\u05b7\u05d3\u05bc\u05b8\u05bd\u05d9", "\u05d9\u05b8\u05d0\u05b5\u05e8", "\u05ea\u05bc\u05b4\u05e7\u05b0\u05e6\u05b9\u05bd\u05e3", "\u05d0\u05b7\u05e8\u05b0\u05d1\u05bc\u05b7\u05e2\u05b7\u05ea", "\u05d8\u05b7\u05e2\u05b7\u05dd", "\u05d5\u05b0\u05e7\u05b4\u05d3\u05bc\u05b7\u05e9\u05c1", "\u05e2\u05d5\u05b9\u05dc\u05b8\u05dd", "\u05e0\u05b8\u05e1\u05b8\u05d4", "\u05d4\u05b7\u05e4\u05bc\u05b7\u05dc\u05bc\u05bb\u05d0\u05b4\u05bd\u05d9", "\u05d7\u05b7\u05e0\u05bc\u05b4\u05d9\u05d0\u05b5\u05dc", "\u05d5\u05b0\u05e2\u05b7\u05de\u05bc\u05d5\u05bc\u05d3\u05b5\u05d9", "\u05d4\u05b7\u05d9\u05bc\u05b8\u05e9\u05c1\u05d5\u05bc\u05d1\u05b4\u05d9", "\u05dc\u05b4\u05d2\u05b0\u05d5\u05ba\u05bd\u05e2\u05b7", "\u05d5\u05b0\u05e7\u05b8\u05de\u05b7\u05e5", "\u05ea\u05b4\u05e9\u05c2\u05b0\u05d8\u05b6\u05d4", "\u05d4\u05b7\u05de\u05bc\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05d7\u05b2\u05de\u05b4\u05d9\u05e9\u05b4\u05c1\u05d9", "\u05db\u05bc\u05b0\u05d4\u05bb\u05e0\u05bc\u05b7\u05ea", "\u05d9\u05d5\u05b9\u05e1\u05b5\u05bd\u05e3", "\u05e2\u05bb\u05d6\u05b4\u05bc\u05d9\u05d0\u05b5\u05dc", "\u05d4\u05b6\u05e8\u05b0\u05d0\u05b8\u05d4", "\u05d0\u05b6\u05dc\u05b6\u05e3", "\u05e9\u05b6\u05c1\u05d1\u05b7\u05e2", "\u05d7\u05b8\u05e8\u05b8\u05d4", "\u05e6\u05b7\u05e8", "\u05d4\u05b7\u05e7\u05bc\u05b6\u05e6\u05b6\u05e3", "\u05e9\u05b5\u05c1\u05d1\u05b6\u05d8", "\u05e6\u05b8\u05d1", "\u05e4\u05bc\u05b4\u05d2\u05b0\u05e8\u05b5\u05d9\u05db\u05b6\u05dd", "\u05d5\u05b0\u05e0\u05b7\u05e2\u05b2\u05de\u05b8\u05df", "\u05dc\u05b0\u05de\u05b7\u05d8\u05bc\u05b6\u05d4", "\u05ea\u05b0\u05d4\u05b4\u05d9", "\u05de\u05b0\u05e6\u05b9\u05e8\u05b7\u05e2\u05b7\u05ea", "\u05d5\u05b0\u05d4\u05b7\u05e9\u05c1\u05bc\u05b4\u05d1\u05b0\u05e2\u05b4\u05d9\u05dd", "\u05e9\u05c1\u05b9\u05bd\u05de\u05b0\u05e8\u05b5\u05d9", "\u05e9\u05c1\u05b8\u05de\u05b0\u05e2\u05bd\u05d5\u05b9", "\u05d4\u05b7\u05d6\u05bc\u05b7\u05e8\u05b0\u05d7\u05b4\u05bd\u05d9", "\u05dc\u05b0\u05d3\u05b9\u05e8\u05b9\u05ea\u05b5\u05d9\u05db\u05b6\u05dd", "\u05e6\u05b4\u05df", "\u05d4\u05b7\u05ea\u05bc\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05ea\u05bc\u05b4\u05db\u05b0\u05ea\u05bc\u05b9\u05d1", "\u05dc\u05b0\u05de\u05b8\u05e9\u05c1\u05b0\u05d7\u05b8\u05d4", "\u05d7\u05b6\u05dc\u05b6\u05d1", "\u05d5\u05b7\u05d9\u05bc\u05b9\u05e6\u05b5\u05bd\u05d0", "\u05d5\u05b0\u05db\u05b4\u05ea\u05b0\u05d1\u05d5\u05bc\u05d0\u05b7\u05ea", "\u05e2\u05b4\u05bd\u05d9\u05e8", "\u05d4\u05b7\u05de\u05bc\u05b4\u05d3\u05b0\u05d9\u05b8\u05e0\u05b4\u05d9\u05dd", "\u05de\u05b9\u05e9\u05b6\u05c1\u05d4", "\u05de\u05b4\u05de\u05bc\u05b7\u05e7\u05b0\u05d4\u05b5\u05dc\u05b9\u05ea", "\u05d5\u05b7\u05bd\u05d9\u05bc\u05b7\u05e2\u05b7\u05de\u05b0\u05d3\u05d5\u05bc", "\u05d5\u05b0\u05d4\u05b7\u05d6\u05bc\u05b8\u05e8", "\u05de\u05b4\u05de\u05bc\u05b4\u05ea\u05b0\u05e7\u05b8\u05d4", "\u05e0\u05b9\u05e9\u05c2\u05b5\u05d0", "\u05d5\u05b0\u05e0\u05d5\u05b9\u05e9\u05c1\u05b7\u05e2\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05d5\u05b8\u05d4\u05b5\u05dd", "\u05dc\u05b0\u05d0\u05b4\u05e9\u05c1\u05bc\u05b7\u05d9", "\u05d1\u05bc\u05b8\u05d4\u05bc", "\u05e2\u05b8\u05dc\u05b9\u05d4", "\u05d3\u05b8\u05bc\u05d2\u05b8\u05df", "\u05d4\u05b7\u05e0\u05bc\u05b5\u05e8\u05b9\u05ea", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05d8\u05bc\u05b9\u05e9\u05c1", "\u05ea\u05d5\u05b9\u05e6\u05b0\u05d0\u05b9\u05ea\u05b8\u05d9\u05d5", "\u05d9\u05b4\u05e4\u05b0\u05e7\u05b9\u05d3", "\u05d6\u05b8\u05e7\u05b5\u05df", "\u05d1\u05bc\u05b6\u05df", "\u05e8\u05b0\u05d7\u05b9\u05d1", "\u05e9\u05b0\u05c1\u05dc\u05b4\u05d9\u05e9\u05b4\u05c1\u05d9", "\u05d5\u05b4\u05bd\u05d9\u05d7\u05bb\u05e0\u05bc\u05b6\u05bd\u05da\u05bc\u05b8", "\u05e2\u05b8\u05dc\u05b8\u05d4", "\u05d9\u05b4\u05e9\u05c1\u05bc\u05b8\u05d7\u05b5\u05d8", "\u05d7\u05b8\u05e8\u05b7\u05e9\u05c1", "\u05d5\u05bc\u05ea\u05b0\u05db\u05b7\u05dc", "\u05d1\u05bc\u05b7\u05e9\u05c1\u05bc\u05b0\u05d1\u05b4\u05d9\u05ea", "\u05e6\u05b0\u05e4\u05d5\u05b9\u05df", "\u05d5\u05b0\u05db\u05d5\u05bc\u05de\u05b8\u05d6", "\u05d4\u05b7\u05de\u05bc\u05b4\u05d6\u05b0\u05e8\u05b8\u05e7\u05b9\u05ea", "\u05dc\u05b0\u05d2\u05d5\u05bc\u05e0\u05b4\u05d9", "\u05d0\u05b7\u05d7\u05b4\u05d9\u05dd", "\u05d4\u05b7\u05d9\u05bc\u05b8\u05bd\u05de\u05bc\u05b8\u05d4", "\u05e2\u05b8\u05d2\u05b4\u05d9\u05dc", "\u05d4\u05b6\u05e2\u05b8\u05e0\u05b8\u05df", "\u05d4\u05b7\u05de\u05bc\u05b4\u05d6\u05b0\u05dc\u05b8\u05d2\u05b9\u05ea", "\u05d4\u05b7\u05ea\u05bc\u05b0\u05d0\u05b5\u05e0\u05b4\u05bd\u05d9\u05dd", "\u05e4\u05bc\u05b4\u05d8\u05b0\u05e8\u05b7\u05ea", "\u05e7\u05b3\u05d3\u05b8\u05e9\u05c1\u05b8\u05d9\u05d5", "\u05de\u05b4\u05e7\u05b0\u05e0\u05b6\u05d4", "\u05e9\u05c1\u05b8\u05de\u05bc\u05b8\u05d4", "\u05e0\u05b4\u05d6\u05b0\u05e8\u05d5\u05b9", "\u05d4\u05b7\u05d7\u05b2\u05e0\u05b9\u05db\u05b4\u05d9", "\u05d4\u05b7\u05e0\u05bc\u05b8\u05e1\u05b6\u05da\u05b0", "\u05d5\u05b0\u05e9\u05c1\u05b8\u05de\u05b0\u05e2\u05d5\u05bc", "\u05d5\u05b7\u05e0\u05bc\u05b7\u05e9\u05c1\u05bc\u05b4\u05d9\u05dd", "\u05d0\u05d5\u05bc\u05e8\u05b4\u05d9\u05dd", "\u05de\u05b4\u05dc\u05bc\u05b4\u05d1\u05b0\u05e0\u05b8\u05d4", "\u05d5\u05b0\u05d4\u05b7\u05de\u05bc\u05b8\u05df", "\u05dc\u05bd\u05d5\u05b9", "\u05d2\u05bc\u05d5\u05b9\u05e8\u05b8\u05dc", "\u05dc\u05b0\u05e2\u05b9\u05dc\u05b8\u05d4", "\u05d4\u05b7\u05e7\u05bc\u05b0\u05e0\u05b4\u05d6\u05bc\u05b4\u05d9", "\u05d5\u05b7\u05d9\u05bc\u05b9\u05bd\u05d0\u05de\u05b0\u05e8\u05d5\u05bc", "\u05e4\u05bc\u05b0\u05e7\u05bb\u05d3\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05e2\u05b8\u05d1\u05b7\u05e8", "\u05e0\u05b6\u05e2\u05b1\u05e6\u05b8\u05bd\u05e8\u05b8\u05d4", "\u05d5\u05bc\u05de\u05b4\u05dc\u05b0\u05db\u05bc\u05b8\u05d4", "\u05d5\u05b7\u05d9\u05bc\u05b8\u05d2\u05b8\u05d6", "\u05dc\u05b8\u05d4", "\u05d4\u05b7\u05ea\u05bc\u05b7\u05d7\u05b7\u05e9\u05c1", "\u05d5\u05b0\u05d7\u05b5\u05e4\u05b6\u05e8", "\u05db\u05bc\u05b6\u05ea\u05b6\u05e3", "\u05d5\u05bc\u05e8\u05b0\u05d1\u05b4\u05d9\u05e2\u05b4\u05ea", "\u05d0\u05b6\u05e9\u05c1\u05b0\u05db\u05bc\u05b9\u05dc", "\u05d4\u05b7\u05bd\u05de\u05b0\u05e7\u05b7\u05e0\u05bc\u05b5\u05d0", "\u05d5\u05bc\u05de\u05b4\u05d2\u05b0\u05e8\u05b0\u05e9\u05c1\u05b5\u05d9", "\u05d5\u05b7\u05ea\u05bc\u05b8\u05e0\u05b7\u05d7", "\u05d7\u05b7\u05d8\u05b8\u05bc\u05d0", "\u05dc\u05b0\u05de\u05b4\u05e9\u05c1\u05b0\u05e4\u05bc\u05b0\u05d7\u05b9\u05bd\u05ea\u05b5\u05d9\u05db\u05b6\u05dd", "\u05d5\u05b0\u05dc\u05b7\u05ea\u05bc\u05d5\u05b9\u05e9\u05c1\u05b8\u05d1", "\u05de\u05b4\u05e0\u05b0\u05d7\u05b8\u05d4", "\u05d9\u05b4\u05ea\u05bc\u05b7\u05de\u05bc\u05d5\u05bc", "\u05d5\u05b0\u05e8\u05b8\u05e6\u05b7\u05d7", "\u05d9\u05b4\u05bd\u05e1\u05b0\u05dc\u05b7\u05d7", "\u05d5\u05b0\u05e9\u05c1\u05b4\u05d1\u05b0\u05e2\u05b7\u05ea", "\u05d9\u05b4\u05e0\u05b0\u05d7\u05b8\u05bd\u05dc\u05d5\u05bc", "\u05dc\u05b0\u05e9\u05c1\u05b7\u05db\u05bc\u05b5\u05df", "\u05d5\u05bc\u05e7\u05b0\u05e1\u05b8\u05de\u05b4\u05d9\u05dd", "\u05e4\u05b8\u05bc\u05dc\u05b8\u05d0", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e1\u05b0\u05de\u05b9\u05da\u05b0", "\u05d5\u05b0\u05d4\u05b7\u05d9\u05bc\u05b9\u05e6\u05b5\u05d0", "\u05de\u05b7\u05d8\u05bc\u05b5\u05bd\u05d4\u05d5\u05bc", "\u05d5\u05bc\u05d3\u05b0\u05d1\u05b8\u05bd\u05e9\u05c1", "\u05d4\u05b7\u05e7\u05bc\u05b0\u05e0\u05b8\u05d0\u05b9\u05ea", "\u05de\u05b4\u05d1\u05b0\u05e6\u05b8\u05e8", "\u05d1\u05bc\u05b0\u05d7\u05b6\u05d8\u05b0\u05d0\u05b8\u05d4", "\u05dc\u05b0\u05d1\u05b4\u05dc\u05b0\u05e2\u05b8\u05dd", "\u05db\u05b8\u05bc\u05d1\u05b7\u05e1", "\u05e0\u05b8\u05e1\u05b7\u05e2", "\u05d5\u05b0\u05d4\u05b7\u05e0\u05bc\u05b6\u05e4\u05b6\u05e9\u05c1", "\u05de\u05b7\u05e2\u05b2\u05e9\u05b6\u05c2\u05d4", "\u05dc\u05b0\u05e9\u05c1\u05b6\u05d1\u05b6\u05ea", "\u05d0\u05b7\u05e8\u05b0\u05e6\u05b0\u05db\u05b6\u05dd", "\u05de\u05b4\u05d3\u05bc\u05b8\u05e4\u05b0\u05e7\u05b8\u05d4", "\u05e6\u05b8\u05e8\u05d5\u05b9\u05e8", "\u05d5\u05b0\u05d9\u05b8\u05ea\u05bb\u05e8\u05d5\u05bc", "\u05e0\u05b9\u05d1\u05b7\u05d7", "\u05d1\u05bc\u05b0\u05e4\u05b4\u05d2\u05b0\u05e2\u05d5\u05b9", "\u05d5\u05bc\u05d1\u05b8\u05dc\u05b8\u05e7", "\u05d0\u05b6\u05d3\u05b6\u05df", "\u05d1\u05bc\u05b0\u05e2\u05b7\u05dd", "\u05d5\u05b0\u05e2\u05b7\u05d1\u05b0\u05d3\u05bc\u05b4\u05d9", "\u05d4\u05b7\u05de\u05bc\u05b4\u05d3\u05b0\u05d1\u05bc\u05b8\u05e8", "\u05d5\u05b8\u05d0\u05b6\u05ea\u05bc\u05b0\u05e0\u05b8\u05d4", "\u05d8\u05bc\u05b9\u05d1\u05d5\u05bc", "\u05d2\u05bc\u05b0\u05d1\u05d5\u05bc\u05dc\u05b6\u05bd\u05da\u05b8", "\u05ea\u05bc\u05b5\u05e9\u05c1\u05b0\u05d1\u05d5\u05bc", "\u05e1\u05b0\u05d1\u05b4\u05d9\u05d1\u05b9\u05ea", "\u05e0\u05b0\u05e7\u05b4\u05d9\u05bc\u05b4\u05d9\u05dd", "\u05d4\u05b8\u05d0\u05b6\u05d6\u05b0\u05e8\u05b8\u05d7", "\u05e4\u05b9\u05e8\u05b7\u05e9\u05c1", "\u05d9\u05b8\u05d0\u05b4\u05bd\u05d9\u05e8", "\u05dc\u05b0\u05e6\u05b5\u05d0\u05ea", "\u05d5\u05b0\u05d4\u05b7\u05bd\u05e2\u05b2\u05d1\u05b7\u05e8\u05b0\u05ea\u05bc\u05b8", "\u05d7\u05b5\u05e1\u05b5\u05d3", "\u05e0\u05b6\u05d2\u05b6\u05e3", "\u05d4\u05d5\u05b9\u05dc\u05b4\u05d3", "\u05d4\u05bd\u05b7\u05d7\u05b7\u05d9\u05bc\u05b4\u05d9\u05dd", "\u05e7\u05b5\u05d9\u05e0\u05b4\u05d9", "\u05e2\u05b2\u05e9\u05c2\u05b4\u05d9\u05e8\u05b4\u05ea", "\u05dc\u05b4\u05e8\u05b0\u05d0\u05d5\u05bc\u05d1\u05b5\u05df", "\u05d4\u05b7\u05de\u05bc\u05b0\u05e0\u05b9\u05e8\u05b8\u05d4", "\u05d0\u05b2\u05e9\u05c1\u05b8\u05de\u05b8\u05dd", "\u05d5\u05b7\u05bd\u05d9\u05bc\u05b4\u05dc\u05b0\u05db\u05bc\u05b0\u05d3\u05d5\u05bc", "\u05e2\u05b2\u05d1\u05b9\u05d3\u05b8\u05ea\u05d5\u05b9", "\u05ea\u05bc\u05d5\u05b9\u05dc\u05b0\u05d3\u05b8\u05d4", "\u05d4\u05b7\u05de\u05bc\u05b0\u05e0\u05d5\u05b9\u05e8\u05b8\u05d4", "\u05d4\u05b2\u05e8\u05b7\u05d2\u05b0\u05ea\u05bc\u05b4\u05bd\u05d9\u05da\u05b0", "\u05e2\u05b8\u05e8", "\u05d4\u05b7\u05de\u05b0\u05d0\u05b8\u05bd\u05e8\u05b2\u05e8\u05b4\u05bd\u05d9\u05dd", "\u05d1\u05bc\u05b4\u05e0\u05b0\u05e2\u05bb\u05e8\u05b6\u05bd\u05d9\u05d4\u05b8", "\u05db\u05bc\u05b0\u05de\u05d5\u05b9\u05e9\u05c1", "\u05de\u05b0\u05d1\u05b7\u05e7\u05bc\u05b5\u05e9\u05c1", "\u05e7\u05b6\u05e1\u05b6\u05dd", "\u05d9\u05d5\u05bc\u05de\u05b8\u05bd\u05ea", "\u05de\u05b7\u05d6\u05b0\u05db\u05bc\u05b6\u05e8\u05b6\u05ea", "\u05e9\u05c1\u05b4\u05de\u05b0\u05e2\u05b2\u05da\u05b8", "\u05e0\u05b7\u05d7\u05b7\u05e9\u05c1", "\u05de\u05b0\u05d0\u05b7\u05e1\u05bc\u05b5\u05e3", "\u05ea\u05bc\u05b4\u05d6\u05b0\u05e8\u05b9\u05e7", "\u05ea\u05bc\u05b5\u05d7\u05b8\u05bd\u05dc\u05b0\u05e6\u05d5\u05bc", "\u05d7\u05b8\u05e6\u05b8\u05d4", "\u05e2\u05b7\u05de\u05bc\u05b5\u05da\u05b0", "\u05d0\u05b8\u05e1\u05b7\u05e3", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05e0\u05bc\u05b4\u05d9\u05d7\u05d5\u05bc", "\u05d3\u05b8\u05bc\u05df", "\u05d8\u05b8\u05d4\u05b5\u05e8", "\u05d0\u05b2\u05db\u05b7\u05d1\u05bc\u05b6\u05d3\u05b0\u05da\u05b8", "\u05d4\u05b4\u05e8\u05b0\u05d2\u05d5\u05bc", "\u05d1\u05b0\u05d7\u05b5\u05d9\u05e7\u05b6\u05da\u05b8", "\u05e7\u05bd\u05b7\u05e2\u05b2\u05e8\u05b7\u05ea", "\u05d5\u05b0\u05d4\u05b7\u05de\u05bc\u05b8\u05d0\u05ea\u05b8\u05d9\u05b4\u05dd", "\u05e2\u05b7\u05de\u05bc\u05d5\u05bc\u05d3", "\u05de\u05b7\u05ea\u05b8\u05bc\u05e0\u05b8\u05d4", "\u05d5\u05b0\u05d4\u05b5\u05e4\u05b5\u05e8", "\u05de\u05b5\u05bd\u05d0\u05b7\u05e4\u05bc\u05b0\u05db\u05b6\u05dd", "\u05d0\u05b7\u05e0\u05b0\u05e9\u05c1\u05b5\u05d9", "\u05d5\u05b0\u05d9\u05b9\u05d3\u05b7\u05e2", "\u05d5\u05bc\u05de\u05b4\u05de\u05bc\u05b7\u05ea\u05bc\u05b8\u05e0\u05b8\u05d4", "\u05de\u05b8\u05ea\u05b7\u05d9", "\u05d1\u05b4\u05d2\u05b0\u05d3\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05dc\u05b0\u05e7\u05b9\u05dc\u05b8\u05dd", "\u05e2\u05b7\u05d3", "\u05e6\u05b4\u05bd\u05df", "\u05d4\u05b7\u05e6\u05bc\u05b9\u05d1\u05b0\u05d0\u05b4\u05bd\u05d9\u05dd", "\u05d5\u05b0\u05d4\u05b5\u05e9\u05c1\u05b4\u05d9\u05d1\u05d5\u05bc", "\u05e9\u05c1\u05b5\u05d1\u05b6\u05d8", "\u05d4\u05b7\u05d1\u05bc\u05b0\u05e8\u05b4\u05d9\u05e2\u05b4\u05bd\u05d9", "\u05e2\u05b5\u05e5", "\u05d5\u05bc\u05de\u05b5\u05d0\u05b5\u05ea", "\u05d0\u05b5\u05dc\u05b8\u05d9\u05d5", "\u05d5\u05b0\u05d0\u05d5\u05b9\u05df", "\u05de\u05b4\u05d6\u05b0\u05d1\u05b5\u05bc\u05d7\u05b7", "\u05d4\u05b8\u05e2\u05b7\u05e8\u05b0\u05d1\u05bc\u05b7\u05d9\u05b4\u05dd", "\u05e2\u05b5\u05d3\u05b8\u05d4", "\u05d7\u05b8\u05e8\u05b6\u05e9\u05c2", "\u05de\u05b0\u05d1\u05b9\u05e8\u05b8\u05da\u05b0", "\u05d5\u05b0\u05db\u05b7\u05d0\u05b2\u05e8\u05b4\u05d9", "\u05d4\u05b8\u05e8\u05bb\u05bd\u05d0\u05d5\u05bc\u05d1\u05b5\u05e0\u05b4\u05d9", "\u05e4\u05bc\u05b0\u05e7\u05bb\u05d3\u05b5\u05d9\u05db\u05b6\u05dd", "\u05d9\u05b4\u05ea\u05b0\u05d7\u05b7\u05e9\u05c1\u05bc\u05b8\u05bd\u05d1", "\u05e2\u05b7\u05e6\u05b0\u05de\u05b9\u05bd\u05e0\u05b8\u05d4", "\u05e0\u05b4\u05dc\u05b0\u05d7\u05b7\u05dd", "\u05e4\u05bc\u05b0\u05d3\u05d5\u05bc\u05d9\u05b5\u05d9", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05b0\u05de\u05b4\u05d9\u05d3\u05b8\u05e2\u05b4\u05d9", "\u05d1\u05bc\u05b0\u05de\u05d5\u05b9\u05e2\u05b2\u05d3\u05b5\u05d9\u05db\u05b6\u05dd", "\u05d4\u05b7\u05de\u05bc\u05b7\u05d7\u05b0\u05ea\u05bc\u05b9\u05ea", "\u05d2\u05b0\u05d1\u05d5\u05bc\u05dc\u05b6\u05bd\u05da\u05b8", "\u05de\u05b4\u05d2\u05b0\u05d3\u05bc\u05b9\u05bd\u05dc", "\u05d5\u05b0\u05d9\u05b4\u05dc\u05bc\u05b8\u05d5\u05d5\u05bc", "\u05d0\u05b2\u05d1\u05b8\u05e8\u05b2\u05db\u05b5\u05bd\u05dd", "\u05dc\u05b0\u05e9\u05c2\u05b4\u05db\u05bc\u05b4\u05d9\u05dd", "\u05e9\u05b6\u05c1\u05dc\u05b6\u05dd", "\u05e9\u05c1\u05b8\u05dc\u05bd\u05d5\u05b9\u05dd", "\u05d5\u05b0\u05e0\u05b4\u05e7\u05bc\u05b8\u05d4", "\u05d4\u05b8\u05d0\u05b7\u05e8\u05b0\u05d0\u05b5\u05dc\u05b4\u05bd\u05d9", "\u05d5\u05b7\u05ea\u05bc\u05b4\u05dc\u05b0\u05d7\u05b7\u05e5", "\u05d4\u05b8\u05bd\u05d0\u05b7\u05e9\u05c1\u05b0\u05d1\u05bc\u05b5\u05dc\u05b4\u05d9", "\u05d1\u05bc\u05b0\u05e8\u05b7\u05d7", "\u05e9\u05b8\u05c2\u05db\u05b8\u05e8", "\u05de\u05b7\u05d7\u05b2\u05e0\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05d4\u05b7\u05e0\u05bc\u05b8\u05d6\u05b4\u05d9\u05e8", "\u05d1\u05bc\u05b0\u05e2\u05b4\u05d9\u05bc\u05b5\u05d9", "\u05d1\u05bc\u05b8\u05d4\u05b5\u05df", "\u05d3\u05b8\u05bc\u05d2", "\u05d5\u05b7\u05ea\u05bc\u05b4\u05e7\u05bc\u05b8\u05d1\u05b5\u05e8", "\u05de\u05b4\u05e7\u05b0\u05d3\u05bc\u05b0\u05e9\u05c1\u05d5\u05b9", "\u05ea\u05bc\u05b5\u05d0\u05b8\u05e1\u05b5\u05e3", "\u05d5\u05bc\u05dc\u05b0\u05de\u05b4\u05d1\u05bc\u05b5\u05d9\u05ea", "\u05de\u05b7\u05e9\u05c2\u05bc\u05b8\u05d0\u05b8\u05bd\u05dd", "\u05d1\u05bc\u05b0\u05e2\u05b6\u05e8\u05b0\u05db\u05bc\u05b0\u05da\u05b8", "\u05d1\u05bc\u05b0\u05d4\u05b4\u05e7\u05bc\u05b8\u05d4\u05b5\u05dc", "\u05d5\u05b0\u05e0\u05b8\u05ea\u05b7\u05df", "\u05e1\u05b0\u05ea\u05d5\u05bc\u05e8", "\u05e0\u05b8\u05bd\u05e4\u05b6\u05e9\u05c1", "\u05de\u05b7\u05e2\u05b7\u05df", "\u05e2\u05b8\u05e6\u05d5\u05bc\u05dd", "\u05d5\u05b0\u05d7\u05b6\u05e9\u05c1\u05b0\u05d1\u05bc\u05d5\u05b9\u05df", "\u05d5\u05bc\u05d1\u05b0\u05ea\u05d5\u05b9\u05db\u05b8\u05dd", "\u05ea\u05bc\u05b7\u05de\u05b0\u05e2\u05b4\u05d9\u05d8\u05d5\u05bc", "\u05d5\u05b0\u05d4\u05b7\u05bd\u05e2\u05b2\u05d1\u05b7\u05e8\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05d5\u05b7\u05d0\u05b7\u05db\u05b7\u05dc\u05bc\u05b6\u05d4", "\u05ea\u05bc\u05b0\u05de\u05b4\u05d9\u05de\u05b8\u05d4", "\u05d4\u05b6\u05d7\u05b1\u05d9\u05b5\u05bd\u05d9\u05ea\u05b4\u05d9", "\u05d0\u05bd\u05b6\u05e2\u05b1\u05e9\u05c2\u05b6\u05bd\u05d4", "\u05d5\u05b0\u05dc\u05b7\u05e9\u05c2\u05bc\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05d4\u05b7\u05e4\u05bc\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05d5\u05b0\u05d4\u05b4\u05ea\u05b0\u05e0\u05b7\u05d7\u05b7\u05dc\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05e0\u05b5\u05e8\u05b9\u05ea\u05b6\u05d9\u05d4\u05b8", "\u05e8\u05b9\u05db\u05b5\u05d1", "\u05d1\u05bc\u05b0\u05ea\u05bd\u05d5\u05b9\u05db\u05b0\u05db\u05b6\u05dd", "\u05dc\u05b0\u05d1\u05b6\u05dc\u05b7\u05e2", "\u05e0\u05b4\u05bd\u05d9\u05d7\u05b9\u05d7\u05b4\u05d9", "\u05d1\u05bc\u05b0\u05d0\u05b8\u05dc\u05bd\u05d5\u05bc\u05e9\u05c1", "\u05e8\u05b9\u05e6\u05b5\u05d7\u05b7\u05bd", "\u05dc\u05b8\u05e2\u05b9\u05e9\u05c2\u05b6\u05d4", "\u05dc\u05b0\u05d1\u05b8\u05dc\u05b8\u05e7", "\u05d0\u05b1\u05dc\u05b9\u05d4\u05b5\u05d9\u05db\u05b6\u05bd\u05dd", "\u05d0\u05b2\u05d1\u05b4\u05d9\u05d4\u05b6\u05df", "\u05de\u05b5\u05d7\u05b9\u05e8", "\u05e9\u05c1\u05b0\u05dc\u05b9\u05de\u05b4\u05bd\u05d9", "\u05d1\u05bc\u05b0\u05e0\u05b8\u05e1\u05b0\u05e2\u05b8\u05dd", "\u05de\u05b4\u05e6\u05b0\u05d5\u05b8\u05ea\u05d5\u05b9", "\u05d1\u05bc\u05b0\u05d3\u05b4\u05d9\u05d1\u05b9\u05df", "\u05d5\u05b0\u05d9\u05b5\u05e8\u05b0\u05d3\u05bc\u05b0", "\u05ea\u05bc\u05b4\u05e4\u05b0\u05d3\u05bc\u05b6\u05bd\u05d4", "\u05d1\u05bc\u05b0\u05e9\u05c2\u05b4\u05e0\u05b0\u05d0\u05b8\u05d4", "\u05d5\u05b0\u05d4\u05b8\u05bd\u05d0\u05e1\u05b7\u05e4\u05b0\u05e1\u05bb\u05e3", "\u05d5\u05b0\u05e0\u05b8\u05e9\u05c2\u05b4\u05d9\u05d0", "\u05dc\u05b7\u05d6\u05bc\u05b8\u05d1\u05b7\u05d7", "\u05e7\u05b0\u05d3\u05b9\u05e9\u05c1\u05b4\u05d9\u05dd", "\u05dc\u05b7\u05de\u05bc\u05b4\u05e9\u05c1\u05b0\u05db\u05bc\u05b8\u05df", "\u05d5\u05b0\u05e0\u05b8\u05e9\u05c1\u05d5\u05bc\u05d1\u05b8\u05d4", "\u05dc\u05b7\u05d4\u05b2\u05de\u05b4\u05d9\u05ea\u05b5\u05e0\u05d5\u05bc", "\u05d4\u05b7\u05d1\u05bc\u05b7\u05e8\u05b0\u05d6\u05b6\u05dc", "\u05d4\u05b7\u05bd\u05de\u05bc\u05b4\u05d6\u05b0\u05d1\u05bc\u05b0\u05d7\u05b9\u05ea", "\u05d1\u05b0\u05e2\u05b9\u05e8", "\u05de\u05b0\u05dc\u05b9\u05d0\u05ea", "\u05d4\u05b7\u05e0\u05bc\u05b0\u05e4\u05b4\u05dc\u05b4\u05d9\u05dd", "\u05d1\u05bc\u05b0\u05de\u05b7\u05e7\u05b0\u05d4\u05b5\u05dc\u05b9\u05bd\u05ea", "\u05dc\u05b0\u05d0\u05b7\u05d4\u05b2\u05e8\u05b9\u05df", "\u05d9\u05b0\u05d8\u05b7\u05de\u05bc\u05b0\u05d0\u05d5\u05bc", "\u05d9\u05b9\u05e8\u05b6\u05e9\u05c1\u05b6\u05ea", "\u05d5\u05b0\u05d9\u05b7\u05e2\u05b2\u05e9\u05c2\u05d5\u05bc", "\u05e8\u05b9\u05de\u05b7\u05d7", "\u05d9\u05b0\u05e0\u05b7\u05e7\u05bc\u05b6\u05d4", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05b4\u05de\u05b0\u05e8\u05b9\u05e0\u05b4\u05bd\u05d9", "\u05e0\u05b4\u05d3\u05b0\u05e8\u05b8\u05d4\u05bc", "\u05d7\u05b5\u05df", "\u05d9\u05b8\u05d8\u05b7\u05d1", "\u05de\u05b7\u05d2\u05b5\u05bc\u05e4\u05b8\u05d4", "\u05d9\u05b0\u05db\u05b7\u05e4\u05bc\u05b6\u05e8", "\u05de\u05b7\u05e9\u05c2\u05bc\u05b8\u05d0\u05b8\u05dd", "\u05d5\u05b0\u05e9\u05c1\u05b8\u05db\u05b7\u05d1", "\u05d9\u05b8\u05d0\u05b4\u05d9\u05e8\u05d5\u05bc", "\u05de\u05b5\u05d9\u05d3\u05b8\u05d3", "\u05d1\u05bc\u05b7\u05db\u05bc\u05b0\u05ea\u05bb\u05d1\u05b4\u05d9\u05dd", "\u05d5\u05b0\u05d4\u05b7\u05d9\u05b0\u05d1\u05d5\u05bc\u05e1\u05b4\u05d9", "\u05db\u05bc\u05b9\u05d7\u05b7", "\u05d2\u05bc\u05b7\u05d3\u05bc\u05b5\u05dc", "\u05d4\u05b8\u05e8\u05b8\u05e8", "\u05d5\u05b0\u05e0\u05b8\u05e1\u05b0\u05e2\u05d5\u05bc", "\u05d4\u05b8\u05ea\u05b0\u05e4\u05bc\u05b8\u05e7\u05b0\u05d3\u05d5\u05bc", "\u05d0\u05b6\u05d7\u05b8\u05d3", "\u05d1\u05bc\u05b0\u05de\u05b5\u05ea", "\u05d5\u05b7\u05e2\u05b2\u05e0\u05b8\u05bd\u05e0\u05b0\u05da\u05b8", "\u05d5\u05b0\u05d7\u05b8\u05d2\u05b0\u05dc\u05b8\u05d4", "\u05d9\u05b4\u05e1\u05bc\u05b8\u05e2\u05d5\u05bc", "\u05e0\u05b7\u05d7\u05b2\u05dc\u05b4\u05d9\u05d0\u05b5\u05dc", "\u05e8\u05b6\u05d1\u05b7\u05e2", "\u05de\u05b5\u05e2\u05bd\u05d5\u05b9\u05d3\u05b0\u05da\u05b8", "\u05d7\u05b8\u05dc\u05d5\u05bc\u05e5", "\u05ea\u05bc\u05b4\u05e4\u05b0\u05e7\u05b0\u05d3\u05d5\u05bc", "\u05d1\u05bc\u05b7\u05de\u05bc\u05b4\u05d3\u05b0\u05d1\u05bc\u05b8\u05e8", "\u05d5\u05b0\u05d4\u05b4\u05d8\u05bc\u05b6\u05d4\u05b8\u05bd\u05e8\u05d5\u05bc", "\u05d5\u05b0\u05d3\u05b4\u05e9\u05c1\u05bc\u05b0\u05e0\u05d5\u05bc", "\u05d1\u05bc\u05b0\u05d4\u05b7\u05d0\u05b2\u05e8\u05b4\u05d9\u05da\u05b0", "\u05d5\u05b7\u05d0\u05b2\u05db\u05b7\u05dc\u05b0\u05ea\u05bc\u05b6\u05bd\u05dd", "\u05dc\u05b0\u05d2\u05b5\u05e8\u05b0\u05e9\u05c1\u05d5\u05b9\u05df", "\u05e2\u05b2\u05d3\u05b8\u05ea\u05b0\u05da\u05b8", "\u05d4\u05b7\u05ea\u05bc\u05d5\u05b9\u05dc\u05b8\u05e2\u05b4\u05d9", "\u05e2\u05b7\u05d6\u05bc\u05b8\u05bd\u05df", "\u05e9\u05c1\u05b8\u05d8\u05d5\u05bc", "\u05ea\u05b0\u05e9\u05c1\u05d5\u05bc\u05d1\u05bb\u05df", "\u05e2\u05b9\u05de\u05b5\u05d3", "\u05d0\u05b2\u05e9\u05c1\u05b6\u05e8\u05c4", "\u05d5\u05bc\u05e9\u05c2\u05b0\u05e2\u05b4\u05d9\u05e8", "\u05d7\u05b5\u05dc\u05b6\u05e3", "\u05d1\u05bc\u05b0\u05e2\u05b5\u05d9\u05e0\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05d0\u05b8\u05de\u05b7\u05df", "\u05d5\u05bc\u05e9\u05c1\u05b0\u05de\u05b9\u05bd\u05e0\u05b7\u05ea", "\u05e2\u05b7\u05de\u05bc\u05b4\u05d9\u05e9\u05c1\u05b7\u05d3\u05bc\u05b8\u05bd\u05d9", "\u05d0\u05b6\u05ea\u05bc\u05b5\u05df", "\u05d4\u05b8\u05e8\u05d5\u05bc\u05d7\u05b9\u05ea", "\u05dc\u05b0\u05d9\u05b5\u05e6\u05b6\u05e8", "\u05de\u05b4\u05de\u05bc\u05b5\u05d9", "\u05d9\u05b0\u05e8\u05b5\u05d0\u05ea\u05b6\u05dd", "\u05d5\u05b7\u05d9\u05bc\u05b8\u05e0\u05b4\u05d9\u05d0\u05d5\u05bc", "\u05e2\u05b2\u05e8\u05b8\u05d1\u05b8\u05d4", "\u05d1\u05b6\u05bd\u05d7\u05b8\u05dc\u05b8\u05dc", "\u05de\u05b4\u05d9\u05db\u05b8\u05d0\u05b5\u05dc", "\u05d1\u05b8\u05d0\u05b5\u05e9\u05c1", "\u05d0\u05b2\u05d7\u05bb\u05d6\u05b8\u05bc\u05d4", "\u05d4\u05b7\u05de\u05bc\u05b7\u05dc\u05b0\u05db\u05bc\u05b4\u05d9\u05d0\u05b5\u05dc\u05b4\u05bd\u05d9", "\u05d2\u05b6\u05bc\u05d1\u05b6\u05e8", "\u05d5\u05b7\u05bd\u05d9\u05bc\u05b4\u05e6\u05b0\u05d1\u05bc\u05b0\u05d0\u05d5\u05bc", "\u05dc\u05b0\u05d9\u05b8\u05e9\u05c1\u05d5\u05bc\u05d1", "\u05d4\u05b8\u05e8\u05b4\u05d0\u05e9\u05c1\u05b9\u05e0\u05b4\u05d9\u05dd", "\u05de\u05b4\u05e9\u05c1\u05bc\u05b0\u05e4\u05b8\u05dd", "\u05d5\u05bc\u05de\u05b4\u05d2\u05bc\u05b9\u05e8\u05b7\u05dc", "\u05d5\u05b0\u05d7\u05b8\u05e0\u05d5\u05bc", "\u05de\u05b7\u05e1\u05b0\u05e2\u05b5\u05d9", "\u05d1\u05bc\u05b0\u05e6\u05b7\u05dc\u05b0\u05de\u05b9\u05e0\u05b8\u05bd\u05d4", "\u05dc\u05b0\u05db\u05b9\u05dc", "\u05e9\u05c1\u05b8\u05dc\u05b9\u05d7\u05b7", "\u05d4\u05b8\u05e8\u05b0\u05d1\u05b4\u05d9\u05e2\u05b4\u05d9", "\u05e7\u05b7\u05de\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05e2\u05b6\u05d2\u05b0\u05dc\u05b9\u05ea", "\u05dc\u05b8\u05d5\u05b8\u05d4", "\u05de\u05b5\u05e2\u05b7\u05e6\u05b0\u05de\u05d5\u05b9\u05df", "\u05d4\u05b7\u05d4\u05bd\u05d5\u05bc\u05d0", "\u05e2\u05b2\u05d5\u05ba\u05e0\u05b8\u05dd", "\u05d5\u05b0\u05dc\u05b4\u05e8\u05b0\u05db\u05bb\u05e9\u05c1\u05b8\u05dd", "\u05ea\u05bc\u05b7\u05e2\u05b7\u05e8", "\u05d5\u05b7\u05ea\u05bc\u05b4\u05de\u05b0\u05e2\u05b9\u05dc", "\u05de\u05b7\u05ea\u05b0\u05e0\u05d5\u05bc", "\u05e0\u05b0\u05e7\u05b9\u05dd", "\u05d6\u05b8\u05d2", "\u05db\u05bc\u05d5\u05b9\u05db\u05b8\u05d1", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05e8\u05b0\u05d0\u05d5\u05bc\u05dd", "\u05de\u05b5\u05d0\u05b4\u05ea\u05bc\u05b8\u05dd", "\u05d4\u05b4\u05de\u05bc\u05b8\u05e9\u05c1\u05b7\u05d7", "\u05e4\u05b0\u05bc\u05e8\u05b4\u05d9", "\u05d0\u05b7\u05db\u05bc\u05b6\u05e0\u05bc\u05d5\u05bc", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05db\u05bc\u05d5\u05bc", "\u05d4\u05b8\u05bd\u05d9\u05b0\u05ea\u05b8\u05d4", "\u05ea\u05bc\u05b0\u05d3\u05b7\u05d1\u05bc\u05b5\u05e8", "\u05d1\u05b7\u05e7\u05bc\u05b8\u05bd\u05d1\u05b6\u05e8", "\u05d0\u05b5\u05e9\u05c1", "\u05ea\u05bc\u05d5\u05b9\u05dc\u05b7\u05e2\u05b7\u05ea", "\u05dc\u05d5\u05b9", "\u05de\u05b5\u05e8\u05b4\u05e1\u05bc\u05b8\u05d4", "\u05d7\u05c7\u05d2\u05b0\u05dc\u05b8\u05d4", "\u05d8\u05b4\u05bd\u05d9\u05e8\u05b9\u05ea\u05b8\u05dd", "\u05d0\u05b2\u05d7\u05b5\u05d9\u05db\u05b6\u05dd", "\u05d4\u05b2\u05d3\u05b8\u05e4\u05d5\u05b9", "\u05e9\u05c1\u05b4\u05e0\u05bc\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05d1\u05bc\u05b7\u05d8\u05bc\u05b8\u05e3", "\u05d4\u05b7\u05e7\u05bc\u05b0\u05d4\u05b8\u05ea\u05b4\u05bd\u05d9", "\u05d0\u05b1\u05d3\u05d5\u05b9\u05dd", "\u05d9\u05b4\u05d9\u05e9\u05c1\u05b7\u05e8", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05de\u05b0\u05e9\u05c1\u05b8\u05d7\u05b5\u05dd", "\u05d1\u05bc\u05b0\u05d4\u05b7\u05e6\u05bc\u05b9\u05ea\u05b8\u05dd", "\u05d9\u05d5\u05b9\u05e1\u05b5\u05e3", "\u05d1\u05bc\u05b7\u05bd\u05de\u05b0\u05e1\u05b4\u05dc\u05bc\u05b8\u05d4", "\u05dc\u05b0\u05de\u05b8\u05db\u05b4\u05d9\u05e8", "\u05de\u05b8\u05e8\u05b8\u05d4", "\u05e9\u05c1\u05b8\u05e7\u05b6\u05dc", "\u05ea\u05bc\u05b0\u05e0\u05d5\u05bc\u05e4\u05b9\u05ea", "\u05d5\u05b7\u05d9\u05bc\u05b9\u05d0\u05de\u05b6\u05e8", "\u05d5\u05b7\u05d9\u05bc\u05b8\u05e8\u05b5\u05e2\u05d5\u05bc", "\u05e7\u05b0\u05d4\u05b8\u05ea", "\u05d1\u05b8\u05bc\u05d4\u05b6\u05dd", "\u05d9\u05b0\u05d4\u05b9\u05d5\u05b8\u05d4", "\u05e7\u05b7\u05d1\u05bc\u05b9\u05d4", "\u05e8\u05b4\u05de\u05bc\u05d5\u05b9\u05df", "\u05e0\u05b8\u05e4\u05b6\u05e9\u05c1", "\u05e2\u05b2\u05d5\u05ba\u05e0\u05b8\u05d4", "\u05e9\u05b7\u05c2\u05e8", "\u05d1\u05bc\u05b0\u05d1\u05b6\u05d4\u05b1\u05de\u05b7\u05ea", "\u05de\u05b4\u05e7\u05bc\u05b8\u05d3\u05b5\u05e9\u05c1", "\u05d5\u05b0\u05d6\u05b8\u05e8", "\u05de\u05b6\u05dc\u05b6\u05da\u05b0", "\u05d7\u05b8\u05d2\u05b7\u05d2", "\u05d5\u05b0\u05e9\u05c1\u05b5\u05e8\u05b0\u05ea\u05d5\u05bc", "\u05d5\u05b0\u05d0\u05b4\u05d1\u05bc\u05b7\u05d3\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05ea\u05bc\u05b4\u05e9\u05c1\u05b0\u05de\u05b0\u05e8\u05d5\u05bc", "\u05d2\u05bc\u05b8\u05bd\u05d1\u05b6\u05e8", "\u05de\u05b4\u05e0\u05bc\u05b4\u05d3\u05b0\u05e8\u05b5\u05d9\u05db\u05b6\u05dd", "\u05e0\u05b4\u05d9\u05d7\u05d5\u05b9\u05d7\u05b7", "\u05d4\u05b7\u05e4\u05bc\u05b8\u05e8\u05b8\u05bd\u05d4", "\u05e7\u05b8\u05e6\u05b6\u05d4", "\u05d5\u05b0\u05d4\u05bd\u05b6\u05d0\u05b1\u05d1\u05b4\u05d9\u05d3", "\u05e9\u05c1\u05b4\u05e4\u05b0\u05d8\u05b8\u05bd\u05df", "\u05d9\u05d5\u05bc\u05de\u05b7\u05ea", "\u05d5\u05b7\u05d9\u05bc\u05b5\u05e8\u05b6\u05d3", "\u05d0\u05b2\u05d1\u05b4\u05d9\u05d3\u05b8\u05df", "\u05e0\u05b8\u05d2\u05b7\u05e9\u05c1", "\u05d5\u05b0\u05d8\u05b4\u05d4\u05b7\u05e8\u05b0\u05ea\u05bc\u05b8", "\u05dc\u05b0\u05de\u05b4\u05e7\u05b0\u05e0\u05b5\u05e0\u05d5\u05bc", "\u05e9\u05c1\u05b0\u05e0\u05b8\u05ea\u05d5\u05b9", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05e2\u05b2\u05dc\u05d5\u05bc", "\u05d1\u05bc\u05b0\u05e0\u05b7\u05e4\u05b0\u05e9\u05c1\u05b9\u05ea\u05b8\u05dd", "\u05d5\u05bc\u05e7\u05b0\u05d8\u05b9\u05e8\u05b6\u05ea", "\u05d4\u05b7\u05de\u05bc\u05d5\u05bc\u05e9\u05c1\u05b8\u05d1", "\u05e7\u05c7\u05e8\u05b0\u05d7\u05b4\u05d9", "\u05d2\u05bc\u05d5\u05b9\u05d9", "\u05dc\u05b7\u05e2\u05b2\u05d1\u05b8\u05d3\u05b6\u05d9\u05da\u05b8", "\u05d4\u05b4\u05ea\u05b0\u05e0\u05b7\u05d7\u05b5\u05dc", "\u05d4\u05b7\u05bd\u05d7\u05b6\u05e6\u05b0\u05e8\u05d5\u05b9\u05e0\u05b4\u05d9", "\u05d4\u05b7\u05e8", "\u05d1\u05bc\u05b7\u05e7\u05bc\u05b9\u05d3\u05b6\u05e9\u05c1", "\u05de\u05b0\u05e6\u05b7\u05d5\u05bc\u05b6\u05bd\u05d4", "\u05d5\u05b7\u05d9\u05b0\u05de\u05b7\u05dc\u05bc\u05b5\u05d0", "\u05d5\u05b0\u05d4\u05d5\u05b9\u05e8\u05b7\u05e9\u05c1\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05e1\u05b9\u05dc\u05b6\u05ea", "\u05d4\u05b6\u05bd\u05e2\u05b1\u05dc\u05b4\u05d9\u05ea\u05bb\u05e0\u05d5\u05bc", "\u05d5\u05bc\u05de\u05b4\u05d1\u05bc\u05b8\u05de\u05d5\u05b9\u05ea", "\u05db\u05bc\u05b0\u05e0\u05d5\u05b9\u05d7\u05b7", "\u05dc\u05b7\u05d4\u05b2\u05dc\u05b9\u05da\u05b0", "\u05e9\u05c1\u05b0\u05ea\u05bb\u05dd", "\u05d4\u05b8\u05d0\u05b2\u05dc\u05b8\u05e4\u05b4\u05d9\u05dd", "\u05d4\u05b7\u05e0\u05bc\u05b9\u05d2\u05b5\u05e2\u05b7", "\u05e1\u05b4\u05d9\u05e0\u05b7\u05d9", "\u05d5\u05bc\u05d1\u05b0\u05d0\u05b5\u05d9\u05dc\u05b4\u05dd", "\u05d4\u05b7\u05de\u05bc\u05b8\u05e8\u05b4\u05bd\u05d9\u05dd", "\u05d5\u05b0\u05e0\u05b4\u05e1\u05b0\u05db\u05bc\u05b5\u05d4\u05b6\u05dd", "\u05d0\u05b7\u05d1\u05b0\u05e8\u05b8\u05d4\u05b8\u05dd", "\u05e8\u05b9\u05d0\u05e9\u05c1", "\u05d4\u05b8\u05e8\u05b7\u05d2\u05b0\u05ea\u05bc\u05b4\u05d9", "\u05d1\u05bc\u05b7\u05de\u05bc\u05b0\u05d3\u05b9\u05db\u05b8\u05d4", "\u05ea\u05bc\u05b4\u05e9\u05c1\u05b0\u05e2\u05b8\u05d4", "\u05ea\u05bc\u05d5\u05b9\u05dc\u05b8\u05e2", "\u05d7\u05b4\u05e0\u05bc\u05b8\u05dd", "\u05d9\u05b5\u05e9\u05c1", "\u05de\u05b4\u05e9\u05c1\u05b0\u05e4\u05bc\u05b8\u05d8\u05b8\u05df", "\u05e9\u05c1\u05b8\u05e4\u05b6\u05e8", "\u05e9\u05b8\u05c2\u05e8\u05b7\u05e3", "\u05d0\u05b6\u05e8\u05b6\u05da\u05b0", "\u05ea\u05b7\u05e2\u05b2\u05e9\u05c2\u05b6\u05bd\u05d4", "\u05e2\u05b7\u05dd", "\u05ea\u05bc\u05b0\u05dc\u05d5\u05bc\u05e0\u05bc\u05b9\u05ea\u05b8\u05dd", "\u05e9\u05c1\u05b6\u05bd\u05e4\u05b4\u05d9", "\u05de\u05b5\u05e2\u05b7\u05dc", "\u05d1\u05bc\u05b0\u05e9\u05c2\u05b8\u05e8\u05b8\u05d4\u05bc", "\u05d0\u05b2\u05d7\u05b9\u05ea\u05b8\u05dd", "\u05dc\u05b0\u05d4\u05b8\u05e7\u05b4\u05d9\u05dd", "\u05db\u05b4\u05bc\u05d9", "\u05de\u05b7\u05e8\u05b0\u05d0\u05b6\u05d4", "\u05d5\u05b0\u05d9\u05b8\u05e8\u05b9\u05dd", "\u05d7\u05b2\u05e0\u05d5\u05bc", "\u05e9\u05c2\u05b0\u05e2\u05b9\u05e8\u05b4\u05d9\u05dd", "\u05d0\u05b9\u05db\u05b6\u05dc\u05b6\u05ea", "\u05d5\u05bc\u05de\u05b5\u05d9\u05ea\u05b0\u05e8\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05de\u05b5\u05d9\u05de\u05b6\u05d9\u05da\u05b8", "\u05ea\u05b8\u05ea\u05bb\u05e8\u05d5\u05bc", "\u05e9\u05b8\u05c1\u05dc\u05b7\u05da\u05b0", "\u05d5\u05b0\u05e0\u05d5\u05b9\u05e1\u05b7\u05e3", "\u05e9\u05c1\u05b7\u05dc\u05b0\u05de\u05b5\u05d9\u05db\u05b6\u05dd", "\u05e4\u05b6\u05bc\u05ea\u05b7\u05d7", "\u05e9\u05c1\u05b0\u05db\u05b8\u05d1\u05b0\u05ea\u05bc\u05d5\u05b9", "\u05de\u05b8\u05e1\u05b8\u05da\u05b0", "\u05d5\u05b4\u05bd\u05d9\u05d3\u05b7\u05e2\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05de\u05b5\u05d0\u05b8\u05dc\u05d5\u05bc\u05e9\u05c1", "\u05d1\u05bc\u05b0\u05d7\u05b7\u05e9\u05c1\u05b0\u05de\u05b9\u05e0\u05b8\u05bd\u05d4", "\u05d5\u05b0\u05dc\u05b7\u05d0\u05b2\u05d1\u05b9\u05ea\u05b5\u05bd\u05d9\u05e0\u05d5\u05bc", "\u05db\u05b6\u05bc\u05e8\u05b6\u05dd", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05d4\u05b7\u05e1", "\u05e7\u05b8\u05e6\u05b7\u05e3", "\u05d5\u05b0\u05dc\u05b4\u05e9\u05c1\u05b0\u05d1\u05bb\u05e2\u05b8\u05d4", "\u05de\u05b5\u05e2\u05b4\u05d9\u05bc\u05b4\u05d9\u05dd", "\u05d2\u05b5\u05e8\u05b0\u05e9\u05c1\u05d5\u05b9\u05df", "\u05d9\u05b5\u05d7\u05b8\u05dc\u05b5\u05e7", "\u05e0\u05b9\u05e1\u05b0\u05e2\u05b4\u05d9\u05dd", "\u05dc\u05b7\u05d2\u05bc\u05b5\u05e8\u05b0\u05e9\u05c1\u05bb\u05e0\u05bc\u05b4\u05d9", "\u05d4\u05bd\u05b7\u05d7\u05b7\u05d8\u05bc\u05b8\u05d0\u05b4\u05d9\u05dd", "\u05d5\u05b0\u05d4\u05b4\u05d6\u05bc\u05b4\u05d9\u05e8", "\u05e4\u05b7\u05dc\u05bc\u05d5\u05bc\u05d0", "\u05e2\u05b2\u05d8\u05b8\u05e8\u05b9\u05ea", "\u05d2\u05b5\u05bc\u05e8\u05b0\u05e9\u05bb\u05c1\u05e0\u05b4\u05bc\u05d9", "\u05d4\u05b8\u05d0\u05b6\u05d7\u05b8\u05d3", "\u05d0\u05b9\u05d6\u05b6\u05df", "\u05d5\u05b0\u05e2\u05b8\u05d1\u05b8\u05bd\u05d3\u05d5\u05bc", "\u05d4\u05bd\u05b8\u05e2\u05b5\u05d3\u05b8\u05d4", "\u05d5\u05bc\u05ea\u05b0\u05de\u05bb\u05e0\u05b7\u05ea", "\u05d4\u05d5\u05b9\u05e9\u05b5\u05c1\u05e2\u05b7", "\u05d4\u05b8\u05bd\u05e2\u05b7\u05e8\u05b0\u05d1\u05bc\u05b8\u05bd\u05d9\u05b4\u05dd", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05db\u05b0\u05e8\u05b0\u05ea\u05d5\u05bc", "\u05ea\u05bc\u05b0\u05e8\u05d5\u05bc\u05de\u05b9\u05ea\u05b8\u05d9", "\u05d7\u05b8\u05e6\u05b5\u05e8", "\u05d1\u05b8\u05dc\u05b8\u05bd\u05e7", "\u05d0\u05b6\u05bd\u05e8\u05b6\u05e5", "\u05ea\u05bc\u05b4\u05e9\u05c1\u05b0\u05d1\u05bc\u05b6\u05bd\u05da\u05bc\u05b8", "\u05d5\u05b0\u05e2\u05b4\u05e9\u05c2\u05bc\u05b8\u05e8\u05c4\u05d5\u05b9\u05df", "\u05e6\u05b4\u05e4\u05bc\u05b9\u05e8", "\u05e4\u05b8\u05bc\u05e0\u05b4\u05d9\u05dd", "\u05d1\u05bc\u05b4\u05e7\u05b0\u05e6\u05b5\u05d4", "\u05d6\u05b8\u05db\u05b7\u05e8\u05b0\u05e0\u05d5\u05bc", "\u05d5\u05b0\u05e7\u05b7\u05e8\u05b0\u05e7\u05b7\u05e8", "\u05d4\u05b2\u05d9\u05b8\u05db\u05d5\u05b9\u05dc", "\u05d1\u05b0\u05d9\u05b8\u05d3\u05d5\u05b9", "\u05db\u05bc\u05b0\u05d1\u05b8\u05e9\u05c2\u05b4\u05bd\u05d9\u05dd", "\u05de\u05b7\u05ea\u05bc\u05b8\u05e0\u05b8\u05d4", "\u05dc\u05b0\u05d1\u05b6\u05db\u05b6\u05e8", "\u05d4\u05b5\u05df", "\u05d1\u05bc\u05b0\u05de\u05b4\u05d3\u05b0\u05d9\u05b8\u05bd\u05df", "\u05d9\u05b0\u05e7\u05b4\u05d9\u05de\u05b6\u05e0\u05bc\u05d5\u05bc", "\u05e4\u05bc\u05b0\u05e7\u05b9\u05d3", "\u05d5\u05b0\u05e0\u05b9\u05d0\u05db\u05b5\u05bd\u05dc\u05b8\u05d4", "\u05db\u05bc\u05d5\u05bc\u05dc", "\u05d0\u05b7\u05e8\u05b0\u05e0\u05bd\u05d5\u05b9\u05df", "\u05d0\u05bb\u05de\u05bc\u05d5\u05b9\u05ea", "\u05d4\u05b7\u05bd\u05de\u05bc\u05b8\u05d7\u05b3\u05e8\u05b8\u05ea", "\u05e0\u05b8\u05d3\u05b7\u05e8", "\u05ea\u05bc\u05b7\u05e2\u05b2\u05d1\u05b9\u05e8", "\u05d4\u05b8\u05bd\u05e2\u05b8\u05d6\u05bc\u05b4\u05d9\u05d0\u05b5\u05dc\u05b4\u05d9", "\u05d2\u05b7\u05bc\u05dd", "\u05d4\u05b7\u05e4\u05bc\u05d5\u05bc\u05e0\u05b4\u05bd\u05d9", "\u05db\u05b0\u05bc\u05dc\u05b4\u05d9", "\u05d4\u05b7\u05bd\u05ea\u05bc\u05b7\u05d0\u05b2\u05d5\u05b8\u05d4", "\u05d3\u05bc\u05d5\u05b9\u05e8", "\u05e7\u05d5\u05bc\u05dd", "\u05d5\u05b0\u05d7\u05b8\u05e6\u05b4\u05d9\u05ea\u05b8", "\u05d4\u05b6\u05d0\u05b1\u05de\u05b7\u05e0\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05d5\u05b0\u05d4\u05d5\u05b9\u05e8\u05b4\u05d3\u05d5\u05bc", "\u05e0\u05b7\u05d7\u05b2\u05dc\u05b8\u05d4", "\u05d0\u05b4\u05ea\u05bc\u05b0\u05db\u05b6\u05dd", "\u05de\u05d5\u05b9\u05e6\u05b4\u05d9\u05d0\u05d5\u05b9", "\u05ea\u05bc\u05b0\u05de\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05e0\u05b8\u05d8\u05b8\u05d4", "\u05d5\u05b0\u05e2\u05b6\u05e6\u05b6\u05dd", "\u05ea\u05b0\u05d3\u05b7\u05d1\u05bc\u05b5\u05bd\u05e8", "\u05d5\u05b0\u05d8\u05b7\u05e4\u05bc\u05b0\u05db\u05b6\u05dd", "\u05e9\u05c1\u05b0\u05de\u05b9\u05e0\u05b8\u05d4", "\u05de\u05b4\u05e6\u05bc\u05b7\u05dc\u05b0\u05de\u05b9\u05e0\u05b8\u05d4", "\u05d5\u05bc\u05d1\u05b0\u05e9\u05c1\u05b5\u05de\u05b9\u05ea", "\u05e2\u05b7\u05d1\u05b0\u05d3\u05bc\u05b4\u05d9", "\u05d9\u05b4\u05e0\u05b0\u05d7\u05b8\u05dc\u05d5\u05bc", "\u05e0\u05b7\u05e2\u05b7\u05e8", "\u05dc\u05b0\u05d0\u05b4\u05e9\u05c1\u05bc\u05b8\u05d4", "\u05d4\u05b7\u05bd\u05de\u05bc\u05b7\u05e2\u05b2\u05e9\u05c2\u05b5\u05bd\u05e8", "\u05de\u05b4\u05e7\u05bc\u05b0\u05d4\u05b5\u05dc\u05b8\u05ea\u05b8\u05d4", "\u05d5\u05b0\u05d0\u05b8\u05e1\u05b0\u05e8\u05b8\u05d4", "\u05de\u05d5\u05b9\u05e6\u05b4\u05d9\u05d0\u05b8\u05dd", "\u05d0\u05b4\u05dd", "\u05d4\u05b4\u05db\u05bc\u05b8\u05e8\u05b5\u05ea", "\u05e0\u05d5\u05b9\u05d0\u05b7\u05dc\u05b0\u05e0\u05d5\u05bc", "\u05d5\u05b0\u05e0\u05b4\u05d3\u05b0\u05d1\u05b9\u05ea\u05b5\u05d9\u05db\u05b6\u05dd", "\u05dc\u05b0\u05d0\u05b6\u05dc\u05b0\u05e2\u05b8\u05d6\u05b8\u05e8", "\u05ea\u05bc\u05b4\u05e7\u05bc\u05b7\u05d7", "\u05d9\u05b8\u05e1\u05b7\u05e3", "\u05d1\u05b0\u05bc\u05e8\u05b4\u05d9\u05ea", "\u05d4\u05b6\u05d7\u05b8\u05e6\u05b4\u05d9\u05e8", "\u05d5\u05bc\u05dc\u05b0\u05d0\u05b7\u05d7\u05b9\u05ea\u05d5\u05b9", "\u05d9\u05b8\u05e9\u05c1\u05b4\u05d9\u05e8", "\u05d9\u05b4\u05d2\u05b0\u05d0\u05b8\u05dc", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05d5\u05bc\u05e0\u05b4\u05bd\u05d9", "\u05ea\u05bc\u05b0\u05d1\u05b8\u05e8\u05b5\u05da\u05b0", "\u05de\u05b0\u05e9\u05c1\u05b8\u05e8\u05b5\u05ea", "\u05d4\u05b8\u05bd\u05d0\u05b6\u05e9\u05c1\u05b0\u05db\u05bc\u05d5\u05b9\u05dc", "\u05e9\u05b7\u05c1\u05de\u05bc\u05d5\u05bc\u05e2\u05b7", "\u05d5\u05bc\u05e4\u05b8\u05e8\u05b7\u05e2", "\u05d0\u05b8\u05d1\u05b5\u05dc", "\u05e2\u05b6\u05e9\u05c2\u05b0\u05e8\u05b5\u05d4", "\u05de\u05b8\u05e0\u05b8\u05d4", "\u05ea\u05b8\u05bc\u05de\u05b4\u05d9\u05dd", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e7\u05bc\u05b8\u05e8", "\u05ea\u05bc\u05b7\u05db\u05b0\u05e8\u05b4\u05d9\u05ea\u05d5\u05bc", "\u05d4\u05b8\u05bd\u05d0\u05b7\u05d7\u05b7\u05ea", "\u05d4\u05b8\u05e8\u05b8\u05df", "\u05dc\u05b8\u05d1\u05b7\u05e9\u05c1", "\u05d7\u05bb\u05e9\u05c1\u05b4\u05d9\u05dd", "\u05d5\u05b0\u05e0\u05b4\u05e1\u05b0\u05db\u05bc\u05bd\u05d5\u05b9", "\u05d4\u05b7\u05d9\u05bc\u05b9\u05e0\u05b5\u05e7", "\u05de\u05b5\u05ea", "\u05ea\u05bc\u05b4\u05e8\u05b0\u05d0\u05b6\u05e0\u05bc\u05d5\u05bc", "\u05e0\u05b9\u05e2\u05b8\u05d4", "\u05ea\u05bc\u05d5\u05b9\u05e8\u05b8\u05d4", "\u05d5\u05b0\u05e0\u05d5\u05b9\u05e2\u05b2\u05d3\u05d5\u05bc", "\u05de\u05b4\u05e6\u05b0\u05d5\u05b8\u05d4", "\u05d0\u05b8\u05d4\u05b3\u05dc\u05d5\u05b9", "\u05d5\u05b8\u05e4\u05b0\u05e1\u05b4\u05bd\u05d9", "\u05d7\u05b8\u05dc\u05b7\u05dc", "\u05e2\u05b9\u05dc\u05b8\u05ea\u05d5\u05b9", "\u05e7\u05b7\u05d9\u05b4\u05df", "\u05dc\u05b8\u05d4\u05b6\u05bd\u05df", "\u05d5\u05bc\u05dc\u05b0\u05e0\u05b4\u05e1\u05b0\u05db\u05bc\u05b5\u05d9\u05db\u05b6\u05dd", "\u05de\u05b8\u05e2\u05b7\u05dc", "\u05de\u05b5\u05d7\u05b2\u05e6\u05b5\u05e8\u05b9\u05ea", "\u05d5\u05b7\u05d9\u05bc\u05b8\u05ea\u05bb\u05e8\u05d5\u05bc", "\u05dc\u05b8\u05db\u05b6\u05bd\u05dd", "\u05de\u05b5\u05bd\u05d7\u05b6\u05e9\u05c1\u05b0\u05d1\u05bc\u05d5\u05b9\u05df", "\u05de\u05bd\u05d5\u05bc\u05e1\u05b7\u05d1\u05bc\u05b9\u05ea", "\u05de\u05b7\u05e9\u05c2\u05bc\u05b8\u05d0\u05bd\u05d5\u05b9", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05b8\u05e8\u05b5\u05ea", "\u05d3\u05b6\u05bc\u05e8\u05b6\u05da\u05b0", "\u05dc\u05b0\u05d4\u05b7\u05e7\u05b0\u05d8\u05b4\u05d9\u05e8", "\u05de\u05b7\u05d7\u05b2\u05e0\u05b5\u05d4\u05d5\u05bc", "\u05d5\u05b0\u05d0\u05b8\u05e1\u05b7\u05e3", "\u05e4\u05b6\u05bc\u05d4", "\u05d5\u05b0\u05e9\u05c1\u05b4\u05bd\u05d7\u05b7\u05ea\u05bc\u05b6\u05dd", "\u05d5\u05b0\u05d0\u05d5\u05b9\u05e8\u05b4\u05e9\u05c1\u05b6\u05e0\u05bc\u05d5\u05bc", "\u05d5\u05b0\u05d0\u05b4\u05ea\u05bc\u05b8\u05e0\u05d5\u05bc", "\u05de\u05bd\u05d5\u05b9\u05ea", "\u05de\u05b7\u05e7\u05b0\u05e8\u05b4\u05d9\u05d1\u05b5\u05d9", "\u05d0\u05b2\u05d3\u05b9\u05e0\u05b8\u05d9", "\u05d0\u05b2\u05ea\u05b9\u05e0\u05b0\u05da\u05b8", "\u05d1\u05bc\u05b0\u05bd\u05d4\u05b7\u05e7\u05b0\u05e8\u05b4\u05d1\u05b8\u05dd", "\u05dc\u05b0\u05d6\u05b6\u05e8\u05b7\u05d7", "\u05ea\u05b4\u05e4\u05b0\u05d3\u05bc\u05b6\u05d4", "\u05dc\u05b0\u05d2\u05b4\u05dc\u05b0\u05e2\u05b8\u05d3", "\u05dc\u05b5\u05d0\u05de\u05b9\u05bd\u05e8", "\u05d5\u05bd\u05b7\u05d9\u05bc\u05b7\u05e2\u05b2\u05e9\u05c2\u05d5\u05bc", "\u05d9\u05b8\u05e8\u05b4\u05d9\u05de\u05d5\u05bc", "\u05d5\u05b7\u05d9\u05b0\u05e9\u05c1\u05b7\u05dc\u05bc\u05b0\u05d7\u05d5\u05bc", "\u05db\u05bc\u05bd\u05b7\u05d7\u05b2\u05d2\u05b8\u05d1\u05b4\u05d9\u05dd", "\u05e6\u05b9\u05d0\u05df", "\u05e4\u05bc\u05b0\u05e8\u05b4\u05d9", "\u05e0\u05b8\u05d2\u05b7\u05e2", "\u05e9\u05b0\u05c1\u05d1\u05b4\u05d9", "\u05e2\u05d5\u05b9\u05dc\u05b8\u05d4", "\u05d0\u05b8\u05d7\u05bb\u05d6", "\u05d2\u05bc\u05b5\u05e8\u05b8\u05d4", "\u05d5\u05b0\u05e0\u05b9\u05bd\u05d0\u05d7\u05b2\u05d6\u05d5\u05bc", "\u05de\u05b4\u05e7\u05bc\u05b4\u05d1\u05b0\u05e8\u05d5\u05b9\u05ea", "\u05d1\u05b0\u05bc\u05e2\u05b7\u05d3", "\u05d5\u05bc\u05db\u05b0\u05e9\u05c1\u05d5\u05b9\u05e7", "\u05d5\u05b0\u05dc\u05b7\u05bd\u05e2\u05b2\u05d1\u05b8\u05d3\u05b6\u05d9\u05da\u05b8", "\u05dc\u05b7\u05dc\u05b0\u05d5\u05b4\u05d9\u05bc\u05b4\u05bd\u05dd", "\u05d5\u05b7\u05bd\u05d0\u05d3\u05b9\u05e0\u05b4\u05d9", "\u05d5\u05bc\u05d1\u05b8\u05e0\u05b6\u05d9\u05da\u05b8", "\u05de\u05b4\u05d9\u05bc\u05b4\u05e9\u05c2\u05b0\u05e8\u05b8\u05d0\u05b5\u05bd\u05dc", "\u05d7\u05b2\u05e6\u05b5\u05e8\u05d5\u05b9\u05ea", "\u05de\u05b4\u05e0\u05b0\u05d7\u05b8\u05ea\u05d5\u05b9", "\u05e2\u05b5\u05d3\u05b4\u05d9\u05dd", "\u05e9\u05c2\u05b8\u05d8\u05b4\u05d9\u05ea", "\u05e4\u05b0\u05e7\u05bb\u05d3\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05dc\u05b4\u05e0\u05b0\u05d7\u05b9\u05dc", "\u05de\u05b8\u05dc\u05b5\u05d0", "\u05e2\u05b5\u05d9\u05e0\u05b8\u05df", "\u05d5\u05b0\u05d4\u05d5\u05bc\u05e8\u05b7\u05d3", "\u05de\u05b7\u05d7\u05b0\u05ea\u05bc\u05b9\u05ea\u05b6\u05d9\u05d4\u05b8", "\u05de\u05b5\u05d9\u05de\u05b8\u05d9\u05d5", "\u05d5\u05b7\u05e0\u05bc\u05b4\u05d9\u05e8\u05b8\u05dd", "\u05d1\u05bc\u05b0\u05e7\u05b7\u05e8", "\u05d1\u05b8\u05d4\u05b5\u05df", "\u05d5\u05b0\u05d4\u05b7\u05bd\u05d7\u05b4\u05ea\u05bc\u05b4\u05d9", "\u05de\u05b4\u05dc\u05b0\u05db\u05b8\u05bc\u05d4", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05b8\u05dc\u05b8\u05dc", "\u05e8\u05d5\u05bc\u05d7\u05b7", "\u05dc\u05b8\u05e7\u05b9\u05d1", "\u05ea\u05bc\u05b4\u05e7\u05bc\u05b8\u05d7\u05d5\u05bc", "\u05d5\u05b7\u05d4\u05b2\u05e8\u05b5\u05e2\u05b9\u05ea\u05b6\u05dd", "\u05d4\u05b4\u05ea\u05b0\u05d0\u05b7\u05d5\u05bc\u05d5\u05bc", "\u05de\u05b4\u05e8\u05b0\u05d9\u05b8\u05dd", "\u05d4\u05b8\u05e2\u05b9\u05e4\u05b8\u05bd\u05e8\u05b6\u05ea", "\u05de\u05d5\u05bc\u05e9\u05b4\u05c1\u05d9", "\u05d5\u05bc\u05e0\u05b0\u05d3\u05b8\u05e8\u05b6\u05d9\u05d4\u05b8", "\u05dc\u05b7\u05d3\u05bc\u05b8\u05dd", "\u05e9\u05c2\u05b4\u05de\u05b0\u05d7\u05b7\u05ea\u05b0\u05db\u05b6\u05dd", "\u05d9\u05b4\u05e8\u05b0\u05d0\u05bd\u05d5\u05bc\u05d4\u05b8", "\u05e4\u05bc\u05b8\u05e7\u05b0\u05d3\u05d5\u05bc", "\u05d0\u05b1\u05dc\u05b4\u05d9\u05d0\u05b8\u05d1", "\u05d1\u05b8\u05bc\u05dc\u05b8\u05e7", "\u05d5\u05b0\u05e2\u05b4\u05e0\u05bc\u05d5\u05bc", "\u05d4\u05b7\u05de\u05bc\u05b9\u05e8\u05b4\u05d9\u05dd", "\u05d9\u05d5\u05b9\u05e0\u05b8\u05d4", "\u05d1\u05b7\u05e0\u05bc\u05b6\u05d2\u05b6\u05d1", "\u05e6\u05b8\u05d1\u05b8\u05bd\u05d4", "\u05d9\u05b0\u05d4\u05d5\u05b9\u05e9\u05c1\u05d5\u05bc\u05e2\u05b7", "\u05d4\u05b8\u05d0\u05b8\u05dc\u05b9\u05ea", "\u05dc\u05b0\u05e9\u05c2\u05b8\u05d8\u05b8\u05df", "\u05d4\u05b8\u05e2\u05b2\u05d2\u05b8\u05dc\u05b9\u05ea", "\u05d5\u05bc\u05dc\u05b0\u05d6\u05b6\u05d1\u05b7\u05d7", "\u05e8\u05b4\u05d0\u05e9\u05c1\u05d5\u05b9\u05df", "\u05d5\u05b0\u05d9\u05b8\u05e8\u05b7\u05d3\u05b0\u05ea\u05bc\u05b4\u05d9", "\u05d9\u05b4\u05d3\u05bc\u05b9\u05e8", "\u05d4\u05b7\u05d6\u05bc\u05b5\u05d4", "\u05d5\u05b7\u05d9\u05b0\u05e9\u05c1\u05b7\u05dc\u05bc\u05b7\u05d7", "\u05d7\u05b2\u05e4\u05b8\u05e8\u05d5\u05bc\u05d4\u05b8", "\u05dc\u05b0\u05d7\u05b8\u05de\u05d5\u05bc\u05dc", "\u05de\u05b4\u05de\u05bc\u05b7\u05d8\u05bc\u05d5\u05b9\u05ea", "\u05d1\u05bc\u05b0\u05e4\u05b8\u05e0\u05b6\u05d9\u05d4\u05b8", "\u05d4\u05b7\u05d6\u05bc\u05b4\u05db\u05bc\u05b8\u05e8\u05d5\u05b9\u05df", "\u05de\u05b4\u05e9\u05c1\u05b0\u05e4\u05bc\u05b0\u05d7\u05b9\u05ea\u05b8\u05dd", "\u05d5\u05b8\u05de\u05b5\u05bd\u05ea\u05d5\u05bc", "\u05d1\u05bc\u05b8\u05d4\u05b6\u05dd", "\u05d2\u05b8\u05d5\u05b7\u05e2\u05b0\u05e0\u05d5\u05bc", "\u05d0\u05b4\u05e1\u05bc\u05b8\u05e8", "\u05d4\u05b7\u05e0\u05bc\u05b6\u05d2\u05b6\u05e3", "\u05d5\u05b0\u05d8\u05b4\u05de\u05bc\u05b5\u05d0", "\u05e7\u05b8\u05e6\u05b5\u05d4\u05d5\u05bc", "\u05d4\u05bd\u05d5\u05bc\u05d0", "\u05d5\u05b0\u05d6\u05b9\u05d0\u05ea", "\u05dc\u05b0\u05e9\u05c1\u05b4\u05de\u05b0\u05e8\u05b9\u05df", "\u05de\u05b6\u05e8\u05b4\u05d9", "\u05dc\u05b0\u05d4\u05b7\u05d6\u05bc\u05b4\u05d9\u05e8", "\u05d5\u05b0\u05d0\u05b7\u05bd\u05d9\u05b4\u05dc", "\u05d3\u05bc\u05b4\u05d9\u05d1\u05b9\u05df", "\u05d5\u05b7\u05e2\u05b2\u05d1\u05b9\u05d3\u05b7\u05ea", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05b5\u05dc\u05b8\u05e0\u05b4\u05d9", "\u05dc\u05b4\u05de\u05b0\u05e2\u05b9\u05dc", "\u05d1\u05bc\u05b0\u05ea\u05b8\u05bd\u05d7\u05b7\u05ea", "\u05dc\u05b0\u05de\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05d7\u05b3\u05de\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05e0\u05b7\u05d7\u05b0\u05d1\u05bc\u05b4\u05d9", "\u05dc\u05b0\u05d1\u05b9\u05d0", "\u05e7\u05b9\u05e8\u05b7\u05d7", "\u05dc\u05b4\u05d9\u05e0\u05d5\u05bc", "\u05ea\u05b4\u05e8\u05b0\u05e6\u05b8\u05d4", "\u05ea\u05bc\u05b7\u05e2\u05b2\u05e9\u05c2\u05b6\u05d4", "\u05d4\u05b7\u05e0\u05bc\u05b0\u05d7\u05b8\u05e9\u05c1\u05b4\u05d9\u05dd", "\u05e1\u05d5\u05bc\u05e1\u05b4\u05bd\u05d9", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05bd\u05ea\u05b0\u05d7\u05b7\u05d8\u05bc\u05b0\u05d0\u05d5\u05bc", "\u05de\u05b4\u05bd\u05e9\u05c1\u05b0\u05de\u05b7\u05e8\u05b0\u05ea\u05bc\u05b0\u05da\u05b8", "\u05d4\u05bd\u05b7\u05d7\u05bb\u05e7\u05bc\u05b4\u05d9\u05dd", "\u05de\u05b4\u05d1\u05bc\u05b4\u05dc\u05b0\u05ea\u05bc\u05b4\u05d9", "\u05d1\u05b0\u05db\u05d5\u05b9\u05e8", "\u05e2\u05d5\u05b9\u05d3", "\u05e9\u05b0\u05c1\u05de\u05b4\u05d9\u05e0\u05b4\u05d9", "\u05d4\u05b8\u05e8\u05b9\u05d2", "\u05d9\u05b8\u05e8\u05b7\u05e9\u05c1", "\u05e8\u05b0\u05e2\u05d5\u05bc\u05d0\u05b5\u05bd\u05dc", "\u05d5\u05bc\u05e4\u05b0\u05e7\u05d5\u05bc\u05d3\u05b5\u05d9", "\u05de\u05b8\u05db\u05b4\u05d9\u05e8", "\u05d4\u05b7\u05de\u05bc\u05b0\u05e9\u05c1\u05bb\u05d7\u05b4\u05d9\u05dd", "\u05dc\u05b8\u05e0\u05bb\u05e1", "\u05db\u05bc\u05d5\u05bc\u05df", "\u05e0\u05b0\u05de\u05d5\u05bc\u05d0\u05b5\u05dc", "\u05d5\u05bc\u05de\u05b4\u05bd\u05de\u05bc\u05b7\u05d7\u05b2\u05e6\u05b4\u05d9\u05ea", "\u05d5\u05bc\u05dc\u05b0\u05de\u05b7\u05e9\u05c2\u05bc\u05b8\u05bd\u05d0", "\u05e7\u05b8\u05e8\u05b0\u05e2\u05d5\u05bc", "\u05de\u05b5\u05bd\u05d7\u05b7\u05e9\u05c1\u05b0\u05de\u05b9\u05e0\u05b8\u05d4", "\u05d7\u05b8\u05dc\u05b8\u05dc", "\u05d5\u05b0\u05e0\u05b8\u05e1\u05b8\u05bd\u05e2\u05d5\u05bc", "\u05d0\u05b7\u05e8\u05b0\u05d2\u05bc\u05b8\u05de\u05b8\u05bd\u05df", "\u05d5\u05b0\u05e2\u05b7\u05de\u05bc\u05d5\u05bc\u05d3\u05b8\u05d9\u05d5", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05bd\u05d4\u05b7\u05e8\u05b0\u05d2\u05d5\u05bc", "\u05d1\u05bc\u05b0\u05d7\u05b9\u05e8", "\u05ea\u05bc\u05b4\u05e1\u05bc\u05b8\u05e4\u05d5\u05bc", "\u05d2\u05bc\u05b7\u05d3", "\u05db\u05bc\u05b0\u05d7\u05bb\u05e7\u05bc\u05b7\u05ea", "\u05dc\u05b0\u05e2\u05b5\u05e8\u05b8\u05df", "\u05d4\u05b8\u05bd\u05d0\u05b7\u05e8\u05b0\u05d1\u05bc\u05b8\u05e2\u05b4\u05d9\u05dd", "\u05d0\u05b9\u05ea\u05b0\u05db\u05b8\u05d4", "\u05d9\u05b9\u05de\u05b7\u05d9\u05b4\u05dd", "\u05d5\u05bc\u05d1\u05b0\u05e2\u05b4\u05d9\u05e8\u05b5\u05bd\u05e0\u05d5\u05bc", "\u05e9\u05b0\u05c1\u05d0\u05d5\u05b9\u05dc", "\u05d5\u05b0\u05e6\u05b4\u05d9\u05dd", "\u05d5\u05bc\u05e9\u05c1\u05b0\u05de\u05b9\u05e0\u05b4\u05bd\u05d9\u05dd", "\u05d4\u05b4\u05e9\u05c2\u05b0\u05ea\u05bc\u05b8\u05e8\u05b5\u05bd\u05e8", "\u05e0\u05b9\u05e4\u05b6\u05dc\u05b6\u05ea", "\u05d1\u05bc\u05b0\u05de\u05b4\u05d1\u05b0\u05e6\u05b8\u05e8\u05b4\u05bd\u05d9\u05dd", "\u05d1\u05bc\u05b0\u05de\u05b4\u05e9\u05c1\u05b0\u05e4\u05bc\u05b7\u05d8", "\u05d5\u05b0\u05d4\u05b4\u05e7\u05b0\u05e8\u05b4\u05d9\u05ea\u05b6\u05dd", "\u05d5\u05b7\u05bd\u05d9\u05bc\u05b7\u05db\u05bc\u05b0\u05ea\u05d5\u05bc\u05dd", "\u05e9\u05b8\u05c1\u05e0\u05b6\u05d4", "\u05d5\u05bc\u05dc\u05b0\u05de\u05b7\u05e1\u05bc\u05b7\u05e2", "\u05d9\u05b8\u05e9\u05c1\u05d5\u05bc\u05d1", "\u05e2\u05b2\u05de\u05b8\u05dc\u05b5\u05e7", "\u05db\u05b0\u05bc\u05de\u05d5\u05b9", "\u05de\u05b0\u05e0\u05b7\u05d0\u05b2\u05e6\u05b7\u05d9", "\u05d6\u05b9\u05d0\u05ea", "\u05d5\u05b0\u05d0\u05b7\u05d7\u05b2\u05e8\u05b4\u05d9\u05ea\u05d5\u05b9", "\u05d1\u05b8\u05bc\u05e7\u05b8\u05e8", "\u05d5\u05bc\u05e0\u05b0\u05e9\u05c2\u05b4\u05d9\u05d0", "\u05d5\u05bc\u05d1\u05b5\u05bd\u05d9\u05ea\u05b0\u05db\u05b6\u05dd", "\u05d4\u05b7\u05de\u05bc\u05b4\u05d3\u05b0\u05d9\u05b8\u05e0\u05b4\u05d9", "\u05d5\u05bc\u05de\u05b4\u05db\u05b0\u05e1\u05b8\u05dd", "\u05dc\u05b4\u05d1\u05b0\u05e0\u05b5\u05bd\u05d9", "\u05dc\u05b8\u05e2\u05b9\u05dc\u05b8\u05d4", "\u05e7\u05b0\u05d8\u05b9\u05e8\u05b6\u05ea", "\u05d5\u05b0\u05d3\u05b7\u05dd", "\u05d2\u05b0\u05d3\u05d5\u05b9\u05dc\u05b8\u05bd\u05d4", "\u05dc\u05b0\u05de\u05b7\u05e2\u05b2\u05dc\u05b5\u05d4", "\u05d0\u05b8\u05d5\u05b6\u05df", "\u05d4\u05b8\u05e2\u05b2\u05d1\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05e0\u05b7\u05e4\u05b0\u05e9\u05c1\u05d5\u05b9", "\u05db\u05bc\u05b7\u05d0\u05b2\u05e9\u05c1\u05b6\u05e8", "\u05d9\u05b0\u05bd\u05e9\u05c1\u05b8\u05e8\u05b0\u05ea\u05d5\u05bc", "\u05d5\u05bc\u05ea\u05b0\u05e0\u05d5\u05bc", "\u05d0\u05b4\u05d9\u05ea\u05b8\u05de\u05b8\u05bd\u05e8", "\u05ea\u05b8\u05bc\u05de\u05b4\u05d9\u05d3", "\u05d3\u05b8\u05bc\u05e8\u05b7\u05da\u05b0", "\u05d4\u05b8\u05e2\u05b2\u05e9\u05c2\u05bb\u05d9\u05b8\u05d4", "\u05dc\u05b8\u05d0\u05b6\u05d7\u05b8\u05d3", "\u05dc\u05b0\u05d1\u05d5\u05b9\u05e0\u05b8\u05d4", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05da\u05b0", "\u05d4\u05b6\u05e2\u05b1\u05dc\u05b4\u05d9\u05ea\u05b8", "\u05dc\u05b7\u05de\u05bc\u05b7\u05d8\u05bc\u05b6\u05d4", "\u05e0\u05b0\u05e7\u05b5\u05d1\u05b8\u05d4", "\u05d0\u05d5\u05b9\u05d9\u05b5\u05d1", "\u05d7\u05b6\u05e1\u05b6\u05d3", "\u05e2\u05b9\u05dc\u05b8\u05ea\u05b6\u05da\u05b8", "\u05d9\u05b4\u05ea\u05b0\u05e7\u05b8\u05e2\u05d5\u05bc", "\u05dc\u05b8\u05bd\u05d0\u05b4\u05e9\u05c1\u05bc\u05b8\u05d4", "\u05d1\u05b0\u05d0\u05b9\u05ea\u05b9\u05ea", "\u05ea\u05b4\u05e6\u05b0\u05dc\u05b8\u05bd\u05d7", "\u05d1\u05b8\u05bc\u05e9\u05b8\u05c1\u05df", "\u05d5\u05bc\u05d1\u05b0\u05d1\u05b9\u05d0", "\u05d5\u05bc\u05de\u05b4\u05e9\u05c1\u05b0\u05de\u05b6\u05e8\u05b6\u05ea", "\u05e6\u05d5\u05bc\u05e2\u05b8\u05e8", "\u05e9\u05b8\u05c1\u05d7\u05b7\u05d8", "\u05d4\u05b7\u05e0\u05bc\u05b0\u05e9\u05c2\u05b4\u05d9\u05d0\u05b4\u05d9\u05dd", "\u05d4\u05b8\u05bd\u05d0\u05b7\u05e9\u05c2\u05b0\u05e8\u05b4\u05bd\u05d0\u05b5\u05dc\u05b4\u05d9", "\u05d5\u05b0\u05d0\u05b7\u05d9\u05b4\u05dc", "\u05d1\u05bc\u05b0\u05de\u05b9\u05e2\u05b2\u05d3\u05b5\u05d9\u05db\u05b6\u05dd", "\u05d4\u05b4\u05e1\u05b0\u05db\u05bc\u05b7\u05e0\u05b0\u05ea\u05bc\u05b4\u05d9", "\u05d5\u05b0\u05d4\u05b5\u05dd", "\u05e8\u05b7\u05d1", "\u05db\u05bc\u05b7\u05d0\u05b2\u05e8\u05b4\u05d9", "\u05d4\u05b7\u05e1\u05bc\u05b7\u05e8\u05b0\u05d3\u05bc\u05b4\u05d9", "\u05d5\u05b6\u05bd\u05d0\u05b1\u05e1\u05b8\u05e8\u05b6\u05d4\u05b8", "\u05d4\u05bd\u05b6\u05e2\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05d0\u05b4\u05d9\u05e2\u05b6\u05d6\u05b6\u05e8", "\u05d5\u05bc\u05de\u05b4\u05e7\u05b0\u05e0\u05b7\u05d9", "\u05db\u05bc\u05b8\u05db\u05b8\u05d4", "\u05d9\u05b8\u05e2", "\u05e7\u05b7\u05e8\u05b0\u05e9\u05c1\u05b5\u05d9", "\u05d1\u05bc\u05b0\u05e2\u05b4\u05d9\u05e8\u05b8\u05bd\u05dd", "\u05e1\u05b0\u05d1\u05b4\u05d9\u05d1\u05b9\u05ea\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05d7\u05d5\u05bc\u05dc", "\u05de\u05b4\u05db\u05b0\u05e8\u05b8\u05dd", "\u05de\u05b4\u05e0\u05b0\u05d7\u05b8\u05ea\u05b8\u05d4\u05bc", "\u05d0\u05b2\u05d3\u05b8\u05de\u05b8\u05d4", "\u05de\u05b5\u05d4\u05b8\u05e8\u05b5\u05d9", "\u05d4\u05b7\u05de\u05bc\u05b8\u05db\u05b4\u05d9\u05e8\u05b4\u05d9", "\u05e7\u05b8\u05e6\u05b7\u05e8", "\u05d0\u05b2\u05de\u05b7\u05e8\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05db\u05bc\u05b8\u05e8\u05b0\u05ea\u05d5\u05bc", "\u05d4\u05b7\u05e2\u05b5\u05d9\u05e0\u05b5\u05d9", "\u05d1\u05bc\u05b7\u05d3\u05bc\u05b8\u05bd\u05e8\u05b6\u05da\u05b0", "\u05dc\u05b0\u05e6\u05b4\u05d1\u05b0\u05d0\u05b9\u05ea\u05b8\u05dd", "\u05e0\u05b4\u05d8\u05b0\u05de\u05b5\u05d0\u05ea", "\u05d5\u05b0\u05e8\u05b8\u05d0\u05b4\u05d9\u05ea\u05b8\u05d4", "\u05ea\u05bc\u05b0\u05dc\u05bb\u05e0\u05bc\u05d5\u05b9\u05ea", "\u05d5\u05b0\u05d0\u05b6\u05e9\u05c1\u05b0\u05db\u05bc\u05d5\u05b9\u05dc", "\u05e2\u05b8\u05e0\u05b8\u05d9\u05d5", "\u05e8\u05b8\u05d0\u05b8\u05d4", "\u05d7\u05b7\u05d2", "\u05d8\u05b7\u05d1\u05bc\u05b7\u05e2\u05b7\u05ea", "\u05d5\u05b0\u05d4\u05b4\u05e0\u05bc\u05b4\u05d9\u05d7\u05b7", "\u05de\u05b7\u05d8\u05bc\u05b5\u05d4\u05d5\u05bc", "\u05d4\u05b7\u05db\u05bc\u05b7\u05e4\u05bc\u05b9\u05ea", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05dc\u05b0\u05d1\u05bc\u05b5\u05e9\u05c1", "\u05d5\u05bd\u05b6\u05d0\u05b1\u05e1\u05b8\u05e8\u05b8\u05d4\u05bc", "\u05d5\u05b0\u05d2\u05b7\u05dd", "\u05e2\u05b8\u05e9\u05b8\u05c2\u05d4", "\u05d0\u05b7\u05bd\u05d4\u05b2\u05e8\u05b9\u05df", "\u05d6\u05b8\u05db\u05b8\u05e8", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e7\u05bc\u05b8\u05d7\u05b5\u05d4\u05d5\u05bc", "\u05d9\u05b0\u05e8\u05b4\u05d9\u05e2\u05b9\u05ea", "\u05d1\u05b6\u05bd\u05d2\u05b6\u05d3", "\u05e8\u05b8\u05e4\u05bd\u05d5\u05bc\u05d0", "\u05e7\u05b4\u05e0\u05bc\u05b6\u05bd\u05da\u05b8", "\u05ea\u05b8\u05e9\u05c1\u05b5\u05ea", "\u05e2\u05b2\u05d2\u05b8\u05dc\u05b8\u05d4", "\u05e8\u05b4\u05d1\u05bc\u05b5\u05e2\u05b4\u05bd\u05d9\u05dd", "\u05d5\u05bc\u05e0\u05b0\u05d0\u05bb\u05dd", "\u05d5\u05bc\u05de\u05b4\u05d9\u05bc\u05b4\u05e9\u05c2\u05b0\u05e8\u05b8\u05d0\u05b5\u05dc", "\u05d9\u05b8\u05dc\u05b7\u05d3", "\u05d5\u05b0\u05db\u05b4\u05d1\u05bc\u05b7\u05e1\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05d0\u05b8\u05de\u05b5\u05df", "\u05e2\u05b2\u05e8\u05b8\u05d3", "\u05de\u05b4\u05e7\u05b0\u05dc\u05b8\u05d8", "\u05d0\u05b6\u05d7\u05b8\u05d9\u05d5", "\u05d5\u05b7\u05ea\u05bc\u05b4\u05e8\u05b0\u05d1\u05bc\u05b7\u05e5", "\u05de\u05b5\u05e2\u05b8\u05d5\u05ba\u05df", "\u05e1\u05d5\u05bc\u05e3", "\u05d3\u05b4\u05bc\u05d9\u05d1\u05d5\u05b9\u05df", "\u05d5\u05bc\u05d1\u05b0\u05e0\u05b9\u05ea\u05b8\u05d9\u05d5", "\u05d5\u05b0\u05d3\u05b4\u05d1\u05bc\u05b7\u05e8\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05e0\u05b4\u05d3\u05b0\u05e8\u05d5\u05b9", "\u05d5\u05b4\u05bd\u05d9\u05ea\u05b5\u05d3\u05b9\u05ea\u05b8\u05dd", "\u05d5\u05b0\u05d6\u05b4\u05e7\u05b0\u05e0\u05b5\u05d9", "\u05dc\u05b4\u05e9\u05c1\u05b0\u05d0\u05b5\u05e8\u05d5\u05b9", "\u05de\u05b7\u05e6\u05b8\u05bc\u05d4", "\u05d5\u05bc\u05d1\u05b0\u05e9\u05c2\u05b8\u05e8\u05b8\u05dd", "\u05de\u05b4\u05bd\u05e0\u05bc\u05b7\u05d7\u05b2\u05dc\u05b7\u05ea", "\u05d5\u05b8\u05d0\u05b8\u05dc\u05b6\u05e3", "\u05e9\u05b0\u05c2\u05de\u05b9\u05d0\u05d5\u05dc", "\u05d0\u05b8\u05e8\u05d5\u05b9\u05df", "\u05e2\u05b8\u05e0\u05b8\u05df", "\u05d1\u05b0\u05d7\u05b4\u05d9\u05d3\u05b9\u05ea", "\u05d9\u05b4\u05e1\u05bc\u05b8\u05bd\u05e2\u05d5\u05bc", "\u05d7\u05bb\u05e6\u05bd\u05d5\u05b9\u05ea", "\u05d5\u05b7\u05d9\u05bc\u05b8\u05e8\u05b6\u05d1", "\u05de\u05b4\u05e9\u05c1\u05b0\u05de\u05b6\u05e8\u05b6\u05ea", "\u05e9\u05c1\u05b0\u05dc\u05bb\u05e4\u05b8\u05d4", "\u05d4\u05b8\u05bd\u05d0\u05b4\u05e9\u05c1\u05bc\u05b6\u05d4", "\u05e2\u05b7\u05ea\u05b8\u05bc\u05d4", "\u05dc\u05b0\u05d9\u05b8\u05de\u05b4\u05d9\u05df", "\u05e8\u05b4\u05bd\u05d1\u05b0\u05d1\u05d5\u05b9\u05ea", "\u05db\u05b4\u05bc\u05ea\u05b4\u05bc\u05d9", "\u05d5\u05b7\u05d9\u05bc\u05d5\u05b9\u05e8\u05b6\u05e9\u05c1", "\u05e6\u05b8\u05d1\u05b8\u05d0", "\u05d4\u05b7\u05de\u05b0\u05d0\u05b8\u05e8\u05b0\u05e8\u05b4\u05d9\u05dd", "\u05d1\u05bc\u05b0\u05d4\u05b6\u05de\u05b0\u05ea\u05bc\u05b8\u05dd", "\u05e9\u05c1\u05b0\u05d1\u05d5\u05bc", "\u05d1\u05b0\u05d3\u05b6\u05e8\u05b6\u05da\u05b0", "\u05de\u05b4\u05d2\u05b0\u05e8\u05b8\u05e9\u05c1", "\u05d5\u05b0\u05e2\u05b6\u05e9\u05c2\u05b0\u05e8\u05b4\u05d9\u05dd", "\u05d4\u05b8\u05d0\u05b2\u05ea\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05d5\u05b0\u05e2\u05b4\u05e9\u05c2\u05bc\u05b8\u05e8\u05b9\u05df", "\u05dc\u05b0\u05d7\u05b7\u05d8\u05bc\u05b8\u05bd\u05ea", "\u05e2\u05b9\u05dc\u05b8\u05d4", "\u05d7\u05bb\u05e7\u05b8\u05bc\u05d4", "\u05d4\u05b7\u05e9\u05c2\u05bc\u05b0\u05e8\u05bb\u05e4\u05b4\u05d9\u05dd", "\u05e6\u05b9\u05e4\u05b4\u05d9\u05dd", "\u05d1\u05bc\u05b0\u05de\u05b8\u05e8\u05b8\u05bd\u05d4", "\u05d4\u05b7\u05d1\u05bc\u05b4\u05db\u05bc\u05d5\u05bc\u05e8\u05b4\u05d9\u05dd", "\u05dc\u05b0\u05e0\u05b7\u05d7\u05b5\u05dc", "\u05d4\u05b8\u05e2\u05b5\u05e8\u05b4\u05bd\u05d9", "\u05d9\u05b4\u05d8\u05b0\u05d4\u05b8\u05bd\u05e8", "\u05d5\u05bc\u05d2\u05b0\u05dc\u05d5\u05bc\u05d9", "\u05db\u05bc\u05b0\u05d4\u05bb\u05e0\u05bc\u05b7\u05ea\u05b0\u05db\u05b6\u05bd\u05dd", "\u05d0\u05b6\u05e4\u05b6\u05e1", "\u05d4\u05b7\u05e0\u05bc\u05b0\u05de\u05d5\u05bc\u05d0\u05b5\u05dc\u05b4\u05d9", "\u05ea\u05b0\u05bc\u05e0\u05d5\u05bc\u05e4\u05b8\u05d4", "\u05d0\u05b6\u05dc\u05b4\u05d9\u05e6\u05b8\u05e4\u05b8\u05df", "\u05db\u05b8\u05bc\u05e1\u05b8\u05d4", "\u05db\u05bc\u05b7\u05e9\u05c1\u05bc\u05b8\u05dc\u05b6\u05d2", "\u05e6\u05b8\u05e4\u05b9\u05e0\u05b8\u05d4", "\u05d5\u05bc\u05db\u05b0\u05d3\u05b6\u05e8\u05b6\u05da\u05b0", "\u05d6\u05b8\u05d1", "\u05d4\u05b8\u05d0\u05b9\u05e1\u05b5\u05e3", "\u05dc\u05b0\u05e9\u05c1\u05b4\u05dc\u05bc\u05b5\u05dd", "\u05dc\u05b7\u05e6\u05b0\u05d1\u05bc\u05d5\u05b9\u05ea", "\u05e0\u05b7\u05d7\u05b2\u05dc\u05b8\u05ea\u05b5\u05e0\u05d5\u05bc", "\u05de\u05b4\u05d3\u05bc\u05b4\u05d9\u05d1\u05b9\u05df", "\u05d5\u05b0\u05db\u05b7\u05e2\u05b2\u05d3\u05b8\u05ea\u05d5\u05b9", "\u05db\u05b8\u05bc\u05e0\u05b8\u05e3", "\u05d5\u05bc\u05e9\u05c1\u05b0\u05dc\u05b4\u05d9\u05e9\u05c1\u05b4\u05ea", "\u05e4\u05b7\u05bc\u05e2\u05b7\u05dd", "\u05d5\u05bc\u05d6\u05b0\u05db\u05b7\u05e8\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05d1\u05bc\u05b4\u05e9\u05c2\u05b0\u05d3\u05b5\u05d4", "\u05e8\u05b7\u05e2", "\u05de\u05b9\u05e9\u05c1\u05b6\u05bd\u05d4", "\u05dc\u05b0\u05e4\u05b4\u05d9", "\u05d9\u05b7\u05d0\u05b2\u05de\u05b4\u05d9\u05e0\u05d5\u05bc", "\u05dc\u05b0\u05e2\u05b5\u05d9\u05e0\u05b5\u05d9\u05d4\u05b6\u05bd\u05dd", "\u05d1\u05bc\u05b0\u05e8\u05b7\u05d2\u05b0\u05dc\u05b7\u05d9", "\u05d1\u05bc\u05b0\u05d0\u05b7\u05e8\u05b0\u05d1\u05bc\u05b8\u05e2\u05b8\u05d4", "\u05dc\u05b8\u05db\u05b5\u05df", "\u05e0\u05b0\u05d7\u05b9\u05e9\u05b6\u05c1\u05ea", "\u05d4\u05b7\u05e0\u05bc\u05b9\u05d2\u05b7\u05e2\u05b7\u05ea", "\u05d5\u05b7\u05ea\u05bc\u05b4\u05ea\u05bc\u05b6\u05df", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e9\u05c1\u05b0\u05d7\u05b8\u05d8\u05b5\u05dd", "\u05de\u05b5\u05e2\u05b4\u05bd\u05d9\u05e8", "\u05d4\u05b7\u05de\u05bc\u05b9\u05e6\u05b0\u05d0\u05b4\u05d9\u05dd", "\u05d9\u05b8\u05e9\u05c1\u05b4\u05d9\u05d1\u05d5\u05bc", "\u05db\u05bc\u05b0\u05de\u05d5\u05b9\u05ea", "\u05d5\u05b4\u05d9\u05ea\u05b5\u05d3\u05b9\u05ea\u05b8\u05dd", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05e7\u05b0\u05d4\u05b4\u05dc\u05d5\u05bc", "\u05d9\u05b0\u05d1\u05b4\u05d9\u05d0\u05b5\u05dd", "\u05d1\u05bc\u05b0\u05e0\u05b7\u05d7\u05b2\u05dc\u05b7\u05ea", "\u05d4\u05b7\u05d6\u05bc\u05b6\u05d4", "\u05e9\u05b8\u05c1\u05de\u05b7\u05e2", "\u05d5\u05bc\u05d1\u05b0\u05d4\u05b7\u05d0\u05b2\u05e8\u05b4\u05d9\u05da\u05b0", "\u05db\u05b6\u05bc\u05d1\u05b6\u05e9\u05c2", "\u05d7\u05b9\u05ea\u05b5\u05df", "\u05d5\u05bc\u05e9\u05c1\u05b0\u05e0\u05b5\u05d9", "\u05d4\u05b6\u05d7\u05b8\u05d6\u05b8\u05e7", "\u05de\u05b8\u05bd\u05d9\u05b4\u05dd", "\u05d5\u05b0\u05e1\u05b8\u05d1\u05b4\u05d9\u05d1", "\u05e0\u05b4\u05d2\u05bc\u05b8\u05e8\u05b7\u05e2", "\u05e6\u05d5\u05bc\u05e8\u05b4\u05d9\u05d0\u05b5\u05dc", "\u05d4\u05b7\u05db\u05bc\u05b8\u05e0\u05b8\u05e3", "\u05d5\u05bc\u05dc\u05b0\u05de\u05b7\u05d8\u05bc\u05b5\u05d4", "\u05e2\u05b7\u05d8\u05b0\u05e8\u05b9\u05ea", "\u05de\u05d5\u05b9\u05e2\u05b5\u05d3", "\u05db\u05bc\u05b0\u05dc\u05b4\u05d9", "\u05de\u05d5\u05b9\u05e9\u05b8\u05c1\u05d1", "\u05e7\u05b8\u05e8\u05b7\u05d1", "\u05dc\u05b0\u05de\u05b4\u05e9\u05c1\u05b0\u05db\u05bc\u05b7\u05d1", "\u05e7\u05b3\u05d1\u05b8\u05ea\u05b8\u05d4\u05bc", "\u05d2\u05bc\u05b0\u05d0\u05d5\u05bc\u05d0\u05b5\u05dc", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e7\u05b0\u05d3\u05bc\u05b8\u05e9\u05c1\u05d5\u05bc", "\u05e0\u05d5\u05bc\u05e3", "\u05d1\u05b7\u05bc\u05e2\u05b7\u05dc", "\u05d9\u05d5\u05bc\u05d0\u05b8\u05bd\u05e8", "\u05e6\u05b4\u05e4\u05bc\u05b9\u05bd\u05e8", "\u05d4\u05b8\u05bd\u05e2\u05b7\u05e8\u05b0\u05d1\u05bc\u05b8\u05d9\u05b4\u05dd", "\u05e1\u05b8\u05d1\u05b7\u05d1", "\u05e4\u05bc\u05b7\u05dc\u05b0\u05d8\u05b4\u05d9\u05d0\u05b5\u05dc", "\u05d5\u05b0\u05d4\u05b7\u05de\u05bc\u05b0\u05e0\u05b9\u05e8\u05b8\u05d4", "\u05d5\u05b0\u05d4\u05b6\u05e2\u05b8\u05e0\u05b8\u05df", "\u05de\u05b4\u05d3\u05b8\u05bc\u05d4", "\u05d5\u05bc\u05bd\u05d1\u05b0\u05de\u05d5\u05b9\u05e2\u05b2\u05d3\u05b5\u05d9\u05db\u05b6\u05dd", "\u05d1\u05bc\u05b0\u05d4\u05b7\u05e2\u05b2\u05dc\u05b9\u05bd\u05ea\u05b0\u05da\u05b8", "\u05d0\u05b5\u05dc\u05b5\u05bd\u05da\u05b0", "\u05e9\u05b8\u05c2\u05de\u05b7\u05d7", "\u05d6\u05b7\u05db\u05bc\u05bd\u05d5\u05bc\u05e8", "\u05d5\u05bc\u05d1\u05b7\u05de\u05bc\u05b4\u05d3\u05b0\u05d1\u05bc\u05b8\u05e8", "\u05dc\u05b0\u05de\u05b4\u05e0\u05b0\u05d7\u05b8\u05d4", "\u05d1\u05b8\u05bc\u05d7\u05b7\u05e8", "\u05d5\u05bc\u05de\u05b4\u05e1\u05b0\u05e4\u05bc\u05b8\u05e8", "\u05d7\u05b6\u05d1\u05b0\u05e8\u05d5\u05b9\u05df", "\u05de\u05d5\u05b9\u05e2\u05b5\u05bd\u05d3", "\u05e2\u05b7\u05de\u05bc\u05d5\u05b9\u05df", "\u05de\u05b4\u05d2\u05bc\u05b9\u05d0\u05b5\u05dc", "\u05d5\u05bc\u05d1\u05b0\u05e2\u05b9\u05bd\u05df", "\u05d5\u05bc\u05de\u05b8\u05d0\u05ea\u05b7\u05d9\u05b4\u05dd", "\u05d5\u05b0\u05db\u05b4\u05dc\u05bc\u05b8\u05d4", "\u05e0\u05b8\u05e9\u05c1\u05b5\u05d9\u05e0\u05d5\u05bc", "\u05d0\u05b4\u05e7\u05bc\u05b8\u05e8\u05b6\u05d4", "\u05dc\u05b0\u05d1\u05b4\u05ea\u05bc\u05bd\u05d5\u05b9", "\u05d5\u05b7\u05ea\u05bc\u05b0\u05d3\u05b7\u05d1\u05bc\u05b5\u05e8", "\u05d9\u05b0\u05de\u05bb\u05ea\u05d5\u05bc\u05df", "\u05d4\u05b7\u05de\u05b0\u05d0\u05bd\u05b8\u05e8\u05b2\u05e8\u05b4\u05d9\u05dd", "\u05dc\u05b0\u05d4\u05b7\u05db\u05bc\u05b9\u05ea\u05b8\u05bd\u05d4\u05bc", "\u05d9\u05d5\u05b9\u05db\u05b6\u05d1\u05b6\u05d3", "\u05dc\u05b4\u05e1\u05b0\u05d1\u05b9\u05d1", "\u05d1\u05b0\u05db\u05b9\u05d7\u05b2\u05da\u05b8", "\u05d6\u05b8\u05e0\u05b8\u05d4", "\u05d4\u05bb\u05db\u05bc\u05b8\u05d4", "\u05d1\u05b8\u05e7\u05b8\u05e8", "\u05d1\u05b0\u05e7\u05b8\u05d1\u05b6\u05e8", "\u05de\u05c7\u05d7\u05b3\u05e8\u05b8\u05ea", "\u05d0\u05b6\u05d1\u05b0\u05d7\u05b7\u05e8", "\u05d4\u05b7\u05e4\u05bc\u05b8\u05e8\u05b8\u05d4", "\u05d0\u05b8\u05d1\u05d5\u05b9\u05ea", "\u05d5\u05b7\u05d9\u05b0\u05e6\u05b7\u05d5", "\u05d5\u05b7\u05d9\u05bc\u05b8\u05e8\u05b6\u05dd", "\u05d5\u05b0\u05e0\u05b9\u05d1\u05b7\u05d7", "\u05d4\u05b8\u05e8\u05b9\u05e6\u05b5\u05bd\u05d7\u05b7", "\u05d2\u05bc\u05b9\u05e8\u05b6\u05df", "\u05d1\u05bc\u05b4\u05e6\u05b0\u05d3\u05b4\u05d9\u05bc\u05b8\u05d4", "\u05d4\u05b8\u05e8\u05b2\u05db\u05bd\u05d5\u05bc\u05e9\u05c1", "\u05de\u05b7\u05e1\u05b7\u05bc\u05e2", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05e2\u05b0\u05e4\u05bc\u05b4\u05dc\u05d5\u05bc", "\u05d5\u05b0\u05e0\u05bd\u05d5\u05b9\u05e2\u05b2\u05d3\u05d5\u05bc", "\u05db\u05b8\u05bc\u05dc\u05b5\u05d1", "\u05d1\u05bc\u05b0\u05d9\u05b8\u05d3\u05d5\u05b9", "\u05d4\u05b7\u05d2\u05bc\u05b4\u05d3\u05b0\u05d2\u05bc\u05b8\u05d3", "\u05dc\u05b8\u05d7\u05b7\u05dd", "\u05d5\u05b0\u05ea\u05b4\u05db\u05bc\u05d5\u05b9\u05e0\u05b5\u05df", "\u05d9\u05b4\u05ea\u05b0\u05d7\u05b7\u05d8\u05bc\u05b8\u05d0", "\u05d4\u05b7\u05db\u05bc\u05b0\u05d1\u05b8\u05e9\u05c2\u05b4\u05bd\u05d9\u05dd", "\u05db\u05b8\u05bc\u05e4\u05b7\u05e8", "\u05d5\u05bc\u05de\u05b0\u05e8\u05b9\u05e8\u05b4\u05d9\u05dd", "\u05de\u05b7\u05ea\u05bc\u05b8\u05e0\u05b8\u05dd", "\u05d5\u05b0\u05d4\u05d5\u05b9\u05e7\u05b7\u05e2", "\u05d1\u05b0\u05e9\u05c1\u05b5\u05de\u05b9\u05ea", "\u05d2\u05b8\u05bc\u05d3\u05d5\u05b9\u05dc", "\u05d6\u05b9\u05e2\u05b2\u05de\u05b8\u05d4", "\u05d4\u05b7\u05d9\u05bc\u05b4\u05e6\u05b0\u05d4\u05b8\u05e8\u05b4\u05d9", "\u05d0\u05b1\u05dc\u05b4\u05d9\u05e6\u05b8\u05e4\u05b8\u05df", "\u05d0\u05b1\u05dc\u05b4\u05bd\u05d9\u05e9\u05c1\u05b8\u05de\u05b8\u05e2", "\u05e9\u05b4\u05c1\u05d1\u05b0\u05e2\u05b4\u05d9\u05dd", "\u05d7\u05bb\u05e7\u05bc\u05b8\u05d4", "\u05ea\u05bc\u05b7\u05e2\u05b2\u05d1\u05b4\u05e8\u05b5\u05e0\u05d5\u05bc", "\u05d5\u05b0\u05e9\u05c1\u05d5\u05b9\u05e8", "\u05d5\u05b7\u05d9\u05bc\u05b8\u05e7\u05b8\u05dd", "\u05e4\u05b6\u05e1\u05b7\u05d7", "\u05d1\u05bc\u05b0\u05de\u05b7\u05e9\u05c2\u05bc\u05b8\u05d0", "\u05d1\u05bc\u05b0\u05dc\u05d5\u05bc\u05dc\u05b8\u05d4", "\u05d7\u05b2\u05e0\u05b9\u05ea\u05b5\u05e0\u05d5\u05bc", "\u05d0\u05b1\u05dc\u05b4\u05d9\u05d3\u05b8\u05d3", "\u05dc\u05b8\u05e7\u05b7\u05d7", "\u05d1\u05bc\u05b0\u05e7\u05b4\u05e0\u05b0\u05d0\u05b8\u05ea\u05b4\u05bd\u05d9", "\u05d4\u05b1\u05d9\u05d5\u05bc", "\u05d0\u05b6\u05e8\u05b6\u05e5", "\u05de\u05b4\u05e7\u05bc\u05b4\u05d9\u05e8", "\u05d4\u05b7\u05e0\u05bc\u05b4\u05d3\u05bc\u05b8\u05d4", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05dc\u05bc\u05b9\u05e0\u05d5\u05bc", "\u05e0\u05b8\u05e7\u05b8\u05d4", "\u05de\u05b7\u05e2\u05b2\u05e9\u05c2\u05b5\u05e8", "\u05d5\u05b7\u05ea\u05bc\u05b4\u05e8\u05b0\u05d0\u05b7\u05e0\u05b4\u05d9", "\u05d0\u05b8\u05de\u05b7\u05e8", "\u05dc\u05b0\u05de\u05b4\u05e7\u05b0\u05e8\u05b8\u05d0", "\u05db\u05bc\u05bd\u05b7\u05d0\u05b2\u05e9\u05c1\u05b6\u05e8", "\u05d9\u05b8\u05e8\u05b7\u05d8", "\u05de\u05b4\u05de\u05bc\u05b7\u05bd\u05d7\u05b2\u05e6\u05b4\u05d9\u05ea\u05b8\u05dd", "\u05e1\u05d5\u05b9\u05d3\u05b4\u05bd\u05d9", "\u05d1\u05b7\u05de\u05bc\u05b8\u05bd\u05d9\u05b4\u05dd", "\u05d5\u05b7\u05bd\u05d9\u05bc\u05b9\u05d0\u05de\u05b0\u05e8\u05d5\u05bc", "\u05d5\u05b8\u05e4\u05b8\u05e9\u05c1\u05b7\u05e2", "\u05ea\u05bc\u05b4\u05d9\u05e8\u05b8\u05d0\u05bb\u05bd\u05dd", "\u05d4\u05b7\u05dc\u05bc\u05b8\u05d9\u05b0\u05dc\u05b8\u05d4", "\u05de\u05b0\u05e8\u05b4\u05d9\u05d1\u05b8\u05d4", "\u05de\u05b4\u05e8\u05b0\u05d9\u05b8\u05bd\u05dd", "\u05d1\u05bc\u05b0\u05e7\u05b8\u05d3\u05b5\u05e9\u05c1", "\u05e9\u05b8\u05c1\u05e4\u05b7\u05d8", "\u05d4\u05b7\u05d2\u05bc\u05b8\u05d3\u05d5\u05b9\u05dc", "\u05d4\u05b7\u05e0\u05bc\u05b0\u05e4\u05b4\u05d9\u05dc\u05b4\u05d9\u05dd", "\u05d5\u05b0\u05d3\u05b4\u05d1\u05bc\u05b6\u05e8", "\u05d3\u05bc\u05b4\u05d1\u05bc\u05b7\u05ea", "\u05d7\u05b5\u05e6\u05b4\u05d9", "\u05d4\u05b5\u05e2\u05b8\u05dc\u05b9\u05ea", "\u05d1\u05b8\u05e8\u05b5\u05bd\u05da\u05b0", "\u05d7\u05b2\u05e6\u05bd\u05d5\u05b9\u05e6\u05b0\u05e8\u05b9\u05ea", "\u05d5\u05b0\u05e0\u05b8\u05bd\u05e1\u05b0\u05e2\u05d5\u05bc", "\u05ea\u05b0\u05bc\u05e8\u05d5\u05bc\u05e2\u05b8\u05d4", "\u05d7\u05b5\u05dc\u05b9\u05bd\u05df", "\u05d2\u05bc\u05b5\u05e8\u05b0\u05e9\u05c1\u05d5\u05b9\u05df", "\u05d0\u05b9\u05bd\u05d9\u05b0\u05d1\u05b6\u05d9\u05da\u05b8", "\u05d1\u05b5\u05bc\u05d9\u05df", "\u05e2\u05b4\u05d9\u05e8", "\u05e9\u05b4\u05c1\u05dc\u05b5\u05bc\u05e9\u05c1", "\u05d7\u05b6\u05dc\u05b0\u05d1\u05bc\u05d5\u05b9", "\u05d1\u05bc\u05b0\u05d9\u05b8\u05d8\u05b0\u05d1\u05b8\u05bd\u05ea\u05b8\u05d4", "\u05d9\u05b0\u05de\u05b4\u05d9\u05ea\u05b6\u05bd\u05e0\u05bc\u05d5\u05bc", "\u05e2\u05b8\u05e9\u05b8\u05c2\u05e8", "\u05d5\u05bc\u05d1\u05b0\u05e8\u05b4\u05d9\u05d7\u05b8\u05d9\u05d5", "\u05d0\u05d5\u05b9\u05d9", "\u05d0\u05b2\u05e9\u05c1\u05d5\u05bc\u05e8\u05b6\u05e0\u05bc\u05d5\u05bc", "\u05ea\u05bc\u05b7\u05d0\u05b2\u05d5\u05b8\u05d4", "\u05ea\u05bc\u05d5\u05b9\u05e6\u05b0\u05d0\u05b9\u05ea", "\u05e7\u05b8\u05e8\u05d5\u05b9\u05d1", "\u05e2\u05b7\u05ea\u05bc\u05bb\u05d3\u05b4\u05d9\u05dd", "\u05e4\u05b4\u05bc\u05d9\u05e0\u05b0\u05d7\u05b8\u05e1", "\u05db\u05bc\u05b7\u05de\u05bc\u05b4\u05e1\u05b0\u05e4\u05bc\u05b8\u05e8", "\u05de\u05b4\u05bd\u05d1\u05bc\u05b7\u05dc\u05b0\u05e2\u05b2\u05d3\u05b5\u05d9", "\u05d0\u05b2\u05d3\u05b7\u05d1\u05bc\u05b6\u05e8", "\u05e2\u05b8\u05e8\u05b5\u05d9", "\u05d4\u05b8\u05e2\u05b8\u05bd\u05d9\u05b4\u05df", "\u05d0\u05b6\u05dc", "\u05e4\u05bc\u05b8\u05e8\u05b9\u05db\u05b6\u05ea", "\u05d4\u05b7\u05d8\u05bc\u05b7\u05e3", "\u05d5\u05b0\u05d9\u05b4\u05e9\u05c1\u05b0\u05de\u05b0\u05e8\u05b6\u05bd\u05da\u05b8", "\u05dc\u05b8\u05d0\u05b7\u05d7\u05b2\u05e8\u05b9\u05e0\u05b8\u05d4", "\u05de\u05b4\u05d2\u05bc\u05b6\u05e4\u05b6\u05df", "\u05d1\u05bc\u05b4\u05e7\u05b0\u05d4\u05b5\u05dc\u05b8\u05bd\u05ea\u05b8\u05d4", "\u05d4\u05b4\u05d1\u05b0\u05d3\u05bc\u05b4\u05d9\u05dc", "\u05d0\u05b4\u05ea\u05bc\u05b4\u05d9", "\u05d7\u05b8\u05bd\u05d9\u05b4\u05dc", "\u05d5\u05b0\u05db\u05b8\u05dc\u05b5\u05d1", "\u05d5\u05b0\u05d4\u05b7\u05d7\u05b9\u05e0\u05b4\u05d9\u05dd", "\u05de\u05b7\u05d7\u05b0\u05dc\u05b8\u05d4", "\u05e9\u05c1\u05b8\u05bd\u05de\u05b0\u05e2\u05d5\u05bc", "\u05d5\u05b0\u05d4\u05b7\u05d1\u05bc\u05b8\u05e7\u05b8\u05e8", "\u05d1\u05bc\u05b4\u05dc\u05b0\u05e2\u05b8\u05dd", "\u05ea\u05bc\u05b4\u05de\u05b0\u05e8\u05b9\u05d3\u05d5\u05bc", "\u05d7\u05b5\u05d8\u05b0\u05d0", "\u05de\u05b0\u05d4\u05b5\u05e8\u05b8\u05d4", "\u05d1\u05bc\u05bd\u05b7\u05d7\u05b2\u05dc\u05b7\u05dc", "\u05d4\u05b7\u05de\u05bc\u05b4\u05dc\u05b0\u05d7\u05b8\u05de\u05b8\u05bd\u05d4", "\u05de\u05b4\u05e9\u05b0\u05c1\u05de\u05b6\u05e8\u05b6\u05ea", "\u05d1\u05bc\u05b4\u05e9\u05c1\u05b0\u05e0\u05b8\u05d9\u05b4\u05dd", "\u05e7\u05b0\u05e8\u05b4\u05d9\u05d0\u05b5\u05d9", "\u05de\u05b5\u05d0\u05b9\u05d9\u05b0\u05d1\u05b5\u05d9\u05db\u05b6\u05bd\u05dd", "\u05d5\u05b0\u05d6\u05b6\u05d4", "\u05d1\u05bc\u05b0\u05de\u05d5\u05b9\u05e2\u05b2\u05d3\u05d5\u05b9", "\u05de\u05b4\u05e7\u05bc\u05b5\u05e5", "\u05d1\u05bc\u05b8\u05de\u05b9\u05ea\u05b8\u05dd", "\u05ea\u05bc\u05b5\u05d7\u05b8\u05dc\u05b5\u05e7", "\u05dc\u05b4\u05e7\u05b0\u05d4\u05b8\u05ea", "\u05e0\u05b9\u05e4\u05b7\u05d7", "\u05d0\u05b7\u05d7\u05b2\u05e8\u05b4\u05d9\u05ea\u05b4\u05d9", "\u05e7\u05b0\u05e8\u05b4\u05d0\u05b5\u05d9", "\u05de\u05b7\u05db\u05b8\u05bc\u05d4", "\u05d6\u05b0\u05d1\u05d5\u05bc\u05dc\u05d5\u05bc\u05df", "\u05d5\u05b7\u05bd\u05d9\u05bc\u05b7\u05e2\u05b2\u05de\u05b5\u05d3", "\u05e1\u05b4\u05d9\u05d7\u05d5\u05b9\u05df", "\u05e0\u05b8\u05d3\u05b8\u05e8\u05b8\u05d4", "\u05d1\u05b8\u05bc\u05da\u05b0", "\u05e9\u05c1\u05b4\u05e9\u05c1\u05bc\u05b8\u05d4", "\u05d4\u05b7\u05bd\u05de\u05bc\u05b7\u05d7\u05b2\u05e0\u05d5\u05b9\u05ea", "\u05ea\u05b0\u05d3\u05b7\u05d1\u05bc\u05b5\u05e8", "\u05de\u05b7\u05d7\u05b0\u05ea\u05b8\u05bc\u05d4", "\u05d5\u05b0\u05d4\u05b7\u05e7\u05b0\u05d4\u05b5\u05dc", "\u05d5\u05b0\u05e0\u05b4\u05d6\u05b0\u05e8\u05b0\u05e2\u05b8\u05d4", "\u05db\u05bc\u05b0\u05d1\u05d5\u05b9\u05d3", "\u05ea\u05bc\u05b4\u05e9\u05c1\u05b0\u05dc\u05b8\u05d7\u05d5\u05bc", "\u05d1\u05b0\u05d0\u05b5\u05e8", "\u05d5\u05b0\u05d4\u05b7\u05e9\u05c2\u05bc\u05b9\u05e8\u05b5\u05e3", "\u05dc\u05b0\u05d0\u05b7\u05e4\u05bc\u05b8\u05bd\u05d9\u05d5", "\u05e6\u05b8\u05e8\u05b8\u05d9\u05d5", "\u05db\u05bc\u05b0\u05d2\u05b9\u05d3\u05b6\u05dc", "\u05e6\u05b0\u05dc\u05b8\u05e4\u05b0\u05d7\u05b8\u05bd\u05d3", "\u05d0\u05b8\u05d3\u05b8\u05dd", "\u05e9\u05b8\u05c1\u05dc\u05b7\u05d7", "\u05d1\u05bc\u05b0\u05db\u05b9\u05dc", "\u05de\u05b4\u05de\u05bc\u05b9\u05e1\u05b5\u05e8\u05d5\u05b9\u05ea", "\u05d4\u05b7\u05e0\u05bc\u05b7\u05e2\u05b7\u05e8", "\u05de\u05bd\u05b7\u05e2\u05b2\u05e9\u05c2\u05b5\u05e8", "\u05e9\u05c1\u05d5\u05b9\u05e4\u05b8\u05df", "\u05d1\u05bc\u05b8\u05bd\u05d0\u05b7\u05de\u05bc\u05b8\u05d4", "\u05e6\u05bb\u05e8\u05b4\u05d9\u05dd", "\u05d5\u05b0\u05dc\u05b4\u05d1\u05b0\u05e0\u05b5\u05d9", "\u05e0\u05b8\u05db\u05b8\u05d4", "\u05e2\u05b2\u05e8\u05b4\u05e1\u05b9\u05ea\u05b5\u05d9\u05db\u05b6\u05dd", "\u05d0\u05b7\u05de\u05bc\u05b8\u05d4", "\u05d4\u05b7\u05bd\u05d7\u05b6\u05e6\u05b0\u05e8\u05b9\u05e0\u05b4\u05d9", "\u05d1\u05b0\u05bc\u05d0\u05b5\u05e8\u05b9\u05ea \u05d1\u05b0\u05bc\u05e0\u05b5\u05d9\u05be\u05d9\u05b7\u05e2\u05b2\u05e7\u05b7\u05df", "\u05d5\u05b7\u05bd\u05d9\u05b0\u05e8\u05b7\u05e7\u05bc\u05b0\u05e2\u05d5\u05bc\u05dd", "\u05e4\u05bc\u05b4\u05e8\u05b0\u05e9\u05c1\u05b8\u05d4\u05bc", "\u05e4\u05bc\u05b6\u05d8\u05b6\u05e8", "\u05e4\u05bc\u05b8\u05ea\u05b4\u05d9\u05dc", "\u05d3\u05b8\u05ea\u05b8\u05df", "\u05e4\u05bc\u05b7\u05e8", "\u05de\u05b8\u05db\u05b4\u05bd\u05d9", "\u05d4\u05b7\u05d9\u05b0\u05e9\u05c1\u05b4\u05d9\u05de\u05b9\u05bd\u05df", "\u05d9\u05b0\u05e9\u05c1\u05b8\u05e8\u05b0\u05ea\u05bb\u05d4\u05d5\u05bc", "\u05ea\u05bc\u05b4\u05bd\u05ea\u05b0\u05d7\u05b7\u05d8\u05bc\u05b0\u05d0\u05d5\u05bc", "\u05d7\u05c7\u05e8\u05b0\u05de\u05b8\u05d4", "\u05e2\u05b5\u05d9\u05e0\u05b8\u05d9\u05d5", "\u05d1\u05b8\u05bc\u05e2\u05b7\u05e8", "\u05d4\u05b8\u05dc\u05b7\u05da\u05b0", "\u05d5\u05bc\u05de\u05b8\u05db\u05b4\u05d9\u05e8", "\u05d1\u05b0\u05e2\u05b6\u05e6\u05b6\u05dd", "\u05d1\u05bc\u05b7\u05e2\u05b2\u05d3\u05b7\u05ea", "\u05d5\u05bc\u05d8\u05b0\u05d4\u05b9\u05e8\u05b8\u05d4", "\u05de\u05b7\u05e9\u05b8\u05bc\u05c2\u05d0", "\u05de\u05b4\u05e6\u05b0\u05e8\u05b7\u05d9\u05b4\u05dd", "\u05d6\u05b7\u05e8\u05b0\u05d7\u05b4\u05d9", "\u05d4\u05b7\u05d2\u05bc\u05d5\u05bc\u05e0\u05b4\u05bd\u05d9", "\u05dc\u05b4\u05de\u05b0\u05e2\u05b8\u05bd\u05d8", "\u05d5\u05b0\u05d4\u05b6\u05d7\u05b1\u05e8\u05b4\u05e9\u05c1", "\u05d5\u05b0\u05e2\u05b7\u05bd\u05dc", "\u05d3\u05bc\u05b4\u05d2\u05b0\u05dc\u05d5\u05b9", "\u05d3\u05b0\u05d1\u05b8\u05e8\u05b8\u05d9", "\u05d9\u05b0\u05d3\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05e8\u05b7\u05d1\u05bc\u05b8\u05d4", "\u05e0\u05b0\u05d7\u05b8\u05e9\u05c1\u05b4\u05d9\u05dd", "\u05d2\u05bc\u05b4\u05d3\u05b0\u05e8\u05b9\u05ea", "\u05e2\u05b2\u05d3\u05b8\u05ea\u05d5\u05b9", "\u05d1\u05bc\u05b0\u05e2\u05b7\u05d1\u05b0\u05e8\u05b9\u05e0\u05b8\u05bd\u05d4", "\u05e1\u05b8\u05dc\u05d5\u05bc\u05d0", "\u05d9\u05b8\u05e7\u05b4\u05d9\u05de\u05d5\u05bc", "\u05e2\u05bb\u05d6\u05bc\u05b4\u05d9\u05d0\u05b5\u05bd\u05dc", "\u05d9\u05b4\u05bd\u05d9\u05e8\u05b0\u05e9\u05c1\u05d5\u05bc", "\u05e2\u05b7\u05d9\u05b4\u05df", "\u05e9\u05b8\u05c1\u05e7\u05b8\u05d4", "\u05d5\u05b0\u05db\u05b9\u05dc", "\u05d7\u05b2\u05de\u05d5\u05b9\u05e8", "\u05d5\u05b7\u05d9\u05b0\u05db\u05b7\u05e1", "\u05d2\u05bc\u05b4\u05d3\u05b0\u05e2\u05d5\u05b9\u05e0\u05b4\u05bd\u05d9", "\u05d4\u05b7\u05dc\u05bc\u05b5\u05d5\u05b4\u05d9", "\u05e2\u05b4\u05de\u05bc\u05b8\u05d4\u05b6\u05dd", "\u05d1\u05bc\u05b7\u05de\u05bc\u05b7\u05d2\u05bc\u05b5\u05e4\u05b8\u05d4", "\u05d0\u05b5\u05d9\u05ea\u05b8\u05df", "\u05d5\u05b8\u05d0\u05b7\u05e2\u05b7\u05dc", "\u05d1\u05b5\u05bc\u05df", "\u05e4\u05b7\u05d7\u05b4\u05d9\u05dd", "\u05e9\u05b7\u05c1\u05d3\u05b7\u05bc\u05d9", "\u05e0\u05b8\u05e4\u05b7\u05dc", "\u05e9\u05b7\u05c1\u05d1\u05b8\u05bc\u05ea", "\u05ea\u05bc\u05b4\u05e4\u05b0\u05e7\u05b9\u05d3", "\u05d5\u05bc\u05d1\u05b0\u05e2\u05b7\u05de\u05bc\u05bb\u05d3", "\u05d0\u05b9\u05d9\u05b5\u05d1", "\u05de\u05b7\u05d7\u05b0\u05ea\u05bc\u05d5\u05b9\u05ea", "\u05d1\u05bc\u05b6\u05d7\u05b8\u05bd\u05e8\u05b6\u05d1", "\u05d1\u05bc\u05b4\u05d2\u05b0\u05d3\u05b5\u05d9\u05db\u05b6\u05dd", "\u05d4\u05b8\u05bd\u05d9\u05d5\u05bc", "\u05e2\u05b2\u05d8\u05b8\u05e8\u05d5\u05b9\u05ea", "\u05de\u05b7\u05d9\u05b4\u05dd", "\u05d5\u05bc\u05e4\u05b0\u05e7\u05b7\u05d3\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05e7\u05b8\u05d3\u05b5\u05e9\u05c1 \u05d1\u05b7\u05bc\u05e8\u05b0\u05e0\u05b5\u05e2\u05b7", "\u05d6\u05bc\u05b6\u05d4", "\u05d1\u05bc\u05b0\u05e2\u05d5\u05b9\u05e8", "\u05de\u05b7\u05e1\u05bc\u05b5\u05bd\u05db\u05b9\u05ea\u05b8\u05dd", "\u05dc\u05b0\u05d8\u05b7\u05e4\u05bc\u05b5\u05bd\u05e0\u05d5\u05bc", "\u05e7\u05b6\u05e8\u05b6\u05d1", "\u05d5\u05b0\u05e0\u05b4\u05e7\u05bc\u05b0\u05ea\u05b8\u05d4", "\u05db\u05b8\u05bc\u05dc\u05b8\u05d4", "\u05d5\u05bc\u05d1\u05b0\u05d3\u05b6\u05e8\u05b6\u05da\u05b0", "\u05ea\u05b0\u05de\u05b4\u05d9\u05de\u05b4\u05dd", "\u05e1\u05b4\u05d9\u05e0\u05b8\u05bd\u05d9", "\u05e6\u05b9\u05e2\u05b7\u05df", "\u05d4\u05b8\u05d4\u05b5\u05dd", "\u05d0\u05b7\u05e8\u05b0\u05d3\u05bc\u05b0", "\u05d5\u05bc\u05d1\u05b0\u05e8\u05b8\u05d0\u05e9\u05c1\u05b5\u05d9", "\u05de\u05b4\u05db\u05b0\u05e1\u05b5\u05d4\u05d5\u05bc", "\u05db\u05b9\u05bc\u05d4", "\u05dc\u05b4\u05d1\u05b0\u05e8\u05b4\u05d9\u05e2\u05b8\u05d4", "\u05d1\u05bc\u05b8\u05d0\u05d5\u05bc", "\u05d4\u05b7\u05d7\u05b2\u05d9\u05d5\u05bc", "\u05d9\u05b8\u05d2\u05b0\u05dc\u05b4\u05bd\u05d9", "\u05d8\u05b0\u05de\u05b5\u05d0\u05b4\u05d9\u05dd", "\u05d0\u05b4\u05e9\u05c1\u05b0\u05ea\u05bc\u05d5\u05b9", "\u05d4\u05b7\u05de\u05bc\u05b7\u05d2\u05bc\u05b5\u05e4\u05b8\u05bd\u05d4", "\u05d9\u05b8\u05e0\u05d5\u05bc\u05e1", "\u05d1\u05bc\u05b0\u05d9\u05b8\u05d3", "\u05d5\u05b0\u05d4\u05b4\u05d6\u05bc\u05b8\u05d4", "\u05e0\u05b8\u05bd\u05e9\u05c2\u05b0\u05d0\u05d5\u05bc", "\u05e8\u05b6\u05d7\u05b6\u05dd", "\u05ea\u05b8\u05de\u05bd\u05d5\u05bc\u05ea\u05d5\u05bc", "\u05d0\u05b1\u05dc\u05b9\u05d4\u05b4\u05d9\u05dd", "\u05d9\u05b4\u05e6\u05b0\u05d4\u05b8\u05e8", "\u05d1\u05bc\u05b0\u05e2\u05b7\u05d1\u05b0\u05d3\u05bc\u05b4\u05d9", "\u05dc\u05b8\u05e2\u05b8\u05d9\u05b4\u05df", "\u05de\u05b0\u05e7\u05b7\u05d1\u05bc\u05b0\u05e8\u05b4\u05d9\u05dd", "\u05de\u05b7\u05d8\u05b6\u05bc\u05d4", "\u05d5\u05b0\u05e2\u05b7\u05de\u05bc\u05bb\u05d3\u05b5\u05d9", "\u05d9\u05d5\u05b9\u05e8\u05b4\u05e9\u05c1\u05b6\u05bd\u05e0\u05bc\u05b8\u05d4", "\u05d9\u05bc\u05b9\u05e1\u05b5\u05e3", "\u05e4\u05b7\u05bc\u05e8", "\u05ea\u05bc\u05b0\u05e0\u05d5\u05bc\u05d0\u05b8\u05ea\u05b4\u05bd\u05d9", "\u05d4\u05b4\u05d1\u05bc\u05b4\u05d9\u05d8", "\u05e4\u05b0\u05e7\u05d5\u05bc\u05d3\u05b5\u05d9", "\u05d4\u05b7\u05d9\u05bc\u05b4\u05de\u05b0\u05e0\u05b8\u05d4", "\u05d4\u05b7\u05e0\u05bc\u05b0\u05e4\u05b8\u05e9\u05c1\u05d5\u05b9\u05ea", "\u05e2\u05b5\u05d3\u05d5\u05bc\u05ea", "\u05dc\u05b0\u05e1\u05b6\u05e8\u05b6\u05d3", "\u05dc\u05b7\u05e9\u05c1\u05bc\u05d5\u05b9\u05e8", "\u05d0\u05b1\u05d3\u05b9\u05dd", "\u05dc\u05b5\u05d5\u05b4\u05d9", "\u05dc\u05b0\u05d2\u05bb\u05dc\u05b0\u05d2\u05bc\u05b0\u05dc\u05b9\u05ea\u05b8\u05dd", "\u05d5\u05bc\u05de\u05b8\u05d7\u05b7\u05e5", "\u05e4\u05b8\u05e8\u05b8\u05d4", "\u05d5\u05bc\u05de\u05b4\u05d2\u05b0\u05e8\u05b0\u05e9\u05c1\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05ea\u05bc\u05b7\u05e7\u05b0\u05e8\u05b4\u05d9\u05d1\u05d5\u05bc", "\u05e8\u05d5\u05bc\u05e5", "\u05e8\u05b0\u05d7\u05b9\u05e7\u05b8\u05d4\u05c4", "\u05d5\u05b0\u05d4\u05b7\u05bd\u05e2\u05b2\u05de\u05b7\u05d3\u05b0\u05ea\u05bc\u05b8", "\u05d5\u05b0\u05e0\u05b4\u05db\u05b0\u05d1\u05bc\u05b0\u05e9\u05c1\u05b8\u05d4", "\u05d5\u05b0\u05d8\u05b7\u05e4\u05bc\u05b8\u05bd\u05dd", "\u05d4\u05b5\u05bd\u05e2\u05b8\u05dc\u05d5\u05bc", "\u05d5\u05bc\u05de\u05b5\u05d9\u05d3\u05b8\u05d3", "\u05d1\u05bc\u05b7\u05d7\u05b2\u05dc\u05d5\u05b9\u05dd", "\u05d5\u05b0\u05db\u05b7\u05bd\u05de\u05b0\u05dc\u05b5\u05d0\u05b8\u05d4", "\u05d0\u05b2\u05e0\u05b7\u05d7\u05b0\u05e0\u05d5\u05bc", "\u05d5\u05bc\u05de\u05b8\u05d4", "\u05d5\u05b7\u05d9\u05b0\u05e0\u05b4\u05e2\u05b5\u05dd", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05bd\u05d0\u05b7\u05e1\u05b0\u05e4\u05d5\u05bc", "\u05d9\u05b7\u05d7\u05b2\u05e0\u05d5\u05bc", "\u05d0\u05b2\u05d7\u05b4\u05d9\u05de\u05b7\u05df", "\u05d9\u05b8\u05e8\u05b5\u05d0", "\u05d4\u05b7\u05d3\u05bc\u05b8\u05bd\u05e8\u05b6\u05da\u05b0", "\u05dc\u05b7\u05d0\u05b2\u05d7\u05b5\u05d9", "\u05dc\u05b7\u05d8\u05bc\u05d5\u05b9\u05d1", "\u05e1\u05b4\u05d9\u05e0\u05b8\u05d9", "\u05e0\u05b0\u05e9\u05c2\u05b4\u05bd\u05d9\u05d0\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05d9\u05b0\u05e8\u05b4\u05d9\u05d7\u05d5\u05b9", "\u05d5\u05b7\u05ea\u05bc\u05b4\u05d1\u05b0\u05dc\u05b7\u05e2", "\u05e8\u05b8\u05d2\u05b7\u05dc", "\u05d5\u05bc\u05de\u05b5\u05d0\u05b8\u05bd\u05d4", "\u05e4\u05bc\u05b0\u05d3\u05b8\u05d4\u05e6\u05bd\u05d5\u05bc\u05e8", "\u05e2\u05b2\u05e8\u05b9\u05e2\u05b5\u05bd\u05e8", "\u05e2\u05b7\u05de\u05bc\u05b4\u05bd\u05d9\u05e9\u05c1\u05b7\u05d3\u05bc\u05b8\u05bd\u05d9", "\u05d5\u05bc\u05d1\u05b0\u05d4\u05b7\u05e7\u05b0\u05d4\u05b4\u05d9\u05dc", "\u05d5\u05b7\u05d9\u05bc\u05b5\u05e6\u05b5\u05d0", "\u05db\u05bc\u05b0\u05e1\u05d5\u05bc\u05d9", "\u05d5\u05b7\u05d7\u05b2\u05de\u05b5\u05bd\u05e9\u05c1\u05b6\u05ea", "\u05d9\u05b0\u05dc\u05b4\u05d3\u05b0\u05ea\u05bc\u05b4\u05d9\u05d4\u05d5\u05bc", "\u05d5\u05b0\u05d3\u05b4\u05d9\u05d1\u05b9\u05df", "\u05ea\u05b4\u05e7\u05bc\u05b3\u05d1\u05b6\u05e0\u05bc\u05d5\u05bc", "\u05ea\u05bc\u05b8\u05e8\u05b4\u05d9\u05de\u05d5\u05bc", "\u05d1\u05bc\u05b8\u05e8\u05b4\u05d0\u05e9\u05c1\u05d5\u05b9\u05df", "\u05e9\u05b4\u05c1\u05d9\u05e8", "\u05de\u05d5\u05b9\u05e6\u05b8\u05d0", "\u05de\u05b7\u05e9\u05c2\u05b0\u05db\u05bc\u05b4\u05d9\u05bc\u05b9\u05ea\u05b8\u05dd", "\u05de\u05b7\u05d3\u05bc\u05d5\u05bc\u05e2\u05b7", "\u05d5\u05b0\u05d0\u05b7\u05e9\u05c2\u05b0\u05e8\u05b4\u05d9\u05d0\u05b5\u05dc", "\u05d2\u05bc\u05b7\u05d3\u05bc\u05b4\u05d9", "\u05de\u05b5\u05d1\u05b4\u05d9\u05d0", "\u05d9\u05b0\u05dc\u05b4\u05d3\u05b5\u05d9", "\u05d9\u05b4\u05e9\u05c2\u05b0\u05e8\u05b8\u05d0\u05b5\u05dc", "\u05e0\u05b6\u05d3\u05b6\u05e8", "\u05d5\u05b0\u05dc\u05b7\u05d0\u05b2\u05d1\u05b4\u05d9\u05e8\u05b8\u05dd", "\u05dc\u05b4\u05e0\u05b0\u05d3\u05bc\u05b9\u05e8", "\u05d4\u05b8\u05d0\u05b4\u05bd\u05d9\u05e2\u05b6\u05d6\u05b0\u05e8\u05b4\u05d9", "\u05e0\u05b7\u05e4\u05b0\u05e9\u05c1\u05b5\u05e0\u05d5\u05bc", "\u05d4\u05b8\u05d0\u05b9\u05de\u05b5\u05df", "\u05d4\u05b7\u05e0\u05bc\u05b0\u05d7\u05b8\u05dc\u05b4\u05d9\u05dd", "\u05e2\u05b7\u05de\u05bc\u05b4\u05d9\u05d0\u05b5\u05dc", "\u05dc\u05b0\u05d0\u05b5\u05dc\u05d5\u05b9\u05df", "\u05d2\u05bc\u05b4\u05d3\u05b0\u05e2\u05b9\u05e0\u05b4\u05bd\u05d9", "\u05d5\u05bc\u05de\u05b4\u05e9\u05c1\u05b0\u05e4\u05bc\u05b7\u05d7\u05b7\u05ea", "\u05d5\u05bc\u05dc\u05b0\u05d1\u05b8\u05e0\u05b6\u05bd\u05d9\u05da\u05b8", "\u05ea\u05bc\u05b4\u05ea\u05bc\u05b5\u05e0\u05d5\u05bc", "\u05db\u05b7\u05bc\u05e4\u05b9\u05bc\u05e8\u05b6\u05ea", "\u05ea\u05b9\u05bc\u05dd", "\u05d8\u05b8\u05de\u05b5\u05d0", "\u05e0\u05b8\u05e9\u05c2\u05b9\u05d0", "\u05de\u05b5\u05bd\u05d0\u05b2\u05d2\u05b7\u05d2", "\u05db\u05b8\u05bc\u05d4\u05b7\u05df", "\u05d5\u05bc\u05de\u05b8\u05d7\u05b8\u05d4", "\u05d5\u05b7\u05bd\u05d9\u05b0\u05e0\u05b7\u05e9\u05c1\u05bc\u05b0\u05db\u05d5\u05bc", "\u05e0\u05b0\u05e9\u05c2\u05b4\u05bd\u05d9\u05d0\u05b5\u05d4\u05b6\u05dd", "\u05e0\u05b0\u05ea\u05b7\u05e0\u05b0\u05d0\u05b5\u05dc", "\u05d5\u05b0\u05d0\u05b7\u05ea\u05bc\u05b8\u05d4", "\u05dc\u05b0\u05e4\u05b7\u05dc\u05bc\u05d5\u05bc\u05d0", "\u05e0\u05b8\u05d2\u05b7\u05d3", "\u05e7\u05b9\u05d1", "\u05d4\u05b7\u05bd\u05d7\u05b6\u05d1\u05b0\u05e8\u05b4\u05d9", "\u05e9\u05c1\u05b6\u05de\u05b6\u05df", "\u05d4\u05b7\u05de\u05bc\u05b6\u05db\u05b6\u05e1", "\u05d0\u05b2\u05d1\u05b4\u05d9\u05d7\u05b8\u05d9\u05b4\u05dc", "\u05d1\u05bc\u05b0\u05de\u05b4\u05e9\u05c1\u05b0\u05de\u05b6\u05e8\u05b6\u05ea", "\u05e0\u05b0\u05d3\u05b4\u05d9\u05d1\u05b5\u05d9", "\u05d1\u05bc\u05b0\u05e0\u05b9\u05ea\u05b6\u05bd\u05d9\u05d4\u05b8", "\u05d3\u05bc\u05b9\u05d1\u05b0\u05e8\u05b4\u05bd\u05d9\u05dd", "\u05dc\u05b8\u05d0\u05b5\u05d9\u05dc\u05b4\u05dd", "\u05d0\u05b4\u05d9\u05e9\u05c1\u05b5\u05bd\u05da\u05b0", "\u05d0\u05b9\u05d3\u05d5\u05b9\u05ea", "\u05d5\u05b0\u05e0\u05b4\u05e1\u05b0\u05db\u05bc\u05d5\u05b9", "\u05d4\u05b8\u05e2\u05b9\u05d1\u05b5\u05d3", "\u05de\u05b4\u05ea\u05bc\u05d5\u05bc\u05e8", "\u05d5\u05b0\u05e0\u05b6\u05d0\u05b1\u05e1\u05b7\u05e4\u05b0\u05ea\u05bc\u05b8", "\u05d1\u05bc\u05b0\u05e7\u05b7\u05e0\u05b0\u05d0\u05d5\u05b9", "\u05d5\u05b0\u05d2\u05b5\u05e8\u05b7\u05e9\u05c1\u05b0\u05ea\u05bc\u05b4\u05bd\u05d9\u05d5", "\u05d5\u05bc\u05e9\u05c2\u05b0\u05e2\u05b4\u05d9\u05e8\u05b5\u05d9", "\u05e2\u05b7\u05de\u05bc\u05b6\u05bd\u05d9\u05da\u05b8", "\u05d5\u05b7\u05e2\u05b2\u05e9\u05c2\u05b4\u05d9\u05e8\u05b4\u05d9\u05ea", "\u05ea\u05b8\u05d1\u05b4\u05d9\u05d0\u05d5\u05bc", "\u05de\u05b4\u05de\u05bc\u05b7\u05d7\u05b2\u05e6\u05b4\u05ea", "\u05dc\u05b0\u05d6\u05b8\u05e8\u05b8\u05d0", "\u05d9\u05b7\u05e2\u05b7\u05df", "\u05d5\u05b0\u05e7\u05b8\u05d1\u05b0\u05e0\u05d5\u05b9", "\u05db\u05bc\u05b4\u05dc\u05b0\u05d7\u05b9\u05da\u05b0", "\u05de\u05b0\u05e0\u05b9\u05e8\u05b7\u05ea", "\u05d4\u05b7\u05e7\u05bc\u05b9\u05d3\u05b6\u05e9\u05c1", "\u05e8\u05b8\u05d2\u05d5\u05b9\u05dd", "\u05d5\u05b7\u05e2\u05b2\u05e0\u05b7\u05df", "\u05dc\u05b6\u05d7\u05b6\u05dd", "\u05d9\u05b4\u05d2\u05bc\u05b0\u05e2\u05d5\u05bc", "\u05dc\u05b8\u05bd\u05da\u05b0", "\u05d5\u05b0\u05dc\u05b8\u05de\u05b8\u05d4", "\u05d5\u05b0\u05d7\u05b7\u05d8\u05bc\u05b8\u05d0\u05ea\u05b8\u05dd", "\u05d4\u05b8\u05bd\u05e2\u05b5\u05d3\u05d5\u05bc\u05ea", "\u05d0\u05b5\u05e4\u05b6\u05e8", "\u05e9\u05c1\u05b0\u05dc\u05b9\u05d7\u05b7", "\u05d9\u05b6\u05d7\u05b1\u05d8\u05b8\u05d0", "\u05e2\u05b8\u05de\u05b7\u05d3", "\u05e1\u05b4\u05d9\u05d7\u05bd\u05d5\u05b9\u05df", "\u05d1\u05bc\u05b0\u05de\u05b4\u05e9\u05c1\u05b0\u05e2\u05d5\u05b9\u05dc", "\u05d0\u05b5\u05dc\u05b5\u05d9\u05e0\u05d5\u05bc", "\u05d5\u05b7\u05d9\u05bc\u05b8\u05e0\u05b6\u05e3", "\u05d1\u05bc\u05b0\u05d4\u05b6\u05de\u05b0\u05ea\u05bc\u05b5\u05e0\u05d5\u05bc", "\u05d9\u05b8\u05e9\u05b7\u05c1\u05d1", "\u05d4\u05b8\u05e8\u05b7\u05d2", "\u05d9\u05b7\u05e2\u05b2\u05d6\u05b5\u05d9\u05e8", "\u05d4\u05b2\u05e8\u05b9\u05bd\u05d2\u05d5\u05bc", "\u05d4\u05b4\u05e7\u05b0\u05d4\u05b4\u05d9\u05dc\u05d5\u05bc", "\u05d0\u05b6\u05e9\u05c1\u05b0\u05de\u05b9\u05e8", "\u05d5\u05b0\u05d4\u05b5\u05e7\u05b4\u05d9\u05de\u05d5\u05bc", "\u05dc\u05b0\u05e4\u05bb\u05d5\u05b8\u05d4", "\u05d5\u05b0\u05e0\u05b4\u05d8\u05b0\u05de\u05b8\u05bd\u05d0\u05b8\u05d4", "\u05d5\u05bc\u05d1\u05b0\u05de\u05b9\u05e9\u05c1\u05b6\u05d4", "\u05de\u05b4\u05e7\u05bc\u05b4\u05e8\u05b0\u05d9\u05b7\u05ea", "\u05ea\u05bc\u05d5\u05b9\u05ea\u05b4\u05d9\u05e8\u05d5\u05bc", "\u05e1\u05b4\u05d9\u05df", "\u05d4\u05b8\u05e8\u05b0\u05d2\u05b5\u05e0\u05b4\u05d9", "\u05dc\u05b0\u05e4\u05b8\u05e0\u05b7\u05d9", "\u05d1\u05bc\u05b0\u05de\u05b7\u05d8\u05bc\u05b5\u05d4\u05d5\u05bc", "\u05d1\u05b8\u05bc\u05e7\u05b7\u05e9\u05c1", "\u05d5\u05bc\u05d1\u05b7\u05d7\u05b2\u05e0\u05b9\u05ea", "\u05de\u05b4\u05d1\u05bc\u05b0\u05db\u05d5\u05b9\u05e8", "\u05de\u05b4\u05d1\u05b0\u05d8\u05b8\u05d0", "\u05d5\u05b0\u05e2\u05b5\u05d9\u05e0\u05d5\u05b9", "\u05e0\u05b0\u05d1\u05d5\u05b9", "\u05d5\u05b0\u05e1\u05b8\u05de\u05b0\u05db\u05d5\u05bc", "\u05d0\u05b6\u05dc\u05b0\u05d9\u05b8\u05e1\u05b8\u05e3", "\u05de\u05b4\u05dc\u05b0\u05d7\u05b8\u05de\u05b8\u05d4", "\u05d5\u05bc\u05e4\u05b8\u05e6\u05b0\u05ea\u05b8\u05d4", "\u05dc\u05b7\u05e2\u05b2\u05d3\u05b7\u05ea", "\u05d1\u05bc\u05b7\u05e0\u05bc\u05b8\u05e9\u05c1\u05b4\u05d9\u05dd", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e9\u05c1\u05b0\u05d8\u05b0\u05d7\u05d5\u05bc", "\u05dc\u05b7\u05d7\u05b2\u05e0\u05bb\u05db\u05bc\u05b7\u05ea", "\u05d5\u05bc\u05d1\u05b4\u05de\u05b0\u05e7\u05d5\u05b9\u05dd", "\u05d9\u05b8\u05db\u05b9\u05dc", "\u05d5\u05bc\u05de\u05b4\u05e0\u05bc\u05b7\u05d7\u05b2\u05dc\u05b4\u05d9\u05d0\u05b5\u05dc", "\u05d5\u05d4\u05d9\u05d4", "\u05d4\u05b7\u05d9\u05bc\u05b4\u05e9\u05c1\u05b0\u05d5\u05b4\u05d9", "\u05de\u05b7\u05e2\u05b0\u05e9\u05c2\u05b7\u05e8", "\u05e9\u05b6\u05c1\u05de\u05b6\u05df", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05d1\u05b0\u05e0\u05d5\u05bc", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e1\u05bc\u05b7\u05e2", "\u05d1\u05b0\u05e8\u05b4\u05d9\u05e2\u05b8\u05d4", "\u05e9\u05c1\u05b0\u05dc\u05b9\u05e9\u05c1\u05b8\u05d4", "\u05dc\u05b0\u05e0\u05b5\u05bd\u05e1", "\u05d0\u05b7\u05d7\u05b2\u05e8\u05b8\u05bd\u05d9", "\u05d6\u05b0\u05e0\u05d5\u05bc\u05ea\u05b5\u05d9\u05db\u05b6\u05dd", "\u05dc\u05b0\u05d0\u05b8\u05e9\u05c1\u05b8\u05dd", "\u05e7\u05b9\u05e8\u05b6\u05d7", "\u05d5\u05bc\u05d1\u05b0\u05d4\u05b5\u05e2\u05b8\u05dc\u05b9\u05ea\u05d5\u05b9", "\u05de\u05b4\u05e9\u05c1\u05b0\u05e4\u05bc\u05b8\u05d8", "\u05d1\u05bc\u05b0\u05de\u05b4\u05d3\u05b0\u05d9\u05b8\u05df", "\u05ea\u05bc\u05b4\u05e4\u05b0\u05e7\u05b0\u05d3\u05b5\u05bd\u05dd", "\u05d5\u05bc\u05ea\u05b0\u05e7\u05b7\u05e2\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05e9\u05b0\u05c1\u05db\u05b8\u05d1\u05b8\u05d4", "\u05e1\u05bb\u05db\u05bc\u05d5\u05b9\u05ea", "\u05d4\u05b8\u05e8\u05b7\u05d1", "\u05d8\u05b6\u05e8\u05b6\u05dd", "\u05d5\u05bc\u05dc\u05b0\u05db\u05b7\u05e4\u05bc\u05b5\u05e8", "\u05d4\u05b7\u05d9\u05bc\u05b8\u05de\u05b4\u05bd\u05d9\u05dd", "\u05ea\u05bc\u05b7\u05e8\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05d5\u05b0\u05e9\u05c1\u05b6\u05db\u05b6\u05dd", "\u05d0\u05b8\u05ea\u05d5\u05b9\u05df", "\u05e9\u05b8\u05c2\u05e8\u05b8\u05e3", "\u05d5\u05b0\u05dc\u05b7\u05e0\u05b0\u05e4\u05bc\u05b4\u05dc", "\u05e2\u05b8\u05e8\u05b5\u05d9\u05d4\u05b6\u05bd\u05dd", "\u05d5\u05b0\u05d0\u05b8\u05de\u05b0\u05e8\u05d5\u05bc", "\u05dc\u05b0\u05d4\u05b7\u05d8\u05bc\u05b9\u05ea\u05b8\u05d4\u05bc", "\u05d4\u05b7\u05d9\u05bc\u05b7\u05d7\u05b0\u05dc\u05b0\u05d0\u05b5\u05dc\u05b4\u05bd\u05d9", "\u05de\u05b4\u05e6\u05b0\u05e8\u05b8\u05d9\u05b4\u05dd", "\u05db\u05bc\u05b0\u05e2\u05b5\u05d9\u05df", "\u05de\u05b7\u05dc\u05b0\u05e7\u05b8\u05d7\u05b6\u05d9\u05d4\u05b8", "\u05d0\u05b4\u05e9\u05c1\u05bc\u05b6\u05d4", "\u05d4\u05b7\u05e0\u05bc\u05b8\u05bd\u05d2\u05b6\u05e3", "\u05ea\u05bc\u05b8\u05d7\u05b7\u05e9\u05c1", "\u05d1\u05b6\u05dc\u05b7\u05e2", "\u05e2\u05bb\u05d2\u05d5\u05b9\u05ea", "\u05de\u05b0\u05d0\u05d5\u05bc\u05de\u05b8\u05d4", "\u05dc\u05b0\u05de\u05b4\u05e9\u05c1\u05b0\u05e4\u05bc\u05b0\u05d7\u05d5\u05b9\u05ea\u05b8\u05dd", "\u05ea\u05bc\u05b0\u05e9\u05c1\u05b7\u05dc\u05bc\u05b5\u05d7\u05d5\u05bc", "\u05d0\u05b8\u05d1\u05b7\u05d3\u05b0\u05ea\u05bc\u05b8", "\u05e4\u05bc\u05b7\u05d0\u05b2\u05ea\u05b5\u05d9", "\u05e2\u05b5\u05ea", "\u05ea\u05bc\u05b5\u05d9\u05de\u05b8\u05e0\u05b8\u05d4", "\u05d5\u05b0\u05e2\u05b4\u05e9\u05c2\u05bc\u05b8\u05e8\u05d5\u05b9\u05df", "\u05d4\u05b8\u05d0\u05b2\u05d7\u05b4\u05d9\u05e8\u05b8\u05de\u05b4\u05bd\u05d9", "\u05d9\u05d5\u05b9\u05e9\u05c1\u05b5\u05d1", "\u05d5\u05bc\u05e9\u05c1\u05b0\u05de\u05b9\u05e0\u05b6\u05d4", "\u05d5\u05bc\u05de\u05b4\u05d2\u05bc\u05b0\u05d1\u05b8\u05e2\u05d5\u05b9\u05ea", "\u05d9\u05b0\u05e4\u05bb\u05e0\u05b6\u05bc\u05d4", "\u05d5\u05b0\u05e7\u05b4\u05e0\u05bc\u05b5\u05d0", "\u05d7\u05b8\u05dc\u05b7\u05e5", "\u05d0\u05b7\u05e0\u05b0\u05ea\u05b8\u05bc\u05d4", "\u05d9\u05b4\u05de\u05b0\u05d7\u05b8\u05bd\u05e5", "\u05d1\u05b0\u05de\u05b9\u05e9\u05c1\u05b6\u05bd\u05d4", "\u05d1\u05bc\u05b7\u05de\u05bc\u05b7\u05d9\u05b4\u05dd", "\u05e0\u05b0\u05d3\u05b8\u05e8\u05b6\u05d9\u05d4\u05b8", "\u05d8\u05b7\u05e4\u05bc\u05b5\u05e0\u05d5\u05bc", "\u05e7\u05b8\u05d3\u05b5\u05e9\u05c1\u05b8\u05d4", "\u05d5\u05b0\u05e0\u05b4\u05de\u05b0\u05e8\u05b8\u05d4", "\u05dc\u05b0\u05d0\u05b6\u05d7\u05b8\u05d9\u05d5", "\u05dc\u05b5\u05bd\u05d0\u05dc\u05b9\u05d4\u05b5\u05d9\u05d4\u05b6\u05bd\u05df", "\u05e2\u05b2\u05d1\u05b9\u05bd\u05d3\u05b7\u05ea\u05b0\u05db\u05b6\u05dd", "\u05e7\u05b5\u05bd\u05d3\u05b0\u05de\u05b8\u05d4", "\u05e9\u05b4\u05c1\u05e9\u05b4\u05bc\u05c1\u05d9\u05dd", "\u05e7\u05b4\u05e8\u05b0\u05d9\u05b8\u05ea\u05b7\u05d9\u05b4\u05dd", "\u05d1\u05bc\u05b0\u05d1\u05b9\u05bd\u05d0\u05b2\u05db\u05b6\u05dd", "\u05d5\u05b0\u05db\u05b5\u05df", "\u05d9\u05b8\u05de\u05b4\u05d9\u05dd", "\u05d4\u05b7\u05e0\u05bc\u05b4\u05e9\u05c1\u05b0\u05e7\u05b8\u05e3", "\u05e9\u05b0\u05c2\u05dc\u05b8\u05d5", "\u05e7\u05b4\u05e0\u05bc\u05b5\u05d0", "\u05d6\u05b6\u05e8\u05b6\u05d3", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e7\u05bc\u05b9\u05d3", "\u05de\u05b2\u05dc\u05b0\u05d0\u05b8\u05da\u05b0", "\u05d9\u05d5\u05b9\u05e8\u05b4\u05d9\u05d3\u05d5\u05bc", "\u05d9\u05b8\u05de\u05b4\u05d9\u05df", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05d5\u05bc\u05de\u05b4\u05bd\u05d9\u05dd", "\u05e8\u05b0\u05d2\u05b8\u05dc\u05b4\u05d9\u05dd", "\u05d4\u05b5\u05d0\u05b8\u05e1\u05b5\u05e3", "\u05d0\u05b8\u05d1", "\u05e9\u05b5\u05c1\u05ea", "\u05e9\u05b8\u05c1\u05d1\u05b7\u05e2", "\u05d0\u05b8\u05e9\u05c1\u05b7\u05dd", "\u05d4\u05b8\u05bd\u05e2\u05b2\u05e8\u05b0\u05d1\u05bc\u05b7\u05d9\u05b4\u05dd", "\u05d0\u05b5\u05dc\u05b6\u05d9\u05da\u05b8", "\u05d5\u05bc\u05e4\u05b8\u05e8\u05b0\u05e9\u05c2\u05d5\u05bc", "\u05d5\u05b0\u05dc\u05b7\u05e2\u05b2\u05de\u05b9\u05d3", "\u05e9\u05c1\u05b0\u05de\u05b9\u05ea\u05b8\u05bd\u05dd", "\u05e0\u05b4\u05de\u05b0\u05e8\u05b8\u05d4", "\u05d5\u05b7\u05d9\u05bc\u05b9\u05e6\u05b4\u05d0\u05b5\u05e0\u05d5\u05bc", "\u05de\u05b8\u05e6\u05b8\u05d0", "\u05de\u05b4\u05e9\u05c1\u05b0\u05e7\u05b8\u05dc\u05b8\u05d4\u05bc", "\u05d1\u05bc\u05b4\u05e9\u05c1\u05b0\u05de\u05b9\u05ea\u05b8\u05dd", "\u05d0\u05b2\u05ea\u05b9\u05e0\u05d5\u05b9", "\u05de\u05b7\u05e2\u05b2\u05e9\u05c2\u05b6\u05bd\u05d4", "\u05d1\u05bc\u05b0\u05e6\u05b4\u05d3\u05bc\u05b5\u05d9\u05db\u05b6\u05dd", "\u05db\u05bc\u05b0\u05d3\u05b6\u05e8\u05b6\u05da\u05b0", "\u05d7\u05b7\u05d5\u05bc\u05ba\u05ea\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05e9\u05c1\u05b7\u05d1\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05e7\u05b0\u05e0\u05b8\u05d0\u05b9\u05ea", "\u05d5\u05b0\u05d0\u05b6\u05e9\u05c1\u05b6\u05d3", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e4\u05b0\u05ea\u05bc\u05b7\u05d7", "\u05e0\u05b8\u05ea\u05b7\u05df", "\u05d7\u05b8\u05d2", "\u05d4\u05b7\u05bd\u05d7\u05b6\u05e4\u05b0\u05e8\u05b4\u05bd\u05d9", "\u05d0\u05b7\u05e8\u05b0\u05d1\u05bc\u05b8\u05e2\u05b8\u05d4", "\u05db\u05b6\u05e1\u05b6\u05e3", "\u05ea\u05bc\u05b4\u05ea\u05b0\u05e7\u05b0\u05e2\u05d5\u05bc", "\u05de\u05b7\u05e1\u05b0\u05e2\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05de\u05b4\u05bd\u05de\u05bc\u05b4\u05e9\u05c1\u05b0\u05e4\u05bc\u05b0\u05d7\u05b9\u05ea", "\u05d5\u05b0\u05d0\u05b8\u05de\u05b0\u05e8\u05b8\u05d4", "\u05e0\u05b7\u05d7\u05b0\u05e0\u05d5\u05bc", "\u05d5\u05b7\u05e2\u05b2\u05e0\u05b8\u05d1\u05b4\u05d9\u05dd", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05dc\u05bc\u05b4\u05d9\u05e0\u05d5\u05bc", "\u05e0\u05d5\u05bc\u05e1", "\u05d6\u05b8\u05d1\u05b7\u05d7", "\u05e9\u05c1\u05b8\u05e4\u05b8\u05d8", "\u05dc\u05b9\u05d0", "\u05d4\u05b4\u05ea\u05b0\u05d9\u05b7\u05e6\u05bc\u05b5\u05d1", "\u05e4\u05bc\u05b7\u05dc\u05b0\u05d8\u05b4\u05d9", "\u05e9\u05c2\u05d5\u05bc\u05dd", "\u05d2\u05b9\u05bc\u05e8\u05b6\u05df", "\u05e4\u05b6\u05bc\u05e8\u05b6\u05e5", "\u05d0\u05b1\u05dc\u05b9\u05d4\u05b5\u05d9\u05d4\u05b6\u05df", "\u05dc\u05b0\u05e9\u05c1\u05b4\u05d1\u05b0\u05e2\u05b7\u05ea", "\u05de\u05b8\u05e7\u05d5\u05b9\u05dd", "\u05d5\u05b0\u05d7\u05b9\u05de\u05b6\u05e5", "\u05d1\u05bc\u05b0\u05ea\u05b8\u05bd\u05e8\u05b7\u05d7", "\u05e7\u05b0\u05e9\u05c2\u05d5\u05b9\u05ea", "\u05d1\u05bc\u05b0\u05e2\u05b5\u05d9\u05e0\u05b6\u05d9\u05da\u05b8", "\u05d0\u05b6\u05e7\u05bc\u05b9\u05d1", "\u05d0\u05b6\u05d1\u05b6\u05df", "\u05d5\u05b0\u05d4\u05b8\u05d0\u05b9\u05d4\u05b6\u05dc", "\u05d5\u05bc\u05d1\u05b0\u05e2\u05b5\u05d9\u05e0\u05b5\u05d9", "\u05d4\u05b7\u05d9\u05bc\u05b7\u05d7\u05b0\u05e6\u05b0\u05d0\u05b5\u05dc\u05b4\u05d9", "\u05d9\u05b4\u05e8\u05b0\u05d0\u05d5\u05bc", "\u05dc\u05b8\u05d0\u05b4\u05d9\u05e9\u05c1", "\u05e1\u05b5\u05e4\u05b6\u05e8", "\u05d4\u05b8\u05d9\u05b8\u05d4", "\u05d1\u05bc\u05b4\u05d2\u05b0\u05d3\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05d1\u05bc\u05b4\u05e9\u05c1\u05b0\u05d1\u05bb\u05e2\u05b7\u05ea", "\u05ea\u05e0\u05d5\u05d0\u05d5\u05df", "\u05e9\u05c1\u05b0\u05e0\u05b5\u05bd\u05d9", "\u05d4\u05b2\u05e4\u05b5\u05e8\u05b8\u05dd", "\u05d0\u05b6\u05e8\u05b0\u05d0\u05b6\u05e0\u05bc\u05d5\u05bc", "\u05e2\u05b2\u05d1\u05b9\u05d3\u05b8\u05ea\u05bd\u05d5\u05b9", "\u05d4\u05b8\u05d0\u05b9\u05bd\u05d4\u05b6\u05dc", "\u05d4\u05b7\u05d4\u05b4\u05d5\u05d0", "\u05d4\u05b7\u05e7\u05bc\u05b0\u05d4\u05b8\u05ea\u05b4\u05d9\u05dd", "\u05d4\u05b7\u05d2\u05bc\u05b5\u05e8\u05b0\u05e9\u05c1\u05bb\u05e0\u05bc\u05b4\u05d9", "\u05d5\u05bc\u05e4\u05b0\u05e7\u05bb\u05d3\u05b8\u05d9\u05d5", "\u05d4\u05b8\u05e8\u05b4\u05d9\u05ea\u05b4\u05d9", "\u05d5\u05b0\u05d4\u05b4\u05e7\u05b0\u05e8\u05b4\u05d9\u05d1\u05b8\u05d4", "\u05e2\u05d5\u05b9\u05dc\u05b7\u05ea", "\u05d9\u05b0\u05d4\u05d5\u05bc\u05d3\u05b8\u05d4", "\u05d5\u05b0\u05d4\u05b8\u05d0\u05b2\u05e0\u05b8\u05e9\u05c1\u05b4\u05d9\u05dd", "\u05e9\u05b8\u05c1\u05dc\u05d5\u05b9\u05e9\u05c1", "\u05d5\u05b0\u05e1\u05b7\u05dc", "\u05d0\u05b8\u05e9\u05c1\u05d5\u05bc\u05d1\u05b8\u05d4", "\u05e0\u05b7\u05d7\u05b2\u05dc\u05b8\u05ea\u05b8\u05bd\u05df", "\u05dc\u05b6\u05d0\u05b0\u05e1\u05b9\u05e8", "\u05d1\u05bc\u05b0\u05d0\u05b6\u05d1\u05b6\u05df", "\u05d9\u05b4\u05e9\u05c1\u05b0\u05d1\u05bc\u05b0\u05e8\u05d5\u05bc", "\u05e9\u05b8\u05c2\u05e0\u05b5\u05d0", "\u05d5\u05b0\u05e9\u05c1\u05b8\u05dd", "\u05d4\u05b7\u05db\u05bc\u05b9\u05ea\u05b4\u05d9", "\u05d1\u05bc\u05b8\u05de\u05bd\u05d5\u05b9\u05ea", "\u05e2\u05b4\u05de\u05b0\u05d3\u05d5\u05bc", "\u05d4\u05b6\u05bd\u05e2\u05b8\u05e4\u05b8\u05e8", "\u05e0\u05d5\u05b9\u05e6\u05b4\u05d9\u05d0", "\u05d5\u05b0\u05d4\u05b4\u05e7\u05b0\u05d4\u05b7\u05dc\u05b0\u05ea\u05bc\u05b8", "\u05e9\u05b8\u05c1\u05dc\u05b7\u05e3", "\u05db\u05bc\u05b5\u05bd\u05df", "\u05d4\u05d5\u05b9\u05e9\u05c1\u05b5\u05e2\u05b7", "\u05e2\u05b8\u05de\u05b0\u05d3\u05d5\u05b9", "\u05e4\u05bc\u05b0\u05ea\u05d5\u05b9\u05e8\u05b8\u05d4", "\u05d7\u05b7\u05d9\u05bc\u05b8\u05ea\u05b8\u05bd\u05dd", "\u05e2\u05b5\u05e0\u05b8\u05d1", "\u05d4\u05b4\u05d5\u05d0", "\u05d5\u05b7\u05bd\u05d9\u05b0\u05db\u05b7\u05d1\u05bc\u05b0\u05e1\u05d5\u05bc", "\u05e7\u05b8\u05bd\u05d1\u05b8\u05d4", "\u05e6\u05b8\u05e4\u05d5\u05b9\u05df", "\u05d5\u05b0\u05dc\u05b4\u05e6\u05b0\u05e0\u05b4\u05d9\u05e0\u05b4\u05dd", "\u05db\u05bc\u05b4\u05e0\u05b0\u05d7\u05b8\u05dc\u05b4\u05d9\u05dd", "\u05d2\u05b8\u05bc\u05d3", "\u05d5\u05bc\u05dc\u05b0\u05d6\u05b7\u05e8\u05b0\u05e2\u05b2\u05da\u05b8", "\u05e9\u05b5\u05c1\u05e0\u05b4\u05d9", "\u05d5\u05b4\u05bd\u05d9\u05db\u05b7\u05d6\u05bc\u05b5\u05d1", "\u05de\u05b5\u05d7\u05b2\u05de\u05b5\u05e9\u05c1", "\u05d4\u05b7\u05e4\u05bc\u05b8\u05e0\u05b4\u05d9\u05dd", "\u05de\u05b4\u05e6\u05b0\u05e8\u05b8\u05bd\u05d9\u05b0\u05de\u05b8\u05d4", "\u05e0\u05b6\u05e4\u05b6\u05e9\u05c1", "\u05e9\u05c1\u05b0\u05de\u05d5\u05b9\u05ea", "\u05ea\u05bc\u05b0\u05e0\u05b7\u05e7\u05bc\u05b5\u05e8", "\u05d5\u05bc\u05d1\u05b0\u05d0\u05b5\u05dc\u05bc\u05b6\u05d4", "\u05d9\u05b7\u05d7\u05b5\u05dc", "\u05d5\u05b8\u05d0\u05b6\u05dc\u05b6\u05e3", "\u05d5\u05b0\u05dc\u05b4\u05e7\u05b0\u05d4\u05b8\u05ea", "\u05e0\u05b7\u05db\u05bc\u05b6\u05d4", "\u05d5\u05b0\u05d4\u05bd\u05b7\u05e2\u05b2\u05de\u05b7\u05d3\u05b0\u05ea\u05bc\u05b8", "\u05d4\u05b7\u05bd\u05ea\u05bc\u05b7\u05d0\u05b2\u05d5\u05b8\u05bd\u05d4", "\u05d9\u05b8\u05d3\u05b7\u05e2", "\u05d3\u05b4\u05d1\u05bc\u05b7\u05ea", "\u05d5\u05b0\u05d7\u05b8\u05d9\u05d5\u05bc", "\u05ea\u05bc\u05b7\u05de\u05b0\u05e0\u05d5\u05bc", "\u05d1\u05b0\u05ea\u05b9\u05db\u05b0\u05db\u05b6\u05dd", "\u05d7\u05b8\u05dc\u05b8\u05d1", "\u05d9\u05b4\u05ea\u05b0\u05e0\u05b7\u05e9\u05c2\u05bc\u05b8\u05d0", "\u05d4\u05b2\u05d1\u05b4\u05bd\u05d9\u05d0\u05b9\u05e0\u05bb\u05dd", "\u05e9\u05b0\u05c1\u05de\u05b9\u05e0\u05b6\u05d4", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05d3\u05b0\u05e7\u05b9\u05e8", "\u05d5\u05b0\u05e0\u05bd\u05d5\u05b9\u05e1\u05b0\u05e4\u05b8\u05d4", "\u05de\u05b4\u05bd\u05d9\u05bc\u05b7\u05e2\u05b2\u05e7\u05b9\u05d1", "\u05d0\u05b9\u05ea\u05b8\u05bd\u05dd", "\u05db\u05b0\u05bc\u05e0\u05b7\u05e2\u05b7\u05e0\u05b4\u05d9", "\u05e4\u05bc\u05b8\u05e8", "\u05d1\u05bc\u05b4\u05e9\u05c1\u05b0\u05d1\u05bb\u05e2\u05b8\u05bd\u05d4", "\u05e2\u05d5\u05bc\u05dc", "\u05d9\u05b8\u05d3", "\u05d5\u05b0\u05bd\u05d4\u05b8\u05d9\u05b0\u05ea\u05b8\u05d4", "\u05d1\u05bc\u05b0\u05d2\u05b6\u05e9\u05c1\u05b6\u05ea", "\u05d0\u05b5\u05dd", "\u05e1\u05b0\u05dc\u05b7\u05bd\u05d7", "\u05d5\u05b7\u05ea\u05bc\u05b5\u05d8", "\u05d5\u05bc\u05e9\u05c2\u05b0\u05d1\u05b8\u05dd", "\u05d6\u05d5\u05bc\u05e8", "\u05d4\u05b2\u05d1\u05b5\u05d0\u05ea\u05b6\u05dd", "\u05db\u05bc\u05b4\u05d6\u05b0\u05e8\u05b7\u05e2", "\u05e8\u05b0\u05e4\u05b8\u05d0", "\u05e4\u05bc\u05b8\u05bd\u05e8\u05b6\u05e5", "\u05d4\u05b8\u05d9\u05bd\u05d5\u05bc", "\u05ea\u05b0\u05e0\u05b4\u05d9\u05d0\u05d5\u05bc\u05df", "\u05d9\u05b0\u05e8\u05b5\u05db\u05b8\u05d4\u05bc", "\u05e7\u05b9\u05d3\u05b6\u05e9\u05c1", "\u05e9\u05c2\u05b8\u05bd\u05e8\u05b7\u05d7", "\u05d5\u05bc\u05de\u05b4\u05e8\u05b0\u05d9\u05b8\u05dd", "\u05e2\u05b8\u05d3\u05b7\u05d9", "\u05e0\u05b8\u05d3\u05b8\u05d1", "\u05d4\u05b7\u05de\u05bc\u05b7\u05de\u05b0\u05e2\u05b4\u05d9\u05d8", "\u05de\u05b4\u05e4\u05bc\u05b0\u05e7\u05d5\u05bc\u05d3\u05b5\u05d9", "\u05de\u05b4\u05ea\u05bc\u05b8\u05e8\u05b7\u05d7", "\u05d5\u05b0\u05d4\u05b6\u05d7\u05b1\u05e8\u05b4\u05d9\u05e9\u05c1", "\u05d4\u05b4\u05bd\u05ea\u05b0\u05d2\u05bc\u05b7\u05dc\u05bc\u05b0\u05d7\u05d5\u05b9", "\u05e9\u05b0\u05c1\u05d1\u05b7\u05d8", "\u05d5\u05b0\u05dc\u05d5\u05bc", "\u05d5\u05b0\u05e7\u05b7\u05d1\u05bc\u05b9\u05ea\u05d5\u05b9", "\u05d5\u05b7\u05e2\u05b2\u05dc\u05b4\u05d9\u05ea\u05b6\u05dd", "\u05e7\u05b7\u05e2\u05b2\u05e8\u05b9\u05ea", "\u05d1\u05bc\u05b0\u05d3\u05b8\u05e4\u05b0\u05e7\u05b8\u05bd\u05d4", "\u05d4\u05b7\u05d7\u05b4\u05d9\u05e8\u05b9\u05ea", "\u05dc\u05b0\u05de\u05b7\u05d8\u05bc\u05d5\u05b9\u05ea", "\u05d4\u05b7\u05d3\u05bc\u05d5\u05b9\u05e8", "\u05d1\u05bc\u05bb\u05e7\u05bc\u05b4\u05d9", "\u05de\u05b4\u05d6\u05b0\u05e8\u05b8\u05e7", "\u05dc\u05b7\u05e2\u05b2\u05d5\u05ba\u05df", "\u05d9\u05b8\u05e8\u05b7\u05e7", "\u05d1\u05bc\u05b4\u05dc\u05b0\u05e2\u05b8\u05bd\u05dd", "\u05d5\u05b7\u05e0\u05bc\u05b7\u05e7\u05b0\u05e8\u05b5\u05d1", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e9\u05c2\u05bc\u05b8\u05d0\u05bb\u05d4\u05d5\u05bc", "\u05dc\u05b4\u05d1\u05b0\u05d4\u05b6\u05de\u05b0\u05ea\u05bc\u05b8\u05dd", "\u05d2\u05bc\u05b0\u05d3\u05b9\u05dc\u05b9\u05ea", "\u05de\u05b4\u05e9\u05c1\u05b0\u05e8\u05b7\u05ea", "\u05de\u05b4\u05d3\u05b0\u05d9\u05b8\u05df", "\u05d1\u05bc\u05b7\u05e9\u05c1\u05bc\u05b4\u05d8\u05bc\u05b4\u05d9\u05dd", "\u05d1\u05bc\u05b0\u05e4\u05d5\u05bc\u05e0\u05b9\u05bd\u05df", "\u05d4\u05b7\u05bd\u05de\u05bc\u05b7\u05e2\u05b2\u05e9\u05c2\u05b5\u05e8", "\u05d9\u05b4\u05e9\u05b8\u05bc\u05c2\u05e9\u05db\u05b8\u05e8", "\u05e7\u05e8\u05d5\u05d0\u05d9", "\u05d5\u05b0\u05d4\u05bd\u05b7\u05d7\u05b2\u05e8\u05b7\u05de\u05b0\u05ea\u05bc\u05b4\u05d9", "\u05d5\u05b0\u05e7\u05b8\u05e8\u05b0\u05d1\u05bc\u05b8\u05e0\u05d5\u05b9", "\u05e6\u05b0\u05d1\u05b8\u05d0\u05d5\u05b9", "\u05de\u05b0\u05e0\u05d5\u05bc\u05d7\u05b8\u05d4", "\u05d0\u05b2\u05d1\u05b9\u05ea\u05b8\u05bd\u05dd", "\u05d0\u05b7\u05d7\u05b6\u05e8\u05b6\u05ea", "\u05dc\u05b0\u05d4\u05b7\u05e0\u05bc\u05b4\u05d9\u05d7\u05d5\u05b9", "\u05e0\u05b9\u05db\u05b7\u05d7", "\u05d0\u05b1\u05dc\u05b4\u05d9\u05e6\u05d5\u05bc\u05e8", "\u05e9\u05c2\u05b0\u05e4\u05b8\u05ea\u05b6\u05d9\u05d4\u05b8", "\u05e4\u05bc\u05b0\u05e0\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05d4\u05b7\u05de\u05bc\u05b4\u05ea\u05b0\u05d0\u05b7\u05d5\u05bc\u05b4\u05bd\u05d9\u05dd", "\u05d9\u05b0\u05e8\u05b5\u05e9\u05c1\u05b8\u05d4", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05b4\u05de\u05b0\u05e2\u05b4\u05d9", "\u05d0\u05b7\u05d7\u05b7\u05e8", "\u05dc\u05b8\u05e2\u05b5\u05d3\u05b8\u05d4", "\u05de\u05b0\u05dc\u05b5\u05d0\u05b9\u05ea", "\u05e7\u05b8\u05d3\u05b5\u05bd\u05e9\u05c1\u05d5\u05bc", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e9\u05c1\u05bc\u05b8\u05d0\u05b2\u05e8\u05d5\u05bc", "\u05e4\u05b8\u05ea\u05d5\u05bc\u05d7\u05b7", "\u05d5\u05d9\u05d9\u05e8\u05e9", "\u05d9\u05b8\u05e2\u05b7\u05e5", "\u05d9\u05b8\u05dc\u05b4\u05d9\u05d3", "\u05d4\u05b5\u05e0\u05b4\u05d9\u05d0", "\u05dc\u05b0\u05d6\u05b4\u05d1\u05b0\u05d7\u05b5\u05d9", "\u05e4\u05bc\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05de\u05b5\u05d0\u05b9\u05d1\u05b9\u05ea", "\u05d7\u05b5\u05e8\u05b6\u05dd", "\u05ea\u05bc\u05b4\u05e7\u05b0\u05e6\u05b8\u05e8", "\u05de\u05b4\u05e9\u05c1\u05b0\u05db\u05bc\u05b7\u05d1", "\u05de\u05d5\u05b9\u05e6\u05b8\u05d0\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05d1\u05bc\u05b0\u05de\u05b9\u05ea\u05d5\u05b9", "\u05db\u05bc\u05b0\u05de\u05b4\u05e1\u05b0\u05e4\u05bc\u05b8\u05e8\u05b8\u05bd\u05dd", "\u05ea\u05bc\u05b4\u05d1\u05bc\u05b8\u05e0\u05b6\u05d4", "\u05de\u05d5\u05b9\u05d0\u05b8\u05d1", "\u05d4\u05b8\u05bd\u05d0\u05b9\u05ea\u05d5\u05b9\u05ea", "\u05dc\u05b6\u05d4\u05b8\u05d1\u05b8\u05d4", "\u05de\u05b5\u05d4\u05b2\u05dc\u05b9\u05da\u05b0", "\u05dc\u05b0\u05e2\u05b7\u05de\u05b0\u05e8\u05b8\u05dd", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05dc\u05b0\u05db\u05bc\u05b9\u05d3", "\u05d4\u05b8\u05bd\u05d0\u05b8\u05d7\u05bb\u05d6", "\u05ea\u05b8\u05de\u05b4\u05d9\u05d3", "\u05de\u05b4\u05dc\u05b0\u05de\u05b8\u05e2\u05b0\u05dc\u05b8\u05d4", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e8\u05b0\u05d0\u05d5\u05bc", "\u05d1\u05bc\u05b7\u05db\u05bc\u05b8\u05ea\u05b5\u05e3", "\u05dc\u05b0\u05d7\u05b6\u05e6\u05b0\u05e8\u05b9\u05df", "\u05de\u05b8\u05e9\u05c1\u05d5\u05bc", "\u05d9\u05b7\u05d0\u05b2\u05db\u05b4\u05dc\u05b5\u05e0\u05d5\u05bc", "\u05dc\u05b7\u05d0\u05b2\u05d1\u05b9\u05ea\u05b8\u05bd\u05d9\u05d5", "\u05d5\u05bc\u05de\u05b4\u05de\u05bc\u05b7\u05d7\u05b2\u05e6\u05b4\u05ea", "\u05ea\u05b8\u05e8\u05b4\u05bd\u05d9\u05e2\u05d5\u05bc", "\u05e8\u05b0\u05d0\u05b5\u05dd", "\u05d5\u05b0\u05d9\u05b8\u05d0\u05b4\u05d9\u05e8", "\u05d0\u05b5\u05dc\u05bc\u05b6\u05d4", "\u05e0\u05b0\u05d7\u05b9\u05e9\u05c1\u05b6\u05ea", "\u05d5\u05b7\u05ea\u05bc\u05b4\u05d1\u05b0\u05db\u05bc\u05d5\u05bc", "\u05e4\u05bc\u05b4\u05d9\u05d4\u05b8", "\u05ea\u05bc\u05b0\u05ea\u05b8\u05d0\u05d5\u05bc", "\u05e2\u05b2\u05d1\u05b9\u05d3\u05b8\u05d4", "\u05d5\u05b2\u05e0\u05b4\u05d6\u05b0\u05db\u05bc\u05b7\u05e8\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05de\u05b8\u05d7\u05b8\u05bd\u05e8", "\u05d1\u05bc\u05b0\u05bd\u05de\u05b5\u05e2\u05b7\u05d9\u05b4\u05da\u05b0", "\u05d4\u05b7\u05de\u05bc\u05b7\u05dc\u05b0\u05e7\u05d5\u05b9\u05d7\u05b7", "\u05d4\u05b8\u05e8\u05b4\u05d1\u05b0\u05dc\u05b8\u05d4", "\u05d5\u05b0\u05d4\u05b5\u05e7\u05b4\u05d9\u05dd", "\u05d6\u05b8\u05e2\u05b7\u05dd", "\u05d5\u05b0\u05d9\u05b8\u05e8\u05b5\u05dd", "\u05d9\u05b8\u05de\u05bb\u05ea\u05d5\u05bc", "\u05d9\u05b4\u05e7\u05bc\u05b8\u05e8\u05b5\u05d4", "\u05e0\u05b0\u05ea\u05b7\u05ea\u05bc\u05b4\u05bd\u05d9\u05dd", "\u05ea\u05bc\u05b7\u05d1\u05b0\u05e2\u05b5\u05e8\u05b8\u05d4", "\u05de\u05b4\u05bd\u05d6\u05b0\u05e8\u05b0\u05e7\u05b5\u05d9", "\u05e4\u05b6\u05e8\u05b7\u05d7", "\u05d1\u05b4\u05e0\u05b0\u05d3\u05b8\u05d1\u05b8\u05d4", "\u05dc\u05b0\u05e4\u05b6\u05e8\u05b6\u05e5", "\u05db\u05bc\u05b8\u05bd\u05dc", "\u05d7\u05b7\u05dc\u05b8\u05bc\u05d4", "\u05e9\u05c1\u05d5\u05b9\u05e7", "\u05e7\u05b8\u05d4\u05b8\u05dc", "\u05d5\u05b0\u05e9\u05c2\u05b8\u05d0", "\u05de\u05b5\u05e8\u05b0\u05e4\u05b4\u05d9\u05d3\u05b4\u05dd", "\u05e0\u05b8\u05d6\u05b4\u05d9\u05e8", "\u05e9\u05c1\u05b0\u05dc\u05b8\u05e9\u05c1\u05b0\u05ea\u05bc\u05b0\u05db\u05b6\u05dd", "\u05d0\u05b2\u05e0\u05b4\u05d9", "\u05e2\u05b5\u05e7\u05b6\u05d1", "\u05d5\u05b0\u05d0\u05d5\u05b9\u05ea\u05b8\u05d4\u05bc", "\u05d0\u05bb\u05de\u05b0\u05e0\u05b8\u05dd", "\u05d0\u05b8\u05de\u05d5\u05b9\u05e8", "\u05db\u05bc\u05b0\u05d4\u05bb\u05e0\u05bc\u05b8\u05bd\u05d4", "\u05e1\u05b0\u05d1\u05b4\u05d9\u05d1\u05b9\u05ea\u05b5\u05d9\u05e0\u05d5\u05bc", "\u05d4\u05b7\u05e0\u05bc\u05b8\u05d7\u05b8\u05e9\u05c1", "\u05d1\u05bc\u05b7\u05d0\u05b2\u05db\u05b9\u05dc", "\u05e9\u05c2\u05b0\u05e2\u05b7\u05e8", "\u05de\u05b4\u05e9\u05b0\u05c1\u05d7\u05b8\u05d4", "\u05e9\u05c1\u05b8\u05dc\u05d5\u05b9\u05e9\u05c1", "\u05e2\u05b6\u05d1\u05b6\u05d3", "\u05d0\u05d5\u05bc\u05dc\u05b7\u05d9", "\u05d9\u05b4\u05bd\u05d6\u05bc\u05b7\u05dc", "\u05e4\u05bc\u05bd\u05b8\u05e7\u05b0\u05d3\u05d5\u05bc", "\u05e9\u05b6\u05c1\u05e7\u05b6\u05dc", "\u05dc\u05b0\u05d4\u05b4\u05dc\u05bc\u05b8\u05d7\u05b6\u05dd", "\u05e8\u05b0\u05d0\u05d5\u05bc\u05d1\u05b5\u05df", "\u05db\u05bc\u05b4\u05ea\u05b0\u05e8\u05d5\u05bc\u05de\u05b7\u05ea", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05d3\u05bc\u05b7\u05e8", "\u05d1\u05b8\u05bc\u05e8\u05d5\u05bc\u05da\u05b0", "\u05ea\u05bc\u05d5\u05bc\u05e8", "\u05db\u05bc\u05b0\u05de\u05b4\u05ea\u05b0\u05d0\u05b9\u05e0\u05b0\u05e0\u05b4\u05d9\u05dd", "\u05de\u05b4\u05e9\u05b0\u05c1\u05de\u05b8\u05e8", "\u05d5\u05b0\u05d7\u05b8\u05d3\u05b7\u05dc", "\u05de\u05b4\u05db\u05bc\u05b8\u05d1\u05bd\u05d5\u05b9\u05d3", "\u05d2\u05b4\u05bc\u05dc\u05b0\u05e2\u05b8\u05d3", "\u05d5\u05b7\u05d0\u05b2\u05d1\u05b4\u05d9\u05e8\u05b8\u05dd", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05bd\u05e2\u05b2\u05de\u05b4\u05d3\u05b5\u05d4\u05d5\u05bc", "\u05d0\u05b6\u05e4\u05b0\u05e8\u05b7\u05d9\u05b4\u05dd", "\u05d1\u05bc\u05b0\u05de\u05d5\u05b9\u05e9\u05c1\u05b0\u05d1\u05b9\u05ea\u05b8\u05dd", "\u05d5\u05b0\u05dc\u05b8\u05d0\u05b8\u05e8\u05b6\u05e5", "\u05dc\u05b8\u05e0\u05b6\u05e4\u05b6\u05e9\u05c1", "\u05d9\u05d5\u05b9\u05dd", "\u05dc\u05b0\u05e6\u05b5\u05d0\u05ea\u05b8\u05dd", "\u05de\u05b5\u05bd\u05e8\u05b7\u05e2\u05b0\u05de\u05b0\u05e1\u05b5\u05e1", "\u05d5\u05b0\u05d8\u05b8\u05d7\u05b2\u05e0\u05d5\u05bc", "\u05d9\u05b4\u05d1\u05b0\u05e8\u05b8\u05d0", "\u05d5\u05b7\u05bd\u05d9\u05bc\u05b7\u05d7\u05b2\u05d1\u05b9\u05e9\u05c1", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05b0\u05d1\u05b4\u05d9\u05e2\u05b4\u05d9", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05d2\u05bc\u05b5\u05d3", "\u05e9\u05c1\u05d5\u05bc\u05d1", "\u05d5\u05b0\u05e7\u05b8\u05de\u05d5\u05bc", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05b0\u05de\u05b5\u05e0\u05b8\u05d4", "\u05dc\u05b0\u05d0\u05b8\u05e9\u05c1\u05b5\u05e8", "\u05e2\u05b2\u05d1\u05d5\u05b9\u05e8", "\u05e2\u05b2\u05d1\u05b9\u05d3\u05b7\u05ea", "\u05dc\u05b0\u05db\u05b7\u05e1\u05bc\u05b9\u05ea", "\u05ea\u05dc\u05d5\u05e0\u05d5", "\u05d5\u05b7\u05d4\u05b2\u05e8\u05b5\u05de\u05b9\u05ea\u05b6\u05dd", "\u05d4\u05b7\u05d8\u05bc\u05b8\u05d4\u05b9\u05e8", "\u05d1\u05bc\u05b0\u05d0\u05b5\u05e8", "\u05e8\u05b6\u05e7\u05b6\u05dd", "\u05d1\u05b7\u05bc\u05e8\u05b0\u05d6\u05b6\u05dc", "\u05ea\u05b7\u05d7\u05b2\u05e0\u05b4\u05d9\u05e4\u05d5\u05bc", "\u05e0\u05b8\u05e9\u05b4\u05c2\u05d9\u05d0", "\u05e6\u05b9\u05e8\u05b0\u05e8\u05b4\u05d9\u05dd", "\u05ea\u05bc\u05b4\u05d6\u05b0\u05db\u05bc\u05b0\u05e8\u05d5\u05bc", "\u05dc\u05b0\u05d3\u05b4\u05d2\u05b0\u05dc\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05e9\u05b8\u05c1\u05db\u05b7\u05dd", "\u05de\u05b5\u05bd\u05e2\u05b7\u05de\u05bc\u05b6\u05d9\u05d4\u05b8", "\u05d8\u05b8\u05d4\u05b3\u05e8\u05b8\u05ea\u05d5\u05b9", "\u05d4\u05b8\u05e4\u05b5\u05e8", "\u05d7\u05b8\u05e4\u05b5\u05e5", "\u05e2\u05b8\u05dc\u05b7\u05d9", "\u05d5\u05b0\u05d0\u05b6\u05bd\u05e2\u05b1\u05e9\u05c2\u05b6\u05d4", "\u05d4\u05b7\u05e4\u05bc\u05b4\u05d3\u05b0\u05d9\u05d5\u05b9\u05dd", "\u05e8\u05b8\u05e2\u05b8\u05d4", "\u05d5\u05b0\u05d2\u05b4\u05dc\u05bc\u05b7\u05d7", "\u05e9\u05c1\u05b5\u05e9\u05c1\u05b6\u05ea", "\u05dc\u05b0\u05d0\u05b8\u05d1\u05b4\u05d9\u05d5", "\u05db\u05bc\u05b7\u05e4\u05bc\u05d5\u05b9\u05ea", "\u05de\u05b7\u05de\u05b0\u05dc\u05b8\u05db\u05b8\u05d4", "\u05d0\u05b8\u05e9\u05b8\u05c1\u05dd", "\u05d4\u05b8\u05d0\u05b5\u05d9\u05e4\u05b8\u05d4", "\u05e0\u05b7\u05d7\u05b0\u05e9\u05c1\u05d5\u05b9\u05df", "\u05d1\u05b7\u05bc\u05d3", "\u05de\u05b5\u05e8\u05b4\u05de\u05bc\u05b9\u05df", "\u05d4\u05b7\u05d1\u05bc\u05b0\u05d3\u05b9\u05bd\u05dc\u05b7\u05d7", "\u05d0\u05b8\u05d3\u05d5\u05b9\u05df", "\u05d7\u05b7\u05d8\u05bc\u05b7\u05d0\u05ea\u05b0\u05db\u05b6\u05dd", "\u05db\u05bc\u05b8\u05e8\u05d5\u05bc\u05d4\u05b8", "\u05de\u05b0\u05e7\u05d5\u05b9\u05de\u05b6\u05da\u05b8", "\u05d7\u05b2\u05dc\u05d5\u05bc\u05e6\u05b4\u05d9\u05dd", "\u05d1\u05bc\u05b4\u05d2\u05b0\u05d1\u05bb\u05dc\u05b9\u05ea", "\u05dc\u05d5\u05bc\u05df", "\u05d4\u05b7\u05d7\u05d5\u05bc\u05e4\u05b8\u05de\u05b4\u05bd\u05d9", "\u05e6\u05b4\u05e4\u05bc\u05d5\u05bc\u05d9", "\u05d5\u05b0\u05d0\u05b8\u05bd\u05e9\u05c1\u05b0\u05de\u05b8\u05d4", "\u05e9\u05c1\u05b0\u05d2\u05b8\u05d2\u05b8\u05d4", "\u05de\u05b5\u05d7\u05b2\u05e8\u05b8\u05d3\u05b8\u05d4", "\u05dc\u05b0\u05d3\u05b8\u05ea\u05b8\u05df", "\u05dc\u05b0\u05d0\u05b4\u05e9\u05c1\u05b0\u05ea\u05bc\u05d5\u05b9", "\u05d1\u05b0\u05d0\u05b7\u05e8\u05b0\u05e6\u05b6\u05da\u05b8", "\u05d6\u05b8\u05bd\u05e8\u05b7\u05e2", "\u05d4\u05b7\u05bd\u05d4\u05b9\u05dc\u05b0\u05db\u05b4\u05d9\u05dd", "\u05d0\u05b5\u05d9\u05dc", "\u05d9\u05b8\u05e6\u05b8\u05d0\u05e0\u05d5\u05bc", "\u05d4\u05b4\u05e7\u05b0\u05e8\u05b4\u05d9\u05d1\u05bb\u05dd", "\u05d5\u05b7\u05d0\u05b2\u05db\u05b7\u05dc\u05bc\u05b6\u05d4", "\u05d5\u05b0\u05d4\u05d5\u05b9\u05e6\u05b4\u05d9\u05d0", "\u05d5\u05bc\u05de\u05b7\u05d6\u05bc\u05b5\u05d4", "\u05dc\u05b0\u05d0\u05b9\u05bd\u05d4\u05b6\u05dc", "\u05d4\u05b7\u05d2\u05bc\u05b6\u05d1\u05b6\u05e8", "\u05e9\u05b8\u05c1\u05db\u05b7\u05d1", "\u05de\u05b7\u05e2\u05b2\u05e9\u05b5\u05c2\u05e8", "\u05dc\u05b0\u05db\u05b7\u05e8\u05b0\u05de\u05b4\u05d9", "\u05d0\u05b8\u05db\u05b7\u05dc", "\u05d1\u05bc\u05b4\u05d1\u05b0\u05e0\u05b5\u05d9", "\u05de\u05b4\u05d9", "\u05e9\u05c1\u05b0\u05e7\u05b5\u05d3\u05b4\u05bd\u05d9\u05dd", "\u05d4\u05b7\u05e0\u05bc\u05b4\u05e6\u05b0\u05de\u05b8\u05d3\u05b4\u05d9\u05dd", "\u05d9\u05b0\u05d2\u05b7\u05dc\u05bc\u05b0\u05d7\u05b6\u05bd\u05e0\u05bc\u05d5\u05bc", "\u05e8\u05b8\u05d7\u05b7\u05e5", "\u05e9\u05b8\u05c1\u05dd", "\u05d5\u05b0\u05e0\u05b4\u05e1\u05b0\u05dc\u05b7\u05d7", "\u05d1\u05b8\u05e8\u05b5\u05d7\u05b7\u05d9\u05b4\u05dd", "\u05d4\u05b4\u05e0\u05bc\u05b6\u05e0\u05bc\u05d5\u05bc", "\u05e9\u05b0\u05c1\u05e0\u05b7\u05d9\u05b4\u05dd", "\u05d4\u05b4\u05ea\u05b0\u05e2\u05b7\u05dc\u05bc\u05b7\u05dc\u05b0\u05ea\u05bc\u05b0", "\u05d5\u05bc\u05d1\u05b0\u05e0\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05d1\u05b0\u05e0\u05b6\u05e4\u05b6\u05e9\u05c1", "\u05de\u05b4\u05e1\u05b0\u05e4\u05bc\u05b7\u05e8\u05b0\u05db\u05b6\u05dd", "\u05dc\u05b0\u05d0\u05b7\u05dc\u05b0\u05e4\u05b5\u05d9", "\u05de\u05b4\u05e1\u05b0\u05e4\u05b8\u05bc\u05e8", "\u05dc\u05b0\u05e9\u05c1\u05b8\u05e8\u05b0\u05ea\u05b8\u05bd\u05dd", "\u05d0\u05b7\u05d7\u05b5\u05e8", "\u05d4\u05b6\u05d7\u05b1\u05e8\u05b4\u05e9\u05c1", "\u05e0\u05b4\u05d3\u05b8\u05bc\u05d4", "\u05de\u05b4\u05e7\u05b0\u05e9\u05b8\u05c1\u05d4", "\u05d4\u05b7\u05de\u05bc\u05b9\u05e9\u05c1\u05b0\u05dc\u05b4\u05d9\u05dd", "\u05d1\u05bc\u05b8\u05e0\u05d5\u05b9\u05ea", "\u05dc\u05b8\u05d1\u05b4\u05d9\u05d0", "\u05de\u05b0\u05e0\u05d5\u05b9\u05e8\u05b8\u05d4", "\u05db\u05b4\u05dc\u05bc\u05b4\u05d9\u05ea\u05b4\u05d9", "\u05e9\u05b8\u05c1\u05d1\u05b8\u05d4", "\u05dc\u05b0\u05d9\u05b4\u05e6\u05b0\u05d7\u05b8\u05e7", "\u05d4\u05b7\u05db\u05bc\u05b0\u05e0\u05b7\u05e2\u05b2\u05e0\u05b4\u05d9", "\u05dc\u05b0\u05e9\u05c1\u05b7\u05d3", "\u05e7\u05b8\u05e8\u05b0\u05d1\u05bc\u05b8\u05e0\u05b4\u05d9", "\u05e9\u05c1\u05d5\u05b9\u05e8", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05ea\u05b0\u05d9\u05b7\u05bd\u05dc\u05b0\u05d3\u05d5\u05bc", "\u05d5\u05b7\u05d9\u05b0\u05d2\u05b7\u05dc", "\u05d4\u05b8\u05d0\u05b8\u05dc\u05b8\u05d4", "\u05d0\u05b7\u05d6\u05b0\u05db\u05b8\u05bc\u05e8\u05b8\u05d4", "\u05d1\u05b7\u05d3\u05bc\u05b6\u05d1\u05b6\u05e8", "\u05d0\u05b2\u05d1\u05b4\u05d9\u05d4\u05d5\u05bc\u05d0", "\u05d1\u05b4\u05bc\u05e0\u05b0\u05d9\u05b8\u05de\u05b4\u05d9\u05df", "\u05d4\u05b7\u05ea\u05bc\u05b0\u05e9\u05c1\u05b4\u05d9\u05e2\u05b4\u05d9", "\u05e0\u05b8\u05e9\u05c1\u05b7\u05da\u05b0", "\u05e0\u05b8\u05d8\u05b7\u05e2", "\u05d5\u05bc\u05d1\u05b7\u05d1\u05bc\u05b0\u05d4\u05b5\u05de\u05b8\u05bd\u05d4", "\u05e0\u05b8\u05d0\u05b7\u05e5", "\u05d4\u05b7\u05e0\u05bc\u05d5\u05b9\u05e2\u05b8\u05d3\u05b4\u05d9\u05dd", "\u05d1\u05bc\u05b0\u05e7\u05b4\u05d1\u05b0\u05e8\u05b9\u05ea", "\u05d4\u05b7\u05de\u05bc\u05bd\u05d5\u05b9\u05d8", "\u05d4\u05b8\u05e2\u05b9\u05d3\u05b0\u05e4\u05b4\u05d9\u05dd", "\u05d1\u05bc\u05b0\u05e2\u05b7\u05dc\u05b0\u05de\u05b9\u05df", "\u05e7\u05e8\u05d9\u05d0\u05d9", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05b8\u05e0\u05b8\u05bd\u05d4", "\u05d1\u05bc\u05b8\u05e9\u05c2\u05b8\u05bd\u05e8", "\u05e0\u05b6\u05d2\u05b6\u05d1", "\u05d9\u05bd\u05b7\u05e2\u05b7\u05d1\u05b0\u05e8\u05d5\u05bc", "\u05d5\u05b0\u05e6\u05b8\u05d1\u05b0\u05ea\u05b8\u05d4", "\u05e6\u05b8\u05e2\u05b7\u05e7", "\u05d5\u05bc\u05d1\u05b6\u05e2\u05b8\u05e9\u05c2\u05d5\u05b9\u05e8", "\u05e0\u05b4\u05e1\u05b0\u05db\u05bc\u05bd\u05d5\u05b9", "\u05e2\u05b8\u05e8\u05b7\u05db\u05b0\u05ea\u05bc\u05b4\u05d9", "\u05d5\u05bc\u05de\u05b4\u05d1\u05bc\u05b6\u05df", "\u05dc\u05b8\u05d0\u05b8\u05bd\u05d9\u05b4\u05dc", "\u05d5\u05b0\u05d9\u05b8\u05e0\u05bb\u05e1\u05d5\u05bc", "\u05d2\u05bc\u05b4\u05dc\u05b0\u05e2\u05b8\u05d3\u05b8\u05d4", "\u05db\u05b8\u05bc\u05d1\u05d5\u05b9\u05d3", "\u05d0\u05b2\u05ea\u05b9\u05bd\u05e0\u05b0\u05da\u05b8", "\u05d5\u05b0\u05e0\u05b7\u05e2\u05b2\u05dc\u05b8\u05d4", "\u05d5\u05bc\u05d1\u05b8\u05dc\u05b0\u05e2\u05b8\u05d4", "\u05d9\u05b7\u05e8\u05b0\u05d3\u05bc\u05b5\u05df", "\u05d0\u05b6\u05e7\u05bc\u05b8\u05d7\u05b2\u05da\u05b8", "\u05d5\u05b0\u05e9\u05c1\u05b5\u05e8\u05b5\u05ea", "\u05d9\u05b6\u05d4\u05b0\u05d3\u05bc\u05b8\u05e4\u05b6\u05e0\u05bc\u05d5\u05bc", "\u05d3\u05bc\u05b4\u05d1\u05b0\u05dc\u05b8\u05ea\u05b8\u05bd\u05d9\u05b0\u05de\u05b8\u05d4", "\u05d8\u05b6\u05e8\u05b6\u05e3", "\u05de\u05b4\u05e9\u05c2\u05bc\u05bb\u05de\u05d5\u05b9", "(\u05c6)", "\u05d4\u05b8\u05bd\u05d0\u05b6\u05d6\u05b0\u05e8\u05b8\u05d7", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05e0\u05bc\u05b7\u05d7", "\u05d5\u05b0\u05d9\u05b4\u05d4\u05b0\u05d9\u05d5\u05bc", "\u05d1\u05b4\u05d8\u05b0\u05e0\u05b8\u05d4\u05bc", "\u05de\u05b5\u05bd\u05ea\u05d5\u05bc", "\u05de\u05b4\u05e4\u05bc\u05b8\u05e0\u05b7\u05d9", "\u05dc\u05b7\u05d7\u05b9\u05bd\u05d3\u05b6\u05e9\u05c1", "\u05de\u05b0\u05e6\u05b9\u05e8\u05b8\u05bd\u05e2\u05b7\u05ea", "\u05d5\u05bc\u05e9\u05c1\u05b0\u05e0\u05b4\u05d9\u05bc\u05b4\u05dd", "\u05db\u05bc\u05b7\u05d0\u05b2\u05d4\u05b8\u05dc\u05b4\u05d9\u05dd", "\u05e2\u05b8\u05e0\u05b8\u05e7", "\u05d4\u05b6\u05d7\u05b8\u05de\u05d5\u05bc\u05dc\u05b4\u05bd\u05d9", "\u05d5\u05bc\u05d3\u05b0\u05d1\u05b7\u05e9\u05c1", "\u05d4\u05b8\u05d0\u05b2\u05e8\u05d5\u05b9\u05d3\u05b4\u05d9", "\u05d9\u05b8\u05e8\u05b5\u05da\u05b0", "\u05d5\u05b0\u05e1\u05b8\u05de\u05b7\u05db\u05b0\u05ea\u05bc\u05b8", "\u05d9\u05b8\u05e4\u05b5\u05e8", "\u05d7\u05b5\u05dc\u05b6\u05e7", "\u05d2\u05b8\u05bc\u05d0\u05b7\u05dc", "\u05dc\u05b0\u05d2\u05bb\u05dc\u05b0\u05d2\u05bc\u05b0\u05dc\u05b9\u05ea\u05b8\u05bd\u05dd", "\u05dc\u05b4\u05e9\u05c1\u05b0\u05e4\u05d5\u05bc\u05e4\u05b8\u05dd", "\u05d7\u05bb\u05e7\u05bc\u05b9\u05ea\u05b8\u05d9\u05d5", "\u05dc\u05b8\u05db\u05b6\u05dd", "\u05db\u05bc\u05b4\u05ea\u05b0\u05d1\u05d5\u05bc\u05d0\u05b7\u05ea", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05d6\u05b0\u05d1\u05bc\u05b7\u05d7", "\u05dc\u05b0\u05de\u05b4\u05e9\u05c1\u05b0\u05e4\u05bc\u05b7\u05d7\u05b7\u05ea", "\u05d5\u05b0\u05d4\u05b7\u05de\u05bc\u05b7\u05d2\u05bc\u05b5\u05e4\u05b8\u05d4", "\u05d5\u05b0\u05e2\u05b7\u05ea\u05bc\u05b8\u05d4", "\u05e9\u05b8\u05c1\u05d0\u05d5\u05bc\u05dc", "\u05d1\u05bc\u05b0\u05e0\u05b7\u05d7\u05b2\u05dc\u05b7\u05ea\u05b0\u05db\u05b6\u05dd", "\u05e4\u05b4\u05bc\u05e1\u05b0\u05d2\u05b8\u05bc\u05d4", "\u05d5\u05b0\u05d4\u05b7\u05e9\u05c1\u05bc\u05bb\u05dc\u05b0\u05d7\u05b8\u05df", "\u05d0\u05b6\u05e4\u05b0\u05e8\u05b8\u05d9\u05b4\u05dd", "\u05e0\u05b8\u05d8\u05b0\u05ea\u05b8\u05d4", "\u05de\u05b5\u05e2\u05b2\u05e4\u05b7\u05e8", "\u05d1\u05b8\u05e2\u05b2\u05e8\u05b8\u05d4", "\u05d1\u05bc\u05b0\u05e8\u05b4\u05de\u05bc\u05b9\u05df", "\u05d0\u05b8\u05d1\u05b7\u05d3\u05b0\u05e0\u05d5\u05bc", "\u05de\u05b4\u05d6\u05bc\u05b6\u05d4", "\u05d5\u05b0\u05e6\u05b8\u05de\u05b4\u05d9\u05d3", "\u05d7\u05b8\u05e8\u05d5\u05b9\u05df", "\u05d0\u05b7\u05da\u05b0", "\u05d4\u05b7\u05e7\u05bc\u05b0\u05d8\u05b9\u05bd\u05e8\u05b6\u05ea", "\u05de\u05b8\u05e6\u05b8\u05ea\u05b4\u05d9", "\u05d5\u05b0\u05d8\u05b7\u05e4\u05bc\u05b5\u05e0\u05d5\u05bc", "\u05d0\u05b7\u05d9\u05b4\u05dc", "\u05d1\u05b4\u05bc\u05db\u05bc\u05d5\u05bc\u05e8", "\u05d5\u05b0\u05e0\u05b6\u05d7\u05b0\u05e9\u05c1\u05b7\u05d1", "\u05de\u05b0\u05e6\u05b8\u05d0\u05b8\u05bd\u05ea\u05b0\u05e0\u05d5\u05bc", "\u05d0\u05b7\u05e3", "\u05d7\u05b7\u05d8\u05bc\u05b8\u05d0\u05ea\u05b8\u05dd", "\u05d1\u05bc\u05b6\u05d8\u05b6\u05df", "\u05d5\u05b0\u05d9\u05b8\u05e1\u05b5\u05e8", "\u05e2\u05b7\u05de\u05bc\u05b4\u05d9\u05e0\u05b8\u05d3\u05b8\u05d1", "\u05d5\u05b7\u05d0\u05b2\u05d1\u05b4\u05d9\u05e8\u05b8\u05bd\u05dd", "\u05d5\u05b7\u05d4\u05b2\u05e9\u05c1\u05b4\u05d1\u05b9\u05ea\u05b4\u05d9", "\u05d0\u05b2\u05e9\u05c1\u05b6\u05e8", "\u05ea\u05bc\u05b4\u05d1\u05b0\u05dc\u05b8\u05e2\u05b5\u05e0\u05d5\u05bc", "\u05d5\u05b0\u05e9\u05c1\u05b8\u05d0\u05b7\u05dc", "\u05d1\u05bc\u05b8\u05e0\u05bd\u05d5\u05bc", "\u05e2\u05b2\u05dc\u05b5\u05d9\u05db\u05b6\u05bd\u05dd", "\u05e2\u05b2\u05e9\u05c2\u05d5\u05bc", "\u05e9\u05b8\u05c1\u05ea\u05b8\u05d4", "\u05d4\u05b8\u05bd\u05d0\u05b8\u05d3\u05b8\u05dd", "\u05d5\u05b0\u05d9\u05b4\u05ea\u05b0\u05e0\u05b6\u05d7\u05b8\u05dd", "\u05d5\u05b0\u05d4\u05b4\u05ea\u05b0\u05d5\u05b7\u05d3\u05bc\u05d5\u05bc", "\u05d5\u05bc\u05e8\u05b0\u05e7\u05b4\u05d9\u05e7", "\u05d0\u05b9\u05d4\u05b6\u05dc", "\u05d4\u05b7\u05d8\u05bc\u05b0\u05de\u05b5\u05d0\u05b8\u05d4", "\u05d1\u05bc\u05b7\u05d7\u05b2\u05de\u05b4\u05e9\u05c1\u05bc\u05b8\u05d4", "\u05d1\u05b0\u05e2\u05b5\u05d9\u05e0\u05b5\u05d9\u05e0\u05d5\u05bc", "\u05d4\u05b5\u05e4\u05b7\u05e8", "\u05d5\u05b0\u05db\u05b9\u05d4", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e9\u05c1\u05b0\u05db\u05bc\u05b9\u05df", "\u05dc\u05b0\u05ea\u05b7\u05d7\u05b7\u05df", "\u05d1\u05bc\u05b7\u05d7\u05b9\u05d3\u05b6\u05e9\u05c1", "\u05d1\u05bc\u05b0\u05de\u05b4\u05e6\u05b0\u05e8\u05b8\u05d9\u05b4\u05dd", "\u05e0\u05b7\u05d7\u05b7\u05dc", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05d2\u05b0\u05de\u05b9\u05dc", "\u05d3\u05b8\u05bc\u05d1\u05b8\u05e8", "\u05dc\u05b0\u05e9\u05c1\u05b5\u05dc\u05b8\u05d4", "\u05d7\u05b8\u05d9\u05b8\u05d4", "\u05e9\u05b5\u05c1\u05dd", "\u05e6\u05b0\u05d1\u05b8\u05d0", "\u05d1\u05bc\u05b0\u05e7\u05d5\u05b9\u05dc", "\u05d5\u05b0\u05e2\u05b8\u05e9\u05c2\u05d5\u05bc", "\u05d5\u05b0\u05e9\u05c2\u05b4\u05d9\u05de\u05d5\u05bc", "\u05e8\u05b0\u05d2\u05b8\u05dc\u05b4\u05bd\u05d9\u05dd", "\u05d5\u05b0\u05d4\u05b5\u05d1\u05b5\u05d9\u05d0\u05ea\u05b4\u05d9", "\u05d5\u05bc\u05d1\u05b0\u05e0\u05bb\u05d7\u05b9\u05d4", "\u05de\u05b4\u05dc\u05bc\u05b4\u05d1\u05bc\u05b4\u05d9", "\u05d4\u05b7\u05de\u05bc\u05b4\u05d6\u05b0\u05e8\u05b8\u05e7", "\u05e0\u05b4\u05d8\u05bc\u05b8\u05d9\u05d5\u05bc", "\u05d5\u05bc\u05d2\u05b0\u05d3\u05b5\u05e8\u05b9\u05ea", "\u05e1\u05b4\u05bd\u05d9\u05df", "\u05e9\u05c1\u05b8\u05dc\u05b7\u05d7\u05b0\u05ea\u05bc\u05b8", "\u05e0\u05b7\u05e2\u05b2\u05dc\u05b6\u05bd\u05d4", "\u05d0\u05b5\u05dc\u05b6\u05bc\u05d4", "\u05d9\u05b4\u05e1\u05b0\u05e2\u05d5\u05bc", "\u05d4\u05b5\u05e8\u05b9\u05de\u05bc\u05d5\u05bc", "\u05d5\u05b7\u05ea\u05bc\u05b5\u05dc\u05b6\u05da\u05b0", "\u05db\u05bc\u05b0\u05d8\u05b7\u05e2\u05b7\u05dd", "\u05d4\u05b7\u05d2\u05bc\u05b8\u05e8", "\u05dc\u05b0\u05d7\u05b9\u05d1\u05b8\u05d1", "\u05e1\u05b0\u05d1\u05b4\u05d9\u05d1\u05d5\u05b9\u05ea", "\u05dc\u05b0\u05e2\u05b7\u05e0\u05bc\u05b9\u05ea", "\u05de\u05b4\u05d3\u05b0\u05d1\u05b8\u05bc\u05e8", "\u05d4\u05b7\u05d9\u05bc\u05b4\u05e6\u05b0\u05e8\u05b4\u05d9", "\u05d8\u05b8\u05d1\u05b7\u05dc", "\u05d5\u05b0\u05d4\u05b4\u05e0\u05bc\u05b7\u05d7\u05b0\u05ea\u05bc\u05b8\u05dd", "\u05d9\u05b7\u05d7\u05b2\u05e8\u05b4\u05d9\u05e9\u05c1", "\u05de\u05b5\u05d0\u05b8\u05d4", "\u05dc\u05b0\u05d7\u05b6\u05d1\u05b6\u05e8", "\u05de\u05d5\u05bc\u05dc", "\u05d5\u05b7\u05d4\u05b2\u05e9\u05c1\u05b4\u05db\u05bc\u05b9\u05ea\u05b4\u05d9", "\u05e9\u05b5\u05c2\u05e2\u05b4\u05d9\u05e8", "\u05e9\u05c1\u05b5\u05e0\u05b4\u05d9", "\u05dc\u05b4\u05e0\u05b0\u05de\u05d5\u05bc\u05d0\u05b5\u05dc", "\u05d1\u05bc\u05b8\u05d6\u05b8\u05bd\u05d6\u05d5\u05bc", "\u05ea\u05bc\u05b4\u05e9\u05c1\u05b0\u05dc\u05b0\u05d7\u05d5\u05bc", "\u05dc\u05b0\u05d9\u05b7\u05d7\u05b0\u05e6\u05b0\u05d0\u05b5\u05dc", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05b4\u05d8\u05bc\u05b4\u05d9\u05dd", "\u05de\u05b4\u05de\u05bc\u05b4\u05d3\u05b0\u05d1\u05bc\u05b7\u05e8", "\u05d5\u05bc\u05de\u05b4\u05e7\u05b0\u05e0\u05b6\u05d4", "\u05d5\u05b7\u05e0\u05bc\u05b4\u05e6\u05b0\u05e2\u05b7\u05e7", "\u05d1\u05bc\u05d5\u05b9\u05d0", "\u05d3\u05b0\u05d1\u05b8\u05e8\u05b4\u05d9", "\u05d5\u05bc\u05dc\u05b0\u05d0\u05b4\u05e1\u05bc\u05b7\u05e8", "\u05d4\u05b2\u05de\u05b4\u05df", "\u05e6\u05b4\u05d9\u05e6\u05b4\u05ea", "\u05db\u05bc\u05b7\u05dc\u05bc\u05d5\u05b9\u05ea", "\u05d5\u05b0\u05d4\u05b4\u05ea\u05b0\u05d0\u05b7\u05d5\u05bc\u05b4\u05d9\u05ea\u05b6\u05dd", "\u05e2\u05b7\u05d6", "\u05d5\u05b7\u05ea\u05bc\u05b4\u05e7\u05b0\u05e8\u05b7\u05d1\u05b0\u05e0\u05b8\u05d4", "\u05d5\u05bc\u05db\u05b0\u05db\u05b8\u05dc", "\u05de\u05b7\u05e9\u05c2\u05bc\u05b8\u05d0\u05d5\u05b9", "\u05dc\u05b0\u05e0\u05b7\u05bd\u05e2\u05b2\u05de\u05b8\u05df", "\u05db\u05bc\u05b0\u05de\u05b4\u05e9\u05c1\u05b0\u05e4\u05bc\u05b8\u05d8\u05b8\u05bd\u05dd", "\u05d5\u05bc\u05dc\u05b0\u05d0\u05b6\u05d6\u05b0\u05e8\u05b7\u05d7", "\u05e9\u05c1\u05bb\u05e4\u05bc\u05b7\u05da\u05b0", "\u05dc\u05b5\u05d1\u05b8\u05d1", "\u05de\u05b5\u05e2\u05b7\u05dc\u05b0\u05de\u05b9\u05df", "\u05e4\u05bc\u05b0\u05e7\u05bb\u05d3\u05bc\u05b7\u05ea", "\u05d5\u05b0\u05dc\u05b4\u05d1\u05b0\u05e0\u05b9\u05ea\u05b6\u05d9\u05da\u05b8", "\u05d0\u05b8\u05e9\u05c1\u05b5\u05e8", "\u05e0\u05b0\u05d0\u05bb\u05dd", "\u05d9\u05b8\u05d4\u05b0\u05e6\u05b8\u05d4", "\u05d7\u05d5\u05b9\u05e8\u05b4\u05bd\u05d9", "\u05de\u05b4\u05e6\u05b0\u05e8\u05b4\u05d9", "\u05d1\u05bc\u05b0\u05e8\u05b4\u05ea\u05b0\u05de\u05b8\u05bd\u05d4", "\u05d4\u05b4\u05e9\u05c1\u05bc\u05b8\u05d1\u05b7\u05e2", "\u05d1\u05bc\u05b0\u05e7\u05b7\u05e8\u05b0\u05e7\u05b7\u05e2", "\u05e9\u05c2\u05b8\u05d0\u05b5\u05d4\u05d5\u05bc", "\u05e2\u05b2\u05e6\u05b6\u05e8\u05b6\u05ea", "\u05d1\u05bc\u05b7\u05e6\u05bc\u05b8\u05d1\u05b8\u05d0", "\u05d1\u05bc\u05b0\u05e9\u05c1\u05b8\u05dc\u05b0\u05d7\u05b4\u05d9", "\u05d6\u05b6\u05d1\u05b7\u05d7", "\u05dc\u05b6\u05bd\u05e2\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05d0\u05b5\u05d6\u05d5\u05b9\u05d1", "\u05d1\u05bc\u05b0\u05e8\u05b4\u05d9\u05d0\u05b8\u05d4", "\u05e4\u05b6\u05e8\u05b6\u05e5", "\u05d4\u05bd\u05b8\u05e2\u05b2\u05e0\u05b8\u05e7", "\u05e0\u05b0\u05d1\u05bd\u05d5\u05b9", "\u05e2\u05b8\u05e4\u05b8\u05e8", "\u05d1\u05bc\u05b4\u05e0\u05b0\u05e1\u05b9\u05e2\u05b7", "\u05dc\u05b7\u05d0\u05b2\u05d7\u05b4\u05d9\u05e8\u05b8\u05dd", "\u05d4\u05b7\u05e0\u05bc\u05b5\u05e8\u05bd\u05d5\u05b9\u05ea", "\u05dc\u05b0\u05e2\u05b5\u05d9\u05e0\u05b8\u05bd\u05d9\u05b4\u05dd", "\u05d4\u05b2\u05e8\u05b5\u05e2\u05b9\u05ea\u05b8", "\u05d5\u05b0\u05e9\u05c1\u05b4\u05d1\u05b0\u05e2\u05b4\u05d9\u05dd", "\u05de\u05b8\u05df", "\u05dc\u05b8\u05e0\u05b8\u05bd\u05e4\u05b6\u05e9\u05c1", "\u05db\u05bc\u05b0\u05e4\u05b7\u05bd\u05e2\u05b7\u05dd", "\u05d7\u05b7\u05d8\u05bc\u05b9\u05d0\u05ea", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05bd\u05e9\u05c1\u05b0\u05ea\u05bc\u05b7\u05d7\u05b2\u05d5\u05bc\u05d5\u05bc", "\u05d3\u05bc\u05b4\u05de\u05bc\u05b4\u05d9\u05ea\u05b4\u05d9", "\u05d5\u05bc\u05e0\u05b0\u05e1\u05b8\u05db\u05b6\u05bd\u05d9\u05d4\u05b8", "\u05d4\u05b8\u05bd\u05e8\u05b0\u05e9\u05c1\u05b8\u05e2\u05b4\u05d9\u05dd", "\u05de\u05b4\u05e0\u05b0\u05d7\u05b8\u05ea\u05b8\u05dd", "\u05e9\u05c2\u05b8\u05e8\u05b8\u05e3", "\u05db\u05bc\u05b5\u05dc\u05b6\u05d9\u05d4\u05b8", "\u05dc\u05b0\u05e2\u05b5\u05e8\u05b4\u05d9", "\u05d0\u05b5\u05dc", "\u05d4\u05b8\u05bd\u05d0\u05b7\u05e8\u05b0\u05d3\u05bc\u05b4\u05d9", "\u05de\u05b4\u05e9\u05c1\u05b0\u05db\u05bc\u05b7\u05df", "\u05e0\u05b4\u05d8\u05b0\u05de\u05b0\u05d0\u05b8\u05d4", "\u05d1\u05bc\u05b8\u05d0\u05e0\u05d5\u05bc", "\u05d5\u05b0\u05d0\u05b5\u05d6\u05d5\u05b9\u05d1", "\u05d5\u05b0\u05e7\u05b7\u05dc\u05b0\u05e2\u05b5\u05d9", "\u05dc\u05b0\u05d1\u05b4\u05e0\u05b0\u05d9\u05b8\u05de\u05b4\u05df", "\u05d6\u05b6\u05d4", "\u05d0\u05b7\u05e8\u05b0\u05e0\u05b9\u05bd\u05df", "\u05dc\u05b7\u05e9\u05c1\u05bc\u05b4\u05de\u05b0\u05e2\u05b9\u05e0\u05b4\u05bd\u05d9", "\u05d5\u05b7\u05d9\u05bc\u05b9\u05e6\u05b4\u05d9\u05d0\u05d5\u05bc", "\u05d7\u05b2\u05e0\u05d5\u05b9\u05da\u05b0", "\u05de\u05b4\u05db\u05bc\u05b8\u05dc", "\u05d7\u05b7\u05d5\u05b8\u05bc\u05d4", "\u05d6\u05b8\u05db\u05b7\u05e8", "\u05d0\u05b7\u05d9\u05b4\u05df", "\u05d4\u05b7\u05e7\u05bc\u05d5\u05b9\u05dc", "\u05d5\u05bc\u05e4\u05b0\u05e7\u05bb\u05d3\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05d4\u05b7\u05de\u05bc\u05bb\u05db\u05bc\u05b8\u05d4", "\u05d1\u05bc\u05b7\u05dc\u05bc\u05b7\u05d9\u05b0\u05dc\u05b8\u05d4", "\u05d1\u05bc\u05b0\u05e9\u05c1\u05b8\u05d1\u05bb\u05e2\u05b9\u05ea\u05b5\u05d9\u05db\u05b6\u05dd", "\u05d0\u05b4\u05e9\u05b8\u05bc\u05c1\u05d4", "\u05e4\u05bc\u05b8\u05e2\u05b7\u05dc", "\u05dc\u05b0\u05de\u05d5\u05b9\u05e6\u05b8\u05d0\u05b5\u05d9\u05d4\u05b6\u05bd\u05dd", "\u05d5\u05bc\u05db\u05b0\u05de\u05b4\u05e9\u05c1\u05b0\u05e4\u05bc\u05b8\u05d8\u05d5\u05b9", "\u05dc\u05b4\u05de\u05b0\u05e7\u05b9\u05de\u05d5\u05b9", "\u05e2\u05b2\u05d3\u05b8\u05ea\u05bd\u05d5\u05b9", "\u05dc\u05b8\u05d0\u05b7\u05d9\u05b4\u05dc", "\u05de\u05b4\u05d1\u05bc\u05b5\u05d9\u05ea", "\u05d4\u05b7\u05d3\u05bc\u05b8\u05d2\u05b8\u05d4", "\u05d5\u05bc\u05de\u05b4\u05d2\u05b0\u05e8\u05b8\u05e9\u05c1", "\u05d1\u05bc\u05b7\u05d0\u05b2\u05db\u05b8\u05dc\u05b0\u05db\u05b6\u05dd", "\u05d5\u05b0\u05d4\u05b4\u05e0\u05bc\u05d5\u05b9", "\u05e2\u05b6\u05e8\u05b6\u05d1", "\u05dc\u05b0\u05de\u05b4\u05e9\u05c1\u05b0\u05db\u05bc\u05b7\u05df", "\u05ea\u05b0\u05bc\u05e8\u05d5\u05bc\u05de\u05b8\u05d4", "\u05d5\u05b7\u05bd\u05d9\u05bc\u05b7\u05e2\u05b2\u05dc\u05b5\u05d4\u05d5\u05bc", "\u05e6\u05b7\u05dc\u05b0\u05de\u05b5\u05d9", "\u05d4\u05b7\u05e1\u05bc\u05b5\u05da\u05b0", "\u05d1\u05b7\u05bc\u05ea", "\u05d5\u05b0\u05c4\u05d0\u05b7\u05c4\u05d4\u05b2\u05c4\u05e8\u05b9\u05c4\u05df\u05c4", "\u05d9\u05b9\u05e6\u05b5\u05d0", "\u05d1\u05bc\u05b4\u05de\u05b0\u05d7\u05b9\u05e7\u05b5\u05e7", "\u05dc\u05b4\u05e6\u05b0\u05e4\u05d5\u05b9\u05df", "\u05e0\u05b7\u05e4\u05b0\u05e9\u05c1\u05b8\u05bd\u05d4\u05bc", "\u05d1\u05bc\u05b0\u05d0\u05b4\u05d9\u05e9\u05c1\u05b8\u05d4\u05bc", "\u05d4\u05b4\u05e7\u05b0\u05e8\u05b4\u05d1", "\u05d0\u05b5\u05de\u05b6\u05e8", "\u05d4\u05b2\u05dc\u05b4\u05bd\u05d9\u05e0\u05b9\u05ea\u05b6\u05dd", "\u05de\u05d5\u05b9\u05e9\u05c1\u05b0\u05d1\u05b9\u05ea\u05b5\u05d9\u05db\u05b6\u05bd\u05dd", "\u05e8\u05b8\u05d1\u05b8\u05d4", "\u05d2\u05bc\u05b0\u05de\u05b7\u05dc\u05bc\u05b4\u05bd\u05d9", "\u05de\u05b7\u05d7\u05b0\u05dc\u05b4\u05d9", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05bd\u05ea\u05b0\u05d0\u05b7\u05d1\u05bc\u05b0\u05dc\u05d5\u05bc", "\u05d5\u05b0\u05d9\u05b4\u05e9\u05c2\u05b0\u05e8\u05b8\u05d0\u05b5\u05dc", "\u05d7\u05b8\u05dc\u05b8\u05d4", "\u05d9\u05b4\u05e8\u05b0\u05e6\u05b7\u05d7", "\u05e9\u05b8\u05c1\u05db\u05b7\u05df", "\u05d1\u05bc\u05b8\u05e2\u05b8\u05bd\u05e8\u05b6\u05d1", "\u05d4\u05b7\u05bd\u05d4\u05b7\u05e1\u05b0\u05db\u05bc\u05b5\u05df", "\u05d5\u05b7\u05d9\u05bc\u05b8\u05e7\u05b8\u05e5", "\u05e6\u05b4\u05e0\u05b8\u05d4", "\u05e9\u05b6\u05c1\u05de\u05b6\u05e9\u05c1", "\u05dc\u05b7\u05e4\u05bc\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05d1\u05bc\u05b0\u05e6\u05bb\u05e8\u05d5\u05b9\u05ea", "\u05de\u05b7\u05dc\u05b0\u05db\u05bb\u05ea\u05bd\u05d5\u05b9", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e1\u05b0\u05e4\u05bc\u05b9\u05e7", "\u05d5\u05b0\u05d4\u05b6\u05bd\u05e2\u05b1\u05de\u05b4\u05d3\u05b8\u05d4\u05bc", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05e4\u05b0\u05e9\u05c1\u05b5\u05d8", "\u05e0\u05b5\u05d7\u05b8\u05dc\u05b5\u05e5", "\u05e2\u05b8\u05db\u05b0\u05e8\u05b8\u05bd\u05df", "\u05d4\u05b8\u05e2\u05b5\u05e8\u05b8\u05e0\u05b4\u05bd\u05d9", "\u05d1\u05bc\u05b0\u05de\u05b7\u05d9\u05b4\u05dd", "\u05d1\u05bc\u05b8\u05d0\u05b4\u05d9\u05dd", "\u05d5\u05b0\u05d2\u05b6\u05e4\u05b6\u05df", "\u05d5\u05bc\u05d1\u05b5\u05e8\u05b5\u05da\u05b0", "\u05e4\u05bc\u05b9\u05e7\u05b5\u05d3", "\u05ea\u05bc\u05b0\u05e8\u05d5\u05bc\u05de\u05b7\u05ea\u05b0\u05db\u05b6\u05dd", "\u05e7\u05b4\u05d1\u05b0\u05e8\u05d5\u05b9\u05ea", "\u05d0\u05b6\u05e9\u05c1\u05b0\u05db\u05bc\u05d5\u05b9\u05dc", "\u05d5\u05b7\u05d4\u05b2\u05d1\u05b4\u05bd\u05d9\u05d0\u05b9\u05ea\u05b4\u05d9\u05d5", "\u05d5\u05b0\u05dc\u05b7\u05db\u05bc\u05b0\u05d1\u05b8\u05e9\u05c2\u05b4\u05d9\u05dd", "\u05e2\u05b5\u05d9\u05e0\u05b8\u05bd\u05df", "\u05d4\u05b5\u05e0\u05b8\u05bc\u05d4", "\u05e6\u05b4\u05dc\u05bc\u05b8\u05dd", "\u05d7\u05b9\u05d3\u05b6\u05e9\u05c1", "\u05de\u05b0\u05d0\u05b9\u05d3", "\u05e2\u05d5\u05bc\u05e8", "\u05d6\u05b4\u05db\u05b0\u05e8\u05d5\u05b9\u05df", "\u05e1\u05d5\u05bc\u05e8", "\u05e8\u05b0\u05d1\u05b4\u05d9\u05e2\u05b4\u05d9", "\u05d5\u05b0\u05e9\u05c1\u05b8\u05bd\u05e4\u05b0\u05d8\u05d5\u05bc", "\u05d1\u05b6\u05bc\u05d2\u05b6\u05d3", "\u05d5\u05b0\u05d8\u05b4\u05bd\u05d4\u05b7\u05e8\u05b0\u05ea\u05bc\u05b8", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05b8\u05d0\u05d5\u05bc\u05dc\u05b4\u05bd\u05d9", "\u05dc\u05b4\u05d2\u05b0\u05d1\u05bb\u05dc\u05b9\u05ea\u05b6\u05bd\u05d9\u05d4\u05b8", "\u05ea\u05b0\u05d8\u05b7\u05de\u05bc\u05b5\u05d0", "\u05e4\u05b4\u05bc\u05ea\u05b0\u05d0\u05d5\u05b9\u05dd", "\u05d5\u05b0\u05d7\u05b4\u05d8\u05bc\u05b0\u05d0\u05d5\u05b9", "\u05e9\u05b0\u05c1\u05de\u05b9\u05e0\u05b4\u05d9\u05dd", "\u05e4\u05bc\u05b8\u05e7\u05b7\u05d3", "\u05e0\u05b4\u05d8\u05b0\u05de\u05b8\u05d0\u05b8\u05d4", "\u05d9\u05b6\u05e7\u05b6\u05d1", "\u05e9\u05c1\u05b0\u05e4\u05b8\u05bd\u05de\u05b8\u05d4", "\u05d4\u05b4\u05d9\u05df", "\u05d1\u05bc\u05bd\u05b7\u05d7\u05b2\u05e6\u05b9\u05e6\u05b0\u05e8\u05d5\u05b9\u05ea", "\u05d4\u05b8\u05e8\u05b4\u05de\u05bc\u05b9\u05e0\u05b4\u05d9\u05dd", "\u05db\u05b0\u05e7\u05b9\u05e8\u05b7\u05d7", "\u05d9\u05b7\u05e2\u05b2\u05e7\u05b9\u05d1", "\u05db\u05bc\u05b0\u05d4\u05bb\u05e0\u05bc\u05b8\u05ea\u05b8\u05dd", "\u05d0\u05b8\u05d1\u05b7\u05d3", "\u05de\u05b0\u05dc\u05b9\u05d0", "\u05e7\u05d5\u05b9\u05dc", "\u05e9\u05c1\u05b9\u05e4\u05b0\u05db\u05bd\u05d5\u05b9", "\u05e4\u05bc\u05b0\u05e7\u05bb\u05d3\u05b8\u05d9\u05d5", "\u05d5\u05b0\u05d0\u05b8\u05bd\u05de\u05b0\u05e8\u05d5\u05bc", "\u05d1\u05bc\u05b0\u05dc\u05b4\u05d1\u05b0\u05e0\u05b8\u05bd\u05d4", "\u05d5\u05b0\u05d0\u05d5\u05b9\u05e0\u05b8\u05df", "\u05e2\u05b8\u05d1\u05b7\u05d3", "\u05de\u05b5\u05e8\u05b7\u05e2\u05b0\u05de\u05b0\u05e1\u05b5\u05e1", "\u05de\u05b4\u05de\u05bc\u05b7\u05d8\u05bc\u05b6\u05d4", "\u05db\u05bc\u05b0\u05d4\u05bb\u05e0\u05bc\u05b7\u05ea\u05b0\u05db\u05b6\u05dd", "\u05d4\u05b6\u05e2\u05b8\u05e8\u05b4\u05bd\u05d9\u05dd", "\u05d5\u05b7\u05d9\u05b0\u05db\u05b7\u05e4\u05bc\u05b5\u05e8", "\u05dc\u05b7\u05e9\u05c2\u05bc\u05b6\u05d4", "\u05de\u05b5\u05d4\u05bd\u05d5\u05b9\u05d3\u05b0\u05da\u05b8", "\u05d4\u05b8\u05d0\u05b5\u05e9\u05c1", "\u05d0\u05b6\u05e1\u05b0\u05e4\u05b8\u05d4", "\u05d4\u05b7\u05e7\u05bc\u05b8\u05e8\u05b5\u05d1", "\u05ea\u05b4\u05e9\u05c1\u05b0\u05d2\u05bc\u05d5\u05bc", "\u05dc\u05b4\u05e0\u05b0\u05e4\u05bc\u05b9\u05dc", "\u05dc\u05b5\u05d1", "\u05d5\u05b0\u05e0\u05b8\u05e1\u05b7\u05e2", "\u05d4\u05b7\u05e4\u05bc\u05b0\u05e2\u05d5\u05b9\u05e8", "\u05d0\u05b6\u05dc\u05b0\u05e2\u05b8\u05d6\u05b8\u05e8", "\u05e9\u05c2\u05b0\u05e8\u05b5\u05e4\u05b7\u05ea", "\u05e2\u05b7\u05de\u05b0\u05e8\u05b8\u05dd", "\u05d4\u05b7\u05de\u05bc\u05b4\u05d3\u05b0\u05d9\u05b8\u05e0\u05b4\u05d9\u05ea", "\u05d7\u05b5\u05dc\u05bd\u05d5\u05b9\u05df", "\u05d5\u05b7\u05d9\u05bc\u05d5\u05b9\u05e1\u05b6\u05e3", "\u05e4\u05bc\u05b0\u05ea\u05b4\u05d9\u05dc", "\u05ea\u05bc\u05b7\u05e2\u05b2\u05d1\u05b4\u05d9\u05e8\u05d5\u05bc", "\u05d5\u05b0\u05e9\u05c1\u05b4\u05de\u05b0\u05e2\u05b4\u05bd\u05d9", "\u05d1\u05b0\u05bc\u05db\u05d5\u05b9\u05e8", "\u05d0\u05b5\u05d9\u05dc\u05b4\u05dd", "\u05d5\u05b0\u05d4\u05b5\u05de\u05b7\u05ea\u05bc\u05b8\u05d4", "\u05de\u05b0\u05e8\u05b4\u05d9\u05ea\u05b6\u05dd", "\u05e7\u05b4\u05d9\u05e8", "\u05d1\u05b8\u05dc\u05b8\u05e7", "\u05e2\u05b7\u05dc", "\u05e2\u05b8\u05de\u05b8\u05dc", "\u05ea\u05b4\u05e9\u05c2\u05b0\u05d0\u05d5\u05bc", "\u05d9\u05b9\u05d3\u05b7\u05e2\u05b7\u05ea", "\u05de\u05b5\u05e8\u05b6\u05d7\u05b6\u05dd", "\u05d1\u05bc\u05b7\u05d7\u05b2\u05e6\u05b5\u05e8\u05bd\u05d5\u05b9\u05ea", "\u05e8\u05b5\u05d9\u05d7\u05b7", "\u05de\u05b4\u05bd\u05e7\u05b0\u05e8\u05b8\u05d0", "\u05d5\u05b0\u05d9\u05b7\u05e2\u05b0\u05d6\u05b5\u05e8", "\u05ea\u05bc\u05b4\u05ea\u05b0\u05e0\u05b7\u05d7\u05b2\u05dc\u05d5\u05bc", "\u05d4\u05b7\u05e7\u05bc\u05b0\u05e2\u05b8\u05e8\u05b8\u05d4", "\u05de\u05b4\u05e7\u05b0\u05e0\u05b5\u05e0\u05d5\u05bc", "\u05d7\u05b8\u05e0\u05d5\u05bc", "\u05e4\u05bc\u05b6\u05e8\u05b7\u05e2", "\u05de\u05b4\u05dc\u05bc\u05b6\u05d7\u05b6\u05dd", "\u05d5\u05b7\u05d9\u05bc\u05b5\u05d0\u05b8\u05e1\u05b5\u05e3", "\u05dc\u05b0\u05d7\u05b7\u05d2\u05bc\u05b4\u05d9", "\u05d1\u05bc\u05b8\u05d4\u05b5\u05e0\u05bc\u05b8\u05d4", "\u05d1\u05bc\u05b0\u05de\u05d5\u05b9\u05ea", "\u05d1\u05b7\u05d3\u05bc\u05b8\u05bd\u05d9\u05d5", "\u05d5\u05b0\u05d4\u05b4\u05d1\u05bc\u05b4\u05d9\u05d8", "\u05d0\u05b8\u05de\u05b5\u05bd\u05df", "\u05d5\u05b0\u05ea\u05b4\u05e0\u05bc\u05b7\u05e9\u05c2\u05bc\u05b5\u05d0", "\u05d1\u05bc\u05b4\u05db\u05b0\u05dc\u05b4\u05d9", "\u05d6\u05b6\u05e8\u05b7\u05e2", "\u05e7\u05c7\u05e8\u05b0\u05d1\u05b8\u05bc\u05df", "\u05de\u05b0\u05d1\u05b8\u05e8\u05b2\u05db\u05b6\u05d9\u05da\u05b8", "\u05dc\u05b7\u05d9\u05b0\u05dc\u05b8\u05d4", "\u05dc\u05b0\u05e9\u05c1\u05d5\u05bc\u05ea\u05b6\u05dc\u05b7\u05d7", "\u05d8\u05d5\u05b9\u05d1", "\u05d4\u05b7\u05ea\u05bc\u05d5\u05b9\u05e8\u05b8\u05d4", "\u05d0\u05b1\u05dc\u05b4\u05d9\u05e9\u05c1\u05b8\u05de\u05b8\u05e2", "\u05dc\u05b7\u05e2\u05b2\u05d1\u05b9\u05d3\u05b8\u05d4", "\u05e4\u05bc\u05b4\u05d9\u05e0\u05b0\u05d7\u05b8\u05e1", "\u05de\u05b5\u05e2\u05b6\u05e6\u05b0\u05d9\u05d5\u05b9\u05df", "\u05ea\u05bc\u05b4\u05bd\u05ea\u05b0\u05e0\u05b7\u05e9\u05c2\u05bc\u05b0\u05d0\u05d5\u05bc", "\u05db\u05bc\u05b0\u05e8\u05b8\u05bd\u05d2\u05b7\u05e2", "\u05d7\u05b7\u05dc\u05bc\u05b8\u05d4", "\u05d9\u05b7\u05e2\u05b2\u05e7\u05b8\u05bd\u05df", "\u05d5\u05b0\u05d4\u05b7\u05d7\u05d5\u05b9\u05e0\u05b4\u05dd", "\u05e9\u05c1\u05b0\u05dc\u05b7\u05d7\u05b0\u05ea\u05bc\u05b8\u05e0\u05d5\u05bc", "\u05dc\u05b4\u05e9\u05c1\u05b0\u05d2\u05b8\u05d2\u05b8\u05d4", "\u05ea\u05b0\u05d1\u05b8\u05e8\u05b2\u05db\u05b6\u05bd\u05e0\u05bc\u05d5\u05bc", "\u05d4\u05b8\u05bd\u05e2\u05b2\u05d1\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05d1\u05bc\u05b0\u05e8\u05b4\u05d9\u05ea\u05b4\u05d9", "\u05e9\u05b0\u05c1\u05d1\u05d5\u05bc\u05e2\u05b8\u05d4", "\u05d3\u05b4\u05bc\u05d9\u05df", "\u05de\u05b5\u05bd\u05e2\u05b7\u05d1\u05b0\u05e8\u05b9\u05e0\u05b8\u05d4", "\u05ea\u05b4\u05e1\u05bc\u05b9\u05d1", "\u05de\u05b4\u05df", "\u05e4\u05bc\u05b8\u05e0\u05b8\u05bd\u05d9\u05d5", "\u05d8\u05bb\u05de\u05b0\u05d0\u05b8\u05d4", "\u05d1\u05b7\u05de\u05bc\u05b7\u05d9\u05b4\u05dd", "\u05d5\u05bc\u05e9\u05c1\u05b0\u05de\u05b4\u05d9\u05d3\u05b8\u05e2", "\u05dc\u05bc\u05b8\u05e0\u05d5\u05bc", "\u05d4\u05b7\u05d7\u05b2\u05de\u05b4\u05e9\u05c1\u05bc\u05b4\u05d9\u05dd", "\u05db\u05bc\u05b0\u05d1\u05b9\u05d3\u05b4\u05d9", "\u05e4\u05bc\u05b0\u05e2\u05b8\u05de\u05b4\u05bd\u05d9\u05dd", "\u05d5\u05b0\u05d4\u05b4\u05e9\u05c1\u05b0\u05e7\u05b8\u05d4\u05bc", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05ea\u05b0\u05d9\u05b7\u05e6\u05bc\u05b5\u05d1", "\u05d5\u05bc\u05de\u05b4\u05e0\u05b0\u05d7\u05b8\u05ea\u05b8\u05d4\u05bc", "\u05d5\u05b0\u05e9\u05c1\u05b4\u05e9\u05c1\u05bc\u05b4\u05bd\u05d9\u05dd", "\u05d1\u05b8\u05bc\u05d4\u05bc", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05e7\u05b0\u05e8\u05b4\u05d9\u05d1\u05d5\u05bc", "\u05e9\u05c1\u05b0\u05dc\u05b8\u05e9\u05c1\u05b0\u05ea\u05bc\u05b8\u05bd\u05dd", "\u05d4\u05b8\u05e8\u05b9\u05d0\u05b4\u05d9\u05dd", "\u05e7\u05b6\u05dc\u05b7\u05e2", "\u05d5\u05b0\u05d4\u05b4\u05e0\u05bc\u05b5\u05d4", "\u05d9\u05b0\u05e0\u05b7\u05d0\u05b2\u05e6\u05bb\u05e0\u05b4\u05d9", "\u05e6\u05d5\u05bc\u05e2\u05b8\u05bd\u05e8", "\u05d4\u05b7\u05de\u05bc\u05b0\u05e8\u05b8\u05e8\u05b4\u05bd\u05d9", "\u05e7\u05b4\u05e8\u05b0\u05d9\u05b7\u05ea", "\u05dc\u05b4\u05e9\u05c1\u05b0\u05de\u05d5\u05b9\u05ea", "\u05d1\u05b4\u05d9", "\u05d5\u05bc\u05de\u05b4\u05e0\u05b0\u05d7\u05b8\u05ea\u05d5\u05b9", "\u05d1\u05bc\u05b0\u05d0\u05b9\u05d1\u05b9\u05bd\u05ea", "\u05dc\u05b0\u05e6\u05b4\u05d1\u05b0\u05d0\u05b9\u05ea\u05b8\u05bd\u05dd", "\u05d1\u05b0\u05e9\u05c2\u05b8\u05e8\u05bd\u05d5\u05b9", "\u05d5\u05b0\u05e2\u05b8\u05dc\u05b4\u05d9\u05e0\u05d5\u05bc", "\u05e2\u05b7\u05ea\u05bc\u05d5\u05bc\u05d3", "\u05db\u05bc\u05b7\u05d7\u05b2\u05d6\u05b5\u05d4", "\u05e9\u05b6\u05c1\u05e4\u05b6\u05d8", "\u05d5\u05b7\u05ea\u05bc\u05b4\u05e7\u05b0\u05e8\u05b6\u05d0\u05df\u05b8", "\u05d1\u05b8\u05bc\u05db\u05b8\u05d4", "\u05de\u05b4\u05e6\u05bc\u05b0\u05d1\u05b8\u05d0", "\u05e0\u05b0\u05ea\u05b9\u05df", "\u05d1\u05bc\u05b7\u05d7\u05b2\u05e8\u05b8\u05d3\u05b8\u05bd\u05d4", "\u05dc\u05b0\u05e2\u05b9\u05dc\u05b9\u05bd\u05ea\u05b5\u05d9\u05db\u05b6\u05dd", "\u05d1\u05bc\u05bd\u05b7\u05de\u05bc\u05b7\u05d7\u05b2\u05e0\u05b6\u05bd\u05d4", "\u05d1\u05bc\u05b7\u05e2\u05b6\u05e6\u05b6\u05dd", "\u05e6\u05b4\u05d9\u05e5", "\u05ea\u05d5\u05b9\u05dc\u05b8\u05e2\u05b7\u05ea", "\u05d1\u05b9\u05bc\u05e7\u05b6\u05e8", "\u05e2\u05b5\u05d1\u05b6\u05e8", "\u05e6\u05b0\u05dc\u05c7\u05e4\u05b0\u05d7\u05b8\u05d3", "\u05d0\u05b1\u05dc\u05b4\u05d9\u05d0\u05b8\u05bd\u05d1", "\u05d1\u05bc\u05b4\u05e0\u05b0\u05e2\u05bb\u05e8\u05b6\u05d9\u05d4\u05b8", "\u05de\u05b7\u05d8\u05bc\u05d5\u05b9\u05ea\u05b8\u05bd\u05dd", "\u05e9\u05c2\u05b8\u05e8\u05b5\u05bd\u05d9", "\u05d5\u05b0\u05e0\u05b4\u05e9\u05c1\u05b0\u05e7\u05b8\u05e4\u05b8\u05d4", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05b4\u05de\u05b0\u05e2\u05b9\u05e0\u05b4\u05d9", "\u05ea\u05bc\u05b4\u05d8\u05b0\u05de\u05b8\u05d0", "\u05d1\u05bc\u05b7\u05de\u05bc\u05b7\u05e7\u05bc\u05b5\u05bd\u05dc", "\u05de\u05b5\u05d0\u05b8\u05d3\u05b8\u05dd", "\u05de\u05b7\u05d7\u05b0\u05ea\u05bc\u05b8\u05ea\u05bd\u05d5\u05b9", "\u05d5\u05b0\u05d9\u05b8\u05e4\u05bb\u05e6\u05d5\u05bc", "\u05d1\u05bc\u05b0\u05de\u05b9\u05e2\u05b2\u05d3\u05d5\u05b9", "\u05d5\u05b0\u05d0\u05b8\u05e6\u05b7\u05dc\u05b0\u05ea\u05bc\u05b4\u05d9", "\u05de\u05d5\u05b9\u05e6\u05b4\u05d0\u05b5\u05d9", "\u05d0\u05b8\u05bd\u05e8\u05b8\u05d4", "\u05dc\u05b0\u05d8\u05b7\u05e4\u05bc\u05b0\u05db\u05b6\u05dd", "\u05d9\u05b6\u05bd\u05d7\u05b1\u05d6\u05b6\u05d4", "\u05de\u05b8\u05e9\u05b7\u05c1\u05d7", "\u05d0\u05b2\u05e9\u05c1\u05b4\u05d9\u05d1\u05b6\u05bd\u05e0\u05bc\u05b8\u05d4", "\u05e9\u05b5\u05c1\u05e9\u05c1", "\u05e6\u05bb\u05d5\u05bc\u05b8\u05bd\u05d4", "\u05de\u05b4\u05e9\u05b0\u05c1\u05e4\u05b8\u05bc\u05d7\u05b8\u05d4", "\u05e0\u05b8\u05e1\u05b8\u05e2\u05d5\u05bc", "\u05de\u05b0\u05e2\u05b7\u05d8", "\u05d5\u05b0\u05d4\u05b4\u05e9\u05c1\u05b0\u05dc\u05b4\u05d9\u05da\u05b0", "\u05ea\u05bc\u05b8\u05e8\u05d5\u05bc", "\u05d4\u05b7\u05d1\u05bc\u05b0\u05e6\u05b8\u05dc\u05b4\u05d9\u05dd", "\u05d5\u05b7\u05d9\u05b0\u05db\u05b7\u05d4\u05b5\u05df", "\u05d4\u05bd\u05b7\u05d7\u05b6\u05d1\u05b0\u05e8\u05b9\u05e0\u05b4\u05d9", "\u05d7\u05b8\u05d3\u05b8\u05e9\u05c1", "\u05d9\u05b0\u05e8\u05b5\u05db\u05b5\u05da\u05b0", "\u05dc\u05b0\u05de\u05b4\u05e7\u05b0\u05dc\u05b8\u05d8", "\u05d4\u05b7\u05e4\u05bc\u05b0\u05e7\u05bb\u05d3\u05b4\u05bd\u05d9\u05dd", "\u05d0\u05b6\u05bd\u05e2\u05b1\u05d1\u05b9\u05bd\u05e8\u05b8\u05d4", "\u05e9\u05b5\u05c1\u05db\u05b8\u05e8", "\u05d4\u05b7\u05db\u05bc\u05bb\u05e9\u05c1\u05b4\u05d9\u05ea", "\u05d5\u05bc\u05d1\u05b8\u05d0\u05d5\u05bc", "\u05db\u05bc\u05b0\u05dc\u05b4\u05d9\u05dc", "\u05e0\u05b7\u05e4\u05b0\u05ea\u05b8\u05bc\u05dc\u05b4\u05d9", "\u05e7\u05b8\u05bd\u05d1\u05b0\u05e8\u05d5\u05bc", "\u05d4\u05b8\u05e2\u05b2\u05de\u05b8\u05dc\u05b5\u05e7\u05b4\u05d9", "\u05d9\u05b8\u05e9\u05b8\u05c1\u05e8", "\u05e4\u05b0\u05bc\u05e2\u05d5\u05b9\u05e8", "\u05de\u05b5\u05d0\u05b7\u05dc\u05b0\u05e4\u05b5\u05d9", "\u05de\u05b8\u05d4", "\u05d1\u05b7\u05d1\u05bc\u05b9\u05e7\u05b6\u05e8", "\u05e4\u05bc\u05b0\u05dc\u05b5\u05d9\u05d8\u05b4\u05dd", "\u05ea\u05bc\u05b8\u05d0\u05b9\u05e8", "\u05dc\u05b7\u05d9\u05b4\u05dc", "\u05d4\u05b7\u05db\u05bc\u05b7\u05e8\u05b0\u05de\u05b4\u05bd\u05d9", "\u05de\u05b5\u05e2\u05b8\u05e8\u05b8\u05d9\u05d5", "\u05d4\u05b4\u05e7\u05b0\u05e8\u05b4\u05d9\u05d1\u05d5\u05bc", "\u05d4\u05b5\u05d1\u05b4\u05d9\u05d0\u05d5\u05bc", "\u05de\u05b0\u05e0\u05b7\u05e9\u05b6\u05bc\u05c1\u05d4", "\u05ea\u05d5\u05b9\u05e8\u05b4\u05d9\u05e9\u05c1\u05d5\u05bc", "\u05d0\u05b4\u05ea\u05bc\u05b8\u05e0\u05d5\u05bc", "\u05d9\u05b6\u05ea\u05b6\u05e8", "\u05d5\u05b0\u05e9\u05c1\u05b4\u05e9\u05c1\u05bc\u05b4\u05d9\u05dd", "\u05d5\u05b0\u05de\u05b7\u05d8\u05bc\u05b5\u05d4", "\u05d5\u05bc\u05d1\u05b4\u05dc\u05b0\u05e2\u05b8\u05dd", "\u05d5\u05b7\u05d9\u05bc\u05b8\u05e9\u05c1\u05b6\u05ea", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05b4\u05dc\u05bc\u05b5\u05de\u05b4\u05bd\u05d9", "\u05e4\u05b9\u05bc\u05d4", "\u05d9\u05b8\u05e6\u05b8\u05d0", "\u05e8\u05b8\u05e9\u05b8\u05c1\u05e2", "\u05d5\u05bc\u05e4\u05b0\u05e7\u05bb\u05d3\u05bc\u05b7\u05ea", "\u05e6\u05d5\u05bc\u05e8", "\u05dc\u05b0\u05d9\u05b4\u05de\u05b0\u05e0\u05b8\u05d4", "\u05d1\u05bc\u05b0\u05e9\u05c1\u05b5\u05de\u05bd\u05d5\u05b9\u05ea", "\u05e0\u05b4\u05ea\u05b0\u05e4\u05bc\u05b8\u05bd\u05e9\u05c2\u05b8\u05d4", "\u05d5\u05b8\u05db\u05b8\u05e8\u05b6\u05dd", "\u05d5\u05b7\u05d9\u05bc\u05b9\u05e1\u05b6\u05e3", "\u05d5\u05bc\u05de\u05b7\u05e8\u05b0\u05d0\u05b6\u05d4", "\u05ea\u05bc\u05b7\u05e9\u05c1\u05b0\u05de\u05b4\u05bd\u05d9\u05d3\u05d5\u05bc", "\u05d5\u05bc\u05dc\u05b0\u05e9\u05c1\u05b7\u05dc\u05b0\u05de\u05b5\u05d9\u05db\u05b6\u05bd\u05dd", "\u05d5\u05b0\u05d4\u05b4\u05e6\u05bc\u05b4\u05d9\u05dc\u05d5\u05bc", "\u05db\u05b9\u05bc\u05dc", "\u05dc\u05b4\u05d2\u05b0\u05d1\u05bb\u05dc\u05b9\u05ea\u05b6\u05d9\u05d4\u05b8", "\u05d5\u05b7\u05d9\u05bc\u05b5\u05d0\u05b8\u05db\u05b5\u05dc", "\u05d9\u05d5\u05b9\u05e9\u05c1\u05b0\u05d1\u05b6\u05d9\u05d4\u05b8", "\u05db\u05bc\u05b8\u05db\u05b6\u05dd", "\u05e4\u05bc\u05b0\u05d3\u05b8\u05d4", "\u05d9\u05b8\u05e8\u05b7\u05d3", "\u05ea\u05b0\u05d7\u05b7\u05dc\u05bc\u05b0\u05dc\u05d5\u05bc", "\u05d6\u05b0\u05de\u05d5\u05b9\u05e8\u05b8\u05d4", "\u05d0\u05b4\u05d9\u05e9\u05c1", "\u05d1\u05bc\u05b8\u05d6\u05b0\u05d6\u05d5\u05bc", "\u05d5\u05b0\u05d9\u05b4\u05bd\u05d4\u05b0\u05d9\u05d5\u05bc", "\u05d1\u05b8\u05bc\u05e7\u05b7\u05e2", "\u05dc\u05b0\u05de\u05b7\u05e1\u05b0\u05e2\u05b5\u05d9\u05d4\u05b6\u05bd\u05dd", "\u05de\u05b5\u05bd\u05d0\u05b5\u05ea\u05b8\u05dd", "\u05d9\u05b8\u05e0\u05b4\u05d9\u05d0", "\u05e9\u05c1\u05b0\u05d3\u05b5\u05d9\u05d0\u05bd\u05d5\u05bc\u05e8", "\u05ea\u05b4\u05db\u05bc\u05b8\u05dc\u05b5\u05dd", "\u05ea\u05b8\u05bc\u05d5\u05b6\u05da\u05b0", "\u05dc\u05b4\u05d1\u05b0\u05e0\u05b9\u05ea\u05b8\u05bd\u05d9\u05d5", "\u05e0\u05b4\u05d1\u05b0\u05e0\u05b0\u05ea\u05b8\u05d4", "\u05d1\u05b8\u05bc\u05e9\u05b8\u05c2\u05e8", "\u05e0\u05b9\u05ea\u05b5\u05df", "\u05d9\u05b4\u05d3\u05b0\u05d1\u05bc\u05b0\u05e7\u05d5\u05bc", "\u05d1\u05bc\u05b7\u05d7\u05b2\u05e6\u05b9\u05bd\u05e6\u05b0\u05e8\u05b9\u05ea", "\u05d2\u05b8\u05bc\u05d3\u05b7\u05dc", "\u05dc\u05b5\u05bd\u05d0\u05dc\u05b9\u05d4\u05b8\u05d9\u05d5", "\u05d7\u05d5\u05bc\u05e8", "\u05d4\u05b2\u05e8\u05b7\u05e7", "\u05d5\u05bc\u05e9\u05c1\u05b0\u05d1\u05b4\u05d9\u05db\u05b6\u05bd\u05dd", "\u05d9\u05d5\u05b9\u05e6\u05b4\u05d9\u05d0\u05b5\u05dd", "\u05d4\u05b4\u05db\u05bc\u05b8\u05d4", "\u05de\u05b4\u05e7\u05b0\u05dc\u05b8\u05d8\u05d5\u05b9", "\u05de\u05b4\u05d3\u05bc\u05b8\u05dc\u05b0\u05d9\u05b8\u05d5", "\u05d4\u05b8\u05e2\u05b9\u05de\u05b0\u05d3\u05b4\u05d9\u05dd", "\u05e8\u05b9\u05e6\u05b5\u05d7\u05b7", "\u05db\u05bc\u05b4\u05e1\u05b0\u05dc\u05bd\u05d5\u05b9\u05df", "\u05d9\u05b7\u05dd", "\u05de\u05b4\u05e9\u05c1\u05b0\u05db\u05bc\u05b0\u05e0\u05b9\u05ea\u05b6\u05d9\u05da\u05b8", "\u05db\u05bc\u05b8\u05d1\u05b5\u05d3", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05bd\u05e2\u05b7\u05d1\u05b0\u05e8\u05d5\u05bc", "\u05e9\u05c1\u05b0\u05e7\u05b8\u05dc\u05b4\u05d9\u05dd", "\u05e0\u05b6\u05d0\u05b1\u05e1\u05b7\u05e3", "\u05d4\u05b7\u05bd\u05ea\u05bc\u05b7\u05d7\u05b2\u05e0\u05b4\u05bd\u05d9", "\u05de\u05b7\u05d7\u05b2\u05e0\u05b6\u05d4", "\u05d1\u05bc\u05b0\u05de\u05b4\u05db\u05b0\u05e1\u05b5\u05d4", "\u05dc\u05b8\u05e9\u05c2\u05b5\u05d0\u05ea", "\u05e0\u05b7\u05e2\u05b2\u05d1\u05b9\u05e8", "\u05e2\u05b7\u05de\u05bc\u05b4\u05d9\u05e0\u05b8\u05d3\u05b8\u05bd\u05d1", "\u05dc\u05b0\u05de\u05b7\u05dc\u05b0\u05db\u05bc\u05b4\u05d9\u05d0\u05b5\u05dc", "\u05dc\u05b0\u05de\u05b7\u05d7\u05b2\u05e0\u05b5\u05d4", "\u05ea\u05bc\u05b5\u05e4\u05b6\u05df", "\u05ea\u05b4\u05bc\u05e8\u05b0\u05e6\u05b8\u05d4", "\u05d5\u05b7\u05d9\u05bc\u05b8\u05e6\u05b5\u05bd\u05e5", "\u05e0\u05b0\u05d1\u05b4\u05d9\u05d0\u05b4\u05d9\u05dd", "\u05d1\u05bc\u05b0\u05e0\u05b4\u05db\u05b0\u05dc\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05e4\u05b9\u05bd\u05d4", "\u05d1\u05b4\u05bc\u05dc\u05b0\u05ea\u05b4\u05bc\u05d9", "\u05d3\u05b7\u05bc\u05e2\u05b7\u05ea", "\u05d9\u05b4\u05e9\u05b0\u05c2\u05e8\u05b8\u05d0\u05b5\u05dc", "\u05e8\u05b4\u05e7\u05bc\u05bb\u05e2\u05b5\u05d9", "\u05db\u05bc\u05b0\u05d2\u05b7\u05e0\u05bc\u05b9\u05ea", "\u05d9\u05b8\u05e8\u05b9\u05e7", "\u05d7\u05b6\u05e9\u05c1\u05b0\u05d1\u05bc\u05d5\u05b9\u05df", "\u05dc\u05b0\u05d7\u05b5\u05dc\u05b6\u05e7", "\u05d4\u05b7\u05d1\u05bc\u05b7\u05dc\u05b0\u05e2\u05b4\u05d9", "\u05d0\u05b8\u05d6\u05b7\u05df", "\u05d4\u05b7\u05bd\u05e0\u05bc\u05b7\u05e2\u05b2\u05de\u05b4\u05bd\u05d9", "\u05dc\u05d5\u05bc\u05d0", "\u05db\u05bc\u05b7\u05de\u05bc\u05b7\u05e8\u05b0\u05d0\u05b6\u05d4", "\u05ea\u05b7\u05bc\u05d7\u05b7\u05e9\u05c1", "\u05d4\u05b7\u05bd\u05de\u05bc\u05b7\u05d7\u05b2\u05e0\u05bd\u05d5\u05b9\u05ea", "\u05e4\u05bc\u05b6\u05dc\u05b6\u05ea", "\u05d4\u05b2\u05dc\u05b9\u05da\u05b0", "\u05d5\u05b7\u05bd\u05d9\u05bc\u05b4\u05dc\u05b0\u05db\u05bc\u05b0\u05d3\u05bb\u05d4\u05b8", "\u05d0\u05b8\u05d7\u05b4\u05d9\u05d5", "\u05e7\u05b8\u05d8\u05b7\u05e8", "\u05d1\u05bc\u05b4\u05e8\u05b0\u05d1\u05b4\u05e2\u05b4\u05d9\u05ea", "\u05e6\u05b0\u05d3\u05b4\u05d9\u05bc\u05b8\u05bd\u05d4", "\u05db\u05bc\u05b0\u05e8\u05b8\u05d2\u05b7\u05e2", "\u05dc\u05b0\u05e0\u05b8\u05e9\u05c2\u05b4\u05d9\u05d0", "\u05d5\u05b0\u05db\u05b9\u05bd\u05d4", "\u05d5\u05bc\u05db\u05b0\u05e0\u05b4\u05e1\u05b0\u05db\u05bc\u05d5\u05b9", "\u05d5\u05b0\u05e2\u05b5\u05d3", "\u05ea\u05bc\u05b4\u05d2\u05bc\u05b0\u05e2\u05d5\u05bc", "\u05db\u05bc\u05b4\u05e0\u05bc\u05b6\u05e8\u05b6\u05ea", "\u05d1\u05bc\u05b7\u05d7\u05b2\u05e6\u05b9\u05e6\u05b0\u05e8\u05d5\u05b9\u05ea", "\u05de\u05b0\u05d0\u05b7\u05ea", "\u05db\u05bc\u05b6\u05e9\u05c2\u05b6\u05d1", "\u05d4\u05b7\u05e7\u05b0\u05e8\u05b4\u05d1", "\u05e2\u05b4\u05e9\u05b8\u05bc\u05c2\u05e8\u05d5\u05b9\u05df", "\u05de\u05b7\u05d8\u05bc\u05b5\u05bd\u05d4", "\u05e4\u05b8\u05bc\u05d3\u05b8\u05d4", "\u05ea\u05bc\u05d5\u05b9\u05e8", "\u05d0\u05b7\u05e4\u05bc\u05b7\u05d9\u05b4\u05dd", "\u05dc\u05b8\u05e8\u05b7\u05d1", "\u05e7\u05b8\u05e8\u05b0\u05d1\u05bc\u05b8\u05e0\u05b8\u05d4\u05bc", "\u05d7\u05b7\u05d9\u05b4\u05dc", "\u05d7\u05b6\u05dc\u05b0\u05d1\u05bc\u05b8\u05dd", "\u05d4\u05b7\u05d1\u05bc\u05b8\u05d6", "\u05d3\u05bc\u05b9\u05d1\u05b0\u05e8\u05b9\u05ea", "\u05e9\u05c1\u05b0\u05dc\u05b4\u05e9\u05c1\u05b4\u05d9\u05ea", "\u05de\u05b4\u05d6\u05bc\u05b8\u05db\u05b8\u05e8", "\u05d4\u05d5\u05bc\u05d0", "\u05e0\u05b8\u05d4\u05b8\u05e8", "\u05d4\u05b8\u05d0\u05b8\u05ea\u05d5\u05b9\u05df", "\u05d5\u05bc\u05d1\u05b0\u05db\u05b5\u05dc\u05b8\u05bd\u05d9\u05d5", "\u05d1\u05bc\u05b6\u05bd\u05d7\u05b8\u05dc\u05b8\u05dc", "\u05d4\u05b8\u05d0\u05b5\u05dc\u05b9\u05e0\u05b4\u05d9", "\u05d5\u05b0\u05db\u05b8\u05ea\u05b7\u05d1", "\u05d4\u05b2\u05de\u05b4\u05ea\u05bc\u05b6\u05dd", "\u05d4\u05d5\u05b9\u05e8\u05b4\u05d9\u05e9\u05c1\u05d5\u05b9", "\u05e6\u05d5\u05bc\u05e8\u05b4\u05bd\u05d9", "\u05e9\u05b8\u05c2\u05d3\u05b6\u05d4", "\u05d6\u05d5\u05bc\u05d1", "\u05d5\u05b0\u05d4\u05b7\u05db\u05bc\u05b0\u05e0\u05b7\u05e2\u05b2\u05e0\u05b4\u05d9", "\u05e8\u05b8\u05de\u05b8\u05d4", "\u05de\u05b4\u05e9\u05b0\u05c1\u05db\u05b8\u05bc\u05df", "\u05d5\u05b7\u05d0\u05b2\u05d3\u05b8\u05e0\u05b8\u05d9\u05d5", "\u05d9\u05b8\u05bd\u05e7\u05b6\u05d1", "\u05e0\u05b5\u05d3\u05b6\u05e8", "\u05d4\u05b8\u05bd\u05d0\u05b2\u05d1\u05b7\u05d8\u05bc\u05b4\u05d7\u05b4\u05d9\u05dd", "\u05d4\u05b7\u05e6\u05bc\u05b7\u05e8", "\u05d4\u05bd\u05b8\u05d0\u05b2\u05dc\u05b8\u05e4\u05b4\u05d9\u05dd", "\u05d5\u05bc\u05de\u05b4\u05e0\u05b0\u05d7\u05b7\u05ea", "\u05d0\u05bd\u05b8\u05d4\u05b3\u05dc\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05e7\u05b7\u05bd\u05e2\u05b2\u05e8\u05b7\u05ea", "\u05db\u05bc\u05b7\u05d1\u05bc\u05b0\u05d3\u05b6\u05bd\u05da\u05b8", "\u05d3\u05bc\u05b8\u05ea\u05b8\u05df", "\u05d9\u05bd\u05b6\u05d7\u05b1\u05d6\u05b6\u05d4", "\u05e1\u05b8\u05d1\u05b4\u05d9\u05d1", "\u05e4\u05b0\u05e7\u05bb\u05d3\u05b8\u05d9\u05d5", "\u05d7\u05b6\u05d8\u05b0\u05d0\u05d5\u05b9", "\u05d5\u05b0\u05d4\u05b4\u05d1\u05b0\u05d3\u05bc\u05b7\u05dc\u05b0\u05ea\u05bc\u05b8", "\u05d5\u05bd\u05b6\u05d0\u05b1\u05e1\u05b8\u05e8\u05b6\u05d9\u05d4\u05b8", "\u05d5\u05bc\u05e0\u05b0\u05e9\u05c2\u05b4\u05d9\u05d0\u05b5\u05d9", "\u05d5\u05bc\u05db\u05b0\u05d0\u05b7\u05de\u05bc\u05b8\u05ea\u05b7\u05d9\u05b4\u05dd", "\u05d5\u05b0\u05d4\u05d5\u05b9\u05dc\u05b5\u05da\u05b0", "\u05e8\u05b0\u05e2\u05d5\u05bc\u05d0\u05b5\u05dc", "\u05d0\u05b8\u05d1\u05b4\u05bd\u05d9\u05d5", "\u05de\u05b4\u05d6\u05b0\u05e8\u05b8\u05d7", "\u05d9\u05b7\u05e0\u05b0\u05d7\u05b5\u05e0\u05b4\u05d9", "\u05d9\u05b0\u05e7\u05b4\u05d9\u05de\u05b6\u05bd\u05e0\u05bc\u05b8\u05d4", "\u05d0\u05b7\u05dc\u05b0\u05de\u05b8\u05e0\u05b8\u05d4", "\u05dc\u05b0\u05d9\u05b7\u05d7\u05b0\u05dc\u05b0\u05d0\u05b5\u05dc", "\u05d7\u05b5\u05e4\u05b6\u05e8", "\u05e7\u05b6\u05de\u05b7\u05d7", "\u05de\u05b4\u05d2\u05bc\u05b0\u05d1\u05d5\u05bc\u05dc", "\u05d4\u05b2\u05d1\u05b4\u05d9\u05d0\u05b9\u05ea\u05b8\u05e0\u05d5\u05bc", "\u05d4\u05b7\u05e0\u05bc\u05b5\u05e1", "\u05d0\u05b5\u05e4\u05b9\u05bd\u05d3", "\u05db\u05bc\u05b9\u05d4", "\u05e2\u05b4\u05de\u05bc\u05b8\u05bd\u05da\u05b0", "\u05e0\u05b0\u05d1\u05b4\u05d9\u05d0\u05b2\u05db\u05b6\u05dd", "\u05d9\u05b8\u05de\u05b4\u05d9\u05ea", "\u05e6\u05b7\u05d5", "\u05d0\u05b2\u05d3\u05bb\u05de\u05bc\u05b8\u05d4", "\u05d5\u05b7\u05ea\u05bc\u05b4\u05e9\u05c1\u05b0\u05e7\u05b7\u05e2", "\u05d6\u05b4\u05de\u05b0\u05e8\u05b4\u05d9", "\u05d9\u05b8\u05d3\u05b8\u05dd", "\u05de\u05b4\u05e4\u05bc\u05b0\u05e8\u05b4\u05d9", "\u05de\u05b5\u05d7\u05b7\u05e8\u05b0\u05e6\u05b7\u05e0\u05bc\u05b4\u05d9\u05dd", "\u05e8\u05b8\u05d6\u05b8\u05d4", "\u05d4\u05b8\u05d0\u05b2\u05d3\u05b8\u05de\u05b8\u05bd\u05d4", "\u05d5\u05b7\u05bd\u05d9\u05bc\u05b7\u05d7\u05b2\u05e0\u05d5\u05bc", "\u05de\u05b0\u05dc\u05b8\u05d0\u05db\u05b8\u05d4", "\u05db\u05bc\u05b7\u05d0\u05b2\u05e8\u05b8\u05d6\u05b4\u05d9\u05dd", "\u05ea\u05bc\u05b7\u05e8\u05b0\u05d1\u05bc\u05d5\u05bc\u05ea", "\u05d1\u05bc\u05b8\u05ea\u05bc\u05b5\u05d9\u05e0\u05d5\u05bc", "\u05de\u05b8\u05e2\u05b7\u05d8", "\u05e2\u05b9\u05bd\u05dc", "\u05d1\u05b8\u05bc\u05dd", "\u05d0\u05b8\u05d1\u05b4\u05bd\u05d9\u05e0\u05d5\u05bc", "\u05e2\u05b9\u05dc\u05b8\u05ea\u05bd\u05d5\u05b9", "\u05d4\u05b2\u05d9\u05b4\u05e7\u05b0\u05e8\u05b0\u05da\u05b8", "\u05ea\u05b4\u05e0\u05b0\u05d7\u05b8\u05dc", "\u05de\u05b7\u05dc\u05b0\u05d0\u05b8\u05db\u05b4\u05d9\u05dd", "\u05e2\u05b4\u05de\u05bc\u05b4\u05bd\u05d9", "\u05d1\u05bc\u05b4\u05d9", "\u05d1\u05bc\u05b0\u05de\u05b4\u05ea\u05b0\u05e7\u05b8\u05bd\u05d4", "\u05e6\u05b4\u05d5\u05bc\u05b8\u05d4", "\u05db\u05bc\u05b0\u05d1\u05b7\u05dc\u05bc\u05b7\u05e2", "\u05e0\u05b4\u05db\u05bc\u05b0\u05dc\u05d5\u05bc", "\u05d7\u05b8\u05d3\u05b0\u05e9\u05c1\u05b5\u05d9\u05db\u05b6\u05dd", "\u05e1\u05b6\u05dc\u05b7\u05e2", "\u05de\u05b4\u05e6\u05b0\u05d5\u05ba\u05ea\u05b8\u05d9", "\u05e7\u05b8\u05d3\u05b7\u05e9\u05c1", "\u05d1\u05bc\u05b9\u05d0\u05b8\u05bd\u05dd", "\u05d4\u05b7\u05d1\u05bc\u05b0\u05d3\u05b4\u05d9\u05dc", "\u05d4\u05b7\u05d9\u05b0\u05e9\u05c1\u05b4\u05de\u05b9\u05ea", "\u05de\u05b4\u05e7\u05bc\u05b4\u05e8\u05b0\u05d1\u05bc\u05bd\u05d5\u05b9", "\u05d8\u05b7\u05e4\u05bc\u05b8\u05dd", "\u05e0\u05b4\u05e0\u05b0\u05d7\u05b7\u05dc", "\u05d1\u05bc\u05b4\u05e8\u05b0\u05e4\u05b4\u05d9\u05d3\u05b4\u05dd", "\u05d0\u05b8\u05e1\u05b0\u05e8\u05b8\u05d4", "\u05de\u05b4\u05dc\u05b0\u05d0\u05d5\u05bc", "\u05d4\u05b6\u05d0\u05b8\u05e0\u05b9\u05db\u05b4\u05d9", "\u05d5\u05b0\u05e0\u05b4\u05e1\u05b0\u05db\u05bc\u05b8\u05bd\u05d4\u05bc", "\u05d5\u05b0\u05d2\u05b4\u05d3\u05b0\u05e8\u05b9\u05ea", "\u05e0\u05d5\u05b9\u05ea\u05b7\u05e8", "\u05d4\u05b7\u05de\u05bc\u05d5\u05bc\u05e9\u05c1\u05b4\u05d9", "\u05d5\u05b0\u05e6\u05b4\u05d5\u05bc\u05b4\u05d9\u05ea\u05b8\u05d4", "\u05d1\u05bc\u05b0\u05de\u05d5\u05b9\u05e2\u05b2\u05d3\u05bd\u05d5\u05b9", "\u05d4\u05b7\u05dc\u05b0\u05d5\u05b4\u05d9\u05bc\u05b4\u05bd\u05dd", "\u05e7\u05b8\u05d4\u05b7\u05dc", "\u05db\u05bc\u05b0\u05dc\u05b8\u05d0\u05b5\u05bd\u05dd", "\u05d5\u05b7\u05e2\u05b2\u05d1\u05b9\u05d3\u05b8\u05d4", "\u05d1\u05b0\u05d7\u05b6\u05d8\u05b0\u05d0\u05d5\u05b9", "\u05e0\u05b0\u05d7\u05b7\u05e9\u05c1", "\u05d4\u05b7\u05de\u05b0\u05d0\u05b8\u05bd\u05e8\u05b2\u05e8\u05b4\u05d9\u05dd", "\u05e6\u05b8\u05e8\u05d5\u05bc\u05e2\u05b7", "\u05d5\u05bc\u05d1\u05b8\u05e2\u05b6\u05e8\u05b6\u05d1", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05b0\u05d1\u05b4\u05d9", "\u05d9\u05b5\u05e9\u05c1\u05b5\u05d1", "\u05d9\u05b4\u05e9\u05c1\u05b0\u05db\u05bc\u05b9\u05df", "\u05de\u05b4\u05dc\u05b0\u05d7\u05b2\u05de\u05b9\u05ea", "\u05d5\u05b0\u05d9\u05b8\u05e1\u05b7\u05e3", "\u05d3\u05b9\u05d3\u05b5\u05d9\u05d4\u05b6\u05df", "\u05d9\u05b7\u05d7\u05b2\u05e0\u05bd\u05d5\u05bc", "\u05d1\u05bc\u05b8\u05de\u05d5\u05b9\u05ea", "\u05d7\u05b6\u05e9\u05b0\u05c1\u05d1\u05bc\u05d5\u05b9\u05df", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05d9\u05e8\u05b7\u05e9\u05c1", "\u05e7\u05b4\u05e0\u05b0\u05d0\u05b8\u05d4", "\u05d5\u05bc\u05de\u05b5\u05bd\u05d9\u05ea\u05b0\u05e8\u05b5\u05d9\u05d4\u05b6\u05bd\u05dd", "\u05d2\u05bc\u05d5\u05bc\u05e8", "\u05db\u05b8\u05bc\u05ea\u05b7\u05d1", "\u05ea\u05bc\u05b4\u05e4\u05b0\u05e7\u05b0\u05d3\u05b5\u05dd", "\u05e4\u05b4\u05d2\u05b0\u05e8\u05b5\u05d9\u05db\u05b6\u05dd", "\u05de\u05b7\u05d7\u05b2\u05d6\u05b5\u05d4", "\u05d4\u05b7\u05d1\u05bc\u05b7\u05db\u05b0\u05e8\u05b4\u05d9", "\u05d1\u05bc\u05b7\u05d9\u05d4\u05d5\u05b8\u05d4", "\u05dc\u05b7\u05d7\u05b0\u05de\u05b5\u05e0\u05d5\u05bc", "\u05de\u05b7\u05ea\u05bc\u05b0\u05e0\u05b9\u05ea\u05b5\u05d9\u05db\u05b6\u05dd", "\u05d9\u05b4\u05d2\u05bc\u05b8\u05e8\u05b5\u05bd\u05e2\u05b7", "\u05e7\u05d5\u05bc\u05de\u05b8\u05d4", "\u05e4\u05b5\u05bc\u05d0\u05b8\u05d4", "\u05e0\u05b6\u05d6\u05b6\u05e8", "\u05d5\u05b8\u05d7\u05d5\u05bc\u05e6\u05b8\u05d4", "\u05ea\u05bc\u05b7\u05e7\u05b0\u05d8\u05b4\u05d9\u05e8", "\u05d0\u05b9\u05bd\u05d9\u05b0\u05d1\u05b7\u05d9", "\u05dc\u05b7\u05e0\u05bc\u05b6\u05e1\u05b6\u05da\u05b0", "\u05de\u05b5\u05bd\u05d9\u05ea\u05b8\u05e8\u05b8\u05d9\u05d5", "\u05d4\u05b7\u05d8\u05bc\u05b7\u05dc", "\u05d2\u05bc\u05b7\u05de\u05b0\u05dc\u05b4\u05d9\u05d0\u05b5\u05dc", "\u05d1\u05bc\u05d5\u05b9", "\u05d5\u05b0\u05dc\u05b8\u05bd\u05e7\u05b0\u05d8\u05d5\u05bc", "\u05d9\u05b7\u05e9\u05c1\u05b0\u05e7\u05b6\u05d4", "\u05e9\u05c1\u05b0\u05d1\u05bb\u05e2\u05b8\u05d4", "\u05dc\u05b0\u05e9\u05c1\u05b4\u05de\u05b0\u05e2\u05d5\u05b9\u05df", "\u05db\u05b8\u05dc\u05b5\u05d1", "\u05d5\u05b7\u05bd\u05d9\u05bc\u05b4\u05e7\u05b0\u05e8\u05b0\u05d1\u05d5\u05bc", "\u05d5\u05b0\u05d4\u05b4\u05e9\u05c1\u05b0\u05d1\u05bc\u05b4\u05d9\u05e2\u05b7", "\u05d4\u05b9\u05e8", "\u05e6\u05b0\u05d3\u05b8\u05d3", "\u05d1\u05b7\u05db\u05bc\u05b0\u05d1\u05b8\u05e9\u05c2\u05b4\u05d9\u05dd", "\u05e6\u05d5\u05bc\u05e8\u05b4\u05bd\u05d9\u05e9\u05c1\u05b7\u05d3\u05bc\u05b8\u05bd\u05d9", "\u05e2\u05b4\u05e9\u05c2\u05bc\u05b8\u05e8\u05d5\u05b9\u05df", "\u05d2\u05bc\u05b0\u05d1\u05d5\u05bc\u05dc", "\u05e2\u05b7\u05e7\u05b0\u05e8\u05b8\u05d1", "\u05d4\u05b7\u05e7\u05bc\u05b8\u05e8\u05b9\u05d1", "\u05d5\u05b0\u05ea\u05b8\u05e7\u05b0\u05e2\u05d5\u05bc", "\u05dc\u05b0\u05d0\u05b8\u05d7\u05b4\u05d9\u05d5", "\u05dc\u05b7\u05d0\u05b2\u05d7\u05bb\u05d6\u05bc\u05b8\u05d4", "\u05d5\u05b0\u05d4\u05b7\u05de\u05bc\u05b5\u05d0\u05d5\u05b9\u05ea", "\u05d4\u05b7\u05d0\u05b7\u05bd\u05d7\u05b5\u05d9\u05db\u05b6\u05dd", "\u05e1\u05b8\u05dc\u05b7\u05d7\u05b0\u05ea\u05bc\u05b4\u05d9", "\u05d9\u05b4\u05d2\u05bc\u05b7\u05e2", "\u05d9\u05b4\u05e0\u05b0\u05d7\u05b2\u05dc\u05d5\u05bc", "\u05d5\u05b0\u05db\u05b4\u05d9", "\u05d4\u05b7\u05bd\u05de\u05b0\u05d0\u05b8\u05e8\u05b2\u05e8\u05b4\u05d9\u05dd", "\u05dc\u05b7\u05de\u05bc\u05b4\u05dc\u05b0\u05d7\u05b8\u05de\u05b8\u05d4", "\u05db\u05bc\u05b9\u05dc", "\u05d9\u05b0\u05d4\u05d5\u05bc\u05d3\u05b8\u05bd\u05d4", "\u05de\u05b5\u05bd\u05d9\u05d3\u05b0\u05d1\u05b8\u05bd\u05d0", "\u05d5\u05b0\u05e0\u05b7\u05e4\u05b0\u05e9\u05c1\u05b5\u05e0\u05d5\u05bc", "\u05d4\u05b7\u05bd\u05d7\u05b6\u05dc\u05b0\u05e7\u05b4\u05bd\u05d9", "\u05d4\u05b7\u05e0\u05bc\u05b6\u05e4\u05b6\u05e9\u05c1", "\u05d5\u05b0\u05e0\u05b4\u05db\u05b0\u05d1\u05bc\u05b8\u05d3\u05b4\u05d9\u05dd", "\u05d9\u05b7\u05d7\u05b2\u05e0\u05b4\u05d9\u05e3", "\u05db\u05bc\u05b6\u05bd\u05d1\u05b6\u05e9\u05c2", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05d5\u05bc\u05d7\u05b8\u05de\u05b4\u05d9", "\u05dc\u05b0\u05ea\u05b4\u05ea\u05bc\u05b4\u05d9", "\u05d5\u05b0\u05d4\u05b7\u05bd\u05de\u05bc\u05b4\u05d6\u05b0\u05d1\u05bc\u05b0\u05d7\u05b9\u05ea", "\u05d5\u05b0\u05d0\u05b7\u05d3\u05b0\u05e0\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05dc\u05b4\u05de\u05b0\u05e1\u05b8\u05e8", "\u05d5\u05b0\u05e0\u05b6\u05e2\u05b0\u05dc\u05b7\u05dd", "\u05de\u05b4\u05e7\u05bc\u05b4\u05d1\u05b0\u05e8\u05b9\u05ea", "\u05d5\u05b7\u05d9\u05bc\u05b5\u05e2\u05b8\u05dc\u05d5\u05bc", "\u05e0\u05b8\u05d1\u05b7\u05d8", "\u05d9\u05b8\u05e2\u05b7\u05d3", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e7\u05bc\u05b8\u05d3\u05b5\u05e9\u05c1", "\u05e6\u05b8\u05d5\u05b8\u05d4", "\u05e2\u05d5\u05b9\u05d2", "\u05e8\u05b9\u05d1\u05b7\u05e2", "\u05e7\u05b0\u05d4\u05b7\u05dc", "\u05de\u05b5\u05bd\u05e2\u05b2\u05d1\u05b9\u05e8", "\u05d7\u05b8\u05e0\u05b8\u05d4", "\u05e2\u05b5\u05bd\u05e5", "\u05d0\u05b1\u05e1\u05b8\u05e8\u05b6\u05d9\u05d4\u05b8", "\u05d5\u05b0\u05d7\u05b5\u05dc\u05b6\u05e7", "\u05d1\u05bc\u05b0\u05d4\u05b7\u05e7\u05b0\u05e8\u05b4\u05d9\u05d1\u05b0\u05db\u05b6\u05dd", "\u05d1\u05bc\u05b0\u05d2\u05b4\u05e9\u05c1\u05b0\u05ea\u05bc\u05b8\u05dd", "\u05d4\u05b8\u05dc\u05b0\u05d0\u05b8\u05d4", "\u05de\u05b5\u05d7\u05b2\u05e6\u05b7\u05e8", "\u05d5\u05b7\u05d7\u05b2\u05de\u05b4\u05d9\u05e9\u05c1\u05b4\u05ea\u05d5\u05b9", "\u05e2\u05b8\u05d5\u05ba\u05df", "\u05d1\u05bc\u05b0\u05e1\u05d5\u05bc\u05e4\u05b8\u05d4", "\u05d5\u05b0\u05e0\u05b4\u05e9\u05c1\u05b0\u05e2\u05b7\u05df", "\u05de\u05b4\u05dc\u05bc\u05b4\u05d1\u05bc\u05b4\u05bd\u05d9", "\u05d1\u05bc\u05b0\u05e2\u05b7\u05bd\u05e8\u05b0\u05d1\u05b9\u05ea", "\u05d4\u05b8\u05bd\u05d0\u05b6\u05d7\u05b8\u05bd\u05d3", "\u05d0\u05b2\u05d7\u05b4\u05d9\u05e2\u05b6\u05d6\u05b6\u05e8", "\u05dc\u05b0\u05d4\u05b7\u05e7\u05b0\u05e8\u05b4\u05d9\u05d1", "\u05e2\u05b2\u05d1\u05b9\u05d3\u05b8\u05ea\u05b8\u05bd\u05dd", "\u05d5\u05bc\u05e4\u05b4\u05d2\u05b0\u05e8\u05b5\u05d9\u05db\u05b6\u05dd", "\u05d0\u05bd\u05b8\u05e1\u05b0\u05e8\u05b8\u05d4", "\u05d5\u05b0\u05d4\u05b6\u05e2\u05b1\u05d1\u05b4\u05d9\u05e8\u05d5\u05bc", "\u05d2\u05bc\u05b8\u05d5\u05b7\u05e2\u05b0\u05e0\u05d5\u05bc", "\u05d1\u05bc\u05b7\u05e1\u05bc\u05b6\u05dc\u05b7\u05e2", "\u05d0\u05b5\u05ea\u05b8\u05dd", "\u05db\u05b7\u05bc\u05e3", "\u05d1\u05bc\u05b0\u05e0\u05b7\u05d7\u05b2\u05dc\u05b8\u05ea\u05d5\u05b9", "\u05e9\u05c1\u05b8\u05d8\u05d5\u05b9\u05d7\u05b7", "\u05de\u05b0\u05e0\u05b8\u05e2\u05b2\u05da\u05b8", "\u05d5\u05b0\u05e9\u05c1\u05b5\u05bd\u05e9\u05c1\u05b6\u05ea", "\u05d3\u05bc\u05b4\u05d1\u05b0\u05dc\u05b8\u05ea\u05b8\u05d9\u05b0\u05de\u05b8\u05d4", "\u05d5\u05b0\u05e0\u05b4\u05e1\u05b0\u05ea\u05bc\u05b0\u05e8\u05b8\u05d4", "\u05d5\u05b0\u05e0\u05b8\u05e1\u05b8\u05e2\u05d5\u05bc", "\u05d0\u05b8\u05d6", "\u05d9\u05b4\u05e9\u05c1\u05b0\u05ea\u05bc\u05b6\u05bd\u05d4", "\u05d7\u05b7\u05d9", "\u05dc\u05b7\u05d0\u05b2\u05d1\u05b9\u05ea\u05b8\u05dd", "\u05e0\u05b0\u05ea\u05b7\u05ea\u05bc\u05b4\u05d9\u05dd", "\u05d4\u05b7\u05bd\u05d7\u05b6\u05d1\u05b0\u05e8\u05b9\u05e0\u05b4\u05d9", "\u05d5\u05bc\u05d1\u05b5\u05df", "\u05d0\u05b9\u05d4\u05b8\u05dc\u05b6\u05d9\u05da\u05b8", "\u05d5\u05bc\u05e9\u05c1\u05b0\u05dc\u05b9\u05e9\u05c1", "\u05de\u05b7\u05e8\u05b0\u05d0\u05b8\u05d4", "\u05d9\u05b5\u05bd\u05d0\u05b8\u05de\u05b7\u05e8", "\u05de\u05b5\u05e8\u05b4\u05ea\u05b0\u05de\u05b8\u05d4", "\u05dc\u05b0\u05e7\u05b9\u05e8\u05b7\u05d7", "\u05d9\u05b4\u05bd\u05e1\u05b0\u05dc\u05b7\u05bd\u05d7", "\u05d4\u05b7\u05e0\u05bc\u05b0\u05e9\u05c2\u05b4\u05d9\u05d0\u05b4\u05dd", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05bd\u05e2\u05b2\u05e0\u05d5\u05bc", "\u05d6\u05b9\u05e8\u05b7\u05e7", "\u05dc\u05b0\u05d1\u05b4\u05ea\u05bc\u05d5\u05b9", "\u05dc\u05b8\u05e7\u05b8\u05d7\u05b0\u05ea\u05bc\u05b4\u05d9", "\u05dc\u05b0\u05d1\u05b8\u05e2\u05b5\u05bd\u05e8", "\u05d5\u05b0\u05d4\u05b7\u05de\u05bc\u05b8\u05e1\u05b8\u05da\u05b0", "\u05d1\u05bc\u05b0\u05db\u05b4\u05d9\u05ea\u05b6\u05dd", "\u05e9\u05c1\u05b9\u05de\u05b5\u05e2\u05b7", "\u05d4\u05b6\u05bd\u05e2\u05b1\u05dc\u05b4\u05d9\u05ea\u05b8\u05e0\u05d5\u05bc", "\u05d5\u05b7\u05d9\u05bc\u05b8\u05e7\u05bb\u05de\u05d5\u05bc", "\u05d2\u05b0\u05bc\u05d1\u05d5\u05bc\u05dc", "\u05d6\u05b4\u05e4\u05b0\u05e8\u05b9\u05e0\u05b8\u05d4", "\u05d1\u05bc\u05b7\u05e4\u05bc\u05b8\u05e8\u05d5\u05bc\u05e8", "\u05e2\u05b1\u05e0\u05d5\u05bc", "\u05d5\u05bc\u05d1\u05b0\u05e2\u05b7\u05de\u05bc\u05d5\u05bc\u05d3", "\u05d0\u05d5\u05bc\u05dc\u05b8\u05dd", "\u05d9\u05b4\u05e9\u05c2\u05b0\u05e8\u05b9\u05bd\u05e3", "\u05db\u05bc\u05b0\u05db\u05b8\u05dc", "\u05d4\u05b8\u05d9\u05d5\u05b9", "\u05e8\u05b8\u05e6\u05b7\u05d7", "\u05e9\u05b7\u05c1\u05e2\u05b7\u05e8", "\u05de\u05b4\u05d9\u05bc\u05b8\u05d8\u05b0\u05d1\u05b8\u05ea\u05b8\u05d4", "\u05d5\u05b0\u05d0\u05b9\u05e8\u05b0\u05e8\u05b6\u05d9\u05da\u05b8", "\u05de\u05b7\u05e2\u05b2\u05e9\u05c2\u05b5\u05d4", "\u05d5\u05b0\u05e2\u05b8\u05e9\u05c2\u05b8\u05bd\u05d4", "\u05d7\u05b7\u05d8\u05b8\u05bc\u05d0\u05b8\u05d4", "\u05d4\u05b4\u05db\u05bc\u05b4\u05d9\u05ea\u05b7\u05e0\u05b4\u05d9", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e9\u05c1\u05b0\u05ea\u05bc\u05b7\u05d7\u05d5\u05bc", "\u05e8\u05b9\u05e2\u05b4\u05d9\u05dd", "\u05d5\u05b0\u05e0\u05b5\u05d3\u05b6\u05e8", "\u05d0\u05b5\u05ea", "\u05d5\u05b0\u05e9\u05c1\u05b8\u05de\u05b7\u05e2", "\u05ea\u05b8\u05d0\u05b9\u05e8", "\u05d1\u05b7\u05de\u05bc\u05d5\u05b9\u05d8", "\u05d5\u05b7\u05d9\u05bc\u05b8\u05e7\u05d5\u05bc\u05de\u05d5\u05bc", "\u05dc\u05b0\u05d9\u05b8\u05db\u05b4\u05d9\u05df", "\u05ea\u05bc\u05b0\u05e9\u05c1\u05b7\u05dc\u05bc\u05b0\u05d7\u05d5\u05bc\u05dd", "\u05d9\u05bc\u05b7\u05e8\u05b0\u05d0\u05b5\u05e0\u05b4\u05d9", "\u05d0\u05b2\u05e8\u05b8\u05dd", "\u05d1\u05bc\u05b9\u05db\u05b6\u05d4", "\u05d5\u05bc\u05de\u05b4\u05e9\u05c1\u05b0\u05de\u05b7\u05e8\u05b0\u05ea\u05bc\u05b8\u05dd", "\u05e2\u05b8\u05dc\u05b6\u05d9\u05d4\u05b8", "\u05d5\u05b0\u05d0\u05b6\u05e9\u05c1\u05b0\u05de\u05b0\u05e2\u05b8\u05d4", "\u05d0\u05b6\u05e8\u05b0\u05d0\u05b6\u05d4", "\u05de\u05b5\u05d4\u05b9\u05e8", "\u05d5\u05d9\u05dc\u05d5\u05e0\u05d5", "\u05d2\u05b5\u05bc\u05e8", "\u05d0\u05b6\u05d6\u05b0\u05e2\u05b9\u05dd", "\u05e0\u05b0\u05e7\u05b8\u05de\u05b8\u05d4", "\u05d9\u05b4\u05e1\u05b0\u05de\u05b0\u05db\u05d5\u05bc", "\u05d9\u05b6\u05e8\u05b6\u05e7", "\u05de\u05b4\u05e7\u05b0\u05e0\u05b6\u05bd\u05d4", "\u05d0\u05b9\u05ea\u05b8\u05da\u05b0", "\u05dc\u05b0\u05e2\u05b8\u05e8\u05b6\u05d9\u05d4\u05b8", "\u05d1\u05bc\u05b0\u05de\u05b9\u05e9\u05c1\u05b6\u05d4", "\u05d5\u05bc\u05d1\u05b5\u05d0\u05dc\u05b9\u05d4\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05e7\u05b8\u05e6\u05b8\u05d4", "\u05db\u05bc\u05b7\u05e3", "\u05e9\u05c1\u05b5\u05e9\u05c1\u05b7\u05d9", "\u05db\u05b5\u05bc\u05df", "\u05db\u05bb\u05e9\u05c1\u05b4\u05d9\u05ea", "\u05d4\u05b7\u05de\u05bc\u05b7\u05d7\u05b0\u05dc\u05b4\u05d9", "\u05ea\u05b0\u05bc\u05db\u05b5\u05dc\u05b6\u05ea", "\u05d9\u05b8\u05dc\u05b7\u05da\u05b0", "\u05d1\u05bc\u05b4\u05d2\u05b0\u05d1\u05bb\u05dc\u05d5\u05b9", "\u05d1\u05bc\u05b7\u05d4\u05b2\u05e8\u05b4\u05bd\u05d9\u05de\u05b0\u05db\u05b6\u05dd", "\u05d9\u05d5\u05b9\u05d1\u05b5\u05dc", "\u05e9\u05c1\u05b8\u05de\u05b0\u05e2\u05d5\u05b9", "\u05e9\u05c1\u05b4\u05d1\u05b0\u05e2\u05b4\u05d9\u05dd", "\u05d4\u05b7\u05d6\u05bc\u05b0\u05e7\u05b5\u05e0\u05b4\u05d9\u05dd", "\u05dc\u05b0\u05e8\u05b5\u05d9\u05d7\u05b7", "\u05e0\u05b4\u05d8\u05bc\u05b6\u05d4", "\u05d1\u05bc\u05b9\u05e7\u05b6\u05e8", "\u05db\u05bc\u05b7\u05e4\u05bc\u05b6\u05d9\u05d4\u05b8", "\u05d5\u05b4\u05bd\u05d9\u05e9\u05c1\u05b8\u05e8\u05b0\u05ea\u05d5\u05bc\u05da\u05b8", "\u05d0\u05b2\u05e9\u05b6\u05c1\u05e8", "\u05d3\u05bc\u05b6\u05d2\u05b6\u05dc", "\u05d4\u05b7\u05d1\u05bc\u05b0\u05d4\u05b5\u05de\u05b8\u05d4", "\u05ea\u05bc\u05b0\u05d0\u05b7\u05d1\u05bc\u05b5\u05d3\u05d5\u05bc", "\u05d2\u05bc\u05b7\u05dd", "\u05dc\u05b4\u05e6\u05b0\u05d1\u05b9\u05d0", "\u05e2\u05b4\u05dd", "\u05ea\u05bc\u05b4\u05db\u05bc\u05b8\u05e8\u05b5\u05ea", "\u05d0\u05b5\u05d9\u05dc\u05b4\u05de\u05b8\u05d4", "\u05de\u05b4\u05e7\u05b0\u05e8\u05b8\u05d0", "\u05d0\u05b4\u05d9\u05ea\u05b8\u05de\u05b8\u05e8", "\u05e9\u05c1\u05b6\u05e7\u05b6\u05dc", "\u05d4\u05b5\u05dd", "\u05d1\u05b8\u05e8\u05b5\u05da\u05b0", "\u05ea\u05bc\u05b9\u05d0\u05db\u05b0\u05dc\u05d5\u05bc\u05df", "\u05de\u05b7\u05d7\u05b0\u05ea\u05bc\u05b9\u05ea", "\u05e0\u05b7\u05d7\u05b2\u05dc\u05b8\u05ea\u05b8\u05df", "\u05ea\u05bc\u05b4\u05e1\u05bc\u05b8\u05d2\u05b5\u05e8", "\u05d5\u05b7\u05d9\u05bc\u05d5\u05b9\u05e6\u05b4\u05d9\u05d0\u05d5\u05bc", "\u05de\u05b4\u05e4\u05bc\u05d5\u05bc\u05e0\u05b9\u05df", "\u05e9\u05c1\u05b5\u05de\u05b9\u05ea", "\u05d5\u05b0\u05d0\u05b6\u05bd\u05dc", "\u05e2\u05b2\u05dc\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05d5\u05b0\u05d4\u05b4\u05ea\u05b0\u05d7\u05b7\u05d6\u05bc\u05b7\u05e7\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05d4\u05b8\u05e8\u05d0\u05d5\u05bc\u05d1\u05b5\u05e0\u05b4\u05d9", "\u05de\u05b4\u05dc\u05bc\u05b5\u05d0", "\u05de\u05b0\u05e9\u05c1\u05b8\u05dc\u05d5\u05b9", "\u05d1\u05b7\u05bc\u05d6", "\u05dc\u05b7\u05d0\u05b2\u05e9\u05c1\u05b6\u05e8", "\u05dc\u05b0\u05e9\u05c1\u05d5\u05bc\u05d7\u05b8\u05dd", "\u05dc\u05b4\u05d9", "\u05dc\u05b4\u05e4\u05b0\u05e7\u05bb\u05d3\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05d0\u05b8\u05e0\u05b9\u05db\u05b4\u05d9", "\u05d5\u05b8\u05d4\u05b5\u05d1", "\u05db\u05b8\u05bc\u05d1\u05b7\u05d3", "\u05d5\u05b0\u05d9\u05b4\u05e6\u05b0\u05d4\u05b8\u05e8", "\u05d9\u05b0\u05e8\u05b5\u05d7\u05bd\u05d5\u05b9", "\u05e9\u05b4\u05c1\u05e9\u05b4\u05bc\u05c1\u05d9", "\u05d9\u05b4\u05e6\u05b9\u05e7", "\u05e0\u05b8\u05e7\u05b7\u05d1", "\u05d3\u05b8\u05bc\u05dd", "\u05d5\u05b0\u05d2\u05b8\u05d3\u05b5\u05e8", "\u05ea\u05bc\u05b4\u05e0\u05bc\u05b8\u05d2\u05b0\u05e4\u05d5\u05bc", "\u05de\u05b0\u05e7\u05d5\u05b9\u05de\u05b8\u05dd", "\u05d9\u05b8\u05dd", "\u05dc\u05b0\u05d8\u05b7\u05d4\u05b2\u05e8\u05b8\u05bd\u05dd", "\u05de\u05b5\u05d0\u05b5\u05d9\u05dc\u05b4\u05dd", "\u05d0\u05b6\u05e6\u05b0\u05e2\u05b8\u05d3\u05b8\u05d4", "\u05d4\u05b8\u05d0\u05b8\u05d6\u05b0\u05e0\u05b4\u05d9", "\u05e9\u05b8\u05c1\u05d2\u05b7\u05d2", "\u05d5\u05b0\u05d4\u05b4\u05d9\u05d0", "\u05d5\u05bc\u05e9\u05c1\u05b0\u05e0\u05b4\u05d9", "\u05e0\u05b4\u05e9\u05c1\u05b0\u05ea\u05bc\u05b6\u05d4", "\u05db\u05b4\u05e1\u05bc\u05b8\u05d4\u05d5\u05bc", "\u05e2\u05b9\u05dc\u05b9\u05ea\u05b5\u05d9\u05db\u05b6\u05dd", "\u05e4\u05bc\u05b7\u05e2\u05b2\u05de\u05b8\u05d9\u05b4\u05dd", "\u05dc\u05b0\u05e2\u05b9\u05dc\u05b7\u05ea", "\u05de\u05b0\u05d0\u05b9\u05bd\u05d3", "\u05d4\u05b4\u05d1\u05bc\u05b8\u05d3\u05b0\u05dc\u05d5\u05bc", "\u05d3\u05bc\u05b0\u05e2\u05d5\u05bc\u05d0\u05b5\u05bd\u05dc", "\u05d9\u05b0\u05d2\u05b8\u05e8\u05b5\u05dd", "\u05dc\u05b0\u05de\u05b8\u05e8\u05b4\u05bd\u05d9\u05dd", "\u05dc\u05b7\u05e4\u05bc\u05b8\u05e8\u05b9\u05db\u05b6\u05ea", "\u05d1\u05b8\u05bc\u05e0\u05b8\u05d4", "\u05e2\u05b7\u05bd\u05e8\u05b0\u05d1\u05b9\u05ea", "\u05e7\u05b8\u05d3\u05b5\u05e9\u05c1", "\u05d9\u05b7\u05d1\u05bc\u05b9\u05e7", "\u05d5\u05b0\u05e8\u05b8\u05d0\u05b8\u05d4", "\u05d4\u05b7\u05e6\u05bc\u05b0\u05e4\u05d5\u05b9\u05e0\u05b4\u05d9", "\u05e0\u05b7\u05e4\u05b0\u05ea\u05bc\u05b8\u05dc\u05b4\u05d9", "\u05d0\u05b6\u05e8\u05b6\u05d6", "\u05db\u05bc\u05b7\u05de\u05bc\u05b5\u05ea", "\u05d5\u05b0\u05d9\u05b8\u05e8\u05b7\u05e9\u05c1\u05b0\u05e0\u05d5\u05bc", "\u05e2\u05d5\u05b9\u05e8", "\u05dc\u05bc\u05b8\u05d4\u05b6\u05dd", "\u05dc\u05b4\u05e7\u05b0\u05e8\u05b8\u05d0\u05ea\u05b6\u05bd\u05da\u05b8", "\u05db\u05bc\u05b8\u05d6\u05b0\u05d1\u05bc\u05b4\u05d9", "\u05d5\u05b0\u05d0\u05b6\u05e4\u05b0\u05e8\u05b8\u05bd\u05d9\u05b4\u05dd", "\u05d9\u05b4\u05e9\u05c1\u05b0\u05de\u05b0\u05e2\u05d5\u05bc", "\u05d0\u05b7\u05ea\u05bc\u05b0", "\u05ea\u05bc\u05bd\u05d5\u05b9\u05e6\u05b0\u05d0\u05b9\u05ea\u05b8\u05d9\u05d5", "\u05de\u05b7\u05dc\u05b0\u05e7\u05d5\u05b9\u05d7\u05b7", "\u05d4\u05b4\u05ea\u05b0\u05e4\u05bc\u05b7\u05dc\u05bc\u05b5\u05dc", "\u05e0\u05b0\u05ea\u05d5\u05bc\u05e0\u05b4\u05dd", "\u05d1\u05bc\u05b7\u05d7\u05b2\u05e6\u05b5\u05e8\u05b9\u05bd\u05ea", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05d7\u05b2\u05e8\u05b5\u05dd", "\u05d3\u05bc\u05b9\u05d1\u05b5\u05bd\u05e8", "\u05dc\u05b4\u05d2\u05b0\u05d1\u05d5\u05bc\u05dc", "\u05e8\u05b8\u05db\u05b7\u05d1\u05b0\u05ea\u05bc\u05b8", "\u05e7\u05b8\u05d8\u05b8\u05df", "\u05ea\u05bc\u05b8\u05e9\u05c1\u05bb\u05d1\u05d5\u05bc", "\u05d5\u05b7\u05d7\u05b2\u05e6\u05b4\u05d9", "\u05e1\u05b8\u05dc\u05b7\u05d7", "\u05e1\u05b4\u05d9\u05d7\u05b9\u05df", "\u05d5\u05b0\u05e2\u05b8\u05d1\u05b7\u05d3", "\u05d0\u05b7\u05e8\u05b0\u05e0\u05d5\u05b9\u05df", "\u05de\u05b5\u05bd\u05d4\u05b7\u05e8", "\u05d0\u05b8\u05d7\u05d5\u05b9\u05ea", "\u05d1\u05b0\u05bc\u05d0\u05b5\u05e8", "\u05d4\u05b8\u05d0\u05b1\u05de\u05b9\u05e8\u05b4\u05bd\u05d9", "\u05dc\u05b5\u05d5\u05b4\u05d9\u05b4\u05bc\u05d9", "\u05de\u05b5\u05bd\u05d9\u05ea\u05b0\u05e8\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05e8\u05b5\u05d0\u05e9\u05c1\u05b4\u05d9\u05ea\u05b8\u05dd", "\u05e7\u05b3\u05d4\u05b8\u05ea\u05b4\u05d9", "\u05e2\u05b5\u05de\u05b6\u05e7", "\u05d5\u05b7\u05e2\u05b2\u05e9\u05c2\u05b4\u05d9\u05ea\u05b6\u05dd", "\u05ea\u05b7\u05e2\u05b7\u05e8", "\u05d5\u05b0\u05d7\u05b4\u05e6\u05bc\u05b8\u05d9\u05d5", "\u05d7\u05b2\u05de\u05b8\u05ea", "\u05d4\u05b7\u05bd\u05d7\u05b4\u05d9\u05e8\u05b9\u05ea", "\u05db\u05b8\u05bc\u05e8\u05b7\u05e2", "\u05d5\u05b0\u05e9\u05c1\u05b8\u05de\u05b0\u05e8\u05d5\u05bc", "\u05ea\u05bc\u05d5\u05b9\u05dc\u05b0\u05d3\u05b9\u05ea\u05b8\u05dd", "\u05ea\u05bc\u05b0\u05e0\u05d5\u05bc\u05e4\u05b8\u05bd\u05d4", "\u05e0\u05b8\u05ea\u05b9\u05df", "\u05dc\u05b0\u05d0\u05b7\u05e8\u05b0\u05d0\u05b5\u05dc\u05b4\u05d9", "\u05d5\u05b0\u05dc\u05b7\u05de\u05b0\u05e2\u05b7\u05d8", "\u05d5\u05b0\u05d0\u05b6\u05ea\u05bc\u05b0\u05e0\u05b8\u05d4", "\u05d5\u05b7\u05ea\u05bc\u05b5\u05bd\u05e2\u05b8\u05e6\u05b7\u05e8", "\u05dc\u05b0\u05e9\u05c1\u05d5\u05bc\u05e0\u05b4\u05d9", "\u05dc\u05b4\u05de\u05b0\u05e8\u05b8\u05e8\u05b4\u05d9", "\u05ea\u05bc\u05b7\u05e2\u05b2\u05e9\u05c2\u05bd\u05d5\u05bc", "\u05d4\u05b7\u05bd\u05de\u05bc\u05b7\u05d7\u05b2\u05e0\u05b9\u05ea", "\u05e9\u05c1\u05b0\u05dc\u05bb\u05bd\u05de\u05b4\u05d9\u05d0\u05b5\u05dc", "\u05e2\u05e0\u05d5", "\u05d7\u05b8\u05d6\u05b8\u05e7", "\u05e2\u05b2\u05e8\u05b4\u05e1\u05b9\u05ea\u05b5\u05db\u05b6\u05dd", "\u05d5\u05b0\u05e2\u05b9\u05dc\u05b7\u05ea", "\u05d4\u05b7\u05d2\u05bc\u05b5\u05e8", "\u05d6\u05b0\u05e8\u05b5\u05d4", "\u05d1\u05bc\u05b0\u05e1\u05bb\u05db\u05bc\u05b9\u05bd\u05ea", "\u05e9\u05c1\u05b5\u05de\u05bd\u05d5\u05b9\u05ea", "\u05de\u05b4\u05dc\u05bc\u05b4\u05e4\u05b0\u05e0\u05b5\u05d9", "\u05e0\u05b6\u05d2\u05b6\u05d3", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e1\u05bc\u05b8\u05bd\u05e2\u05d5\u05bc", "\u05ea\u05bc\u05b5\u05d0\u05b8\u05e1\u05b5\u05bd\u05e3", "\u05e4\u05b8\u05bc\u05d0\u05e8\u05b8\u05df", "\u05d5\u05b0\u05e9\u05c2\u05b8\u05e8\u05b7\u05e3", "\u05dc\u05b0\u05d2\u05b8\u05d3", "\u05d4\u05b7\u05e0\u05bc\u05b9\u05e2\u05b8\u05d3\u05b4\u05d9\u05dd", "\u05d9\u05b0\u05e4\u05b5\u05e8\u05b6\u05bd\u05e0\u05bc\u05d5\u05bc", "\u05e9\u05c2\u05b0\u05e2\u05b4\u05d9\u05e8", "\u05e0\u05d5\u05bc\u05df", "\u05e8\u05b7\u05e7", "\u05d4\u05b7\u05d9\u05bc\u05b8\u05db\u05b4\u05d9\u05e0\u05b4\u05bd\u05d9", "\u05d5\u05b0\u05e6\u05b8\u05e8\u05b2\u05e8\u05d5\u05bc", "\u05d1\u05bc\u05b4\u05e9\u05c1\u05b0\u05e0\u05b7\u05ea", "\u05e2\u05b8\u05dc\u05b6\u05d9\u05da\u05b8", "\u05dc\u05b0\u05de\u05b8\u05d7\u05b8\u05e8", "\u05d5\u05b0\u05d0\u05b4\u05d9\u05ea\u05b8\u05de\u05b8\u05e8", "\u05d5\u05bc\u05e9\u05c2\u05b0\u05de\u05b9\u05d0\u05d5\u05dc", "\u05de\u05b4\u05d7\u05d5\u05bc\u05e5", "\u05e2\u05b2\u05d1\u05b9\u05bd\u05d3\u05b8\u05ea\u05b8\u05dd", "\u05d4\u05b7\u05bd\u05d7\u05b7\u05d2\u05bc\u05b4\u05d9", "\u05e9\u05b8\u05c1\u05e8\u05b7\u05ea", "\u05d4\u05b7\u05de\u05bc\u05b5\u05d0\u05bd\u05d5\u05b9\u05ea", "\u05d1\u05bc\u05b0\u05ea\u05b5\u05ea", "\u05e8\u05b5\u05bd\u05d9\u05d7\u05b7", "\u05d1\u05bc\u05b4\u05de\u05b0\u05e8\u05b4\u05d9\u05d1\u05b7\u05ea", "\u05d1\u05bc\u05b0\u05bd\u05db\u05d5\u05b9\u05e8", "\u05db\u05b0\u05dc\u05b4\u05bd\u05d9", "\u05d1\u05bc\u05b4\u05e9\u05c1\u05b0\u05d2\u05b8\u05d2\u05b8\u05bd\u05d4", "\u05d4\u05b7\u05de\u05bc\u05bb\u05db\u05bc\u05b6\u05d4", "\u05d5\u05b0\u05d3\u05b4\u05d1\u05bc\u05b7\u05e8\u05b0\u05ea\u05bc\u05b4\u05d9", "\u05d1\u05bc\u05b0\u05de\u05b9\u05ea\u05b8\u05dd", "\u05d3\u05b8\u05db\u05d5\u05bc", "\u05d0\u05b1\u05de\u05b9\u05e8\u05b4\u05d9", "\u05e6\u05b8\u05de\u05b4\u05d9\u05d3", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05e4\u05bc\u05b5\u05dc", "\u05d1\u05bc\u05b7\u05e0\u05bc\u05b6\u05d2\u05b6\u05d1", "\u05de\u05b6\u05d7\u05b1\u05e6\u05b7\u05ea", "\u05d0\u05b6\u05d3\u05b0\u05e8\u05b6\u05e2\u05b4\u05d9", "\u05de\u05b0\u05d2\u05b7\u05d3\u05bc\u05b5\u05e3", "\u05e0\u05b8\u05e6\u05b7\u05d1", "\u05e0\u05b8\u05ea\u05b7\u05ea\u05bc\u05b4\u05bd\u05d9", "\u05d9\u05b8\u05e7\u05d5\u05bc\u05dd", "\u05d1\u05bc\u05b0\u05e0\u05b5\u05d4", "\u05d5\u05bc\u05e0\u05b0\u05d1\u05d5\u05b9", "\u05d7\u05b2\u05de\u05b4\u05e9\u05b4\u05bc\u05c1\u05d9\u05dd", "\u05d5\u05b7\u05ea\u05bc\u05b5\u05e2\u05b8\u05e6\u05b7\u05e8", "\u05de\u05b5\u05d0\u05b4\u05ea\u05bc\u05b0\u05db\u05b6\u05dd", "\u05db\u05b9\u05e4\u05b6\u05e8", "\u05de\u05b7\u05dc\u05bc\u05b4\u05d9\u05e0\u05b4\u05d9\u05dd", "\u05e4\u05b8\u05bc\u05dc\u05b7\u05dc", "\u05ea\u05b7\u05e2\u05b2\u05e9\u05c2\u05d5\u05bc\u05df", "\u05e0\u05b9\u05e1\u05b5\u05e2\u05b7", "\u05e9\u05b8\u05c2\u05e8\u05b4\u05d9\u05d3", "\u05e8\u05b5\u05d0\u05e9\u05c1\u05b4\u05d9\u05ea", "\u05d4\u05b7\u05e7\u05bc\u05b0\u05e2\u05b8\u05e8\u05b9\u05ea", "\u05d5\u05b0\u05e8\u05b7\u05d1", "\u05d4\u05dc\u05da", "\u05e0\u05b7\u05d7\u05b2\u05dc\u05b8\u05ea\u05bd\u05d5\u05b9", "\u05d1\u05bc\u05b4\u05d8\u05b0\u05e0\u05b5\u05da\u05b0", "\u05d4\u05b7\u05d2\u05bc\u05b4\u05dc\u05b0\u05e2\u05b8\u05d3\u05b4\u05bd\u05d9", "\u05dc\u05b0\u05de\u05b4\u05e0\u05b0\u05d7\u05b8\u05bd\u05d4", "\u05e2\u05b7\u05de\u05b0\u05e8\u05b8\u05bd\u05dd", "\u05d1\u05bc\u05b0\u05e4\u05b6\u05ea\u05b7\u05e2", "\u05ea\u05b4\u05de\u05bc\u05b8\u05e0\u05b7\u05e2", "\u05d5\u05b0\u05e2\u05b7\u05e6\u05b0\u05de\u05b9\u05ea\u05b5\u05d9\u05d4\u05b6\u05dd", "\u05d4\u05b7\u05d6\u05bc\u05b0\u05d1\u05d5\u05bc\u05dc\u05b9\u05e0\u05b4\u05d9", "\u05d7\u05b2\u05e0\u05bb\u05db\u05b8\u05bc\u05d0", "\u05de\u05b8\u05bd\u05ea\u05b0\u05e0\u05d5\u05bc", "\u05d0\u05b6\u05ea\u05b0\u05d5\u05b7\u05d3\u05bc\u05b8\u05e2", "\u05de\u05b4\u05d1\u05bc\u05b0\u05d7\u05bb\u05e8\u05b8\u05d9\u05d5", "\u05d5\u05b0\u05d4\u05b6\u05bd\u05e2\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05dc\u05b0\u05d0\u05b8\u05d6\u05b0\u05e0\u05b4\u05d9", "\u05d1\u05bc\u05b0\u05d2\u05d5\u05b9\u05e8\u05b8\u05dc", "\u05d5\u05bc\u05de\u05b4\u05e0\u05b0\u05d7\u05b8\u05ea\u05b8\u05dd", "\u05de\u05b0\u05e8\u05b8\u05e8\u05b4\u05d9", "\u05d7\u05b8\u05d8\u05b8\u05d0", "\u05d4\u05b7\u05ea\u05bc\u05b0\u05dc\u05b8\u05d0\u05b8\u05d4", "\u05d0\u05b7\u05ea\u05bc\u05b6\u05bd\u05dd", "\u05e0\u05b8\u05d0", "\u05dc\u05b7\u05d7\u05b4\u05d9\u05dd", "\u05d0\u05b8\u05e0\u05b8\u05d4", "\u05d1\u05b8\u05bd\u05d4\u05bc", "\u05d5\u05b0\u05d9\u05b8\u05d2\u05b0\u05d1\u05bc\u05b3\u05d4\u05b8\u05bd\u05d4", "\u05d5\u05b7\u05d9\u05bc\u05b8\u05d0\u05e6\u05b6\u05dc", "\u05e7\u05b0\u05de\u05d5\u05bc\u05d0\u05b5\u05dc", "\u05de\u05b4\u05d6\u05b0\u05d1\u05bc\u05b7\u05d7", "\u05db\u05b8\u05bc\u05e8\u05b7\u05ea", "\u05de\u05b5\u05d7\u05b2\u05e6\u05b5\u05e8\u05d5\u05b9\u05ea", "\u05d1\u05b8\u05e2\u05b4\u05d6\u05bc\u05b4\u05bd\u05d9\u05dd", "\u05d1\u05b0\u05de\u05b5\u05ea", "\u05d0\u05b7\u05d7\u05b2\u05e8\u05b4\u05d9\u05ea", "\u05d0\u05b2\u05d1\u05b9\u05ea\u05b8\u05bd\u05d9\u05d5", "\u05d1\u05b0\u05de\u05b4\u05d3\u05b0\u05d1\u05bc\u05b7\u05e8", "\u05dc\u05b0\u05d1\u05b7\u05d3\u05bc\u05b4\u05d9", "\u05d1\u05bc\u05b0\u05de\u05b4\u05e1\u05b0\u05e4\u05bc\u05b8\u05e8\u05b8\u05dd", "\u05d4\u05b7\u05d6\u05bc\u05b4\u05d9\u05e8\u05d5\u05b9", "\u05d1\u05bc\u05b0\u05e0\u05bd\u05d5\u05bc", "\u05d4\u05b7\u05d0\u05b4\u05dd", "\u05d5\u05bc\u05ea\u05b0\u05d0\u05b5\u05e0\u05b8\u05d4", "\u05dc\u05b4\u05e8\u05b0\u05d2\u05bc\u05d5\u05b9\u05dd", "\u05d4\u05b7\u05dc\u05bc\u05b4\u05d1\u05b0\u05e0\u05b4\u05d9", "\u05e4\u05b6\u05bc\u05e1\u05b7\u05d7", "\u05d5\u05b8\u05d7\u05b8\u05bd\u05d9", "\u05e2\u05b8\u05d6\u05b7\u05d1", "\u05d5\u05b0\u05db\u05b4\u05d1\u05bc\u05b0\u05e1\u05d5\u05bc", "\u05d4\u05b9\u05e8\u05b5\u05d2", "\u05e9\u05b8\u05c1\u05de\u05b7\u05e8", "\u05de\u05b0\u05e7\u05b9\u05e9\u05c1\u05b5\u05e9\u05c1", "\u05d5\u05bc\u05d1\u05b7\u05d2\u05bc\u05d5\u05b9\u05d9\u05b4\u05dd", "\u05e2\u05b6\u05e9\u05b0\u05c2\u05e8\u05b4\u05d9\u05dd", "\u05de\u05b0\u05e2\u05d5\u05b9\u05df", "\u05d4\u05b4\u05e6\u05bc\u05d5\u05bc", "\u05d4\u05bd\u05b7\u05de\u05bc\u05b7\u05d7\u05b2\u05e0\u05b6\u05d4", "\u05de\u05d5\u05bc\u05ea", "\u05d9\u05b7\u05e8\u05b0\u05d3\u05b5\u05bc\u05df", "\u05de\u05d0\u05d5\u05bc\u05dd", "\u05d1\u05bc\u05b0\u05e9\u05c1\u05b5\u05dc\u05b8\u05d4", "\u05e9\u05c2\u05b7\u05dc\u05b0\u05d5\u05b4\u05d9\u05dd", "\u05d9\u05b4\u05bd\u05d4\u05b0\u05d9\u05d5\u05bc", "\u05d1\u05b8\u05bc\u05dc\u05b7\u05dc", "\u05e7\u05b6\u05e6\u05b6\u05e3", "\u05d5\u05b0\u05e0\u05b4\u05e1\u05b0\u05db\u05bc\u05b5\u05d9\u05d4\u05b6\u05bd\u05dd", "\u05e4\u05bc\u05b4\u05e8\u05b0\u05d7\u05b8\u05d4\u05bc", "\u05d5\u05b8\u05d1\u05b8\u05da\u05b0", "\u05db\u05b4\u05e1\u05bc\u05b8\u05d4", "\u05dc\u05b4\u05e9\u05c1\u05b0\u05dc\u05b8\u05de\u05b4\u05bd\u05d9\u05dd", "\u05d1\u05bc\u05b0\u05d9\u05b7\u05e2\u05b2\u05e7\u05b9\u05d1", "\u05d5\u05bc\u05d1\u05b0\u05e8\u05b6\u05d3\u05b6\u05ea", "\u05d5\u05bc\u05d1\u05b5\u05d9\u05ea", "\u05dc\u05b8\u05e7\u05b8\u05bd\u05d7", "\u05c0", "\u05d1\u05b9\u05db\u05b4\u05d9\u05dd", "\u05d5\u05b0\u05e9\u05c1\u05b9\u05d8\u05b0\u05e8\u05b8\u05d9\u05d5", "\u05d4\u05b2\u05e6\u05b9\u05d0\u05df", "\u05d4\u05b7\u05e4\u05bc\u05b7\u05e8\u05b0\u05e6\u05b4\u05d9", "\u05d0\u05b7\u05e8\u05b0\u05d1\u05b8\u05bc\u05e2\u05b4\u05d9\u05dd", "\u05de\u05b6\u05db\u05b6\u05e1", "\u05ea\u05bc\u05b4\u05ea\u05b0\u05e0\u05b6\u05d7\u05b8\u05bd\u05dc\u05d5\u05bc", "\u05d1\u05bc\u05b0\u05de\u05b4\u05e9\u05c1\u05b0\u05de\u05b0\u05e8\u05b9\u05ea\u05b8\u05bd\u05dd", "\u05d0\u05b7\u05ea\u05b8\u05bc\u05d4", "\u05d7\u05b2\u05dc\u05d5\u05bc\u05e5", "\u05e4\u05bc\u05b7\u05d2\u05b0\u05e2\u05b4\u05d9\u05d0\u05b5\u05dc", "\u05d0\u05b2\u05d7\u05b4\u05d9\u05e8\u05b7\u05e2", "\u05e0\u05b7\u05e2\u05b2\u05dc\u05b6\u05d4", "\u05e2\u05b8\u05e0\u05b8\u05d4", "\u05d5\u05b0\u05d0\u05b4\u05ea\u05bc\u05b0\u05db\u05b6\u05dd", "\u05e2\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05d4\u05b7\u05d2\u05bc\u05b8\u05d3\u05b4\u05d9", "\u05d4\u05b8\u05e2\u05b8\u05bd\u05e8\u05b6\u05d1", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e9\u05c1\u05b0\u05d1\u05bc\u05b0", "\u05d7\u05b8\u05d9\u05d5\u05bc", "\u05ea\u05bc\u05b7\u05e2\u05b2\u05dc\u05d5\u05bc", "\u05d5\u05bc\u05d1\u05b4\u05e9\u05c1\u05bc\u05b0\u05dc\u05d5\u05bc", "\u05d4\u05b8\u05e2\u05b9\u05bd\u05d3\u05b0\u05e4\u05b4\u05d9\u05dd", "\u05d5\u05b0\u05db\u05b7\u05d1\u05b0\u05e9\u05c2\u05b8\u05d4", "\u05e9\u05b8\u05c1\u05d0\u05b7\u05e8", "\u05e9\u05b0\u05c1\u05dc\u05d5\u05b9\u05e9\u05b4\u05c1\u05d9\u05dd", "\u05e1\u05b8\u05e4\u05b7\u05e8", "\u05d1\u05bc\u05b0\u05de\u05b9\u05e1\u05b5\u05e8\u05bd\u05d5\u05b9\u05ea", "\u05d0\u05b9\u05ea\u05b9\u05ea\u05b7\u05d9", "\u05d5\u05b0\u05d0\u05b6\u05e4\u05b6\u05e1", "\u05ea\u05bc\u05b4\u05e9\u05c2\u05b0\u05d8\u05b6\u05d4", "\u05d4\u05b7\u05d6\u05bc\u05b0\u05e8\u05b9\u05e2\u05b7", "\u05d1\u05bc\u05b0\u05e8\u05b8\u05e2\u05b8\u05ea\u05b4\u05bd\u05d9", "\u05d4\u05b8\u05d4\u05b8\u05e8", "\u05de\u05b8\u05e1\u05b7\u05da\u05b0", "\u05d9\u05b4\u05d2\u05bc\u05b8\u05e8\u05b7\u05e2", "\u05dc\u05b8\u05d4\u05bc", "\u05d1\u05bc\u05b4\u05e9\u05c1\u05b0\u05de\u05bd\u05d5\u05b9", "\u05e4\u05bc\u05b0\u05d3\u05b7\u05d4\u05b0\u05d0\u05b5\u05dc", "\u05d4\u05b7\u05d7\u05b9\u05e0\u05b4\u05d9\u05dd", "\u05d3\u05b4\u05d1\u05bc\u05b8\u05d4", "\u05d5\u05bc\u05d2\u05b0\u05e8\u05d5\u05bc\u05e9\u05c1\u05b8\u05d4", "\u05e4\u05bc\u05b6\u05df", "\u05d5\u05bc\u05e4\u05b0\u05d3\u05d5\u05bc\u05d9\u05b8\u05d5", "\u05d4\u05b7\u05de\u05bc\u05b7\u05d8\u05bc\u05b9\u05ea", "\u05d5\u05bc\u05ea\u05b0\u05d4\u05b4\u05d9", "\u05dc\u05b0\u05e6\u05b9\u05e0\u05b7\u05d0\u05b2\u05db\u05b6\u05dd", "\u05de\u05b4\u05d9\u05bc\u05b7\u05d9\u05b4\u05df", "\u05e2\u05b7\u05de\u05bc\u05b4\u05d9\u05d4\u05d5\u05bc\u05d3", "\u05e7\u05b8\u05e8\u05b8\u05d0", "\u05d9\u05b7\u05e4\u05b0\u05dc\u05b4\u05d0", "\u05d9\u05b0\u05dc\u05b7\u05d7\u05b2\u05db\u05d5\u05bc", "\u05d4\u05b2\u05d8\u05d5\u05b9\u05d1\u05b8\u05d4", "\u05d1\u05bc\u05b8\u05e7\u05b8\u05e8", "\u05d7\u05b2\u05d6\u05b5\u05d4", "\u05e8\u05b6\u05d2\u05b6\u05dc", "\u05de\u05b5\u05d0\u05b2\u05d7\u05bb\u05d6\u05bc\u05b7\u05ea", "\u05d0\u05b8\u05d7", "\u05e2\u05b8\u05e9\u05c2\u05b8\u05bd\u05d4", "\u05d8\u05bb\u05de\u05b0\u05d0\u05b8\u05ea\u05d5\u05b9", "\u05d1\u05bc\u05b0\u05de\u05b4\u05e9\u05c1\u05b0\u05e2\u05b2\u05e0\u05b9\u05ea\u05b8\u05dd", "\u05e8\u05b0\u05d1\u05b4\u05d9\u05e2\u05b4\u05d9\u05ea", "\u05d5\u05b7\u05ea\u05bc\u05b4\u05dc\u05bc\u05b8\u05d7\u05b5\u05e5", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e9\u05c1\u05b0\u05dc\u05b7\u05d7", "\u05e9\u05c1\u05bb\u05dc\u05b0\u05d7\u05b7\u05df", "\u05de\u05b0\u05d0\u05b7\u05e1\u05b0\u05ea\u05bc\u05b6\u05dd", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05e8\u05b0\u05d0", "\u05e0\u05b8\u05e9\u05b7\u05c2\u05d2", "\u05dc\u05b0\u05d7\u05d5\u05bc\u05e4\u05b8\u05dd", "\u05e7\u05b0\u05e6\u05b5\u05d4", "\u05e0\u05b8\u05e9\u05b8\u05c2\u05d0", "\u05dc\u05b0\u05d4\u05b7\u05e7\u05b0\u05d3\u05bc\u05b4\u05d9\u05e9\u05c1\u05b5\u05e0\u05b4\u05d9", "\u05d4\u05b5\u05e8\u05b4\u05d9\u05de\u05d5\u05bc", "\u05de\u05b4\u05d2\u05b0\u05e8\u05b0\u05e9\u05c1\u05b5\u05d9", "\u05d9\u05b7\u05d6\u05bc\u05b4\u05d9\u05e8", "\u05d5\u05b0\u05db\u05b4\u05e1\u05bc\u05d5\u05bc", "\u05d4\u05b7\u05d3\u05bc\u05b0\u05d1\u05b8\u05e8\u05b4\u05d9\u05dd", "\u05ea\u05bc\u05d5\u05b9\u05da", "\u05d1\u05bc\u05b4\u05e0\u05b0\u05d9\u05b8\u05de\u05b4\u05df", "\u05d7\u05b2\u05e6\u05b9\u05e6\u05b0\u05e8\u05b8\u05d4", "\u05d5\u05bc\u05de\u05b4\u05bd\u05e0\u05bc\u05b7\u05d7\u05b2\u05dc\u05b7\u05ea", "\u05d1\u05bc\u05b0\u05e2\u05b6\u05e6\u05b0\u05d9\u05d5\u05b9\u05df", "\u05ea\u05b8\u05de\u05b4\u05d9\u05dd", "\u05e6\u05b4\u05e4\u05bc\u05d5\u05b9\u05e8", "\u05ea\u05bc\u05b6\u05d7\u05b1\u05d8\u05b8\u05d0", "\u05e9\u05c2\u05b4\u05d1\u05b0\u05de\u05b8\u05d4", "\u05d0\u05b7\u05d4\u05b2\u05e8\u05d5\u05b9\u05df", "\u05e2\u05b9\u05e8\u05b8\u05d4\u05bc", "\u05ea\u05bc\u05b8\u05de\u05b9\u05ea", "\u05e9\u05b6\u05c1\u05e4\u05b6\u05e8", "\u05d5\u05bc\u05de\u05b7\u05d3\u05bc\u05b9\u05ea\u05b6\u05dd", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05de\u05bc\u05bd\u05b8\u05e1\u05b0\u05e8\u05d5\u05bc", "\u05d1\u05bc\u05b0\u05de\u05b8\u05e7\u05d5\u05b9\u05dd", "\u05dc\u05b8\u05bd\u05d0\u05b8\u05ea\u05d5\u05b9\u05df", "\u05d5\u05bc\u05ea\u05b0\u05e8\u05d5\u05bc\u05e2\u05b7\u05ea", "\u05de\u05d5\u05b9\u05dc\u05b6\u05d3\u05b6\u05ea", "\u05dc\u05b0\u05e0\u05b6\u05d2\u05b0\u05d3\u05bc\u05b4\u05bd\u05d9", "\u05d5\u05b0\u05d0\u05b8\u05d1\u05b4\u05d9\u05d4\u05b8", "\u05e4\u05b8\u05bc\u05e8\u05b7\u05d7", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05e6\u05bc\u05b8\u05de\u05b6\u05d3", "\u05ea\u05b8\u05bc\u05e4\u05b7\u05e9\u05c2", "\u05e4\u05bc\u05b7\u05e8\u05b0\u05e0\u05b8\u05bd\u05da\u05b0", "\u05d9\u05b8\u05db\u05d5\u05b9\u05dc", "\u05e2\u05b2\u05d5\u05ba\u05e0\u05b8\u05bd\u05d4\u05bc", "\u05d1\u05b4\u05dc\u05b0\u05e2\u05b8\u05dd", "\u05d5\u05b0\u05d4\u05b7\u05e4\u05b0\u05e9\u05c1\u05b5\u05d8", "\u05dc\u05b7\u05e2\u05b2\u05e9\u05c2\u05d5\u05b9\u05ea", "\u05e4\u05b8\u05bc\u05e7\u05b7\u05d3", "\u05e7\u05b0\u05e8\u05d5\u05bc\u05d0\u05b5\u05d9", "\u05d0\u05b6\u05dc\u05b0\u05d3\u05bc\u05b8\u05d3", "\u05d1\u05b0\u05bc\u05d4\u05b5\u05de\u05b8\u05d4", "\u05e0\u05b5\u05e1", "\u05d0\u05b1\u05dc\u05b9\u05d4\u05b8\u05d9", "\u05de\u05b5\u05d0\u05b5\u05df", "\u05d4\u05b7\u05e1\u05bc\u05b7\u05de\u05bc\u05b4\u05d9\u05dd", "\u05d5\u05bc\u05d1\u05b0\u05db\u05b6\u05e8\u05b6\u05dd", "\u05d9\u05b8\u05d3\u05d5\u05b9", "\u05d4\u05b7\u05e7\u05bc\u05bb\u05d1\u05bc\u05b8\u05d4", "\u05d4\u05b7\u05d2\u05bc\u05b4\u05d3\u05b0\u05d2\u05bc\u05b8\u05bd\u05d3", "\u05e4\u05bc\u05b8\u05d0\u05e8\u05b8\u05df", "\u05e2\u05b2\u05d3\u05b7\u05ea", "\u05d1\u05bc\u05b0\u05e8\u05b4\u05e1\u05bc\u05b8\u05bd\u05d4", "\u05d3\u05b4\u05d1\u05bc\u05b7\u05e8\u05b0\u05e0\u05d5\u05bc", "\u05d5\u05b0\u05d4\u05b5\u05d8\u05b7\u05d1\u05b0\u05e0\u05d5\u05bc", "\u05de\u05b7\u05dc\u05b0\u05d0\u05b8\u05db\u05b6\u05d9\u05da\u05b8", "\u05db\u05bc\u05b0\u05e0\u05b8\u05bd\u05e2\u05b7\u05df", "\u05e2\u05b8\u05d1\u05b7\u05e8\u05b0\u05e0\u05d5\u05bc", "\u05d0\u05b6\u05dc\u05b0\u05e2\u05b8\u05dc\u05b5\u05d0", "\u05d9\u05b4\u05e7\u05b0\u05e8\u05b8\u05d1\u05d5\u05bc", "\u05dc\u05b7\u05d8\u05bc\u05b8\u05de\u05b5\u05d0", "\u05e1\u05b7\u05dc", "\u05d9\u05b4\u05d8\u05bc\u05b7\u05de\u05bc\u05b8\u05d0", "\u05d1\u05bc\u05b0\u05d4\u05b7\u05e7\u05b0\u05e8\u05b4\u05d9\u05d1\u05b8\u05dd", "\u05db\u05b7\u05d1\u05bc\u05b5\u05d3", "\u05d1\u05b0\u05e7\u05b8\u05d3\u05b5\u05e9\u05c1", "\u05dc\u05b0\u05d9\u05b4\u05e9\u05c1\u05b0\u05d5\u05b4\u05d9", "\u05e9\u05b8\u05c1\u05e0\u05b8\u05d4", "\u05ea\u05b4\u05e4\u05b0\u05e7\u05b9\u05d3", "\u05ea\u05bc\u05b4\u05d9\u05e8\u05d5\u05b9\u05e9\u05c1", "\u05d4\u05b7\u05d9\u05bc\u05b8\u05de\u05b4\u05dd", "\u05de\u05b4\u05e7\u05b0\u05d3\u05b8\u05bc\u05e9\u05c1", "\u05e9\u05c1\u05b4\u05d2\u05b0\u05d2\u05b8\u05ea\u05b8\u05bd\u05dd", "\u05e9\u05c2\u05b8\u05d0", "\u05d7\u05b8\u05de\u05b5\u05e9\u05c1", "\u05db\u05b9\u05bc\u05d4\u05b5\u05df", "\u05d0\u05b8\u05d1\u05b8\u05bd\u05d3\u05b0\u05e0\u05d5\u05bc", "\u05d0\u05b7\u05e9\u05bc\u05c1\u05d5\u05bc\u05e8", "\u05d5\u05b7\u05ea\u05bc\u05b5\u05dc\u05b6\u05d3", "\u05e0\u05b8\u05e9\u05c2\u05b8\u05d0\u05ea\u05b8\u05d4", "\u05d1\u05b8\u05bc\u05d6\u05b8\u05d4", "\u05d4\u05b7\u05db\u05bc\u05b5\u05dc\u05b4\u05d9\u05dd", "\u05ea\u05bc\u05b0\u05e8\u05d5\u05bc\u05de\u05b9\u05ea", "\u05d9\u05b7\u05e7\u05b0\u05e8\u05b4\u05d9\u05d1\u05d5\u05bc", "\u05d0\u05b7\u05e8\u05b0\u05e6\u05d5\u05b9", "\u05e9\u05c1\u05b7\u05de\u05b0\u05e0\u05b8\u05d4\u05bc", "\u05de\u05b4\u05d3\u05bc\u05b8\u05de\u05b8\u05d4\u05bc", "\u05dc\u05b7\u05d7\u05b0\u05de\u05b4\u05d9", "\u05d4\u05b7\u05d1\u05bc\u05b0\u05de\u05b7\u05bd\u05d7\u05b2\u05e0\u05b4\u05d9\u05dd", "\u05d4\u05b7\u05de\u05bc\u05b0\u05e0\u05b7\u05e7\u05bc\u05b4\u05d9\u05bc\u05b9\u05ea", "\u05e2\u05b5\u05d6", "\u05db\u05bc\u05b7\u05d3\u05bc\u05b8\u05d2\u05b8\u05df", "\u05d2\u05b8\u05d5\u05b7\u05e2", "\u05e2\u05b7\u05de\u05bc\u05b6\u05d9\u05da\u05b8", "\u05d5\u05bc\u05e9\u05c1\u05b0\u05dc\u05b9\u05e9\u05c1\u05b8\u05d4", "\u05d4\u05b7\u05db\u05bc\u05b7\u05e3", "\u05dc\u05b0\u05d0\u05b7\u05e9\u05c1\u05b0\u05d1\u05bc\u05b5\u05dc", "\u05ea\u05bc\u05b4\u05ea\u05b0\u05d7\u05b7\u05d8\u05bc\u05b8\u05bd\u05d0\u05d5\u05bc", "\u05d1\u05b0\u05d0\u05b5\u05d9\u05d1\u05b8\u05d4", "\u05d0\u05b7\u05dc", "\u05dc\u05b8\u05bd\u05d4\u05bc", "\u05d5\u05b0\u05d9\u05b8\u05e8\u05b7\u05e9\u05c1", "\u05d4\u05b7\u05bd\u05d7\u05b4\u05d9\u05bc\u05b4\u05d9\u05ea\u05b6\u05dd", "\u05d4\u05b7\u05bd\u05d7\u05b8\u05e8\u05b0\u05de\u05b8\u05bd\u05d4", "\u05d4\u05b8\u05bd\u05d0\u05b1\u05de\u05b9\u05e8\u05b4\u05d9", "\u05d4\u05b5\u05d7\u05b8\u05dc\u05b0\u05e6\u05d5\u05bc", "\u05d1\u05b8\u05e8\u05b7\u05da\u05b0", "\u05e6\u05bb\u05d5\u05bc\u05b8\u05d4", "\u05db\u05b6\u05bc\u05e1\u05b6\u05e3", "\u05dc\u05b8\u05e9\u05c1\u05b8\u05d1\u05b6\u05ea", "\u05d1\u05b7\u05e9\u05c1\u05bc\u05b6\u05de\u05b6\u05df", "\u05d5\u05b0\u05d4\u05b8\u05bd\u05e2\u05b2\u05de\u05b8\u05dc\u05b5\u05e7\u05b4\u05d9", "\u05d8\u05b8\u05d4\u05d5\u05b9\u05e8", "\u05e7\u05b6\u05d3\u05b6\u05dd", "\u05d1\u05b0\u05d9\u05d5\u05b9\u05dd", "\u05e9\u05b4\u05c1\u05de\u05b0\u05e2\u05d5\u05b9\u05df", "\u05d1\u05bc\u05b0\u05e2\u05b5\u05d9\u05e0\u05b5\u05d9", "\u05dc\u05b0\u05d0\u05b8\u05dc\u05b8\u05d4", "\u05d7\u05b6\u05e8\u05b6\u05d1", "\u05db\u05bc\u05b8\u05ea\u05b4\u05d9\u05ea", "\u05e7\u05b8\u05d3\u05d5\u05b9\u05e9\u05c1", "\u05dc\u05b8\u05e0\u05d5\u05bc", "\u05e2\u05b7\u05e9\u05b0\u05c1\u05ea\u05b5\u05bc\u05d9", "\u05d3\u05b8\u05d1\u05b7\u05e8", "\u05d4\u05b4\u05e9\u05c1\u05b0\u05d0\u05b4\u05bd\u05d9\u05e8", "\u05d1\u05bc\u05b4\u05d2\u05b0\u05d5\u05b7\u05e2", "\u05d5\u05bc\u05d1\u05b0\u05e2\u05b4\u05d9\u05e8\u05b8\u05bd\u05dd", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05bb\u05ea\u05b7\u05dc\u05b0\u05d7\u05b4\u05d9", "\u05d0\u05b7\u05e8\u05b0\u05d1\u05b7\u05bc\u05e2", "\u05d5\u05bc\u05d1\u05b4\u05e0\u05b0\u05e1\u05b9\u05e2\u05b7", "\u05d5\u05bc\u05dc\u05b0\u05de\u05b4\u05e0\u05b0\u05d7\u05b9\u05ea\u05b5\u05d9\u05db\u05b6\u05dd", "\u05d5\u05b7\u05ea\u05bc\u05b4\u05e1\u05bc\u05b8\u05d2\u05b5\u05e8", "\u05e7\u05b0\u05d7\u05d5\u05bc", "\u05d1\u05b7\u05de\u05bc\u05b5\u05ea", "\u05e9\u05c1\u05b9\u05bd\u05de\u05b0\u05e8\u05b4\u05d9\u05dd", "\u05e4\u05bc\u05b8\u05d3\u05b9\u05d4", "\u05d4\u05b7\u05e9\u05c1\u05bc\u05b4\u05db\u05b0\u05de\u05b4\u05bd\u05d9", "\u05d1\u05bc\u05b8\u05d0\u05b7\u05de\u05bc\u05b8\u05d4", "\u05d4\u05b7\u05e7\u05bc\u05b0\u05dc\u05b9\u05e7\u05b5\u05bd\u05dc", "\u05d0\u05b7\u05d3\u05bc\u05b8\u05e8", "\u05d5\u05b7\u05ea\u05bc\u05b4\u05e4\u05b0\u05ea\u05bc\u05b7\u05d7", "\u05d5\u05b7\u05d9\u05bc\u05b7\u05bd\u05e2\u05b2\u05dc\u05d5\u05bc", "\u05d5\u05b7\u05e2\u05b2\u05d1\u05b8\u05d3\u05b6\u05d9\u05da\u05b8", "\u05d4\u05b6\u05bd\u05d7\u05b8\u05e6\u05b5\u05e8", "\u05e9\u05c1\u05b0\u05dc\u05bb\u05de\u05b4\u05d9\u05d0\u05b5\u05dc", "\u05d5\u05b7\u05d9\u05bc\u05b4\u05d7\u05b7\u05e8", "\u05d1\u05b7\u05bc\u05d9\u05b4\u05ea", "\u05dc\u05b0\u05d0\u05d5\u05b9\u05ea", "\u05d5\u05b7\u05d7\u05b2\u05de\u05b9\u05e8\u05b4\u05d9\u05dd", "\u05d2\u05bc\u05b7\u05d3\u05bc\u05b4\u05d9\u05d0\u05b5\u05dc", "\u05d9\u05b4\u05d8\u05b0\u05de\u05b8\u05d0", "\u05d5\u05bc\u05dc\u05b0\u05e4\u05b4\u05d9", "\u05d7\u05b9\u05de\u05b6\u05e5", "\u05d9\u05b7\u05d9\u05b4\u05df", "\u05d4\u05bd\u05b7\u05e2\u05b7\u05de\u05b0\u05e8\u05b8\u05de\u05b4\u05d9", "\u05e8\u05b8\u05e2\u05b8\u05ea\u05bd\u05d5\u05b9"], "type": "scattergl", "x": [-1.2391278743743896, 0.6619284749031067, 2.7566497325897217, -0.24320681393146515, -0.11353886127471924, 0.7373192310333252, -0.1888541430234909, -0.3708237111568451, -0.5761190056800842, -0.17170880734920502, -0.5787983536720276, -1.2988632917404175, -1.809494972229004, -0.14592845737934113, 0.7377080917358398, -1.7110313177108765, 1.0950971841812134, -0.3358604311943054, -2.3022892475128174, 1.6141234636306763, 8.370843887329102, 1.705938696861267, -1.5388400554656982, 0.21233874559402466, 2.188234806060791, 0.42063015699386597, -0.06148117408156395, -1.5758516788482666, -1.2578086853027344, -0.23567365109920502, -0.7796710133552551, -2.001105546951294, 0.8450528383255005, 1.267575740814209, 2.331352710723877, -0.3806132376194, -14.96287727355957, 2.3491034507751465, 2.465280055999756, -1.7571815252304077, 2.068479299545288, 0.6257684826850891, -1.55148184299469, -0.4073941111564636, -0.2909526824951172, 1.4067102670669556, -0.30261874198913574, 0.8208689093589783, -2.827544689178467, -2.899693250656128, -3.713775873184204, 1.673766016960144, -3.252615213394165, -0.9176924228668213, 1.5629404783248901, 6.384072780609131, -2.34851336479187, -0.12812548875808716, 3.084228038787842, -0.6137719750404358, -0.6005858182907104, 1.278481364250183, -0.6196869015693665, 4.390882968902588, -0.9279343485832214, 0.7431236505508423, -1.2329602241516113, 0.5427101850509644, -5.90709924697876, 2.455091953277588, 0.8696076273918152, 2.2817468643188477, -0.29904139041900635, 3.2715272903442383, 3.37565279006958, -2.1923720836639404, 0.3263958692550659, -1.726463794708252, -0.29822736978530884, 0.8592954874038696, -1.5397601127624512, 0.23802094161510468, -0.29907092452049255, -0.00619391119107604, -3.657909393310547, -22.21417808532715, -0.3528464734554291, 1.56017005443573, 1.2870115041732788, -8.188369750976562, -1.050696611404419, -0.7960201501846313, 3.249858856201172, -1.3484948873519897, -1.9180511236190796, 0.13260744512081146, 4.822758197784424, -1.7596272230148315, 2.2713277339935303, -6.1398797035217285, -0.6534198522567749, -1.031592845916748, 2.2762057781219482, -0.7510802149772644, 0.811551034450531, -3.9664437770843506, -0.1451663076877594, -2.7175588607788086, 1.6402339935302734, -9.956238746643066, 0.36186033487319946, -1.716313362121582, -1.5964587926864624, -0.26587921380996704, -0.3699941337108612, 0.1625470668077469, 1.631056308746338, -3.6853744983673096, -3.5835044384002686, 2.2854552268981934, -2.8089497089385986, -0.4994634985923767, -0.8680618405342102, -0.27573591470718384, 0.8625378608703613, -1.6862643957138062, 0.3822847902774811, -8.3154878616333, 1.2151196002960205, 1.430191159248352, -2.7821552753448486, -2.1298582553863525, 1.5484246015548706, 0.22631211578845978, 0.9296566247940063, -0.4513128101825714, -2.740748882293701, 1.1391438245773315, -2.619802236557007, -0.08582153916358948, 2.0039923191070557, 5.171509742736816, -1.4980964660644531, -1.5165566205978394, -1.0877063274383545, -3.3744986057281494, 0.6675336956977844, 1.9175666570663452, -1.242014765739441, -0.9069169759750366, -1.6152701377868652, -1.2205959558486938, -0.7474373579025269, 7.396864414215088, -5.3026299476623535, -0.45323485136032104, -0.06414490938186646, 8.58010196685791, -0.8803700804710388, -2.2658534049987793, 8.613945960998535, 1.3569201231002808, 2.291428804397583, 0.18330018222332, -0.34360504150390625, 1.52981698513031, 1.1642507314682007, -0.03373055160045624, -0.49251383543014526, 0.20299741625785828, 0.9035030603408813, 0.37344688177108765, -0.5045227408409119, -2.8794569969177246, 2.2089717388153076, -1.3256443738937378, -0.8832046985626221, 0.028849849477410316, -0.8179575800895691, -0.04609198123216629, -1.1982626914978027, 2.2985332012176514, 2.8843584060668945, 0.44388121366500854, -2.4643137454986572, -4.175143241882324, 1.7691564559936523, 0.5454356074333191, 2.2267508506774902, 1.6088603734970093, -0.7152355313301086, -0.17239345610141754, -1.4216010570526123, -0.7785159945487976, 1.252955436706543, 1.1667717695236206, -0.9834887385368347, -3.438732385635376, 0.3280235826969147, 0.5635980367660522, 0.3939513862133026, 1.2423574924468994, 8.63048267364502, 11.065287590026855, 0.23244473338127136, 0.8660293221473694, -2.458848237991333, -3.8680224418640137, -1.8170771598815918, -2.5730113983154297, 0.8316730260848999, -3.4248037338256836, 7.076292514801025, -0.10212771594524384, -0.7306550741195679, -0.1551058292388916, -4.168986797332764, 1.054869294166565, -1.099239468574524, 2.265798568725586, 0.8862007260322571, 1.4402652978897095, -0.49864405393600464, -1.7512295246124268, 0.32286176085472107, -0.898945152759552, -0.9407497048377991, 1.464464783668518, 2.070394992828369, 0.10106213390827179, -5.251884460449219, -2.585545063018799, -0.4708194136619568, 0.7426949143409729, 0.008267813362181187, -0.785801112651825, 1.9212350845336914, -1.6932960748672485, 15.810168266296387, -0.7122501730918884, 0.5936986207962036, 0.16583772003650665, 0.7377341389656067, 0.025774125009775162, 2.962904930114746, 0.9859674572944641, -1.0710852146148682, 2.6706812381744385, 0.6124294996261597, 8.614943504333496, -0.7852110266685486, -1.0652137994766235, -0.4503173530101776, 0.47657954692840576, -0.3494492471218109, 1.1943877935409546, 1.5686789751052856, 12.375537872314453, 1.9652434587478638, -0.7920029759407043, 3.4428446292877197, 3.702939033508301, 2.3829128742218018, 1.1097146272659302, 0.07783576846122742, -0.9354328513145447, 1.580214500427246, -0.9076789617538452, -0.3691290616989136, -1.5993989706039429, 2.6379103660583496, -2.2409651279449463, -1.8576865196228027, 1.297690987586975, -0.5797131657600403, 0.9660131931304932, -0.9543569087982178, 0.9951784610748291, -0.41887366771698, -0.7278375029563904, -1.4916565418243408, -0.9586730599403381, -0.5746997594833374, 2.131474256515503, 2.70361065864563, 1.965098261833191, -2.040229320526123, 1.568858027458191, 2.038123607635498, 0.6786407232284546, -2.1077675819396973, -1.0815482139587402, 0.19561327993869781, 1.6759065389633179, -2.885697603225708, 0.4302814304828644, 1.0772886276245117, -0.28331246972084045, -0.9089875817298889, -0.37380513548851013, -0.786749005317688, -3.2957208156585693, -0.27239909768104553, 2.347167730331421, -1.1389193534851074, 1.9603025913238525, -6.772032737731934, -1.463943362236023, 0.2870473563671112, 3.757725954055786, -1.1578083038330078, 0.5444701910018921, -1.2590241432189941, 0.8837906122207642, 12.460786819458008, 1.0922985076904297, 0.9955926537513733, 1.9422184228897095, 0.02567340061068535, 1.019810676574707, -1.2839715480804443, 1.2364044189453125, -1.0714483261108398, -3.0562918186187744, -0.08184707164764404, 1.5693751573562622, 0.06957266479730606, -4.031288146972656, -1.1323673725128174, -0.3453712463378906, -0.7027683258056641, 0.39129412174224854, 1.9966776371002197, 2.4219396114349365, -1.488183856010437, -0.00405037309974432, 2.0400874614715576, 1.9576433897018433, 3.0449631214141846, -1.5066263675689697, -2.796525001525879, -0.5857101678848267, -0.5337854027748108, 2.2272040843963623, 2.6881566047668457, -0.9588039517402649, -1.2335971593856812, 0.560668408870697, -2.642895221710205, -0.6470767855644226, -3.175929546356201, -3.0919387340545654, 0.11731411516666412, -2.8857295513153076, 2.1872475147247314, 2.0890235900878906, 0.2741485834121704, -0.6690415143966675, 7.1192193031311035, 2.3437256813049316, -0.2983833849430084, 2.6544127464294434, -0.5405760407447815, 0.03763522580265999, -0.9239792227745056, -1.7044990062713623, -3.9939682483673096, -0.46302443742752075, -0.3580493628978729, -3.879523754119873, -9.912067413330078, -0.8491690158843994, 0.4504249691963196, 14.816568374633789, 0.21213079988956451, 0.5118483901023865, 0.9320715069770813, 0.055066175758838654, -0.7334623336791992, 15.253324508666992, -0.31489914655685425, 1.2121107578277588, -0.52901691198349, -0.7929237484931946, -0.10668736696243286, 1.1825834512710571, 1.3884931802749634, -3.9090771675109863, -0.6149575710296631, 12.351839065551758, 2.1088204383850098, 0.3228525221347809, 1.7782409191131592, -0.6901472806930542, -0.7804218530654907, 7.533548355102539, -5.716802597045898, -0.5182892680168152, -0.8053305149078369, 1.9926382303237915, 8.48075008392334, -0.8568848371505737, 1.1522984504699707, 0.9663355350494385, -2.9618241786956787, -0.8723215460777283, 0.14249290525913239, 0.18097643554210663, 0.179801344871521, 3.5365073680877686, -0.592272937297821, -1.7461973428726196, -0.3467623293399811, -0.6406252980232239, -0.20548810064792633, 12.286239624023438, -1.7173030376434326, 0.07779242098331451, -0.5668333768844604, -3.43438982963562, -0.7146028280258179, 0.7630636692047119, -3.077031135559082, -0.5735257267951965, 1.565809726715088, 1.233245611190796, -0.3376833498477936, 0.7126988768577576, -1.9300240278244019, -3.1565957069396973, -1.4703056812286377, -0.8122256994247437, 0.9680423736572266, 3.6335532665252686, 5.028602600097656, -0.7636520862579346, 2.3334383964538574, -1.6494052410125732, -0.606286346912384, -2.671428680419922, -1.2575805187225342, 1.382778286933899, 5.541737079620361, -0.7750869393348694, 17.527957916259766, -2.536855697631836, 1.5489866733551025, 0.033784858882427216, -2.2125844955444336, 0.5700008869171143, 2.895733594894409, -0.7864789366722107, 0.4562614858150482, 0.5645990967750549, -3.3530688285827637, 0.6545248627662659, -3.0553700923919678, -5.7677412033081055, 1.988099217414856, 0.5538683533668518, 0.7008165717124939, 0.3959062695503235, -2.618940830230713, -0.31725171208381653, 0.2604440152645111, -1.4894471168518066, -4.229443073272705, -5.398831844329834, -0.20546765625476837, 0.12120701372623444, -6.100133895874023, -0.4256708025932312, 1.3955790996551514, -0.11201714724302292, 1.9696294069290161, 0.8208100199699402, -7.964280605316162, 0.7674494981765747, 0.49848857522010803, -3.406262159347534, 5.290591239929199, -0.02571660466492176, -6.139339923858643, -0.7009021043777466, 4.1086812019348145, 0.9943363070487976, -3.448585271835327, -2.2825779914855957, 5.451013088226318, -2.553237199783325, 1.0215528011322021, -2.186702251434326, 0.5166531205177307, -1.9743568897247314, -1.088784098625183, 1.5056184530258179, -0.7862902283668518, 0.011063302867114544, -0.7616826295852661, 0.04743075743317604, -0.8603803515434265, 1.3671338558197021, -3.5175135135650635, 0.6077784299850464, 7.913806915283203, -1.1202843189239502, 0.4792848825454712, -0.9671174883842468, 3.349041700363159, 9.837257385253906, -0.7914460301399231, 4.258127212524414, -1.1891496181488037, 0.9953256249427795, 0.433755099773407, 0.4258258044719696, 0.49094247817993164, 8.608818054199219, 6.024787902832031, -0.675800621509552, -1.1851903200149536, 0.8088779449462891, 1.5100913047790527, -1.1229479312896729, -1.1912730932235718, 0.5913835167884827, 3.0956084728240967, -1.0391616821289062, -0.17718324065208435, -2.4707276821136475, -1.0977524518966675, -2.3733739852905273, 2.4066648483276367, -3.6256041526794434, -2.494615316390991, 0.5517080426216125, 0.6944579482078552, -2.0999224185943604, -1.586376428604126, -0.013511915691196918, 1.9449864625930786, 3.855520009994507, -1.053145170211792, 0.6794472932815552, 0.8812069296836853, -2.405219078063965, 1.9142143726348877, -1.146655797958374, 1.850329875946045, -0.19699329137802124, -0.9142659306526184, 0.13150934875011444, -0.8403146862983704, 13.085555076599121, -0.08170220255851746, -0.4981074631214142, -1.3086358308792114, 1.8142931461334229, 7.839580059051514, -5.7743940353393555, 1.571606159210205, 1.1300504207611084, -4.00147819519043, 1.2509955167770386, -2.6096653938293457, -0.20169635117053986, -2.9964067935943604, 1.1393016576766968, -0.5432969331741333, 0.9665641188621521, 1.0246467590332031, -1.9862186908721924, 0.5644056797027588, 1.4404927492141724, -0.8047619462013245, -2.988255500793457, 1.734727382659912, -1.0069183111190796, 0.9362889528274536, 1.9818663597106934, -0.7856190204620361, -1.6845896244049072, 4.242635250091553, 7.89450740814209, -0.2940177619457245, 2.623476982116699, 7.572906970977783, 0.6445493698120117, -0.5318717360496521, 0.45616209506988525, -2.675766944885254, 0.9660362601280212, -0.8772711753845215, 2.351832866668701, -0.9415300488471985, 4.005881309509277, 2.1261560916900635, -5.762914180755615, -3.7030322551727295, 0.20184244215488434, 2.361421823501587, -0.5375864505767822, 8.222495079040527, -0.10318578779697418, 1.4606870412826538, -0.28770118951797485, 0.8152862191200256, -0.5967310070991516, 2.1366193294525146, -0.2555958330631256, -0.8760930299758911, -1.0308928489685059, -1.1924878358840942, 0.6279320120811462, 0.9176145195960999, 3.1554532051086426, -0.16250483691692352, 0.8775537014007568, 1.171401858329773, -0.18368031084537506, 1.9628268480300903, 0.12024104595184326, 0.7953307628631592, -0.9760627150535583, -1.1683220863342285, -1.3744432926177979, -0.1406061053276062, 0.03588267043232918, -0.5294011831283569, 8.581122398376465, 12.287991523742676, 1.512328028678894, 1.4692848920822144, -2.730179786682129, -0.5546458959579468, -0.6355317831039429, 8.611566543579102, -2.018796682357788, -0.8050389289855957, -1.424214243888855, -2.6236939430236816, -0.6853870749473572, 0.37947389483451843, -3.0860257148742676, -0.8792592883110046, -1.1910640001296997, -1.9913641214370728, 1.1617255210876465, -0.13600696623325348, 4.057627201080322, 1.1492840051651, -1.2149710655212402, -0.404559850692749, 3.6559855937957764, -1.5930976867675781, 0.49892082810401917, 1.8462082147598267, 0.8233648538589478, 3.5935258865356445, -11.8534574508667, -0.38651585578918457, -0.01845632866024971, 1.6723952293395996, -0.7870756387710571, 0.7511359453201294, 2.096546173095703, 2.146009683609009, -4.387435436248779, -0.6771907210350037, -4.1658549308776855, 0.853594958782196, -1.3650283813476562, -0.024805378168821335, -0.8887994289398193, 0.3351113498210907, 0.013600906357169151, 0.9311371445655823, -1.783854365348816, -1.2551875114440918, 0.23849251866340637, -0.3852289617061615, -1.251039743423462, 0.9347712397575378, -3.29819393157959, -1.3610173463821411, 4.2940263748168945, 0.2884589433670044, 0.006795660126954317, -0.7855072021484375, 2.2521910667419434, -1.7312308549880981, -0.5825201272964478, 0.19304227828979492, 3.1448309421539307, 0.54407799243927, 1.9679783582687378, 0.8595677614212036, 1.6688472032546997, 0.9674359560012817, -5.6721367835998535, 8.604701042175293, -2.573124647140503, 0.2783336043357849, 0.05467844754457474, -0.38572728633880615, -1.0807665586471558, -0.1561494618654251, 8.609091758728027, 0.05183696374297142, 3.149249315261841, 0.8318275213241577, -0.9958599805831909, 0.21083468198776245, 0.5374398231506348, 0.019438361749053, -2.1363742351531982, -0.2757643759250641, 0.4773370325565338, -0.8713132739067078, 3.3988356590270996, -2.514313220977783, -4.035504341125488, -3.0864765644073486, 0.946799099445343, -2.548027276992798, 0.4361361861228943, -3.450620412826538, -0.25735917687416077, 1.7779093980789185, -2.7539594173431396, 3.154895305633545, -2.8196327686309814, 1.0778886079788208, -0.7885702848434448, -16.429241180419922, -0.8827717304229736, -0.35639065504074097, 1.9293845891952515, -1.1179150342941284, -0.5446486473083496, -0.05118301138281822, 1.9973655939102173, 0.960989236831665, -1.139306664466858, -1.6783803701400757, -0.958987295627594, -1.916488766670227, -8.350074768066406, 1.7553688287734985, 1.4182889461517334, 0.18954138457775116, -0.6950327754020691, 6.421947956085205, 1.8089276552200317, 8.597701072692871, 0.006135222967714071, -0.7310250997543335, 2.0224521160125732, 0.18728779256343842, -1.9005906581878662, -0.787238597869873, 1.571352481842041, -0.09585806727409363, -1.8741573095321655, -1.2996852397918701, -0.08046597987413406, -0.016615701839327812, 7.661630153656006, 1.5168102979660034, 2.3322999477386475, -2.565138339996338, 3.517784833908081, -0.7610140442848206, 0.5189688205718994, -4.662154197692871, -0.9977778196334839, -1.683760166168213, 0.3460540175437927, -3.439650297164917, -0.524840772151947, -1.787968397140503, 1.5824319124221802, -0.7487452030181885, -0.0438590869307518, 0.15353405475616455, -0.8089615106582642, -0.8222542405128479, -1.5324875116348267, -3.100290298461914, 2.106330633163452, 7.612919807434082, 1.0172778367996216, -1.0026371479034424, 3.9927194118499756, -0.17837251722812653, -2.0919301509857178, -1.1877073049545288, 1.3155441284179688, 1.9206814765930176, -0.8138694763183594, -0.011378784663975239, 0.456827312707901, -1.3107255697250366, -1.222883701324463, -1.8609628677368164, 2.2201197147369385, -0.7340465784072876, -0.8776324987411499, 0.5605944991111755, 0.9578443169593811, -0.48592251539230347, -1.2392385005950928, 3.067357063293457, 8.33664321899414, -0.7121676802635193, 0.726914644241333, 0.7395191788673401, 0.32802456617355347, 1.2190364599227905, 2.148521900177002, 0.7608408331871033, -0.9007473587989807, -2.233752489089966, -0.4622141718864441, 0.42865046858787537, -0.4340115487575531, 1.4000762701034546, -0.2059980034828186, -3.042067527770996, 0.1898757964372635, 3.5426130294799805, 3.8051390647888184, -1.4510483741760254, -1.0372463464736938, -0.8979973793029785, -2.2496962547302246, -0.9443720579147339, 1.6717017889022827, -1.533027172088623, -42.470542907714844, -0.8047745227813721, 6.537466049194336, -0.4709324538707733, 0.26666226983070374, -1.2922284603118896, 12.637771606445312, -4.166718006134033, -1.2407184839248657, 3.2692482471466064, -0.8632009625434875, -1.0368525981903076, -2.281696081161499, 0.04902295023202896, 1.5814303159713745, -0.09009131789207458, 2.880976676940918, 0.41375863552093506, 2.3711307048797607, -1.0806479454040527, -1.3111506700515747, -4.146996974945068, -1.847506046295166, 0.20176364481449127, -1.9714066982269287, -1.5013538599014282, -3.6673028469085693, -2.89994740486145, -3.922971487045288, 12.295568466186523, 3.820138931274414, -1.091542363166809, -4.039958953857422, 2.6051933765411377, -0.27424317598342896, 1.9250377416610718, 1.1372045278549194, -1.1557037830352783, 0.6549244523048401, -6.728635311126709, -1.4525222778320312, -6.143911361694336, -0.6995088458061218, 2.781789541244507, -0.09297088533639908, 0.21040993928909302, -1.2408735752105713, 2.441405773162842, 0.5956540107727051, -7.344812870025635, -2.630343198776245, -2.7466678619384766, -0.20428124070167542, 0.9999544024467468, -1.268941044807434, -2.5218563079833984, -3.421839952468872, -2.4405128955841064, 1.071734070777893, 8.58952808380127, 0.3851516544818878, 2.714810371398926, 0.48195022344589233, 0.331389456987381, -1.0913372039794922, 10.003156661987305, 0.1372021585702896, 4.773741722106934, 0.09752046316862106, -2.5654916763305664, 1.062339425086975, 3.2189087867736816, 0.7575860619544983, 2.5700464248657227, 1.0998492240905762, 0.27534574270248413, -13.887608528137207, 1.01300048828125, -0.8792223334312439, 2.6589372158050537, 2.16955828666687, 0.9759668111801147, 1.0601379871368408, -1.4951461553573608, -0.7879871726036072, -1.1933566331863403, 1.4645023345947266, 11.287781715393066, 1.0671035051345825, -0.5244131088256836, 1.3531253337860107, -0.9999560117721558, -2.1222126483917236, 1.370229959487915, -2.900080442428589, -3.0760300159454346, -2.10294508934021, -2.2096121311187744, -0.9867574572563171, 1.0332437753677368, 1.0748528242111206, 3.4449474811553955, 2.418858528137207, 3.553405284881592, 0.32971835136413574, -0.0939481258392334, -0.07847251743078232, 0.5189783573150635, -1.1814221143722534, -0.643229603767395, 2.6824395656585693, 0.05520577356219292, -0.66892409324646, 0.47759780287742615, 0.8759567141532898, 1.2809010744094849, -3.1642274856567383, -0.3846501410007477, 1.5696202516555786, 5.932685375213623, 1.4072870016098022, -0.9653926491737366, -0.17405515909194946, -1.0817246437072754, 1.7254499197006226, 2.3871147632598877, -0.5551779270172119, -3.428166151046753, -4.298928260803223, -1.5376509428024292, -0.5777238607406616, -2.8221166133880615, -7.6454267501831055, 0.26515862345695496, 1.3494659662246704, -0.21969947218894958, 0.9591092467308044, 2.9546585083007812, -2.8405239582061768, -0.29170775413513184, 0.010164710693061352, -2.294849395751953, 1.04652738571167, -0.3760231137275696, -0.8823460936546326, -1.9985809326171875, -0.8724995851516724, 0.9989283680915833, 0.1777319461107254, 1.50424325466156, -0.7221773266792297, 0.3869919776916504, 6.250656604766846, -2.2680675983428955, 0.8012931942939758, 1.3781824111938477, 2.050419569015503, -0.23100735247135162, 0.5777433514595032, -2.2268898487091064, -1.0147417783737183, -0.7794390320777893, -0.8945894241333008, 1.4227221012115479, 2.3952853679656982, 1.130426287651062, 0.058581240475177765, 2.1382553577423096, -1.0254783630371094, 2.832744836807251, -0.873851478099823, -0.3576934039592743, 0.19568654894828796, 2.759084463119507, -0.25674253702163696, -4.509575366973877, -1.9078425168991089, 2.239884853363037, -0.7842319011688232, 1.8457725048065186, -9.345880508422852, 0.8959729075431824, 0.9049505591392517, -2.2990708351135254, -2.027482748031616, -0.22397775948047638, 0.1539539247751236, -1.0921107530593872, -1.1172780990600586, -0.5915156602859497, 3.049959897994995, -0.14097434282302856, -1.1170198917388916, 0.23509685695171356, -1.4608887434005737, 2.321019411087036, 1.3086868524551392, 0.32531216740608215, -1.203148365020752, 0.18086393177509308, -0.46163591742515564, 0.292111873626709, -0.855159342288971, -1.2839957475662231, -0.10712093114852905, 0.005712824873626232, -4.104368686676025, 4.17948055267334, -0.42891985177993774, 0.4530063271522522, 2.6671741008758545, -1.5790064334869385, -1.0373824834823608, 1.9982160329818726, -1.8930103778839111, 4.240650177001953, -0.5172497630119324, 1.6782910823822021, -4.158446311950684, -0.8279054164886475, 2.6214048862457275, -1.207495093345642, -2.3685781955718994, 0.6690758466720581, -0.8720505237579346, 4.575637340545654, 1.0213631391525269, -0.7860504984855652, -3.9215049743652344, -0.008063812740147114, -11.769268035888672, -9.48522663116455, -1.5384713411331177, -0.004952327813953161, 0.7323596477508545, 1.9379205703735352, 7.164057731628418, -0.9131961464881897, -0.09780146181583405, -3.0826690196990967, -0.1918099820613861, 2.1902341842651367, -0.8762231469154358, 2.3763234615325928, -2.818162441253662, -0.7720217704772949, -3.3222134113311768, 0.9043870568275452, 5.412070274353027, -3.2035884857177734, -1.0491405725479126, 0.11691665649414062, -0.35047897696495056, -4.165502548217773, 0.32478880882263184, 2.0556387901306152, 0.5477924942970276, 0.22354096174240112, 0.9914308786392212, -0.916375458240509, -0.31869250535964966, -0.1958017349243164, 2.441758871078491, 1.3082610368728638, -0.6658033132553101, -0.13437597453594208, 2.1493475437164307, -1.0729749202728271, 2.234022378921509, -2.069246292114258, -0.3808601200580597, -0.8568445444107056, 0.6206275820732117, 4.682829856872559, 1.7117265462875366, 8.197827339172363, -1.124829649925232, 0.5543208718299866, -0.07780034095048904, 0.18392862379550934, -0.572714626789093, 2.2528374195098877, -1.4984126091003418, 0.6436232328414917, 0.6950668692588806, 0.3265017867088318, -3.086326837539673, -1.912286400794983, -1.8994284868240356, 1.8431278467178345, 0.4034830927848816, 0.7680768370628357, 0.06410457193851471, -3.06687331199646, 0.7396469712257385, -3.4380879402160645, -0.7609201073646545, -1.9142978191375732, -0.20918093621730804, 0.9389787912368774, -0.6902554035186768, -2.467163562774658, -0.011271611787378788, -3.546116352081299, -1.1882617473602295, -0.9954767823219299, 0.9268492460250854, 3.9423184394836426, -3.617074966430664, -0.5526331663131714, 1.819124460220337, -3.8511886596679688, -1.460944414138794, 1.0604445934295654, -1.663140058517456, -0.11124418675899506, 0.19520661234855652, -1.045964241027832, 1.7524365186691284, -6.101935863494873, 0.3872874081134796, -1.5921225547790527, 0.11663417518138885, 1.9180548191070557, -0.06603215634822845, -3.591221809387207, -0.5048341751098633, 1.5224415063858032, 15.361794471740723, -2.118607521057129, -6.1409807205200195, 4.713254928588867, 4.045491695404053, -0.640563428401947, 4.9833221435546875, -1.9192144870758057, 2.1900343894958496, 3.1649341583251953, -0.5699107646942139, 2.041717052459717, 0.5364017486572266, 1.2363533973693848, -0.16201519966125488, -4.237942695617676, -0.08806464821100235, -1.0986249446868896, 0.0725969448685646, -0.7612438797950745, -1.195553183555603, -0.10015158355236053, -0.5772014260292053, 2.070887804031372, 2.327749252319336, 2.052496910095215, 0.7688361406326294, -1.1893212795257568, -0.3240433931350708, 0.9594507217407227, 2.0753233432769775, -1.723789930343628, 6.044783592224121, 8.212286949157715, -0.40043431520462036, 2.067380666732788, 0.7995193600654602, -9.228209495544434, -0.8029207587242126, -0.24792033433914185, -0.5194366574287415, -0.10807507485151291, -0.0812743529677391, 0.47832462191581726, 0.38166189193725586, -1.1364145278930664, -3.7648301124572754, -3.893873453140259, -3.846787214279175, -1.546089768409729, -0.2411772906780243, -2.6664445400238037, 0.1055338978767395, 0.07934390753507614, 0.054006028920412064, -3.506397247314453, 0.5351368188858032, -4.320569038391113, -0.8200334906578064, -2.377281665802002, 0.15786872804164886, 0.5247527956962585, -0.027846837416291237, 0.9235348701477051, -0.7726801633834839, 0.0896112397313118, -0.20995590090751648, 4.406101703643799, -0.04194042831659317, -0.1475103497505188, 0.42271995544433594, 2.3378846645355225, 2.2879865169525146, -9.454028129577637, 1.3316893577575684, -1.3469815254211426, 0.805587112903595, 1.6097725629806519, 1.4604090452194214, 1.1540348529815674, 0.6182358264923096, 1.8064310550689697, -0.9259316325187683, 1.6404446363449097, -0.9300186038017273, 0.6153541207313538, 2.2774581909179688, -0.8516419529914856, 1.2230602502822876, 2.427034616470337, -2.5756795406341553, 0.33335593342781067, -0.6711969375610352, -4.69682502746582, -2.1390092372894287, 2.731193780899048, -0.19317065179347992, 0.31882140040397644, -1.7549093961715698, -0.41783154010772705, 13.43304443359375, 2.023470163345337, -2.898315906524658, -0.4663943648338318, 1.236324429512024, 2.4081575870513916, 3.0732882022857666, 1.9980392456054688, 2.8491079807281494, -34.13524627685547, -2.504258155822754, 0.9639527797698975, 0.8502789735794067, 1.9538931846618652, -0.0437956377863884, 8.59897232055664, -0.07218260318040848, -0.36504700779914856, 0.9295544624328613, -0.4891185462474823, -4.64011812210083, -0.015310955233871937, 12.299286842346191, 0.550223171710968, -0.7494479417800903, 0.16422253847122192, -3.402446985244751, 0.5878026485443115, -0.02249746024608612, 1.4552206993103027, -3.8725836277008057, 0.46710914373397827, 0.3742344081401825, 0.27422112226486206, -0.7846884727478027, 1.7993683815002441, -1.3124430179595947, -1.4081624746322632, -0.7902507781982422, -1.1248410940170288, 3.5221750736236572, 2.347288131713867, 0.01588507369160652, -0.37792885303497314, -1.5197973251342773, 0.10479310899972916, -2.710540294647217, 1.2301950454711914, 10.72230052947998, -2.1376006603240967, 2.3748624324798584, -3.3277854919433594, 0.5168847441673279, 1.0588369369506836, -1.271469235420227, -1.162774682044983, -2.564302444458008, 0.3539758324623108, 0.04111485183238983, -1.3871381282806396, -1.5299137830734253, 8.107870101928711, 1.4700472354888916, -3.148536443710327, 2.403899908065796, -0.7857410311698914, 17.90781593322754, 2.0189433097839355, 2.453537702560425, -0.504275381565094, -0.630664587020874, 0.7770383358001709, 2.6423745155334473, 8.631131172180176, 1.5124493837356567, -1.1127263307571411, -0.877104640007019, -1.3862268924713135, 1.1825780868530273, -2.8914217948913574, 0.49315330386161804, 0.32849133014678955, -1.4587147235870361, -2.9797675609588623, 2.310237407684326, 1.23296320438385, 1.7902604341506958, -2.521939754486084, 1.7216323614120483, -1.1805262565612793, -2.458787202835083, -5.064272880554199, -0.5705530643463135, -0.3286529779434204, -0.4056021273136139, -1.011035680770874, 0.016971241682767868, 2.3231148719787598, 0.9246830940246582, -1.7605913877487183, 0.09539973735809326, -0.29141536355018616, 0.7477588057518005, -1.2932103872299194, 1.5587455034255981, -3.2551653385162354, -3.2920796871185303, -0.2695932388305664, -1.719942331314087, -0.49400851130485535, -1.6028180122375488, 1.927377462387085, -0.7229045033454895, 3.644285202026367, 1.5829927921295166, -0.4789576232433319, -0.857609212398529, 12.311692237854004, -3.212783098220825, -0.9104476571083069, -5.02079439163208, -1.511381983757019, 0.39499685168266296, -0.12901562452316284, 0.03404741361737251, -0.7862991690635681, 1.029206395149231, -3.890411853790283, -1.2422202825546265, 1.0083547830581665, -0.3090704083442688, 1.7039772272109985, -1.9384644031524658, -1.8248026371002197, -0.22822967171669006, 4.353923797607422, 1.183651089668274, -1.0518105030059814, -2.3026351928710938, 1.0200793743133545, 0.42969563603401184, -0.36777812242507935, -0.8107782602310181, -1.3062578439712524, 1.3340710401535034, 12.462345123291016, 0.6384891867637634, -0.8498508930206299, 1.2263137102127075, 2.0969340801239014, -1.195439100265503, -1.8369925022125244, 2.0091822147369385, 0.8807891011238098, 6.574088096618652, -0.025541959330439568, 0.7382895350456238, -4.360840320587158, 1.2010091543197632, -0.04942528158426285, -1.1146408319473267, -2.1182312965393066, 2.2564163208007812, -2.0656912326812744, 4.003781795501709, -4.127634525299072, -0.6434158682823181, -1.3642023801803589, -5.727838516235352, 0.41186973452568054, 0.2936205267906189, 6.300079345703125, 1.988442063331604, -1.2086948156356812, -1.292810320854187, -0.21749086678028107, -2.6167216300964355, -0.08079613745212555, 0.7090657353401184, -1.7929682731628418, 2.8404417037963867, 2.7307326793670654, 0.05744041129946709, -2.0861079692840576, 2.2911577224731445, 2.8340842723846436, 1.5192253589630127, -2.302748441696167, -1.416759729385376, 2.087369441986084, -2.067784547805786, -2.8075358867645264, -1.5751066207885742, -0.7982749938964844, -3.5651698112487793, -1.4659953117370605, 2.1281604766845703, -1.3539276123046875, -1.9050523042678833, -0.7576930522918701, -0.4572654068470001, 0.6821553111076355, -0.08023284375667572, -0.2258061021566391, 0.8086251020431519, -0.7856572270393372, 0.7812191247940063, -0.09509023278951645, -0.4652782678604126, -0.8797634243965149, 2.8635098934173584, -0.24521251022815704, 1.33183753490448, 1.8207913637161255, -0.24934817850589752, -1.1644279956817627, -0.788559079170227, 2.7395217418670654, -0.4326981008052826, 3.8039090633392334, -0.7805596590042114, -0.3141111135482788, 3.132473945617676, -2.169312000274658, 0.858698308467865, -0.932104229927063, -0.6841803789138794, -3.879394769668579, -9.862302780151367, -0.8944038152694702, -9.243358612060547, -0.7796391844749451, 1.2108891010284424, 0.604243278503418, -3.795743703842163, 1.795860767364502, 17.72897720336914, 1.9020640850067139, 1.570540189743042, -1.1855441331863403, -0.15437474846839905, -0.4840155839920044, -1.052748680114746, 1.1935323476791382, -6.126471519470215, -3.741858959197998, 0.4788229167461395, 3.7171826362609863, -1.2943637371063232, -0.11580535024404526, -1.6285854578018188, 0.7699400782585144, -2.5374252796173096, 1.9921334981918335, -0.5067914128303528, 1.3026496171951294, -4.156763553619385, -4.126408576965332, -5.741702556610107, -1.5887037515640259, 1.4203089475631714, -0.3086639940738678, 0.39891090989112854, -0.47942298650741577, 0.07347994297742844, -0.7903458476066589, 2.6012356281280518, 9.580142974853516, 0.965809166431427, 0.21267876029014587, -1.435080647468567, -3.1253678798675537, -0.8225609064102173, -4.167092323303223, 4.623685359954834, 1.435058355331421, -3.236086368560791, 0.3164163827896118, 0.03675224259495735, -2.036377429962158, -0.26495930552482605, 3.7290897369384766, 0.45314720273017883, -2.9466135501861572, 0.9220932722091675, 3.9848504066467285, 0.6892641186714172, -0.8782560229301453, -2.3115077018737793, 1.838757038116455, -0.6814885139465332, -1.778159737586975, 3.245600700378418, -3.119814395904541, 0.47784358263015747, 2.2694761753082275, 0.43821531534194946, -1.8768177032470703, 0.9255149364471436, 0.7604513168334961, 0.9223872423171997, -0.6821710467338562, 0.23381450772285461, -3.190514087677002, -0.39440596103668213, 0.3691155016422272, 0.42889928817749023, 0.5331907272338867, 4.40710973739624, 1.9455868005752563, -0.3242567777633667, -0.2843644917011261, -0.6156877279281616, 1.510556936264038, -2.8508784770965576, -0.07638862729072571, 2.820833683013916, 1.3265581130981445, 1.9817816019058228, -2.800379753112793, 2.1019134521484375, 1.1062977313995361, -0.602000892162323, -0.36755865812301636, -2.9947988986968994, -1.8911197185516357, -0.2270583212375641, -1.0867234468460083, -4.172918796539307, -2.0214061737060547, 2.2350714206695557, 3.819735527038574, 0.2661103904247284, 6.417355537414551, 1.47227144241333, 0.8521236181259155, 1.0396440029144287, -0.48102205991744995, -0.96431964635849, -1.4238500595092773, 0.6590655446052551, 4.848754405975342, -0.7906510233879089, -1.135561466217041, -3.5012319087982178, -0.14169633388519287, 1.7591925859451294, -1.0578786134719849, 0.7576945424079895, -1.1013225317001343, -1.5605971813201904, -1.0998574495315552, 1.209106683731079, 21.075054168701172, 0.8746381402015686, -2.8853137493133545, -0.7902389764785767, -2.0022666454315186, -3.447143316268921, -10.454475402832031, -2.1411166191101074, -2.358144521713257, -1.2065565586090088, 1.375746726989746, -0.41220083832740784, -2.556057929992676, -1.5108869075775146, -0.5264655947685242, 0.24484406411647797, -1.712942361831665, -3.3276941776275635, 1.25419020652771, -0.2149055451154709, 4.963250637054443, -0.8744087219238281, 1.2116165161132812, -1.816142201423645, 2.937399387359619, -0.5036476254463196, -4.126790523529053, 1.6378734111785889, 0.2772139608860016, 1.0717048645019531, -0.4888140559196472, 0.18446116149425507, 1.271359920501709, -3.0942678451538086, 0.2586084306240082, 4.49666166305542, -0.6056855916976929, -0.7312051653862, 2.200324773788452, 0.47506505250930786, -1.333516001701355, -0.288928359746933, 1.329606056213379, -2.4420762062072754, 0.10894354432821274, 1.7259453535079956, 0.3227768540382385, 1.19461190700531, -0.894009530544281, -12.427922248840332, -3.0379602909088135, 2.59401798248291, -1.1737879514694214, 4.050178050994873, -3.390317678451538, 0.3981669843196869, -3.8663854598999023, 0.9360033273696899, 15.328880310058594, 0.5367186069488525, -0.20244237780570984, -0.7188704609870911, 2.3212809562683105, 3.243345022201538, 19.521862030029297, -0.6403329968452454, 2.92992901802063, -0.5609819889068604, 1.1755186319351196, -4.699931621551514, 1.2162635326385498, 1.0906457901000977, 2.7635438442230225, -0.7462660074234009, -0.7272487878799438, -1.1198467016220093, 0.16025254130363464, 5.254193305969238, 17.34173583984375, 0.7828794121742249, -0.7944172620773315, 2.995013475418091, -0.8953165411949158, -0.6802982687950134, 2.7776379585266113, -1.739829659461975, -2.43386173248291, 1.2009928226470947, 0.09498538821935654, 0.4991500675678253, -0.08208587765693665, -0.05695537105202675, -0.430950790643692, -1.6496469974517822, -1.223142385482788, 2.3751440048217773, 0.11269981414079666, -41.997745513916016, -2.336217164993286, -0.48502156138420105, -1.5103750228881836, 1.3440961837768555, -1.0767130851745605, 0.1553996056318283, -0.1519518494606018, 0.5627733469009399, 2.943916082382202, 0.3728375732898712, -0.03028051368892193, -1.1855286359786987, -3.08142352104187, -3.5269391536712646, -4.992541790008545, 0.1431383639574051, -0.40347981452941895, -3.4113574028015137, 0.3796885907649994, 4.701793193817139, 1.8917592763900757, 1.6972213983535767, 1.080954909324646, -0.07012882828712463, -4.219098091125488, 0.31909942626953125, 0.4944329559803009, -4.322762966156006, 0.1718473881483078, 0.4055609405040741, 8.202977180480957, 1.4606270790100098, -6.346334934234619, 0.6752544045448303, 1.8032588958740234, 2.166714668273926, -0.41708651185035706, 1.2920259237289429, 1.761130690574646, -0.39887768030166626, -2.0505290031433105, 1.1447633504867554, -3.061481237411499, -0.21105602383613586, 1.9754353761672974, 0.2925634980201721, -0.2343365103006363, -2.521003484725952, -0.4569288194179535, 1.6781550645828247, -2.213493585586548, -0.35771191120147705, 1.610893726348877, -0.9197801947593689, -0.9079297184944153, 0.4607371687889099, -1.630722165107727, -2.0492234230041504, -3.074890375137329, -0.9026216268539429, -2.2646756172180176, -0.3193836510181427, -2.6315107345581055, 0.9578222632408142, -1.3540728092193604, -0.6804665923118591, 2.1128838062286377, -1.854234218597412, 0.9996904134750366, 0.7869634628295898, 1.5743498802185059, 2.441433906555176, 1.1421172618865967, -1.3439826965332031, 17.80628204345703, 2.155856132507324, 0.4596031904220581, 1.4911657571792603, -2.2634239196777344, 4.2330546379089355, 0.869849681854248, -3.484196901321411, 0.28742459416389465, -1.3820579051971436, -1.0200759172439575, -0.334109902381897, 1.9103864431381226, -2.5008816719055176, 1.9342875480651855, -0.2200053334236145, 1.4292044639587402, 6.876657485961914, -1.866996169090271, 0.04537064582109451, -1.2860798835754395, -1.844008207321167, -2.527639150619507, -0.511396050453186, 0.8122269511222839, -0.24220772087574005, -1.610304355621338, -1.0014790296554565, 1.188035249710083, 1.0650516748428345, -11.663244247436523, -6.134707927703857, 2.801304578781128, -0.7898440957069397, 13.114990234375, 0.6276040077209473, -0.9594433903694153, 0.394727885723114, 0.8745006918907166, 1.4821405410766602, 5.084089279174805, 1.5020172595977783, 0.8785995244979858, -2.256405830383301, -0.6469630599021912, -1.8269375562667847, 3.2753517627716064, -1.7034339904785156, 2.345736026763916, -1.5975764989852905, -0.21488447487354279, -1.3201321363449097, 0.209612637758255, -1.153003215789795, -3.0021119117736816, 3.586109161376953, 5.485062122344971, 4.143600940704346, 14.490673065185547, 0.07647699117660522, 0.9221590161323547, 13.67318344116211, 4.364546298980713, 13.083089828491211, -2.0382328033447266, 1.3355154991149902, 0.40350982546806335, -0.7401686906814575, -1.8957817554473877, 0.277731716632843, -1.5451617240905762, -0.7090769410133362, -0.6633409261703491, -1.0284216403961182, 8.597384452819824, -0.7815711498260498, -0.6566557884216309, -3.146430492401123, 0.8871307373046875, 1.8184731006622314, -1.1109713315963745, -2.9041149616241455, 12.664366722106934, 0.3473091423511505, -0.4856596887111664, -0.27852433919906616, -3.9478912353515625, -12.036023139953613, 3.419741630554199, 4.535611152648926, 2.791935682296753, -0.7953957319259644, -1.134257435798645, 0.05324631184339523, 3.8695461750030518, 0.5646759271621704, -0.10611843317747116, 1.1924179792404175, 12.28978443145752, -0.4303094446659088, -3.0110723972320557, -0.8866086602210999, 0.48989924788475037, -11.588138580322266, -3.4387619495391846, 0.22640718519687653, -2.3432435989379883, 2.6301403045654297, 0.29573535919189453, -0.7930565476417542, -2.799464225769043, 0.18249325454235077, 0.5092827677726746, 0.5435242056846619, 0.5429452061653137, -1.5213686227798462, -0.13657130300998688, 0.9759796857833862, -2.6152899265289307, 1.421507477760315, -2.2763733863830566, 0.32211971282958984, -0.16842640936374664, 1.6186552047729492, 2.0066146850585938, -0.5549736618995667, 2.9935302734375, -2.011350154876709, 9.419961929321289, -1.7592791318893433, 1.3442695140838623, -0.018814366310834885, 0.37722066044807434, -2.14993953704834, -0.3336851894855499, 1.7222918272018433, 1.235826015472412, 2.2453322410583496, 2.7442798614501953, 4.699699401855469, -0.3548617959022522, -0.06545081734657288, -1.8258979320526123, 0.664344847202301, 0.3919128477573395, -0.05183056741952896, -1.1166185140609741, -2.699305534362793, 1.0982333421707153, 1.4507864713668823, 1.6940009593963623, -3.4423203468322754, 1.7202430963516235, 2.410825252532959, -8.163824081420898, -1.5324509143829346, -0.09404989331960678, 0.9859398603439331, 0.03378778323531151, 2.112947463989258, -1.3626242876052856, 1.4876978397369385, -1.359071135520935, 0.3310108482837677, -0.5600379705429077, 2.764528274536133, -0.8072253465652466, 1.592980146408081, -0.9120073318481445, 1.092345952987671, 2.768190383911133, 0.582728385925293, 1.7097549438476562, -5.178632736206055, 0.571236789226532, 0.42551562190055847, -4.139233589172363, -1.6228795051574707, -1.3486429452896118, -1.1614259481430054, 0.32574498653411865, -2.3158957958221436, 8.598670959472656, 1.0206071138381958, 1.5841333866119385, 0.6961632370948792, 14.417704582214355, 17.594017028808594, 1.6184779405593872, -1.2493665218353271, 2.3746867179870605, -1.081094741821289, 0.32065778970718384, -1.1733676195144653, -1.076425552368164, -1.275403618812561, -0.5295649766921997, 0.6764664649963379, 1.8635283708572388, 2.001805305480957, -0.20837096869945526, -5.894463062286377, 2.8000011444091797, -2.325808048248291, 3.5864152908325195, -1.5580289363861084, 0.6972579956054688, -0.3846219480037689, 1.0834239721298218, 0.212600439786911, 0.3274518549442291, 1.7058091163635254, -1.069876790046692, -1.0828830003738403, -1.6494168043136597, -3.046022653579712, -0.2913048565387726, 0.5659634470939636, -0.26454538106918335, -1.1820896863937378, -0.6111513376235962, 1.4582127332687378, 2.1208701133728027, 1.3739120960235596, 0.046293411403894424, -1.2130974531173706, -0.30581456422805786, 1.2583494186401367, -1.069961428642273, 2.386209011077881, -11.992302894592285, 0.7535422444343567, -1.4758168458938599, -2.6551434993743896, 1.2272977828979492, 0.6746367812156677, -0.8406656384468079, 12.630292892456055, -0.8777244687080383, 0.040990907698869705, 0.21683725714683533, -2.087702512741089, 0.6202799677848816, -0.8869113922119141, 3.8324248790740967, -2.1254420280456543, 0.9729032516479492, -1.1403813362121582, 0.6767346858978271, 0.46189817786216736, 0.9388956427574158, 1.4194494485855103, -1.891643762588501, 1.1699550151824951, -0.9150350689888, -3.9095654487609863, 1.9201173782348633, -2.4835314750671387, 0.6298678517341614, 1.012387752532959, 0.5439800024032593, -3.2546544075012207, -1.309683084487915, 7.578368663787842, 13.062880516052246, 0.7374582290649414, 0.9210861921310425, 13.020840644836426, 0.5788490772247314, 1.5994311571121216, 0.8706969618797302, 0.5591035485267639, -3.097520112991333, -0.29718995094299316, -1.2584072351455688, 1.5643494129180908, 8.631843566894531, -1.2703677415847778, -0.6282532215118408, 1.142848014831543, 17.385644912719727, -1.32759690284729, -0.7835213541984558, 3.8910810947418213, -0.10766219347715378, 0.04821780323982239, -1.063123345375061, 1.0568170547485352, -0.7966784834861755, 4.621670722961426, 1.0722962617874146, -0.0899839699268341, -4.160974025726318, 0.20487244427204132, -0.979414701461792, 0.7308012247085571, 13.009573936462402, -1.3794076442718506, -1.2036548852920532, 0.08071594685316086, 1.4849809408187866, -0.7109917402267456, 2.579782485961914, 3.0257534980773926, -3.4758849143981934, 2.5867645740509033, -0.3938019871711731, 0.9952255487442017, -2.114788770675659, -1.410025715827942, -1.9258180856704712, -0.5884132385253906, 1.0506126880645752, -0.2683470547199249, 0.4070131778717041, -0.5725246071815491, -0.17912599444389343, 0.1552915722131729, 1.8436917066574097, -1.1698219776153564, -0.3016708493232727, -3.436014175415039, 3.0302627086639404, 2.561023712158203, -0.6671757698059082, -0.015616362914443016, -0.5550112128257751, -1.1824008226394653, 0.36296898126602173, 0.07357561588287354, -0.8246636986732483, 7.091264247894287, -0.057143185287714005, -3.292999267578125, -1.9134377241134644, 0.06830741465091705, 1.3692262172698975, -0.4901747405529022, -1.1247860193252563, 0.29242631793022156, 7.770019054412842, -0.07577474415302277, -1.624679684638977, 1.4506264925003052, -1.9989005327224731, -2.1491925716400146, 2.357591152191162, 1.2283090353012085, 1.6593074798583984, 7.4059672355651855, 1.0722646713256836, -2.166825771331787, -0.8958321809768677, -2.5543322563171387, -7.345780849456787, -2.393392324447632, 6.698224067687988, -2.9578657150268555, 0.3200069069862366, 1.6551746129989624, 0.5267711281776428, -0.9341350197792053, 1.4025803804397583, 1.1127057075500488, -0.787536084651947, 1.002286434173584, 13.135268211364746, 0.6776333451271057, -1.0332024097442627, -1.9659295082092285, -0.6619756817817688, -0.5588646531105042, -2.2335402965545654, 3.1987557411193848, -0.5572215914726257, 2.86537766456604, -4.3590617179870605, 4.386618137359619, -2.9041857719421387, 1.1568994522094727, -0.20162691175937653, -3.27327823638916, 11.289243698120117, -2.8157105445861816, -0.35151422023773193, 0.9722962975502014, 0.6226479411125183, -0.00882616639137268, 1.6831899881362915, -3.407275676727295, -6.138655185699463, 1.3957068920135498, 0.16133365035057068, 0.9642589688301086, 1.242174506187439, -1.5923199653625488, -3.883655071258545, 1.885501742362976, -4.1708221435546875, 1.0459587574005127, -0.15642096102237701, -1.8281230926513672, -3.7349982261657715, 3.5687007904052734, -1.402381181716919, 1.7904573678970337, 14.946128845214844, 0.3225196599960327, 3.71087646484375, -0.13080944120883942, 1.0167827606201172, -0.25166812539100647, -2.737975835800171, 1.8340654373168945, 2.083524703979492, -2.611656904220581, -0.9777504205703735, -0.32022520899772644, 0.6199613809585571, 1.5873931646347046, -0.6747208833694458, 3.314990997314453, 0.38091644644737244, -4.682052135467529, -0.8990704417228699, 0.8629684448242188, 0.09889063984155655, 1.3988722562789917, 0.21954308450222015, 1.1893587112426758, -1.040539026260376, -0.8178464770317078, 0.7766851782798767, -1.1110306978225708, -0.3868619501590729, -0.7858732342720032, -0.481161892414093, -1.8820815086364746, -1.378023624420166, -1.7428640127182007, 0.837445080280304, -0.6925152540206909, 1.0804569721221924, 3.935220241546631, -4.163699626922607, -0.3747361898422241, 0.0397946834564209, 1.8943027257919312, 13.18608283996582, 0.1462761014699936, -3.2909228801727295, 0.6030151844024658, -1.2978092432022095, -2.7462382316589355, -0.1336134821176529, -0.10586082935333252, -0.013447780162096024, -2.5305497646331787, 0.08141190558671951, 0.6792853474617004, 0.834293782711029, -4.787709712982178, 7.429385662078857, 4.939291954040527, 4.614419937133789, 1.6318634748458862, 0.8104661107063293, -0.5221148133277893, -1.8469704389572144, -1.283524751663208, 1.3873652219772339, -0.24878855049610138, -0.45030033588409424, -1.6677547693252563, -1.039001703262329, 0.19496703147888184, 1.3305230140686035, -2.846864700317383, -0.6394826769828796, -3.776458740234375, 0.2682926654815674, -3.7557263374328613, -2.566852569580078, -0.38288038969039917, 2.147747755050659, 0.5640425682067871, -1.9730427265167236, -3.7292981147766113, -1.217430830001831, -3.0461812019348145, -3.806746482849121, 1.207451343536377, 2.2043046951293945, -1.2939372062683105, 0.17098793387413025, -0.27659204602241516, 1.8157289028167725, 0.5017589330673218, 3.361455202102661, 0.10849371552467346, -0.6630505323410034, 0.9881903529167175, 1.3414579629898071, -0.5602185130119324, 0.9116241931915283, -3.6604490280151367, -4.174112319946289, 1.23200523853302, 0.5982239842414856, 1.317813515663147, 0.21069307625293732, 7.113238334655762, -1.4036226272583008, -1.555924654006958, -2.837358236312866, 1.5678850412368774, 0.29809704422950745, -8.443082809448242, 0.5258638858795166, 12.237152099609375, 0.3489437699317932, -0.01424400508403778, 1.6107052564620972, -1.0825448036193848, -0.5277855396270752, -1.1563018560409546, -1.3353948593139648, 3.4501290321350098, -1.23057222366333, 2.1638777256011963, -1.3275741338729858, 3.132518768310547, 2.2261886596679688, 2.8922224044799805, -1.5975234508514404, -4.15867280960083, -2.7985503673553467, 0.17386820912361145, 1.563797116279602, -2.7636566162109375, 0.6501471996307373, 0.31299424171447754, 1.4899516105651855, -1.1491488218307495, 1.7585952281951904, 0.1048852875828743, -1.0932620763778687, -2.1910367012023926, -0.8523891568183899, 0.09788403660058975, -0.9145640134811401, -4.173070430755615, 0.3744446039199829, -3.0851125717163086, -1.2252646684646606, 0.13697879016399384, 0.8944554328918457, 3.1644206047058105, -2.504817485809326, -12.13785171508789, -1.5176780223846436, 1.2223013639450073, -2.043121814727783, -0.7886286377906799, -2.6219077110290527, -0.6506257057189941, 0.29532286524772644, -2.113077402114868, 1.0208849906921387, 0.12468819320201874, -5.559260368347168, -0.5519325137138367, 0.19354386627674103, -0.36577141284942627, -0.545836865901947, -1.2671788930892944, -0.9187102317810059, -0.31804272532463074, 2.369786500930786, -0.16775330901145935, -2.0488908290863037, 1.1911380290985107, 0.05766020342707634, 3.9716122150421143, 1.1262587308883667, 1.1032625436782837, -0.18697994947433472, -0.8742413520812988, -0.25124305486679077, 0.9569997191429138, -1.4598689079284668, -1.2176344394683838, -1.0583083629608154, -0.2553427517414093, -7.531976699829102, 3.905709743499756, -2.9160268306732178, 0.0947176069021225, -1.1980761289596558, -0.7864871621131897, 3.7845215797424316, -0.6058077812194824, 1.9610836505889893, 0.29261133074760437, -8.970264434814453, -1.9892469644546509, -0.9408935904502869, 0.7435252666473389, -0.15206581354141235, -0.3945391774177551, 0.037401244044303894, 0.1325119435787201, 0.36614280939102173, 1.8210198879241943, 2.721632719039917, 1.1516982316970825, -0.685907244682312, 3.24038028717041, 3.929565668106079, -0.014897262677550316, -0.6292614340782166, -0.5668293237686157, 1.4646495580673218, -6.075855255126953, -3.4521427154541016, 2.4485268592834473, 0.9732944965362549, 4.4023871421813965, 1.921269416809082, 0.09499155730009079, 0.901118278503418, 2.9423294067382812, 1.1123050451278687, 0.861312210559845, -0.2825799286365509, -6.1103925704956055, 1.2017929553985596, 0.3852260112762451, -1.7204625606536865, 1.4420160055160522, -0.2508293390274048, -0.5943554639816284, 0.7404612898826599, -1.2728952169418335, -0.9649221301078796, -0.17933525145053864, -2.258854627609253, -3.2619800567626953, 1.6051911115646362, 0.21699422597885132, 0.5086798071861267, -1.2349326610565186, -3.4959235191345215, -0.3928993344306946, 0.7224512696266174, 0.08456116914749146, 5.130569934844971, -1.242735743522644, -0.32722991704940796, -0.44884464144706726, 0.6274534463882446, 2.3939974308013916, -2.0431458950042725, 0.9039098620414734, -0.5726464986801147, 0.5720332264900208, 1.713987946510315, -2.1382763385772705, -1.204777479171753, 13.553522109985352, 0.2464456707239151, -1.1242191791534424, 0.11168485879898071, -1.3372905254364014, -1.0509296655654907, 0.39868664741516113, -0.11335081607103348, -0.7424482107162476, 0.16714265942573547, -0.34043553471565247, 0.9548432230949402, -2.4607882499694824, 0.2173793464899063, 2.1458094120025635, -1.470813274383545, -6.800738334655762, -0.11696738749742508, -1.14273202419281, -1.354886531829834, -1.1537002325057983, -0.8746223449707031, -5.804525375366211, 2.240396499633789, 1.3874425888061523, 0.7514165043830872, -2.6347429752349854, -1.429024577140808, -6.177170753479004, -0.7808339595794678, 0.16505326330661774, 1.222531795501709, 0.3146820664405823, 0.3103597164154053, 0.1656758040189743, 0.7976797223091125, -0.7733085751533508, 1.0573664903640747, -0.5892302989959717, -0.705098569393158, -1.0826261043548584, 19.81995391845703, 0.4920794367790222, -15.195415496826172, 2.107701301574707, 2.576544761657715, -0.7093226909637451, 0.6628867387771606, -0.5583319067955017, -4.154606342315674, 3.421644449234009, -0.7888485193252563, 0.716478168964386, -1.3886100053787231, -1.1465555429458618, 4.561147212982178, 1.2589091062545776, 0.9533826112747192, -2.2946090698242188, 1.0591589212417603, 0.9235716462135315, -3.306637763977051, 0.6602817177772522, -0.3757168650627136, -3.5015718936920166, -2.9541985988616943, -1.7372510433197021, 5.923753261566162, 0.9714547395706177, -0.007760883308947086, 3.789358377456665, 1.9309605360031128, 0.5015032887458801, 1.953831434249878, -0.4432414472103119, 1.2733454704284668, 3.531306505203247, 0.1628313809633255, 0.01974206231534481, -2.7458930015563965, 8.596117973327637, 0.09754686802625656, -0.43865975737571716, 1.2259340286254883, -0.017240332439541817, 2.7281811237335205, -6.1253180503845215, 0.38602954149246216, -0.8780356645584106, -1.1581814289093018, -2.8959972858428955, -0.7909684181213379, 1.912630319595337, -0.702717125415802, -1.7064099311828613, 0.9194473624229431, -1.768117904663086, -0.4229731559753418, -1.769819974899292, -1.464800238609314, 1.0218755006790161, 1.2224788665771484, -5.242143154144287, -3.6930134296417236, 1.262813925743103, -0.9462432861328125, 2.304891586303711, -1.3040825128555298, 3.526780843734741, -0.6257380843162537, 1.3034467697143555, 0.3104633092880249, -24.838085174560547, -0.22749364376068115, -0.09018434584140778, 1.9974521398544312, 1.415384292602539, -3.779160261154175, -0.4035964608192444, 13.598915100097656, 4.591667652130127, -2.4619877338409424, -0.9611310362815857, 9.992147445678711, 4.492584705352783, -0.03221702575683594, 2.1651878356933594, -0.6638089418411255, -4.514800548553467, -2.125983715057373, -1.2552745342254639, 2.2148430347442627, -0.8844290971755981, -2.9841508865356445, 0.9024642109870911, 0.2784998416900635, -1.0669751167297363, -2.254023551940918, 0.7320044040679932, -0.9920526742935181, 1.3474633693695068, 0.17128492891788483, 1.8791154623031616, 2.350141763687134, 3.788346529006958, -0.9623979926109314, 2.237241268157959, -2.567401647567749, 0.501437783241272, -1.9835892915725708, 2.672062873840332, 1.1010985374450684, 2.9064762592315674, 0.4951103627681732, -0.8779314160346985, -0.3640759289264679, -1.0390582084655762, -0.5700423717498779, -0.4920820891857147, 1.0938003063201904, 0.10438086092472076, 1.6668062210083008, 0.32877010107040405, 0.3371735215187073, 0.19006098806858063, 1.2299617528915405, -12.120013236999512, 1.0171583890914917, 2.619037389755249, 1.4720302820205688, 0.2228473722934723, -2.500655174255371, 1.1490888595581055, -1.272864818572998, -0.01456945389509201, 4.6241068840026855, 0.7729244232177734, -3.287841796875, -0.112235426902771, 0.42508769035339355, 1.0062516927719116, -2.7431752681732178, 0.5728489756584167, 1.1358497142791748, -0.3124163746833801, -5.086189270019531, -3.1480157375335693, -0.3973705768585205, 0.5702299475669861, -3.1190643310546875, 0.9930959343910217, 0.42951053380966187, 1.504489779472351, -1.8460168838500977, -2.043159246444702, -3.659355640411377, 1.6652616262435913, 1.7246465682983398, -0.058068860322237015, 1.0354328155517578, 2.3103013038635254, -2.0567123889923096, 5.760321140289307, -0.96439129114151, -0.8795811533927917, 0.4062468111515045, 1.9886236190795898, 3.2278106212615967, 0.4547640383243561, -3.971097707748413, 0.9786465167999268, 0.29623696208000183, 1.6294960975646973, -1.7161566019058228, -0.8203081488609314, -2.265803337097168, -0.25534865260124207, 0.26825380325317383, 4.465428829193115, -1.043531894683838, -5.635164737701416, 0.26082322001457214, -0.7038417458534241, -1.0203732252120972, -1.5212310552597046, 0.09758787602186203, 0.33531397581100464, -2.972107172012329, 0.10220262408256531, 4.577102184295654, 0.5018919706344604, -1.3307147026062012, -0.37045952677726746, -0.3825172185897827, 1.532504677772522, 0.5606686472892761, 1.52993905544281, 2.924867630004883, -0.20565016567707062, 0.9970903396606445, -9.385089874267578, 3.194213390350342, 1.5160199403762817, 0.7848244309425354, -0.872155487537384, -13.163066864013672, 0.24498651921749115, -3.0517098903656006, -1.3600798845291138, 0.24920019507408142, -0.4007953405380249, 2.113969087600708, -5.482168197631836, -4.168430328369141, -2.549164295196533, -0.08491909503936768, 8.004806518554688, 6.771958827972412, 1.2507141828536987, 0.653290331363678, 2.378234386444092, -0.23259346187114716, 2.498901605606079, -1.3947197198867798, -0.47931191325187683, 2.4393105506896973, 2.651707649230957, 0.6083922982215881, -4.17291784286499, -2.8094959259033203, -0.1399552822113037, -3.2820615768432617, 1.5967909097671509, 1.3128241300582886, 1.0799411535263062, -1.075899600982666, -3.9761807918548584, 3.2612545490264893, -1.1984925270080566, -0.9903793334960938, -1.0462490320205688, -2.4250926971435547, 0.4101952314376831, 0.8492657542228699, -3.850893259048462, -0.40583550930023193, -0.6473920345306396, 0.04102463647723198, -1.088060736656189, 1.5543593168258667, 0.7397722005844116, -3.909677505493164, 0.5403322577476501, -0.23494072258472443, -3.464771270751953, 1.733560562133789, -3.4048547744750977, -1.2183046340942383, 1.994423508644104, -0.299613893032074, -0.006932367570698261, -0.6324560046195984, -0.9255552887916565, 0.14960706233978271, -0.3305246829986572, -1.1664127111434937, 0.24109679460525513, 0.8209323883056641, 1.359421968460083, 0.4133138060569763, -0.4190114438533783, -0.7224079966545105, -2.926023244857788, -0.7905064821243286, 0.24748830497264862, -0.10361208021640778, -9.672889709472656, 1.267667293548584, 3.3710391521453857, 2.4605977535247803, -0.6691334843635559, -0.5481570363044739, 0.502711832523346, 8.644387245178223, -0.6237188577651978, -0.8776035904884338, 1.2728290557861328, 1.2440991401672363, 0.8220635652542114, 0.1355917751789093, -4.32846212387085, 1.919259786605835, -0.7364658117294312, 3.7870919704437256, -0.4822025001049042, -0.8753993511199951, -1.0465972423553467, -0.7811142206192017, -0.31911998987197876, -1.1833844184875488, -1.152005910873413, -0.6496403217315674, -2.8780357837677, -0.6895813345909119, 0.04233771562576294, 0.16543199121952057, 1.35548734664917, -1.5910625457763672, -4.179802417755127, -2.0408010482788086, 2.680915355682373, 1.3536713123321533, 9.254311561584473, -0.02768983505666256, -5.748735427856445, -0.13406968116760254, -0.3857715129852295, 0.4298948347568512, 1.1943734884262085, -0.7907118797302246, -0.36597740650177, 0.6802453398704529, -1.8069723844528198, 3.011024236679077, 0.31705278158187866, 0.23906336724758148, 2.3899600505828857, -0.9036231637001038, 4.247962474822998, -0.7966977953910828, 0.9571890234947205, -2.0180535316467285, 0.7730115056037903, -0.700890064239502, 0.9247149229049683, 0.6422211527824402, -1.5782575607299805, -1.2941962480545044, 2.4373698234558105, 1.4960771799087524, 0.16537341475486755, 0.21483823657035828, 1.7634243965148926, 12.330652236938477, 0.06863292306661606, -0.10585115104913712, 1.0762473344802856, 0.4677521586418152, 1.0802539587020874, -0.7230355739593506, 0.27470842003822327, -0.8609145879745483, -0.8765652775764465, 1.1875849962234497, 1.9403692483901978, -0.337787002325058, 0.5497727990150452, 3.887248992919922, 0.14078305661678314, 0.33078011870384216, -1.160676121711731, -1.0383259057998657, -0.4483436346054077, 0.6358484029769897, -2.113898515701294, -1.1167442798614502, -0.47702285647392273, -0.5445436239242554, 0.7146112322807312, 0.29160216450691223, -1.1599102020263672, 0.39702749252319336, -0.32791754603385925, 1.9015640020370483, -0.9785797595977783, -1.2355120182037354, -0.9796282649040222, 1.2420717477798462, 1.5742228031158447, 1.553983449935913, 0.23332582414150238, -1.0125876665115356, -0.36440980434417725, 3.1500062942504883, 1.02604079246521, 0.7089489698410034, 0.3454761803150177, 2.1203014850616455, 1.1033793687820435, 0.7466170787811279, -3.824970245361328, 1.2071129083633423, 3.414597988128662, -2.8745250701904297, -2.518101215362549, 0.8235204219818115, 0.44247734546661377, 4.170548915863037, -0.6660993099212646, 0.07973597943782806, -0.04567902535200119, -0.5911774635314941, -0.9256452918052673, 1.0282336473464966, -0.3135168254375458, -1.3578602075576782, 0.32655155658721924, -3.653177261352539, 1.2232264280319214, 2.4621872901916504, -0.7021870017051697, 1.3881311416625977, -10.917401313781738, 12.071199417114258, -0.1737193763256073, -0.8277888894081116, 1.4014848470687866, 4.960794925689697, 0.8824951648712158, -5.0392560958862305, -2.9325809478759766, 0.7242559790611267, 2.6035714149475098, 0.16089127957820892, 6.182252883911133, 8.052459716796875, 1.0494446754455566, 1.1989959478378296, -1.9190759658813477, 0.5922538638114929, 0.7208312153816223, -0.32368341088294983, -1.4017884731292725, 0.9653934836387634, -0.45438191294670105, -2.6029670238494873, -3.3726139068603516, 0.819398820400238, 2.332172393798828, -3.15529727935791, 0.4740466773509979, 1.39878511428833, 1.1733263731002808, -26.022886276245117, -2.3880863189697266, 2.929478883743286, -1.7160392999649048, 0.542759358882904, -1.171776533126831, 2.0174720287323, 7.152649879455566, -1.6925654411315918, 2.4313881397247314, 1.1119831800460815, 0.2823071777820587, 1.035402774810791, 2.333888292312622, -0.23091807961463928, -0.24121081829071045, -1.7801300287246704, 2.4336416721343994, 0.2495826780796051, -0.432757705450058, 1.4125655889511108, -1.0167427062988281, 0.5760860443115234, 0.8994002342224121, 2.4407923221588135, 0.056438542902469635, -0.5384153723716736, -0.8921093344688416, 4.374063968658447, 1.1958110332489014, 0.9933339953422546, -3.606765031814575, -3.8091986179351807, -0.8533616065979004, -1.3248735666275024, 1.1628592014312744, 0.43935084342956543, -0.2925052046775818, 0.774341881275177, 1.9944854974746704, 2.5444860458374023, -0.9580717086791992, -1.4086600542068481, -0.5637381076812744, 2.4646856784820557, -4.155472278594971, 8.218881607055664, -0.31500205397605896, 0.12143941968679428, -2.4936814308166504, 1.9027214050292969, 1.0594183206558228, 1.0799671411514282, -1.2481749057769775, -3.2574806213378906, 1.605432152748108, 3.58404803276062, -6.297763824462891, 1.0445001125335693, 0.2939647436141968, 8.609689712524414, -0.16624446213245392, 0.8422117829322815, -0.6857335567474365, -3.442030429840088, 0.9199988842010498, -1.264835000038147, 2.2977964878082275, 1.0292202234268188, -0.06646820157766342, 0.8177978992462158, 5.820914268493652, 1.6848421096801758, -0.1512349545955658, 2.4269134998321533, 1.1279510259628296, 1.1925108432769775, 2.375455379486084, -0.26629528403282166, 2.5836522579193115, 0.32518270611763, 0.8302361965179443, 0.5116357803344727, 0.5650371313095093, -0.44091781973838806, 8.611348152160645, 6.29775333404541, -0.09569861739873886, 0.30938684940338135, -2.086857557296753, -0.373663991689682, 19.275178909301758, -0.6364948749542236, -2.8537747859954834, -0.48446112871170044, -2.643622636795044, 0.6850154399871826, -2.1070396900177, 2.77182936668396, -4.162595748901367, -0.5777695178985596, 2.464568853378296, 0.766336977481842, -0.8068370223045349, -1.525963306427002, 0.31253811717033386, 2.1843347549438477, 0.4324914515018463, 5.532632827758789, 1.8262474536895752, -0.440367192029953, 1.742998480796814, -0.9225830435752869, -1.107500672340393, 0.33364957571029663, 1.1444233655929565, 0.05709380283951759, 2.4348087310791016, 0.07234574854373932, -1.3385449647903442, 2.0816681385040283, -21.241226196289062, 4.2831597328186035, -0.12594947218894958, 0.8554596900939941, -3.0873284339904785, 0.3773306906223297, 0.3643321692943573, 0.3938394784927368, 0.25982704758644104, -0.4693884551525116, -1.305403709411621, -3.981963634490967, -2.128016233444214, -0.79623943567276, 1.7910635471343994, 0.4520385265350342, -0.24944935739040375, -1.2091169357299805, 1.2756843566894531, -1.9890764951705933, -0.8183817863464355, -1.6280517578125, 0.9292734265327454, -2.955083131790161, 1.411204218864441, -0.31030601263046265, -0.5916934609413147, -1.1360684633255005, -1.7932206392288208, -36.07928466796875, -0.1747671663761139, 0.9444243907928467, -1.216597080230713, -3.064084768295288, -0.4660343825817108, -2.1393420696258545, 3.073754072189331, 8.577072143554688, -2.4853761196136475, -0.6694826483726501, 0.038432829082012177, 0.7518613338470459, -2.5326435565948486, 2.1851203441619873, -0.1177222728729248, 0.8256987929344177, 0.5956779718399048, -3.483778238296509, 0.5877083539962769, -3.456329345703125, 1.9553790092468262, 0.33689314126968384, 1.215187430381775, -0.8943562507629395, 0.19690407812595367, -2.1754252910614014, 1.0383824110031128, 0.2911519408226013, 1.1269679069519043, 0.0919114425778389, 1.0450453758239746, 2.0004260540008545, 0.19010798633098602, -0.2515462636947632, 8.594098091125488, -0.39923733472824097, -2.8105666637420654, -1.1243782043457031, -0.1271524727344513, 7.464242935180664, 1.736317753791809, 1.2437132596969604, -0.5721297264099121, 2.013587474822998, -0.04696712642908096, 3.005678415298462, -1.3027987480163574, -5.009091854095459, 11.471266746520996, 2.060213804244995, -1.6128381490707397, -0.4239708483219147, 0.1377749741077423, 2.862555980682373, 0.3961297273635864, -1.3875068426132202, -2.8283755779266357, -3.0955874919891357, 1.2410740852355957, 2.4101099967956543, 1.2287391424179077, 0.37335699796676636, 0.20854796469211578, 2.1677465438842773, -0.9841088652610779, -3.208413600921631, -0.6595472097396851, -0.8913383483886719, 1.6847026348114014, -0.9744511246681213, 0.26348957419395447, -1.5425878763198853, 9.3247652053833, 0.578281819820404, 0.9395446181297302, 1.898934006690979, 2.258840322494507, 0.3281487226486206, -0.012295205146074295, -2.436910390853882, 2.3029236793518066, 1.3159186840057373, -1.950998306274414, 1.7434602975845337, 7.5999650955200195, 0.8650473952293396, 2.9171416759490967, 0.5989957451820374, -0.836028516292572, -0.749385416507721, 0.5466294288635254, -2.0893967151641846, 3.7449474334716797, -0.8189502954483032, -1.0658586025238037, 14.01621150970459, 2.6623828411102295, 2.1619322299957275, 2.0259833335876465, -1.4012635946273804, -2.8144853115081787, -1.0229238271713257, 0.02376660704612732, -4.1706342697143555, -0.803419291973114, 0.3668759763240814, -0.28335821628570557, -4.157464504241943, -3.5860724449157715, -0.4673413336277008, -2.092580795288086, -3.9789206981658936, -1.613121747970581, 1.7183780670166016, 5.570288181304932, -0.9751041531562805, -0.7244731783866882, -0.1799037754535675, 7.384022235870361, -1.4469614028930664, -0.5440164804458618, -0.8158695697784424, -2.519754409790039, -1.0257495641708374, 2.157650947570801, -0.7807483673095703, -6.869840621948242, 0.17197690904140472, -1.2733145952224731, -42.779022216796875, -3.873563051223755, 1.4303110837936401, -3.0818371772766113, 1.0874789953231812, 1.0479201078414917, -0.5358151197433472, 0.4494180977344513, -0.9800886511802673, 2.2907254695892334, -0.5331263542175293, -0.8941680788993835, -2.918370246887207, -0.3049947917461395, 2.244859457015991, -0.7715074419975281, -0.5178191661834717, 3.699357509613037, -0.4457673728466034, -0.09861798584461212, 0.03211553022265434, -1.0777322053909302, 4.1906208992004395, 0.7769045829772949, 2.4776625633239746, 2.144526720046997, 1.6304980516433716, 0.8606383204460144, 1.0656418800354004, -0.4660903513431549, 1.8770402669906616, 1.3775699138641357, 0.8943192362785339, 0.4362521469593048, 0.4011565148830414, 0.5301568508148193, 13.737398147583008, -3.109994649887085, 1.6919597387313843, -1.9387706518173218, 0.23487284779548645, 0.7718257904052734, -0.7625955939292908, 0.7231461405754089, 0.3686903417110443, 2.882039785385132, 2.0129637718200684, -0.3072800040245056, 5.626606464385986, -0.015790646895766258, -1.1422181129455566, -0.7548468708992004, -2.6490609645843506, -14.191043853759766, -1.458050012588501, 0.7481551170349121, 2.6751503944396973, 4.711084365844727, -2.9247474670410156, -1.471025824546814, 2.6083967685699463, 2.807516098022461, -0.841620683670044, 2.2507524490356445, -6.126681804656982, -0.1963645964860916, -1.8642603158950806, -1.9602959156036377, 0.9671957492828369, -0.4142639636993408, 0.866380512714386, 1.6236834526062012, 2.882858991622925, 0.8697883486747742, -0.4901881515979767, -0.527550220489502, 0.13705258071422577, -3.659841299057007, -1.583605408668518, 0.9361149072647095, 4.516128063201904, 0.810197651386261, 1.5153944492340088, 2.6604020595550537, -1.2996841669082642, 2.1522817611694336, 1.932211995124817, 0.7172393202781677, -0.9734439849853516, 0.9866287708282471, 0.26970013976097107, 2.669651508331299, -1.1216627359390259, -2.3747191429138184, -0.3759123682975769, -0.5733547806739807, 0.42108601331710815, 0.17576123774051666, -1.3036236763000488, -2.2361645698547363, -0.21603451669216156, 2.543234348297119, -2.928788900375366, 0.24067099392414093, 0.017692649737000465, -1.9471290111541748, 0.59427410364151, 0.8717749714851379, 1.8751311302185059, -0.5301979780197144, -0.09595098346471786, 1.6950361728668213, -4.9023542404174805, 0.07235320657491684, -0.9378708600997925, 0.3364049792289734, 1.9866608381271362, 1.3496394157409668, -3.8111751079559326, 2.000854253768921, -1.1105108261108398, -0.9443976283073425, -0.09231323003768921, -1.0158640146255493, -1.0269241333007812, 0.32563096284866333, -1.5471826791763306, -3.1529173851013184, -2.4335246086120605, -0.3095966577529907, -1.7309221029281616, -1.414186716079712, 0.5055664777755737, -0.920505940914154, -3.7444615364074707, -3.8354413509368896, 0.8697516322135925, -0.35674548149108887, -0.1103191077709198, 1.5988985300064087, -2.5351552963256836, -0.16486738622188568, -1.51845121383667, -2.100281238555908, 0.5296784043312073, 3.255455493927002, -0.8368998169898987, -4.855983734130859, 0.29831618070602417, -4.2946014404296875, 1.3734078407287598, 12.279878616333008, -0.06418512761592865, -0.5580313205718994, -0.07295461744070053, -1.1061149835586548, -2.1532416343688965, 1.8994563817977905, -1.3347256183624268, -1.4933255910873413, -0.6350343227386475, -1.1546651124954224, -3.860908031463623, 0.6338693499565125, 0.02999945357441902, -2.363098621368408, 1.2725261449813843, -3.436051607131958, -0.6778818964958191, 3.922964334487915, 5.6029534339904785, 2.822260856628418, -5.058166980743408, 0.14245718717575073, 0.6449503898620605, 1.2897025346755981, 2.83113694190979, -5.237325668334961, -0.5510178804397583, -1.1158602237701416, -0.6146190762519836, -0.5646737217903137, -1.190495252609253, 0.39502254128456116, 2.9546329975128174, -0.0479692742228508, 0.8986926674842834, -0.13923262059688568, -1.1696127653121948, 1.26449453830719, 0.2746403217315674, -0.40201568603515625, -4.148654937744141, 2.1710166931152344, -0.5355424880981445, 3.375152111053467, 0.30278292298316956, 0.24825264513492584, 0.5958206653594971, 0.21083424985408783, -0.9270709156990051, -0.2625885307788849, -3.481113910675049, -1.4055181741714478, -0.8695167899131775, 14.84913444519043, -5.016049861907959, -1.2713725566864014, -1.423295497894287, -0.7993444800376892, -0.8459535837173462, -0.22147978842258453, -1.0577178001403809, 5.061602592468262, -1.8607927560806274, -3.2634568214416504, -0.4286598265171051, 1.24954354763031, 2.776050090789795, 1.748103380203247, 3.521472215652466, 1.7606838941574097, 0.934532880783081, -1.5543980598449707, -0.12139219790697098, -5.771076679229736, 1.0500706434249878, -1.1504852771759033, 0.21445658802986145, -0.7850315570831299, 0.06504549831151962, -1.250255823135376, -1.5042893886566162, -1.4138126373291016, -0.6794252991676331, -0.2961263656616211, 0.1921050101518631, 1.1555002927780151, 1.05971097946167, 2.1912996768951416, -0.4350322484970093, -1.7463551759719849, 13.060309410095215, 1.8068420886993408, -1.4509251117706299, 12.34697151184082, 1.1910971403121948, 3.082556962966919, 1.1247893571853638, 0.9643981456756592, 0.9846603274345398, 1.8578121662139893, 0.31353050470352173, -1.4677045345306396, -0.36512061953544617, 1.567878007888794, 0.07964284718036652, 2.22699236869812, 2.155902624130249, -4.184319496154785, -4.5715179443359375, -0.5582050085067749, 0.0057698520831763744, 0.10286472737789154, -1.1145002841949463, 0.1512569934129715, -0.7906912565231323, -4.948296546936035, -0.5833613872528076, -1.3345950841903687, 3.3057851791381836, -1.0546659231185913, 2.695913076400757, 0.012954752892255783, -3.404232978820801, -1.702748417854309, -0.7040027976036072, 0.0025453015696257353, -0.7878634333610535, -2.5065040588378906, -0.5142307281494141, 2.9915716648101807, -0.8751329183578491, -2.8790667057037354, 0.36451682448387146, -0.14660660922527313, 1.3460670709609985, -3.630613088607788, -3.6446492671966553, -4.169246196746826, -0.5211388468742371, 1.711604356765747, -0.5114903450012207, 2.276082992553711, -1.083626627922058, -0.3922271728515625, -0.4599372148513794, 1.3962125778198242, 0.12206882238388062, 1.373822569847107, -1.2369565963745117, -3.627077102661133, 1.6370291709899902, -4.676068305969238, 4.217479228973389, -0.7893117070198059, 0.9363546967506409, -0.5611104369163513, -3.0933992862701416, 1.5185213088989258, -0.040370311588048935, 1.8246986865997314, -4.207671165466309, -0.5168889164924622, 1.4820002317428589, -2.788616418838501, 0.2045479714870453, 0.16272437572479248, 0.2723027765750885, -1.1706657409667969, -2.7900218963623047, -4.958742618560791, -0.7336220145225525, 6.277759552001953, 4.950995445251465, -0.017280595377087593, 0.8798536658287048, -2.442086696624756, -1.0506927967071533, -3.101858139038086, -0.6645298004150391, 2.7430973052978516, 1.3147599697113037, 1.2912266254425049, -1.3536771535873413, 1.4089645147323608, -1.021265983581543, -3.9315028190612793, 0.9760892987251282, -1.9546256065368652, -3.339656352996826, -9.404409408569336, -3.941032648086548, -0.19757211208343506, 2.656203269958496, 1.1020797491073608, -0.30468741059303284, -2.0801455974578857, 1.8420253992080688], "xaxis": "x", "y": [0.2077058106660843, 0.2573804259300232, -3.1430773735046387, -2.2241744995117188, 0.12088927626609802, -1.042863130569458, 1.5422999858856201, -0.016214493662118912, -0.7188573479652405, -1.8822942972183228, 0.4322144687175751, 0.558271050453186, 1.8164384365081787, 7.689966678619385, -0.9419658780097961, 1.4749106168746948, -1.348613977432251, -0.3478353023529053, -3.408193826675415, 0.7322423458099365, 2.4943408966064453, -9.429533004760742, -2.266382932662964, 0.618040144443512, -1.6761541366577148, 0.9558416604995728, -2.99125337600708, -1.4966421127319336, 0.5086115002632141, -0.34551993012428284, 0.24535423517227173, -1.2640821933746338, 0.6171749830245972, 1.1853563785552979, -1.3651270866394043, -0.7952147722244263, -0.7553468346595764, 1.235802173614502, -1.9524739980697632, 0.7094086408615112, 1.172092080116272, -0.32151585817337036, 0.5321817398071289, -3.122312307357788, 2.3163797855377197, 0.9074825048446655, 0.6170292496681213, 2.239898681640625, 4.456398963928223, 4.667257785797119, 0.8757294416427612, -0.4965679347515106, -1.2327271699905396, -0.6527763605117798, -2.2727489471435547, 7.35446834564209, -0.15391843020915985, -0.7135658264160156, -1.4649310111999512, 0.7014877796173096, 1.1654983758926392, 2.170074939727783, 2.0878825187683105, -6.275125980377197, 0.3459593951702118, 0.13438713550567627, 0.675822913646698, 0.0960206687450409, 1.2768760919570923, 0.4649510681629181, 0.16377554833889008, -0.9382551908493042, -1.5400807857513428, 3.406928539276123, -1.0336776971817017, 1.6581085920333862, -0.26529461145401, 0.800574004650116, 0.45146873593330383, 1.2685967683792114, 4.73373556137085, 0.9289870858192444, -1.5187036991119385, 0.3218528628349304, -6.943292140960693, -5.652738571166992, -1.0311541557312012, 0.9170351624488831, 1.0298795700073242, 2.0234148502349854, -1.2246525287628174, 0.4376968741416931, -0.9012154936790466, -0.02451905980706215, 0.7418510913848877, -1.2038897275924683, 7.903906345367432, 0.20961235463619232, -5.133587837219238, 5.591798782348633, -1.9054691791534424, -0.5246081352233887, -2.1869163513183594, -0.19567470252513885, -0.6217381954193115, 0.21186307072639465, 0.16475825011730194, 0.5645017027854919, -2.9449241161346436, 3.008338212966919, -3.226773500442505, -1.268241286277771, 0.144888773560524, 1.8283742666244507, -0.2117341160774231, 0.6431436538696289, -2.8455660343170166, 4.41759729385376, 0.7469999194145203, 2.1767849922180176, 4.4466633796691895, 1.0961780548095703, -1.2513164281845093, -0.2589230537414551, 1.55889093875885, 0.8166386485099792, 0.16284547746181488, 2.4811434745788574, 5.8753509521484375, -1.1703964471817017, -0.12597845494747162, 0.3222350478172302, -4.008538246154785, -0.8038938045501709, -1.6623791456222534, -0.04695318639278412, 1.3832910060882568, -0.08133385330438614, 2.9016079902648926, -0.29950928688049316, -3.1692233085632324, -12.016539573669434, 0.5395141243934631, -1.7132411003112793, 0.94866943359375, 4.374783992767334, 1.0965723991394043, -0.9580679535865784, 0.5262263417243958, -0.8091884851455688, -1.9257824420928955, 1.0256081819534302, 1.6652151346206665, 1.7344517707824707, -0.24576611816883087, -0.525968074798584, -2.722625494003296, -7.973353862762451, 0.10053875297307968, -2.2764227390289307, -8.022192001342773, 1.8130524158477783, -4.986536979675293, -2.9340388774871826, 1.5010759830474854, -0.37220093607902527, -5.916852951049805, 1.745969533920288, -3.06904673576355, 0.5639781951904297, 1.1837645769119263, -3.241812229156494, -0.8013420701026917, 0.9122939109802246, 0.6995996236801147, 0.9227816462516785, -1.5398532152175903, -1.1364483833312988, 1.206555724143982, 0.6111651062965393, 0.7863142490386963, -1.1858298778533936, -3.833561658859253, 3.282599687576294, -2.6809704303741455, 4.374485969543457, -0.013546542264521122, 1.3531237840652466, 7.9061784744262695, -0.6021556258201599, -0.7664052248001099, 1.6543569564819336, -1.4765411615371704, -0.8628271222114563, -0.5355706810951233, -0.36902040243148804, 0.007684013806283474, 3.813459873199463, 1.1273473501205444, -2.0453040599823, 1.2095521688461304, -2.0175111293792725, -8.036864280700684, 3.3206236362457275, -0.617996871471405, -0.7176762223243713, 1.9400975704193115, 0.04309052228927612, -2.3178515434265137, 0.337660551071167, -0.15055324137210846, -3.7611770629882812, 12.819714546203613, -0.5394697785377502, 0.8827188611030579, -2.020772933959961, 4.368084907531738, -0.05157075449824333, 6.669456958770752, 0.5702866911888123, -2.5442299842834473, -0.8631864190101624, 1.6027934551239014, 0.5795363187789917, -0.8164517283439636, -0.15269580483436584, 1.5838974714279175, 0.7210145592689514, -0.8827081322669983, 0.13105838000774384, -1.787510871887207, 0.3907686769962311, -1.352739930152893, -1.5614465475082397, 1.9667119979858398, -1.1548216342926025, 1.0448040962219238, -0.9645518660545349, -0.09789593517780304, 0.659625768661499, 0.5565515160560608, 2.3576667308807373, -2.281834602355957, 1.3542678356170654, -1.812151312828064, 1.7582505941390991, -0.7486036419868469, -0.4218805134296417, -2.02970552444458, -8.034255981445312, -1.1606504917144775, -1.2194528579711914, 0.528078556060791, 0.576806366443634, 0.8853712677955627, 2.326101064682007, -1.5764375925064087, 5.300368309020996, -2.308140516281128, -1.1581313610076904, -4.3001532554626465, 1.637180209159851, -3.5438268184661865, -0.6027573347091675, -0.02870924025774002, -0.7220391035079956, -1.3991498947143555, -1.0184245109558105, 0.3336739242076874, -0.6938247680664062, 0.6295326948165894, -2.2507739067077637, -0.14552052319049835, 1.2632637023925781, -1.4099313020706177, 2.152853012084961, 0.5506981611251831, 2.4852795600891113, -0.6435307860374451, 0.4616455137729645, -1.091825008392334, -1.293180227279663, -2.6751668453216553, -1.5787591934204102, -0.6493000984191895, 2.0116257667541504, 0.9679278135299683, -1.463604211807251, -0.6059660315513611, -0.24433863162994385, -1.7957735061645508, 0.9996258020401001, -3.1762282848358154, -0.3461422026157379, -0.9563391208648682, -0.5246163606643677, -0.7957665920257568, -1.6062889099121094, -0.5370206236839294, -0.13708792626857758, -1.1602516174316406, -0.19098618626594543, 0.9241777062416077, -3.96955943107605, -2.0940794944763184, -7.737768650054932, 0.9423962235450745, 0.6515056490898132, -0.5333700180053711, -2.3451027870178223, -1.0127562284469604, -0.17084665596485138, -2.1617066860198975, -2.170208692550659, 12.546286582946777, 0.6352417469024658, 0.938541829586029, -0.10975756496191025, 0.4742903709411621, -0.7998462915420532, 0.43941304087638855, -0.6516421437263489, 0.7636762857437134, 4.252892971038818, -0.018354985862970352, -1.572710394859314, -2.026153326034546, 0.18189741671085358, 0.2860685884952545, 0.39319542050361633, -0.7632044553756714, 1.301156759262085, 0.6953455209732056, 1.1437392234802246, 1.4167542457580566, 0.167239248752594, 0.1438535898923874, 0.8996446132659912, -0.8275398015975952, -0.7004231810569763, 4.456972599029541, 0.3195382356643677, -0.396087110042572, -1.1661306619644165, -2.910687208175659, -0.28247925639152527, -1.1927289962768555, 0.2661961615085602, 1.2729132175445557, -0.7336859107017517, -1.3921805620193481, 4.254130840301514, -3.7999868392944336, 4.643223285675049, -3.604428768157959, -0.7282466292381287, -0.5711193084716797, 0.3508515954017639, -3.671611785888672, 1.4749301671981812, -0.19847393035888672, -2.47552752494812, 0.4143689274787903, -1.7457058429718018, -0.8099949955940247, -1.7578296661376953, 1.0612698793411255, 0.005365964025259018, 0.3228745460510254, 0.782903790473938, 0.2166144847869873, 0.20870299637317657, 0.04282446950674057, 8.328487396240234, -0.6148644089698792, -0.3825879693031311, -0.5924960374832153, -1.0837948322296143, 1.5075864791870117, 0.459203839302063, -2.0163936614990234, -1.5934735536575317, -0.5186340808868408, -1.1619446277618408, -0.49896949529647827, -0.8402580618858337, 2.1190881729125977, 2.5319554805755615, 1.047622561454773, -4.3439812660217285, 0.6754822134971619, 0.7482684254646301, -7.194472789764404, -1.6914345026016235, -0.175747349858284, -4.926080703735352, 2.004638910293579, 0.9896880388259888, 0.25465041399002075, -1.372087836265564, 5.5873565673828125, 1.039588451385498, 1.3765220642089844, -0.5885191559791565, -0.1507040113210678, 0.668727695941925, 0.2853810787200928, 0.9103682041168213, -1.0204756259918213, 0.057687338441610336, -1.0471127033233643, 0.5823469161987305, 0.3218972086906433, 1.0393651723861694, 1.4537068605422974, -4.341999530792236, -0.35915327072143555, -0.5677992701530457, 1.0003211498260498, -1.867515206336975, -1.4421947002410889, 1.3397575616836548, 4.246777534484863, -0.320333868265152, -0.3695417642593384, -1.1587681770324707, -0.702976644039154, 0.7089619636535645, -1.1524591445922852, -1.3794008493423462, -0.7508220672607422, 1.1822153329849243, 0.29313525557518005, -0.6321282982826233, -9.536430358886719, 0.28091034293174744, -1.9409451484680176, -0.7010201811790466, 0.01554599218070507, 1.4225964546203613, 0.4052530825138092, -1.7390888929367065, 10.663639068603516, -0.9849894046783447, 17.785585403442383, -1.4038784503936768, -0.2595146596431732, -0.7874969840049744, -2.8475751876831055, -0.027500856667757034, 2.564680576324463, -1.1583231687545776, 1.010230302810669, 0.6666476726531982, 3.1320667266845703, -0.876995861530304, 4.337671279907227, 2.0112595558166504, -2.1555027961730957, -0.9308987259864807, -0.5086324214935303, 1.2096116542816162, 3.2995872497558594, -0.780073344707489, -0.9528372883796692, 0.7772385478019714, -1.423733115196228, 0.8813069462776184, -0.7124682068824768, -2.895413398742676, 5.571341037750244, 1.3288966417312622, -2.098980188369751, -1.1166266202926636, -2.159351348876953, 3.3246984481811523, 2.4260287284851074, 0.4750232994556427, 0.038465846329927444, 4.3577046394348145, -2.673670768737793, -0.7397605180740356, 5.590284824371338, 0.9751583933830261, 1.4855905771255493, 0.2637658417224884, -1.866551399230957, 0.2172558754682541, -5.669066905975342, -2.1894302368164062, -2.0687615871429443, 0.7514170408248901, -3.166532278060913, -0.17619742453098297, -0.6808212399482727, 0.9261960983276367, -0.0754823312163353, 0.13575255870819092, 0.2820255756378174, 0.7136000394821167, -0.12505631148815155, -1.085320234298706, -5.373544216156006, 1.0510419607162476, 4.667971134185791, -1.312734603881836, 0.9614342451095581, 0.0505177266895771, 1.3560677766799927, 9.907270431518555, -1.161519169807434, 1.7804173231124878, -2.4350907802581787, 3.75022292137146, -0.21517522633075714, 1.5371172428131104, -0.19661785662174225, -8.029953002929688, 11.447463035583496, -0.8352862596511841, -1.232709527015686, 1.6282435655593872, 0.19135092198848724, -0.1700586974620819, 1.430878758430481, -0.9125667214393616, -2.685659170150757, -1.2330888509750366, -1.771095871925354, -0.03450879827141762, -1.8364126682281494, 0.6318458318710327, -3.6896233558654785, -1.1160664558410645, -1.9288283586502075, -3.447209358215332, 0.35134032368659973, -0.06279651820659637, 0.16341374814510345, -1.178587555885315, -2.1089351177215576, 1.7469230890274048, 0.9804316163063049, 1.5107353925704956, -1.9297722578048706, -1.5947365760803223, -3.188431978225708, -2.2908425331115723, -1.5362639427185059, -0.11622373014688492, -1.8380130529403687, -1.4457706212997437, 0.05292552709579468, -4.265040397644043, 3.074930191040039, 0.37833282351493835, 0.5712530612945557, 0.46424400806427, 4.118295192718506, 2.026136636734009, 0.5594004392623901, -2.7638659477233887, 0.439766526222229, -4.206833362579346, -1.824381947517395, 0.1800074279308319, 1.9379494190216064, 0.687934160232544, 0.47454801201820374, 2.4971563816070557, -4.08908224105835, 1.5307635068893433, 0.44775390625, -1.9132399559020996, -0.07029515504837036, 4.382658004760742, -1.6777645349502563, 0.9001789093017578, -9.32078742980957, -2.3855090141296387, -1.1596835851669312, 2.983678102493286, -7.397509574890137, -0.08039158582687378, 0.9993593692779541, -5.024434566497803, 15.870933532714844, -3.482422113418579, 0.8091955780982971, -1.3467707633972168, -0.6854760050773621, -0.5167232751846313, 14.140861511230469, -1.147207498550415, -0.5659001469612122, -0.6723652482032776, 4.281521797180176, -0.34283989667892456, 2.4587018489837646, 0.05098972097039223, -0.5599360466003418, -2.5321600437164307, -6.595518589019775, -0.32488423585891724, 0.22796867787837982, -1.5304912328720093, -1.1707671880722046, 1.190392255783081, -3.459718704223633, -1.9144436120986938, -1.1986724138259888, 0.38786378502845764, -3.267824649810791, -0.610148012638092, -0.39838311076164246, 0.21196843683719635, -2.931884527206421, 2.1366829872131348, -0.42342182993888855, 1.2987805604934692, -1.0383814573287964, -0.683556318283081, 2.021145820617676, 0.03865676745772362, 0.6300588846206665, -2.402432918548584, 0.9223666191101074, 0.8883028626441956, -0.6913650035858154, -7.973702907562256, -4.336177825927734, -0.1426362246274948, 0.95313960313797, 0.6832327842712402, -5.749691009521484, -0.6851036548614502, -8.045820236206055, -0.38335099816322327, -1.0030708312988281, -1.258556604385376, 1.0115607976913452, -0.34942159056663513, -3.2546417713165283, -0.634187638759613, -1.2058486938476562, 1.4302529096603394, 1.4683324098587036, -2.532655715942383, 0.08653822541236877, -0.5788756012916565, -1.4053618907928467, -2.2845723628997803, 1.3868229389190674, -0.7503560781478882, -1.7636620998382568, 0.7664320468902588, -3.8909010887145996, -0.3167901635169983, -3.6440913677215576, 10.514622688293457, -0.6849193572998047, -1.2146623134613037, -1.2220035791397095, -1.158633828163147, -1.5483253002166748, -2.0031538009643555, 1.4385043382644653, 0.09700798243284225, -0.571151614189148, 4.366353511810303, -1.7430179119110107, -3.997345447540283, 0.7473793029785156, -2.7626936435699463, 0.8683425784111023, 0.22371983528137207, -0.5076099634170532, 0.6797515153884888, 0.8710156679153442, 0.7503920197486877, 0.05527333915233612, 2.7904229164123535, -0.6019473671913147, 4.31695032119751, -1.0763897895812988, -2.56600022315979, 0.09356168657541275, -0.14482200145721436, -1.1576907634735107, -1.3125741481781006, -1.8609532117843628, -0.15982834994792938, 0.18085435032844543, -6.41124963760376, 0.4012238383293152, -0.17013797163963318, 0.12327108532190323, -11.983420372009277, -0.5890935063362122, -0.21048426628112793, -7.9832305908203125, -3.017925500869751, -1.436997413635254, -0.27314847707748413, 0.5828069448471069, 0.6875275373458862, 1.1039859056472778, -8.009757995605469, 0.6868457794189453, -6.4078521728515625, -0.37888216972351074, 1.1862053871154785, 1.1545913219451904, 4.982967376708984, -1.1523587703704834, 0.42888155579566956, 0.45763108134269714, -0.41098693013191223, -1.1922873258590698, 0.6959312558174133, -2.506589889526367, 0.7301093339920044, 4.271833419799805, 0.12694068253040314, -2.459557294845581, -0.18397730588912964, 0.10333111137151718, -2.083505630493164, 0.48832249641418457, 1.109874963760376, -2.5127344131469727, 0.41711950302124023, -3.8841018676757812, -2.5366673469543457, 1.1620004177093506, 0.5951961874961853, 0.23008333146572113, -3.055692672729492, 1.610053539276123, -0.9839037656784058, 0.10492050647735596, -1.1997658014297485, 2.913724899291992, 0.6039336919784546, 0.8478885889053345, 0.06843987852334976, -4.336968898773193, 2.518623113632202, -4.215747833251953, -1.5787744522094727, 1.142717719078064, 0.3733508586883545, 19.961353302001953, 1.4796926975250244, -7.98227071762085, -1.5697299242019653, 0.24510101974010468, -0.770326554775238, -1.6558794975280762, -4.249882221221924, -0.9095993041992188, -1.3611187934875488, 0.18172194063663483, -0.8361145853996277, 0.5614444613456726, -1.2288933992385864, 0.25714510679244995, 7.469139575958252, -0.34126409888267517, 1.3104174137115479, -0.05275480076670647, -9.229068756103516, 0.0919504314661026, -1.5169199705123901, 4.650688648223877, 0.6705286502838135, 0.8557919859886169, -2.295668601989746, -1.8742607831954956, -0.6528300046920776, -1.2626742124557495, 1.1176772117614746, 0.17056891322135925, -1.2045514583587646, 1.5137889385223389, 0.8643948435783386, 0.7432489395141602, 1.184496521949768, -0.3493576943874359, 0.5836226344108582, -6.586486339569092, -4.44574499130249, 0.1649487018585205, 0.19764064252376556, 0.2195776253938675, -1.736601710319519, -1.7241835594177246, -2.163546085357666, -1.1982676982879639, -1.1905195713043213, 0.5850059986114502, -0.7273569107055664, 1.663525104522705, -1.1987204551696777, 1.655210018157959, 1.6133160591125488, -0.511978030204773, -1.2024011611938477, -0.1837303191423416, 0.4041651487350464, 0.8053558468818665, 1.0925930738449097, -0.8270332217216492, 14.85264778137207, -1.085086703300476, 0.24455595016479492, -2.797739028930664, 0.022578377276659012, -1.6901957988739014, -2.3587772846221924, -3.2682571411132812, -1.1263599395751953, 2.0920770168304443, 0.5428464412689209, 0.42207834124565125, 1.9816005229949951, 0.44860148429870605, -0.8754090070724487, 0.12309530377388, -2.6193456649780273, -2.0931878089904785, -7.758934020996094, 0.19801414012908936, -3.814424514770508, -0.9841822385787964, -0.7813014984130859, -0.522273600101471, -1.484135627746582, -0.43796512484550476, 14.773788452148438, 0.12571871280670166, -1.6476349830627441, 0.03650175407528877, 0.9375488758087158, -0.1903018355369568, 0.5029593110084534, 4.370388031005859, 0.7395036816596985, -0.9594345092773438, 1.7481567859649658, 0.8661336898803711, -0.9121979475021362, -0.19374065101146698, -5.914802551269531, 0.8400330543518066, -4.931289196014404, -0.7823902368545532, -1.6253654956817627, 0.9313699007034302, 0.29092246294021606, 4.352268695831299, -0.078656867146492, 0.1250656396150589, 2.2169811725616455, -0.835818886756897, -4.625673770904541, 4.296284198760986, -2.988839864730835, -4.347015857696533, 2.130192518234253, 1.1456947326660156, 0.8046831488609314, 0.6572172045707703, -1.3580843210220337, -3.348343849182129, 2.6528069972991943, 1.2041380405426025, -1.425057053565979, 2.2682650089263916, -1.3721851110458374, -5.917152404785156, 0.5225951671600342, -1.6689369678497314, 3.213815689086914, 0.7453369498252869, 0.6649414300918579, -2.845015525817871, -0.6218008995056152, 1.7474427223205566, 0.9927518367767334, 0.5710640549659729, -1.3858153820037842, -0.2865283787250519, -0.04677356779575348, 0.22678719460964203, -1.8906216621398926, -1.4677252769470215, -1.3348259925842285, -7.98664665222168, 1.2005982398986816, 0.11664699018001556, 1.3287441730499268, 1.252661108970642, -2.2439465522766113, -2.13802433013916, -0.6368361115455627, 1.054723858833313, -0.9888423085212708, 1.1770827770233154, -0.3045872151851654, -1.5071910619735718, 3.082481622695923, -2.1751646995544434, -0.8325820565223694, -1.8732081651687622, -3.206495761871338, -0.35845744609832764, -1.2047077417373657, -0.16786594688892365, -2.4549472332000732, 0.6053978204727173, 0.5106938481330872, 1.6375216245651245, -1.5615953207015991, -0.329370379447937, -0.6464347243309021, 9.53249740600586, 0.3025834858417511, 0.02568092569708824, 0.4037337899208069, 0.6261882185935974, 1.136808156967163, -1.8390130996704102, 0.2599606215953827, -1.2731138467788696, -2.7085342407226562, 0.4279523193836212, 1.5025607347488403, 1.4432182312011719, 3.2781169414520264, 0.7414001226425171, 0.18173301219940186, -2.5386152267456055, 0.40785548090934753, -1.6536710262298584, -0.11731664091348648, 1.7910090684890747, -1.7651512622833252, 2.0032029151916504, 1.2064881324768066, 1.460712194442749, 0.8257151246070862, -0.027239961549639702, -2.163274049758911, -0.7616289854049683, -1.8320480585098267, -0.3152123987674713, -0.6345647573471069, 5.680789470672607, -1.9294641017913818, 0.9835708737373352, -1.4158653020858765, 0.6713454723358154, -0.3633311986923218, -2.424222230911255, -5.777261257171631, -3.1639018058776855, 1.0549498796463013, -2.2663094997406006, -0.30510413646698, 4.465402603149414, 1.98137629032135, 1.75826096534729, -0.3095186948776245, -0.5964867472648621, -0.05311635136604309, 1.7178595066070557, 4.286952495574951, 0.46707308292388916, 0.5990738272666931, 0.8392593860626221, -0.6100833415985107, 0.00536661921069026, -1.205947756767273, -1.8077808618545532, -1.2794798612594604, 0.335552841424942, -1.6669094562530518, -0.8393340706825256, 3.028003215789795, -0.9934252500534058, -6.416889667510986, 1.9920519590377808, -0.17777979373931885, 0.35484135150909424, -0.7757275104522705, 0.34032464027404785, -0.24784593284130096, -0.33488020300865173, 0.03127142786979675, -1.1619324684143066, -0.17261096835136414, 1.2569184303283691, -6.800449371337891, -1.2210323810577393, -1.6123837232589722, 9.670075416564941, -1.5713070631027222, -0.26856350898742676, -1.1982991695404053, 0.2301006317138672, -0.4942074418067932, -2.2379536628723145, -0.34388455748558044, -2.9779980182647705, -0.6254147887229919, 0.03791587054729462, -0.9370211958885193, 3.059751272201538, 2.64568829536438, 0.2592966556549072, -1.9281280040740967, 0.6787337064743042, -0.44661372900009155, 1.9327589273452759, -0.4131847620010376, 0.5044647455215454, 2.620511293411255, 1.1457512378692627, -2.351149320602417, 0.594721257686615, -1.8869749307632446, -3.010401964187622, 0.5263481140136719, -1.2446621656417847, 1.1295608282089233, -0.5342567563056946, -3.600856304168701, -0.8576834201812744, 1.0552409887313843, -1.0667752027511597, -0.2431769222021103, 1.5680477619171143, 0.7920973896980286, -0.0193384550511837, -0.6373183131217957, 0.8329272270202637, -0.7197852730751038, -0.43669959902763367, -0.9258803129196167, -0.657578706741333, -3.8228509426116943, -2.154449462890625, -1.8065704107284546, 0.23517094552516937, 0.052943892776966095, -1.7657228708267212, 4.361963748931885, -2.3615217208862305, -1.8156521320343018, 0.7906684279441833, -2.7437069416046143, 1.0402159690856934, -0.7484831213951111, -8.432342529296875, 3.731477737426758, -1.1594675779342651, -0.21649384498596191, -1.290825605392456, 10.434676170349121, 6.544594764709473, 1.4466317892074585, 0.7174191474914551, 1.4778050184249878, -0.9815983772277832, -3.437042474746704, 1.7203072309494019, -1.9179041385650635, 4.251433849334717, 0.728403627872467, -0.5653374195098877, -0.021050376817584038, -1.1645138263702393, 4.275457382202148, -0.9373512864112854, 2.050819158554077, -0.32069137692451477, -12.665823936462402, -1.1285113096237183, 0.3327232599258423, 0.8513818979263306, 0.6843851804733276, 4.368246555328369, 1.851933240890503, -0.43325188755989075, -1.1789387464523315, -0.8876048922538757, 1.0062956809997559, 2.373806953430176, -0.058694493025541306, -1.3247408866882324, -1.0793529748916626, -1.1950855255126953, -1.8591886758804321, 0.3571294844150543, 0.8616024851799011, -1.3491336107254028, -0.6049805879592896, 0.8067694306373596, -1.2894494533538818, -0.9374570846557617, 2.3595612049102783, 0.067000612616539, -0.8445028066635132, 3.331282138824463, -0.8990630507469177, -0.9195986390113831, -0.42944401502609253, 1.8736240863800049, -5.7825727462768555, 0.964668333530426, -0.2052573412656784, -0.5715537071228027, -2.2828121185302734, -0.8504974246025085, 0.5443122386932373, -1.0174492597579956, 1.6250004768371582, 0.06195858120918274, 0.47067791223526, 1.0468920469284058, -0.6923072934150696, 1.3391538858413696, 1.2624276876449585, -1.86692214012146, -0.8351832032203674, -3.7023322582244873, 0.7728806138038635, -1.714362382888794, 0.35711899399757385, -0.4630579352378845, -0.7764313220977783, 0.25659871101379395, -3.269501209259033, -0.4575423300266266, -2.218878984451294, 2.297773599624634, -3.6284825801849365, 1.8965110778808594, 0.6349446773529053, 0.8267074227333069, -0.7203279733657837, -5.077981472015381, -1.9105472564697266, 2.4937729835510254, -2.7644894123077393, -6.710886001586914, -0.15198087692260742, 1.7142176628112793, -1.3489657640457153, 8.361907958984375, -0.5443890690803528, -2.315767288208008, 1.8140872716903687, -6.468738079071045, 0.014578609727323055, -0.36927446722984314, 8.502991676330566, -0.5773423314094543, -4.443398952484131, 1.425364375114441, -1.23053777217865, -0.5664379596710205, -11.335758209228516, -0.2415006458759308, 1.4100347757339478, -11.565753936767578, 1.7220447063446045, -1.523589015007019, -0.9471976161003113, -2.102449417114258, -0.8403365015983582, 0.07083573937416077, 1.6213723421096802, -0.14179085195064545, 0.04999218508601189, -1.247746229171753, 0.34142833948135376, -2.089895009994507, -5.798367500305176, -2.5403454303741455, 1.4032199382781982, -0.4209161102771759, 1.0559028387069702, -1.2679626941680908, -0.9886882305145264, -2.2382357120513916, 0.6542182564735413, 0.45406121015548706, 1.7338546514511108, -6.573917388916016, -1.6326558589935303, -0.4230981469154358, 0.6941495537757874, 6.564408302307129, 1.173378586769104, 0.8847581148147583, -0.3976150155067444, -1.0611273050308228, -1.4989259243011475, 0.024998487904667854, -1.9752204418182373, 8.299503326416016, 0.761986494064331, -0.15619884431362152, -0.9330440759658813, 0.4273436367511749, -2.795112371444702, 1.7979429960250854, -3.162919521331787, -0.14441214501857758, -1.3204162120819092, 9.955338478088379, -0.6013721823692322, 0.7241698503494263, -0.9895309209823608, -1.4656832218170166, -0.1448855996131897, -0.5788636803627014, -0.6051653623580933, 1.8634264469146729, -0.9529411196708679, 0.6836642026901245, -0.27433207631111145, 0.37172213196754456, -2.4624454975128174, -0.5116000771522522, -2.3109302520751953, -1.1915663480758667, -2.1190552711486816, 2.528785467147827, 0.7075194120407104, -2.0687692165374756, 1.7683944702148438, -6.572766304016113, -2.0165908336639404, 0.4026276469230652, 1.544187068939209, -0.8523315787315369, -3.413917303085327, 0.7304410934448242, -1.231687068939209, -0.2137272208929062, -2.188624143600464, 2.375128746032715, -0.3053146302700043, -5.721861362457275, 0.16775371134281158, 3.6448960304260254, 1.4058740139007568, 0.6915444731712341, 0.28606632351875305, -1.8804750442504883, 0.35652315616607666, -0.5201787352561951, -0.7892346382141113, -0.39750295877456665, -5.603034973144531, 1.210590124130249, 0.5599694848060608, 0.3751078248023987, -2.9506847858428955, -1.3077032566070557, -4.767569541931152, 1.357252836227417, 0.22223426401615143, -17.542781829833984, 0.4593742787837982, -0.5841607451438904, 1.3489731550216675, -1.116819143295288, 0.3398215174674988, 19.84644889831543, -2.824704647064209, 0.2470368593931198, 1.0551505088806152, -2.175663948059082, -1.765622854232788, 0.8791573643684387, -4.350213527679443, -0.5015854239463806, -2.046828508377075, -0.4053553342819214, -0.3804124593734741, -0.07975853234529495, 1.6746375560760498, -1.1503609418869019, -0.05569373071193695, -3.179389715194702, -3.2650489807128906, -0.3293450176715851, -0.17179203033447266, -0.3179791569709778, -3.0813183784484863, -1.2116029262542725, 2.2918128967285156, 0.2201823890209198, -7.10302734375, 1.1177953481674194, 0.14188723266124725, -1.0783793926239014, -1.4468449354171753, -0.5177009701728821, 1.6939691305160522, 1.986917495727539, 8.664459228515625, -1.5545731782913208, -3.5175819396972656, 0.14936403930187225, -1.53727388381958, -1.9233406782150269, 1.1529674530029297, -0.622617244720459, 0.2670782804489136, 3.9143664836883545, -1.688746452331543, -0.618783175945282, -0.5539037585258484, 1.0223963260650635, -1.7632790803909302, -0.6318761706352234, -0.44871121644973755, -1.1561663150787354, 4.740422248840332, 1.2650121450424194, -0.6498105525970459, 0.22281010448932648, 0.4334218204021454, -1.5210278034210205, 1.8740906715393066, -7.998782157897949, -14.990763664245605, -0.8207279443740845, -1.1993485689163208, -0.4901958107948303, -0.7812241315841675, -0.8804506063461304, 0.2892135977745056, 0.027989991009235382, 0.0004028280091006309, 4.378692150115967, -2.4100003242492676, 2.6717474460601807, -0.17643533647060394, -2.2593250274658203, 0.8026331067085266, 1.0017249584197998, 1.1160670518875122, 1.58110773563385, -1.169325351715088, 0.27969005703926086, 1.913270115852356, -0.7341012358665466, -1.9520153999328613, -5.899229526519775, -1.685164451599121, 1.7009283304214478, 2.068897247314453, -1.9652739763259888, 0.5612756013870239, -0.9626665711402893, -2.050602436065674, 4.76300048828125, 4.3131842613220215, 0.32889875769615173, 0.40438538789749146, -2.636160135269165, -0.07627841830253601, -3.0417916774749756, -1.011210322380066, -4.328227519989014, -2.322901487350464, -1.8777879476547241, -3.69887638092041, -4.3618364334106445, 1.0686063766479492, -0.7229533791542053, 2.235501527786255, 0.3608236312866211, -0.46141621470451355, -0.4594322144985199, 0.3183489143848419, 1.3164927959442139, -1.997226357460022, 0.5762636661529541, 1.0969144105911255, -0.6873335242271423, 0.12886904180049896, -3.7094526290893555, 1.5959985256195068, -0.429284930229187, 1.0314558744430542, -0.5419216752052307, -0.535148024559021, 1.196581482887268, -0.6073505878448486, -0.04977176710963249, -0.9322316646575928, -0.17659474909305573, 1.4997081756591797, 1.3902533054351807, -0.6665213108062744, 8.46786880493164, 0.6412267684936523, -0.6899743676185608, -1.35802161693573, -4.570784568786621, 0.17009283602237701, -1.3172149658203125, -0.9281786680221558, -0.8691728711128235, 1.1401119232177734, -0.7144047021865845, 0.21958968043327332, -0.9771672487258911, -2.1018362045288086, -0.1461213231086731, 0.400236040353775, 1.4740303754806519, 8.787805557250977, -3.9345693588256836, -1.774662733078003, 4.341104507446289, -1.2217624187469482, -0.3184751570224762, 2.009249210357666, -0.29080459475517273, -0.06321945041418076, 7.3434576988220215, -0.8319220542907715, 0.6778913140296936, -1.347138524055481, -3.160845994949341, -2.2535006999969482, 0.1502760350704193, -1.9951554536819458, 0.40346235036849976, -5.286543369293213, 0.31757494807243347, -0.491791307926178, 0.13346542418003082, 2.179595708847046, 0.14338843524456024, -2.780733823776245, 2.499086380004883, -1.7864391803741455, 0.30097877979278564, 1.5722894668579102, 4.268781661987305, -0.3037051856517792, -2.052888870239258, 2.339783191680908, 0.8220779895782471, -1.0537852048873901, 0.28747260570526123, -0.3722964823246002, -0.1077076718211174, -0.24484437704086304, 1.0653104782104492, 1.3675179481506348, -7.740746021270752, 1.960024356842041, -1.1557292938232422, -1.8907305002212524, 1.5277129411697388, -1.1633998155593872, -2.4445347785949707, -2.7903072834014893, 1.4420984983444214, -1.2728034257888794, -1.602606177330017, -0.8761781454086304, 0.3566060960292816, -0.4313841760158539, 5.048504829406738, -2.8000245094299316, -8.943727493286133, -0.18256621062755585, 0.456773042678833, -2.8070905208587646, 0.15229898691177368, -1.650782585144043, -1.3176788091659546, -4.272248268127441, -2.9588637351989746, 2.8267273902893066, -1.1276015043258667, 1.9513039588928223, -1.166593313217163, -1.5870075225830078, 1.176216721534729, -0.982130765914917, -0.7945967316627502, 10.749078750610352, -1.2266247272491455, -0.8402779698371887, 0.31205835938453674, -0.9733152389526367, 0.6924224495887756, -1.2246105670928955, -1.9384181499481201, 5.594030857086182, 0.6462563276290894, -1.821289300918579, -3.858308792114258, -1.0098044872283936, 0.6574718356132507, -1.640742540359497, 0.7639353275299072, 3.774597644805908, -2.8263909816741943, -2.078552722930908, -0.8373081684112549, 0.424152672290802, 4.3463215827941895, 1.2338969707489014, 4.570700168609619, -0.7625262141227722, -0.3636115491390228, -2.3553009033203125, 0.32925790548324585, 0.4978961646556854, -1.1599403619766235, -0.7409018278121948, 2.9359285831451416, -0.9641706347465515, -1.0866093635559082, -0.0307792779058218, 0.6606042385101318, -1.0861358642578125, 4.3693318367004395, -7.535323619842529, -2.0918052196502686, 4.283706188201904, 0.3472287952899933, -0.46650996804237366, 0.7650293111801147, -0.048864491283893585, -0.8467340469360352, -1.5597707033157349, -0.9647682309150696, 14.191818237304688, 5.195763111114502, -0.4317297041416168, 1.9516377449035645, -1.7680771350860596, -2.889965295791626, -2.686471700668335, -1.2742226123809814, 0.7157394289970398, 3.129723072052002, -7.146798610687256, 10.091266632080078, -0.968934178352356, -2.358597993850708, 0.19156545400619507, -0.9487436413764954, -1.1892703771591187, 1.0361649990081787, 1.1235387325286865, -0.2669229805469513, 0.2133554369211197, -3.2351884841918945, 1.4142667055130005, -1.5575737953186035, -6.939249515533447, -4.185943126678467, -1.2271614074707031, -2.0930352210998535, -0.04949754849076271, -0.5801043510437012, 0.28726136684417725, -0.896239161491394, -1.6448044776916504, -0.7987801432609558, -1.61076819896698, -4.767146587371826, 0.7213023900985718, -0.87215256690979, 0.5207226276397705, -0.42710214853286743, -1.137204885482788, 0.12322068959474564, -1.9855037927627563, -1.3107991218566895, 4.355747699737549, -1.434165120124817, -3.6791934967041016, -1.0896263122558594, 0.28798991441726685, -5.65668249130249, -2.7764687538146973, 0.7207618355751038, -1.0222649574279785, -0.29911285638809204, -0.9482784271240234, -1.0133274793624878, 1.3066486120224, 1.0473822355270386, -1.157867670059204, -2.0815553665161133, 0.3356148302555084, -0.294567346572876, -2.592898368835449, 0.29816895723342896, 0.5896360874176025, 1.5329694747924805, 0.048985227942466736, -0.7376008033752441, -0.4557424783706665, 21.362478256225586, 0.8144406676292419, 4.641698360443115, -0.8129674792289734, 4.25095272064209, -1.873708963394165, -8.603157043457031, 1.1131584644317627, 1.7079590559005737, 0.23357948660850525, -0.4032596945762634, -0.2098069041967392, 1.3662430047988892, -3.535249948501587, 0.008628985844552517, 0.4715130031108856, 1.13559091091156, -0.6184190511703491, 5.2230963706970215, -0.21825924515724182, -3.2181179523468018, -0.8270705938339233, 1.2264578342437744, -3.8175408840179443, -2.824864625930786, 0.010055403225123882, 4.357409477233887, 0.7020835876464844, -1.0706143379211426, 2.800201177597046, -1.3577097654342651, -5.387012958526611, -2.845590591430664, 4.273454666137695, 1.0529453754425049, 1.0186668634414673, -1.7154510021209717, 0.3328854441642761, -5.027085304260254, 0.7748970985412598, 0.9729220867156982, 0.2640431225299835, -2.365370750427246, -0.20418433845043182, -0.8527657389640808, -2.3092262744903564, -1.8674287796020508, -1.704160451889038, 1.2073460817337036, 6.413110256195068, 4.243515491485596, -2.005200147628784, -0.5584664344787598, -1.076008915901184, 0.49711140990257263, 0.45391523838043213, -1.5508767366409302, 0.012522729113698006, 13.66426944732666, -8.092565536499023, 1.1959359645843506, 0.4487343430519104, -0.9905373454093933, -4.6503095626831055, 10.856377601623535, -0.5720490217208862, -2.811624526977539, -1.3149943351745605, 2.1465277671813965, -0.718745768070221, 0.4848441183567047, -5.275914669036865, 0.8987473845481873, 0.3232669532299042, -0.5356064438819885, 0.7592611908912659, -0.6817779541015625, -1.6613264083862305, 4.260319232940674, -2.922501564025879, -0.44796738028526306, 0.3729456067085266, -1.1215702295303345, -0.5545501708984375, -0.3900223672389984, 0.7949958443641663, -1.5923943519592285, -0.6750470399856567, 0.38356882333755493, 0.03738629072904587, -0.4954864978790283, 0.8352841734886169, -2.940852165222168, -0.3757263422012329, 0.007386037614196539, -6.9454545974731445, -0.12474583834409714, -2.4400441646575928, 0.3041720986366272, 0.41025954484939575, -0.3951895833015442, 0.08135572820901871, -0.17273849248886108, -1.7394788265228271, -0.964106559753418, 1.5780060291290283, -0.11723671108484268, -3.2413418292999268, -0.3048478066921234, 0.4291541576385498, 4.255632400512695, -0.22756381332874298, 1.5142900943756104, -1.107293725013733, -1.124700665473938, -0.11318308860063553, 1.7596248388290405, 1.686435580253601, 1.048350214958191, -1.2740172147750854, -0.34488043189048767, 2.809448480606079, 0.6337026953697205, -0.0645141676068306, -3.852651357650757, 0.7259666919708252, -0.35804256796836853, 2.2531847953796387, -6.554409027099609, -0.46553319692611694, 1.7409650087356567, -1.3513424396514893, 6.314650535583496, -2.3588082790374756, 0.7927002310752869, -0.550919771194458, -1.6700091361999512, -0.37259814143180847, -3.536119222640991, -4.431722164154053, 4.344760894775391, -0.26872366666793823, -0.2733485698699951, -1.1762449741363525, 0.3880431354045868, -3.3017382621765137, -0.4069623649120331, -4.368329048156738, 0.41908180713653564, -0.4123958945274353, -0.7413644790649414, -0.20402881503105164, -1.0157462358474731, 0.9766103625297546, -1.2966864109039307, 0.5207870602607727, 0.004647425841540098, -0.9972448348999023, 3.1087377071380615, 0.38747596740722656, 0.10728186368942261, -0.59107905626297, 0.261385053396225, 2.7797305583953857, -2.1229965686798096, 0.5073972344398499, 2.8585147857666016, -0.800790548324585, 1.20163094997406, -1.286177158355713, 0.9567787051200867, -0.03435707092285156, 18.010921478271484, -7.578572750091553, 1.0590075254440308, -0.548459529876709, -4.26215934753418, 0.22999721765518188, -0.967609167098999, -0.5938258767127991, 0.047301035374403, 0.9376415014266968, 0.3326598107814789, 1.2613554000854492, 0.04349154978990555, -2.9663288593292236, -2.7486934661865234, 0.23855610191822052, 0.29578152298927307, 6.610801696777344, -0.22535346448421478, -1.9962626695632935, 1.156995415687561, -2.0084803104400635, 0.9207261204719543, 1.8832265138626099, -1.1793094873428345, 0.28895968198776245, -0.013570556417107582, -0.10702558606863022, -0.2213982343673706, -0.5905600786209106, -0.36799755692481995, 5.588837623596191, -0.7671099305152893, -1.1580519676208496, -4.276115417480469, -1.1221174001693726, -0.7330343127250671, 0.6147795915603638, 1.5913598537445068, 0.3596240282058716, -0.3726670742034912, 0.5465046763420105, -1.167161464691162, -0.029971400275826454, -3.456533670425415, -0.47668007016181946, 1.4809826612472534, -0.8016590476036072, -2.2944891452789307, 0.638477087020874, 0.11065930873155594, -0.4210370182991028, -0.9509536623954773, 0.27724719047546387, -2.251598834991455, 7.188760757446289, -11.850449562072754, 1.2473022937774658, 7.947963237762451, -0.4431091248989105, -2.5929067134857178, 3.7573766708374023, -10.163392066955566, -4.282779693603516, 0.9684481024742126, -2.7237865924835205, 1.017918348312378, 0.1360684633255005, 1.3253870010375977, -1.8491038084030151, -3.1669962406158447, -0.44317880272865295, -0.715957760810852, -0.6851941347122192, -8.002957344055176, 0.6766732335090637, -0.46476954221725464, 0.04811229929327965, -5.239114761352539, -3.3325035572052, -0.6083589196205139, 0.10558027774095535, -5.11745548248291, -0.8298043012619019, 0.3290073871612549, -0.20690423250198364, 0.5457680225372314, 2.489647150039673, -5.360957622528076, -1.0231246948242188, -3.9019696712493896, -1.1664791107177734, 2.4891490936279297, 1.3930275440216064, -4.740071773529053, 1.4948610067367554, -0.5376802086830139, -1.6636954545974731, -4.3587822914123535, -0.8175579905509949, 0.5334500074386597, 1.0975090265274048, -1.084718108177185, 4.653172492980957, -0.326837956905365, -0.2382531315088272, 1.8201850652694702, -0.18045873939990997, 0.7512160539627075, -4.036811828613281, 0.38360705971717834, 0.8332066535949707, -1.4181138277053833, -1.4530068635940552, 0.4319123923778534, 2.629260778427124, -0.39054518938064575, 0.2513422966003418, 1.1960644721984863, -1.14541757106781, 0.9833499789237976, 0.4953692853450775, 1.2872161865234375, -1.6223583221435547, 4.944546699523926, 0.354098379611969, -1.831150770187378, 0.5637629628181458, -9.136704444885254, -1.4305986166000366, -2.358304262161255, -0.3875987231731415, -2.79288911819458, -0.9606816172599792, 0.03110441006720066, -1.1116626262664795, -1.4679255485534668, 1.125227689743042, -0.7653543949127197, -10.420413970947266, 0.055221639573574066, -0.925885796546936, 0.20493820309638977, -0.1508171260356903, 0.7223771810531616, -0.8390151858329773, 0.018847521394491196, 1.6331143379211426, -1.9059728384017944, -1.6604149341583252, -0.7194775938987732, -1.8740044832229614, -0.46990230679512024, -2.962637424468994, 1.9911848306655884, 0.9649553298950195, -2.2121851444244385, -1.6012989282608032, -1.1646020412445068, -0.05759753659367561, -1.7341641187667847, -2.453883171081543, -5.341181755065918, 1.5833027362823486, -0.04567611962556839, 0.8226038217544556, -0.7331798672676086, -1.5026297569274902, -1.8477360010147095, 0.34487780928611755, -2.8687145709991455, -2.5877463817596436, -1.1712764501571655, 1.356687068939209, 2.5096232891082764, 2.3144726753234863, 4.34614372253418, -0.5157870650291443, 2.6422042846679688, -0.598444938659668, -1.2929818630218506, -0.6618290543556213, -8.007284164428711, 1.5538791418075562, -3.882391929626465, -1.556836485862732, 5.6015238761901855, 0.1177162453532219, -2.0629262924194336, -0.37703055143356323, -2.886140823364258, -0.5432084798812866, -0.8088393807411194, -1.2306945323944092, -0.1920630782842636, 1.8370579481124878, -0.23893097043037415, -3.43015193939209, -1.2852003574371338, 1.3262866735458374, -0.8776028752326965, -0.4723651707172394, -0.768504798412323, -0.32913845777511597, -0.6129281520843506, -1.2591472864151, 1.072696328163147, -0.5903071165084839, -0.5271375775337219, -0.9030886888504028, 0.26184195280075073, -4.9266815185546875, 0.39117559790611267, 0.27642345428466797, -2.090073347091675, -0.25299277901649475, -0.9173645973205566, -0.7909591794013977, 1.4217908382415771, 0.5688145160675049, 1.3827197551727295, -0.9241433143615723, 1.1389485597610474, -1.4498642683029175, -1.525206446647644, 1.095299482345581, -11.774080276489258, -2.458151340484619, 0.35813018679618835, -0.9945369362831116, 3.4519121646881104, -3.1854312419891357, -1.097200632095337, -2.0102686882019043, 0.5393499732017517, 2.069751739501953, -2.2824196815490723, 0.22868438065052032, -1.203446626663208, 0.782768189907074, -0.5331934094429016, 0.37725213170051575, -0.13466815650463104, 0.5410727262496948, 0.1027684137225151, -2.246934413909912, 0.6282030344009399, 1.0146565437316895, 0.1880938559770584, -1.020177960395813, -0.9550884366035461, 1.396865725517273, -0.27703046798706055, 0.34206852316856384, 0.4770360291004181, -1.0873850584030151, 0.23408068716526031, 1.4166380167007446, -1.035490870475769, -2.731336832046509, 3.33172869682312, 0.36591094732284546, 1.383532166481018, -5.044732093811035, 3.72876238822937, -3.64715838432312, -1.4769670963287354, 0.2077755331993103, -3.4970591068267822, 0.5454762578010559, 0.02600148133933544, 0.6194669604301453, 4.258473873138428, 1.5149844884872437, 0.17952443659305573, -0.0960383266210556, -8.052151679992676, -2.5165181159973145, 0.46127092838287354, -0.251569539308548, 2.7893552780151367, -0.5005894899368286, 1.0841412544250488, -6.615255832672119, -0.5702197551727295, -1.0908993482589722, -1.5743063688278198, 1.336883306503296, -1.863624095916748, 11.851082801818848, 0.44230881333351135, -1.3662850856781006, 4.364209175109863, 0.24697284400463104, -0.8571120500564575, -1.1949048042297363, 17.9011173248291, -1.6063998937606812, 1.1105937957763672, 1.2443437576293945, -0.8986437916755676, 0.11384238302707672, 2.9421987533569336, -5.97480583190918, 0.3477710783481598, -2.5107486248016357, 1.2192789316177368, -0.530077338218689, -1.2121529579162598, -1.83174467086792, 0.3621169626712799, -1.7800086736679077, -1.825197458267212, -0.07224007695913315, 0.6764635443687439, 1.6475090980529785, -2.5098814964294434, -0.1948419064283371, -0.3784053325653076, -0.6750034689903259, -0.49652183055877686, -1.8742934465408325, -2.2228152751922607, -3.776526689529419, 7.21965217590332, -0.21499566733837128, 0.8587676286697388, 0.791619598865509, -3.228675365447998, -0.7668066620826721, 1.2039806842803955, 6.903658390045166, -0.326211154460907, 0.46521005034446716, 0.7425065636634827, -0.23525218665599823, -1.356104850769043, 0.7639757990837097, -0.9964224696159363, -0.5223367810249329, -2.4393932819366455, -0.46426039934158325, -0.12083707004785538, -0.56673264503479, 0.8444949984550476, 0.786451518535614, -1.7332775592803955, -0.06988383084535599, -0.005241844803094864, 4.836295127868652, -3.1143553256988525, 0.7809785008430481, -1.1407033205032349, 1.4577593803405762, -2.7315163612365723, 3.9621942043304443, -0.5482954382896423, 0.9834063649177551, -2.5465071201324463, -0.7429174780845642, 0.2013293206691742, -1.3196868896484375, -1.791283130645752, -2.830523729324341, -1.1617399454116821, 1.6148481369018555, -4.313081741333008, 1.2450326681137085, 0.916435956954956, -0.5305764675140381, 0.6940134167671204, -5.776198387145996, -2.0371556282043457, 1.0076289176940918, -0.692126452922821, -1.5692094564437866, 1.8885107040405273, -10.345311164855957, -0.5022507905960083, -0.030681194737553596, 1.6688520908355713, 4.322617053985596, 0.13986919820308685, 4.44918966293335, 0.8192060589790344, 0.21099719405174255, -0.4607495963573456, -0.2722238004207611, -0.1721123903989792, -1.9335048198699951, 5.589254379272461, 2.08085036277771, 0.48606178164482117, 0.5619794130325317, -8.417137145996094, 0.08532509207725525, -5.793591499328613, -2.0721869468688965, 4.371488571166992, 1.649253487586975, -0.9554327726364136, -2.8659262657165527, -0.2914498448371887, -2.123215675354004, 0.5090869665145874, -1.3117737770080566, -2.867229461669922, -1.9431122541427612, 4.664330005645752, 3.6457176208496094, -2.763654947280884, -0.3282138705253601, -0.7405700087547302, -1.4956161975860596, 0.720098614692688, 1.2277251482009888, 0.9229270219802856, 0.48377853631973267, -0.4048343300819397, -4.426352500915527, 0.19990351796150208, 6.630618572235107, -1.4267873764038086, 0.8065789341926575, -2.29790997505188, -0.38772210478782654, -0.29386505484580994, -2.2944791316986084, -1.3594716787338257, -0.7273114919662476, 1.5817673206329346, 0.6261367797851562, 0.7006193995475769, -1.1552430391311646, 0.1654011756181717, -1.162137508392334, 2.0744643211364746, -3.792804002761841, 0.3271431028842926, 0.5405417084693909, 1.6481642723083496, 1.0210717916488647, -9.620720863342285, -1.8018134832382202, 4.365420818328857, 0.5075507164001465, 0.18305984139442444, -0.7134181261062622, 18.637033462524414, -1.1315910816192627, 2.169562578201294, -0.13719025254249573, -0.15330204367637634, -0.2588072717189789, -2.7993688583374023, -0.05532633140683174, 0.4661005139350891, -0.38062408566474915, 0.3928581476211548, -4.201128005981445, 0.5362737774848938, 1.2719497680664062, 5.186622619628906, 4.0652384757995605, -4.165882110595703, -3.0197560787200928, -0.46043094992637634, -0.016061747446656227, -1.2234067916870117, -0.616992175579071, 0.2032942771911621, 2.50724196434021, 0.19193723797798157, -1.616834044456482, -0.4135279059410095, 2.045602321624756, -1.2479218244552612, 4.47110652923584, 0.6511646509170532, -7.729926586151123, 0.7017940282821655, -0.6762918829917908, 1.1829699277877808, -3.9017117023468018, -0.25317904353141785, 0.01384508702903986, 1.9828790426254272, 2.368006706237793, -1.6204466819763184, -1.9618067741394043, 2.588745355606079, -0.24445930123329163, -1.6632789373397827, 0.5526606440544128, -5.399040222167969, -1.2373820543289185, -1.0343046188354492, 1.5720988512039185, -5.521474361419678, 0.39156246185302734, 0.02459612861275673, 0.6186366081237793, -0.8426825404167175, 0.6311399340629578, 2.951881170272827, -1.9603543281555176, 4.370059490203857, 13.724729537963867, 0.16545546054840088, 0.35611605644226074, -0.6924106478691101, -7.105091094970703, 0.34141379594802856, -1.5361496210098267, 0.5368364453315735, -5.798591136932373, -2.6410133838653564, 2.4625301361083984, -0.48819389939308167, -4.317166805267334, 1.2678987979888916, 0.025555837899446487, -6.1381378173828125, 0.8469009399414062, -0.18160930275917053, -0.11751657724380493, 1.1592957973480225, 2.3038713932037354, 0.38632163405418396, -1.4033194780349731, 1.0658071041107178, -2.8676626682281494, -2.7252728939056396, -3.01009202003479, -1.1684843301773071, 4.364068984985352, -0.3586621582508087, 0.9047819375991821, -4.177651405334473, 0.12367686629295349, -0.251716285943985, 1.17900550365448, -1.1238576173782349, -0.5732560157775879, -0.27469003200531006, -0.4544360339641571, 1.580485463142395, -1.8915332555770874, 0.46772465109825134, -0.5945620536804199, -2.137295961380005, 4.368137836456299, -2.023108720779419, 4.292723655700684, -2.3674774169921875, 0.3995325267314911, -2.994215965270996, 0.3746618330478668, -3.8893959522247314, 3.23814058303833, -0.2254989892244339, -1.674808144569397, 0.9638979434967041, -1.155385971069336, 0.4362972676753998, -0.12431719154119492, -1.426621437072754, -3.1042613983154297, 0.07988489419221878, -3.233804941177368, 0.7404707074165344, 0.3652043044567108, -0.9657647609710693, 0.3456529974937439, 0.3495209217071533, 0.020796122029423714, 0.5414318442344666, -0.4376925528049469, -3.199665069580078, -0.3738110661506653, 3.455087900161743, 0.030561452731490135, -0.7690218091011047, 3.4909801483154297, 0.3281077444553375, -3.1248559951782227, 0.5499244928359985, -0.8017069697380066, -0.5181388258934021, -3.231327772140503, -1.4607435464859009, 2.8878307342529297, 1.0733243227005005, -2.9633920192718506, -6.1318864822387695, -1.349231481552124, 1.7711137533187866, 0.07863184064626694, 0.5032563805580139, -1.1581817865371704, -12.723983764648438, -0.11501912027597427, 2.223111867904663, -0.4436136782169342, 2.3403875827789307, 1.5218945741653442, -1.140674114227295, -0.7290671467781067, 0.5012639164924622, 0.13536061346530914, -0.6072308421134949, 1.3405120372772217, 0.20383018255233765, -0.433459609746933, 1.1771020889282227, -2.873725175857544, 13.827186584472656, -2.5460782051086426, -2.461582660675049, -3.5148510932922363, -1.5031381845474243, -5.780224800109863, -2.9859390258789062, 1.4087815284729004, 3.7847189903259277, 0.6484510898590088, -2.29972243309021, 1.422569751739502, -0.8508363366127014, 0.6889749765396118, 1.6789473295211792, -2.0428948402404785, -0.45905277132987976, -2.5900614261627197, 15.376623153686523, 1.4297820329666138, -0.6968869566917419, -0.2714143991470337, 0.640663743019104, -2.4469399452209473, 0.5956276655197144, 0.03088877536356449, -2.015410900115967, 7.071206569671631, -1.4152711629867554, 1.0244454145431519, -4.255894184112549, 4.349963188171387, -1.096271276473999, -0.5338271856307983, -1.3503800630569458, 1.0519109964370728, -1.1249245405197144, -1.833946943283081, 1.1870876550674438, 1.8962011337280273, 3.7320432662963867, 3.2681703567504883, 0.765521764755249, 1.6227471828460693, -0.4092620313167572, -2.8117456436157227, -2.090532064437866, 0.3363533318042755, -0.038603369146585464, -2.496537208557129, -0.06317164748907089, 0.7187736630439758, 0.8865422606468201, -5.808900356292725, 0.5105290412902832, -1.1020156145095825, -1.2098497152328491, -0.011827127076685429, -0.9138607978820801, 1.0722160339355469, -0.29566070437431335, 2.7221736907958984, 0.2994009852409363, -0.19409377872943878, -4.57745361328125, -2.052708148956299, 0.2881244421005249, -0.6537865996360779, -1.6801726818084717, -6.541454315185547, -2.347259044647217, -1.1730225086212158, 0.8475775718688965, 1.376854658126831, 0.2811264097690582, 1.0273357629776, -5.8734235763549805, 2.3153555393218994, -0.6551656723022461, 0.996289074420929, -1.1145875453948975, 0.6489253640174866, -1.1630018949508667, -1.3526684045791626, -8.466361999511719, -0.9485816955566406, 0.42316746711730957, -0.8510334491729736, 0.31610146164894104, 0.46206897497177124, 1.9673179388046265, 0.34252506494522095, 1.3674554824829102, 0.6087056398391724, 11.382304191589355, -0.9474762082099915, -0.6523797512054443, 0.7148375511169434, -1.6258134841918945, 0.7032878398895264, 0.048280779272317886, -5.769126892089844, 4.3568854331970215, -0.661021888256073, -1.5491937398910522, 1.0589134693145752, -1.9371325969696045, 1.0098034143447876, -0.9668700695037842, 1.5040209293365479, -0.0016667474992573261, 0.0013863269705325365, 0.9493186473846436, -6.891500949859619, 2.0745694637298584, -0.459553062915802, 0.41227635741233826, 0.282392680644989, -5.184906482696533, -0.10947483777999878, -4.241596698760986, -2.164076328277588, -1.369650959968567, -1.65091872215271, -2.2857987880706787, 0.7441092729568481, -1.489958643913269, -0.2005462795495987, -1.502699375152588, -1.3350889682769775, -0.9099329113960266, 1.1122249364852905, 2.2645270824432373, -8.002137184143066, 0.41423678398132324, 0.7089912295341492, -1.882646083831787, -0.5214982628822327, -0.6231080293655396, 5.586383819580078, 0.28553080558776855, 0.9501450657844543, -1.0103552341461182, -0.8075006604194641, -1.1620643138885498, -1.835647463798523, 3.056838274002075, 0.38552772998809814, -0.7299271821975708, 0.27951839566230774, -2.565680503845215, -1.2137434482574463, -2.216219663619995, -3.3263070583343506, 0.232389435172081, 1.2509040832519531, 4.2607598304748535, -5.331993103027344, -0.42274704575538635, 1.265211582183838, 1.5204426050186157, -2.736698627471924, 0.2715184986591339, 7.833069801330566, -6.8285346031188965, -12.088177680969238, 0.1463325172662735, -1.4041552543640137, -2.1687121391296387, -2.3261868953704834, -3.127821445465088, -0.19032277166843414, -5.8639726638793945, 15.606746673583984, -5.344137668609619, 1.0706888437271118, 14.474849700927734, -0.11412576586008072, 0.5242080688476562, -0.9918363094329834, 0.4252101480960846, -2.9825828075408936, 1.221172571182251, -1.8933297395706177, -2.8662092685699463, -0.2969855070114136, 1.5287872552871704, 2.0888452529907227, -2.6884167194366455, -0.16426527500152588, -2.279179573059082, -0.36375299096107483, -0.13793854415416718, -0.6484125852584839, -0.23181597888469696, -2.4643547534942627, 1.9714285135269165, -0.6889227628707886, 1.208998441696167, -0.20234978199005127, -0.5190088152885437, -4.193740367889404, 1.6009010076522827, -0.995327889919281, 0.007075135596096516, -1.150836706161499, -0.9427960515022278, -1.2016260623931885, 0.05841605365276337, 8.609267234802246, 1.4665818214416504, -1.5612059831619263, -0.11119139939546585, -0.023304389789700508, -2.615644693374634, 1.9903459548950195, 0.4394168257713318, 1.0205795764923096, 2.689113140106201, 3.5455217361450195, -0.9857400059700012, -2.8483870029449463, 1.3312166929244995, -1.5203114748001099, -2.486937999725342, -1.970102310180664, 0.4151334762573242, -0.45351770520210266, 2.734562397003174, 0.3661477267742157, 4.3447675704956055, 0.41515782475471497, 0.5784066319465637, -0.3697812259197235, 0.6518044471740723, -0.3648894429206848, -1.3763772249221802, 0.468651682138443, -0.4665561616420746, -0.6241090297698975, -0.9523755311965942, -3.8834822177886963, 4.270627498626709, -0.3780573308467865, -0.7487682700157166, -0.00687197083607316, -0.6673064231872559, 0.6884709596633911, 2.511014223098755, -0.47641023993492126, -0.36713486909866333, 0.45776912569999695, -2.1235861778259277, -0.13261061906814575, -1.7552632093429565, 18.067670822143555, -1.4196268320083618, -0.49561360478401184, -0.016151929274201393, 6.908120632171631, -0.2958824932575226, 1.2982903718948364, 1.2915468215942383, 0.0842893123626709, -0.3428373634815216, 1.116707444190979, -0.035913411527872086, -1.2651971578598022, -3.74499249458313, -0.696546733379364, 2.816805601119995, -6.271485805511475, -1.8537557125091553, 1.1609681844711304, 0.37587741017341614, -0.09581030905246735, 0.16481514275074005, 0.7334423661231995, -0.027154691517353058, -1.1205049753189087, -0.2548618018627167, 1.849565863609314, -8.438504219055176, -0.2787402868270874, -0.1718350648880005, -2.0167484283447266, 0.1480604112148285, 0.4540875554084778, -1.4071623086929321, -1.0403869152069092, 1.281887412071228, -0.33281150460243225, -0.017380574718117714, 2.658491849899292, -2.3473007678985596, -2.4302022457122803, -2.081122636795044, -1.469482421875, -0.9108944535255432, -3.4551846981048584, 4.2545294761657715, -1.627827525138855, 0.9434566497802734, -1.3872781991958618, 0.8300513029098511, 1.0291855335235596, 4.370404243469238, -0.7803179621696472, -4.335744380950928, 14.01700210571289, 11.467872619628906, -1.871507167816162, 0.19441579282283783, -3.67922043800354, -0.7081024646759033, -2.2821879386901855, -1.1737298965454102, 0.6747995615005493, 2.076195001602173, -2.39101505279541, -0.6787892580032349, 4.373819351196289, 4.278036117553711, -6.871741771697998, 4.308553695678711, -0.797738790512085, 0.7361654043197632, 1.719302773475647, -1.3555322885513306, 0.3643467426300049, -7.633565902709961, 1.0053130388259888, -1.2899115085601807, 0.011385031044483185, 5.270809650421143, 0.3571246266365051, -1.082771897315979, 1.26901376247406, -0.2919037640094757, 1.655928611755371, 0.9162720441818237, -1.5007071495056152, 2.0512967109680176, -1.9291242361068726, -1.2881413698196411, 3.2052605152130127, -1.495561122894287, 5.112171649932861, -3.5802223682403564, -0.22549554705619812, 0.764456570148468, 0.7369187474250793, -0.4645310044288635, -0.4843728840351105, -1.0779465436935425, 0.5130935311317444, 0.25366273522377014, 0.8255870342254639, 6.974096298217773, 0.8914393186569214, -0.29656168818473816, -0.08892226219177246, -2.491211414337158, 0.3258165419101715, 1.6335582733154297, 4.6680707931518555, -1.1605119705200195, -2.293677568435669, -0.5719087719917297, 3.280271530151367, -0.07390297204256058, -1.016618251800537, -1.6245012283325195, 0.44976866245269775, -0.32535338401794434, 0.8668830394744873, 1.4570080041885376, -0.10411331802606583, -1.2020971775054932, -1.1079633235931396, -2.0357534885406494, 5.023245811462402, -1.2904740571975708, 0.40992897748947144, -0.8493771553039551, -0.7236693501472473, -0.6910862326622009, 0.20515836775302887, 0.015880312770605087, -0.21527598798274994, -1.163727045059204, -0.08766813576221466, -3.11860990524292, -1.3367910385131836, 0.217664435505867, -0.19132941961288452, 0.37921732664108276, -1.312615990638733, 0.5008219480514526, 0.10491596907377243, 1.181837558746338, 4.3765459060668945, 1.1269348859786987, 6.161985397338867, -0.6501563191413879, 0.6313046216964722, 0.12476592510938644, 2.008230209350586, -1.298751711845398, -0.8009209632873535, -0.2982814610004425, -0.4358915090560913, -1.161238431930542, 1.0566166639328003, -2.072554588317871, 0.3068026900291443, 0.5420314073562622, -1.0219790935516357, 2.1938929557800293, 0.030604014173150063, -1.077692985534668, -2.875204563140869, -0.7718958854675293, -0.6668031811714172, -0.49418362975120544, -1.5650686025619507, -0.1845625340938568, -0.18721337616443634, -0.9769300818443298, -1.555784821510315, 0.8559983968734741, -1.8349864482879639, -0.05622139573097229, -5.659701824188232, -3.9666903018951416, -1.41982901096344, -4.362298011779785, -1.8283469676971436, -1.861973524093628, 0.392023503780365, 3.3812050819396973, -1.7329505681991577, -0.2764419913291931, -0.39835742115974426, 0.09008514881134033, -1.2011756896972656, 0.37944358587265015, -1.3334434032440186, -1.1450605392456055, -0.6385176181793213, -1.809092402458191, 0.3963649272918701, -0.5523070693016052, -0.005131298676133156, 1.7286263704299927, 0.44109681248664856, 0.17359282076358795, 3.02176570892334, -1.3152623176574707, -0.4777585566043854, -0.0627642571926117, 0.2783878743648529, -9.310429573059082, -1.4591734409332275, -0.5698223114013672, -2.1651976108551025, 2.0771853923797607, -0.491483598947525, -1.596858024597168, -0.29633423686027527, -0.06165819615125656, -0.19890393316745758, -4.674631118774414, 0.03465623781085014, 0.6655508279800415, 0.008043897338211536, -5.393910884857178, -0.6218802332878113, 0.12317698448896408, 0.5643219351768494, 4.624229431152344, -0.08546148985624313, -0.6565079689025879, -1.8271790742874146, -0.22421276569366455, -1.4120984077453613, 4.6471991539001465, -2.102200746536255, -0.4970887303352356, -1.3617594242095947, -0.173222154378891, -2.1354334354400635, 0.012958000414073467, -0.14939065277576447, -0.5375601649284363, -0.07835520803928375, -3.7829251289367676, 3.3172264099121094, -1.0436556339263916, -0.28930971026420593, 0.4304323196411133, 0.912618100643158, 0.35056453943252563, -0.4579547643661499, 3.6519055366516113, 3.752711296081543, 5.385409832000732, -0.67176353931427, -1.2044241428375244, -2.3144357204437256, 3.263399362564087, 3.444220781326294, 1.2594391107559204, 2.8982717990875244, -0.06807970255613327, 0.3241150975227356, -1.7529774904251099, 2.940342426300049, 2.528752565383911, -1.5493360757827759, -4.073859214782715, -1.0952327251434326, -3.421417474746704, 0.24215713143348694, 0.04685086011886597, -0.42543521523475647, 1.542975902557373, -1.8243215084075928, -1.0314207077026367, 0.6340305805206299, 1.6742216348648071, 1.5198653936386108, 4.2177042961120605, 0.7504643797874451, 0.11537925153970718, -0.23796311020851135, 13.662666320800781, 1.3794533014297485, -6.078928470611572, -0.8101789355278015, 1.4243788719177246, -2.858157157897949, -0.027674060314893723, -7.014490127563477, -4.981179237365723, -3.404468536376953, -0.1961035430431366, -1.9246158599853516, -2.8627305030822754, -2.400848627090454, -1.4076220989227295, -2.0229456424713135, -2.2238240242004395, -1.2463749647140503, 1.0741868019104004, -0.8549325466156006, 0.05958622321486473, 2.108691453933716, -0.18804608285427094, -0.831072211265564, 0.3782305121421814, -0.6821473240852356, 0.3986040949821472, -1.21677565574646, -0.43039441108703613, -0.5300795435905457, -1.9773980379104614, 10.754352569580078, 0.23235447704792023, -0.015176882967352867, -1.3301949501037598, -2.5115630626678467, -0.5104101300239563, 1.4861807823181152, -3.3145413398742676, -0.6653885841369629, -1.2814232110977173, -0.3884527087211609, -0.4709235429763794, 0.511448860168457, -4.985955238342285, -1.794330358505249, -6.551473617553711, 0.37735313177108765, 0.7099034786224365, -2.4158003330230713, 2.5600225925445557, -0.7021902203559875, -0.1990182250738144, -1.3211932182312012, 3.95475172996521, -0.037213850766420364, -5.282214641571045, -1.8064278364181519, -3.2942757606506348, -1.8206827640533447, -8.017118453979492, -0.0377846360206604, -1.5698747634887695, -0.73430335521698, -1.8784294128417969, 0.7723281383514404, 1.0567822456359863, -1.2434419393539429, 0.254789799451828, 0.5892808437347412, -0.7254024147987366, -14.974960327148438, -0.5415259003639221, -2.2422194480895996, 0.38153785467147827, 0.5074524283409119, 2.1127264499664307, -2.3404793739318848, -0.04479193687438965, -3.8769328594207764, -0.29997196793556213, 0.8494412899017334, 0.41734641790390015, -0.21647098660469055, 0.12672393023967743, -8.019760131835938, 7.4787797927856445, 1.8992669582366943, 0.3923681676387787, 1.7428497076034546, -0.7165206074714661, 4.9593729972839355, 0.24868741631507874, 0.4524807929992676, -0.6158472299575806, -0.42240074276924133, -0.5796365141868591, 2.9499075412750244, -5.959723472595215, 4.363864898681641, -1.293555498123169, -3.9454386234283447, -0.5987817049026489, 0.4413076639175415, -4.171818256378174, 0.32003703713417053, -6.2730278968811035, -1.432106852531433, -7.2933268547058105, -2.265376329421997, 0.6683392524719238, -1.868036150932312, -1.174558401107788, -1.2324661016464233, 0.7115044593811035, -0.6223880052566528, 0.4377330541610718, -1.7824795246124268, -2.888113260269165, -0.9102436304092407, 2.376422166824341, 10.918062210083008, 5.7845563888549805, -1.19111168384552, 0.7744771838188171, 4.157177925109863, 1.651908040046692, -1.1865147352218628, -0.16125939786434174, -1.2458034753799438, 0.1761852502822876, -0.15269991755485535, 0.5759834051132202, 0.3793560266494751, 1.2121862173080444, -0.0738580971956253, 0.4461892545223236, -0.22127698361873627, -0.9508810639381409, 0.6417379379272461, -2.6000683307647705, 0.2930861711502075, -1.4355204105377197, -1.6395883560180664, -3.705134391784668, -1.9695570468902588, 1.399394154548645, -0.29029595851898193, 0.7138679027557373, -2.4520559310913086, 5.722373962402344, 0.1615593284368515, -1.8646003007888794, 1.0021940469741821, -1.116942286491394, 0.23615482449531555, 0.4330129325389862, -1.8686336278915405, -7.984408855438232, -0.08373928815126419, -4.452362060546875, 0.3515160381793976, -0.17072804272174835, 0.2873920798301697, 0.0898732990026474, -2.1431570053100586, 0.6308790445327759, 0.7676434516906738, 2.0850090980529785, -1.55862557888031, 0.6948990225791931, -1.4953511953353882, -1.8103395700454712, -1.4322926998138428, 3.238816261291504, -0.9764536619186401, 0.7584643363952637, 1.1109522581100464, 0.19160480797290802, -0.5785145163536072, 1.6545921564102173, -1.4126219749450684, -3.4117937088012695, 0.4017421305179596, -0.3838595449924469, -7.998294353485107, 2.931617259979248, 4.45687198638916, 1.7004138231277466, 0.4111104905605316, 0.5971472859382629, 0.1582866609096527, -2.8106045722961426, 0.7030100226402283, -2.168276309967041, 0.8870190978050232, -3.9134023189544678, -0.624634861946106, 1.0394450426101685, 10.335195541381836, 1.927688717842102, -0.5027769207954407, 0.434052973985672, -2.493410110473633, -0.03209918364882469, 0.992982029914856, -1.3995782136917114, 4.286319732666016, 0.6480700969696045, -0.06505617499351501, -3.302013397216797, -3.4909026622772217, 0.6191275715827942, 1.0196105241775513, -0.2817462384700775, 0.6629586815834045, 1.776738166809082, 0.24301083385944366, 0.3659611642360687, 0.9018333554267883, -0.522107720375061, -5.6323018074035645, -0.1965782344341278, -7.226038932800293, -1.66015625, -9.336026191711426, -0.6886969804763794, -1.468734622001648, 0.9354947209358215, -0.6427335143089294, 0.7340554594993591, -3.6028075218200684, -1.107383370399475, 1.885650396347046, -0.8130537271499634, -5.0945725440979, -3.242049217224121, -0.49260273575782776, -0.6006614565849304, -0.7200683951377869, -0.5719487071037292, 1.4521818161010742, 1.3725485801696777, -3.923755407333374, 0.2840769290924072, 1.251835584640503, 12.67756175994873, 1.130012035369873, -2.1419482231140137, 1.5956132411956787, -0.6675788164138794, 0.8307718634605408, -0.8655357956886292, 0.041440483182668686, 4.3689398765563965, -0.21070106327533722, -0.6624125838279724, -3.9313154220581055, 4.36356782913208, 3.134377956390381, 1.9760563373565674, -1.7214901447296143, 0.20671844482421875, -2.999441146850586, -1.3124784231185913, -1.3055124282836914, 2.1376450061798096, 0.3834609389305115, -0.8037465810775757, -6.433091163635254, 0.08249175548553467, -1.4673720598220825, -1.3646268844604492, -1.3075027465820312, -1.3599952459335327, -1.8682734966278076, -1.153718113899231, 1.58390212059021, 0.09996749460697174, 0.4131888151168823, 31.610754013061523, -2.739711046218872, 0.17530450224876404, 4.275299072265625, 0.144908145070076, 0.18214254081249237, -0.6035026907920837, -0.9957887530326843, -1.4536339044570923, 1.0180991888046265, -1.5831334590911865, -1.5439305305480957, 4.663516998291016, -0.20570048689842224, -3.5688724517822266, -1.8287898302078247, 1.1660661697387695, -1.2061301469802856, 0.9655909538269043, 0.8304122090339661, 0.39557549357414246, -0.6482387781143188, -6.128785610198975, 0.22646698355674744, 4.3807291984558105, -1.178470253944397, 0.43150901794433594, -0.24322913587093353, -0.9360088109970093, 9.859460988081992e-05, -2.691530466079712, -0.18969382345676422, -2.128782272338867, -2.011314630508423, 0.11578666418790817, -3.9679665565490723, 7.627330780029297, -0.029863595962524414, 2.240694284439087, -2.7655105590820312, -0.1871647983789444, -0.11215285956859589, -4.019465923309326, -0.39783617854118347, -1.6381454467773438, 0.6337056756019592, -1.6667695045471191, -0.28273746371269226, 4.2922773361206055, 0.01821897178888321, -0.2881140410900116, -0.2673991918563843, 3.251394271850586, 17.50492286682129, -0.8621331453323364, -0.13055381178855896, -8.185773849487305, 3.2859549522399902, 1.3698596954345703, 0.2929975092411041, -0.9679856896400452, -3.693674325942993, -6.512032985687256, -1.6374455690383911, 5.575601100921631, -0.7233331799507141, 4.514923095703125, 0.5172720551490784, -1.0552011728286743, -0.8384261131286621, -0.5993791222572327, -3.6433217525482178, 2.2977259159088135, -0.24249868094921112, 0.7852649092674255, -0.3371317684650421, -1.1732105016708374, 0.5911868214607239, 0.7725774645805359, 0.5379520654678345, -4.141595840454102, 3.8947527408599854, 2.2022411823272705, -1.3057692050933838, 0.12295637279748917, -2.168165445327759, -1.557434320449829, 12.452881813049316, -0.586357831954956, 0.6199365854263306, 0.24748463928699493, -0.059226468205451965, -1.310959815979004, 2.69211483001709, -0.02877133898437023, -1.5115824937820435, -3.5188565254211426, 0.5974054932594299, 0.061856579035520554, -2.1786842346191406, 0.016566475853323936, -0.46320462226867676, -0.782708466053009, -2.1081950664520264, -0.8012636303901672, -1.0999201536178589, -0.1280006617307663, -0.9963712096214294, -0.13686323165893555, 0.5716617703437805, 0.0270738136023283, -0.03008587844669819, 7.415950775146484, 0.2277188003063202, 1.60606849193573, 1.5010015964508057, -5.127500534057617, -2.5409369468688965, 1.7799773216247559, 0.26487621665000916, -0.470017671585083, -0.5064870119094849, 1.526637077331543, 0.316962331533432, -1.4813284873962402, -1.4901407957077026, 1.1917208433151245, 4.72705078125, -0.08448757231235504, -2.4134955406188965, 1.129690170288086, 0.38078629970550537, -1.4498014450073242, -1.0454845428466797, 0.5066263675689697, 0.5142555236816406, -2.157322406768799, -0.4456418752670288, 0.7823493480682373, 1.451602816581726, 0.2876157760620117, 0.9025454521179199, -0.732684314250946, 1.0573358535766602, -2.0221333503723145, 4.634834289550781, 0.8198073506355286, 3.1226887702941895, 0.5943845510482788, -0.11916626244783401, -1.3205254077911377, -4.342920780181885, -0.7738397121429443, -1.5749952793121338, 0.6327224373817444, 0.06219453737139702, -2.811392307281494, -2.250253200531006, -1.6474803686141968, -0.03496523201465607, 0.24364283680915833, 0.7859370708465576, 0.9020589590072632, 4.123860836029053, -1.0202250480651855, 0.454041987657547, -0.36021849513053894, -1.8720362186431885, -0.8528254628181458, 2.763584852218628, 5.668804168701172, 7.020084857940674, 1.0516138076782227, -1.0084835290908813, 0.90843665599823, -2.0323827266693115, -0.4164244830608368, 1.9570975303649902, -0.9326475262641907, -1.8250067234039307, -0.4015163481235504, -1.034772515296936, -0.9905200600624084, 1.147081971168518, -3.0287439823150635, 6.062664031982422, -4.553491115570068, -3.319645881652832, 0.9708086252212524, -1.085434913635254, -1.8685874938964844, 0.5577618479728699, 4.35297966003418, -1.5132591724395752, -0.6779425144195557, -0.7730307579040527, -1.1470812559127808, 1.140047311782837, 1.5151000022888184, 1.1505180597305298, -1.0837095975875854, -6.135420799255371, 0.010354878380894661, 0.7545965313911438, 0.5855190753936768, -2.8252437114715576, 2.4199512004852295, -0.2866272032260895, 1.5966675281524658, -0.859708309173584, -2.1297264099121094, -0.8194984197616577, -0.23749469220638275, -4.124383926391602, -4.223925590515137, -1.690342903137207, -0.9706557393074036, -3.614011526107788, -4.080682277679443, -2.2479817867279053, -1.5652663707733154, -3.013709306716919, -0.6558228135108948, -1.2626277208328247, 0.30888235569000244, 2.019538640975952, -1.107478141784668, -1.1136915683746338, -1.6895172595977783, -1.1573294401168823, -0.23446771502494812, -0.6087369322776794, 0.4274353086948395, -0.25699126720428467, -0.17475882172584534, -0.18478764593601227, -1.166336178779602, -4.003023624420166, 4.334836006164551, 0.2825196087360382, -1.2992439270019531, -2.254152536392212, -4.237626552581787, -3.172529935836792, -1.1465256214141846, -4.379649639129639, -1.4892401695251465, -5.194952011108398, -2.0646862983703613, -0.6877126693725586, -0.2274341732263565, 0.6761301755905151, 0.07377849519252777, 0.7501119375228882, 0.3080677092075348, 0.42021313309669495, -0.5513622164726257, -2.3411762714385986, -2.9331395626068115, 4.375935077667236, -1.6950353384017944, 0.14944645762443542, 1.2936118841171265, 0.6517611742019653, -1.239338755607605, -1.7403903007507324, -1.1562265157699585, -1.0772056579589844, -0.7647058963775635, -2.519817590713501, -1.0081744194030762, 0.2803581655025482, 0.16829992830753326, -1.9272254705429077, -1.2308391332626343, -0.20494844019412994, 0.004607878625392914, -0.6571950316429138, -0.9091052412986755, -3.187297821044922, 1.1743210554122925, -1.3116704225540161, -1.1992607116699219, 2.4892189502716064, 1.2639400959014893, -1.5336233377456665, -4.100111484527588, 5.567183494567871, 1.9791017770767212, 4.358534336090088, 0.295236736536026, 2.193591833114624, -2.80304217338562, -2.9566707611083984, 0.20959830284118652, -3.497114896774292, 0.7604942321777344, -1.1389578580856323, -2.8590149879455566, -2.4059112071990967, -2.7033309936523438, 0.8446967005729675, 1.7031917572021484, 9.062850952148438, -4.930017948150635, 0.2463369220495224, -0.4156109690666199, 0.3741205632686615, 0.4064079523086548, -1.9150269031524658, -0.3220655024051666, -0.4442598223686218, 8.454795837402344, -0.06325795501470566, -0.49324727058410645, 0.8758127689361572, -2.4623169898986816, -0.24843661487102509, -4.943604946136475, -0.7193211913108826, 4.274684429168701, -1.0781186819076538, -0.5025853514671326, 20.206626892089844, -11.273247718811035, 0.9051839709281921, 3.4423751831054688, -2.722003936767578, 0.06503109633922577, 4.2576165199279785, -0.08223950117826462, 3.5707499980926514, 0.4434191584587097, -1.386322259902954, -1.65432870388031, 0.30956217646598816, -2.0080478191375732, 0.5522260665893555, -6.115539073944092, 0.8185729384422302, -3.387054443359375, 1.0823416709899902, 0.6622709631919861, 0.8314149379730225, -1.9925364255905151, 3.8954222202301025, 1.7287553548812866, 4.510122299194336, 0.3114977180957794], "yaxis": "y"}],                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "autotypenumbers": "strict", "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Coord0"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "Coord1"}}},                        {"scrollZoom": true, "responsive": true}                    ).then(function(){

var gd = document.getElementById('7f418e2e-7a72-49ae-9499-ab0325968431');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```

```
