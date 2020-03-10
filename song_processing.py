import json
import logging
import os
import re
import nltk
import numpy as np

from crawler import base_url, Song

processing_logger = logging.getLogger('processing')
processing_logger.setLevel(logging.DEBUG)


def normalize(text, query=False):
    text = re.sub("\d+", "", text.lower())  # Lower, no digits
    text = text.replace("_", " ")  # Replace _ to space
    text = text.replace("\n", " ")
    if query:
        text = re.sub("[^\w\s*]", " ", text)  # No punctuactions except *
        text = re.sub("[^a-z  *]", "", text)  # Remove non english letters except *
    else:
        text = re.sub("[^\w\s]", " ", text)  # No punctuactions
        text = re.sub("[^a-z ]", "", text)  # Remove non english letters
    text = re.sub(' +', ' ', text)  # Remove repeated whitespaces
    return text


def tokenize(text):
    return nltk.word_tokenize(text)


def lemmatization(tokens):
    # Lemmatize text using nltk
    lemm = nltk.stem.WordNetLemmatizer()
    return [lemm.lemmatize(token) for token in tokens]


def remove_stop_word(tokens):
    # Remove stopwords with nltk
    return [token for token in tokens if not token in nltk.corpus.stopwords.words('english')]


class PrefixT:
    def __init__(self, letter, parent=None):
        self.children = {}  # Children - dictionary that maps next letter to the parent
        self.l: str = letter  # The content of the node, letter
        self.parent = parent  # Parent of the node

    def add_word(self, word: str):
        word = word + "@"  # @ will indicate end of the word
        cur_tree = self

        # Iteratively add every letter to the tree while going deeper
        for letter in word:
            if not letter in cur_tree.children:
                cur_tree.children[letter] = PrefixT(letter, cur_tree)
            cur_tree = cur_tree.children[letter]

    def get_words(self, cur_word=''):
        # Gets all possible words from self node, doesn't return prefix, DFS
        cur_word += self.l
        if not self.children:
            return {cur_word}
        to_return = set()
        for tree in self.children.values():
            to_return = to_return.union(tree.get_words(cur_word))
        return to_return

    def prefix(self):
        # Returns prefix (with itself) of the words
        prefix = self.l
        if self.parent is None:
            return prefix
        else:
            return self.parent.prefix() + prefix

    def __eq__(self, other):
        return self.l == other.l

    def __hash__(self):
        return self.l.__hash__()

    def __str__(self):
        return f"PrefixT({self.l})"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        # Finds the string in the current node with name "item"
        cur_child = self
        if len(item) == 1:
            if item in self.children:
                return self.children[item]
            else:
                return None
        for i in item:
            if not i in cur_child.children:
                return None
            cur_child = cur_child[i]
        return cur_child


def get_rotations(word):
    # Performs all rotations
    wordword = word + word
    to_return = set()
    length = len(word)

    for i in range(length):
        to_return.add(wordword[i:length + i])  # slide over the wordword and get all rotations

    return to_return


soundex_dictionary = {'a': 0, 'e': 0, 'i': 0, 'o': 0, 'u': 0, 'y': 0, 'h': 0, 'w': 0,
                      'b': 1, 'f': 1, 'p': 1, 'v': 1,
                      'c': 2, 'g': 2, 'j': 2, 'k': 2, 'q': 2, 's': 2, 'x': 2, 'z': 2,
                      'd': 3, 't': 3, 'l': 4, 'm': 5, 'n': 5, 'r': 6}


def make_soundex(word):
    soundex = word[0].upper()  # take the first letter
    previous = ''
    for letter in word[1:]:
        if len(soundex) == 4:  # Stop at 4 letters or when the word have been exhauseted
            break
        number = soundex_dictionary[letter]

        # If the number is not 0 - add it to the soundex
        if number and number != previous:
            soundex += str(number)
        previous = number
    # If the length of the word isn't 4 - fill it with 0s
    if len(soundex) != 4:
        soundex += '0' * (4 - len(soundex))
    return soundex


def levenshtein_distance(word1, word2):
    matrix = np.zeros((len(word1) + 1, len(word2) + 1))  # Initial matrix

    # Fill the corner with ranges of the words
    for i in range(len(word2) + 1):
        matrix[0][i] = i
    for i in range(len(word1) + 1):
        matrix[i][0] = i

    # Go over matrix and compute the distance
    for i in range(1, len(word1) + 1):
        for j in range(1, len(word2) + 1):
            matrix[i][j] = min(matrix[i - 1][j - 1] + (1 if word1[i - 1] != word2[j - 1] else 0),
                               matrix[i][j - 1] + 1, matrix[i - 1][j] + 1)
    # Take lower right corner - this is an answer
    return matrix[-1][-1]


def preprocess(text):
    # Now returns two lists
    # First one with lemmatization
    # Other one without lemmatization
    text = tokenize(normalize(text))
    return remove_stop_word(lemmatization(text)), remove_stop_word(text)  # All-in-one


def get_id(song_url):
    return song_url[len(base_url):].split("/")[0]


def update_indexes(documents, inverted_index, soundex_index, prefix_tree, vocab_aux):
    for song in documents:

        # Clean the text, then add words to the index
        clean, unlemmatized = preprocess(song['text'])

        for word in clean:
            if not word in inverted_index:
                inverted_index[word] = {
                    get_id(song['url'])}
                # If we haven't encountered the word previously - create set with it
            else:
                inverted_index[word].add(get_id(song['url']))  # Otherwise add to the set

        # Now go over unlemmatized part
        vocab_aux |= set(unlemmatized)
        make_soundex_and_tree(unlemmatized, soundex_index, prefix_tree)


def make_soundex_and_tree(words, soundex_index=None, prefix_tree=None):
    if soundex_index is None:
        soundex_index = {}
    if prefix_tree is None:
        prefix_tree = PrefixT('')
    for word in words:

        rotations = get_rotations(word + "$")

        # Add rotations to the prefix tree
        for rot in rotations:
            prefix_tree.add_word(rot)
        sound = make_soundex(word)

        # Add sounds to the soundex index
        if not sound in soundex_index:
            soundex_index[sound] = {word}
        else:
            soundex_index[sound].add(word)
    return soundex_index, prefix_tree


def rotate_back(word):
    # Rotate the word back to the original position and clean the end notifier (@)

    while word[-1] != "$":
        word = word[-1] + word[:-1]

    return word.replace("@", "")[:-1]


def wildcard_query(word, prefix_tree):
    # Perform wildcard querying (supports only single *)
    if not "*" in word:
        return None

    word = word + "$"

    # Rotate until * is in the end
    while word[-1] != "*":
        word = word[-1] + word[:-1]

    # Find possible words in the prefix tree (removing *)
    found = prefix_tree[word[:-1]].get_words()

    # Process answer
    words = list(
        map(
            lambda x: rotate_back(word[:-2] + x),
            list(found)
        )
    )
    return words


def query_prettifier(query):
    # Prettify query for the print
    query = query.copy()
    for i in range(len(query)):
        if isinstance(query[i], list):
            query[i] = f'({" || ".join(query[i])})'
    return " && ".join(query)


def parse_query(inverted_index, soundex_index, prefix_tree, index_db, query):
    # Clean the query
    clean_query = remove_stop_word(tokenize(normalize(query, True)))

    if not clean_query:
        return []  # Query empty - return nothing
    new_query = []

    for word in clean_query:
        # If the word in the index - just add it to the query
        if word in inverted_index or index_db.find_one({'word': word}) is not None:
            new_query.append(word)
        # If the word has *, then perform wildcard query
        # Then lemmatize the result (as we will search in the inverted_index) and remove stop words
        elif '*' in word:
            items = wildcard_query(word, prefix_tree)
            items = remove_stop_word(lemmatization(items))
            new_query.append(list(set(items)))

        # If this word is still unknown - try to fix it
        # find it in soundex_index the query, compute the distance for all words found and take ones with
        # minimum distance
        else:

            soundex = make_soundex(word)
            if soundex in soundex_index:
                sound_words = soundex_index[soundex]
                distances = []

                for w in sound_words:
                    distances.append(levenshtein_distance(w, word))

                minimum = min(distances)
                final = [sound_word for sound_word, dis in zip(sound_words, distances) if dis == minimum]
                # Lemmatize words and remove stop words (as we will use inverted_index)
                final = remove_stop_word(lemmatization(final))
                new_query.append(list(set(final)))
            else:
                return []  # No docs with this word

    if not new_query:
        return []  # Empty query
    print(f"Query was perceived as: {query_prettifier(new_query)}")

    # Sort by strings first - lists last
    new_query = sorted(new_query, key=lambda x: isinstance(x, list))
    return new_query


def search_in_index(inverted_index, new_query):
    # The query has the following structure:
    # It is a list of lists or string
    # If an element of the query is a list - all docs of items of this list are OR'ed
    # And then they will be AND'ed with each other
    if not new_query:
        return set()
    docs = set()

    try:
        # Take the initial docs set
        if isinstance(new_query[0], list):
            # If the value is a list - take OR over all values
            l = new_query[0]
            docs = inverted_index[l[0]].copy()

            for i in l[1:]:
                docs |= inverted_index[i]
        else:
            # If the value is a string -just take it
            docs = inverted_index[new_query[0]].copy()

        for words in new_query[1:]:
            # Now go over remaining items
            if isinstance(words, list):
                # If the list - OR it's items and then AND with the current docs
                new_docs = inverted_index[words[0]].copy()
                for word in words[1:]:
                    new_docs |= inverted_index[word]
                docs &= new_docs

            else:
                # If string - just AND with the current docs
                docs &= inverted_index[words]
        return docs
    except KeyError:
        return docs


def find_by_query(query, inverted_index, soundex_index, prefix_tree):
    new_query = parse_query(inverted_index, soundex_index, prefix_tree, query)
    docs = search_in_index(inverted_index, new_query)
    return docs


index_dir = './index'


def store_index(inverted_index):
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    processing_logger.debug(f'Storing inverted index of length {len(inverted_index.keys())}')
    while inverted_index:
        word, doc_ids = inverted_index.popitem()
        idx_file = os.path.join(index_dir + word)
        if os.path.exists(idx_file):
            index = open(idx_file, 'r')
            previous_index = json.loads(index.read())
            to_store = list(doc_ids) + previous_index
            index.close()
        else:
            to_store = list(doc_ids)
        index = open(idx_file, 'w+')
        index.write(json.dumps(to_store))
