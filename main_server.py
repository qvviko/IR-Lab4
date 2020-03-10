import json
import logging
from threading import Thread, Lock
from time import sleep
from typing import Iterable
import sys
from flask import Flask, render_template, make_response
import os
from flask import send_from_directory, request, redirect
from pymongo.errors import BulkWriteError

from song_processing import update_indexes, parse_query, \
    search_in_index, PrefixT, get_id, make_soundex_and_tree
import nltk
from pymongo import MongoClient

app = Flask(__name__, static_url_path='/static')
# Load nltk defaults
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Init indexes and other variables
soundex_index = {}
inverted_index = {}
prefix_tree = PrefixT('')
n_documents = 0
t_before_save = 10
max_doc_on_page = 100

# Init db connection
# TODO: check sharding?
client = MongoClient(os.environ.get("MONGO_URL"), int(os.environ.get("MONGO_PORT")),
                     username=os.environ.get("MONGO_USER"), password=os.environ.get("MONGO_PASS"))
db = client['IR-db']
seen_db = db['seen']
doc_db = db['docs']
vocab_db = db['vocab']
index_db = db['index']
deleted_db = db['deleted']

# Make indexes on primary key for index and docs
index_db.create_index('word', unique=True)
doc_db.create_index('id', unique=True)

# Create lock for writing to the dbs
write_lock = Lock()

# Get idx for set of deleted docs
if deleted_db.find_one() is None:
    returned = deleted_db.insert_one({'deleted': []})
    deleted_idx = returned.inserted_id
else:
    deleted_idx = deleted_db.find_one()['_id']

# Get idx for set of seen docs
if seen_db.find_one() is None:
    seen_db.insert_one({'seen': []})
seen_idx = seen_db.find_one()['_id']
seen_aux = []

# Get idx for vocabulary
vc_db_res = vocab_db.find_one()
if vc_db_res is None:
    returned = vocab_db.insert_one({'vocab': []})
    vocab_idx = returned.inserted_id
    vocab_aux = set()
else:
    vocab_aux = set(vc_db_res['vocab'])
    vocab_idx = vc_db_res['_id']

# Set up logger
app_logger = logging.getLogger('main_app')
app_logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
app_logger.addHandler(logging.StreamHandler(sys.stdout))


def store_aux_info():
    # Function for storing auxiliary information - seen documents, inverted index and vocabulary
    for key, value in inverted_index.items():
        word_db = index_db.find_one({'word': key})
        if word_db is None:
            index_db.insert_one({'word': key, 'docs': list(value)})
        else:
            index_db.update_one({'_id': word_db['_id']}, {'$addToSet': {'docs': {'$each': list(value)}}})
    inverted_index.clear()
    app_logger.debug("Spilled inverted index to mongo")

    seen_db.update_one({'_id': seen_idx}, {'$addToSet': {'seen': {'$each': seen_aux}}})
    seen_aux.clear()
    app_logger.debug("Spilled seen docs to mongo")

    vocab_db.update_one({'_id': vocab_idx}, {'$set': {'vocab': list(vocab_aux)}})
    vocab_aux.clear()
    app_logger.debug("Spilled vocabulary to mongo")


def open_index(word):
    # Helper function for searching for words in mongo
    return set(index_db.find_one({'word': word})['docs'])


def search_in_mongo(query):
    # Algorithm is almost the same as for inverted index, but we use open_idex instead
    # The query has the following structure:
    # It is a list of lists or string
    # If an element of the query is a list - all docs of items of this list are OR'ed
    # And then they will be AND'ed with each other
    if not query:
        return set()
    docs = set()

    # Take the initial docs set
    if isinstance(query[0], list):
        # If the value is a list - take OR over all values
        l = query[0]
        docs = open_index(l[0])

        for i in l[1:]:
            docs |= open_index(i)
    else:
        # If the value is a string -just take it
        docs = open_index(query[0])

    for words in query[1:]:
        # Now go over remaining items
        if isinstance(words, list):
            # If the list - OR it's items and then AND with the current docs
            new_docs = open_index(words[0])
            for word in words[1:]:
                new_docs |= open_index(word)
            docs &= new_docs

        else:
            # If string - just AND with the current docs
            docs &= open_index(words)
    return docs


def store_daemon():
    # Daemon for spilling aux information to the disk
    while True:
        global n_documents
        sleep(t_before_save)
        app_logger.debug(f'Timeout, spilling to disk..')
        with write_lock:
            store_aux_info()
            n_documents = 0


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/doc/<id>')
def doc(id):
    # Page to check document
    doc = doc_db.find_one({'id': id})
    if doc is None:
        return make_response('No doc found', 404)
    doc['text'] = doc['text'].split("\n")
    return render_template('song.html', doc=doc)


@app.route('/search', methods=['GET'])
def search():
    # Page to search for the documents
    if 'query' not in request.args.keys():
        return redirect("/")

    # Firsly - parse a query
    app_logger.debug(f'Received a query {request.args["query"]}')
    new_query = parse_query(inverted_index, soundex_index, prefix_tree, index_db, request.args['query'])

    # Then check in the aux index
    documents = search_in_index(inverted_index, new_query)
    app_logger.debug(f"Search in index found {documents}")

    # Then in the mongo index
    mongo_documents = search_in_mongo(new_query)
    app_logger.debug(f"Search in mongo found {mongo_documents}")

    # Combine and remove deleted docs
    documents |= mongo_documents
    documents -= set(deleted_db.find_one()['deleted'])
    app_logger.debug(f"After removing deleted documents {documents}")

    # Get the doc info, remove '_id' from them and give out max_doc_on_page of them to the user
    to_return = list(doc_db.find({'id': {'$in': list(documents)}}))
    for r in to_return:
        r.pop('_id')
    app_logger.debug(f"Got a {len(to_return)} of documents, returning")
    return render_template('search.html', query=request.args['query'], documents=to_return[:max_doc_on_page])


@app.route('/update', methods=['POST'])
def update():
    # Update endpoint for crawler to send it's docs
    global n_documents
    app_logger.debug(f'Received a doc, number of documents in aux index is {n_documents}')
    js = json.loads(request.json)
    if not isinstance(js, list):
        js = [js]

    # Acquire lock, update indexes
    with write_lock:
        update_indexes(js, inverted_index, soundex_index, prefix_tree, vocab_aux)

        # Update seen aux
        for doc in js:
            seen_aux.append(get_id(doc['url']))
            app_logger.debug(f'Recevied docid: {seen_aux[-1]}')
        app_logger.debug(f"Current seen_aux is {len(seen_aux)}")
        app_logger.debug(f"Saving documents to mongodb")

        # Try to insert - if fails, the document already in the mongo!
        try:
            doc_db.insert_many(js)
        except BulkWriteError as bwe:
            app_logger.error(bwe.details)
            make_response("Problem on save", 400)

        n_documents += 1

    return make_response('Updated', 200)


@app.route('/delete/<id>', methods=['DELETE'])
def delete(id):
    # Endpoint for marking document as deleted
    deleted_db.update_one({'_id': deleted_idx}, {'$addToSet': {'deleted': id}})
    return make_response(f'Doc {id} deleted', 204)


if __name__ == "__main__":
    app_logger.debug(f'Restoring soundex and prefix tree')
    soundex_index, prefix_tree = make_soundex_and_tree(vocab_db.find_one()['vocab'])
    daemon_thread = Thread(target=store_daemon, daemon=True)
    daemon_thread.start()
    app.run(host=os.environ.get("MAIN_SERVER_URL", '0.0.0.0'), port=int(os.environ.get("MAIN_SERVER_PORT", 8080)),
            debug=True)
