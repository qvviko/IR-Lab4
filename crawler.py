import logging
import sys
from time import sleep

import requests
from bs4 import BeautifulSoup
import json
import os

from pymongo import MongoClient

lyrics_url = "https://www.lyrics.com/random.php"
error_url = "https://www.lyrics.com/no-lyrics.php"
base_url = "https://www.lyrics.com/lyric/"
seen = []

# Setup db connection (for restoring seen documents)
client = MongoClient(os.environ.get("MONGO_URL"), int(os.environ.get("MONGO_PORT")),
                     username=os.environ.get("MONGO_USER"), password=os.environ.get("MONGO_PASS"))
db = client['IR-db']
app_logger = logging.getLogger('crawler')
app_logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
app_logger.addHandler(logging.StreamHandler(sys.stdout))


# Class where we store all songs
class Song:
    def __init__(self, url=None):
        self.url = url
        self.title = None
        self.text = None
        self.artists = None
        self.get()

    def __str__(self):
        return f"Song {self.title} by {', '.join(self.artists)}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.url == other.url

    def __hash__(self):
        return self.title.__hash__()

    def get_id(self):
        return self.url[len(base_url):].split("/")[0]

    def get(self):
        # Try to get from the directory, if not - download a random song from the website

        if not self.download():
            raise FileNotFoundError(self.url)
        app_logger.info(f"Loaded {self.url}")

    def download(self):
        try:
            # Downloads a random songs from the website
            page = requests.get(lyrics_url)

            # Store url
            self.url = page.url
            if page.url == error_url:  # If broken url - skip
                return False
            soup = BeautifulSoup(page.content, 'html.parser')
            # Get title
            self.title = soup.find(["h1", "h2"], {"id": "lyric-title-text"}).text
            # Get artist name
            artists = soup.find("h3", {"class": "lyric-artist"}).findAll('a')
            self.artists = []
            for artist in artists[:-1]:  # The last item is this list is "buy this song" promotions
                self.artists.append(artist.text)
                # Get song body
            self.text = soup.find("pre", {"class": "lyric-body"}).text.replace("\r", "")
        except Exception as e:
            print(e)
            return False
        return True


def get_songs(seen):
    # loads lyrics
    # Otherwise download the remaining at random
    while True:
        try:
            s = Song()
        except FileNotFoundError:
            continue
        if s.get_id() not in seen:
            seen.append(s.get_id())
            yield s


def restore_seen():
    try:
        return db['seen'].find_one()['seen']
    except IndexError:
        return []


# Make server url (to which we will send docs)
server_url = f'http://{os.environ.get("MAIN_SERVER_URL", "localhost")}:{os.environ.get("MAIN_SERVER_PORT", 8080)}'
if __name__ == "__main__":
    # Restore already seen documents
    seen = restore_seen()
    if seen is None:
        seen = []
    app_logger.info(f"Restored n={len(seen)} of seen")
    while True:
        # Finds an article and sends it
        for song in get_songs(seen):
            # Send parsed article to the server
            app_logger.debug(f"Sending {song.url} to the main server")
            js = {'title': song.title, 'artists': song.artists, 'text': song.text, 'url': song.url, 'id': song.get_id()}
            answ = requests.post(server_url + '/update', json=json.dumps(js))
            sleep(0.1)
