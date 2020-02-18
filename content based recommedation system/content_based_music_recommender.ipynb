{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Music recommender system\n",
    "\n",
    "One of the most used machine learning algorithms is recommendation systems. A **recommender** (or recommendation) **system** (or engine) is a filtering system which aim is to predict a rating or preference a user would give to an item, eg. a film, a product, a song, etc.\n",
    "\n",
    "Which type of recommender can we have?   \n",
    "\n",
    "There are two main types of recommender systems: \n",
    "- Content-based filters\n",
    "- Collaborative filters\n",
    "  \n",
    "> Content-based filters predicts what a user likes based on what that particular user has liked in the past. On the other hand, collaborative-based filters predict what a user like based on what other users, that are similar to that particular user, have liked.\n",
    "\n",
    "### 1) Content-based filters\n",
    "\n",
    "Recommendations done using content-based recommenders can be seen as a user-specific classification problem. This classifier learns the user's likes and dislikes from the features of the song.\n",
    "\n",
    "The most straightforward approach is **keyword matching**.\n",
    "\n",
    "In a few words, the idea behind is to extract meaningful keywords present in a song description a user likes, search for the keywords in other song descriptions to estimate similarities among them, and based on that, recommend those songs to the user.\n",
    "\n",
    "*How is this performed?*\n",
    "\n",
    "In our case, because we are working with text and words, **Term Frequency-Inverse Document Frequency (TF-IDF)** can be used for this matching process.\n",
    "  \n",
    "We'll go through the steps for generating a **content-based** music recommender system."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Importing required libraries\n",
    "\n",
    "First, we'll import all the required libraries."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from typing import List, Dict"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We have already used the **TF-IDF score before** when performing Twitter sentiment analysis. \n",
    "\n",
    "Likewise, we are going to use TfidfVectorizer from the Scikit-learn package again."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "So imagine that we have the [following dataset](https://www.kaggle.com/mousehead/songlyrics/data#). \n",
    "\n",
    "This dataset contains name, artist, and lyrics for *57650 songs in English*. The data has been acquired from LyricsFreak through scraping."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "songs = pd.read_csv('content based recommedation system/songdata.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>artist</th>\n",
       "      <th>song</th>\n",
       "      <th>link</th>\n",
       "      <th>text</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ABBA</td>\n",
       "      <td>Ahe's My Kind Of Girl</td>\n",
       "      <td>/a/abba/ahes+my+kind+of+girl_20598417.html</td>\n",
       "      <td>Look at her face, it's a wonderful face  \\nAnd...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ABBA</td>\n",
       "      <td>Andante, Andante</td>\n",
       "      <td>/a/abba/andante+andante_20002708.html</td>\n",
       "      <td>Take it easy with me, please  \\nTouch me gentl...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>ABBA</td>\n",
       "      <td>As Good As New</td>\n",
       "      <td>/a/abba/as+good+as+new_20003033.html</td>\n",
       "      <td>I'll never know why I had to go  \\nWhy I had t...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ABBA</td>\n",
       "      <td>Bang</td>\n",
       "      <td>/a/abba/bang_20598415.html</td>\n",
       "      <td>Making somebody happy is a question of give an...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ABBA</td>\n",
       "      <td>Bang-A-Boomerang</td>\n",
       "      <td>/a/abba/bang+a+boomerang_20002668.html</td>\n",
       "      <td>Making somebody happy is a question of give an...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  artist                   song                                        link  \\\n",
       "0   ABBA  Ahe's My Kind Of Girl  /a/abba/ahes+my+kind+of+girl_20598417.html   \n",
       "1   ABBA       Andante, Andante       /a/abba/andante+andante_20002708.html   \n",
       "2   ABBA         As Good As New        /a/abba/as+good+as+new_20003033.html   \n",
       "3   ABBA                   Bang                  /a/abba/bang_20598415.html   \n",
       "4   ABBA       Bang-A-Boomerang      /a/abba/bang+a+boomerang_20002668.html   \n",
       "\n",
       "                                                text  \n",
       "0  Look at her face, it's a wonderful face  \\nAnd...  \n",
       "1  Take it easy with me, please  \\nTouch me gentl...  \n",
       "2  I'll never know why I had to go  \\nWhy I had t...  \n",
       "3  Making somebody happy is a question of give an...  \n",
       "4  Making somebody happy is a question of give an...  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "songs.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Because of the dataset being so big, we are going to resample only 5000 random songs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "songs = songs.sample(n=5000).drop('link', axis=1).reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can notice also the presence of `\\n` in the text, so we are going to remove it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "songs['text'] = songs['text'].str.replace(r'\\n', '')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "After that, we use TF-IDF vectorizerthat calculates the TF-IDF score for each song lyric, word-by-word. \n",
    "\n",
    "Here, we pay particular attention to the arguments we can specify."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "tfidf = TfidfVectorizer(analyzer='word', stop_words='english')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "lyrics_matrix = tfidf.fit_transform(songs['text'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*How do we use this matrix for a recommendation?* \n",
    "\n",
    "We now need to calculate the similarity of one lyric to another. We are going to use **cosine similarity**.\n",
    "\n",
    "We want to calculate the cosine similarity of each item with every other item in the dataset. So we just pass the lyrics_matrix as argument."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "cosine_similarities = cosine_similarity(lyrics_matrix) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Once we get the similarities, we'll store in a dictionary the names of the 50  most similar songs for each song in our dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "similarities = {}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(len(cosine_similarities)):\n",
    "    # Now we'll sort each element in cosine_similarities and get the indexes of the songs. \n",
    "    similar_indices = cosine_similarities[i].argsort()[:-50:-1] \n",
    "    # After that, we'll store in similarities each name of the 50 most similar songs.\n",
    "    # Except the first one that is the same song.\n",
    "    similarities[songs['song'].iloc[i]] = [(cosine_similarities[i][x], songs['song'][x], songs['artist'][x]) for x in similar_indices][1:]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "After that, all the magic happens. We can use that similarity scores to access the most similar items and give a recommendation.\n",
    "\n",
    "For that, we'll define our Content based recommender class."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {},
   "outputs": [],
   "source": [
    "class ContentBasedRecommender:\n",
    "    def __init__(self, matrix):\n",
    "        self.matrix_similar = matrix\n",
    "\n",
    "    def _print_message(self, song, recom_song):\n",
    "        rec_items = len(recom_song)\n",
    "        \n",
    "        print(f'The {rec_items} recommended songs for {song} are:')\n",
    "        for i in range(rec_items):\n",
    "            print(f\"Number {i+1}:\")\n",
    "            print(f\"{recom_song[i][1]} by {recom_song[i][2]} with {round(recom_song[i][0], 3)} similarity score\") \n",
    "            print(\"--------------------\")\n",
    "        \n",
    "    def recommend(self, recommendation):\n",
    "        # Get song to find recommendations for\n",
    "        song = recommendation['song']\n",
    "        # Get number of songs to recommend\n",
    "        number_songs = recommendation['number_songs']\n",
    "        # Get the number of songs most similars from matrix similarities\n",
    "        recom_song = self.matrix_similar[song][:number_songs]\n",
    "        # print each item\n",
    "        self._print_message(song=song, recom_song=recom_song)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now, instantiate class"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {},
   "outputs": [],
   "source": [
    "recommedations = ContentBasedRecommender(similarities)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Then, we are ready to pick a song from the dataset and make a recommendation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [],
   "source": [
    "recommendation = {\n",
    "    \"song\": songs['song'].iloc[10],\n",
    "    \"number_songs\": 4 \n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The 4 recommended songs for Jehovah And All That Jazz are:\n",
      "Number 1:\n",
      "Sing by Glen Campbell with 0.166 similarity score\n",
      "--------------------\n",
      "Number 2:\n",
      "Devil Cried by Black Sabbath with 0.149 similarity score\n",
      "--------------------\n",
      "Number 3:\n",
      "Angelique by Kenny Loggins with 0.141 similarity score\n",
      "--------------------\n",
      "Number 4:\n",
      "Up The Devil's Pay by Old 97's with 0.131 similarity score\n",
      "--------------------\n"
     ]
    }
   ],
   "source": [
    "recommedations.recommend(recommendation)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "And we can pick another random song and recommend again:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {},
   "outputs": [],
   "source": [
    "recommendation2 = {\n",
    "    \"song\": songs['song'].iloc[120],\n",
    "    \"number_songs\": 4 \n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The 4 recommended songs for I Do It For Your Love are:\n",
      "Number 1:\n",
      "I Love You by Lionel Richie with 0.189 similarity score\n",
      "--------------------\n",
      "Number 2:\n",
      "Just One Love by Michael Bolton with 0.187 similarity score\n",
      "--------------------\n",
      "Number 3:\n",
      "I'm Gonna Sit Right Down And Write Myself A Letter by Nat King Cole with 0.184 similarity score\n",
      "--------------------\n",
      "Number 4:\n",
      "If I Love Again by Barbra Streisand with 0.183 similarity score\n",
      "--------------------\n"
     ]
    }
   ],
   "source": [
    "recommedations.recommend(recommendation2)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
