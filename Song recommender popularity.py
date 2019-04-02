# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:32:20 2018

@author: Ramiro GM
"""

import pandas
import numpy as np
import Recommender
import item_similarity_recommender 
# =============================================================================
# The triplet_file contains user_id, song_id and 
# times listened. 
# The songs_metadata_file contains song_id,
#  title, release_by and artist_name.
# =============================================================================

triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'
#songs_metadata_file = 'https://raw.githubusercontent.com/vyashemang/popularity_based_recommendation/master/song_data.csv'

song_df_1 = pandas.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']
# merging the 2 datasets
song_df_2 = pandas.read_csv(songs_metadata_file)
song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")
# to see all the columns
song_df = song_df.head(10000)

# if I'd like to include in one column the song name and the
# artist
song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']

# groups the dataframe according the maximum listening counts
song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
#Sums the listening counts of each song
grouped_sum = song_grouped['listen_count'].sum()
# to be able to see the whole columns and rows in the terminal
pandas.set_option('display.expand_frame_repr', False)    
# getiing the % of popularity
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])

users = song_df['user_id'].unique()
len(users)
items = song_df['song'].unique()
len(items)

from sklearn.cross_validation import train_test_split
train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)

model = Recommender.Popularity_Recommender()
model.create(train_data, 'user_id', 'song')
model.recommend(users[10])

# By item-similarity collaborative filtering approach

item_model = item_similarity_recommender.item_similarity_recommender()
item_model.create(train_data, 'user_id', 'song')

# predicting the songs that a user will probably like

#Print the songs for the user in training data
user_id = users[15]
user_items = item_model.get_user_items(user_id)
#
print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)
    
    print("----------------------------------------------------------------------")
    print("Recommendation process going on:")
    print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
recomended = item_model.recommend(user_id)
# finding similar songs to any songs in our dataset
similar_songs = item_model.get_similar_items(['U Smile - Justin Bieber'])
#item_model.generate_top_recommendations(user_id, cooccurence_matrix,items,users)