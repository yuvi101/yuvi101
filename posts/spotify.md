---
title: "Building a Spotify Recommendation System"
date: 2022-04-11T11:21:51+03:00
#draft: true
author: "Yuval"


featured_image: "/posts/spotify_logo.jpg"

resources:
  -name: "/posts/spotify_logo.jpg"
  src: "/posts/spotify_logo.jpg"

tags: ["Python", "API", "Machine Learning"]
categories: ["Recommendation Systems"]

---
In this post I'll show the steps I took in order to build my own spotify recommendation system that will automaticlly add new playlists to my account based on the songs I liked.
![spotify logo](https://images.unsplash.com/photo-1634037227397-34c8c46d585c?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80)

### Using Spotify API and gathering data
Importing all the packages we are gonna need:
```python
import spotipy
from sklearn.neighbors import KNeighborsClassifier
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
```
Defining personal information that we will later need in order to work with the API:
```python
client_ID ='YOUR_CLIENT_ID'
client_SECRET='YOUR_CLIENT_SECRET'
redirect_URL='http://localhost:9000'
user = 'YOUR_USERNAME' #you can get this by clicking on your account at the top right corner of spotify app/web
```

```python
def authentication(scope):
    return spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_ID,
                           client_secret=client_SECRET, redirect_uri=redirect_URL, scope=scope))
```

```Python
scope = "user-read-recently-played"
sp = authentication(scope)
playlists = sp.current_user_playlists()
playlists_ids = []
for idx, item in enumerate(playlists['items']):
    playlists_ids.append(item['id'])
```
Getting a list of all track ids from my playlists:
```python
all_tracks = []
for pid in playlists_ids:
    auth_manager = SpotifyClientCredentials(client_ID, client_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    playlist_tracks = sp.playlist_items(playlist_id=pid , fields='items,name,id')
    for idx, item in enumerate(playlist_tracks['items']):
        track = item['track']
        all_tracks.append((track['artists'][0]['name'], track['name'], track['id'], pid))
all_tracks = list(map(list, all_tracks))
```
Converting the list to a dataframe and adding to each track audio features such as: acousticness, energy, instrumentalness, liveness, loudness etc.
```python
values = []
keys = (sp.audio_features(all_tracks[0][2]))[0].keys()
for i in range(len(all_tracks)):
    r = sp.audio_features(all_tracks[i][2])
    for key in r[0]:
        values.append(r[0][key])
    all_tracks[i] = all_tracks[i] + values
    values = []
df = pd.DataFrame(all_tracks, columns = ['artist_name', 'track_name', 'track_id', 'playlist_id'] + list(keys))```
```
Instead off ranking by hand my tracks I gave the rank 0 to all tracks in a specific playlist and rank=1 to all the other.
```python
df['ratings'] = df['playlist_id'].apply(lambda x: 1 if x == '37i9dQZF1DWYbUY40ZDAwb' else 0)
```

Spliting the data to train and test:
```python
y = df.iloc[:, 22:23]
X1 = df.iloc[:, 4:14]
X2 = df.iloc[:, 20:22]
X = pd.concat([X1,X2], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
```
### The ML models
Chossing the optimal number of features for the random forest classifier:
```python
rf = RandomForestClassifier(n_estimators=1000, random_state=42)
rfecv = RFECV(estimator=rf, step=1, n_jobs=-1, cv=StratifiedKFold(2), verbose=1, scoring='roc_auc')
rfecv.fit(X_train, y_train.values.ravel())
print("Optimal number of features: {}".format(rfecv.n_features_))
```

Rescaling all the features to be between 0 and 1:
```python
minmax_scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(minmax_scaler.fit_transform(X_train), columns=X_train.columns)
```
Implementing the random forest classifier:
```python
rfc = RandomForestClassifier(n_estimators=1000, random_state=42)
rfc_gcv_parameters = {'min_samples_leaf': [1, 3, 5, 8], 'max_depth': [3, 4, 5, 8, 12, 16, 20],}
rfe_gcv = GridSearchCV(rfc, rfc_gcv_parameters, n_jobs=-1, cv=StratifiedKFold(2), verbose=1, scoring='roc_auc')
rfe_gcv.fit(X_train, y_train.values.ravel())
rfe_gcv.best_estimator_, rfe_gcv.best_score_
print('random forest classifier:')
print(classification_report(y_test.values.ravel(), rfe_gcv.predict(X_test)))
```

Implementing the k-nearest-neighbors classifier:
```Python
knn = KNeighborsClassifier(n_jobs=-1)
knn_gcv_params = {'n_neighbors': range(1, 10)}
knn_gcv = GridSearchCV(knn, knn_gcv_params, n_jobs=-1, cv=StratifiedKFold(2), verbose=1, scoring='roc_auc')
knn_gcv.fit(X_train, y_train.values.ravel())
knn_gcv.best_params_, knn_gcv.best_score_
print('knn:')
print(classification_report(y_test.values.ravel(), knn_gcv.predict(X_test)))
```

### Using the Random Forest model to recommend:
I used kaggle's dataset (https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify?select=genres_v2.csv) as a source from which the algorithm will find tracks with the highest probability to be added to my playlist.
```Python
spotify_dataset = pd.read_csv(r'address_of_your_file')
X = X[X.columns.intersection(spotify_dataset.columns)]
X_recommend = spotify_dataset.copy()
X_recommend = X_recommend[X_recommend.columns.intersection(X.columns)]
spotify_dataset['ratings'] = rfe_gcv.predict(X_recommend)
spotify_dataset['prob_ratings'] = rfe_gcv.predict_proba(X_recommend)[:,1]
result = (spotify_dataset[spotify_dataset['ratings'] ==  1].sort_values(by='prob_ratings', ascending=False))
result = result.drop_duplicates(subset=['song_name'], keep='last')
```

```Python
print(result[['song_name', 'id']].head(20))
```
The output:
```text
                                     song_name                      id
14041                                      Bet  1hv5qektVlfqlu1P858s5G
21346                                     Ride  0yNi8hNyv0DzMVFtKIAP1S
20300                                  RACECAR  7JdarpX08FLhmfNZpxqgUO
9418                                  Blackout  5TEQB7WfKZZNhNKoaMRiB7
9715                          Grandma's Porch.  7r7O86q51J9iHzfjOcwlTj
9575                                  Cash App  4bMLzfbjYJ9v3wvlpI6wtE
18412                Prisoner (feat. Dua Lipa)  5JqZ3oqF00jkT81foAFvqg
9895                                        C4  3NiqzLj51KIwM9yMfai276
18404                         Watermelon Sugar  6UelLqGlWMcVH1E5c4H7lY
7992                                  Take One  6fI3tBVND8zUXZi9rr2Yps
20390                             First Person  6bvV5L5afKykg819xAIJWt
7583                          Let My People Go  07ZzLIfJvo14UJyhKjN3z4
15873                                     Cp24  356tIL4ewLC8zHQCjhMrrF
6322                            Do What I Want  4IWGnyOHDrVZEtPWfs4s7q
10927                             Laid To Rest  1iDaAHOQvaxWGXx0VMYwAd
287                                Analog Keys  7yMvF3mjdsFStdOiMpiFNx
1126                          Over the Rainbow  5ocuRCDSWiUMZcWI4Utd9g
15130       Out Of Love (feat. Internet Money)  0IJA9KP6rT55jrP1YpTdhx
20067  Creepshow (feat. IDK & Chip tha Ripper)  0iLfMB2S2ilazjeFH91NiT
18420                          You Got Me Like  2oygttOZA8dTFxHevUYGKm
```

### Add the recoomended tracks to my account
```Python
tracks_to_add = result[['id']].head(20)
scope = "playlist-modify-private"
sp = authentication(scope)
add_new_playlist = sp.user_playlist_create(user=user, name="recommended #1", public=False)
sp.playlist_add_items(playlist_id=add_new_playlist['id'],items=tracks_to_add['id'])
```
### Final result
All tracks had been added to my account successfuly and a new playlist were added. Although the results are not perfect (the dataset which I used does not contain the latests hits and my model does not take into consideration the 'popularity' factor) I do like some of the recommended songs, but I may be biased :wink:.
![spotify_screenshot](/posts/spotify_screenshot.png)
