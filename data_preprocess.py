import pandas as pd
import sys
sys. path. append(".")
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreprocess:

    def __init__(self, ds):
        self.ds = ds
        self.columns = ds.columns
        self.initGenreLabelEncoder()

    # initialise the GenreLabelEncoder
    def initGenreLabelEncoder(self):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(
            ["r&b", "rap", "classical", "salsa", "edm", "hip hop", "trap", "techno", "jazz", "metal", "country",
             "rock",
             "reggae", "latin",
             "disco", "soul", "chanson", "blues", "dance", "electro", "punk", "folk", "pop"])

    # returrn a set of artist if it exist
    def getListArtists(self):
        try:
            artists = self.ds["artist_name"]
            artists_set = list(dict.fromkeys(artists))
            return artists_set
        except Exception:
            print('No artist column in this dataset...')
            return []

    # init the Artist Label Encoder
    def initArtistsLabelEncoder(self):
        self.ale = preprocessing.LabelEncoder()
        artists = self.getListArtists()
        self.ale.fit(
            artists)

    # print global informations of the dataset
    def dsInfos(self):
        print('___ data infos ---')
        print(self.ds.info())
        print('----------')
        print(self.ds.head(10))
        print('----------')
        print(self.ds.columns)
        print('----------')

    # print statistical properties of the dataset
    def statisticalProperties(self):
        print('----------')
        print("Statistical properties")
        print('----------')
        print(self.ds.describe())
        print('----------')
        nb_val, nb_col = self.ds.shape
        print(self.ds.shape)
        print('----------')
        return self.ds.describe()

    # check if there are nan values in the dataset
    def nanValues(self, df):
        for column in df.columns:
            nbNan = df[column].isna().sum()
            if (nbNan > 0):
                print(column)
                print(nbNan)
                # No Nan values

    # convert release_date into int
    def convertReleaseDate(self):
        self.ds['release_date'] = pd.to_datetime(self.ds['release_date']).apply(lambda x: x.value)

    # preprocess the dataset and return the data
    def preprocessDs(self, isTrainingDatas):
        # replace boolean by int
        self.ds.replace({False: 0, True: 1}, inplace=True)

        self.convertReleaseDate()

        if(isTrainingDatas):
            genres = self.ds["genre"]
            self.ds.drop(['genre'], axis=1, inplace=True)

            y_data = self.le.transform(genres)
            X_data = self.ds.to_numpy()
            return (X_data, y_data)
        X_data = self.ds.to_numpy()
        return X_data

    # return the dataset
    def getDs(self):
        return self.ds

    # print the real labels by invert transform the LabelEncoding
    def printLabes(self, y):
        labels = self.le.inverse_transform(y)
        print(labels)
        return labels

    # encode Object columns
    def encode(self):
        label_encoder = preprocessing.LabelEncoder()
        track_name_array = self.ds['track_name'].to_numpy()
        label_encoder.fit(track_name_array)
        label_encoder.classes_
        self.ds["track_name"] = label_encoder.transform(track_name_array)

        genre_array = self.ds['genres'].to_numpy()
        label_encoder.fit(genre_array)
        label_encoder.classes_
        self.ds["genres"] = label_encoder.transform(genre_array)

    # plot correlations in a graph with colors
    def correlations(self):
        corr = self.ds.corr()
        plt.figure()
        sns.heatmap(corr, annot=True)
        plt.show()

    # preprocess the subset for exercise 2
    def preprocessSubset(self):
        self.initArtistsLabelEncoder()
        encoded_artists = self.ale.transform(self.ds['artist_name'])

        self.ds['artist_name'] = encoded_artists
        self.encode()
        self.convertReleaseDate()

        self.ds.drop(['id'], axis=1, inplace=True)
        self.ds.replace({False: 0, True: 1}, inplace=True)

        return self.ds




