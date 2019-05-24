# -*- Encoding:UTF-8 -*-

import pandas as pd
import numpy as np
import sys
import pickle
# sys.path.insert(0, '/home/lxu/Documents/Probabilistic-Matrix-Factorization')
# from  PreprocessAssistment import PreprocessAssistmentSkillBuilder, PreprocessAssistmentProblemSkill

class Data:
    def __init__(self, name='ml-1m'):
        self.dataName = name
        self.dataPath = "./data/" + self.dataName + "/"
        # Static Profile
        self.UserInfo = self.getUserInfo()
        self.MovieInfo = self.getMovieInfo()

        self.data = self.getData()


    def getUserInfo(self):
        if self.dataName == "ml-1m":
            userInfoPath = self.dataPath + "users.dat"

            users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
            users = pd.read_table(userInfoPath, sep='::', header=None, names=users_title, engine='python')
            users = users.filter(regex='UserID|Gender|Age|JobID')
            users_orig = users.values

            # 将性别映射到0,1
            gender_map = {'F': 0, 'M': 1}
            users['Gender'] = users['Gender'].map(gender_map)
            # 将年龄组映射到0-6
            age_map = {val: idx for idx, val in enumerate(set(users['Age']))}
            users['Age'] = users['Age'].map(age_map)

            return users

    def getMovieInfo(self):
        if self.dataName == "ml-1m":
            movieInfoPath = self.dataPath + "movies.dat"

            movies_title = ['MovieID', 'Title', 'Genres']
            movies = pd.read_table(movieInfoPath, sep='::', header=None, names=movies_title, engine='python')
            movies = movies.filter(regex='MovieID|Genres')

            #电影类型映射到0-18
            genres_set = set()
            for val in movies['Genres'].str.split('|'):
                genres_set.update(val)
            genres2int = {val: idx for idx, val in enumerate(genres_set)}
            genres_map = {val: [genres2int[row] for row in val.split('|')] for ii, val in enumerate(set(movies['Genres']))}
            movies['Genres'] = movies['Genres'].map(genres_map)

            return movies

    def getData(self):
        if self.dataName == "ml-1m":
            dataPath = self.dataPath + "ratings.dat"

            ratings_title = ['UserID', 'MovieID', 'Rating', 'TimeStamp']
            ratings = pd.read_table(dataPath, sep='::', header=None, names=ratings_title, engine='python')
            rating = ratings.Rating
            rating[rating < 2.5] = 0
            rating[rating > 2.5] = 1
            # rating = rating.astype('bool')
            ratings.Rating = rating
            data = pd.merge(pd.merge(ratings, self.UserInfo), self.MovieInfo)
            data = data.sort_values(by=['TimeStamp'])

            return data

class AssistmentData:
    def __init__(self, name = 'Assistment15', dataPath = '../', item = 'skill'):
        self.dataName = name
        self.dataPath = dataPath
        self.item =  item
        self.data = self.getDataCSV()
        self.user_num = len(np.unique(self.data.user_id))
        if self.dataName == 'Assistment15':
            self.item_num = len(np.unique(self.data.skill_id))
        elif self.dataName == 'Assistment09':
            if self.item == 'skill':
                self.item_num = len(np.unique(self.data.skill_id))
            elif self.item == 'problem':
                self.item_num = len(np.unique(self.data.problem_id))

    def getData(self):
        if self.dataName == 'Assistment15':
            dataPath = self.dataPath + 'Assistment15-skill.pickle'
            pickle_in = open(dataPath, 'rb')
            data = pickle.load(pickle_in)
            return data[['user_id','skill_id', 'correct']]
        if self.dataName == 'Assistment09':
            if self.item=='skill':
                dataPath = self.dataPath + 'Assistment09-skill.pickle'
            elif self.item=='problem':
                dataPath = self.dataPath + 'Assistment09-problem.pickle'

            pickle_in = open(dataPath, 'rb')
            data = pickle.load(pickle_in)
            # data.drop(columns=['skill_ids'], inplace=True)
            # data.rename(index=str, columns={"problem_id": "skill_id"})
            if self.item=='skill':
                return data[['user_id','skill_id', 'correct']]
            elif self.item=='problem':
                return data[['user_id','problem_id', 'correct']]


    def getDataCSV(self):
        dataPath = self.dataPath + self.dataName + '-' + self.item + '.csv'
        data = pd.read_csv(dataPath)
        return data[['user_id', self.item +'_id', 'correct']]
        print('Load data succefully!')

if __name__ == '__main__':
    data = Data()
    print(data.MovieInfo)


