import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix


class MatrixRecommender:
    def __init__(self, data, songs, k_features):
        self.data = data
        self.k_features = k_features
        self.songs = songs
        self.model = self._computeSVD(data, k_features)
        self.all_ratings = self._get_all_ratings()
        
    def _computeSVD(data, k_features):
        u, s, vt = svds(self.data, self.k_features)    
        sigma = np.diag(s)
        return np.dot(np.dot(u, sigma), vt)
    
    def _get_all_ratings():
        return pd.DataFrame(self.model, columns = self.songs)
    
    def recommend_movies(item, num_recommendations):
        recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])]).merge(
            pd.DataFrame(sorted_user_predictions).reset_index(), how ='left',left_on = 'movieId', right_on = 'movieId').rename(
            columns = {user_row_number:'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :-1]
        
        return recommendations