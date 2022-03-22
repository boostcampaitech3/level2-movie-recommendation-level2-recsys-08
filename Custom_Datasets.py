import pandas as pd
import os


def context_data(data_path='~/input/data/train'):
    year_data = pd.read_csv(os.path.join(data_path, 'years.tsv'), sep='\t')
    writer_data = pd.read_csv(os.path.join(data_path, 'writers.tsv'), sep='\t')
    title_data = pd.read_csv(os.path.join(data_path, 'titles.tsv'), sep='\t')
    genre_data = pd.read_csv(os.path.join(data_path, 'genres.tsv'), sep='\t')
    director_data = pd.read_csv(os.path.join(data_path, 'directors.tsv'), sep='\t')    
    
    return year_data, writer_data, title_data, genre_data, director_data


def load_data(data_path='~/input/data/train'):        
    train_df = pd.read_csv(os.path.join(data_path, 'train_ratings.csv')) # 전체 학습 데이터
    n_u = train_df['user'].size
    n_m = train_df['item'].size
    n_t = train_df['time'].size
    
    return n_u, n_m, n_t
    
