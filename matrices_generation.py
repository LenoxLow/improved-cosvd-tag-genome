import pandas as pd
import numpy as np
import math

def ref_key(userid, movieid):
    return (str(int(userid)) + " " + str(int(movieid))) 

def generateTagsOrigin(rate, tags):
    temp_df = rate
    temp_df = temp_df.iloc[:,0:3]

    uni_tags = tags.tag.unique()
    uni_tags = {'tag':uni_tags, 'tid':range(len(uni_tags))}
    u_tag = pd.DataFrame(uni_tags)
    tags = pd.merge(tags, u_tag, how='left', on=['tag'])

    w = tags.groupby(['tid', 'movieId'], as_index=False)['userId'].count()
    w.columns = ['tid', 'movieId', 'cn']
    temp = w.groupby('movieId', as_index=False)['cn'].sum()
    temp = temp.set_index('movieId')
    iteration = range(len(w))
    w['val'] = np.array(pd.Series(iteration).map(lambda x: w.cn[x] / temp.loc[w.movieId[1], 'cn']))
    
    f = tags.groupby(['userId', 'tid'], as_index=False).agg({'movieId': 'count'})
    f.columns = ['userId', 'tid', 'cn']
    temp = f.groupby('userId', as_index=False)['cn'].sum()
    temp = temp.set_index('userId')
    iteration = range(len(f))
    f['val'] = np.array(pd.Series(iteration).map(lambda x: f.cn[x] / temp.loc[f.userId[x], 'cn']))
    
    nl_alpha = -0.006
    
    nl_ut = tags.groupby(['userId', 'tid'], as_index=False).agg({'timestamp': 'max'})
    nl_ut = nl_ut.sort_values(by=['userId', 'timestamp'], ascending=[True,False]).reset_index(drop=True)
    g = nl_ut.groupby(['userId'])
    nl_ut['times'] = g['timestamp'].rank(method='first', ascending=False) - 1
    nl_ut['val'] = nl_alpha * nl_ut['times']
    nl_ut['val'] = nl_ut['val'].map(lambda x: math.exp(x)).tolist()
    
    nl_it = tags.groupby(['movieId', 'tid'], as_index=False).agg({'timestamp': 'max'})
    nl_it = nl_it.sort_values(by=['movieId', 'timestamp'], ascending=[True,False]).reset_index(drop=True)
    g = nl_it.groupby(['movieId'])
    nl_it['times'] = g['timestamp'].rank(method='first', ascending=False) - 1
    nl_it['val'] = nl_alpha * nl_it['times']
    nl_it['val'] = nl_it['val'].map(lambda x: math.exp(x)).tolist()
    
    ru = temp_df.groupby(['userId'], as_index=False).agg({'rating': 'mean'})
    ru = ru.rename(index=str, columns={"rating": "ru"})
    temp = f[['userId']].drop_duplicates()
    ru = pd.merge(temp, ru, how="left", on=['userId'])
    ru.fillna(0, inplace=True)
    
    p_ut = pd.DataFrame(f[['userId','tid']], columns=['userId','tid'])
    
    how='outer'
    overall = pd.merge(temp_df, tags, how=how, on=['userId', 'movieId'])
    overall = overall.drop(columns=['tag','timestamp'])
    overall.rating.fillna(0, inplace=True)
    temp_rt = overall[ (-pd.isna(overall.userId)) & (-pd.isna(overall.tid))]
    rt = overall.groupby(['userId', 'tid'], as_index=False).agg({'rating': 'mean'})
    rt = rt.rename(index=str, columns={"rating": "rt"})
    overall = pd.merge(overall, w, how=how, on=['movieId', 'tid'])
    overall = overall.drop(columns=['cn'])
    overall = overall.rename(index=str, columns={"val": "w_it"})
    overall.w_it.fillna(0, inplace=True)
    overall = pd.merge(overall, ru, how=how, on=['userId'])
    overall.ru.fillna(0, inplace=True)
    overall['r_bias'] = overall.rating - overall.ru
    overall['b_it'] = overall.r_bias * overall.w_it

    b_it = overall[~pd.isna(overall.tid)].groupby(['userId', 'tid'], as_index=False).agg({'w_it': 'sum', 'b_it':'sum'})
    b_it['val'] = b_it.b_it / b_it.w_it
    
    ru = ru.set_index('userId')
    rt = rt.set_index(['userId', 'tid'])
    f = f.set_index(['userId', 'tid'])
    b_it = b_it.set_index(['userId', 'tid'])
    nl_ut = nl_ut.set_index(['userId', 'tid'])

    p_ut = p_ut[p_ut[['userId', 'tid']].apply(tuple, axis=1).isin(overall[['userId', 'tid']].apply(tuple, axis=1))]
    
    p_ut['val'] = list(map(lambda x,y: ru.loc[x, 'ru'] + b_it.loc[(x, y), 'val'] 
              + 1.7 * f.loc[(x,y),'val'] * (rt.loc[(x,y), 'rt'] - ru.loc[x, 'ru']) 
              + 0.05 * nl_ut.loc[(x,y), 'val'] , p_ut.userId, p_ut.tid))
    
    f_it = pd.DataFrame(w[['movieId','tid']], columns=['movieId','tid'])
    w = w.set_index(['movieId', 'tid'])
    nl_it = nl_it.set_index(['movieId', 'tid'])
    f_it['val'] = list(map(lambda x,y: w.loc[(x,y), 'val'] + 0.05 * nl_it.loc[(x,y), 'val'], f_it.movieId, f_it.tid))
    
    ratings = overall.iloc[:,0:4]
    ratings['ref_key'] = ratings.apply(lambda x: ref_key(x['userId'], x['movieId']), axis=1)
    return p_ut, f_it, tags, ratings