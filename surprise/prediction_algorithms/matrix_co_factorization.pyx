"""
the :mod:`matrix_factorization` module includes some algorithms using matrix
factorization.
"""
# command for .pyd file generation
# python setup.py build_ext --inplace

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

cimport numpy as np  # noqa
import numpy as np
import pandas as pd
from six.moves import range

from .algo_base import AlgoBase
from .predictions import PredictionImpossible
from ..utils import get_rng

import os
import math
import time
import sys
import array
from functools import reduce

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
    temp_rt = overall[ (~pd.isna(overall.userId)) & (~pd.isna(overall.tid))]
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

    p_ut['val'] = list(map(lambda x,y: ru.loc[x, 'ru'] + b_it.loc[(x, y), 'val']
              + 1.7 * f.loc[(x,y),'val'] * (rt.loc[(x,y), 'rt'] - ru.loc[x, 'ru'])
              + 0.05 * nl_ut.loc[(x,y), 'val'] , p_ut.userId, p_ut.tid))

    f_it = pd.DataFrame(w[['movieId','tid']], columns=['movieId','tid'])
    w = w.set_index(['movieId', 'tid'])
    nl_it = nl_it.set_index(['movieId', 'tid'])
    f_it['val'] = list(map(lambda x,y: w.loc[(x,y), 'val'] + 0.05 * nl_it.loc[(x,y), 'val'], f_it.movieId, f_it.tid))

    ratings = overall.iloc[:,0:4]
    return p_ut, f_it, tags, ratings

def generateTagsWithTagGenome(rate, tags, genome_tag, genome_score):
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
    temp_rt = overall[ (~pd.isna(overall.userId)) & (~pd.isna(overall.tid))]
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

    p_ut['val'] = list(map(lambda x,y: ru.loc[x, 'ru'] + b_it.loc[(x, y), 'val']
              + 1.7 * f.loc[(x,y),'val'] * (rt.loc[(x,y), 'rt'] - ru.loc[x, 'ru'])
              + 0.05 * nl_ut.loc[(x,y), 'val'] , p_ut.userId, p_ut.tid))

    genome_tag = genome_tag[genome_tag.tag.isin(tags.tag)]
    genome_tag = pd.merge(genome_tag, u_tag, how='left', on=['tag'])
    genome_score = pd.merge(genome_score, genome_tag[['tagId', 'tid']], how='left', on=['tagId'])

    f_it = genome_score[genome_score.movieId.isin(tags.movieId) & genome_score.tagId.isin(genome_tag.tagId)]
    f_it = f_it.drop(['tagId'], axis=1)
    f_it = f_it.rename(index=str, columns={"relevance": "val"})

    ratings = overall.iloc[:,0:4]
    return p_ut, f_it, tags, ratings

def generateTagsWithTagGenomeHybrid(rate, tags, genome_tag, genome_score):
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
    temp_rt = overall[ (~pd.isna(overall.userId)) & (~pd.isna(overall.tid))]
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

    p_ut['val'] = list(map(lambda x,y: ru.loc[x, 'ru'] + b_it.loc[(x, y), 'val']
              + 1.7 * f.loc[(x,y),'val'] * (rt.loc[(x,y), 'rt'] - ru.loc[x, 'ru'])
              + 0.05 * nl_ut.loc[(x,y), 'val'] , p_ut.userId, p_ut.tid))

    genome_tag = genome_tag[genome_tag.tag.isin(tags.tag)]
    genome_tag = pd.merge(genome_tag, u_tag, how='left', on=['tag'])
    genome_score = pd.merge(genome_score, genome_tag[['tagId', 'tid']], how='left', on=['tagId'])
    genome_score = genome_score.drop(['tagId'], axis=1)
    genome_score = genome_score.rename(index=str, columns={"relevance": "val"})

    f_it = pd.DataFrame(w[['movieId','tid']], columns=['movieId','tid'])
    f_it = f_it.append(genome_score[['movieId', 'tid']], ignore_index=True)
    f_it = f_it.drop_duplicates()
    f_it.reset_index(drop=True, inplace=True)

    w = w.set_index(['movieId', 'tid'])
    nl_it = nl_it.set_index(['movieId', 'tid'])
    genome_score = genome_score.set_index(['movieId', 'tid'])

    f_it['val'] = list(map(lambda x,y: genome_score.loc[(x,y), 'val'] + 0.05 * nl_it.loc[(x,y), 'val'] if (((x,y) in genome_score.index) & ((x,y) in nl_it.index)) else genome_score.loc[(x,y), 'val'] if (x,y) in genome_score.index else w.loc[(x,y), 'val'] + 0.05 * nl_it.loc[(x,y), 'val'], f_it.movieId, f_it.tid))
    ratings = overall.iloc[:,0:4]
    return p_ut, f_it, tags, ratings

# class CoSVDv5(AlgoBase):
#     def __init__ (
#         self
#         , n_factors = 40
#         , n_epochs = 1
#         , biased = True
#         , init_mean = 0
#         , init_std_dev=.1
#         , lr_all=.005
#         , reg_all=.02
#         , lr_bu=None
#         , lr_bi=None
#         , lr_bt=None
#         , lr_pu=None
#         , lr_qi=None
#         , lr_rt=None
#         , reg_p=.001
#         , reg_r=.035
#         , reg_f=1.5
#         , random_state=None
#         , verbose=False
#         , p_ut=None
#         , f_it=None
#         , tags=None
#         , ratings=None
#         ):
#
#         self.n_factors = n_factors
#         self.n_epochs = n_epochs
#         self.biased = biased
#         self.init_mean = init_mean
#         self.init_std_dev = init_std_dev
#         self.lr_bu = lr_bu if lr_bu is not None else lr_all
#         self.lr_bi = lr_bi if lr_bi is not None else lr_all
#         self.lr_bt = lr_bt if lr_bt is not None else lr_all
#
#         self.lr_pu = lr_pu if lr_pu is not None else lr_all
#         self.lr_qi = lr_qi if lr_qi is not None else lr_all
#         self.lr_rt = lr_rt if lr_rt is not None else lr_all
#
#         self.reg_p = reg_p
#         self.reg_f = reg_f
#         self.reg_r = reg_r
#
#         self.random_state = random_state
#         self.verbose = verbose
#
#         self.p_ut = p_ut
#         self.f_it = f_it
#         self.tags = tags
#         self.ratings = ratings
#         AlgoBase.__init__(self)
#
#     def fit(self,trainset):
#         AlgoBase.fit(self, trainset)
#         self.sgd(trainset)
#         return self
#
#
#     def sgd(self, trainset):
#         cdef np.ndarray[np.double_t] bu
#         cdef np.ndarray[np.double_t] bi
#         cdef np.ndarray[np.double_t] bt
#
#         cdef np.ndarray[np.double_t, ndim=2] pu
#         cdef np.ndarray[np.double_t, ndim=2] qi
#         cdef np.ndarray[np.double_t, ndim=2] rt
#
#         cdef int u, i, t, f, raw_u, raw_i, raw_t
#         cdef double r, p_put, p_fit, err_r, err_p, err_f, dot_r, dot_p, dot_f, puf, qif, rtf, global_mean_p, global_mean_f
#         cdef double global_mean_r = self.trainset.global_mean
#
#         cdef double lr_bu = self.lr_bu
#         cdef double lr_bi = self.lr_bi
#         cdef double lr_bt = self.lr_bt
#
#         cdef double lr_pu = self.lr_pu
#         cdef double lr_qi = self.lr_qi
#         cdef double lr_rt = self.lr_rt
#
#         cdef double reg_p = self.reg_p
#         cdef double reg_f = self.reg_f
#         cdef double reg_r = self.reg_r
#
#         p_ut = self.p_ut
#         f_it = self.f_it
#         tags = self.tags
#         ratings = self.ratings
#
#         cdef int n_factors = self.n_factors
#         raw_user = np.zeros(trainset.n_users, int)
#         raw_item = np.zeros(trainset.n_items, int)
#
#
#         data_ref = []
#         for u, i, r in trainset.all_ratings():
#             data_ref.append(str(int(trainset.to_raw_uid(u))) + " " + str(int(trainset.to_raw_iid(i))))
#
#         raw_user = np.zeros(trainset.n_users, int)
#         for i in trainset.all_users():
#             raw_user[i] = trainset.to_raw_uid(i)
#
#         raw_item = np.zeros(trainset.n_items, int)
#         for i in trainset.all_items():
#             raw_item[i] = trainset.to_raw_iid(i)
#
#         raw_data = ratings[ratings.ref_key.isin(data_ref)]
#
#         uni_tid = raw_data.tid.unique()
#         uni_tid = uni_tid[~np.isnan(uni_tid)]
#         u_t = pd.DataFrame({'tid': uni_tid,
#                             'tid_inner':range(len(uni_tid))})
#
#         raw_data = pd.merge(raw_data, u_t, how ='left', on=['tid'])
#
#         final_p = p_ut[p_ut.userId.isin(raw_data.userId) & p_ut.tid.isin(raw_data.tid)]
#         final_f = f_it[f_it.movieId.isin(raw_data.movieId) & f_it.tid.isin(raw_data.tid)]
#
#         final_p = pd.merge(final_p, u_t, how='left', on=['tid'])
#         final_f = pd.merge(final_f, u_t, how='left', on=['tid'])
#
#         p_ut = final_p.drop(['tid'], axis=1)
#         f_it = final_f.drop(['tid'], axis=1)
#
#         rng = get_rng(self.random_state)
#
#         bu = np.zeros(trainset.n_users, np.double)
#         bi = np.zeros(trainset.n_items, np.double)
#
#         bt = np.zeros(len(p_ut.tid_inner.unique()), np.double)
#
#         pu = rng.normal(self.init_mean, self.init_std_dev,
#                         (trainset.n_users, n_factors))
#         qi = rng.normal(self.init_mean, self.init_std_dev,
#                         (trainset.n_items, n_factors))
#         rt = rng.normal(self.init_mean, self.init_std_dev,
#                         (len(p_ut.tid_inner.unique()), n_factors))
#
#         global_mean_p = np.mean(p_ut.val)
#         global_mean_f = np.mean(f_it.val)
#
#         for current_epoch in range(self.n_epochs):
#             if self.verbose:
#                 print("Processing epoch {}".format(current_epoch), end='\r')
#
#             for rn in range(len(raw_data)):
#                 r = raw_data.loc[rn, 'rating']
#                 raw_u = raw_data.loc[rn, 'userId']
#                 raw_i = raw_data.loc[rn, 'movieId']
#                 known_item = True
#                 known_user = True
#                 try:
#                     temp = trainset.to_inner_iid(raw_i)
#                 except:
#                     known_item = False
#
#                 try:
#                     temp = trainset.to_inner_uid(raw_u)
#                 except:
#                     known_user = False
#
#                 if (r != 0) | (~np.isnan(r)):
#                     u = trainset.to_inner_uid(raw_u)
#                     i = trainset.to_inner_iid(raw_i)
#
#                     if pd.isna(raw_data.loc[rn, 'tid_inner']) | np.isnan(raw_data.loc[rn, 'tid_inner']):
#                         t = -1
#                     else:
#                         t = raw_data.loc[rn, 'tid_inner']
#
#                     t = int(t)
#
#                     if t != -1:
#                         p_put = p_ut.val[(p_ut.userId == raw_u) & (p_ut.tid_inner == t)].values[0]
#                         p_fit = f_it.val[(f_it.movieId == raw_i) & (f_it.tid_inner == t)].values[0]
#
#                     dot_r = 0.0
#                     dot_p = 0.0
#                     dot_f = 0.0
#
#                     dot_r = np.dot(qi[i], pu[u])
#
#                     if t != -1:
#                         dot_p = np.dot(rt[t], pu[u])
#                         dot_f = np.dot(rt[t], qi[i])
#
#
#                     err_r = r - (global_mean_r + bu[u] + bi[i] + dot_r)
#
#                     if t != -1:
#                         err_p = p_put - (global_mean_p + bu[u] + bt[t] + dot_p)
#                         err_f = p_fit - (global_mean_f + bi[i] + bt[t] + dot_f)
#                     else:
#                         err_p = 0.0
#                         err_f = 0.0
#
#                     if self.biased:
#                         bu[u] -= lr_bu * (-1 * err_r - reg_p * err_p + reg_r * bu[u])
#                         bi[i] -= lr_bi * (-1 * err_r - reg_f * err_f + reg_r * bi[i])
#                         if t != -1:
#                             bt[t] -= lr_bt * (-1 * reg_p * err_p - reg_f * err_f + reg_r * bt[t])
#
#                     rv_sum = err_r * qi[i]
#                     ru_sum = err_r * pu[u]
#
#                     if t != -1:
#                         pz_sum = err_p * rt[t]
#                         fz_sum = err_f * rt[t]
#                         pu_sum = err_p * pu[u]
#                         fv_sum = err_f * qi[i]
#                     else:
#                         pz_sum = 0
#                         fz_sum = 0
#                         pu_sum = 0
#                         fv_sum = 0
#
#                     pu[u] += lr_pu * (rv_sum + reg_p * pz_sum - reg_r * pu[u])
#                     qi[i] += lr_pu * (ru_sum + reg_f * fz_sum - reg_r * qi[i])
#                     if t != -1:
#                         rt[t] += lr_pu * (reg_p * pu_sum + reg_f * fv_sum - reg_r * rt[t])
#                 else:
#                     t = int(raw_data.loc[rn, 'tid_inner'])
#                     if known_item & known_user:
#                         p_put = p_ut.val[(p_ut.userId == raw_u) & (p_ut.tid_inner == t)].values[0]
#                         p_fit = f_it.val[(f_it.movieId == raw_i) & (f_it.tid_inner == t)].values[0]
#
#                         dot_p = np.dot(rt[t], pu[u])
#                         dot_f = np.dot(rt[t], qi[i])
#
#                         err_p = p_put - (global_mean_p + bu[u] + bt[t] + dot_p)
#                         err_f = p_fit - (global_mean_f + bi[i] + bt[t] + dot_f)
#
#                         if self.biased:
#                             bt[t] -= lr_bt * (-1 * reg_p * err_p - reg_f * err_f + reg_r * bt[t])
#
#                         pu_sum = err_p * pu[u]
#                         fv_sum = err_f * qi[i]
#
#                         rt -= lr_pu * (-1 * reg_p * pu_sum - reg_f * fv_sum + reg_r * rt)
#                     elif known_user:
#                         u = trainset.to_inner_uid(raw_u)
#
#                         p_put = p_ut.val[(p_ut.userId == raw_u) & (p_ut.tid_inner == t)].values[0]
#
#                         dot_p = np.dot(rt[t], pu[u])
#
#                         err_p = p_put - (global_mean_p + bu[u] + bt[t] + dot_p)
#
#                         if self.biased:
#                             bt[t] -= lr_bt * (-1 * reg_p * err_p + reg_r * bt[t])
#
#                         pu_sum = err_p * pu[u]
#
#                         rt -= lr_pu * (-1 * reg_p * pu_sum + reg_r * rt)
#                     elif known_item:
#                         i = trainset.to_inner_iid(raw_i)
#
#                         p_fit = f_it.val[(f_it.movieId == raw_i) & (f_it.tid_inner == t)].values[0]
#
#                         dot_f = np.dot(rt[t], qi[i])
#
#                         err_f = p_fit - (global_mean_f + bi[i] + bt[t] + dot_f)
#
#                         if self.biased:
#                             bt[t] -= lr_bt * (-1 * reg_f * err_f + reg_r * bt[t])
#
#                         fv_sum = err_f * qi[i]
#
#                         rt -= lr_pu * (-1 * reg_f * fv_sum + reg_r * rt)
#
#         self.bu = bu
#         self.bi = bi
#         self.pu = pu
#         self.qi = qi
#
#     def estimate(self, u, i):
#         known_user = self.trainset.knows_user(u)
#         known_item = self.trainset.knows_item(i)
#
#         if self.biased:
#             est = self.trainset.global_mean
#
#             if known_user:
#                 est += self.bu[u]
#
#             if known_item:
#                 est += self.bi[i]
#
#             if known_user and known_item:
#                 est += np.dot(self.qi[i], self.pu[u])
#         else:
#             if known_user and known_item:
#                 est = np.dot(self.qi[i], self.pu[u])
#             else:
#                 raise PredictionImpossible('User and item are unknown')
#         return est
#
# class CoSVDv6(AlgoBase):
#     def __init__ (
#         self
#         , n_factors = 40
#         , n_epochs = 1
#         , biased = True
#         , init_mean = 0
#         , init_std_dev=.1
#         , lr_all=.005
#         , reg_all=.02
#         , lr_bu=None
#         , lr_bi=None
#         , lr_bt=None
#         , lr_pu=None
#         , lr_qi=None
#         , lr_rt=None
#         , reg_p=.001
#         , reg_r=.035
#         , reg_f=1.5
#         , random_state=None
#         , verbose=False
#         , p_ut=None
#         , f_it=None
#         , tags=None
#         , ratings=None
#         ):
#
#         self.n_factors = n_factors
#         self.n_epochs = n_epochs
#         self.biased = biased
#         self.init_mean = init_mean
#         self.init_std_dev = init_std_dev
#         self.lr_bu = lr_bu if lr_bu is not None else lr_all
#         self.lr_bi = lr_bi if lr_bi is not None else lr_all
#         self.lr_bt = lr_bt if lr_bt is not None else lr_all
#
#         self.lr_pu = lr_pu if lr_pu is not None else lr_all
#         self.lr_qi = lr_qi if lr_qi is not None else lr_all
#         self.lr_rt = lr_rt if lr_rt is not None else lr_all
#
#         self.reg_p = reg_p
#         self.reg_f = reg_f
#         self.reg_r = reg_r
#
#         self.random_state = random_state
#         self.verbose = verbose
#
#         self.p_ut = p_ut
#         self.f_it = f_it
#         self.tags = tags
#         self.ratings = ratings
#         AlgoBase.__init__(self)
#
#     def fit(self,trainset):
#         AlgoBase.fit(self, trainset)
#         self.sgd(trainset)
#         return self
#
#     def sgd(self, trainset):
#         cdef np.ndarray[np.double_t] bu
#         cdef np.ndarray[np.double_t] bi
#         cdef np.ndarray[np.double_t] bt
#
#         cdef np.ndarray[np.double_t, ndim=2] pu
#         cdef np.ndarray[np.double_t, ndim=2] qi
#         cdef np.ndarray[np.double_t, ndim=2] rt
#
#         cdef int u, i, t, f, raw_u, raw_i, raw_t
#         cdef double r, p_put, p_fit, err_r, err_p, err_f, dot_r, dot_p, dot_f, puf, qif, rtf, global_mean_p, global_mean_f
#         cdef double global_mean_r = self.trainset.global_mean
#
#         cdef double lr_bu = self.lr_bu
#         cdef double lr_bi = self.lr_bi
#         cdef double lr_bt = self.lr_bt
#
#         cdef double lr_pu = self.lr_pu
#         cdef double lr_qi = self.lr_qi
#         cdef double lr_rt = self.lr_rt
#
#         cdef double reg_p = self.reg_p
#         cdef double reg_f = self.reg_f
#         cdef double reg_r = self.reg_r
#
#         p_ut = self.p_ut
#         f_it = self.f_it
#         tags = self.tags
#         ratings = self.ratings
#
#         cdef int n_factors = self.n_factors
#         raw_user = np.zeros(trainset.n_users, int)
#         raw_item = np.zeros(trainset.n_items, int)
#
#
#         data_ref = []
#         for u, i, r in trainset.all_ratings():
#             data_ref.append(str(int(trainset.to_raw_uid(u))) + " " + str(int(trainset.to_raw_iid(i))))
#
#         raw_user = np.zeros(trainset.n_users, int)
#         for i in trainset.all_users():
#             raw_user[i] = trainset.to_raw_uid(i)
#
#         raw_item = np.zeros(trainset.n_items, int)
#         for i in trainset.all_items():
#             raw_item[i] = trainset.to_raw_iid(i)
#
#         raw_data = ratings[ratings.ref_key.isin(data_ref)]
#
#         uni_tid = raw_data.tid.unique()
#         uni_tid = uni_tid[~np.isnan(uni_tid)]
#         u_t = pd.DataFrame({'tid': uni_tid,
#                             'tid_inner':range(len(uni_tid))})
#
#         raw_data = pd.merge(raw_data, u_t, how ='left', on=['tid'])
#
#         final_p = p_ut[p_ut.userId.isin(raw_data.userId) & p_ut.tid.isin(raw_data.tid)]
#         final_f = f_it[f_it.movieId.isin(raw_data.movieId) & f_it.tid.isin(raw_data.tid)]
#
#         final_p = pd.merge(final_p, u_t, how='left', on=['tid'])
#         final_f = pd.merge(final_f, u_t, how='left', on=['tid'])
#
#         p_ut = final_p.drop(['tid'], axis=1)
#         f_it = final_f.drop(['tid'], axis=1)
#
#         rng = get_rng(self.random_state)
#
#         bu = np.zeros(trainset.n_users, np.double)
#         bi = np.zeros(trainset.n_items, np.double)
#
#         bt = np.zeros(len(p_ut.tid_inner.unique()), np.double)
#
#         pu = rng.normal(self.init_mean, self.init_std_dev,
#                         (trainset.n_users, n_factors))
#         qi = rng.normal(self.init_mean, self.init_std_dev,
#                         (trainset.n_items, n_factors))
#         rt = rng.normal(self.init_mean, self.init_std_dev,
#                         (len(p_ut.tid_inner.unique()), n_factors))
#
#         global_mean_p = np.mean(p_ut.val)
#         global_mean_f = np.mean(f_it.val)
#
#         for current_epoch in range(self.n_epochs):
#             if self.verbose:
#                 print("Processing epoch {}".format(current_epoch), end='\r')
#
#             for rn in range(len(raw_data)):
#                 r = raw_data.loc[rn, 'rating']
#                 raw_u = raw_data.loc[rn, 'userId']
#                 raw_i = raw_data.loc[rn, 'movieId']
#                 known_item = True
#                 known_user = True
#                 try:
#                     temp = trainset.to_inner_iid(raw_i)
#                 except:
#                     known_item = False
#
#                 try:
#                     temp = trainset.to_inner_uid(raw_u)
#                 except:
#                     known_user = False
#
#                 if (r != 0) | (~np.isnan(r)):
#                     u = trainset.to_inner_uid(raw_u)
#                     i = trainset.to_inner_iid(raw_i)
#
#                     if pd.isna(raw_data.loc[rn, 'tid_inner']) | np.isnan(raw_data.loc[rn, 'tid_inner']):
#                         t = -1
#                     else:
#                         t = raw_data.loc[rn, 'tid_inner']
#
#                     t = int(t)
#
#                     if t != -1:
#                         p_put = p_ut.val[(p_ut.userId == raw_u) & (p_ut.tid_inner == t)].values[0]
#                         p_fit = f_it.val[(f_it.movieId == raw_i) & (f_it.tid_inner == t)].values[0]
#
#                     dot_r = 0.0
#                     dot_p = 0.0
#                     dot_f = 0.0
#
#                     dot_r = np.dot(qi[i], pu[u])
#
#                     if t != -1:
#                         dot_p = np.dot(rt[t], pu[u])
#                         dot_f = np.dot(rt[t], qi[i])
#
#
#                     err_r = r - (global_mean_r + bu[u] + bi[i] + dot_r)
#
#                     if t != -1:
#                         err_p = p_put - (global_mean_p + bu[u] + bt[t] + dot_p)
#                         err_f = p_fit - (global_mean_f + bi[i] + bt[t] + dot_f)
#                     else:
#                         err_p = 0.0
#                         err_f = 0.0
#
#                     if self.biased:
#                         bu[u] -= lr_bu * (-1 * err_r - reg_p * err_p + reg_r * bu[u])
#                         bi[i] -= lr_bi * (-1 * err_r - reg_f * err_f + reg_r * bi[i])
#                         if t != -1:
#                             bt[t] -= lr_bt * (-1 * reg_p * err_p - reg_f * err_f + reg_r * bt[t])
#
#                     rv_sum = err_r * qi[i]
#                     ru_sum = err_r * pu[u]
#
#                     if t != -1:
#                         pz_sum = err_p * rt[t]
#                         fz_sum = err_f * rt[t]
#                         pu_sum = err_p * pu[u]
#                         fv_sum = err_f * qi[i]
#                     else:
#                         pz_sum = 0
#                         fz_sum = 0
#                         pu_sum = 0
#                         fv_sum = 0
#
#                     pu[u] += lr_pu * (rv_sum + reg_p * pz_sum - reg_r * pu[u])
#                     qi[i] += lr_pu * (ru_sum + reg_f * fz_sum - reg_r * qi[i])
#                     if t != -1:
#                         rt[t] += lr_pu * (reg_p * pu_sum + reg_f * fv_sum - reg_r * rt[t])
#                 else:
#                     print("test")
#                     t = int(raw_data.loc[rn, 'tid_inner'])
#                     if known_item & known_user:
#                         p_put = p_ut.val[(p_ut.userId == raw_u) & (p_ut.tid_inner == t)].values[0]
#                         p_fit = f_it.val[(f_it.movieId == raw_i) & (f_it.tid_inner == t)].values[0]
#
#                         dot_p = np.dot(rt[t], pu[u])
#                         dot_f = np.dot(rt[t], qi[i])
#
#                         err_p = p_put - (global_mean_p + bu[u] + bt[t] + dot_p)
#                         err_f = p_fit - (global_mean_f + bi[i] + bt[t] + dot_f)
#
#                         if self.biased:
#                             bt[t] -= lr_bt * (-1 * reg_p * err_p - reg_f * err_f + reg_r * bt[t])
#
#                         pu_sum = err_p * pu[u]
#                         fv_sum = err_f * qi[i]
#
#                         rt -= lr_pu * (-1 * reg_p * pu_sum - reg_f * fv_sum + reg_r * rt)
#
#         self.bu = bu
#         self.bi = bi
#         self.pu = pu
#         self.qi = qi
#
#     def estimate(self, u, i):
#         known_user = self.trainset.knows_user(u)
#         known_item = self.trainset.knows_item(i)
#
#         if self.biased:
#             est = self.trainset.global_mean
#
#             if known_user:
#                 est += self.bu[u]
#
#             if known_item:
#                 est += self.bi[i]
#
#             if known_user and known_item:
#                 est += np.dot(self.qi[i], self.pu[u])
#         else:
#             if known_user and known_item:
#                 est = np.dot(self.qi[i], self.pu[u])
#             else:
#                 raise PredictionImpossible('User and item are unknown')
#         return est
#
# class CoSVDv7(AlgoBase):
#     def __init__ (
#         self
#         , n_factors = 40
#         , n_epochs = 1
#         , biased = True
#         , init_mean = 0
#         , init_std_dev=.1
#         , lr_all=.005
#         , reg_all=.02
#         , lr_bu=None
#         , lr_bi=None
#         , lr_bt=None
#         , lr_pu=None
#         , lr_qi=None
#         , lr_rt=None
#         , reg_p=.001
#         , reg_r=.035
#         , reg_f=1.5
#         , random_state=None
#         , verbose=False
#         , p_ut=None
#         , f_it=None
#         , tags=None
#         , ratings=None
#         ):
#
#         self.n_factors = n_factors
#         self.n_epochs = n_epochs
#         self.biased = biased
#         self.init_mean = init_mean
#         self.init_std_dev = init_std_dev
#         self.lr_bu = lr_bu if lr_bu is not None else lr_all
#         self.lr_bi = lr_bi if lr_bi is not None else lr_all
#         self.lr_bt = lr_bt if lr_bt is not None else lr_all
#
#         self.lr_pu = lr_pu if lr_pu is not None else lr_all
#         self.lr_qi = lr_qi if lr_qi is not None else lr_all
#         self.lr_rt = lr_rt if lr_rt is not None else lr_all
#
#         self.reg_p = reg_p
#         self.reg_f = reg_f
#         self.reg_r = reg_r
#
#         self.random_state = random_state
#         self.verbose = verbose
#
#         self.p_ut = p_ut
#         self.f_it = f_it
#         self.tags = tags
#         self.ratings = ratings
#         AlgoBase.__init__(self)
#
#     def fit(self,trainset):
#         AlgoBase.fit(self, trainset)
#         self.sgd(trainset)
#         return self
#
#     def sgd(self, trainset):
#
#         cdef np.ndarray[np.double_t] bu
#         cdef np.ndarray[np.double_t] bi
#         cdef np.ndarray[np.double_t] bt
#
#         cdef np.ndarray[np.double_t, ndim=2] pu
#         cdef np.ndarray[np.double_t, ndim=2] qi
#         cdef np.ndarray[np.double_t, ndim=2] rt
#
#         cdef int u, i, t, f, raw_u, raw_i, raw_t
#         cdef double r, p_put, p_fit, err_r, err_p, err_f, dot_r, dot_p, dot_f, puf, qif, rtf, global_mean_p, global_mean_f
#         cdef double global_mean_r = self.trainset.global_mean
#
#         cdef double lr_bu = self.lr_bu
#         cdef double lr_bi = self.lr_bi
#         cdef double lr_bt = self.lr_bt
#
#         cdef double lr_pu = self.lr_pu
#         cdef double lr_qi = self.lr_qi
#         cdef double lr_rt = self.lr_rt
#
#         cdef double reg_p = self.reg_p
#         cdef double reg_f = self.reg_f
#         cdef double reg_r = self.reg_r
#
#         p_ut = self.p_ut
#         f_it = self.f_it
#         tags = self.tags
#         ratings = self.ratings
#
#         cdef int n_factors = self.n_factors
#         raw_user = np.zeros(trainset.n_users, int)
#         raw_item = np.zeros(trainset.n_items, int)
#
#
#         data_ref = []
#         for u, i, r in trainset.all_ratings():
#             data_ref.append(str(int(trainset.to_raw_uid(u))) + " " + str(int(trainset.to_raw_iid(i))))
#
#         raw_user = np.zeros(trainset.n_users, int)
#         for i in trainset.all_users():
#             raw_user[i] = trainset.to_raw_uid(i)
#
#         raw_item = np.zeros(trainset.n_items, int)
#         for i in trainset.all_items():
#             raw_item[i] = trainset.to_raw_iid(i)
#
#         raw_data = ratings[ratings.ref_key.isin(data_ref)]
#
#         uni_tid = raw_data.tid.unique()
#         uni_tid = uni_tid[~np.isnan(uni_tid)]
#         u_t = pd.DataFrame({'tid': uni_tid,
#                             'tid_inner':range(len(uni_tid))})
#
#         raw_data = pd.merge(raw_data, u_t, how ='left', on=['tid'])
#
#         final_p = p_ut[p_ut.userId.isin(raw_data.userId) & p_ut.tid.isin(raw_data.tid)]
#         final_f = f_it[f_it.movieId.isin(raw_data.movieId) & f_it.tid.isin(raw_data.tid)]
#
#         final_p = pd.merge(final_p, u_t, how='left', on=['tid'])
#         final_f = pd.merge(final_f, u_t, how='left', on=['tid'])
#
#         p_ut = final_p.drop(['tid'], axis=1)
#         f_it = final_f.drop(['tid'], axis=1)
#
#         rng = get_rng(self.random_state)
#
#         bu = np.zeros(trainset.n_users, np.double)
#         bi = np.zeros(trainset.n_items, np.double)
#
#         bt = np.zeros(len(p_ut.tid_inner.unique()), np.double)
#
#         pu = rng.normal(self.init_mean, self.init_std_dev,
#                         (trainset.n_users, n_factors))
#         qi = rng.normal(self.init_mean, self.init_std_dev,
#                         (trainset.n_items, n_factors))
#         rt = rng.normal(self.init_mean, self.init_std_dev,
#                         (len(p_ut.tid_inner.unique()), n_factors))
#
#         global_mean_p = np.mean(p_ut.val)
#         global_mean_f = np.mean(f_it.val)
#
#         for current_epoch in range(self.n_epochs):
#             if self.verbose:
#                 print("Processing epoch {}".format(current_epoch), end='\r')
#
#             for rn in range(len(raw_data)):
#                 r = raw_data.loc[rn, 'rating']
#                 raw_u = raw_data.loc[rn, 'userId']
#                 raw_i = raw_data.loc[rn, 'movieId']
#                 known_item = True
#                 known_user = True
#                 try:
#                     temp = trainset.to_inner_iid(raw_i)
#                 except:
#                     known_item = False
#
#                 try:
#                     temp = trainset.to_inner_uid(raw_u)
#                 except:
#                     known_user = False
#
#                 if (r != 0) | (~np.isnan(r)):
#                     u = trainset.to_inner_uid(raw_u)
#                     i = trainset.to_inner_iid(raw_i)
#
#                     if pd.isna(raw_data.loc[rn, 'tid_inner']) | np.isnan(raw_data.loc[rn, 'tid_inner']):
#                         t = -1
#                     else:
#                         t = raw_data.loc[rn, 'tid_inner']
#
#                     t = int(t)
#
#                     if t != -1:
#                         p_put = p_ut.val[(p_ut.userId == raw_u) & (p_ut.tid_inner == t)].values[0]
#                         p_fit = f_it.val[(f_it.movieId == raw_i) & (f_it.tid_inner == t)].values[0]
#
#                     dot_r = 0.0
#                     dot_p = 0.0
#                     dot_f = 0.0
#
#                     dot_r = np.dot(qi[i], pu[u])
#
#                     if t != -1:
#                         dot_p = np.dot(rt[t], pu[u])
#                         dot_f = np.dot(rt[t], qi[i])
#
#                     err_r = r - (global_mean_r + bu[u] + bi[i] + dot_r)
#
#                     if t != -1:
#                         err_p = p_put - (global_mean_p + bu[u] + bt[t] + dot_p)
#                         err_f = p_fit - (global_mean_f + bi[i] + bt[t] + dot_f)
#                     else:
#                         err_p = 0.0
#                         err_f = 0.0
#
#                     if self.biased:
#                         bu[u] -= lr_bu * (-1 * err_r - reg_p * err_p + reg_r * bu[u])
#                         bi[i] -= lr_bi * (-1 * err_r - reg_f * err_f + reg_r * bi[i])
#                         if t != -1:
#                             bt[t] -= lr_bt * (-1 * reg_p * err_p - reg_f * err_f + reg_r * bt[t])
#
#                     rv_sum = err_r * qi[i]
#                     ru_sum = err_r * pu[u]
#
#                     if t != -1:
#                         pz_sum = err_p * rt[t]
#                         fz_sum = err_f * rt[t]
#                         pu_sum = err_p * pu[u]
#                         fv_sum = err_f * qi[i]
#                     else:
#                         pz_sum = 0
#                         fz_sum = 0
#                         pu_sum = 0
#                         fv_sum = 0
#
#                     pu[u] += lr_pu * (rv_sum + reg_p * pz_sum - reg_r * pu[u])
#                     qi[i] += lr_pu * (ru_sum + reg_f * fz_sum - reg_r * qi[i])
#                     if t != -1:
#                         rt[t] += lr_pu * (reg_p * pu_sum + reg_f * fv_sum - reg_r * rt[t])
#
#         self.bu = bu
#         self.bi = bi
#         self.pu = pu
#         self.qi = qi
#
#     def estimate(self, u, i):
#         known_user = self.trainset.knows_user(u)
#         known_item = self.trainset.knows_item(i)
#
#         if self.biased:
#             est = self.trainset.global_mean
#
#             if known_user:
#                 est += self.bu[u]
#
#             if known_item:
#                 est += self.bi[i]
#
#             if known_user and known_item:
#                 est += np.dot(self.qi[i], self.pu[u])
#         else:
#             if known_user and known_item:
#                 est = np.dot(self.qi[i], self.pu[u])
#             else:
#                 raise PredictionImpossible('User and item are unknown')
#         return est
#
# class CoSVDv8(AlgoBase):
#     def __init__ (
#         self
#         , n_factors = 40
#         , n_epochs = 1
#         , biased = True
#         , init_mean = 0
#         , init_std_dev=.1
#         , lr_all=.005
#         , reg_all=.02
#         , lr_bu=None
#         , lr_bi=None
#         , lr_bt=None
#         , lr_pu=None
#         , lr_qi=None
#         , lr_rt=None
#         , reg_p=.001
#         , reg_r=.035
#         , reg_f=1.5
#         , random_state=None
#         , verbose=False
#         , p_ut=None
#         , f_it=None
#         , tags=None
#         , ratings=None
#         ):
#
#         self.n_factors = n_factors
#         self.n_epochs = n_epochs
#         self.biased = biased
#         self.init_mean = init_mean
#         self.init_std_dev = init_std_dev
#         self.lr_bu = lr_bu if lr_bu is not None else lr_all
#         self.lr_bi = lr_bi if lr_bi is not None else lr_all
#         self.lr_bt = lr_bt if lr_bt is not None else lr_all
#
#         self.lr_pu = lr_pu if lr_pu is not None else lr_all
#         self.lr_qi = lr_qi if lr_qi is not None else lr_all
#         self.lr_rt = lr_rt if lr_rt is not None else lr_all
#
#         self.reg_p = reg_p
#         self.reg_f = reg_f
#         self.reg_r = reg_r
#
#         self.random_state = random_state
#         self.verbose = verbose
#
#         self.p_ut = p_ut
#         self.f_it = f_it
#         self.tags = tags
#         self.ratings = ratings
#         AlgoBase.__init__(self)
#
#     def fit(self,trainset):
#         AlgoBase.fit(self, trainset)
#         self.sgd(trainset)
#         return self
#
#     def sgd(self, trainset):
#         cdef np.ndarray[np.double_t] bu
#         cdef np.ndarray[np.double_t] bi
#         cdef np.ndarray[np.double_t] bt
#
#         cdef np.ndarray[np.double_t, ndim=2] pu
#         cdef np.ndarray[np.double_t, ndim=2] qi
#         cdef np.ndarray[np.double_t, ndim=2] rt
#
#         cdef int u, i, t, f, raw_u, raw_i, raw_t
#         cdef double r, p_put, p_fit, err_r, err_p, err_f, dot_r, dot_p, dot_f, puf, qif, rtf, global_mean_p, global_mean_f
#         cdef double global_mean_r = self.trainset.global_mean
#
#         cdef double lr_bu = self.lr_bu
#         cdef double lr_bi = self.lr_bi
#         cdef double lr_bt = self.lr_bt
#
#         cdef double lr_pu = self.lr_pu
#         cdef double lr_qi = self.lr_qi
#         cdef double lr_rt = self.lr_rt
#
#         cdef double reg_p = self.reg_p
#         cdef double reg_f = self.reg_f
#         cdef double reg_r = self.reg_r
#
#         p_ut = self.p_ut
#         f_it = self.f_it
#         tags = self.tags
#         ratings = self.ratings
#
#         cdef int n_factors = self.n_factors
#         raw_user = np.zeros(trainset.n_users, int)
#         raw_item = np.zeros(trainset.n_items, int)
#
#
#         data_ref = []
#         for u, i, r in trainset.all_ratings():
#             data_ref.append(str(int(trainset.to_raw_uid(u))) + " " + str(int(trainset.to_raw_iid(i))))
#
#         raw_user = np.zeros(trainset.n_users, int)
#         for i in trainset.all_users():
#             raw_user[i] = trainset.to_raw_uid(i)
#
#         raw_item = np.zeros(trainset.n_items, int)
#         for i in trainset.all_items():
#             raw_item[i] = trainset.to_raw_iid(i)
#
#         raw_data = ratings[ratings.ref_key.isin(data_ref)]
#
#         uni_tid = raw_data.tid.unique()
#         uni_tid = uni_tid[~np.isnan(uni_tid)]
#         u_t = pd.DataFrame({'tid': uni_tid,
#                             'tid_inner':range(len(uni_tid))})
#
#         raw_data = pd.merge(raw_data, u_t, how ='left', on=['tid'])
#
#         final_p = p_ut[p_ut.userId.isin(raw_data.userId) & p_ut.tid.isin(raw_data.tid)]
#         final_f = f_it[f_it.movieId.isin(raw_data.movieId) & f_it.tid.isin(raw_data.tid)]
#
#         final_p = pd.merge(final_p, u_t, how='left', on=['tid'])
#         final_f = pd.merge(final_f, u_t, how='left', on=['tid'])
#
#         p_ut = final_p.drop(['tid'], axis=1)
#         f_it = final_f.drop(['tid'], axis=1)
#
#         rng = get_rng(self.random_state)
#
#         bu = np.zeros(trainset.n_users, np.double)
#         bi = np.zeros(trainset.n_items, np.double)
#
#         bt = np.zeros(len(p_ut.tid_inner.unique()), np.double)
#
#         pu = rng.normal(self.init_mean, self.init_std_dev,
#                         (trainset.n_users, n_factors))
#         qi = rng.normal(self.init_mean, self.init_std_dev,
#                         (trainset.n_items, n_factors))
#         rt = rng.normal(self.init_mean, self.init_std_dev,
#                         (len(p_ut.tid_inner.unique()), n_factors))
#
#         global_mean_p = np.mean(p_ut.val)
#         global_mean_f = np.mean(f_it.val)
#
#         raw_data = raw_data.sort_values(by=['ref_key']).reset_index()
#
#         for current_epoch in range(self.n_epochs):
#             if self.verbose:
#                 print("Processing epoch {}".format(current_epoch), end='\r')
#             prev_ref = ""
#             for rn in range(len(raw_data)):
#                 ref = raw_data.loc[rn, 'ref_key']
#                 r = raw_data.loc[rn, 'rating']
#                 raw_u = raw_data.loc[rn, 'userId']
#                 raw_i = raw_data.loc[rn, 'movieId']
#
#                 known_item = True
#                 known_user = True
#                 try:
#                     temp = trainset.to_inner_iid(raw_i)
#                 except:
#                     known_item = False
#
#                 try:
#                     temp = trainset.to_inner_uid(raw_u)
#                 except:
#                     known_user = False
#
#                 if (r != 0) | (~np.isnan(r)):
#                     u = trainset.to_inner_uid(raw_u)
#                     i = trainset.to_inner_iid(raw_i)
#
#                     if pd.isna(raw_data.loc[rn, 'tid_inner']) | np.isnan(raw_data.loc[rn, 'tid_inner']):
#                         t = -1
#                     else:
#                         t = raw_data.loc[rn, 'tid_inner']
#
#                     t = int(t)
#
#                     if t != -1:
#                         p_put = p_ut.val[(p_ut.userId == raw_u) & (p_ut.tid_inner == t)].values[0]
#                         p_fit = f_it.val[(f_it.movieId == raw_i) & (f_it.tid_inner == t)].values[0]
#
#                     dot_r = 0.0
#                     dot_p = 0.0
#                     dot_f = 0.0
#
#                     if (prev_ref <> ref) | (rn == 0):
#                         dot_r = np.dot(qi[i], pu[u])
#                         err_r = r - (global_mean_r + bu[u] + bi[i] + dot_r)
#                         rv_sum = err_r * qi[i]
#                         ru_sum = err_r * pu[u]
#
#                     if t != -1:
#                         dot_p = np.dot(rt[t], pu[u])
#                         dot_f = np.dot(rt[t], qi[i])
#                         err_p = p_put - (global_mean_p + bu[u] + bt[t] + dot_p)
#                         err_f = p_fit - (global_mean_f + bi[i] + bt[t] + dot_f)
#                     else:
#                         err_p = 0.0
#                         err_f = 0.0
#
#                     if self.biased:
#                         bu[u] -= lr_bu * (-1 * err_r - reg_p * err_p + reg_r * bu[u])
#                         bi[i] -= lr_bi * (-1 * err_r - reg_f * err_f + reg_r * bi[i])
#                         if t != -1:
#                             bt[t] -= lr_bt * (-1 * reg_p * err_p - reg_f * err_f + reg_r * bt[t])
#
#                     if t != -1:
#                         pz_sum = err_p * rt[t]
#                         fz_sum = err_f * rt[t]
#                         pu_sum = err_p * pu[u]
#                         fv_sum = err_f * qi[i]
#                     else:
#                         pz_sum = 0
#                         fz_sum = 0
#                         pu_sum = 0
#                         fv_sum = 0
#
#                     pu[u] += lr_pu * (rv_sum + reg_p * pz_sum - reg_r * pu[u])
#                     qi[i] += lr_pu * (ru_sum + reg_f * fz_sum - reg_r * qi[i])
#                     if t != -1:
#                         rt[t] += lr_pu * (reg_p * pu_sum + reg_f * fv_sum - reg_r * rt[t])
#                 prev_ref = ref
#
#         self.bu = bu
#         self.bi = bi
#         self.pu = pu
#         self.qi = qi
#
#     def estimate(self, u, i):
#         known_user = self.trainset.knows_user(u)
#         known_item = self.trainset.knows_item(i)
#
#         if self.biased:
#             est = self.trainset.global_mean
#
#             if known_user:
#                 est += self.bu[u]
#
#             if known_item:
#                 est += self.bi[i]
#
#             if known_user and known_item:
#                 est += np.dot(self.qi[i], self.pu[u])
#         else:
#             if known_user and known_item:
#                 est = np.dot(self.qi[i], self.pu[u])
#             else:
#                 raise PredictionImpossible('User and item are unknown')
#         return est

class CoSVDv9(AlgoBase):
    def __init__ (
        self
        , n_factors = 40
        , n_epochs = 1
        , biased = True
        , init_mean = 0
        , init_std_dev=.1

        , lr_all=.005
        , reg_all=.02
        , lr_bu=None
        , lr_bi=None
        , lr_bt=None
        , lr_pu=None
        , lr_qi=None
        , lr_rt=None

        , reg_p=.001
        , reg_r=.035
        , reg_f=1.5

        , random_state=None
        , verbose=False
        , tags=None
        ):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_bt = lr_bt if lr_bt is not None else lr_all

        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.lr_rt = lr_rt if lr_rt is not None else lr_all

        self.reg_p = reg_p
        self.reg_f = reg_f
        self.reg_r = reg_r

        self.random_state = random_state
        self.verbose = verbose
        self.tags = tags
        AlgoBase.__init__(self)



    def fit(self,trainset):
        temp = pd.Series(list(trainset.all_ratings()))
        rate = pd.DataFrame(temp.apply(lambda x: trainset.to_raw_uid(x[0])), columns=['userId'])
        rate['movieId'] = temp.apply(lambda x: trainset.to_raw_iid(x[1]))
        rate['rating'] = temp.apply(lambda x: x[2])

        p_ut , f_it, tags, ratings = generateTagsOrigin(rate, self.tags)
        self.p_ut = p_ut
        self.f_it = f_it
        self.ratings = ratings

        AlgoBase.fit(self, trainset)
        self.sgd(trainset)
        return self


    def sgd(self, trainset):

        cdef np.ndarray[np.double_t] bu
        cdef np.ndarray[np.double_t] bi
        cdef np.ndarray[np.double_t] bt

        cdef np.ndarray[np.double_t, ndim=2] pu
        cdef np.ndarray[np.double_t, ndim=2] qi
        cdef np.ndarray[np.double_t, ndim=2] rt

        cdef int u, i, t, f, raw_u, raw_i, raw_t
        cdef double r, p_put, p_fit, err_r, err_p, err_f, dot_r, dot_p, dot_f, puf, qif, rtf, global_mean_p, global_mean_f
        cdef double global_mean_r = self.trainset.global_mean

        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_bt = self.lr_bt

        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi
        cdef double lr_rt = self.lr_rt

        cdef double reg_p = self.reg_p
        cdef double reg_f = self.reg_f
        cdef double reg_r = self.reg_r

        p_ut = self.p_ut
        f_it = self.f_it
        tags = self.tags
        ratings = self.ratings

        cdef int n_factors = self.n_factors
        raw_user = np.zeros(trainset.n_users, int)
        raw_item = np.zeros(trainset.n_items, int)

        temp = pd.Series(list(trainset.all_ratings()))
        data_ref = pd.DataFrame(temp.apply(lambda x: trainset.to_raw_uid(x[0])), columns=['userId'])
        data_ref['movieId'] = temp.apply(lambda x: trainset.to_raw_iid(x[1]))

        raw_data = ratings[ratings[['userId', 'movieId']].apply(tuple, axis=1).isin(data_ref[['userId', 'movieId']].apply(tuple, axis=1))]

        uni_tid = raw_data.tid.unique()
        uni_tid = uni_tid[~np.isnan(uni_tid)]
        u_t = pd.DataFrame({'tid': uni_tid,
                            'tid_inner':range(len(uni_tid))})

        raw_data = pd.merge(raw_data, u_t, how ='left', on=['tid'])

        final_p = p_ut[p_ut.userId.isin(raw_data.userId) & p_ut.tid.isin(raw_data.tid)]
        final_f = f_it[f_it.movieId.isin(raw_data.movieId) & f_it.tid.isin(raw_data.tid)]

        final_p = pd.merge(final_p, u_t, how='left', on=['tid'])
        final_f = pd.merge(final_f, u_t, how='left', on=['tid'])

        p_ut = final_p.drop(['tid'], axis=1)
        f_it = final_f.drop(['tid'], axis=1)

        rng = get_rng(self.random_state)

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)

        bt = np.zeros(len(p_ut.tid_inner.unique()), np.double)

        pu = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_users, n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, n_factors))
        rt = rng.normal(self.init_mean, self.init_std_dev,
                        (len(p_ut.tid_inner.unique()), n_factors))

        global_mean_p = np.mean(p_ut.val)
        global_mean_f = np.mean(f_it.val)

        raw_data['ref_key'] = raw_data.apply(lambda x: str(int(x['userId'])) + " " + str(int(x['movieId'])), axis=1)
        raw_data = raw_data.sort_values(by=['ref_key']).reset_index()

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch), end='\r')
            prev_ref = ""
            for rn in range(len(raw_data)):
                ref = raw_data.loc[rn, 'ref_key']
                r = raw_data.loc[rn, 'rating']
                raw_u = raw_data.loc[rn, 'userId']
                raw_i = raw_data.loc[rn, 'movieId']

                known_item = True
                known_user = True
                try:
                    temp = trainset.to_inner_iid(raw_i)
                except:
                    known_item = False

                try:
                    temp = trainset.to_inner_uid(raw_u)
                except:
                    known_user = False

                if (r != 0) | (~np.isnan(r)):
                    u = trainset.to_inner_uid(raw_u)
                    i = trainset.to_inner_iid(raw_i)

                    if pd.isna(raw_data.loc[rn, 'tid_inner']) | np.isnan(raw_data.loc[rn, 'tid_inner']):
                        t = -1
                    else:
                        t = raw_data.loc[rn, 'tid_inner']

                    t = int(t)

                    if t != -1:
                        p_put = p_ut.val[(p_ut.userId == raw_u) & (p_ut.tid_inner == t)].values[0]
                        p_fit = f_it.val[(f_it.movieId == raw_i) & (f_it.tid_inner == t)].values[0]

                    dot_r = 0.0
                    dot_p = 0.0
                    dot_f = 0.0

                    if (prev_ref <> ref) | (rn == 0):
                        dot_r = np.dot(qi[i], pu[u])
                        err_r = r - (global_mean_r + bu[u] + bi[i] + dot_r)
                        rv_sum = err_r * qi[i]
                        ru_sum = err_r * pu[u]

                    if t != -1:
                        dot_p = np.dot(rt[t], pu[u])
                        dot_f = np.dot(rt[t], qi[i])
                        err_p = p_put - (global_mean_p + bu[u] + bt[t] + dot_p)
                        err_f = p_fit - (global_mean_f + bi[i] + bt[t] + dot_f)
                    else:
                        err_p = 0.0
                        err_f = 0.0

                    if self.biased:
                        bu[u] -= lr_bu * (-1 * err_r - reg_p * err_p + reg_r * bu[u])
                        bi[i] -= lr_bi * (-1 * err_r - reg_f * err_f + reg_r * bi[i])
                        if t != -1:
                            bt[t] -= lr_bt * (-1 * reg_p * err_p - reg_f * err_f + reg_r * bt[t])

                    if t != -1:
                        pz_sum = err_p * rt[t]
                        fz_sum = err_f * rt[t]
                        pu_sum = err_p * pu[u]
                        fv_sum = err_f * qi[i]
                    else:
                        pz_sum = 0
                        fz_sum = 0
                        pu_sum = 0
                        fv_sum = 0

                    pu[u] += lr_pu * (rv_sum + reg_p * pz_sum - reg_r * pu[u])
                    qi[i] += lr_pu * (ru_sum + reg_f * fz_sum - reg_r * qi[i])
                    if t != -1:
                        rt[t] += lr_pu * (reg_p * pu_sum + reg_f * fv_sum - reg_r * rt[t])
                prev_ref = ref

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])
        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown')
        return est

class CoSVD(AlgoBase):
    def __init__ (
        self
        , n_factors = 40
        , n_epochs = 1
        , biased = True
        , init_mean = 0
        , init_std_dev=.1

        , lr_all=.005
        , reg_all=.02
        , lr_bu=None
        , lr_bi=None
        , lr_bt=None
        , lr_pu=None
        , lr_qi=None
        , lr_rt=None

        , reg_p=.001
        , reg_r=.035
        , reg_f=1.5

        , random_state=None
        , verbose=False
        , tags=None
        ):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_bt = lr_bt if lr_bt is not None else lr_all

        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.lr_rt = lr_rt if lr_rt is not None else lr_all

        self.reg_p = reg_p
        self.reg_f = reg_f
        self.reg_r = reg_r

        self.random_state = random_state
        self.verbose = verbose
        self.tags = tags
        AlgoBase.__init__(self)



    def fit(self,trainset):
        temp = pd.Series(list(trainset.all_ratings()))
        rate = pd.DataFrame(temp.apply(lambda x: trainset.to_raw_uid(x[0])), columns=['userId'])
        rate['movieId'] = temp.apply(lambda x: trainset.to_raw_iid(x[1]))
        rate['rating'] = temp.apply(lambda x: x[2])

        p_ut , f_it, tags, ratings = generateTagsOrigin(rate, self.tags)
        self.p_ut = p_ut
        self.f_it = f_it
        self.ratings = ratings

        AlgoBase.fit(self, trainset)
        self.sgd(trainset)
        return self


    def sgd(self, trainset):

        cdef np.ndarray[np.double_t] bu
        cdef np.ndarray[np.double_t] bi
        cdef np.ndarray[np.double_t] bt

        cdef np.ndarray[np.double_t, ndim=2] pu
        cdef np.ndarray[np.double_t, ndim=2] qi
        cdef np.ndarray[np.double_t, ndim=2] rt

        cdef int u, i, t, f, raw_u, raw_i, raw_t
        cdef double r, p_put, p_fit, err_r, err_p, err_f, dot_r, dot_p, dot_f, puf, qif, rtf, global_mean_p, global_mean_f
        cdef double global_mean_r = self.trainset.global_mean

        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_bt = self.lr_bt

        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi
        cdef double lr_rt = self.lr_rt

        cdef double reg_p = self.reg_p
        cdef double reg_f = self.reg_f
        cdef double reg_r = self.reg_r

        p_ut = self.p_ut
        f_it = self.f_it
        tags = self.tags
        ratings = self.ratings

        cdef int n_factors = self.n_factors
        raw_user = np.zeros(trainset.n_users, int)
        raw_item = np.zeros(trainset.n_items, int)

        temp = pd.Series(list(trainset.all_ratings()))
        data_ref = pd.DataFrame(temp.apply(lambda x: trainset.to_raw_uid(x[0])), columns=['userId'])
        data_ref['movieId'] = temp.apply(lambda x: trainset.to_raw_iid(x[1]))

        raw_data = ratings[ratings[['userId', 'movieId']].apply(tuple, axis=1).isin(data_ref[['userId', 'movieId']].apply(tuple, axis=1))]

        uni_tid = raw_data.tid.unique()
        uni_tid = uni_tid[~np.isnan(uni_tid)]
        u_t = pd.DataFrame({'tid': uni_tid,
                            'tid_inner':range(len(uni_tid))})

        raw_data = pd.merge(raw_data, u_t, how ='left', on=['tid'])

        final_p = p_ut[p_ut.userId.isin(raw_data.userId) & p_ut.tid.isin(raw_data.tid)]
        final_f = f_it[f_it.movieId.isin(raw_data.movieId) & f_it.tid.isin(raw_data.tid)]

        final_p = pd.merge(final_p, u_t, how='left', on=['tid'])
        final_f = pd.merge(final_f, u_t, how='left', on=['tid'])

        p_ut = final_p.drop(['tid'], axis=1)
        f_it = final_f.drop(['tid'], axis=1)

        rng = get_rng(self.random_state)

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)

        bt = np.zeros(len(p_ut.tid_inner.unique()), np.double)

        pu = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_users, n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, n_factors))
        rt = rng.normal(self.init_mean, self.init_std_dev,
                        (len(p_ut.tid_inner.unique()), n_factors))

        global_mean_p = np.mean(p_ut.val)
        global_mean_f = np.mean(f_it.val)

        raw_data['ref_key'] = raw_data.apply(lambda x: str(int(x['userId'])) + " " + str(int(x['movieId'])), axis=1)
        raw_data = raw_data.sort_values(by=['ref_key']).reset_index()

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch), end='\r')
            prev_ref = ""
            for rn in range(len(raw_data)):
                ref = raw_data.loc[rn, 'ref_key']
                r = raw_data.loc[rn, 'rating']
                raw_u = raw_data.loc[rn, 'userId']
                raw_i = raw_data.loc[rn, 'movieId']

                known_item = True
                known_user = True
                try:
                    temp = trainset.to_inner_iid(raw_i)
                except:
                    known_item = False

                try:
                    temp = trainset.to_inner_uid(raw_u)
                except:
                    known_user = False

                if (r != 0) | (~np.isnan(r)):
                    u = trainset.to_inner_uid(raw_u)
                    i = trainset.to_inner_iid(raw_i)

                    if pd.isna(raw_data.loc[rn, 'tid_inner']) | np.isnan(raw_data.loc[rn, 'tid_inner']):
                        t = -1
                    else:
                        t = raw_data.loc[rn, 'tid_inner']

                    t = int(t)

                    if t != -1:
                        p_put = p_ut.val[(p_ut.userId == raw_u) & (p_ut.tid_inner == t)].values[0]
                        p_fit = f_it.val[(f_it.movieId == raw_i) & (f_it.tid_inner == t)].values[0]

                    dot_r = 0.0
                    dot_p = 0.0
                    dot_f = 0.0

                    if (prev_ref <> ref) | (rn == 0):
                        dot_r = np.dot(qi[i], pu[u])
                        err_r = r - (global_mean_r + bu[u] + bi[i] + dot_r)
                        rv_sum = err_r * qi[i]
                        ru_sum = err_r * pu[u]

                    if t != -1:
                        dot_p = np.dot(rt[t], pu[u])
                        dot_f = np.dot(rt[t], qi[i])
                        err_p = p_put - (global_mean_p + bu[u] + bt[t] + dot_p)
                        err_f = p_fit - (global_mean_f + bi[i] + bt[t] + dot_f)
                    else:
                        err_p = 0.0
                        err_f = 0.0

                    if self.biased:
                        bu[u] -= lr_bu * (-1 * err_r - reg_p * err_p + reg_r * bu[u])
                        bi[i] -= lr_bi * (-1 * err_r - reg_f * err_f + reg_r * bi[i])
                        if t != -1:
                            bt[t] -= lr_bt * (-1 * reg_p * err_p - reg_f * err_f + reg_r * bt[t])

                    if t != -1:
                        pz_sum = err_p * rt[t]
                        fz_sum = err_f * rt[t]
                        pu_sum = err_p * pu[u]
                        fv_sum = err_f * qi[i]
                    else:
                        pz_sum = 0
                        fz_sum = 0
                        pu_sum = 0
                        fv_sum = 0

                    pu[u] += lr_pu * (rv_sum + reg_p * pz_sum - reg_r * pu[u])
                    qi[i] += lr_pu * (ru_sum + reg_f * fz_sum - reg_r * qi[i])
                    if t != -1:
                        rt[t] += lr_pu * (reg_p * pu_sum + reg_f * fv_sum - reg_r * rt[t])
                prev_ref = ref

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])
        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown')
        return est

# class CoSVDGenome(AlgoBase):
#     def __init__ (
#         self
#         , n_factors = 40
#         , n_epochs = 1
#         , biased = True
#         , init_mean = 0
#         , init_std_dev=.1
#
#         , lr_all=.005
#         , reg_all=.02
#         , lr_bu=None
#         , lr_bi=None
#         , lr_bt=None
#         , lr_pu=None
#         , lr_qi=None
#         , lr_rt=None
#
#         , reg_p=.001
#         , reg_r=.035
#         , reg_f=1.5
#
#         , random_state=None
#         , verbose=False
#         , tags=None
#
#         , genome="normal"
#         , genome_tag=None
#         , genome_score=None
#         ):
#
#         self.n_factors = n_factors
#         self.n_epochs = n_epochs
#         self.biased = biased
#         self.init_mean = init_mean
#         self.init_std_dev = init_std_dev
#         self.lr_bu = lr_bu if lr_bu is not None else lr_all
#         self.lr_bi = lr_bi if lr_bi is not None else lr_all
#         self.lr_bt = lr_bt if lr_bt is not None else lr_all
#
#         self.lr_pu = lr_pu if lr_pu is not None else lr_all
#         self.lr_qi = lr_qi if lr_qi is not None else lr_all
#         self.lr_rt = lr_rt if lr_rt is not None else lr_all
#
#         self.reg_p = reg_p
#         self.reg_f = reg_f
#         self.reg_r = reg_r
#
#         self.random_state = random_state
#         self.verbose = verbose
#         self.tags = tags
#
#         self.genome = genome
#         self.genome_tag = genome_tag
#         self.genome_score = genome_score
#         AlgoBase.__init__(self)
#
#
#
#     def fit(self,trainset):
#         temp = pd.Series(list(trainset.all_ratings()))
#         rate = pd.DataFrame(temp.apply(lambda x: trainset.to_raw_uid(x[0])), columns=['userId'])
#         rate['movieId'] = temp.apply(lambda x: trainset.to_raw_iid(x[1]))
#         rate['rating'] = temp.apply(lambda x: x[2])
#
#         if self.genome == "normal":
#             p_ut , f_it, tags, ratings = generateTagsWithTagGenome(rate, self.tags, self.genome_tag, self.genome_score)
#         elif self.genome == "hybrid":
#             p_ut , f_it, tags, ratings = generateTagsWithTagGenomeHybrid(rate, self.tags, self.genome_tag, self.genome_score)
#         else:
#             print("please specific the type of tag genome matrix")
#         self.p_ut = p_ut
#         self.f_it = f_it
#         self.ratings = ratings
#
#         AlgoBase.fit(self, trainset)
#         self.sgd(trainset)
#         return self
#
#
#     def sgd(self, trainset):
#
#         cdef np.ndarray[np.double_t] bu
#         cdef np.ndarray[np.double_t] bi
#         cdef np.ndarray[np.double_t] bt
#
#         cdef np.ndarray[np.double_t, ndim=2] pu
#         cdef np.ndarray[np.double_t, ndim=2] qi
#         cdef np.ndarray[np.double_t, ndim=2] rt
#
#         cdef int u, i, t, f, raw_u, raw_i, raw_t
#         cdef double r, p_put, p_fit, err_r, err_p, err_f, dot_r, dot_p, dot_f, puf, qif, rtf, global_mean_p, global_mean_f
#         cdef double global_mean_r = self.trainset.global_mean
#
#         cdef double lr_bu = self.lr_bu
#         cdef double lr_bi = self.lr_bi
#         cdef double lr_bt = self.lr_bt
#
#         cdef double lr_pu = self.lr_pu
#         cdef double lr_qi = self.lr_qi
#         cdef double lr_rt = self.lr_rt
#
#         cdef double reg_p = self.reg_p
#         cdef double reg_f = self.reg_f
#         cdef double reg_r = self.reg_r
#
#         p_ut = self.p_ut
#         f_it = self.f_it
#         tags = self.tags
#         ratings = self.ratings
#
#         cdef int n_factors = self.n_factors
#         raw_user = np.zeros(trainset.n_users, int)
#         raw_item = np.zeros(trainset.n_items, int)
#
#         temp = pd.Series(list(trainset.all_ratings()))
#         data_ref = pd.DataFrame(temp.apply(lambda x: trainset.to_raw_uid(x[0])), columns=['userId'])
#         data_ref['movieId'] = temp.apply(lambda x: trainset.to_raw_iid(x[1]))
#
#         raw_data = ratings[ratings[['userId', 'movieId']].apply(tuple, axis=1).isin(data_ref[['userId', 'movieId']].apply(tuple, axis=1))]
#
#         uni_tid = raw_data.tid.unique()
#         uni_tid = uni_tid[~np.isnan(uni_tid)]
#         u_t = pd.DataFrame({'tid': uni_tid,
#                             'tid_inner':range(len(uni_tid))})
#
#         raw_data = pd.merge(raw_data, u_t, how ='left', on=['tid'])
#
#         #final_p = p_ut[p_ut[['userId', 'tid']].apply(tuple, axis=1).isin(raw_data[['userId', 'tid']].apply(tuple, axis=1))]
#         #final_f = f_it[f_it[['movieId', 'tid']].apply(tuple, axis=1).isin(raw_data[['movieId', 'tid']].apply(tuple, axis=1))]
#
#         final_p = p_ut[p_ut.userId.isin(raw_data.userId) & p_ut.tid.isin(raw_data.tid)]
#         final_f = f_it[f_it.movieId.isin(raw_data.movieId) & f_it.tid.isin(raw_data.tid)]
#
#         final_p = pd.merge(final_p, u_t, how='left', on=['tid'])
#         final_f = pd.merge(final_f, u_t, how='left', on=['tid'])
#
#         p_ut = final_p.drop(['tid'], axis=1)
#         f_it = final_f.drop(['tid'], axis=1)
#
#         rng = get_rng(self.random_state)
#
#         bu = np.zeros(trainset.n_users, np.double)
#         bi = np.zeros(trainset.n_items, np.double)
#
#         bt = np.zeros(len(p_ut.tid_inner.unique()), np.double)
#
#         pu = rng.normal(self.init_mean, self.init_std_dev,
#                         (trainset.n_users, n_factors))
#         qi = rng.normal(self.init_mean, self.init_std_dev,
#                         (trainset.n_items, n_factors))
#         rt = rng.normal(self.init_mean, self.init_std_dev,
#                         (len(p_ut.tid_inner.unique()), n_factors))
#
#         global_mean_p = np.mean(p_ut.val)
#         global_mean_f = np.mean(f_it.val)
#
#         raw_data['ref_key'] = raw_data.apply(lambda x: str(int(x['userId'])) + " " + str(int(x['movieId'])), axis=1)
#         raw_data = raw_data.sort_values(by=['ref_key']).reset_index()
#         f_movie_list = f_it.movieId.unique()
#         f_tag_list = f_it.tid_inner.unique()
#
#         for current_epoch in range(self.n_epochs):
#             if self.verbose:
#                 print("Processing epoch {}".format(current_epoch), end='\r')
#             prev_ref = ""
#             for rn in range(len(raw_data)):
#                 ref = raw_data.loc[rn, 'ref_key']
#                 r = raw_data.loc[rn, 'rating']
#                 raw_u = raw_data.loc[rn, 'userId']
#                 raw_i = raw_data.loc[rn, 'movieId']
#
#                 known_item = True
#                 known_user = True
#                 try:
#                     temp = trainset.to_inner_iid(raw_i)
#                 except:
#                     known_item = False
#
#                 try:
#                     temp = trainset.to_inner_uid(raw_u)
#                 except:
#                     known_user = False
#
#                 if (r != 0) | (~np.isnan(r)):
#                     u = trainset.to_inner_uid(raw_u)
#                     i = trainset.to_inner_iid(raw_i)
#                     if pd.isna(raw_data.loc[rn, 'tid_inner']) | np.isnan(raw_data.loc[rn, 'tid_inner']) | (raw_i not in f_movie_list) | (raw_data.loc[rn, 'tid_inner'] not in f_tag_list):
#                         t = -1
#                     else:
#                         t = raw_data.loc[rn, 'tid_inner']
#
#                     t = int(t)
#
#                     if t != -1:
#                         #print(raw_data.loc[rn, 'tid'])
#                         #print(raw_i)
#                         #print(raw_u)
#                         p_put = p_ut.val[(p_ut.userId == raw_u) & (p_ut.tid_inner == t)].values[0]
#                         p_fit = f_it.val[(f_it.movieId == raw_i) & (f_it.tid_inner == t)].values[0]
#
#                     dot_r = 0.0
#                     dot_p = 0.0
#                     dot_f = 0.0
#
#                     if (prev_ref <> ref) | (rn == 0):
#                         dot_r = np.dot(qi[i], pu[u])
#                         err_r = r - (global_mean_r + bu[u] + bi[i] + dot_r)
#                         rv_sum = err_r * qi[i]
#                         ru_sum = err_r * pu[u]
#
#                     if t != -1:
#                         dot_p = np.dot(rt[t], pu[u])
#                         dot_f = np.dot(rt[t], qi[i])
#                         err_p = p_put - (global_mean_p + bu[u] + bt[t] + dot_p)
#                         err_f = p_fit - (global_mean_f + bi[i] + bt[t] + dot_f)
#                     else:
#                         err_p = 0.0
#                         err_f = 0.0
#
#                     if self.biased:
#                         bu[u] -= lr_bu * (-1 * err_r - reg_p * err_p + reg_r * bu[u])
#                         bi[i] -= lr_bi * (-1 * err_r - reg_f * err_f + reg_r * bi[i])
#                         if t != -1:
#                             bt[t] -= lr_bt * (-1 * reg_p * err_p - reg_f * err_f + reg_r * bt[t])
#
#                     if t != -1:
#                         pz_sum = err_p * rt[t]
#                         fz_sum = err_f * rt[t]
#                         pu_sum = err_p * pu[u]
#                         fv_sum = err_f * qi[i]
#                     else:
#                         pz_sum = 0
#                         fz_sum = 0
#                         pu_sum = 0
#                         fv_sum = 0
#
#                     pu[u] += lr_pu * (rv_sum + reg_p * pz_sum - reg_r * pu[u])
#                     qi[i] += lr_pu * (ru_sum + reg_f * fz_sum - reg_r * qi[i])
#                     if t != -1:
#                         rt[t] += lr_pu * (reg_p * pu_sum + reg_f * fv_sum - reg_r * rt[t])
#                 prev_ref = ref
#
#         self.bu = bu
#         self.bi = bi
#         self.pu = pu
#         self.qi = qi
#
#     def estimate(self, u, i):
#         known_user = self.trainset.knows_user(u)
#         known_item = self.trainset.knows_item(i)
#
#         if self.biased:
#             est = self.trainset.global_mean
#
#             if known_user:
#                 est += self.bu[u]
#
#             if known_item:
#                 est += self.bi[i]
#
#             if known_user and known_item:
#                 est += np.dot(self.qi[i], self.pu[u])
#         else:
#             if known_user and known_item:
#                 est = np.dot(self.qi[i], self.pu[u])
#             else:
#                 raise PredictionImpossible('User and item are unknown')
#         return est
