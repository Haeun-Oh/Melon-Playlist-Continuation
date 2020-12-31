# arena_util.py
# -*- coding: utf-8 -*-
import os
import io
import json
import distutils.dir_util
from collections import Counter
import numpy as np
import scipy.sparse as spr
def write_json(data, fname): #json파일 만들기
    def _conv(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath(parent)
    with io.open(fname, "w", encoding="utf8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)

def load_json(fname):#json파일 불러오기
    with open(fname, encoding='utf8') as f:
        json_obj = json.load(f)

    return json_obj

def debug_json(r):#json debug
    print(json.dumps(r, ensure_ascii=False, indent=4))

# evaluate.py
# -*- coding: utf-8 -*-
# import fire

# from arena_util import load_json
class CustomEvaluator:
    def _idcg(self, l): #idcg
        return sum((1.0 / np.log(i + 2) for i in range(l)))

    def __init__(self):
        self._idcgs = [self._idcg(i) for i in range(101)]

    def _ndcg(self, gt, rec): #ndcg
        dcg = 0.0
        for i, r in enumerate(rec):
            if r in gt:
                dcg += 1.0 / np.log(i + 2)

        return dcg / self._idcgs[len(gt)]

    def _eval(self, gt_fname, rec_fname): #ndcg 계산하기
        gt_playlists = load_json(gt_fname)
        gt_dict = {g["id"]: g for g in gt_playlists}
        rec_playlists = load_json(rec_fname)

        music_ndcg = 0.0
        tag_ndcg = 0.0

        for rec in rec_playlists:
            gt = gt_dict[rec["id"]]
            music_ndcg += self._ndcg(gt["songs"], rec["songs"][:100])
            tag_ndcg += self._ndcg(gt["tags"], rec["tags"][:10])

        music_ndcg = music_ndcg / len(rec_playlists)
        tag_ndcg = tag_ndcg / len(rec_playlists)
        score = music_ndcg * 0.85 + tag_ndcg * 0.15

        return music_ndcg, tag_ndcg, score

    def evaluate(self, gt_fname, rec_fname): #necg결과 보여주기
        try:
            music_ndcg, tag_ndcg, score = self._eval(gt_fname, rec_fname)
            print(f"Music nDCG: {music_ndcg:.6}")
            print(f"Tag nDCG: {tag_ndcg:.6}")
            print(f"Score: {score:.6}")
        except Exception as e:
            print(e)
#############################################
# if __name__ == "__main__":
#     fire.Fire(ArenaEvaluator)
import pandas as pd
from tqdm import tqdm
song_meta = pd.read_json("./data/song_meta.json")
train = pd.read_json("./data/train.json")
test = pd.read_json("./data/val.json")

train['istrain'] = 1
test['istrain'] = 0

n_train = len(train)
n_test = len(test)

# train + test
plylst = pd.concat([train, test], ignore_index=True)

# playlist id
plylst["nid"] = range(n_train + n_test)

# id <-> nid
plylst_id_nid = dict(zip(plylst["id"],plylst["nid"]))
plylst_nid_id = dict(zip(plylst["nid"],plylst["id"]))

####
plylst_tag = plylst['tags'] #playlist - playlist의 tag
tag_counter = Counter([tg for tgs in plylst_tag for tg in tgs])
tag_dict = {x: tag_counter[x] for x in tag_counter}

tag_id_tid = dict()
tag_tid_id = dict()
for i, t in enumerate(tag_dict):
  tag_id_tid[t] = i
  tag_tid_id[i] = t

n_tags = len(tag_dict)

####
plylst_song = plylst['songs'] #playlist - playlist에 있는 노래들
song_counter = Counter([sg for sgs in plylst_song for sg in sgs]) #type: collections.Counter
song_dict = {x: song_counter[x] for x in song_counter} #type: dict

song_id_sid = dict() #song의 id: 순서대로 번호 매기기
song_sid_id = dict()
for i, t in enumerate(song_dict):
  song_id_sid[t] = i
  song_sid_id[i] = t

n_songs = len(song_dict)

#song id의 순서번호를 넣어준다.
plylst['songs_id'] = plylst['songs'].map(lambda x: [song_id_sid.get(s) for s in x if song_id_sid.get(s) != None])
#tag id의 순서번호를 넣어준다.
plylst['tags_id'] = plylst['tags'].map(lambda x: [tag_id_tid.get(t) for t in x if tag_id_tid.get(t) != None])

plylst_use = plylst[['istrain','nid','updt_date','songs_id','tags_id']]
plylst_use.loc[:,'num_songs'] = plylst_use['songs_id'].map(len) # playlist의 song개수 추가
plylst_use.loc[:,'num_tags'] = plylst_use['tags_id'].map(len) #playlist의 tag개수 추가
plylst_use = plylst_use.set_index('nid') #palylist번호 맨 앞으로

plylst_train = plylst_use.iloc[:n_train,:]
plylst_test = plylst_use.iloc[n_train:,:]

# sample test
np.random.seed(33)
n_sample = 300

# test = plylst_test.iloc[np.random.choice(range(n_test), n_sample, replace=False),:]

# real test
test = plylst_test
# print(len(test))

row = np.repeat(range(n_train), plylst_train['num_songs'])
col = [song for songs in plylst_train['songs_id'] for song in songs]
dat = np.repeat(1, plylst_train['num_songs'].sum())
train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_songs))
#->playlist와 song의 co-occurence matrix

row = np.repeat(range(n_train), plylst_train['num_tags'])
col = [tag for tags in plylst_train['tags_id'] for tag in tags]
dat = np.repeat(1, plylst_train['num_tags'].sum())
train_tags_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_tags)) #sparse matrix->CSR(Comporess Spare Row) matrix의 index로 압축하여 저장/ 0이 많은 경우 압축률이 좋다.
#->palylist와 tag의 co-occurence matrix

test_row = np.repeat(range(n_test), plylst_test['num_songs'])
test_col = [song for songs in plylst_test['songs_id'] for song in songs]
test_dat = np.repeat(1, plylst_test['num_songs'].sum())
test_songs_A = spr.csr_matrix((test_dat, (test_row, test_col)), shape=(n_test, n_songs))
# -> test playlist와 song의 co-occurence matrix

test_row = np.repeat(range(n_test), plylst_test['num_tags'])
test_col = [song for songs in plylst_test['tags_id'] for song in songs]
test_dat = np.repeat(1, plylst_test['num_tags'].sum())
test_tags_A = spr.csr_matrix((test_dat, (test_row, test_col)), shape=(n_test, n_tags))

train_songs_A_T = train_songs_A.T.tocsr()
train_tags_A_T = train_tags_A.T.tocsr()

def rec(pids):
    tt = 1

    res = []

    for pid in pids:
        p1 = test_songs_A[pid-n_train]
        pt = p1.T
        p = pt.toarray()

        p2 = test_tags_A[pid - n_train]
        pt2 = p2.T
        pp = pt2.toarray()

        songs_already = test.loc[pid, "songs_id"]
        tags_already = test.loc[pid, "tags_id"]

        simpls = train_songs_A.dot(p)
        simpls2 = np.zeros_like(simpls)

        inds = train_songs_A.dot(p).reshape(-1).argsort()[-9000:][::-1] #.reshape(-1) == .reshape(1, -1)처럼 1차원 배열 반환
        vals = simpls[inds]  # .reshape(-1) == .reshape(1, -1)처럼 1차원 배열 반환

        m =np.max(vals)
        if(m==0):
            m+= 0.01

        vals2 = ((vals - np.min(vals)) * (1/m))**2
        simpls2[inds] = vals
####################################################################################
        simplst = train_tags_A.dot(pp)
        simpls2t = np.zeros_like(simplst)

        indst = train_tags_A.dot(pp).reshape(-1).argsort()[-9000:][::-1]  # .reshape(-1) == .reshape(1, -1)처럼 1차원 배열 반환
        valst = simpls[indst]  # .reshape(-1) == .reshape(1, -1)처럼 1차원 배열 반환

        mt = np.max(valst)
        if (mt == 0):
            mt += 0.01

        vals2t = ((valst - np.min(valst)) * (1 / mt)) ** 2
        simpls2t[indst] = valst

        cand_song = train_songs_A_T[:, inds].dot(vals2)
        cand_song_idx = cand_song.reshape(-1).argsort()[-200:][::-1] #내림차순 정렬

        cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:100] #playlist에 원래 있던 song이 아닌 것들 100개
        rec_song_idx = [song_sid_id[i] for i in cand_song_idx]

        cand_tag = train_tags_A_T[:, indst].dot(vals2t)
        cand_tag_idx = cand_tag.reshape(-1).argsort()[-15:][::-1]

        cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:10]
        rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]

        res.append({
            "id": plylst_nid_id[pid],
            "songs": rec_song_idx,
            "tags": rec_tag_idx
        })

        if tt % 1000 == 0:
            print(tt)

        tt += 1
    return res

answers = rec(test.index)

write_json(answers, "./results.json")

# evaluator = CustomEvaluator()
# evaluator.evaluate("./answers/val.json", "./results.json")