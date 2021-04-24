import glob
import json
import math
import os
import re

from tqdm import tqdm

dataset_dir = 'dataset'
out_dir = 'output'
data_files = list(map(os.path.basename, glob.glob(f"{dataset_dir}/*.txt")))
indexed_files = list()
final_index_file = f'{out_dir}/index.json'

with open('stopwords.txt') as stopwords_file:
    stopwords = [w.lower().strip() for w in stopwords_file.readlines()]


def clean(s: str) -> str:
    return re.sub('[^A-Za-z0-9]+', '', s).lower().strip()


def words_filter(s: str) -> bool:
    return not (s.isnumeric() or s in stopwords)


def get_tokens(fn: str) -> list:
    with open(f'{dataset_dir}/{fn}') as txt_file:
        data = txt_file.read()
    rv = [clean(s) for s in re.split(r'[^A-Za-z0-9]+', data) if len(s.strip()) > 2]
    rv = list(filter(words_filter, rv))
    del data
    return rv


def file_tf(fn: str) -> dict:
    rv = {}
    tokens = get_tokens(fn)
    tokens_set = set(tokens)
    tokens_len = len(tokens_set)

    for t in tqdm(tokens_set, desc=fn):
        if t:
            count = tokens.count(t)
            tf = float(count) / tokens_len
            rv[t] = tf

    del tokens
    del tokens_set

    return rv


def get_index(fp: str) -> dict:
    with open(f'{out_dir}/{fp}') as index_file:
        index = json.load(index_file)
    return index


def get_num_docs(t: str) -> list:
    rv = []
    for index_fn in indexed_files:
        index = get_index(index_fn)
        if t in index:
            rv.append({'file': index_fn, 'tf': index[t]})

    return rv


def reindex():
    print("Reindexing dataset...")
    old_files = glob.glob(f"{out_dir}/*")
    for f in old_files:
        os.remove(f)

    for f in data_files:
        tf_map = file_tf(f)
        with open(f'{out_dir}/{f}.json', 'w') as outfile:
            outfile.write(json.dumps(tf_map, indent=2))
            print(f'Index created for: {f}', flush=True)

    global indexed_files
    indexed_files = list(map(os.path.basename, glob.glob(f"{out_dir}/*.txt.json")))

    all_tokens = set()
    for f in indexed_files:
        index = get_index(f)
        all_tokens.update(index.keys())

    final_index = {}

    for t in tqdm(all_tokens, desc=f'Calculating TF-IDF'):
        docs = get_num_docs(t)
        idf = math.log(float(1 + len(indexed_files)) / (1 + len(docs)))
        for d in docs:
            d['tfidf'] = d['tf'] * idf

        final_index[t] = sorted(docs, key=lambda x: x['tfidf'], reverse=True)

    with open(final_index_file, 'w') as f:
        f.write(json.dumps(final_index, indent=2))
        print('Index updated...', flush=True)


def search(t: str):
    if not os.path.exists(final_index_file):
        print("Index file not found!")
        reindex()

    t = clean(t)
    with open(final_index_file) as f:
        index = json.load(f)

    if t in index:
        for r in map(lambda x: f"{x['file'].replace('.json', '')} - {x['tfidf']:.16f}", index[t]):
            print(r)
    else:
        print(f'{t}: Not found!')
