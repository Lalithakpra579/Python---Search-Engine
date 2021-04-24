import argparse

import tfidf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--search', type=str, help='search the word')
    parser.add_argument('-i', '--index', default=False, action='store_true', help='index the dataset')
    args = parser.parse_args()

    if args.index:
        tfidf.reindex()

    if args.search:
        tfidf.search(args.search)


if __name__ == '__main__':
    main()
