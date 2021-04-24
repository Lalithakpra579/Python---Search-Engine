# TF-IDF python implementation

Implemented from http://www.tfidf.com/

### Before run:

```shell
pip3 install tqdm
```

### How to run

```shell
$: python3 main.py -h

#    usage: main.py [-h] [-s SEARCH] [-i]
#    
#    optional arguments:
#      -h, --help            show this help message and exit
#      -s SEARCH, --search SEARCH
#                            search the word
#      -i, --index           index the dataset
```

```shell
$: python3 main.py -s king

#  Andrew Lang___The Blue Fairy Book.txt - 0.0218087233081963
#  Andrew Lang___New Collected Rhymes.txt - 0.0046507494391743
#  Benjamin Disraeli___Ixion In Heaven.txt - 0.0036492440480141
#  Benjamin Disraeli___Alroy.txt - 0.0027756795253740
#  William Somerset Maugham___Orientations.txt - 0.0011496831233314
#  William Makepeace Thackeray___Roundabout Papers.txt - 0.0007258588875991
#  William Wordsworth___The Poetical Works of William Wordsworth, Volume 4.txt - 0.0006414692379514
#  William Wymark Jacobs___A Mixed Proposal, The Lady of the Barge and Others, Part 9.txt - 0.0003207157998348
#  Bret Harte___A Drift from Redwood Camp.txt - 0.0002853988814006
#  William Wordsworth___Poems in Two Volumes, Volume 2.txt - 0.0002023792278943
#  Anthony Trollope___Aaron Trow.txt - 0.0001702260783738
#  William Dean Howells___The White Mr. Longfellow.txt - 0.0001351889438213
#  William Penn___A Brief Account of the Rise and Progress of the People Called Quakers.txt - 0.0000943219909678
#  William Somerset Maugham___The Trembling of a Leaf.txt - 0.0000926363137826
```