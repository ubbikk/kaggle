#!/usr/bin/env bash

#Google Drive
#word2vec
cd kg
mkdir data
cd data

#Download word2vec
gdrive download 0B_ww4khOD5AlZWh4eEtlUENRTWc
gzip -d GoogleNews-vectors-negative300.bin.gz

#git lfs pull