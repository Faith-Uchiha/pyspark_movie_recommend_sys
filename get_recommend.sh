#!/bin/bash

echo 'Usage:U610 10 means recommend 10 movies to user 610;M999 10 means recommend movie 999 to 10 users'
docker exec -it hadoop-node1 bash -c 'python3 /code/movie_rec2.py U610 10'
