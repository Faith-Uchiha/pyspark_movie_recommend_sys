#!/bin/bash

echo upload movie_lens data to hdfs!

docker exec -it hadoop-node1 hadoop fs -put ml-latest-small /
docker exec -it hadoop-node1 hadoop fs -ls /

echo upload success!