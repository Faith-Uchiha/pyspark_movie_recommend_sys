# pyspark_movie_recommend_sys
a movie recommend system based on pyspark,hadoop

这是一个基于pyspark,hadoop实现的电影推荐系统。hadoop的hdfs作为数据的存储平台，spark用于模型训练，使用算法为spark的MLlib内置的交替最小二乘法ALS

数据集
- dataset:[movie lens latest small](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)

算法
- algorithm:als in spark

环境
- python 3.6
- hadoop 2.8.5
- pyspark

使用前先新建tools文件夹，并把下列两个安装包复制到tools中
- jdk-8u212-linux-x64.tar.gz
- hadoop-2.8.5.tar.gz

使用步骤

1. 执行./build_docker_image.sh。该脚本会根据Dockerfile进行镜像构建
2. 执行./create_network.sh。该脚本用于创建容器使用的网络
3. 执行./start_container.sh。该脚本会启动三个容器构建伪分布式hadoop集群，一主二从
4. 执行./upload_data_to_hdfs.sh。该脚本用于上传数据集到hdfs
5. 执行./get_recommend.sh。该脚本用于得到推荐结果，可根据需要自行修改该脚本

注意
- upload_data_to_hdfs.sh和get_recommend.sh 两个脚本无法执行，需要执行命令“chmod a+x 文件名”
- 如果出现ssh connection refused，说明某个容器的sshd服务未启动，可以手动启动，或者执行clean_container.sh再执行start_container 脚本

