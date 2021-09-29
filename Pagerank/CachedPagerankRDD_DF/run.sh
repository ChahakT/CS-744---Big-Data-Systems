#!/usr/bin/env bash

PYTHON_BINARY='/usr/bin/python3'
SPARK_INSTALL_DIR='/mnt/data/spark-3.1.2-bin-hadoop3.2'
PYSPARK_INCLUDE="${SPARK_INSTALL_DIR}/python"
PY4J_INCLUDE="${SPARK_INSTALL_DIR}/python/lib/py4j-0.10.9-src.zip"

export SPARK_MASTER='spark://c220g5-111031vm-1.wisc.cloudlab.us:7077'
export INPUT_FILE='hdfs://10.10.1.1:9000/enwiki-pages-articles/*'
export OUTPUT_DIR='hdfs://10.10.1.1:9000/pagerank-naive-rdddf'
export PYTHONPATH="${PYTHONPATH}:${PYSPARK_INCLUDE}:${PY4J_INCLUDE}"

${PYTHON_BINARY} 'cached_pagerank_rdd_df.py'
