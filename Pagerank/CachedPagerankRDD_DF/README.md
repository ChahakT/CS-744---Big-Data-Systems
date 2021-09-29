## Cached Pagerank DataFrame & RDD

A dataframe & RDD based hybrid implementation for the Cached Pagerank algorithm

The links dataset is cached

To run:
Set the following vars correctly in ```run.sh```

PYTHON_BINARY = path to Python binary <br />
SPARK_INSTALL_DIR = Spark Install directory <br/> 
SPARK_MASTER = url to the spark master <br/>
INPUT_FILE = input file to run PageRank on (HDFS Path) <br/>
OUTPUT_DIR = output directory for the result (HDFS or local FS) <br/>

To run the code, just execute run.sh
```./run.sh```