## Partitioned Pagerank DataFrame & RDD

A dataframe & RDD based hybrid implementation for the Partitioned Pagerank algorithm

We have implemented hash partitioning in this, and experimented with various partition sized

To run:
Set the following vars correctly in ```run.sh```

PYTHON_BINARY = path to Python binary <br />
SPARK_INSTALL_DIR = Spark Install directory <br/> 
SPARK_MASTER = url to the spark master <br/>
INPUT_FILE = input file to run PageRank on (HDFS Path) <br/>
OUTPUT_DIR = output directory for the result (HDFS or local FS) <br/>

To run the code, just execute run.sh
```./run.sh```