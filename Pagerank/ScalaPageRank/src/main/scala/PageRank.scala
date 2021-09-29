import org.apache.spark.sql.SparkSession

object PageRank {
  def main(args: Array[String]) {
    //Define the HDFS locations for input and output based on user args
    val hdfsPath = "hdfs://10.10.1.1:9000/"
    val inputFile = hdfsPath + args(0)
    val outputFile = hdfsPath + args(1)
    //Build the spark context
    val spark = SparkSession
      .builder
      .appName("ScalaPageRank")
      .config("spark.default.parallelism", "5")
      .getOrCreate()

    val lines = spark.sparkContext.textFile(inputFile)
    //Filter the RDD so that it doesn't start with tab/space, contains only key and value, and we consider only those links with ":" that have the word "Category" prepended to it
    val filteredLines = lines.filter(!_.startsWith("#"))
                    .map(line => line.split("\\t+"))
                    .map(_.filter(_.nonEmpty))
                    .filter(_.length == 2)
                    .map(_.map(_.toLowerCase()))
                    .filter(_.forall(x => !x.contains(":") || x.startsWith("Category:")))
                    .map(parts => parts(0) -> parts(1))

    //Perform groupByKey in order to have data convert from 
    //FromNode ToNode                       FromNode  ToNode
    //1         2                           1         [2,3,4]
    //1         3               to          2          5
    //1         4
    //2         5 
    val links = filteredLines.distinct().groupByKey().cache()
    //Set initial ranks of FromNodes to 1
    var ranks = links.mapValues(v => 1.0)

    //Perform 10 iterations of getting updated ranks for ToNodes. The new rank for a node is now going to be 0.15 + 0.85 * (rank(FromNode) / numOutboundLinks(FromNode))
    for (i <- 1 to 10) {
      val contribs = links.join(ranks).values.flatMap{ case (urls, rank) =>
        val numOutboundLinks = urls.size
        urls.map(url => (url, rank / numOutboundLinks))
      }
      ranks = contribs.reduceByKey(_ + _).mapValues(0.15 + 0.85 * _)
    }
    ranks.collect()
    ranks.saveAsTextFile(outputFile)
  }
}

