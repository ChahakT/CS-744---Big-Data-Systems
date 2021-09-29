import org.apache.spark.sql.SparkSession

object Sorting {
  def main(args: Array[String]) {
    //Get the input file from hdfs location
    val hdfsPath = "hdfs://10.10.1.1:9000/"
    val inputFile = hdfsPath + args(0)
    val outputFile = hdfsPath + args(1)
    //Build a spark session if it doesn't exist
    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
    //Read the data in csv format, include the headers (to perform sort etc)
    val inputData = spark.read.format("csv").option("header", "true").load(inputFile).cache()
    //Perform sort on these 2 variables - "countryCode" and then "timestamp"
    val sorted = inputData.sort("cca2", "timestamp")
    sorted.select("cca2", "timestamp").show()
    //Write the data back to hdfs as output file
    sorted.write.option("header", "true").csv(outputFile)
    //Stop the spark session
    spark.stop()
  }
}

