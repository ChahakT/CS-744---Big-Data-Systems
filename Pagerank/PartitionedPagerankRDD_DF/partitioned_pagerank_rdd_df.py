# Group 23 Assignment 0 Pagerank Code (Partitioned, DF)
# Date : 28th September 2021

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, split, explode, collect_list, col, size


APP_NAME = 'PartitionedPageRankDFG23'


def get_spark_session(spark_master_url: str) -> SparkSession:
    """
    Returns a spark session
    """
    return SparkSession.builder \
        .appName(APP_NAME) \
        .master(spark_master_url) \
        .config("spark.driver.memory", "30G") \
        .config("spark.executor.memory", "30G") \
        .config("spark.executor.cores", 5) \
        .config("spark.driver.cores", 2) \
        .config("spark.task.cpus", 1) \
        .getOrCreate()


def category_filter(url):
    """
    Filter out urls that have ':' in them unless they begin with category
    """
    return (url.startswith('category:') | ~url.contains(':'))


def get_links_and_ranks_dataframes(spark_session: SparkSession, input_file: str):
    """
    Build initial dataframes:
    links_df [FromNodes, ToNodes]
    Link0 -> [ToNode0 .... ToNoden]
    ...
    Linkn -> [ToNode0 .... ToNoden]

    ranks_df
    Link0 -> 1.0
    ...
    Linkn -> 1.0
    """
    data_df = spark_session.read.text(input_file)

    # Split input file on tab (\t)
    split_data_df = \
        data_df\
        .withColumn("FromNodes", split(data_df['value'], '\\t+').getItem(0))\
        .withColumn("ToNodes", split(data_df['value'], '\\t+').getItem(1))\
        .drop('value')

    # Filter split columns
    split_data_df = \
        split_data_df.filter(
            category_filter(split_data_df['FromNodes'])
            | category_filter(split_data_df['ToNodes']))

    # Aggregate all destination urls from source urls
    links_df = \
        split_data_df\
        .groupBy('FromNodes')\
        .agg(collect_list("ToNodes").alias('ToNodes')).coalesce(100).cache()

    # Assign an initial rank of 1.0
    ranks_df = \
        links_df\
        .select("FromNodes")\
        .withColumn("Rank", lit(1.0))

    return links_df, ranks_df


def run_partitioned_pagerank_iterations(spark_session: SparkSession, input_file: str, num_iterations: int = 10):
    """
    Runs PageRank iterations
    """
    links_df, ranks_df = get_links_and_ranks_dataframes(spark_session, input_file)

    for _ in range(num_iterations):
        # Join the links dataframe with the ranks dataframe from the previous iteration.
        contribs_df1 = links_df.join(ranks_df, links_df['FromNodes'] == ranks_df['FromNodes'], 'leftouter').na.fill(1.0)

        # Adjust rank as (prior rank/number of destination nodes)
        contribs_df = contribs_df1.withColumn("NewContrib", col("Rank") / size('ToNodes'))

        # Obtain sum of rank contributions from all
        contribs = \
            contribs_df \
                .select(explode(contribs_df.ToNodes).alias('FromNodes'), contribs_df.NewContrib.alias('Rank')) \
                .groupBy('FromNodes') \
                .agg({'Rank': 'sum'})

        # Adjust rank as (0.85*contributions) + 0.15
        ranks_df = contribs.withColumn("Rank", col("sum(Rank)") * 0.85 + 0.15).coalesce(100)

    return ranks_df


def run(spark_master:str, input_file: str, output_dir: str, num_iterations: int = 10):
    spark_session = get_spark_session(spark_master)
    final_ranks = run_partitioned_pagerank_iterations(spark_session, input_file, num_iterations)
    final_ranks.rdd.saveAsTextFile(output_dir)

    spark_session.stop()


if __name__ == "__main__":
    spark_master = os.environ.get('SPARK_MASTER')
    input_file = os.environ.get('INPUT_FILE')
    output_dir = os.environ.get('OUTPUT_DIR')

    if not (spark_master or input_file or output_dir):
        raise ValueError('spark_master, input_file and output_dir must be provided')

    run(spark_master=spark_master,
        input_file=input_file,
        output_dir=output_dir,
        num_iterations=10)