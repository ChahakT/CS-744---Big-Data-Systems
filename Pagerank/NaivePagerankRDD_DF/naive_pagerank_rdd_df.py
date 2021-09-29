import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

APP_NAME = 'NaivePageRankRDDDFG23'


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


def category_filter(url: str):
    """
    Filter out urls that have ':' in them unless they begin with category
    """
    return (url.startswith('category:') | ~(':' in url))


def compute_contributions(urls, rank: float):
    """Calculates URL contributions to the rank of other URLs."""
    num_urls = len(urls)
    for url in urls:
        yield url, rank / num_urls


def parse_neighbours(url: str):
    """Parses a urls pair string into urls pair."""
    return url.lower().split('\t')


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
    link_columns = ["fromNode", "toNode"]
    rank_columns = ["fromNode", "rank"]

    lines = \
        spark_session.read.text(input_file).rdd\
        .map(lambda r: r[0])\
        .filter(lambda x: not x.startswith('#'))

    filtered_lines = lines \
        .map(lambda url: url.lower().split('\t')) \
        .filter(lambda x: len(x) == 2 & category_filter(x[0]) & category_filter(x[1]))

    links = filtered_lines.\
        groupByKey()\
        .mapValues(list)
    links_df = links.toDF(link_columns)

    ranks = links.map(lambda _: (_[0], 1.0))
    ranks_df = ranks.toDF(rank_columns)

    return links_df, ranks_df


def run_naive_pagerank_iterations(spark_session: SparkSession, input_file: str, num_iterations: int = 10):
    """
        Runs PageRank iterations
        """
    links_df, ranks_df = get_links_and_ranks_dataframes(spark_session, input_file)
    rank_columns = ["fromNode", "rank"]

    for iteration in range(num_iterations):
        # Join the links dataframe with the ranks dataframe from the previous iteration.
        # Adjust rank as (prior rank/number of destination nodes)
        # And then obtain sum of rank contributions from all
        contribs_df = \
            links_df\
            .join(ranks_df, links_df['fromNode'] == ranks_df['fromNode'], "leftouter").na.fill(1.0)\
            .rdd\
            .flatMap(lambda url_urls_rank: compute_contributions(url_urls_rank[1], url_urls_rank[3]))\
            .reduceByKey(lambda a,b: a+b)\
            .toDF(rank_columns)

        ranks_df = contribs_df.withColumn("rank", ((col("rank") * 0.85) + 0.15))

    return ranks_df


def run(spark_master:str, input_file: str, output_dir: str, num_iterations: int = 10):
    spark_session = get_spark_session(spark_master)
    final_ranks = run_naive_pagerank_iterations(spark_session, input_file, num_iterations)
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
