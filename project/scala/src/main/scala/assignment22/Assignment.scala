package assignment22

import javassist.runtime.Desc
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{MinMaxScaler, VectorAssembler}
import org.apache.spark.sql.functions.desc
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.functions.{col, when}

class Assignment {

  val spark: SparkSession = SparkSession.builder()
    .appName("Assignment")
    .config("spark.driver.host", "localhost")
    .master("local")
    .getOrCreate()

  // the data frame to be used in tasks 1 and 4
  val dataD2: DataFrame = spark
    .read
    .option("inferSchema","true")
    .option("header", "true")
    .csv("data/dataD2.csv")

  // the data frame to be used in task 2
  val dataD3: DataFrame = spark
    .read
    .option("inferSchema","true")
    .option("header", "true")
    .csv("data/dataD3.csv")

  // the data frame to be used in task 3 (based on dataD2 but containing numeric labels)
  val dataD2WithLabels: DataFrame = dataD2
      .select(col("a"), col("b"), when(col("LABEL")==="Fatal", 1)
      .otherwise(0).as("Label"))


  /*
  The task is to implement k-means clustering with Apache Spark for two-dimensional data. Example
data can be found from file data/dataD2.csv. The task is to compute cluster means using DataFrames and MLlib.
The number of means (k) is given as a parameter. Data for k-means algorithm should be scaled.
   */
  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b"))
      .setOutputCol("features")
    val transformedData = vectorAssembler.transform(df)

    // Let's scale the data
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    val scaledModel = scaler.fit(transformedData)
    val scaledData = scaledModel.transform(transformedData)

    // Clustering
    val kmeans = new KMeans()
      .setK(k).setSeed(1L).setFeaturesCol("scaledFeatures")
    val kmModel = kmeans.fit(scaledData)
    val clusters = kmModel.clusterCenters

    val array_of_douples = clusters.map(x => (x(0), x(1)))
    array_of_douples
  }

  /*
  The task is to implement k-means clustering with Apache Spark for three-dimensional data. Example
data can be found in file data/dataD3.csv. The task is to compute cluster means with DataFrames and MLlib.
The number of means (k) is given as a parameter.
   */
  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b", "c"))
      .setOutputCol("features")
    val transformedData = vectorAssembler.transform(df)

    // Let's scale the data
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    val scaledModel = scaler.fit(transformedData)
    val scaledData = scaledModel.transform(transformedData)

    // Clustering
    val kmeans = new KMeans()
      .setK(k).setFeaturesCol("scaledFeatures")
    val kmModel = kmeans.fit(scaledData)
    val clusters = kmModel.clusterCenters

    val array_of_douples = clusters.map(x => (x(0), x(1), x(2)))
    array_of_douples
  }

  /*
  Use two-dimensional data (like in the file data/dataD2.csv), map the LABEL column to a numeric
scale, and store the resulting data frame to dataD2WithLabels variable. And then, cluster in three
dimensions (including columns a, b, and the numeric value of LABEL) and return two-dimensional
clusters means (the values corresponding to columns a and b) for those two clusters that have the
largest count of Fatal data points.
   */
  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b", "Label"))
      .setOutputCol("features")
    val transformedData = vectorAssembler.transform(df)

    // Let's scale the data
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    val scaledModel = scaler.fit(transformedData)
    val scaledData = scaledModel.transform(transformedData)

    // Clustering
    val kmeans = new KMeans()
      .setK(k).setSeed(1L).setFeaturesCol("scaledFeatures")
    val kmModel = kmeans.fit(scaledData)
    val clusters = kmModel.clusterCenters

    // Let's find the two largest clusters and store their indexes in variables
    val largest_clusters = kmModel.summary.predictions.groupBy("prediction")
      .sum("Label").sort(desc("sum(Label)")).take(2)
    val first = largest_clusters(0)(0).asInstanceOf[Int]
    val second = largest_clusters(1)(0).asInstanceOf[Int]

    // Let's filter out all other clusters except for the two largest ones
    val array_of_douples = clusters
                            .filter(x => clusters.indexOf(x) == first || clusters.indexOf(x) == second)
                            .map(x => (x(0), x(1)))
    array_of_douples
  }

  /*
  The silhouette method can be used to find the optimal number of clusters in the data. Implement a
function which returns an array of (k, score) pairs, where k is the number of clusters, and score is
the silhouette score for the clustering. You can assume that the data is given in the same format as
the data for Basic task 1, i.e., two-dimensional data with columns a and b. Parameter low is the lowest
k and high is the highest one.
   */
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b"))
      .setOutputCol("features")
    val transformedData = vectorAssembler.transform(df)

    // Let's scale the data
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    val scaledModel = scaler.fit(transformedData)
    val scaledData = scaledModel.transform(transformedData)

    // Count silhouette score for all clusters
    val sil_score = Range.inclusive(low, high).map(k =>{
      val kmeans = new KMeans()
        .setK(k).setSeed(1L).setFeaturesCol("scaledFeatures")
      val kmModel = kmeans.fit(scaledData)

      // Make predictions
      val predictions = kmModel.transform(scaledData)
      // Evaluate clustering by computing Silhouette score
      val evaluator = new ClusteringEvaluator().setFeaturesCol("scaledFeatures")
      val silhouetteScore = evaluator.evaluate(predictions)
      (k, silhouetteScore)
    })
    // Array of (k, score) pairs
    sil_score.toArray
  }
}
