package edu.gatech.cse6250.helper

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.spark.sql.{ SQLContext, SparkSession }
import org.apache.spark.{ SparkConf, SparkContext }

/**
 * @author Yu Jing <yjing43@gatech.edu>
 */
object SparkHelper {
  lazy val sparkMasterURL = "local[*]"

  lazy val spark: SparkSession = SparkHelper.createSparkSession(
    appName = "CSE6250: Learning From Sleep Data",
    masterUrl = sparkMasterURL,
    cfg = {
      _.set("spark.executor.memory", "4G")
        .set("spark.driver.memory", "4G")
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .set("spark.kryoserializer.buffer", "24")
      // .set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName)
      // .set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)
    })

  lazy val sc: SparkContext = spark.sparkContext

  lazy val sqlContext: SQLContext = spark.sqlContext

  def hdfs(sc: SparkContext = sc): FileSystem = {
    val hadoopConf: Configuration = sc.hadoopConfiguration
    org.apache.hadoop.fs.FileSystem.get(new java.net.URI("hdfs://localhost:9000"), hadoopConf)
  }

  def createSparkSession(
    appName: String,
    masterUrl: String = sparkMasterURL,
    cfg: SparkConf => SparkConf = { in => in }): SparkSession = {
    val session = SparkSession.builder().config(sparkConf(appName, masterUrl, cfg)).getOrCreate()
    //    val hadoopConfig = session.sparkContext.hadoopConfiguration
    //    hadoopConfig.set("fs.hdfs.impl", classOf[DistributedFileSystem].getName)
    //    hadoopConfig.set("fs.file.impl", classOf[LocalFileSystem].getName)
    session
  }

  def sparkConf(appName: String, masterUrl: String, cfg: SparkConf => SparkConf): SparkConf = {
    cfg(new SparkConf()
      .setAppName(appName)
      .setMaster(masterUrl)
    //      .set("spark.executor.memory", "1G")
    //      .set("spark.driver.memory", "500M")
    )
  }

  def createSparkSession: SparkSession = createSparkSession("CSE6250 Project: Learning From Sleep Data")
}
