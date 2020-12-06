package edu.gatech.cse6250.main

import java.io.File
import edu.gatech.cse6250.EDFOps.EDFOps
import edu.gatech.cse6250.helper.SparkHelper.spark
import org.apache.spark.sql.functions.lit

object Main {

  def main(args: Array[String]): Unit = {
    /*
    Driver program.
     */

    //        val filepath = "./data/shhs1-200092.edf"

    // getListOfFiles adapted from code at:
    //    https://alvinalexander.com/scala/how-to-list-files-in-directory-filter-names-scala/
    def getListOfFiles(dir: String): List[String] = {
      val file = new File(dir)
      file.listFiles.filter(_.isFile)
        .map(_.getPath).toList
    }
    val fileList = getListOfFiles("./data/")
    //    val fileList = getListOfFiles("/home/tim/shhs/polysomnography/edfs/shhs1")
    /*
    Tukey's tapered-cosine & periodogram creation
     */
    // Specify window size
    val d = Array(250, 250, 250)

    // Specify amount of window overlap.
    val nOverlap = Array(125, 125, 125)

    // Select signals to analyze
    val chosenSignalArray = Array("EEG", "ECG", "EMG")

    // Choose ratio of window to taper
    val r = 0.25

    // Specify the resolution of the periodogram. Note that this is per signal,
    // i.e., if resolution = 100, the total feature vector size will be 3 * 100 = 300
    val resolution = 250

    // Enable down-sampling if sample rate is greater than intended
    val maxValPerChannelMap = Array("SaO2", "H.R.", "EEG(sec)", "ECG", "EMG", "EOG(L)", "EOG(R)", "EEG", "SOUND",
      "AIRFLOW", "THOR RES", "ABDO RES", "POSITION", "LIGHT", "NEW AIR", "OX stat").zip(
        Array(1, 1, 125, 125, 125, 50, 50, 125, 10, 10, 10, 10, 1, 1, 10, 1)).toMap

    // number of files to process before write
    val nPar = 50
    var count = 0.0
    fileList.sliding(nPar, nPar)
      .toArray.zipWithIndex
      .foreach {
        case (fileGroup, idx) => // split files into groups of nPar files
          val sqlContext = spark.sqlContext
          import sqlContext.implicits._
          fileGroup.map {
            file => // process each file
              count += 1
              var fileName = file
                .split("/")
                .last
                .split(".edf")(0) // save sleep patient NSRRID to assign to dataframe as id
              println("%d%%: currently working on %s".format((count / (fileList.size.toFloat) * 100).toInt, fileName));
              val (channelArray, rawSampleArray, allSignals) = EDFOps.loadEDF(file) // process EDF file
              val newSampleArray = channelArray
                .zipWithIndex
                .map {
                  case (channel, i) =>
                    maxValPerChannelMap // perform downsampling operation
                      .getOrElse(channel, rawSampleArray(i))
                }
              EDFOps // compute periodograms and transform to feature vector
                .computeWelchsPeriodogram(
                  d,
                  nOverlap, chosenSignalArray, r,
                  channelArray, newSampleArray, allSignals,
                  resolution)
                .toDF("Channel", "value")
                .withColumn("id", lit(fileName))
          }.reduce(_ union _)
            .write // write intermediate Parquet files to disk every nPar files
            .parquet("./target/intermediate/parquet/%s".format(idx)) // FOR TESTING/DEMO
        //            .parquet("./intermediate/parquet/%s".format(idx)) // FOR MAIN DATA
      }

    val finalDF = spark // process intermediate files
      .read
      .parquet("./target/intermediate/parquet/*") // FOR TESTING/DEMO
    //      .parquet("./intermediate/parquet/*")                // FOR TESTING/DEMO

    println("output dataframe has %s rows".format(finalDF.count))
    //
    finalDF
      .show

    finalDF // save final file as single .csv for later processing
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      //      .save("./output/Periodograms.csv")
      .save("./output/Periodograms.csv")

    println("all done!")
  }
}
