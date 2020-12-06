package edu.gatech.cse6250.EDFOps

import breeze.interpolation.{ CubicInterpolator, LinearInterpolator }
import breeze.linalg.linspace
import org.apache.spark.rdd.RDD
import edu.gatech.cse6250.helper.SparkHelper
import java.nio.file.{ Files, Paths }
import java.nio.charset.StandardCharsets
import scala.collection.mutable.ListBuffer
import breeze.linalg.DenseVector
import breeze.numerics.{ abs, log, pow }
import scala.math.{ Pi, cos }
import breeze.signal.{ fourierFreq, fourierTr }

object EDFOps {

  // test val filepath = "./shhs/polysomnography/edfs/shhs1/shhs1-200092.edf"

  def loadEDF(filepath: String): (Array[String], Array[Int], Array[Array[Double]]) = {

    /*
    Function to parse individual EDF files and load them into memory for later conversion or analysis.

    @param filepath: the path to the EDF file
    @return Array(
      Array of channel names,
      Array of sample rates (same size as channel names),
      Array of signal Arrays (1 array per channel)
     )

     */

    // Read in the data. Since we plan to parse the whole EDF we just read it in at once
    val byteArray = Files.readAllBytes(Paths.get(filepath))

    // Split out the data in the first header to human-readable ASCII text
    val header1 = new String(byteArray.slice(0, 256), StandardCharsets.UTF_8)

    // Process relevant information from first header
    val nrecords = header1 // the number of records in the data
      .slice(236, 244) // (usually equal to sleep study duration in seconds)
      .trim
      .toInt

    // val duration = header1                           // the duration of each record. Usually 1 second.
    // .slice(244,252)
    // .trim
    // .toInt

    val nsignals = header1 // the number of individual signals in the EDF
      .slice(252, 256)
      .trim
      .toShort

    // Split out the data in the second header to human-readable ASCII text
    val header2 = new String(byteArray.slice(256, 256 + nsignals * 256), StandardCharsets.UTF_8)

    // Process relevant information from second header
    val channelArray = header2 // array of the channel names. size = nsignals
      .slice(0, 16 * nsignals)
      .split("  +")

    val minDigVal = header2 // the minimum digital value for each signal. The data here
      .slice(104 * nsignals, 112 * nsignals) // are 16-bit signed integers, so the range of physical minimum
      .split("\\s+") // and maximum values is [-32768, 32767]. See
      .map(v => v.toFloat) // https://www.edfplus.info/specs/edfplus.html for more info

    val maxDigVal = header2 // See comment for minDigVal
      .slice(112 * nsignals, 120 * nsignals)
      .split("\\s+")
      .map(v => v.toFloat)

    val minPhysVal = header2 // the sensor range is is typically different from the range
      .slice(120 * nsignals, 128 * nsignals) // afforded by the 16-bit data values. Therefore a rescaling
      .split("\\s+") // operation is needed to convert the bitwise values into
      .map(v => v.toLong) // actual sensor values. the minima and maxima for both physical
    // and digital values are thus passed to a rescale funciton.
    val maxPhysVal = header2
      .slice(128 * nsignals, 136 * nsignals)
      .split("\\s+")
      .map(v => v.toLong)

    val nsamples = header2
      .slice(nsignals * 216, nsignals * 224)
      .split(" +")
      .map(v => v.toShort)

    // Process data stream (separate the data from the headers)
    val dataByteArray = byteArray
      .slice(header1.size + header2.size, byteArray.size)

    // Order the bytes in the correct way to be read
    val bb = java.nio.ByteBuffer.wrap(dataByteArray)
    bb.order(java.nio.ByteOrder.LITTLE_ENDIAN)
    val sensorVals = new Array[Short](dataByteArray.length / 2)
    bb
      .asShortBuffer
      .get(sensorVals)

    sensorVals.map(sV => sV.toLong)

    // create the function to scale the data from analog to digital
    def rescale(m: Long, rmin: Long, rmax: Long, tmin: Float, tmax: Float): Double = {
      /*
      Rescales data from one range to another
      @param m: the value to be rescaled
      @param rmin[Long]: the minimum value of the original range
      @param rmax[Long]: the maximum value of the original range
      @param tmin[Long]: the minimum value of the target range
      @param tmax[Long]: the maximum value of the target range
      @return Double: the new value rescaled to the target range
       */
      (m.toFloat - rmin.toFloat) / (rmax.toFloat - rmin.toFloat) * (tmax - tmin) + tmin
    }

    // Create array to store channel Arrays
    val signalArrayBuffer = new ListBuffer[ListBuffer[Long]]()
    for (i <- 0 to nsignals - 1) {
      signalArrayBuffer += ListBuffer()
    }

    // Go through the data and assign it to the correct array
    var dataCounter = 0 // from 0 to nrecords -1 go through each record (second)
    for (i <- 0 to nrecords - 1) { // and process for each signal the number of samples per
      var signalCounter = 0 // record.
      for (j <- nsamples) {
        for (ns <- 0 to j - 1) {
          signalArrayBuffer(signalCounter) += sensorVals(dataCounter)
          dataCounter += 1
        }
        signalCounter += 1
      }
    }

    // Send the data to an Array of Arrays. Each sub-array i <- 0 to nsignals corresponds to the
    // data in the channel channelArray[i]
    var allSignals = signalArrayBuffer
      .zipWithIndex.map {
        case (arr, i) => arr.map(
          v => rescale(v, minPhysVal(i), maxPhysVal(i), minDigVal(i), maxDigVal(i))).toArray
      }.toArray

    // return the channels and the signalArray out
    (channelArray, nsamples.map(sample => sample.toInt), allSignals)
  }

  def getTukeyTaperedSignalSnip(r: Double = .5, signalSnip: DenseVector[Double]): DenseVector[Double] = {
    /*
    Compute the Tukey tapered cosine window function
    @param r:                           the ratio of window to taper
    @param signalSnip:                  the window of signal to taper
    @return breeze.linalg.DenseVector:  The window function applied to the window.
     */
    def getTukeyVal(x: Double, r: Double): Double = {
      /*
      Tukey mathematics
      @param x: the position in the signal relative to the window size. x=50 means the position of the
                signal at exactly halfway between the start and end of the snip.
      @param r: as in the outer function, the ratio of window to taper.
      @return Double: the tapered value at x.
       */
      if ((x >= 0) && (x < r / 2.0)) {
        (1.0 / 2.0) * (1.0 + cos(2.0 * Pi / r * (x - r / 2.0)))
      } else if ((x >= r / 2.0) && (x < 1.0 - r / 2.0)) {
        (1.0)
      } else {
        (1.0 / 2.0) * (1.0 + cos(2.0 * Pi / r * (x - 1.0 + r / 2.0)))
      }
    }
    val L = signalSnip.size
    val tukeyTaper = DenseVector
      .tabulate[Double](L) {
        i => getTukeyVal(i / L.toDouble, r)
      }
    signalSnip * tukeyTaper
  }

  def computeWelchsPeriodogram(d: Array[Int], nOverlap: Array[Int], chosenSignalArray: Array[String], r: Double,
    channelArray: Array[String], sampleArray: Array[Int], allSignals: Array[Array[Double]],
    resolution: Int = 100): RDD[(String, String)] = {
    /*
    Conpute the periodograms.
    @param d:                 Array of integers that correspond to window size for each signal.
    @param nOverlap:          Array of integers that correspond to the amount of overlap between
                              windows for each signal. It is recommended to use a value close to
                              50% of the value chosen in d.
    @param chosenSignalArray: Array of strings representing the chosen signals to calculate periodograms for.
    @param channelArray:      The original channel array. An output of the loadEDF function.
    @param sampleArray:       The original sample Array. An output of the loadEDF function, though it may be necessary
                              to adjust the sampling rates if the samples were taken at a higher rate than the
                              study recommends. See https://sleepdata.org/datasets/shhs/pages/08-equipment-shhs1.md
                              for more info.
    @param allSignalsArray:   The original signal array processed from the loadEDF function.
    @param resolution:        The amount of periodogram to keep for each signal. Default is 100 points.
    @return RDD:              The RDD of (signal:String, periodogram:String). Note that the periodograms are stored
                              as strings in this RDD.
     */

    val spark = SparkHelper.spark
    val sc = spark.sparkContext

    import spark.implicits._

    val selectedChannels = ListBuffer[Int]()
    for (chan <- chosenSignalArray) {
      selectedChannels += channelArray.indexWhere(_ == chan)
    }
    val channelAnalysisValuesArray = selectedChannels.map(allSignals).toArray

    val periodograms = channelAnalysisValuesArray.zipWithIndex.map {
      case (arr, i) => arr.sliding(d(i), nOverlap(i)).map(
        arr => DenseVector(arr.padTo(d(i), 0.0))).map(
          DV => getTukeyTaperedSignalSnip(r, DV)).map(
            dTV => fourierTr(dTV)).reduce(_ + _).map(
              DV => log(
                pow(
                  abs(DV), 2)))
    }
    val f = periodograms.zipWithIndex.map {
      case (p, i) => fourierFreq(p.size, sampleArray(selectedChannels(i)))
    }

    /*
    // Both the LinearInterpolator and CubicInterpolator functions are very expensive,
    // so we will use a cheaper alternative
    // at the expense of some fidelity.

    val welchsDF = sc.parallelize(chosenSignalArray.zip(f.zip(periodograms)
      .zipWithIndex
      .map{ case ((freq, pgram) ,idx) =>
        val splinterp = CubicInterpolator(freq, pgram);
        DenseVector.tabulate(resolution){
          i => splinterp(i/resolution.toDouble * (sampleArray(selectedChannels(idx)) / 2 ))
        }
      }
      .map(DV => DV.toArray)))
      .toDF
     */

    val welchsRDD = sc.parallelize(chosenSignalArray
      .zip(periodograms
        .zipWithIndex
        .map {
          case (pgram, idx) =>
            val cheapInterpolator = linspace(0, d(idx) / 2 - 1, resolution).map(num => num.toInt).toArray.toIndexedSeq;
            pgram(cheapInterpolator).toArray.mkString(",")
        }))
    welchsRDD
  }
}

// filepath.takeRight(10).slice(0,6)

