package org.ncut.its.spark.mllib

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils

object logistic_regression {
  def main(args: Array[String]) {
    //1 构建Spark对象
    val conf = new SparkConf().setAppName("logistic_regression")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // 读取样本数据1，格式为LIBSVM format
    val data = MLUtils.loadLibSVMFile(sc, "hdfs://192.168.180.79:9000/user/huangmeiling/sample_libsvm_data.txt")

    //样本数据划分训练样本与测试样本
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    //训练样本集
    val training = splits(0).cache()
    //测试样本集
    val test = splits(1)

    //新建逻辑回归模型，并训练
    val numIterations = 100
    val stepSize = 1
    //随机抽样比例
    val miniBatchFraction = 0.5
    //训练模型
    val model = LogisticRegressionWithSGD.train(training, numIterations, stepSize, miniBatchFraction)
    //    val model = new LogisticRegressionWithLBFGS().
    //      setNumClasses(10).
    //      run(training)
    model.weights
    model.intercept

    //对测试样本进行测试
    val predictionAndLabels = test.map {
      //满足格式才会去处理
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        //预测值，实际值
        (prediction, label)
    }
    val print_predict = predictionAndLabels.take(20)
    println("prediction" + "\t" + "label")
    for (i <- 0 to print_predict.length - 1) {
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    }

    // 误差计算
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    println("Precision = " + precision)

    //保存模型
    val ModelPath = "/user/huangmeiling/logistic_regression_model"
    model.save(sc, ModelPath)
    val sameModel = LogisticRegressionModel.load(sc, ModelPath)
    //对比，查看是否是保存的模型
    sameModel.weights
    sameModel.intercept
  }

}
