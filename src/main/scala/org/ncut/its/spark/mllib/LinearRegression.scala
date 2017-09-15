package org.ncut.its.spark.mllib

import org.apache.log4j.{ Level, Logger }
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LinearRegressionModel

object LinearRegression {

  def main(args: Array[String]) {
    // 构建Spark对象
    val conf = new SparkConf().setAppName("LinearRegressionWithSGD")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    //读取样本数据，转化为标准格式
    val data_path1 = "data/lpsa.data"
    val data = sc.textFile(data_path1)
    val examples = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()
    val numExamples = examples.count()

    // 新建线性回归模型，并设置训练参数
    val numIterations = 100 //迭代次数
    val stepSize = 1 //步长
    val miniBatchFraction = 1.0
    val model = LinearRegressionWithSGD.train(examples, numIterations, stepSize, miniBatchFraction)//训练样本
    model.weights//权重
    model.intercept//偏置

    // 对样本进行测试
    val prediction = model.predict(examples.map(_.features))//预测值
    val predictionAndLabel = prediction.zip(examples.map(_.label))//(预测值，实际值)
    val print_predict = predictionAndLabel.take(20)
    println("prediction" + "\t" + "label")
    for (i <- 0 to print_predict.length - 1) {
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    }
    // 计算测试误差
    val loss = predictionAndLabel.map {
      case (p, l) =>
        val err = p - l
        err * err
    }.reduce(_ + _)
    val rmse = math.sqrt(loss / numExamples)//均方根误差
    println(s"Test RMSE = $rmse.")

    // 模型保存
    val ModelPath = "/user/huangmeiling/LinearRegressionModel"
    model.save(sc, ModelPath)
    val sameModel = LinearRegressionModel.load(sc, ModelPath)
    sameModel.weights
    sameModel.intercept
  }
}
