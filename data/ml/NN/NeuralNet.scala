package NN

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg._

import breeze.linalg.{
  Matrix => BM,
  CSCMatrix => BSM,
  DenseMatrix => BDM,
  Vector => BV,
  DenseVector => BDV,
  SparseVector => BSV,
  axpy => brzAxpy,
  svd => brzSvd
}
import breeze.numerics.{
  exp => Bexp,
  tanh => Btanh
}

import scala.collection.mutable.ArrayBuffer
import java.util.Random
import scala.math._

/**
 * label��Ŀ�����
 * nna��������ÿ��ڵ�����ֵ,a(0),a(1),a(2)
 * error���������Ŀ��ֵ��������
 */
case class NNLabel(label: BDM[Double], nna: ArrayBuffer[BDM[Double]], error: BDM[Double]) extends Serializable

/**
 * ���ò���
 */
case class NNConfig(
  size: Array[Int],
  layer: Int,
  activation_function: String,
  learningRate: Double,
  momentum: Double,
  scaling_learningRate: Double,
  weightPenaltyL2: Double,
  nonSparsityPenalty: Double,
  sparsityTarget: Double,
  inputZeroMaskedFraction: Double,
  dropoutFraction: Double,
  testing: Double,
  output_function: String) extends Serializable

/**
 * NN(neural network)
 */

class NeuralNet(
  private var size: Array[Int],
  private var layer: Int,
  private var activation_function: String,
  private var learningRate: Double,
  private var momentum: Double,
  private var scaling_learningRate: Double,
  private var weightPenaltyL2: Double,
  private var nonSparsityPenalty: Double,
  private var sparsityTarget: Double,
  private var inputZeroMaskedFraction: Double,
  private var dropoutFraction: Double,
  private var testing: Double,
  private var output_function: String,
  private var initW: Array[BDM[Double]]) extends Serializable with Logging {
  //            var size=Array(5, 10, 7, 1)
  //            var layer=4
  //            var activation_function="tanh_opt"
  //            var learningRate=2.0
  //            var momentum=0.5
  //            var scaling_learningRate=1.0
  //            var weightPenaltyL2=0.0
  //            var nonSparsityPenalty=0.0
  //            var sparsityTarget=0.05
  //            var inputZeroMaskedFraction=0.0
  //            var dropoutFraction=0.0
  //            var testing=0.0
  //            var output_function="sigm"
  /**
   * size = architecture;
   * n = numel(nn.size);
   * activation_function = sigm   �����㺯��Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
   * learningRate = 2;            ѧϰ��learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
   * momentum = 0.5;              Momentum
   * scaling_learningRate = 1;    Scaling factor for the learning rate (each epoch)
   * weightPenaltyL2  = 0;        ����L2 regularization
   * nonSparsityPenalty = 0;      Ȩ��ϡ��ȳͷ�ֵon sparsity penalty
   * sparsityTarget = 0.05;       Sparsity target
   * inputZeroMaskedFraction = 0; ����noise,Used for Denoising AutoEncoders
   * dropoutFraction = 0;         ÿһ��mini-batch��������ѵ��ʱ������ӵ�x%��������ڵ�Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
   * testing = 0;                 Internal variable. nntest sets this to one.
   * output = 'sigm';             �������output unit 'sigm' (=logistic), 'softmax' and 'linear'   *
   */
  def this() = this(NeuralNet.Architecture, 3, NeuralNet.Activation_Function, 2.0, 0.5, 1.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, NeuralNet.Output, Array(BDM.zeros[Double](1, 1)))

  /** ����������ṹ. Default: [10, 5, 1]. */
  def setSize(size: Array[Int]): this.type = {
    this.size = size
    this
  }

  /** ���������������. Default: 3. */
  def setLayer(layer: Int): this.type = {
    this.layer = layer
    this
  }

  /** ���������㺯��. Default: sigm. */
  def setActivation_function(activation_function: String): this.type = {
    this.activation_function = activation_function
    this
  }

  /** ����ѧϰ������. Default: 2. */
  def setLearningRate(learningRate: Double): this.type = {
    this.learningRate = learningRate
    this
  }

  /** ����Momentum. Default: 0.5. */
  def setMomentum(momentum: Double): this.type = {
    this.momentum = momentum
    this
  }

  /** ����scaling_learningRate. Default: 1. */
  def setScaling_learningRate(scaling_learningRate: Double): this.type = {
    this.scaling_learningRate = scaling_learningRate
    this
  }

  /** ��������L2����. Default: 0. */
  def setWeightPenaltyL2(weightPenaltyL2: Double): this.type = {
    this.weightPenaltyL2 = weightPenaltyL2
    this
  }

  /** ����Ȩ��ϡ��ȳͷ�����. Default: 0. */
  def setNonSparsityPenalty(nonSparsityPenalty: Double): this.type = {
    this.nonSparsityPenalty = nonSparsityPenalty
    this
  }

  /** ����Ȩ��ϡ���Ŀ��ֵ. Default: 0.05. */
  def setSparsityTarget(sparsityTarget: Double): this.type = {
    this.sparsityTarget = sparsityTarget
    this
  }

  /** ����Ȩ�ؼ�����������. Default: 0. */
  def setInputZeroMaskedFraction(inputZeroMaskedFraction: Double): this.type = {
    this.inputZeroMaskedFraction = inputZeroMaskedFraction
    this
  }

  /** ����Ȩ��Dropout����. Default: 0. */
  def setDropoutFraction(dropoutFraction: Double): this.type = {
    this.dropoutFraction = dropoutFraction
    this
  }

  /** ����testing. Default: 0. */
  def setTesting(testing: Double): this.type = {
    this.testing = testing
    this
  }

  /** �����������. Default: linear. */
  def setOutput_function(output_function: String): this.type = {
    this.output_function = output_function
    this
  }

  /** ���ó�ʼȨ��. Default: 0. */
  def setInitW(initW: Array[BDM[Double]]): this.type = {
    this.initW = initW
    this
  }

  /**
   * �����������㷨.
   */
  def NNtrain(train_d: RDD[(BDM[Double], BDM[Double])], opts: Array[Double]): NeuralNetModel = {
    val sc = train_d.sparkContext
    var initStartTime = System.currentTimeMillis()
    var initEndTime = System.currentTimeMillis()
    // �������� �㲥����
    var nnconfig = NNConfig(size, layer, activation_function, learningRate, momentum, scaling_learningRate,
      weightPenaltyL2, nonSparsityPenalty, sparsityTarget, inputZeroMaskedFraction, dropoutFraction, testing,
      output_function)
    // ��ʼ��Ȩ��
    var nn_W = NeuralNet.InitialWeight(size)
    if (!((initW.length == 1) && (initW(0) == (BDM.zeros[Double](1, 1))))) {
      for (i <- 0 to initW.length - 1) {
        nn_W(i) = initW(i)
      }
    }
    var nn_vW = NeuralNet.InitialWeightV(size)
    //    val tmpw = nn_W(0)
    //    for (i <- 0 to tmpw.rows - 1) {
    //      for (j <- 0 to tmpw.cols - 1) {
    //        print(tmpw(i, j) + "\t")
    //      }
    //      println()
    //    }

    // ��ʼ��ÿ���ƽ�������nn.p
    // average activations (for use with sparsity)
    var nn_p = NeuralNet.InitialActiveP(size)

    // �������ݻ��֣�ѵ�����ݡ������������
    val validation = opts(2)
    val splitW1 = Array(1.0 - validation, validation)
    val train_split1 = train_d.randomSplit(splitW1, System.nanoTime())
    val train_t = train_split1(0)
    val train_v = train_split1(1)

    // m:ѵ������������
    val m = train_t.count
    // batchsize����batch gradientʱ��Ĵ�С 
    // ����batch������
    val batchsize = opts(0).toInt
    val numepochs = opts(1).toInt
    val numbatches = (m / batchsize).toInt
    var L = Array.fill(numepochs * numbatches.toInt)(0.0)
    var n = 0
    var loss_train_e = Array.fill(numepochs)(0.0)
    var loss_val_e = Array.fill(numepochs)(0.0)
    // numepochs��ѭ���Ĵ��� 
    for (i <- 1 to numepochs) {
      initStartTime = System.currentTimeMillis()
      val splitW2 = Array.fill(numbatches)(1.0 / numbatches)
      // ���ݷ���Ȩ�أ��������ÿ����������  
      val bc_config = sc.broadcast(nnconfig)
      for (l <- 1 to numbatches) {
        // Ȩ�� 
        val bc_nn_W = sc.broadcast(nn_W)
        val bc_nn_vW = sc.broadcast(nn_vW)

        //        println(i + "\t" + l)
        //        println("W1")
        //        val tmpw0 = bc_nn_W.value(0)
        //        for (i <- 0 to tmpw0.rows - 1) {
        //          for (j <- 0 to tmpw0.cols - 1) {
        //            print(tmpw0(i, j) + "\t")
        //          }
        //          println()
        //        }
        //        println("W2")
        //        val tmpw1 = bc_nn_W.value(1)
        //        for (i <- 0 to tmpw1.rows - 1) {
        //          for (j <- 0 to tmpw1.cols - 1) {
        //            print(tmpw1(i, j) + "\t")
        //          }
        //          println()
        //        }
        //        println("W3")
        //        val tmpw2 = bc_nn_W.value(2)
        //        for (i <- 0 to tmpw2.rows - 1) {
        //          for (j <- 0 to tmpw2.cols - 1) {
        //            print(tmpw2(i, j) + "\t")
        //          }
        //          println()
        //        }

        // ��������
        val train_split2 = train_t.randomSplit(splitW2, System.nanoTime())
        val batch_xy1 = train_split2(l - 1)
        //        val train_split3 = train_t.filter { f => (f._1 >= batchsize * (l - 1) + 1) && (f._1 <= batchsize * (l)) }
        //        val batch_xy1 = train_split3.map(f => (f._2, f._3))
        // Add noise to input (for use in denoising autoencoder)
        // ����noise������denoising autoencoder��Ҫʹ�õ��Ĳ���  
        // �ⲿ����μ���Extracting and Composing Robust Features with Denoising Autoencoders����ƪ����  
        // �������ķ������ǰ�ѵ�������е�һЩ���ݵ�����Ϊ0��inputZeroMaskedFraction��ʾ�˵����ı���  
        //val randNoise = NeuralNet.RandMatrix(batch_x.numRows.toInt, batch_x.numCols.toInt, inputZeroMaskedFraction)
        val batch_xy2 = if (bc_config.value.inputZeroMaskedFraction != 0) {
          NeuralNet.AddNoise(batch_xy1, bc_config.value.inputZeroMaskedFraction)
        } else batch_xy1

        //        val tmpxy = batch_xy2.map(f => (f._1.toArray,f._2.toArray)).toArray.map {f => ((new ArrayBuffer() ++ f._1) ++ f._2).toArray}
        //        for (i <- 0 to tmpxy.length - 1) {
        //          for (j <- 0 to tmpxy(i).length - 1) {
        //            print(tmpxy(i)(j) + "\t")
        //          }
        //          println()
        //        }

        // NNff�ǽ���ǰ�򴫲�
        // nn = nnff(nn, batch_x, batch_y);
        val train_nnff = NeuralNet.NNff(batch_xy2, bc_config, bc_nn_W)

        //        val tmpa0 = train_nnff.map(f => f._1.nna(0)).take(20)
        //        println("tmpa0")
        //        for (i <- 0 to 10) {
        //          for (j <- 0 to tmpa0(i).cols - 1) {
        //            print(tmpa0(i)(0, j) + "\t")
        //          }
        //          println()
        //        }
        //        val tmpa1 = train_nnff.map(f => f._1.nna(1)).take(20)
        //        println("tmpa1")
        //        for (i <- 0 to 10) {
        //          for (j <- 0 to tmpa1(i).cols - 1) {
        //            print(tmpa1(i)(0, j) + "\t")
        //          }
        //          println()
        //        }
        //        val tmpa2 = train_nnff.map(f => f._1.nna(2)).take(20)
        //        println("tmpa2")
        //        for (i <- 0 to 10) {
        //          for (j <- 0 to tmpa2(i).cols - 1) {
        //            print(tmpa2(i)(0, j) + "\t")
        //          }
        //          println()
        //        }

        // sparsity���㣬����ÿ��ڵ��ƽ��ϡ���
        nn_p = NeuralNet.ActiveP(train_nnff, bc_config, nn_p)
        val bc_nn_p = sc.broadcast(nn_p)

        // NNbp�Ǻ��򴫲�
        // nn = nnbp(nn);
        val train_nnbp = NeuralNet.NNbp(train_nnff, bc_config, bc_nn_W, bc_nn_p)

        //        val tmpd0 = rdd5.map(f => f._2(2)).take(20)
        //        println("tmpd0")
        //        for (i <- 0 to 10) {
        //          for (j <- 0 to tmpd0(i).cols - 1) {
        //            print(tmpd0(i)(0, j) + "\t")
        //          }
        //          println()
        //        }
        //        val tmpd1 = rdd5.map(f => f._2(1)).take(20)
        //        println("tmpd1")
        //        for (i <- 0 to 10) {
        //          for (j <- 0 to tmpd1(i).cols - 1) {
        //            print(tmpd1(i)(0, j) + "\t")
        //          }
        //          println()
        //        }
        //        val tmpdw0 = rdd5.map(f => f._3(0)).take(20)
        //        println("tmpdw0")
        //        for (i <- 0 to 10) {
        //          for (j <- 0 to tmpdw0(i).cols - 1) {
        //            print(tmpdw0(i)(0, j) + "\t")
        //          }
        //          println()
        //        }
        //        val tmpdw1 = rdd5.map(f => f._3(1)).take(20)
        //        println("tmpdw1")
        //        for (i <- 0 to 10) {
        //          for (j <- 0 to tmpdw1(i).cols - 1) {
        //            print(tmpdw1(i)(0, j) + "\t")
        //          }
        //          println()
        //        }

        // nn = NNapplygrads(nn) returns an neural network structure with updated
        // weights and biases
        // ����Ȩ�ز�����w=w-��*[dw + ��w]    
        val train_nnapplygrads = NeuralNet.NNapplygrads(train_nnbp, bc_config, bc_nn_W, bc_nn_vW)
        nn_W = train_nnapplygrads(0)
        nn_vW = train_nnapplygrads(1)

        //        val tmpw2 = train_nnapplygrads(0)(0)
        //        for (i <- 0 to tmpw2.rows - 1) {
        //          for (j <- 0 to tmpw2.cols - 1) {
        //            print(tmpw2(i, j) + "\t")
        //          }
        //          println()
        //        }
        //        val tmpw3 = train_nnapplygrads(0)(1)
        //        for (i <- 0 to tmpw3.rows - 1) {
        //          for (j <- 0 to tmpw3.cols - 1) {
        //            print(tmpw3(i, j) + "\t")
        //          }
        //          println()
        //        }

        // error and loss
        // ���������
        val loss1 = train_nnff.map(f => f._1.error)
        val (loss2, counte) = loss1.treeAggregate((0.0, 0L))(
          seqOp = (c, v) => {
            // c: (e, count), v: (m)
            val e1 = c._1
            val e2 = (v :* v).sum
            val esum = e1 + e2
            (esum, c._2 + 1)
          },
          combOp = (c1, c2) => {
            // c: (e, count)
            val e1 = c1._1
            val e2 = c2._1
            val esum = e1 + e2
            (esum, c1._2 + c2._2)
          })
        val Loss = loss2 / counte.toDouble
        L(n) = Loss * 0.5
        n = n + 1
      }
      // ���㱾�ε�����ѵ��������������
      // Full-batch train mse
      val evalconfig = NNConfig(size, layer, activation_function, learningRate, momentum, scaling_learningRate,
        weightPenaltyL2, nonSparsityPenalty, sparsityTarget, inputZeroMaskedFraction, dropoutFraction, 1.0,
        output_function)
      loss_train_e(i - 1) = NeuralNet.NNeval(train_t, sc.broadcast(evalconfig), sc.broadcast(nn_W))
      if (validation > 0) loss_val_e(i - 1) = NeuralNet.NNeval(train_v, sc.broadcast(evalconfig), sc.broadcast(nn_W))

      // ����ѧϰ����
      // nn.learningRate = nn.learningRate * nn.scaling_learningRate;
      nnconfig = NNConfig(size, layer, activation_function, nnconfig.learningRate * nnconfig.scaling_learningRate, momentum, scaling_learningRate,
        weightPenaltyL2, nonSparsityPenalty, sparsityTarget, inputZeroMaskedFraction, dropoutFraction, testing,
        output_function)
      initEndTime = System.currentTimeMillis()

      // ��ӡ������
      printf("epoch: numepochs = %d , Took = %d seconds; Full-batch train mse = %f, val mse = %f.\n", i, scala.math.ceil((initEndTime - initStartTime).toDouble / 1000).toLong, loss_train_e(i - 1), loss_val_e(i - 1))
    }
    val configok = NNConfig(size, layer, activation_function, learningRate, momentum, scaling_learningRate,
      weightPenaltyL2, nonSparsityPenalty, sparsityTarget, inputZeroMaskedFraction, dropoutFraction, 1.0,
      output_function)
    new NeuralNetModel(configok, nn_W)
  }

}

/**
 * NN(neural network)
 */
object NeuralNet extends Serializable {

  // Initialization mode names
  val Activation_Function = "sigm"
  val Output = "linear"
  val Architecture = Array(10, 5, 1)

  /**
   * �����������
   * �����ֵ>=Fraction��ֵ���䣬�����Ϊ0
   */
  def AddNoise(rdd: RDD[(BDM[Double], BDM[Double])], Fraction: Double): RDD[(BDM[Double], BDM[Double])] = {
    val addNoise = rdd.map { f =>
      val features = f._2
      val a = BDM.rand[Double](features.rows, features.cols)
      val a1 = a :>= Fraction
      val d1 = a1.data.map { f => if (f == true) 1.0 else 0.0 }
      val a2 = new BDM(features.rows, features.cols, d1)
      val features2 = features :* a2
      (f._1, features2)
    }
    addNoise
  }

  /**
   * ��ʼ��Ȩ��
   * ��ʼ��Ϊһ����С�ġ��ӽ�������ֵ
   */
  def InitialWeight(size: Array[Int]): Array[BDM[Double]] = {
    // ��ʼ��Ȩ�ز���
    // weights and weight momentum
    // nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));
    // nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
    val n = size.length
    val nn_W = ArrayBuffer[BDM[Double]]()
    for (i <- 1 to n - 1) {
      val d1 = BDM.rand(size(i), size(i - 1) + 1)
      d1 :-= 0.5
      val f1 = 2 * 4 * sqrt(6.0 / (size(i) + size(i - 1)))
      val d2 = d1 :* f1
      //val d3 = new DenseMatrix(d2.rows, d2.cols, d2.data, d2.isTranspose)
      //val d4 = Matrices.dense(d2.rows, d2.cols, d2.data)
      nn_W += d2
    }
    nn_W.toArray
  }

  /**
   * ��ʼ��Ȩ��vW
   * ��ʼ��Ϊ0
   */
  def InitialWeightV(size: Array[Int]): Array[BDM[Double]] = {
    // ��ʼ��Ȩ�ز���
    // weights and weight momentum
    // nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
    val n = size.length
    val nn_vW = ArrayBuffer[BDM[Double]]()
    for (i <- 1 to n - 1) {
      val d1 = BDM.zeros[Double](size(i), size(i - 1) + 1)
      nn_vW += d1
    }
    nn_vW.toArray
  }

  /**
   * ��ʼÿһ���ƽ�������
   * ��ʼ��Ϊ0
   */
  def InitialActiveP(size: Array[Int]): Array[BDM[Double]] = {
    // ��ʼÿһ���ƽ�������
    // average activations (for use with sparsity)
    // nn.p{i}     = zeros(1, nn.size(i));  
    val n = size.length
    val nn_p = ArrayBuffer[BDM[Double]]()
    nn_p += BDM.zeros[Double](1, 1)
    for (i <- 1 to n - 1) {
      val d1 = BDM.zeros[Double](1, size(i))
      nn_p += d1
    }
    nn_p.toArray
  }

  /**
   * ���������ĳЩ������ڵ��Ȩ�ز�����
   * �����ֵ>=Fraction������ֵ���䣬�����Ϊ0
   */
  def DropoutWeight(matrix: BDM[Double], Fraction: Double): Array[BDM[Double]] = {
    val aa = BDM.rand[Double](matrix.rows, matrix.cols)
    val aa1 = aa :> Fraction
    val d1 = aa1.data.map { f => if (f == true) 1.0 else 0.0 }
    val aa2 = new BDM(matrix.rows: Int, matrix.cols: Int, d1: Array[Double])
    val matrix2 = matrix :* aa2
    Array(aa2, matrix2)
  }

  /**
   * sigm�����
   * X = 1./(1+exp(-P));
   */
  def sigm(matrix: BDM[Double]): BDM[Double] = {
    val s1 = 1.0 / (Bexp(matrix * (-1.0)) + 1.0)
    s1
  }

  /**
   * tanh�����
   * f=1.7159*tanh(2/3.*A);
   */
  def tanh_opt(matrix: BDM[Double]): BDM[Double] = {
    val s1 = Btanh(matrix * (2.0 / 3.0)) * 1.7159
    s1
  }

  /**
   * nnff�ǽ���ǰ�򴫲�
   * �����������е�ÿ���ڵ�����ֵ;
   */
  def NNff(
    batch_xy2: RDD[(BDM[Double], BDM[Double])],
    bc_config: org.apache.spark.broadcast.Broadcast[NNConfig],
    bc_nn_W: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]]): RDD[(NNLabel, Array[BDM[Double]])] = {
    // ��1��:a(1)=[1 x]
    // ����ƫ����b
    val train_data1 = batch_xy2.map { f =>
      val lable = f._1
      val features = f._2
      val nna = ArrayBuffer[BDM[Double]]()
      val Bm1 = new BDM(features.rows, 1, Array.fill(features.rows * 1)(1.0))
      val features2 = BDM.horzcat(Bm1, features)
      val error = BDM.zeros[Double](lable.rows, lable.cols)
      nna += features2
      NNLabel(lable, nna, error)
    }

    //    println("bc_size " + bc_config.value.size(0) + bc_config.value.size(1) + bc_config.value.size(2))
    //    println("bc_layer " + bc_config.value.layer)
    //    println("bc_activation_function " + bc_config.value.activation_function)
    //    println("bc_output_function " + bc_config.value.output_function)
    //
    //    println("tmpw0 ")
    //    val tmpw0 = bc_nn_W.value(0)
    //    for (i <- 0 to tmpw0.rows - 1) {
    //      for (j <- 0 to tmpw0.cols - 1) {
    //        print(tmpw0(i, j) + "\t")
    //      }
    //      println()
    //    }

    // feedforward pass
    // ��2��n-1����㣬a(i)=f(a(i-1)*w(i-1)')
    //val tmp1 = train_data1.map(f => f.nna(0).data).take(1)(0)
    //val tmp2 = new BDM(1, tmp1.length, tmp1)
    //val nn_a = ArrayBuffer[BDM[Double]]()
    //nn_a += tmp2
    val train_data2 = train_data1.map { f =>
      val nn_a = f.nna
      val dropOutMask = ArrayBuffer[BDM[Double]]()
      dropOutMask += new BDM[Double](1, 1, Array(0.0))
      for (j <- 1 to bc_config.value.layer - 2) {
        // ����ÿ�����
        // Calculate the unit's outputs (including the bias term)
        // nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}')
        // nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');            
        val A1 = nn_a(j - 1)
        val W1 = bc_nn_W.value(j - 1)
        val aw1 = A1 * W1.t
        val nnai1 = bc_config.value.activation_function match {
          case "sigm" =>
            val aw2 = NeuralNet.sigm(aw1)
            aw2
          case "tanh_opt" =>
            val aw2 = NeuralNet.tanh_opt(aw1)
            //val aw2 = Btanh(aw1 * (2.0 / 3.0)) * 1.7159
            aw2
        }
        // dropout����
        // Dropout��ָ��ģ��ѵ��ʱ���������ĳЩ������ڵ��Ȩ�ز�����������������Щ�ڵ������ʱ��Ϊ��������ṹ��һ����
        // ��������Ȩ�صñ���������ֻ����ʱ�����¶��ѣ�����Ϊ�´���������ʱ�������ֵù�����
        // ���� http://www.cnblogs.com/tornadomeet/p/3258122.html   
        val dropoutai = if (bc_config.value.dropoutFraction > 0) {
          if (bc_config.value.testing == 1) {
            val nnai2 = nnai1 * (1.0 - bc_config.value.dropoutFraction)
            Array(new BDM[Double](1, 1, Array(0.0)), nnai2)
          } else {
            NeuralNet.DropoutWeight(nnai1, bc_config.value.dropoutFraction)
          }
        } else {
          val nnai2 = nnai1
          Array(new BDM[Double](1, 1, Array(0.0)), nnai2)
        }
        val nnai2 = dropoutai(1)
        dropOutMask += dropoutai(0)
        // Add the bias term
        // ����ƫ����b
        // nn.a{i} = [ones(m,1) nn.a{i}];
        val Bm1 = BDM.ones[Double](nnai2.rows, 1)
        val nnai3 = BDM.horzcat(Bm1, nnai2)
        nn_a += nnai3
      }
      (NNLabel(f.label, nn_a, f.error), dropOutMask.toArray)
    }

    // ��������
    val train_data3 = train_data2.map { f =>
      val nn_a = f._1.nna
      // nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
      // nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';          
      val An1 = nn_a(bc_config.value.layer - 2)
      val Wn1 = bc_nn_W.value(bc_config.value.layer - 2)
      val awn1 = An1 * Wn1.t
      val nnan1 = bc_config.value.output_function match {
        case "sigm" =>
          val awn2 = NeuralNet.sigm(awn1)
          //val awn2 = 1.0 / (Bexp(awn1 * (-1.0)) + 1.0)
          awn2
        case "linear" =>
          val awn2 = awn1
          awn2
      }
      nn_a += nnan1
      (NNLabel(f._1.label, nn_a, f._1.error), f._2)
    }

    // error and loss
    // ���������
    // nn.e = y - nn.a{n};
    // val nn_e = batch_y - nnan
    val train_data4 = train_data3.map { f =>
      val batch_y = f._1.label
      val nnan = f._1.nna(bc_config.value.layer - 1)
      val error = (batch_y - nnan)
      (NNLabel(f._1.label, f._1.nna, error), f._2)
    }
    train_data4
  }

  /**
   * sparsity���㣬����ϡ���
   * ����ÿ���ڵ��ƽ��ֵ
   */
  def ActiveP(
    train_nnff: RDD[(NNLabel, Array[BDM[Double]])],
    bc_config: org.apache.spark.broadcast.Broadcast[NNConfig],
    nn_p_old: Array[BDM[Double]]): Array[BDM[Double]] = {
    val nn_p = ArrayBuffer[BDM[Double]]()
    nn_p += BDM.zeros[Double](1, 1)
    // calculate running exponential activations for use with sparsity
    // sparsity���㣬����sparsity��nonSparsityPenalty �Ƕ�û�ﵽsparsitytarget�Ĳ����ĳͷ�ϵ�� 
    for (i <- 1 to bc_config.value.layer - 1) {
      val pi1 = train_nnff.map(f => f._1.nna(i))
      val initpi = BDM.zeros[Double](1, bc_config.value.size(i))
      val (piSum, miniBatchSize) = pi1.treeAggregate((initpi, 0L))(
        seqOp = (c, v) => {
          // c: (nnasum, count), v: (nna)
          val nna1 = c._1
          val nna2 = v
          val nnasum = nna1 + nna2
          (nnasum, c._2 + 1)
        },
        combOp = (c1, c2) => {
          // c: (nnasum, count)
          val nna1 = c1._1
          val nna2 = c2._1
          val nnasum = nna1 + nna2
          (nnasum, c1._2 + c2._2)
        })
      val piAvg = piSum / miniBatchSize.toDouble
      val oldpi = nn_p_old(i)
      val newpi = (piAvg * 0.01) + (oldpi * 0.09)
      nn_p += newpi
    }
    nn_p.toArray
  }

  /**
   * NNbp�Ǻ��򴫲�
   * ����Ȩ�ص�ƽ��ƫ����
   */
  def NNbp(
    train_nnff: RDD[(NNLabel, Array[BDM[Double]])],
    bc_config: org.apache.spark.broadcast.Broadcast[NNConfig],
    bc_nn_W: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]],
    bc_nn_p: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]]): Array[BDM[Double]] = {
    // ��n��ƫ������d(n)=-(y-a(n))*f'(z)��sigmoid����f'(z)���ʽ:f'(z)=f(z)*[1-f(z)]
    // sigm: d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
    // {'softmax','linear'}: d{n} = - nn.e;
    val train_data5 = train_nnff.map { f =>
      val nn_a = f._1.nna
      val error = f._1.error
      val dn = ArrayBuffer[BDM[Double]]()
      val nndn = bc_config.value.output_function match {
        case "sigm" =>
          val fz = nn_a(bc_config.value.layer - 1)
          (error * (-1.0)) :* (fz :* (1.0 - fz))
        case "linear" =>
          error * (-1.0)
      }
      dn += nndn
      (f._1, f._2, dn)
    }
    // ��n-1����2�㵼����d(n)=-(w(n)*d(n+1))*f'(z) 
    val train_data6 = train_data5.map { f =>
      // ���� f(z) ��sigmoid���� f(z)=1/[1+e^(-z)]��f'(z)���ʽ��f'(z)=f(z)*[1-f(z)]    
      // ���� f(z) tanh f(z)=1.7159*tanh(2/3.*A) ��f'(z)���ʽ��f'(z)=1.7159 * 2/3 * (1 - 1/(1.7159)^2 * f(z).^2)   
      // train_data5.map(f => f._1.nna).take(1)
      // train_data5.map(f => f._3).take(1)
      // train_data5.map(f => f._2).take(1)
      //      val di = ArrayBuffer(BDM((0.011181628780251586)))
      //      val nn_a = ArrayBuffer[BDM[Double]]()
      //      val a1 = BDM((1.0, 0.312605257000000, 0.848582961000000, 0.999014768000000, 0.278330771000000, 0.462701179000000))
      //      val a2 = BDM((1.0, 0.838091550300577, 0.996782915917104, 0.118033012437165, 0.312605257000000, 0.848582961000000, 0.999014768000000, 0.278330771000000, 0.462701179000000, 0.278330771000000, 0.462701179000000))
      //      val a3 = BDM((1.0, 0.312605257000000, 0.848582961000000, 0.999014768000000, 0.278330771000000, 0.462701179000000, 0.278330771000000, 0.462701179000000))
      //      val a4 = BDM((0.9826605123949446))
      //      nn_a += a1
      //      nn_a += a2
      //      nn_a += a3
      //      nn_a += a4
      //      val dropout = Array(BDM.zeros[Double](1,1), BDM.zeros[Double](1,1), BDM.zeros[Double](1,1)) 
      val nn_a = f._1.nna
      val di = f._3
      val dropout = f._2
      for (i <- (bc_config.value.layer - 2) to 1 by -1) {
        // f'(z)���ʽ
        val nnd_act = bc_config.value.activation_function match {
          case "sigm" =>
            val d_act = nn_a(i) :* (1.0 - nn_a(i))
            d_act
          case "tanh_opt" =>
            val fz2 = (1.0 - ((nn_a(i) :* nn_a(i)) * (1.0 / (1.7159 * 1.7159))))
            val d_act = fz2 * (1.7159 * (2.0 / 3.0))
            d_act
        }
        // ϡ��ȳͷ�������:-(t/p)+(1-t)/(1-p)
        // sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
        val sparsityError = if (bc_config.value.nonSparsityPenalty > 0) {
          val nn_pi1 = bc_nn_p.value(i)
          val nn_pi2 = (bc_config.value.sparsityTarget / nn_pi1) * (-1.0) + (1.0 - bc_config.value.sparsityTarget) / (1.0 - nn_pi1)
          val Bm1 = new BDM(nn_pi2.rows, 1, Array.fill(nn_pi2.rows * 1)(1.0))
          val sparsity = BDM.horzcat(Bm1, nn_pi2 * bc_config.value.nonSparsityPenalty)
          sparsity
        } else {
          val nn_pi1 = bc_nn_p.value(i)
          val sparsity = BDM.zeros[Double](nn_pi1.rows, nn_pi1.cols + 1)
          sparsity
        }
        // ������d(n)=-( w(n)*d(n+1)+ sparsityError )*f'(z) 
        // d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act;
        val W1 = bc_nn_W.value(i)
        val nndi1 = if (i + 1 == bc_config.value.layer - 1) {
          //in this case in d{n} there is not the bias term to be removed  
          val di1 = di(bc_config.value.layer - 2 - i)
          val di2 = (di1 * W1 + sparsityError) :* nnd_act
          di2
        } else {
          // in this case in d{i} the bias term has to be removed
          val di1 = di(bc_config.value.layer - 2 - i)(::, 1 to -1)
          val di2 = (di1 * W1 + sparsityError) :* nnd_act
          di2
        }
        // dropoutFraction
        val nndi2 = if (bc_config.value.dropoutFraction > 0) {
          val dropouti1 = dropout(i)
          val Bm1 = new BDM(nndi1.rows: Int, 1: Int, Array.fill(nndi1.rows * 1)(1.0))
          val dropouti2 = BDM.horzcat(Bm1, dropouti1)
          nndi1 :* dropouti2
        } else nndi1
        di += nndi2
      }
      di += BDM.zeros(1, 1)
      // ����������Ҫ��ƫ����ֵ��dw(n)=(1/m)��d(n+1)*a(n)
      //  nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
      val dw = ArrayBuffer[BDM[Double]]()
      for (i <- 0 to bc_config.value.layer - 2) {
        val nndW = if (i + 1 == bc_config.value.layer - 1) {
          (di(bc_config.value.layer - 2 - i).t) * nn_a(i)
        } else {
          (di(bc_config.value.layer - 2 - i)(::, 1 to -1)).t * nn_a(i)
        }
        dw += nndW
      }
      (f._1, di, dw)
    }
    val train_data7 = train_data6.map(f => f._3)

    // Sample a subset (fraction miniBatchFraction) of the total data
    // compute and sum up the subgradients on this subset (this is one map-reduce)
    val initgrad = ArrayBuffer[BDM[Double]]()
    for (i <- 0 to bc_config.value.layer - 2) {
      val init1 = if (i + 1 == bc_config.value.layer - 1) {
        BDM.zeros[Double](bc_config.value.size(i + 1), bc_config.value.size(i) + 1)
      } else {
        BDM.zeros[Double](bc_config.value.size(i + 1), bc_config.value.size(i) + 1)
      }
      initgrad += init1
    }
    val (gradientSum, miniBatchSize) = train_data7.treeAggregate((initgrad, 0L))(
      seqOp = (c, v) => {
        // c: (grad, count), v: (grad)
        val grad1 = c._1
        val grad2 = v
        val sumgrad = ArrayBuffer[BDM[Double]]()
        for (i <- 0 to bc_config.value.layer - 2) {
          val Bm1 = grad1(i)
          val Bm2 = grad2(i)
          val Bmsum = Bm1 + Bm2
          sumgrad += Bmsum
        }
        (sumgrad, c._2 + 1)
      },
      combOp = (c1, c2) => {
        // c: (grad, count)
        val grad1 = c1._1
        val grad2 = c2._1
        val sumgrad = ArrayBuffer[BDM[Double]]()
        for (i <- 0 to bc_config.value.layer - 2) {
          val Bm1 = grad1(i)
          val Bm2 = grad2(i)
          val Bmsum = Bm1 + Bm2
          sumgrad += Bmsum
        }
        (sumgrad, c1._2 + c2._2)
      })
    // ��ƽ��ֵ
    val gradientAvg = ArrayBuffer[BDM[Double]]()
    for (i <- 0 to bc_config.value.layer - 2) {
      val Bm1 = gradientSum(i)
      val Bmavg = Bm1 :/ miniBatchSize.toDouble
      gradientAvg += Bmavg
    }
    gradientAvg.toArray
  }

  /**
   * NNapplygrads��Ȩ�ظ���
   * Ȩ�ظ���
   */
  def NNapplygrads(
    train_nnbp: Array[BDM[Double]],
    bc_config: org.apache.spark.broadcast.Broadcast[NNConfig],
    bc_nn_W: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]],
    bc_nn_vW: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]]): Array[Array[BDM[Double]]] = {
    // nn = nnapplygrads(nn) returns an neural network structure with updated
    // weights and biases
    // ����Ȩ�ز�����w=w-��*[dw + ��w]    
    val W_a = ArrayBuffer[BDM[Double]]()
    val vW_a = ArrayBuffer[BDM[Double]]()
    for (i <- 0 to bc_config.value.layer - 2) {
      val nndwi = if (bc_config.value.weightPenaltyL2 > 0) {
        val dwi = train_nnbp(i)
        val zeros = BDM.zeros[Double](dwi.rows, 1)
        val l2 = BDM.horzcat(zeros, dwi(::, 1 to -1))
        val dwi2 = dwi + (l2 * bc_config.value.weightPenaltyL2)
        dwi2
      } else {
        val dwi = train_nnbp(i)
        dwi
      }
      val nndwi2 = nndwi :* bc_config.value.learningRate
      val nndwi3 = if (bc_config.value.momentum > 0) {
        val vwi = bc_nn_vW.value(i)
        val dw3 = nndwi2 + (vwi * bc_config.value.momentum)
        dw3
      } else {
        nndwi2
      }
      // nn.W{i} = nn.W{i} - dW;
      W_a += (bc_nn_W.value(i) - nndwi3)
      // nn.vW{i} = nn.momentum*nn.vW{i} + dW;
      val nnvwi1 = if (bc_config.value.momentum > 0) {
        val vwi = bc_nn_vW.value(i)
        val vw3 = nndwi2 + (vwi * bc_config.value.momentum)
        vw3
      } else {
        bc_nn_vW.value(i)
      }
      vW_a += nnvwi1
    }
    Array(W_a.toArray, vW_a.toArray)
  }

  /**
   * nneval�ǽ���ǰ�򴫲�������������
   * �����������е�ÿ���ڵ�����ֵ��������ƽ�����;
   */
  def NNeval(
    batch_xy: RDD[(BDM[Double], BDM[Double])],
    bc_config: org.apache.spark.broadcast.Broadcast[NNConfig],
    bc_nn_W: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]]): Double = {
    // NNff�ǽ���ǰ�򴫲�
    // nn = nnff(nn, batch_x, batch_y);
    val train_nnff = NeuralNet.NNff(batch_xy, bc_config, bc_nn_W)
    // error and loss
    // ���������
    val loss1 = train_nnff.map(f => f._1.error)
    val (loss2, counte) = loss1.treeAggregate((0.0, 0L))(
      seqOp = (c, v) => {
        // c: (e, count), v: (m)
        val e1 = c._1
        val e2 = (v :* v).sum
        val esum = e1 + e2
        (esum, c._2 + 1)
      },
      combOp = (c1, c2) => {
        // c: (e, count)
        val e1 = c1._1
        val e2 = c2._1
        val esum = e1 + e2
        (esum, c1._2 + c2._2)
      })
    val Loss = loss2 / counte.toDouble
    Loss * 0.5
  }
}

