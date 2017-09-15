package util

import java.util.Random
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
  cos => Bcos,
  tanh => Btanh
}
import scala.math.Pi

object RandSampleData extends Serializable {
  // Rosenbrock:
  //��(100*(x(i+1)-x(i) 2) 2 + (x(i)-1) 2)
  // Rastrigin:
  //��(x(i) 2 -10*cos(2*3.14*x(i))+10)
  // Sphere :
  //��(x(i) 2)
  /**
   * ���Ժ���: Rosenbrock, Rastrigin
   * �������n2ά���ݣ������ݲ��Ժ�������Y
   * n1 �У�n2 �У�b1 ���ޣ�b2 ���ޣ�function ���㺯��
   */
  def RandM(
    n1: Int,
    n2: Int,
    b1: Double,
    b2: Double,
    function: String): BDM[Double] = {
    //    val n1 = 2
    //    val n2 = 3
    //    val b1 = -30
    //    val b2 = 30
    val bdm1 = BDM.rand(n1, n2) * (b2 - b1).toDouble + b1.toDouble
    val bdm_y = function match {
      case "rosenbrock" =>
        val xi0 = bdm1(::, 0 to (bdm1.cols - 2))
        val xi1 = bdm1(::, 1 to (bdm1.cols - 1))
        val xi2 = (xi0 :* xi0)
        val m1 = ((xi1 - xi2) :* (xi1 - xi2)) * 100.0 + ((xi0 - 1.0) :* (xi0 - 1.0))
        val m2 = m1 * BDM.ones[Double](m1.cols, 1)
        m2
      case "rastrigin" =>
        val xi0 = bdm1
        val xi2 = (xi0 :* xi0)
        val sicos = Bcos(xi0 * 2.0 * Pi) * 10.0
        val m1 = xi2 - sicos + 10.0
        val m2 = m1 * BDM.ones[Double](m1.cols, 1)
        m2
      case "sphere" =>
        val xi0 = bdm1
        val xi2 = (xi0 :* xi0)
        val m1 = xi2
        val m2 = m1 * BDM.ones[Double](m1.cols, 1)
        m2
    }
    val randm = BDM.horzcat(bdm_y, bdm1)
    randm
  }
}
