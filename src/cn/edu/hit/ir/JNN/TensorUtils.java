package cn.edu.hit.ir.JNN;

import java.util.Random;
import java.util.Vector;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

/**
 * a collection of utility functions for tensor.
 */
class TensorUtils {
  public static void constant(Tensor d, double c) {
    CommonOps.fill(d.v, c);
  }

  public static void zero(Tensor d) {
	constant(d, 0);
  }

  public static void randomize(Tensor d) {
    TensorUtils.randomize(d, (double)(Math.sqrt(6.f) / Math.sqrt(d.d.sumDims())));
  }

  public static void randomize(Tensor d, double scale) {
    // TODO optimize this
    Random rand = new Random();
    for (int i = 0; i < d.d.size(); ++i) {
      d.v.set(i, rand.nextFloat() * scale);
    }
  }

  public static void randomBernoulli(Tensor v, double p, double scale) {
	   
  }
  
  public static void randomizeNormal(double mean, double stddev, Tensor v) {
	  
	  
  }
  

  public static double accessElement(Tensor v, int index) {
    return v.v.get(index);
  }

  public static double accessElement(Tensor v, Dim index) {
    // return v[index[0], index[1]];
    return v.v.get(index.at(0), index.at(1));
  }

  public static void setElement(Tensor v, int index, float value) {
    v.v.set(index, value);
  }

  public static void setElements(final Tensor v, final Vector<Double> vec) {
	double[] tmp = new double[vec.size()];   
	for (int i = 0; i < vec.size(); i++)
      tmp[i] = vec.get(i);
    v.v = new DenseMatrix64F(vec.size(), 1, true, tmp);
  }
  
  public static void copyElements(final Tensor v, final Tensor v_src) {
	v.v = v_src.v;
  }
}