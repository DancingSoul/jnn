package main.java.cn.edu.hit.ir.JNN;

import java.util.List;
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
    constant(d, 0d);
  }

  public static void randomize(Tensor d) {
    TensorUtils.randomize(d, (double)(Math.sqrt(6.d) / Math.sqrt(d.d.sumDims())));
  }

  public static void randomize(Tensor d, double scale) {
    // TODO optimize this
    Random rand = new Random();
    for (int i = 0; i < d.d.size(); ++i) {
      d.v.set(i, rand.nextFloat() * scale);
    }
  }

  public static void randomBernoulli(Tensor d, double p, double scale) {
  }
  
  public static void randomizeNormal(double mean, double stddev, Tensor v) {
  }

  public static double accessElement(Tensor d, int index) {
    return d.v.get(index);
  }

  public static double accessElement(Tensor d, Dim index) {
    // return v[index[0], index[1]];
    return d.v.get(index.at(0), index.at(1));
  }

  public static void setElement(Tensor v, int index, double value) {
    v.v.set(index, value);
  }

  public static void setElements(final Tensor d, final List<Double> vec) {
    for (int i = 0; i < vec.size(); i++) {
       d.v.set(i, vec.get(i));
    }
    // memcpy(v.v, &vec[0], sizeof(real) * vec.size());
  }

  public static void copyElements(final Tensor d, final Tensor src) {
    for (int i = 0; i < src.v.getNumElements(); ++i) {
      d.v.set(i, src.v.get(i));
    }
  }
}