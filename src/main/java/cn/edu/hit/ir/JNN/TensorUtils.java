package cn.edu.hit.ir.JNN;

import java.util.List;
import java.util.Random;
import java.util.Vector;

import org.ejml.ops.CommonOps;

public class TensorUtils {
  /**
   * TensorUtils a collection of utility functions for tensor.
   *
   * TODO: currently, uninitialized Tensor is not check. Operating the uninitialized
   * one will raise a nullpointer exception.
   */
  public static void constant(Tensor d, double c) {
    CommonOps.fill(d.v, c);
  }

  public static void zero(Tensor d) {
    constant(d, 0d);
  }

  public static void randomize(Tensor d) {
    TensorUtils.randomize(d, Math.sqrt(6.d) / Math.sqrt(d.d.getSumDimensions()));
  }

  public static void randomize(Tensor d, double scale) {
    // TODO: optimize this, random generator should be obtained from a global
    // random number generator.
    Random rand = new Random();
    for (int i = 0; i < d.d.size(); ++i) {
      d.v.set(i, (rand.nextDouble() - 0.5) * scale * 2);
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
  }

  public static void copyElements(final Tensor tgt, final Tensor src) {
    for (int i = 0; i < src.v.getNumElements(); ++i) {
      tgt.v.set(i, src.v.get(i));
    }
  }
  
  public static double toScalar(Tensor t) {
    assert (t.d.size() == 1);
    return t.v.get(0);
  }
  public static Vector<Double> toVector(Tensor t) {
    Vector<Double> res = new Vector<Double>(t.d.size());
    for (int i = 0; i < t.v.numRows; i++) {
      res.addElement(t.v.get(i, 0));
    }  
    return res;
  }
  
}