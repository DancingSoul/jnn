package cn.edu.hit.ir.JNN;

import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Random;
import java.util.Vector;

public class TensorUtils {
  /**
   * TensorUtils a collection of utility functions for tensor.
   *
   * TODO: currently, uninitialized Tensor is not check. Operating the uninitialized
   * one will raise a nullpointer exception.
   */
  public static void constant(Tensor d, double c) {
    d.v.assign(c);
  }

  public static void zero(Tensor d) {
    d.v = Nd4j.zeros(d.v.shape());
    //constant(d, 0d);
  }

  public static void randomize(Tensor d) {
    TensorUtils.randomize(d, Math.sqrt(6.d) / Math.sqrt(d.d.getSumDimensions()));
  }

  public static void randomize(Tensor d, double scale) {
    // TODO: optimize this, random generator should be obtained from a global
    // random number generator.
    DefaultRandom rand = RandomEngine.getInstance().rnd;
    d.v = Nd4j.rand(d.v.shape(), -scale, scale, rand);
  }

  public static void randomBernoulli(Tensor d, double p, double scale) {
    DefaultRandom rand = RandomEngine.getInstance().rnd;
    for (int i = 0; i < d.v.length(); ++i) {
      d.v.putScalar(i, rand.nextDouble() <= p ? scale : 0);
    }
  }
  
  public static void randomizeNormal(double mean, double stddev, Tensor v) {
  }

  public static double accessElement(Tensor d, int index) {
    return d.v.getDouble(index);
  }

  public static double accessElement(Tensor d, Dim index) {
    // return v[index[0], index[1]];
    return d.v.getDouble(index.at(0), index.at(1));
  }

  public static void setElement(Tensor v, int index, double value) {
    v.v.putScalar(index, value);
  }

  public static void setElements(final Tensor d, final List<Double> vec) {
    for (int i = 0; i < vec.size(); i++) {
       d.v.putScalar(i, vec.get(i));
    }
  }

  public static void copyElements(final Tensor tgt, final Tensor src) {
    for (int i = 0; i < src.v.length(); ++i) {
      tgt.v.putScalar(i, src.v.getDouble(i));
    }
  }
  
  public static double toScalar(Tensor t) {
    assert (t.d.size() == 1);
    return t.v.getDouble(0);
  }
  public static Vector<Double> toVector(Tensor t) {
    Vector<Double> res = new Vector<Double>(t.d.size());
    for (int i = 0; i < t.v.size(0); i++) {
      res.addElement(t.v.getDouble(i, 0));
    }  
    return res;
  }
  
}