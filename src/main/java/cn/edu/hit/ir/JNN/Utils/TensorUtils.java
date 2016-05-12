package cn.edu.hit.ir.JNN.Utils;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.RandomEngine;
import cn.edu.hit.ir.JNN.Tensor;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
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
    //d.v = Nd4j.zeros(d.v.shape());
    constant(d, 0d); //I think assign is better than create new class.
  }

  public static void randomize(Tensor d) {
    TensorUtils.randomize(d, Math.sqrt(6.d) / Math.sqrt(d.d.getSumDimensions()));
  }

  public static void randomize(Tensor d, double scale) {
    // TODO: optimize this, random generator should be obtained from a global
    // random number generator.
    DefaultRandom rand = RandomEngine.getInstance().rnd;
    d.v = Nd4j.rand(d.v.shape(), -scale, scale, rand);
    d.v.setOrder('f');
  }

  public static void randomBernoulli(Tensor d, double p, double scale) {
    DefaultRandom rand = RandomEngine.getInstance().rnd;
    for (int i = 0; i < d.v.length(); ++i) {
      d.v.putScalar(i, rand.nextDouble() <= p ? scale : 0);
    }
  }
  
  public static void randomizeNormal(double mean, double stddev, Tensor d) {
    DefaultRandom rand = RandomEngine.getInstance().rnd;
    for (int i = 0; i < d.v.length(); i++) {
      d.v.putScalar(i, mean + rand.nextGaussian() * stddev);
    }
  }

  public static double accessElement(Tensor d, int index) {
    return d.v.getDouble(index);
  }

  public static double accessElement(Tensor d, Dim index) {
    // return v[index[0], index[1]];
    // return d.v.getDouble(index.at(0) * index.at(1) + index.at(1));
    //TODO
    return 0.0;
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
    tgt.v.assign(src.v);
  }
  
  public static double toScalar(Tensor t) {
    assert (t.d.size() == 1);
    return t.v.getDouble(0);
  }
  public static Vector<Double> toVector(Tensor t) {
    Vector<Double> res = new Vector<Double>(t.d.size());
    for (int i = 0; i < t.v.length(); i++) {
      res.addElement(t.v.getDouble(i));
    }
    return res;
  }
  
}