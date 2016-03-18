package cn.edu.hit.ir.JNN;

import cn.edu.hit.ir.JNN.Tensor;
import java.util.Arrays;
import java.util.Random;

/**
 * a collection of utility functions for tensor.
 */
class TensorUtils {
  public static void constant(Tensor d, float c) {
    Arrays.fill(d.v, c);
  }

  public static void zero(Tensor d) {
    Arrays.fill(d.v, 0.f);
  }

  public static void randomize(Tensor d) {
    TensorUtils.randomize(d, (float)(Math.sqrt(6.f) / Math.sqrt(d.d.sumDims())));
  }

  public static void randomize(Tensor d, float scale) {
    // TODO optimize this
    Random rand = new Random();
    for (int i = 0; i < d.d.size(); ++i) {
      d.v[i] = rand.nextFloat() * scale;
    }
  }

  public static float accessElement(Tensor v, int index) {
    return v.v[index];
  }

  public static float accessElement(Tensor v, Dim index) {
    // return v[index[0], index[1]];
    return 0.f;
  }

  public static void setElement(Tensor v, int index, float value) {
    v.v[index] = value;
  }

  public static void setElement(Tensor v, float[] vec) {
    v.v = Arrays.copyOf(vec, v.d.size());
  }
}