package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

public class Tanh extends Node {
  public Tanh(List<Integer> x) {
    super(x);
  }

  public String asString(final Vector<String> argNames) {
    return "";
  }

  public Dim dimForward(final Vector<Dim> xs) {
    return xs.get(0);
  }

  public boolean supportsMultibatch() {
    return true;
  }

  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    for (int i = 0; i < fx.v.size(0); ++i) {
      for (int j = 0; j < fx.v.size(1); ++j) {
        fx.v.putScalar(new int[]{i, j}, Math.tanh(xs.get(0).v.getDouble(i, j)));
      }
    }
  }

  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i_, Tensor dEdxi) {
    for (int i = 0; i < fx.v.size(0); ++i) {
      for (int j = 0; j < fx.v.size(1); ++j) {
        dEdxi.v.putScalar(new int[]{i, j}, dEdf.v.getDouble(i, j) * (1.0 - Math.pow(fx.v.getDouble(i, j), 2))
                + dEdxi.v.getDouble(i, j));
      }
    }
  }
}
