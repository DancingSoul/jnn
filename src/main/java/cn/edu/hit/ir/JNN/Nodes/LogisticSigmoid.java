package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

/**
 * \sigmoid(x_1)
 *
 *  - arity: 1
 */
public class LogisticSigmoid extends Node {
  public LogisticSigmoid(List<Integer> a) {
    super(a);
  }

  @Override
  public String asString(final Vector<String> argNames) {
    return "\\sigma(" + argNames.get(0) + ")";
  }

  @Override
  public Dim dimForward(final Vector<Dim> xs) {
    assert (xs.size() == 1);
    return xs.get(0);
  }

  public final double sigmoid(double x) {
    return 1.0 / (1 + Math.exp(-x));
  }

  @Override
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    assert (xs.size() == 1);

    for (int i = 0; i < fx.v.numRows; ++i) {
      for (int j = 0; j < fx.v.numCols; ++j) {
        fx.v.set(i, j, sigmoid(xs.get(0).v.get(i, j)));
      }
    }
  }

  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i_, Tensor dEdxi) {
    for (int i = 0; i < fx.v.numRows; ++i) {
      for (int j = 0; j < fx.v.numCols; ++j) {
        double y = fx.v.get(i, j);
        dEdxi.v.add(i, j, dEdf.v.get(i, j) * y * (1 - y));
      }
    }
  }

  public Dim dim;
}
