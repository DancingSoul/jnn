package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

/**
 * \tanh(x_1)
 *
 *  - arity: 1
 */
public class Tanh extends Node {
  public Tanh(List<Integer> x) {
    super(x);
  }

  @Override
  public String asString(final Vector<String> argNames) {
    return "Tanh(" + argNames.get(0) + ")";
  }

  @Override
  public Dim dimForward(final Vector<Dim> xs) {
    return xs.get(0);
  }

  public boolean supportsMultibatch() {
    return true;
  }

  @Override
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    for (int i = 0; i < fx.v.length(); ++i) {
      fx.v.putScalar(i, Math.tanh(xs.get(0).v.getDouble(i)));
    }
  }

  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i_, Tensor dEdxi) {
    for (int i = 0; i < fx.v.length(); ++i) {
      dEdxi.v.putScalar(i, dEdf.v.getDouble(i) * (1.0 - Math.pow(fx.v.getDouble(i), 2))
              + dEdxi.v.getDouble(i));
    }
  }
}
