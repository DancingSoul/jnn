package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;
import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

/**
 * \ReLU(x_1)
 *
 *  - arity: 1
 */
public class Rectify extends Node {
  public Rectify(List<Integer> a) {
    super(a);
  }

  @Override
  public String asString(final Vector<String> argNames) {
    return "ReLU(" + argNames.get(0) + ")";
  }

  @Override
  public Dim dimForward(final Vector<Dim> xs) {
    assert(xs.size() == 1);
    return xs.get(0);
  }

  @Override
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    for (int i = 0; i < fx.v.numRows; ++i) {
      for (int j = 0; j < fx.v.numCols; ++j) {
        fx.v.set(i, j, Math.max(0, xs.get(0).v.get(i, j)));
      }
    }
  }

  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i_, Tensor dEdxi) {
    for (int i = 0; i < fx.v.numRows; ++i) {
      for (int j = 0; j < fx.v.numCols; ++j) {
        dEdxi.v.add(i, j, (fx.v.get(i, j) != 0. ? dEdf.v.get(i, j) : 0.));
      }
    }
  }
}
