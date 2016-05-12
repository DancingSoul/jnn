package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.factory.Nd4j;

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
    assert(xs.size() == 1);

    /*for (int i = 0; i < fx.v.length(); ++i) {
      fx.v.putScalar(i, sigmoid(xs.get(0).v.getDouble(i)));
    }*/

    Nd4j.getExecutioner().exec(new Sigmoid(xs.get(0).v, fx.v));
  }

  @Override
  public void backwardImpl(final Vector<Tensor> xs,
      final Tensor fx, final Tensor dEdf, int i_, Tensor dEdxi) {
    for (int i = 0; i < fx.v.length(); ++i) {
      double y = fx.v.getDouble(i);
      dEdxi.v.putScalar(i, dEdf.v.getDouble(i) * y * (1 - y) + dEdxi.v.getDouble(i));
    }
    //dEdxi.vec().addi(dEdf.v.muli(fx.v).muli(fx.v.rsub(1.)));
  }

  public Dim dim;
}
