package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.api.ops.impl.transforms.TanhDerivative;
import org.nd4j.linalg.factory.Nd4j;

/**
 * \tanh(x_1)
 *
 *  - arity: 1
 */
public class MyTanh extends Node {
  public MyTanh(List<Integer> x) {
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
    Nd4j.getExecutioner().exec(new Tanh(xs.get(0).v, fx.v));
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
