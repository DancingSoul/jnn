package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;
import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear;
import org.nd4j.linalg.factory.Nd4j;

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
    Nd4j.getExecutioner().exec(new RectifedLinear(xs.get(0).v, fx.v));
  }

  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i_, Tensor dEdxi) {
    for (int i = 0; i < fx.v.size(0); ++i) {
      for (int j = 0; j < fx.v.size(1); ++j) {
        dEdxi.v.putScalar(new int[]{i, j},
            (fx.v.getDouble(i, j) != 0. ? dEdf.v.getDouble(i, j) : 0.) + dEdxi.v.getDouble(i, j));
      }
    }
  }
}
