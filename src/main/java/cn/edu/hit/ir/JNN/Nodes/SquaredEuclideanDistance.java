package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;
import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

public class SquaredEuclideanDistance extends Node {
  public SquaredEuclideanDistance(List<Integer> x) {
    dim = new Dim();
    args.setSize(x.size());
    for (int i = 0; i < x.size(); i++) {
      args.setElementAt(x.get(i), i);
    }
  }

  @Override
  public String asString(final Vector<String> argNames) {
    return "|| " + argNames.get(0) + " - " + argNames.get(1) + " ||^2";
  }

  @Override
  public Dim dimForward(final Vector<Dim> xs) {
    assert(xs.size() == 2);
    if (!xs.get(0).singleBatch().equals(xs.get(1).singleBatch())) {
      StringBuilder s = new StringBuilder(
          "Bad input dimensions in SquaredEuclideanDistance: ");
      s.append(xs.get(0)).append(" ").append(xs.get(1));
      throw new IllegalArgumentException(s.toString());
    }
    return Dim.createBatches(1, Math.max(xs.get(0).bd, xs.get(1).bd));
  }

  public boolean supportsMultibatch() {
    return true;
  }

  @Override
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    assert (xs.size() == 2);
    fx.v.putScalar(0, xs.get(0).v.squaredDistance(xs.get(1).v));
  }

  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
    assert (i < 2);

    double scale = dEdf.v.getDouble(0) * 2;
    if (i == 1) scale = -scale;
    dEdxi.v.addi(xs.get(0).v.sub(xs.get(1).v).mul(scale));
  }
}
