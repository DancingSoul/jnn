package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.NormOps;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

public class SquaredEuclideanDistance extends Node {
  public SquaredEuclideanDistance(List<Integer> x) {
    dim = new Dim();
    args.setSize(x.size());
    for (int i = 0; i < x.size(); i++) {
      args.setElementAt(x.get(i), i);
    }
  }

  public String asString(final Vector<String> argNames) {
    return "";
  }

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

  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    assert (xs.size() == 2);
    DenseMatrix64F x1 = xs.get(0).v;
    DenseMatrix64F x2 = xs.get(1).v;
    DenseMatrix64F tmp = new DenseMatrix64F(x1.numRows, x1.numCols);
    CommonOps.subtract(x1, x2, tmp);
    fx.v.set(0, NormOps.normP2(tmp));
  }

  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
    assert (i < 2);
    DenseMatrix64F x1 = xs.get(0).v;
    DenseMatrix64F x2 = xs.get(1).v;
    DenseMatrix64F tmp = new DenseMatrix64F(x1.numRows, x1.numCols);
    double scale = dEdf.v.get(0) * 2;
    if (i == 1) scale = -scale;
    CommonOps.subtract(x1, x2, tmp);
    CommonOps.scale(scale, tmp);
    CommonOps.addEquals(dEdxi.v, tmp);
  }
}
