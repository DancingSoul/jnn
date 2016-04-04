package cn.edu.hit.ir.JNN.Nodes;

import java.util.Arrays;
import java.util.List;
import java.util.Vector;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

public class MatrixMultiply extends Node {
  public MatrixMultiply(List<Integer> x) {
    dim = new Dim();
    args.setSize(x.size());
    for (int i = 0; i < x.size(); i++) {
      args.setElementAt(x.get(i), i);
    }
  }

  public String asString(final Vector<String> argNames) {
    return "";
  }

  @Override
  public Dim dimForward(final Vector<Dim> xs) {
    assert(xs.size() == 2);
    if (xs.get(0).getNumCols() != xs.get(1).getNumRows()) {
      StringBuilder s = new StringBuilder(
          "Mismatched input dimensions in MatrixMultiply: ");
      s.append(xs.get(0)).append(" ").append(xs.get(1));
      throw new IllegalArgumentException(s.toString());
    }
    if (xs.get(1).getNumDimensions() == 1) {
      return Dim.createBatches(xs.get(0).getNumRows(), Math.max(xs.get(0).bd, xs.get(1).bd));
    }
    return Dim.createBatches(xs.get(0).getNumRows(), xs.get(1).getNumCols(),
        Math.max(xs.get(0).bd, xs.get(1).bd));
  }

  @Override
  public boolean supportsMultibatch() {
    return true;
  }

  @Override
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    assert (xs.size() == 2);
    assert (fx.d.bd == Math.max(xs.get(0).d.bd, xs.get(1).d.bd));
    if (xs.get(0).d.bd == 1) {
      CommonOps.mult(xs.get(0).v, xs.get(1).v, fx.v);
    } else {
      //...
    }
  }

  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
    assert (i < 2);
    int maxB = Math.max(xs.get(0).d.bd, xs.get(1).d.bd);
    if (i == 0) {
      for (int b = 0; b < maxB; ++b) {
        DenseMatrix64F tmp = new DenseMatrix64F(dEdf.v.numRows, xs.get(1).v.numRows);
        CommonOps.transpose(xs.get(1).v);
        CommonOps.mult(dEdf.v, xs.get(1).v, tmp);
        CommonOps.transpose(xs.get(1).v);
        CommonOps.addEquals(dEdxi.v, tmp);
      }
    } else {
      if (xs.get(0).d.bd == 1) {
        DenseMatrix64F tmp = new DenseMatrix64F(xs.get(0).v.numCols, dEdf.v.numCols);
        CommonOps.transpose(xs.get(0).v);
        CommonOps.mult(xs.get(0).v, dEdf.v, tmp);
        CommonOps.transpose(xs.get(0).v);
        CommonOps.addEquals(dEdxi.v, tmp);
      } else {
        //...
      }
    }
  }
}
