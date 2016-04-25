package cn.edu.hit.ir.JNN.Nodes;

import java.util.Arrays;
import java.util.List;
import java.util.Vector;


import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

public class MatrixMultiply extends Node {
  public MatrixMultiply(List<Integer> x) {
    super(x);
  }

  @Override
  public String asString(final Vector<String> argNames) {
    return argNames.get(0) + " * " + argNames.get(1);
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
      fx.v = xs.get(0).getBatchMatrix(0).mmul(xs.get(1).colbatchMatrix());
      //fx.v = xs.get(0).vec().mmul(xs.get(1).vec());
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
        dEdxi.addBatchMatrix(b, dEdf.getBatchMatrix(b).mmul(xs.get(1).getBatchMatrix(b).transpose()));
      }
    } else {
      if (xs.get(0).d.bd == 1) {
        dEdxi.vec().addi(xs.get(0).getBatchMatrix(0).transpose().mmul(dEdf.colbatchMatrix()).reshape(dEdxi.d.size()));
      } else {
        for (int b = 0; b < maxB; ++b) {
          dEdxi.addBatchMatrix(b, xs.get(0).getBatchMatrix(b).transpose().mmul(dEdf.getBatchMatrix(b)));
        }
      }
    }
  }
}
