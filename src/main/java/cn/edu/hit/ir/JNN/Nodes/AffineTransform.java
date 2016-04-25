package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;


import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

public class AffineTransform extends Node {
  public AffineTransform(List<Integer> x) {
    super(x);
  }

  public String asString(final Vector<String> argNames) {
    StringBuilder sb = new StringBuilder(argNames.get(0));
    for (int i = 1; i < argNames.size(); i += 2) {
      sb.append(" + ").append(argNames.get(i)).append(" * ").append(argNames.get(i + 1));
    }
    return sb.toString();
  }

  @Override
  public Dim dimForward(final Vector<Dim> xs) {
    if ((xs.size() - 1) % 2 != 0) {
      StringBuilder s = new StringBuilder(
              "Bad number of inputs for AffineTransform: ");
      throw new IllegalArgumentException(s.toString());
    }
    Dim d = xs.get(0);
    for (int i = 1; i < xs.size(); i += 2) {
      if (xs.get(i).getNumCols() != xs.get(i + 1).getNumRows() ||
              xs.get(0).getNumRows() != xs.get(i).getNumRows() ||
              xs.get(0).getNumCols() != xs.get(i).getNumCols()) {
        StringBuilder s = new StringBuilder(
                "Bad dimensions for AffineTransform: ");
        throw new IllegalArgumentException(s.toString());
      }
      d.bd = Math.max(Math.max(d.bd, xs.get(i).bd), xs.get(i + 1).bd);
    }
    return d;
  }

  @Override
  public boolean supportsMultibatch() {
    return true;
  }

  @Override
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    assert(xs.size() % 2 == 1);
    if (xs.size() == 1) {
      fx.v = xs.get(0).v.dup();
      return;
    } else {
      // Add, using broadcasting or not
      if (fx.d.bd > 1 && xs.get(0).d.bd == 1) {
        fx.rowcolMatrix();
        for (int i = 0; i < fx.v.size(1); i++) {
          fx.v.getColumn(i).assign(xs.get(0).vec().transpose());
        }
      } else {
        for (int b = 0; b < fx.d.bd; ++b) {
          fx.setBatchMatrix(b, xs.get(0).getBatchMatrix(b));
        }
      }
      //Multiply
      for (int i = 1; i < xs.size(); i += 2) {
        if (xs.get(i).d.bd == 1 && xs.get(i + 1).d.bd == fx.d.bd) {
          fx.vec().addi(xs.get(i).getBatchMatrix(0).mmul(xs.get(i + 1).colbatchMatrix()).reshape(fx.d.size()));
        } else {
          assert(xs.get(i + 1).d.bd == 1 || xs.get(i + 1).d.bd == xs.get(i).d.bd);
          for (int b = 0; b < fx.d.bd; ++b) {
            fx.addBatchMatrix(b, xs.get(i).getBatchMatrix(b).mmul(xs.get(i + 1).getBatchMatrix(b)));
          }
        }
      }
    }
  }

  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
    assert(i < xs.size());
    if (i == 0) {
      dEdxi.v.addi(dEdf.v);
    } else if (i % 2 == 1){
      dEdxi.v.addi(dEdf.v.mmul(xs.get(i + 1).v.transpose()));
    } else {
      dEdxi.v.addi(xs.get(i - 1).v.transpose().mmul(dEdf.v));
    }
  }
}