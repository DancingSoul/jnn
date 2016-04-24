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
      fx.v = xs.get(0).v.dup();
      for (int i = 1; i < xs.size(); i += 2) {
        fx.v.addi(xs.get(i).v.mmul(xs.get(i + 1).v));
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