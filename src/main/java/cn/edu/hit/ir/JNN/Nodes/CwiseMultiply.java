package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

public class CwiseMultiply extends Node {
  public CwiseMultiply(List<Integer> x) {
    super(x);
  }

  public String asString(final Vector<String> argNames) {
    return "";
  }

  @Override
  public Dim dimForward(final Vector<Dim> xs) {
    assert(xs.size() == 2);
    Dim d = xs.get(0).truncate();
    if (d.singleBatch() != xs.get(1).truncate().singleBatch()) {
      StringBuilder s = new StringBuilder(
              "Mismatched input dimensions in CwiseMultiply: ");
      s.append(xs.get(0)).append(" ").append(xs.get(1));
      throw new IllegalArgumentException(s.toString());
    }
    d.bd = Math.max(xs.get(1).bd, d.bd);
    return d;
  }

  @Override
  public boolean supportsMultibatch() {
    return true;
  }

  @Override
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    assert(xs.size() == 2);
    fx.v = xs.get(0).vec().mul(xs.get(1).vec());
  }

  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
    assert(xs.size() == 2);
    if (i == 0) {
      dEdxi.vec().addi(dEdf.vec().mul(xs.get(1).vec()));
    } else {
      dEdxi.vec().addi(dEdf.vec().mul(xs.get(0).vec()));
    }
  }
}