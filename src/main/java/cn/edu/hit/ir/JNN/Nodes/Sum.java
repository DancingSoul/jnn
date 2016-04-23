package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

public class Sum extends Node {
  public Sum(List<Integer> x) {
    super(x);
  }

  public String asString(final Vector<String> argNames) {
    return "";
  }

  @Override
  public Dim dimForward(final Vector<Dim> xs) {
    Dim d = xs.get(0).truncate();
    for (int i = 1; i < xs.size(); ++i) {
      if (!d.singleBatch().equals(xs.get(i).truncate().singleBatch())) {
        StringBuilder s = new StringBuilder("Mismatched input dimensions in Sum: arg#");
        s.append(i).append(": ")
            .append(xs.get(i).singleBatch())
            .append(" != ")
            .append(d.singleBatch());
        throw new IllegalArgumentException(s.toString());
      }
      d.bd = Math.max(xs.get(i).bd, d.bd);
    }
    return d;
  }

  @Override
  public boolean supportsMultibatch() {
    return true;
  }

  @Override
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    int numArgs = xs.size();
    if (numArgs == 1) {
      fx.v = xs.get(0).v;
      return;
    }
    for (int i = 0; i < numArgs; i++) {
      fx.v.addi(xs.get(i).v);
    }
  }

  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
    dEdxi.v.addi(dEdf.v);
  }
}
