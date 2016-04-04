package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

public class Tanh extends Node {
  public Tanh(List<Integer> x) {
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
    return xs.get(0);
  }

  public boolean supportsMultibatch() {
    return true;
  }

  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    for (int i = 0; i < fx.v.numRows; ++i) {
      for (int j = 0; j < fx.v.numCols; ++j) {
        fx.v.set(i, j, Math.tanh(xs.get(0).v.get(i, j)));
      }
    }
  }

  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i_, Tensor dEdxi) {
    for (int i = 0; i < fx.v.numRows; ++i) {
      for (int j = 0; j < fx.v.numCols; ++j) {
        dEdxi.v.add(i, j, dEdf.v.get(i, j) * (1.0 - Math.pow(fx.v.get(i, j), 2)));
      }
    }
  }
}
