package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

public class ConstantMinusX extends Node {
  public double c;
  public ConstantMinusX(List<Integer> x, double c_) {
    super(x);
    c = c_;
  }

  public String asString(final Vector<String> argNames) {
    return "";
  }

  @Override
  public Dim dimForward(final Vector<Dim> xs) {
    assert(xs.size() == 1);
    return xs.get(0);
  }

  @Override
  public boolean supportsMultibatch() {
    return true;
  }

  @Override
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    fx.v = xs.get(0).v.sub(c).neg();
  }

  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
    dEdxi.v.subi(dEdf.v);
  }
}