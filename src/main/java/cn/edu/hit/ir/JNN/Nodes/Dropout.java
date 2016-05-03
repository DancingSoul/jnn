package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;
import cn.edu.hit.ir.JNN.Utils.TensorUtils;

public class Dropout extends Node {
  public double p;
  private Tensor aux;
  public Dropout(List<Integer> x, double p_) {
    super(x);
    p = p_;
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
    aux = new Tensor(dim);
    TensorUtils.randomBernoulli(aux, 1.0 - p, 1.0 / (1.0 - p));
    fx.v = xs.get(0).vec().mul(aux.vec());
  }

  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
    dEdxi.vec().addi(dEdf.vec().mul(aux.vec()));
  }
}