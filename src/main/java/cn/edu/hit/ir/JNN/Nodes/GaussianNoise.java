package cn.edu.hit.ir.JNN.Nodes;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;
import cn.edu.hit.ir.JNN.Utils.TensorUtils;

import java.util.List;
import java.util.Vector;

/**
 * Created by dancingsoul on 2016/5/4.
 */
public class GaussianNoise extends Node{
  double stddev;
  private Tensor aux;
  public GaussianNoise(final List<Integer> a, double stddev_) {
    super(a);
    stddev = stddev_;
  }
  @Override
  public String asString(final Vector<String> argNames) {return "";}

  @Override
  public Dim dimForward(final Vector<Dim> xs) {
    assert(xs.size() == 1);
    return xs.get(0);
  }

  public boolean supportsMultibatch() {
    return true;
  }

  @Override
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    aux = new Tensor(dim);
    TensorUtils.randomizeNormal(0, stddev, aux);
    fx.v.assign(xs.get(0).v.add(aux.v));
  }
  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i_, Tensor dEdxi) {
    dEdxi.v.addi(dEdf.v);
  }
}
