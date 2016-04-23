package cn.edu.hit.ir.JNN.Nodes;

import java.util.Arrays;
import java.util.Vector;

import cn.edu.hit.ir.JNN.AtomicDouble;
import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

public class ScalarInputNode extends Node {
  public ScalarInputNode(AtomicDouble s) {
    dim = new Dim();
    data = s;
  }

  public String asString(final Vector<String> argNames) {
    return "";
  }

  public Dim dimForward(final Vector<Dim> xs) {
    return new Dim(Arrays.asList(1));
  }

  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    fx.v.putScalar(0, data.doubleValue());
  }

  public void backwardImpl(Vector<Tensor> xs, Tensor fx, Tensor dEdf, int i, Tensor dEdxi) {
    throw new RuntimeException(
        "called backward() on a arity 0 node");
  }

  AtomicDouble data;
}
