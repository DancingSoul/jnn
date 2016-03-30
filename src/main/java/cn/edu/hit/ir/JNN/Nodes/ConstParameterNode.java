package cn.edu.hit.ir.JNN.Nodes;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Parameters;
import cn.edu.hit.ir.JNN.Tensor;

import java.util.Vector;

class ConstParameterNode extends Node {
  ConstParameterNode(Parameters p) {
    dim = new Dim(p.dim);
    params = p;
  }

  public String asString(final Vector<String> argNames) {
    //...
    return "";
  }

  public Dim dimForward(final Vector<Dim> xs) {
    assert (xs.size() == 0);
    return dim;
  }

  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    assert (xs.size() == 0);
    fx.v = params.values.v;
  }

  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
    //"called backward() on arity 0 node : i = ??
    //abort();
  }

  public Dim dim;
  public Parameters params;
}