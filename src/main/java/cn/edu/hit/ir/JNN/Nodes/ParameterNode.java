package cn.edu.hit.ir.JNN.Nodes;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;
import cn.edu.hit.ir.JNN.Parameters;
import java.util.Vector;

class ParameterNode extends AbstractParameterNode {
  ParameterNode(Parameters p) {

  }

  public Dim dimForward(Vector<Dim> xs) {
    assert (xs.size() == 0);
    return null;
  }

  public String asString(final Vector<String> argNames) {
    //...
    return "";
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

  public void accumulateGrad(Tensor g) {
    params.accumulateGrad(g);
  }

  public Dim dim;
  public Parameters params;
}