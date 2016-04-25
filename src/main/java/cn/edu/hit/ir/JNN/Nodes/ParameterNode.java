package cn.edu.hit.ir.JNN.Nodes;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;
import cn.edu.hit.ir.JNN.Parameters;
import java.util.Vector;

public class ParameterNode extends AbstractParameterNode {
  public ParameterNode(Parameters p) {
    params = p;
    dim = new Dim(p.dim);
  }

  @Override
  public Dim dimForward(Vector<Dim> xs) {
    assert (xs.size() == 0);
    return dim;
  }

  @Override
  public String asString(final Vector<String> argNames) {
    return "Parameters(" + dim + ", " + params + ")";
  }

  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    assert (xs.size() == 0);
    fx.v = params.values.v;
  }

  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
    throw new RuntimeException(
        "called backward() on a arity 0 node");
  }

  @Override
  public void accumulateGrad(Tensor g) {
    params.accumulateGrad(g);
  }

  public String getName() {
    return "ParameterNode";
  }

  public Parameters params;
}