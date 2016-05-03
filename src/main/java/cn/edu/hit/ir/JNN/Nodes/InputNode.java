package cn.edu.hit.ir.JNN.Nodes;

import java.util.Vector;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;
import cn.edu.hit.ir.JNN.Utils.TensorUtils;

public class InputNode extends Node {
  public InputNode(final Dim d, final Vector<Double> dat) {
    dim = new Dim(d);
    data = dat;
  }

  @Override
  public Dim dimForward(Vector<Dim> xs) {
    return dim;
  }

  @Override
  public String asString(final Vector<String> argNames) {
    return "Constant(" + dim + ")";
  }

  @Override
  public void forwardImpl(Vector<Tensor> xs, Tensor fx) {
    assert(xs.size() == 0);
    boolean isInputAddressAligned = false;
    if (!isInputAddressAligned) {
      TensorUtils.setElements(fx, data);
    } else {
      // TODo
    }
  }

  @Override
  public void backwardImpl(Vector<Tensor> xs, Tensor fx, Tensor dEdf, int i, Tensor dEdxi) {
    throw new RuntimeException(
        "called backward() on a arity 0 node");
  }

  Vector <Double> data;
}
