package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

public class Sum extends Node{
  public Sum(List<Integer> x) {
    dim = new Dim();
    args.setSize(x.size());
    for (int i = 0; i < x.size(); i++) {
      args.setElementAt(x.get(i), i);
    }
  }
  public String asString(final Vector<String> argNames) {
    return "";
  }
  public Dim dimForward(final Vector <Dim> xs) {
    return null;
  }
  public boolean supportsMultibatch() {
    return true;
  }
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    int numArgs = xs.size();
    if (numArgs == 1) {
      fx.v = xs.get(0).v;
      return;
    }
    DenseMatrix64F res = fx.v;
    for (int i = 0; i < numArgs; i++) 
      CommonOps.addEquals(res, xs.get(i).v);
  }
  public void backwardImpl(final Vector<Tensor> xs,
      final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
    CommonOps.addEquals(dEdxi.v, dEdf.v);
  }
}
