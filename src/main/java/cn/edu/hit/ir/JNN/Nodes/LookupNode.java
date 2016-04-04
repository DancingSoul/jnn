package cn.edu.hit.ir.JNN.Nodes;

import java.util.Vector;
import java.util.concurrent.atomic.AtomicInteger;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.LookupParameters;
import cn.edu.hit.ir.JNN.Tensor;

public class LookupNode extends AbstractParameterNode {
  public LookupNode(LookupParameters p, AtomicInteger ind) {
    dim = new Dim(p.dim);
    index = ind;
    params = p;
  }

  public LookupNode(LookupParameters p, final Vector<Integer> inds) {
    dim = new Dim(p.dim);
    indices = inds;
    params = p;
    dim.bd = inds.size();
  }

  public String asString(final Vector<String> argNames) {
    return "";
  }

  @Override
  public Dim dimForward(final Vector<Dim> xs) {
    return dim;
  }

  @Override
  public boolean supportsMultibatch() {
    return true;
  }

  @Override
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    assert (xs.size() == 0);
    if (index != null) {
      assert (index.get() < params.values.size());
      assert (fx.d.getNumBatchElements() == 1);
      fx.v = params.values.get(index.get()).v;
    } else {
      assert (indices != null);
      assert (fx.d.getNumBatchElements() == indices.size());
      for (int b = 0; b < indices.size(); ++b) {
        int i = indices.get(b);
        assert (i < params.values.size());
        //...???

      }
    }
  }

  @Override
  public void backwardImpl(Vector<Tensor> xs, Tensor fx, Tensor dEdf, int i, Tensor dEdxi) {
    throw new RuntimeException(
        "called backward() on a arity 0 node");
  }

  public void accumulateGrad(final Tensor g) {
    if (index != null) {
      params.accumulateGrad(index.get(), g);
    } else {
      assert (indices != null);
      //...???
    }
  }

  //Dim dim;
  AtomicInteger index;
  Vector<Integer> indices;
  LookupParameters params;
}