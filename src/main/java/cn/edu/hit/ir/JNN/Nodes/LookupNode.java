package cn.edu.hit.ir.JNN.Nodes;

import java.util.Vector;
import java.util.concurrent.atomic.AtomicInteger;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.LookupParameters;
import cn.edu.hit.ir.JNN.Tensor;

public class LookupNode extends AbstractParameterNode {
  public LookupNode(LookupParameters p, final Vector<Integer> inds) {
    dim = new Dim(p.dim);
    indices = inds;
    params = p;
    dim.bd = inds.size();
  }

  @Override
  public String asString(final Vector<String> argNames) {
    return "LookupParameter(|x|=" + params.values.size() + " --> " + params.dim + ")";
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
    if (indices.size() == 1) {
      assert (indices.get(0) < params.values.size());
      assert (fx.d.getNumBatchElements() == 1);
      fx.v = params.values.get(indices.get(0)).v;
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
    if (indices.size() == 1) {
      params.accumulateGrad(indices.get(0), g);
    } else {
      assert (indices != null);
      //...???
    }
  }
  
  public String getName() {
    return "LookupNode";
  }

  //Dim dim;
  public Vector<Integer> indices;
  public LookupParameters params;
}