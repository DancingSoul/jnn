package cn.edu.hit.ir.JNN.Nodes;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;
import cn.edu.hit.ir.JNN.Utils.TensorUtils;

import java.util.List;
import java.util.Vector;

/**
 * Created by dancingsoul on 2016/5/4.
 */
public class PickNegLogSoftmax extends Node{
  public Vector<Integer> vals;
  public PickNegLogSoftmax(final List<Integer> a, Vector<Integer> vals_) {
    super(a);
    vals = vals_;
  }
  @Override
  public String asString(final Vector<String> argNames) {return "";}

  @Override
  public Dim dimForward(final Vector<Dim> xs) {
    assert(xs.size() == 1);
    /*if (!LooksLikeVector(xs.get(0))) {
      throw new RuntimeException("Bad Input Dimensions in PickNegLogSoftmax");
    }*/
    return Dim.createBatches(1, xs.get(0).bd);
  }

  public boolean supportsMultibatch() {
    return true;
  }

  @Override
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    if (xs.get(0).d.getNumCols() == 1) {
      if (vals.size() == 1) {
        //TODO
      } else {
        //TODO
      }
    } else {
      throw new RuntimeException("PickNegLogSoftmax :: forward not yet implemented for multiple columns");
    }
  }
  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i_, Tensor dEdxi) {

  }
}
