package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

public class MatrixMultiply extends Node{
  public MatrixMultiply(List<Integer> x) {
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
    assert(xs.size() == 2);
    assert(fx.d.bd == Math.max(xs.get(0).d.bd, xs.get(1).d.bd));
    if (xs.get(0).d.bd == 1) {
      CommonOps.mult(fx.v, xs.get(0).v, xs.get(1).v);
    } else {
      //...
    }
  }
  public void backwardImpl(final Vector<Tensor> xs,
      final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
    assert(i < 2);
    int maxB = Math.max(xs.get(0).d.bd, xs.get(1).d.bd);
    if (i == 0) {
      for (int b = 0; b < maxB; ++b) {
        DenseMatrix64F tmp = new DenseMatrix64F(dEdf.v.numRows, xs.get(1).v.numRows);
        CommonOps.transpose(xs.get(1).v);
        CommonOps.mult(tmp, dEdf.v, xs.get(1).v);
        CommonOps.transpose(xs.get(1).v);
        CommonOps.addEquals(dEdxi.v, tmp);
      }
    } else {
      if (xs.get(0).d.bd == 1) {
        DenseMatrix64F tmp = new DenseMatrix64F(xs.get(0).v.numCols, dEdf.v.numCols);
        CommonOps.transpose(xs.get(0).v);
        CommonOps.mult(tmp, xs.get(0).v, dEdf.v);
        CommonOps.transpose(xs.get(0).v);
        CommonOps.addEquals(dEdxi.v, tmp);
      } else {
        //...
      }
    }
  }
}
