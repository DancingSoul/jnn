package cn.edu.hit.ir.JNN.Nodes;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;
import java.util.Vector;

/**
 * Created by dancingsoul on 2016/5/23.
 */
public class Concatenate extends Node{
  Vector<Integer> sp;
  public Concatenate(List<Integer> a) {
    super(a);
    sp = new Vector<Integer>();
    sp.setSize(a.size() + 1);
  }

  @Override
  public String asString(final Vector<String> argNames) {
    return "";
  }

  @Override
  public Dim dimForward(final Vector<Dim> xs) {
    int newRows = 0;
    Dim dr = xs.get(0);
    for (Dim c : xs) {
      newRows += c.at(0);
      dr.set(0, c.at(0));
      if (dr.singleBatch().equals(c.singleBatch()) == false) {
        StringBuilder s = new StringBuilder(
                "Bad input dimensions in Concatenate ");
        throw new IllegalArgumentException(s.toString());
      }
      dr.bd = Math.max(dr.bd, c.bd);
    }
    dr.set(0, newRows);
    return dr;
  }


  @Override
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    assert(xs.size() > 0);

    INDArray res = xs.get(0).v;
    sp.set(0, 0);
    sp.set(1, res.length());
    for (int i = 1; i < xs.size(); i++) {
      res = Nd4j.concat(0, res, xs.get(i).v);
      sp.set(i + 1, res.length());
    }
    fx.v = res;
  }

  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {

    dEdxi.v = dEdf.v.get(NDArrayIndex.interval(sp.get(i), sp.get(i + 1)), NDArrayIndex.all());
  }
  public Dim dim;
}
