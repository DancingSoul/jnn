package cn.edu.hit.ir.JNN;

import cn.edu.hit.ir.JNN.Utils.TensorUtils;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashSet;
import java.util.Vector;

public class LookupParameters extends AbstractParameters {
  public Dim dim;
  public Vector<Tensor> values;
  public Vector<Tensor> grads;
  public Vector<Tensor> gradsCheck;
  public HashSet<Integer> nonZeroGrads;

  LookupParameters() {
    dim = new Dim();
    values = new Vector<Tensor>();
    grads = new Vector<Tensor>();
    gradsCheck = new Vector<Tensor>();
    nonZeroGrads = new HashSet<Integer>();
  }

  LookupParameters(int n, Dim d){
    dim = new Dim(d);
    values = new Vector<Tensor>(n);
    grads = new Vector<Tensor>(n);
    gradsCheck = new Vector<Tensor>(n);
    nonZeroGrads = new HashSet<Integer>();

    for (int i = 0; i < n; i++){
      values.add(i, new Tensor(d));
      TensorUtils.randomize(values.get(i));

      grads.add(i, new Tensor(d));
      TensorUtils.zero(grads.get(i));
      
      gradsCheck.add(i, new Tensor(d));
      TensorUtils.zero(gradsCheck.get(i));
    }
  }

  public void scaleParameters(double a){
    for (Tensor p : values) {
      p.v.muli(a);
    }
  }

  public double squaredL2norm(){
    double a = 0;
    for (int i = 0; i < values.size(); ++i) {
      a += Math.pow(values.get(i).v.norm2Number().doubleValue(), 2);
    }
    return a;
  }

  public double gSquaredL2norm(){
    double a = 0;
    for (Integer i : nonZeroGrads) {
      a += Math.pow(grads.get(i).v.norm2Number().doubleValue(), 2);
    }
    return a;
  }

  public int size(){
    return values.size() * dim.size();
  }

  public void initialize(int index, final Vector <Double> val) {
    values.get(index).v = Nd4j.zeros(val.size(), 1);
    for (int i = 0; i < val.size(); i++) {
      values.get(index).v.putScalar(i, val.get(i));
    }
  }

  public void copy(final LookupParameters param) {
    assert(dim.equals(param.dim));
    for (int i = 0; i < param.values.size(); ++i)
      TensorUtils.copyElements(values.get(i), param.values.get(i));
  }

  public void accumulateGrad(int index, final Tensor d) {
    grads.get(index).v.addi(d.v);
  }

  public void clear() {
    for (Integer i : nonZeroGrads) {
      TensorUtils.zero(grads.get(i));
    }
    nonZeroGrads.clear();
  }
}