package cn.edu.hit.ir.JNN;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.NormOps;

import java.util.HashSet;
import java.util.Vector;

public class LookupParameters extends AbstractParameters {
  public Dim dim;
  public Vector<Tensor> values;
  public Vector<Tensor> grads;
  public HashSet<Integer> nonZeroGrads;

  LookupParameters() {
    dim = new Dim();
    values = new Vector<Tensor>();
    grads = new Vector<Tensor>();
    nonZeroGrads = new HashSet<Integer>();
  }

  LookupParameters(int n, Dim d){
    dim = new Dim(d);
    values = new Vector<Tensor>(n);
    grads = new Vector<Tensor>(n);
    nonZeroGrads = new HashSet<Integer>();

    for (int i = 0; i < n; i++){
      values.add(i, new Tensor(d));
      TensorUtils.randomize(values.get(i));

      grads.add(i, new Tensor(d));
      TensorUtils.zero(grads.get(i));
    }
  }

  public void scaleParameters(double a){
    for (Tensor p : values) {
      CommonOps.scale(a, p.v);
    }
  }

  public double squaredL2norm(){
    double a = 0;
    for (int i = 0; i < values.size(); ++i) {
      a += Math.pow(NormOps.normP2(values.get(i).v), 2);
    }
    return a;
  }

  public double gSquaredL2norm(){
    double a = 0;
    for (Integer i : nonZeroGrads) {
      a += Math.pow(NormOps.normP2(grads.get(i).v), 2);
    }
    return a;
  }

  public int size(){
    return values.size() * dim.size();
  }

  public void initialize(int index, final Vector <Double> val) {
    double[] tmp = new double[val.size()];
    for (int i = 0; i < val.size(); i++) {
      tmp[i] = val.get(i);
    }
    values.get(index).v = new DenseMatrix64F(val.size(), 1, true, tmp);
  }

  public void copy(final LookupParameters param) {
    assert(dim.equals(param.dim));
    for (int i = 0; i < param.values.size(); ++i)
      TensorUtils.copyElements(values.get(i), param.values.get(i));
  }

  public void accumulateGrad(int index, final Tensor d) {
    CommonOps.addEquals(grads.get(index).v, d.v);
  }

  public void clear() {
    for (Integer i : nonZeroGrads) {
      TensorUtils.zero(grads.get(i));
    }
    nonZeroGrads.clear();
  }
}