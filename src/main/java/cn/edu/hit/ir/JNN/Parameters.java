package cn.edu.hit.ir.JNN;

import org.ejml.ops.CommonOps;
import org.ejml.ops.NormOps;

public class Parameters extends AbstractParameters {
  public Dim dim;
  public Tensor values;
  public Tensor g;

  Parameters() {
  }

  Parameters(Dim d, double scale){
    dim = new Dim(d);
    values = new Tensor(d);
    g = new Tensor(d);

    if (Math.abs(scale) > 1e-10) {
      TensorUtils.randomize(values, scale);
    } else {
      TensorUtils.randomize(values);
    }

    TensorUtils.zero(g);
  }

  public void scaleParameters(double a){
    CommonOps.scale(a, g.v);
  }

  public final double squaredL2norm(){
    return Math.pow(NormOps.normP2(values.v), 2);
  }

  public final double gSquaredL2norm(){
    return Math.pow(NormOps.normP2(g.v), 2);
  }

  public final int size(){
    return dim.size();
  }

  public void copy(final Parameters param) {
    assert(dim.equals(param.dim));
    TensorUtils.copyElements(values, param.values);
  }

  public void accumulateGrad(final Tensor d){
    CommonOps.addEquals(g.v, d.v);
  }

  public void clear() {
    TensorUtils.zero(g);
  }
}
