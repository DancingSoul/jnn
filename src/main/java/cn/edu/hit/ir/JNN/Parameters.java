package cn.edu.hit.ir.JNN;


import org.nd4j.linalg.factory.Nd4j;

public class Parameters extends AbstractParameters {
  public Dim dim;
  public Tensor values;
  public Tensor g;
  public Tensor gCheck;

  Parameters() {
    dim = new Dim();
    values = new Tensor();
    g = new Tensor();
    gCheck = new Tensor();
  }

  public Parameters(Dim d, double scale){
    dim = new Dim(d);
    values = new Tensor(d);
    g = new Tensor(d);
    gCheck = new Tensor(d);

    if (Math.abs(scale) > 1e-10) {
      TensorUtils.randomize(values, scale);
    } else {
      TensorUtils.randomize(values);
    }

    TensorUtils.zero(g);
    TensorUtils.zero(gCheck);
  }

  public void scaleParameters(double a){
    g.v.muli(a);
  }

  public final double squaredL2norm(){
    return Math.pow(values.v.norm2Number().doubleValue(), 2);
  }

  public final double gSquaredL2norm(){
    return Math.pow(values.v.norm2Number().doubleValue(), 2);
  }

  public final int size(){
    return dim.size();
  }

  public void copy(final Parameters param) {
    assert(dim.equals(param.dim));
    TensorUtils.copyElements(values, param.values);
  }

  public void accumulateGrad(final Tensor d) {
    g.v.addi(d.v);
  }
  public void clear() {
    TensorUtils.zero(g);
  }
}
