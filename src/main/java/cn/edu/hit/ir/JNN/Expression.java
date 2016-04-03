package cn.edu.hit.ir.JNN;

public class Expression {
  public ComputationGraph pg;
  public int i;
  Expression() {
    pg = null;
  }
  Expression(ComputationGraph pg_, int i_) {
    pg = pg_;
    i = i_;
  }
  public Tensor value() {
    return pg.getValue(i);
  }
}
