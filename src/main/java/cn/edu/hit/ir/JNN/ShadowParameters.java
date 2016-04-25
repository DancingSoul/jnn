package cn.edu.hit.ir.JNN;

/**
 * Created by dancingsoul on 2016/4/25.
 */
public class ShadowParameters {
  public Tensor h;
  public ShadowParameters(final Parameters p) {
    h = new Tensor(p.dim);
  }
}
