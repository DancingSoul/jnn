package cn.edu.hit.ir.JNN;

abstract class AbstractParameters {
  public abstract void scaleParameters(double a);
  public abstract double squaredL2norm();
  public abstract double gSquaredL2norm();
  public abstract int size();
}
