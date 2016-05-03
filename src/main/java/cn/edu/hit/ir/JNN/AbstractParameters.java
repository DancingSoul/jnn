package cn.edu.hit.ir.JNN;

import cn.edu.hit.ir.JNN.Trainers.AbstractTrainer;

import java.io.Serializable;

abstract class AbstractParameters implements Serializable{
  private static final long serialVersionUID = 2238574422776967031L;
  public abstract void scaleParameters(double a);
  public abstract double squaredL2norm();
  public abstract double gSquaredL2norm();
  public abstract int size();
}
