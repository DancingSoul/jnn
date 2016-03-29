package cn.edu.hit.ir.JNN;

public abstract class AbstractExecutionEngine {
  AbstractExecutionEngine() {
  }

  protected ComputationGraph cg;

  public abstract void invalidate();

  public abstract Tensor forward();

  public abstract Tensor forward(int i);

  public abstract Tensor incrementalForward();

  public abstract Tensor incrementalForward(int i);

  public abstract Tensor getValue(int i);

  public abstract void backward();

  public abstract void backward(int i);

  protected AbstractExecutionEngine(ComputationGraph cg_) {
    cg = cg_;
  }
}