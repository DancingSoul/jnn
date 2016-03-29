package cn.edu.hit.ir.JNN.Nodes;

import cn.edu.hit.ir.JNN.Tensor;

public abstract class AbstractParameterNode extends Node {
  public abstract void accumulateGrad(final Tensor g);
}