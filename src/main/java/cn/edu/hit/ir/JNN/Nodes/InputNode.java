package cn.edu.hit.ir.JNN.Nodes;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

import java.util.Vector;

class InputNode extends Node {
  InputNode() {

  }

  public Dim dimForward(Vector<Dim> xs) {
    return null;
  }

  public String asString(final Vector<String> argNames) {
    return "";
  }

  public void forwardImpl(Vector<Tensor> xs, Tensor fx) {

  }

  public void backwardImpl(Vector<Tensor> xs, Tensor fx, Tensor dEdf, int i, Tensor dEdxi) {

  }

  public Dim dimForward() {
    return null;
  }
}
