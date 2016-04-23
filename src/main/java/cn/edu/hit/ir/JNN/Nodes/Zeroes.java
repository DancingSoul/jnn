package cn.edu.hit.ir.JNN.Nodes;

import java.util.Vector;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;
import org.nd4j.linalg.factory.Nd4j;

public class Zeroes extends Node{
  Zeroes(final Dim d){
    dim = new Dim(d);
  }
  public String asString(final Vector<String> argNames) {
    return "";
  }
  public Dim dimForward(final Vector <Dim> xs) {
    return dim;
  }
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    fx.v = Nd4j.zerosLike(fx.v);
  }
  public void backwardImpl(final Vector<Tensor> xs,
      final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
    throw new RuntimeException(
        "called backward() on a arity 0 node");
  }
  public Dim dim;
}
