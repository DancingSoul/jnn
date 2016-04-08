package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

public abstract class Node {
  public Node() {
    dim = new Dim();
    args = new Vector<Integer>();
  }

  public Node(List<Integer> x) {
    dim = new Dim();
    args = new Vector<Integer>(); 
    args.setSize(x.size());
    for (int i = 0; i < x.size(); i++) {
      args.setElementAt(x.get(i), i);
    }
  }

  public abstract Dim dimForward(final Vector<Dim> xs);

  public abstract String asString(final Vector<String> args);

  public int auxStorageSize() {
    return 0;
  }

  public abstract void forwardImpl(final Vector<Tensor> xs, Tensor fx);

  public abstract void backwardImpl(final Vector<Tensor> xs,
                                    final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi);

  public boolean supportsMultibatch(){return false;}

  public void forward(final Vector<Tensor> xs, Tensor fx) {
    if (this.supportsMultibatch() || fx.d.getNumBatchElements() == 1) {
      forwardImpl(xs, fx);
    } else {
      //...  
    }
  }

  public void backward(final Vector<Tensor> xs, final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
    if (this.supportsMultibatch() || fx.d.getNumBatchElements() == 1) {
      backwardImpl(xs, fx, dEdf, i, dEdxi);
    } else {
      //...
    }
  }

  public final int arity() {
    return args.size();
  }

  public Dim dim;
  public Vector<Integer> args;
}
