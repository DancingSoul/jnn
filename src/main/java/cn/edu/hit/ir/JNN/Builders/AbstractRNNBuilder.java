package cn.edu.hit.ir.JNN.Builders;

import java.util.Vector;

import cn.edu.hit.ir.JNN.ComputationGraph;
import cn.edu.hit.ir.JNN.Expression;

public abstract class AbstractRNNBuilder {
  protected int cur;
  private RNNStateMachine sm;
  private Vector <Integer> head;
  
  public AbstractRNNBuilder() {
    head = new Vector<Integer>();
    cur = -1;
  }
  public int state() {
    return cur;
  }
  public void newGraph(ComputationGraph cg) {
    sm.transition(RNNOp.new_graph);
    newGraphImpl(cg);
  }
  
  public void startNewSequence() {
    startNewSequence(new Vector <Expression>());
  }
  
  public void startNewSequence(final Vector<Expression> h0) {
    sm.transition(RNNOp.start_new_sequence);
    cur = -1;
    head.clear();
    startNewSequenceImpl(h0);
    
  }
  
  public Expression addInput(final Expression x) {
    sm.transition(RNNOp.add_input);
    head.addElement(cur);
    int rcp = cur;
    cur = head.size() - 1;
    return addInputImpl(rcp, x);
  }
  
  public Expression addInput(final int prev, final Expression x) {
    sm.transition(RNNOp.add_input);
    head.addElement(prev);
    cur = head.size() - 1;
    return addInputImpl(prev, x);
  }
  
  public void rewindOneStep() {
    cur = head.get(cur);
  }
  
  public abstract Expression back();
  
  public abstract Vector<Expression> finalH();
  public abstract Vector<Expression> getH(int i);
  public abstract Vector<Expression> finalS();
  public abstract int numH0Components();
  public abstract Vector<Expression> getS(int i);
  public abstract void copy(final AbstractRNNBuilder params);
  
  protected abstract void newGraphImpl(ComputationGraph cg);
  protected abstract void startNewSequenceImpl(final Vector <Expression> h0);
  protected abstract Expression addInputImpl(int prev, final Expression x);
}
