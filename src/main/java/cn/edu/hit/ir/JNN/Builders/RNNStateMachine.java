package cn.edu.hit.ir.JNN.Builders;

public class RNNStateMachine {
  private RNNState q_;
  public RNNStateMachine() {
    q_ = RNNState.CREATED;
  }
  public void failure(RNNOp op) {
    System.out.println("State transition error: currently in state" + 
        q_ +  " but received operation" + op );
    System.exit(0);
  }
  public void transition(RNNOp op) {
    if (q_ == RNNState.CREATED) {
      if (op == RNNOp.new_graph) { q_ = RNNState.GRAPH_READY; return; }
      failure(op);
    } else if (q_ == RNNState.GRAPH_READY) {
      if (op == RNNOp.new_graph) { return; }
      if (op == RNNOp.start_new_sequence) { q_ = RNNState.READING_INPUT; return; } 
      failure(op);
    } else if (q_ == RNNState.READING_INPUT) {
      if (op == RNNOp.add_input) { return; }
      if (op == RNNOp.start_new_sequence) { return; }
      if (op == RNNOp.new_graph) { q_ = RNNState.GRAPH_READY; return; }
      failure(op);
    }
  }
}
