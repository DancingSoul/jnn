package cn.edu.hit.ir.JNN.Builders;

public enum RNNState {
  CREATED(1), GRAPH_READY(2), READING_INPUT(3);
  private int nCode;
  private RNNState(int nCode_) {
    this.nCode = nCode_;
  }
  public String toString() {
    return String.valueOf ( this . nCode );
  }
}
