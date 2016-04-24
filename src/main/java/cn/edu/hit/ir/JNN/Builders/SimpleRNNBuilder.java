package cn.edu.hit.ir.JNN.Builders;

import java.util.Arrays;
import java.util.Vector;

import cn.edu.hit.ir.JNN.ComputationGraph;
import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Expression;
import cn.edu.hit.ir.JNN.Model;
import cn.edu.hit.ir.JNN.Parameters;

public class SimpleRNNBuilder extends AbstractRNNBuilder{
  private static int X2H = 0;
  private static int H2H = 1;
  private static int HB = 2;
  private static int L2H = 3;
  //first index is layer, then x2h h2h hb
  private Vector <Vector<Parameters>> params;
  //first index is layer, then x2h h2h hb
  private Vector <Vector<Expression>> paramVars;
  //first index is time, second is layer
  private Vector <Vector<Expression>> h;
  /*initial value of h
   * defaluts to zero matrix input
   */
  private Vector <Expression> h0;
  private int layers;
  boolean lagging;
  
  public SimpleRNNBuilder() {
    super();
    params = new Vector<Vector<Parameters>>();
    paramVars = new Vector<Vector<Expression>>();
    h = new Vector<Vector<Expression>>();
    h0 = new Vector<Expression>();
  }
  public SimpleRNNBuilder(int layers, int inputDim, int hiddenDim,
      Model model) {
   this(layers, inputDim, hiddenDim, model, false); 
  }
  public SimpleRNNBuilder(int layers_, int inputDim, int hiddenDim,
      Model model, boolean supportLags) {
    params = new Vector<Vector<Parameters>>();
    paramVars = new Vector<Vector<Expression>>();
    h = new Vector<Vector<Expression>>();
    h0 = new Vector<Expression>();
    layers = layers_;
    lagging = supportLags;
    int layerInputDim = inputDim;
    for (int i = 0; i < layers; ++i) {
      Parameters pX2h = model.addParameters(Dim.create(hiddenDim, layerInputDim));
      Parameters pH2h = model.addParameters(Dim.create(hiddenDim, hiddenDim));
      Parameters pHb = model.addParameters(Dim.create(hiddenDim));
      Vector<Parameters> ps = new Vector<Parameters>(Arrays.asList(pX2h, pH2h, pHb));
      if (lagging) 
        ps.addElement(model.addParameters(Dim.create(hiddenDim, hiddenDim)));;
      params.addElement(ps);
      layerInputDim = hiddenDim;
    }
  }
  
  protected void newGraphImpl(ComputationGraph cg) {
    paramVars.clear();
    for (int i = 0; i < layers; ++i) {
      Parameters pX2h = params.get(i).get(X2H);
      Parameters pH2h = params.get(i).get(H2H);
      Parameters pHb = params.get(i).get(HB);
      Expression iX2h = Expression.Creator.parameter(cg, pX2h);
      Expression iH2h = Expression.Creator.parameter(cg, pH2h);
      Expression iHb = Expression.Creator.parameter(cg, pHb);
      Vector<Expression> vars = new Vector<Expression>(
          Arrays.asList(iX2h, iH2h, iHb));
      if (lagging) {
        Parameters pL2h = params.get(i).get(L2H);
        Expression iL2h = Expression.Creator.parameter(cg, pL2h);
        vars.addElement(iL2h);
      }
      paramVars.addElement(vars);
    }
  }
  
  protected void startNewSequenceImpl(final Vector<Expression> h_0) {
    h.clear();
    h0 = h_0;
    if (h0.size() > 0) { assert(h0.size() == layers); }
  }
  
  protected Expression addInputImpl(int prev, final Expression in) {
    final int t = h.size();
    Vector<Expression> tmp = new Vector<Expression>(layers);
    tmp.setSize(layers);
    h.addElement(tmp);
    Expression x = in;
    for (int i = 0; i < layers; ++i) {
      final Vector<Expression> vars = paramVars.get(i);
      //y <--- f(x)
      Expression y = Expression.Creator.affineTransform(
          Arrays.asList(vars.get(2), vars.get(0), x));
      //y <--- g(y_prev)
      if (prev == -1 && h0.size() > 0) {
        y = Expression.Creator.affineTransform(
            Arrays.asList(y, vars.get(1), h0.get(i)));
      } else if (prev >= 0) {
        y = Expression.Creator.affineTransform(
            Arrays.asList(y, vars.get(1), h.get(prev).get(i)));
      }
      //x <---- tanh
      x =  Expression.Creator.tanh(y);
      h.get(t).set(i, x);
    }
    return h.get(t).lastElement();
  }
  public Expression addAuxiliaryInput(final Expression in, final Expression aux) {
    final int t = h.size();
    Vector<Expression> tmp = new Vector<Expression>(layers);
    tmp.setSize(layers);
    h.addElement(tmp);
    Expression x = in;
   
    for (int i = 0; i < layers; ++i) {
      final Vector<Expression> vars = paramVars.get(i);
      assert(vars.size() >= L2H + 1);
      Expression y = Expression.Creator.affineTransform(
          Arrays.asList(vars.get(HB), vars.get(X2H), x, vars.get(L2H), aux));
      if (t == 0 && h0.size() > 0) {
        y = Expression.Creator.affineTransform(
            Arrays.asList(y, vars.get(H2H), h0.get(i)));
      } else if (t >= 1) {
        y = Expression.Creator.affineTransform(
            Arrays.asList(y, vars.get(H2H), h.get(t - 1).get(i)));
      }
      x =  Expression.Creator.tanh(y);
      h.get(t).set(i, x); 
    }
    return h.get(t).lastElement();
  }
  public Expression back() {
    return cur == -1 ? h0.lastElement() : h.get(cur).lastElement();
  }
  public Vector<Expression> finalH() {
    return h0.size() == 0 ? h0 : h.lastElement();
  }
  public Vector<Expression> finalS() {
    return finalH();
  }
  public Vector<Expression> getH(int i) {
    return i == -1 ? h0 : h.get(i);
  }
  public Vector<Expression> getS(int i) {
    return getH(i);
  }
  public void copy(final AbstractRNNBuilder rnn) {
    final SimpleRNNBuilder rnnSimple = (SimpleRNNBuilder)rnn;
    assert(params.size() == rnnSimple.params.size());
    for (int i = 0; i < rnnSimple.params.size(); ++i) {
      params.get(i).get(0).copy(rnnSimple.params.get(i).get(0));
      params.get(i).get(1).copy(rnnSimple.params.get(i).get(1));
      params.get(i).get(2).copy(rnnSimple.params.get(i).get(2));
    }
  } 
  public int numH0Components(){
    return layers;
  }
}
