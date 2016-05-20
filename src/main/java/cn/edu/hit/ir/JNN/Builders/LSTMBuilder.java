package cn.edu.hit.ir.JNN.Builders;

import java.util.Arrays;
import java.util.Vector;

import cn.edu.hit.ir.JNN.ComputationGraph;
import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Expression;
import cn.edu.hit.ir.JNN.Model;
import cn.edu.hit.ir.JNN.Parameters;

public class LSTMBuilder extends AbstractRNNBuilder{
  //enum
  private int X2I = 0;
  private int H2I = 1;
  private int C2I = 2;
  private int BI = 3;
  private int X2O = 4;
  private int H2O = 5;
  private int C2O = 6;
  private int BO = 7;
  private int X2C = 8;
  private int H2C = 9;
  private int BC = 10;
  
  //first index is layer, then ...
  public Vector<Vector<Parameters>> params;
  //first index is layer, then ...
  public Vector<Vector<Expression>> paramVars;
  //first index is time, second is layer
  public Vector<Vector<Expression>> h, c;
  //initial values of h and c at each layer
  // - both default to zero matrix input
  public boolean hasInitialState;
  public Vector<Expression> h0;
  public Vector<Expression> c0;
  int layers;
  double dropoutRate;
  
  public LSTMBuilder(int layers_, int inputDim, int hiddenDim, Model model) {
    params = new Vector<Vector<Parameters>>();
    paramVars = new Vector<Vector<Expression>>();
    h = new Vector<Vector<Expression>>();
    c = new Vector<Vector<Expression>>();
    h0 = new Vector<Expression>();
    c0 = new Vector<Expression>();
    
    layers = layers_;
    int layerInputDim = inputDim;
    for (int i = 0; i < layers; ++i) {
      //i
      Parameters pX2i = model.addParameters(Dim.create(hiddenDim, layerInputDim));
      Parameters pH2i = model.addParameters(Dim.create(hiddenDim, hiddenDim));
      Parameters pC2i = model.addParameters(Dim.create(hiddenDim, hiddenDim));
      Parameters pBi = model.addParameters(Dim.create(hiddenDim));
      //o
      Parameters pX2o = model.addParameters(Dim.create(hiddenDim, layerInputDim));
      Parameters pH2o = model.addParameters(Dim.create(hiddenDim, hiddenDim));
      Parameters pC2o = model.addParameters(Dim.create(hiddenDim, hiddenDim));
      Parameters pBo = model.addParameters(Dim.create(hiddenDim));
      //c
      Parameters pX2c = model.addParameters(Dim.create(hiddenDim, layerInputDim));
      Parameters pH2c = model.addParameters(Dim.create(hiddenDim, hiddenDim));
      Parameters pBc = model.addParameters(Dim.create(hiddenDim));
      layerInputDim = hiddenDim; //output(hidden) from 1st layer is input to next
    
      Vector <Parameters> ps = new Vector<Parameters>(
          Arrays.asList(pX2i, pH2i, pC2i, pBi,
              pX2o, pH2o, pC2o, pBo, pX2c, pH2c, pBc));
      params.addElement(ps);
    }
    dropoutRate = 0.0; 
  }
  public void setDropout(double d) {
    dropoutRate = d;
  }
  //in general, you should disable dropout at test time
  public void disableDropout() {
    dropoutRate = 0.0;
  }
  public Expression back() {
    return (cur == -1 ? h0.lastElement() : h.get(cur).lastElement());
  }
  public Vector<Expression> finalH() {
    return (h.size() == 0 ? h0 : h.lastElement());
  }
  public Vector<Expression> finalS() {
    Vector<Expression> ret = new Vector<Expression>(c.size() == 0 ? c0 : c.lastElement());
    for (Expression myH : finalH()) 
      ret.addElement(myH);
    return ret;
  }
  public int numH0Components() {
    return 2 * layers;
  }
  
  public Vector<Expression> getH(int i) {
    return (i == -1 ? h0 : h.get(i));
  }
  public Vector<Expression> getS(int i) {
    Vector<Expression> ret = new Vector<Expression>(i == -1 ? c0 : c.get(i));
    for (Expression myH : getH(i))
      ret.addElement(myH);
    return ret;
  }
  public void copy(final AbstractRNNBuilder rnn) {
    final LSTMBuilder rnnLSTM = (LSTMBuilder)rnn;
    assert(params.size() == rnnLSTM.params.size());
    for (int i = 0; i < params.size(); ++i)
      for (int j = 0; j < params.get(i).size(); ++j)
        params.get(i).get(j).copy(rnnLSTM.params.get(i).get(j));
  }
  
  protected void newGraphImpl(ComputationGraph cg) {
    paramVars.clear();
    for (int i = 0; i < layers; ++i) {
      Vector<Parameters> p = params.get(i);
      //i
      Expression iX2i = Expression.Creator.parameter(cg, p.get(X2I));
      Expression iH2i = Expression.Creator.parameter(cg, p.get(H2I));
      Expression iC2i = Expression.Creator.parameter(cg, p.get(C2I));
      Expression iBi = Expression.Creator.parameter(cg, p.get(BI));
      //o
      Expression iX2o = Expression.Creator.parameter(cg, p.get(X2O));
      Expression iH2o = Expression.Creator.parameter(cg, p.get(H2O));
      Expression iC2o = Expression.Creator.parameter(cg, p.get(C2O));
      Expression iBo = Expression.Creator.parameter(cg, p.get(BO));
      //c
      Expression iX2c = Expression.Creator.parameter(cg, p.get(X2C));
      Expression iH2c = Expression.Creator.parameter(cg, p.get(H2C));
      Expression iBc = Expression.Creator.parameter(cg, p.get(BC));
      
      Vector<Expression> vars = new Vector<Expression>(
          Arrays.asList(iX2i, iH2i, iC2i, iBi, iX2o, iH2o, iC2o, iBo,
              iX2c, iH2c, iBc));
      paramVars.addElement(vars);
    }
  }
  
  protected void startNewSequenceImpl(final Vector<Expression> hinit) {
    h.clear();
    c.clear();
    if (hinit.size() > 0) {
      assert(layers * 2 == hinit.size());
      h0.setSize(layers);
      c0.setSize(layers);
      for (int i = 0; i < layers; ++i) {
        c0.set(i, hinit.get(i));
        h0.set(i, hinit.get(i + layers));
      }
      hasInitialState = true;
    } else {
      hasInitialState = false;
    }
  }
  
  protected Expression addInputImpl(int prev, final Expression x) {
    h.addElement(new Vector<Expression>(layers));
    c.addElement(new Vector<Expression>(layers));
    Vector<Expression> ht = h.lastElement();
    Vector<Expression> ct = c.lastElement();
    Expression in = x;
    for (int i = 0; i < layers; ++i) {
      final Vector<Expression> vars = paramVars.get(i);
      Expression iHTm1 = new Expression(), iCTm1 = new Expression();
      boolean hasPrevState = (prev >= 0 || hasInitialState);
      if (prev < 0) {
        if (hasInitialState) {
          //initial value for h and c at timestamp 0 in layer i
          //defaults to zero matrix input if not set in addParameterEdges 
          iHTm1 = h0.get(i);
          iCTm1 = c0.get(i);
        }
      } else { //t > 0
        iHTm1 = h.get(prev).get(i);
        iCTm1 = c.get(prev).get(i);
      }
      //apply dropout according to http://arxiv.org/pdf/1409.2329v5.pdf
      if (Math.abs(dropoutRate) > 1E-10) in = Expression.Creator.dropout(in, dropoutRate);
      //input
      Expression iAit;
      if (hasPrevState) 
        iAit = Expression.Creator.affineTransform(
            Arrays.asList(vars.get(BI), vars.get(X2I), in, vars.get(H2I),
                iHTm1, vars.get(C2I), iCTm1));
      else 
        iAit = Expression.Creator.affineTransform(
            Arrays.asList(vars.get(BI), vars.get(X2I), in));
      Expression iIt = Expression.Creator.logistic(iAit);
      //forget
      Expression iFt = Expression.Creator.constantMinusX(1.0, iIt);
      //write memory cell
      Expression iAwt;
      if (hasPrevState) 
        iAwt = Expression.Creator.affineTransform(
            Arrays.asList(vars.get(BC), vars.get(X2C), in, vars.get(H2C), iHTm1));
      else 
        iAwt = Expression.Creator.affineTransform(
            Arrays.asList(vars.get(BC), vars.get(X2C), in));
      
      Expression iWt = Expression.Creator.tanh(iAwt);
      //output
      if (hasPrevState) {
        Expression iNwt = Expression.Creator.cwiseMultiply(iIt, iWt);
        Expression iCrt = Expression.Creator.cwiseMultiply(iFt, iCTm1);
        ct.add(i, Expression.Creator.add(iCrt, iNwt));
      } else {
        ct.add(i, Expression.Creator.cwiseMultiply(iIt, iWt));
      }
      Expression iAot;
      if (hasPrevState) 
        iAot = Expression.Creator.affineTransform(
            Arrays.asList(vars.get(BO), vars.get(X2O), in, vars.get(H2O),
               iHTm1, vars.get(C2O), ct.get(i)));
      else 
        iAot = Expression.Creator.affineTransform(
          Arrays.asList(vars.get(BO), vars.get(X2O), in));
      Expression iOt = Expression.Creator.logistic(iAot);
      Expression phT = Expression.Creator.tanh(ct.get(i));
      in = Expression.Creator.cwiseMultiply(iOt, phT);
      ht.add(i, in);
    }
    if (Math.abs(dropoutRate) > 1E-10)
      return Expression.Creator.dropout(ht.lastElement(), dropoutRate);
    else return ht.lastElement();
  }
}
