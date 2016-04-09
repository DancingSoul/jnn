package cn.edu.hit.ir.JNN;

import java.lang.reflect.Constructor;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.atomic.AtomicInteger;

import cn.edu.hit.ir.JNN.Nodes.ConstParameterNode;
import cn.edu.hit.ir.JNN.Nodes.InputNode;
import cn.edu.hit.ir.JNN.Nodes.LookupNode;
import cn.edu.hit.ir.JNN.Nodes.Node;
import cn.edu.hit.ir.JNN.Nodes.ParameterNode;
import cn.edu.hit.ir.JNN.Nodes.ScalarInputNode;

public class ComputationGraph {
  public Vector<Node> nodes;
  public Vector<Integer> parameterNodes;

  AbstractExecutionEngine ee;

  public ComputationGraph() {
    nodes = new Vector<Node>();
    parameterNodes = new Vector<Integer>();
    ee = new SimpleExecutionEngine(this);
  }
  
  void clear() {
    nodes.clear();
    parameterNodes.clear();
  }
  
  public int addInput(AtomicDouble s) {
    int newNodeIndex = nodes.size();
    nodes.addElement(new ScalarInputNode(s));
    setDimForNewNode(newNodeIndex);
    return newNodeIndex;
  }

  public int addInput(final Dim d, final Vector <Double> data) {
    int newNodeIndex = nodes.size();
    nodes.addElement(new InputNode(d, data));
    setDimForNewNode(newNodeIndex);
    return newNodeIndex;
  }

  public int addParameters(Parameters p) {
    int newNodeIndex = nodes.size();
    ParameterNode newNode = new ParameterNode(p);
    nodes.addElement(newNode);
    parameterNodes.addElement(newNodeIndex);
    setDimForNewNode(newNodeIndex);
    return newNodeIndex;
  }
  
  public int addConstParameters(Parameters p) {
    int newNodeIndex = nodes.size();
    ConstParameterNode newNode = new ConstParameterNode(p);
    nodes.addElement(newNode);
    setDimForNewNode(newNodeIndex);
    return newNodeIndex;
  }

//  public int addLookup(LookupParameters p, AtomicInteger index){
//    int newNodeIndex = nodes.size();
//    LookupNode newNode = new LookupNode(p, index);
//    nodes.addElement(newNode);
//    parameterNodes.addElement(newNodeIndex);
//    setDimForNewNode(newNodeIndex);
//    return newNodeIndex;
//  }

  public int addLookup(LookupParameters p, final Vector <Integer> indices) {
    int newNodeIndex = nodes.size();
    LookupNode newNode = new LookupNode(p, indices);
    nodes.addElement(newNode);
    parameterNodes.addElement(newNodeIndex);
    setDimForNewNode(newNodeIndex);
    return newNodeIndex;
  }

//  public int addConstLookup(LookupParameters p, AtomicInteger index){
//    int newNodeIndex = nodes.size();
//    LookupNode newNode = new LookupNode(p, index);
//    nodes.addElement(newNode);
//    setDimForNewNode(newNodeIndex);
//    return newNodeIndex;
//  }

  public int addConstLookup(LookupParameters p, final Vector <Integer> indices) {
    int newNodeIndex = nodes.size();
    LookupNode newNode = new LookupNode(p, indices);
    nodes.addElement(newNode);
    setDimForNewNode(newNodeIndex);
    return newNodeIndex;
  }
  
  /*public int addFunction(Node func, List <Integer> arguments) {
    int newNodeIndex = nodes.size();
    try {
      Constructor<?> cons = func.getClass().getConstructor(arguments.getClass());
      nodes.addElement((Node)cons.newInstance(arguments));
    } catch(Exception e) {
      e.printStackTrace();
    } 
    setDimForNewNode(newNodeIndex);
    return newNodeIndex;
  }*/
  
  public int addFunction(Node func) {
    int newNodeIndex = nodes.size();
    nodes.addElement(func);
    setDimForNewNode(newNodeIndex);
    return newNodeIndex;
  }

  private void setDimForNewNode(final int i) {
    Node node = nodes.get(i);
    Vector<Dim> xds = new Vector<Dim>(node.arity());
    int ai = 0;
    for (int arg : node.args) {
      xds.add(ai, new Dim(nodes.get(arg).dim));
      ++ai;
    }
    node.dim = node.dimForward(xds);
  }

  public Tensor incrementalForward() {
    return ee.incrementalForward();
  }

  public Tensor forward() {
    return ee.forward();
  }

  public Tensor getValue(int i) {
    return ee.getValue(i);
  }

  public Tensor getValue(final Expression e) {
    return this.getValue(e.i);
  }

  public void invalidate() {
    ee.invalidate();
  }

  public void backward() {
    ee.backward();
  }

  public void backward(int i) {
    ee.backward(i);
  }

  public void PrintGrapviz() {
    //...
  }
  
  public void checkParameterNode(Node a) {
    final double eta = 1E-8;
    Parameters b = ((ParameterNode)a).params;
    for (int i = 0; i < b.values.v.numRows; i++) {
      for (int j = 0; j < b.values.v.numCols; j++) {
        b.values.v.add(i, j, eta);
        double x = TensorUtils.toScalar(forward());
        b.values.v.add(i, j, -eta * 2);
        double y = TensorUtils.toScalar(forward());
        b.values.v.add(i, j, eta);
        b.gCheck.v.set(i, j, (x - y) / 2.0 / eta);
      }
    }
  }
  
  public void checkLookupNode(Node a) {
    final double eta = 1E-8;
    LookupParameters b = ((LookupNode)a).params;
    Vector <Integer> c = ((LookupNode)a).indices;
    for (int k = 0; k < c.size(); k++) {
      int index = c.get(k);
      for (int i = 0; i < b.values.get(index).v.numRows; ++i) {
        for (int j = 0; j < b.values.get(index).v.numCols; ++j) {
          b.values.get(index).v.add(i, j, eta);
          double x = TensorUtils.toScalar(forward());
          b.values.get(index).v.add(i, j, -eta * 2);
          double y = TensorUtils.toScalar(forward());
          b.values.get(index).v.add(i, j, eta);
          b.gradsCheck.get(index).v.set(i, j, (x - y) / 2.0 / eta);
        }
      }
    }
  }
  
  public void gradientCheck() {
    for (int i = 0; i < parameterNodes.size(); ++i) {
      int j = parameterNodes.get(i);
      if (nodes.get(j).getName() == "ParameterNode") {
        checkParameterNode(nodes.get(j));
      } else {
        checkLookupNode(nodes.get(j));
      }
    }
  }
}
