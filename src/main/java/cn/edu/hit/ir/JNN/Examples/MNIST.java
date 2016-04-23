package cn.edu.hit.ir.JNN.Examples;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Collections;
import java.util.Date;
import java.util.Vector;

import cn.edu.hit.ir.JNN.ComputationGraph;
import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Expression;
import cn.edu.hit.ir.JNN.Model;
import cn.edu.hit.ir.JNN.Parameters;
import cn.edu.hit.ir.JNN.TensorUtils;
import cn.edu.hit.ir.JNN.Trainers.SimpleSGDTrainer;

class MLCBuilder {
  final static int HIDDEN_SIZE = 30;
  Parameters pW;
  Parameters pb;
  Parameters pV;
  Parameters pa;
  
  MLCBuilder(Model m) {
    pW = m.addParameters(Dim.create(HIDDEN_SIZE, 784));
    pb = m.addParameters(Dim.create(HIDDEN_SIZE));
    pV = m.addParameters(Dim.create(10, HIDDEN_SIZE));
    pa = m.addParameters(Dim.create(10));
  }
  
  public Expression buildPredictionScores(Model m, ComputationGraph cg, Vector <Double> xValues) {
    Expression W = Expression.Creator.parameter(cg, pW);
    Expression b = Expression.Creator.parameter(cg, pb);
    Expression V = Expression.Creator.parameter(cg, pV);
    Expression a = Expression.Creator.parameter(cg, pa);
    
    Expression x = Expression.Creator.input(cg, Dim.create(784), xValues);

    Expression h = Expression.Creator.logistic(
        Expression.Creator.add(Expression.Creator.multiply(W, x), b));
    Expression yPredict = Expression.Creator.logistic(
        Expression.Creator.add(Expression.Creator.multiply(V, h), a));
    return yPredict;
  }
}

public class MNIST {

  public static void readFile(String fileName, Vector <Vector<Double> > x,
                              Vector <Vector<Double> > y) {
    try { 
      BufferedReader reader = new BufferedReader(new FileReader(fileName));
      String line = null;
      while((line = reader.readLine()) != null){ 
        String item[] = line.split(",");
        
        
        Vector <Double> vec = new Vector<Double>(784);
        for (int i = 1; i < item.length; i++) {
          double tmp = Double.parseDouble(item[i]);
          vec.addElement(tmp > 0 ? 1.0 : 0.0);
        }
        x.addElement(vec);
        
        vec = new Vector <Double>(10);
        
        for (int i = 0; i < 10; i++) { 
          vec.addElement(0.0);
        }
        
        vec.set(Integer.parseInt(item[0]), 1.0);
        y.addElement(vec);
        
        //System.out.println(item.length); 
      } 
    } catch (Exception e) { 
      e.printStackTrace(); 
    } 
  }
  
  
  
  
  public static void main(String args[]) {
    Vector <Vector<Double> > xTrain = new Vector<Vector<Double> >();
    Vector <Vector<Double> > yTrain = new Vector<Vector<Double> >();
    
    readFile("mnist_train.csv", xTrain, yTrain);

    System.out.println("Done reading train.");
    
    Vector <Vector<Double> > xTest = new Vector<Vector<Double> >();
    Vector <Vector<Double> > yTest = new Vector<Vector<Double> >();
    
    readFile("mnist_test.csv", xTest, yTest);

    System.out.println("Done reading test.");

    System.out.println(new Date().getTime());

    Model m = new Model();
    SimpleSGDTrainer sgd = new SimpleSGDTrainer(m);
    MLCBuilder mlc = new MLCBuilder(m);


    Vector<Integer> label = new Vector<Integer>();
    label.setSize(xTrain.size());
    for (int i = 0; i < label.size(); i++)
      label.set(i, i);

    System.out.println(label.size());
    for (int iteration = 0; iteration < 1; ++iteration) {
      double lossIter = 0.0;
      Collections.shuffle(label);
      for (int i = 0; i < 2000; i++) {
        ComputationGraph cg = new ComputationGraph();
        Expression yPredict = mlc.buildPredictionScores(m, cg, xTrain.get(label.get(i)));
        Vector<Double> p = TensorUtils.toVector(cg.forward());
        
        Expression y = Expression.Creator.input(cg, Dim.create(10), yTrain.get(label.get(i)));
        Expression loss = Expression.Creator.squaredDistance(yPredict, y);
        lossIter += TensorUtils.toScalar(cg.incrementalForward());
        cg.backward();
        sgd.update(1.0);
        if (i % 100 == 0) System.out.println(i);
      }
      sgd.updateEpoch();
      lossIter /= 100;
      System.out.println("E = " + lossIter);
    }
    System.out.println(new Date().getTime());
    int cnt = 0;
    for (int i = 0; i < xTest.size(); i++) {
      ComputationGraph cg = new ComputationGraph();
      Vector<Double> tmp = xTest.get(i);
      Expression yPredict = mlc.buildPredictionScores(m, cg, xTest.get(i));
      Vector<Double> p = TensorUtils.toVector(cg.forward());
      int p1 = -1, p2 = -1;
      double mx = 0.0;
      for (int j = 0; j < 10; j++) {
        if (p.get(j) > mx){
          mx = p.get(j);
          p1 = j;
        }
        if (yTest.get(i).get(j) == 1.0) 
          p2 = j;
      }
      if (p1 == p2) cnt++;
      //System.out.println(mx + " " + p1 + " : " + p2);
    }
    System.out.println(cnt + " / " + xTest.size());
  }
}
