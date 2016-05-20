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
import cn.edu.hit.ir.JNN.Utils.SerializationUtils;
import cn.edu.hit.ir.JNN.Utils.TensorUtils;
import cn.edu.hit.ir.JNN.Trainers.SimpleSGDTrainer;

class MLCBuilder {
  private final static int HIDDEN_SIZE = 30;
  private Parameters pW;
  private Parameters pb;
  private Parameters pV;
  private Parameters pa;

  MLCBuilder(Model m) {
    pW = m.addParameters(Dim.create(HIDDEN_SIZE, 784));
    pb = m.addParameters(Dim.create(HIDDEN_SIZE));
    pV = m.addParameters(Dim.create(10, HIDDEN_SIZE));
    pa = m.addParameters(Dim.create(10));
  }

  public Expression buildPredictionScores(Model m, ComputationGraph cg, Vector<Double> xValues) {
    Expression W = Expression.Creator.parameter(cg, pW);
    Expression b = Expression.Creator.parameter(cg, pb);
    Expression V = Expression.Creator.parameter(cg, pV);
    Expression a = Expression.Creator.parameter(cg, pa);

    Expression x = Expression.Creator.input(cg, Dim.create(784), xValues);

    Expression h = Expression.Creator.logistic(
            Expression.Creator.add(Expression.Creator.multiply(W, x), b));
    return Expression.Creator.logistic(
            Expression.Creator.add(Expression.Creator.multiply(V, h), a));
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

        vec = new Vector<Double>(10);

        for (int i = 0; i < 10; i++) {
          vec.addElement(0.0);
        }

        vec.set(Integer.parseInt(item[0]), 1.0);
        y.addElement(vec);
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public static void main(String args[]) {
    Vector<Vector<Double>> xTrain = new Vector<Vector<Double>>();
    Vector<Vector<Double>> yTrain = new Vector<Vector<Double>>();
    readFile("mnist_train.csv", xTrain, yTrain);
    System.out.println("Done reading train with " + xTrain.size() + " instance(s).");

    Vector<Vector<Double>> xTest = new Vector<Vector<Double>>();
    Vector<Vector<Double>> yTest = new Vector<Vector<Double>>();
    readFile("mnist_test.csv", xTest, yTest);
    System.out.println("Done reading test with " + xTest.size() + " instance(s).");

    System.out.println("Done reading test.");

    Long startOfTraining = new Date().getTime();

    Model m = new Model();
    SimpleSGDTrainer sgd = new SimpleSGDTrainer(m);
    MLCBuilder mlc = new MLCBuilder(m);


    //SerializationUtils.loadModel("MNIST.obj", m);
    Vector<Integer> label = new Vector<Integer>();
    label.setSize(xTrain.size());
    for (int i = 0; i < label.size(); i++) {
      label.set(i, i);
    }


    System.err.println(label.size());
    int maxIteration = 1;
    int numInstances = Math.min(2000, label.size());
    if (args.length >= 1) {
      maxIteration = Integer.parseInt(args[0]);
    }
    if (args.length >= 2) {
      numInstances = Integer.parseInt(args[1]);
    }
    System.err.println("Going to train " + maxIteration + " iteration(s).");
    System.err.println("Going to train " + numInstances + " instance(s).");
    for (int iteration = 0; iteration < maxIteration; ++iteration) {
      double lossIter = 0.0;
      Collections.shuffle(label);
      for (int i = 0; i < numInstances; i++) {
        ComputationGraph cg = new ComputationGraph();
        Expression yPredict = mlc.buildPredictionScores(m, cg, xTrain.get(label.get(i)));
        Vector<Double> p = TensorUtils.toVector(cg.forward());

        Expression y = Expression.Creator.input(cg, Dim.create(10), yTrain.get(label.get(i)));
        Expression loss = Expression.Creator.squaredDistance(yPredict, y);
        lossIter += TensorUtils.toScalar(cg.incrementalForward());
        cg.backward();
        sgd.update(1.0);

        if (i % 100 == 0) System.err.println("[" + iteration + "," + i + "]");
      }
      sgd.updateEpoch();
      lossIter /= numInstances;
      System.err.println("Iteration #" + iteration + " E = " + lossIter);
    }
    System.err.println("consume: " + (new Date().getTime() - startOfTraining));

    int cnt = 0;
    for (int i = 0; i < xTest.size(); i++) {
      ComputationGraph cg = new ComputationGraph();
      Vector<Double> tmp = xTest.get(i);
      Expression yPredict = mlc.buildPredictionScores(m, cg, xTest.get(i));
      Vector<Double> p = TensorUtils.toVector(cg.forward());
      int p1 = -1, p2 = -1;
      double mx = 0.0;
      for (int j = 0; j < 10; j++) {
        if (p.get(j) > mx) {
          mx = p.get(j);
          p1 = j;
        }
        if (yTest.get(i).get(j) == 1.0)
          p2 = j;
      }
      if (p1 == p2) cnt++;
      //System.out.println(mx + " " + p1 + " : " + p2);
    }
    System.err.println(cnt + " / " + xTest.size());
    SerializationUtils.save("MNIST.obj", m);

  }
}