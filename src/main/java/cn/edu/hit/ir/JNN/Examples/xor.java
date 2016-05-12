package cn.edu.hit.ir.JNN.Examples;

import java.util.Vector;

import cn.edu.hit.ir.JNN.AtomicDouble;
import cn.edu.hit.ir.JNN.ComputationGraph;
import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Expression;
import cn.edu.hit.ir.JNN.Model;
import cn.edu.hit.ir.JNN.Utils.SerializationUtils;
import cn.edu.hit.ir.JNN.Utils.TensorUtils;
import cn.edu.hit.ir.JNN.Trainers.SimpleSGDTrainer;
import org.nd4j.linalg.api.ndarray.INDArray;

public class xor {
  public static void main(String args[]){
    final int HIDDEN_SIZE = 8;
    final int ITERATIONS = 30;

    Model m = new Model();
    SimpleSGDTrainer sgd = new SimpleSGDTrainer(m);
    ComputationGraph cg = new ComputationGraph();



    Expression W = Expression.Creator.parameter(cg, m.addParameters(Dim.create(HIDDEN_SIZE, 2)));
    Expression b = Expression.Creator.parameter(cg, m.addParameters(Dim.create(HIDDEN_SIZE)));
    Expression V = Expression.Creator.parameter(cg, m.addParameters(Dim.create(1, HIDDEN_SIZE)));
    Expression a = Expression.Creator.parameter(cg, m.addParameters(Dim.create(1)));

    //SerializationUtils.loadModel("xor.obj", m);

    Vector<Double> xValues = new Vector<Double>(2);
    
    Expression x = Expression.Creator.input(cg, Dim.create(2), xValues);
    
    AtomicDouble yValue = new AtomicDouble();
    Expression y = Expression.Creator.input(cg, yValue);
    
    Expression h = Expression.Creator.tanh(
        Expression.Creator.add(Expression.Creator.multiply(W, x), b));
    Expression yPredict = Expression.Creator.add(Expression.Creator.multiply(V, h), a);
    Expression loss = Expression.Creator.squaredDistance(yPredict, y);


    xValues.setSize(2);
    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
      double lossIter = 0.0;
      for (int mi = 0; mi < 4; ++mi) {
        int x1 = mi % 2;
        int x2 = (mi / 2) % 2;
        xValues.set(0, x1 == 1 ? 1.0 : -1.0);
        xValues.set(1, x2 == 1 ? 1.0 : -1.0);
        yValue.set((x1 != x2) ? 1.0 : -1.0);
        cg.gradientCheck();
        lossIter += TensorUtils.toScalar(cg.forward());
        cg.backward();
        System.out.println(m.gradientCheck());
        sgd.update(1.0);
      }
      sgd.updateEpoch();
      lossIter /= 4.0;
      System.out.println("E = " + lossIter);
    }
    //SerializationUtils.save("xor.obj", m);
  }
}
