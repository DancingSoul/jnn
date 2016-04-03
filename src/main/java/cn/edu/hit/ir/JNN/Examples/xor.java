package cn.edu.hit.ir.JNN.Examples;

import java.util.Arrays;
import java.util.Vector;

import cn.edu.hit.ir.JNN.AtomicDouble;
import cn.edu.hit.ir.JNN.ComputationGraph;
import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Expression;
import cn.edu.hit.ir.JNN.ExpressionUtils;
import cn.edu.hit.ir.JNN.Model;
import cn.edu.hit.ir.JNN.TensorUtils;
import cn.edu.hit.ir.JNN.Trainer.SimpleSGDTrainer;

public class xor {
  public static void main(String args[]){
    final int HIDDEN_SIZE = 8;
    final int ITERATIONS = 30;
    Model m = new Model();
    SimpleSGDTrainer sgd = new SimpleSGDTrainer(m);
    ComputationGraph cg = new ComputationGraph();
    
    Expression W = ExpressionUtils.parameter(cg, m.addParameters(
        new Dim(Arrays.asList(HIDDEN_SIZE, 2))));
    Expression b = ExpressionUtils.parameter(cg, m.addParameters(new Dim(Arrays.asList(HIDDEN_SIZE))));
    Expression V = ExpressionUtils.parameter(cg, m.addParameters(new Dim(Arrays.asList(1, HIDDEN_SIZE))));
    Expression a = ExpressionUtils.parameter(cg, m.addParameters(new Dim(Arrays.asList(1))));
    
    Vector <Double> xValues = new Vector <Double> (2);
    Expression x = ExpressionUtils.input(cg, new Dim(Arrays.asList(2)), xValues);
    AtomicDouble yValue = new AtomicDouble();
    Expression y = ExpressionUtils.input(cg, yValue);
    Expression h = ExpressionUtils.tanh(
        ExpressionUtils.add(ExpressionUtils.multiply(W, x), b));
    Expression y_pred = ExpressionUtils.add(ExpressionUtils.multiply(V, h), a);
    Expression loss = ExpressionUtils.squaredDistance(y_pred, y);
    
    for (int iter = 0; iter < ITERATIONS; ++iter) {
      double loss_ = 0.0;
      for (int mi = 0; mi < 4; ++mi) {
        int x1 = mi % 2;
        int x2 = (mi / 2) % 2;
        xValues.set(0, x1 == 1 ? 1.0 : -1.0);
        xValues.set(1, x2 == 1 ? 1.0 : -1.0);
        yValue.set((x1 != x2) ? 1.0 : -1.0);
        loss_ += TensorUtils.toScalar(cg.forward());
        cg.backward();
        sgd.update(1.0);
      }
      sgd.updateEpoch();;
      loss_ /= 4.0;
      System.out.println("E = " + loss_);
    }
  }
}
