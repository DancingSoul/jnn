package cn.edu.hit.ir.JNN.Nodes;

import cn.edu.hit.ir.JNN.ComputationGraph;
import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Expression;
import cn.edu.hit.ir.JNN.Model;
import cn.edu.hit.ir.JNN.Trainers.SimpleSGDTrainer;
import javassist.compiler.ast.Expr;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.Vector;

/**
 * Created by dancingsoul on 2016/5/19.
 */
public class ComplexNN {
  @Test
  public void testGradient() throws Exception {
    Model m = new Model();
    SimpleSGDTrainer sgd = new SimpleSGDTrainer(m);
    ComputationGraph cg = new ComputationGraph();

    Vector<Double> xV = new Vector<Double>(Arrays.asList(1.0, 2.0));
    Expression W = Expression.Creator.parameter(cg, m.addParameters(Dim.create(100, 2)));
    Expression W2 = Expression.Creator.parameter(cg, m.addParameters(Dim.create(100, 100)));
    Expression W3 = Expression.Creator.parameter(cg, m.addParameters(Dim.create(100, 100)));
    Expression W4 = Expression.Creator.parameter(cg, m.addParameters(Dim.create(100, 100)));

    Expression B1 = Expression.Creator.parameter(cg, m.addParameters(Dim.create(100)));
    Expression B2 = Expression.Creator.parameter(cg, m.addParameters(Dim.create(100)));
    Expression B3 = Expression.Creator.parameter(cg, m.addParameters(Dim.create(100)));
    Expression x = Expression.Creator.input(cg, Dim.create(2, 1), xV);

    Expression A = Expression.Creator.logistic(Expression.Creator.affineTransform(Arrays.asList(B1, W, x)));

    //Expression A = Expression.Creator.logistic(Expression.Creator.add(B1, Expression.Creator.multiply(W, x)));
    Expression A1 = Expression.Creator.tanh(Expression.Creator.affineTransform(Arrays.asList(B1, W2, A)));
    Expression A2 = Expression.Creator.logistic(Expression.Creator.affineTransform(Arrays.asList(B2, W3, A)));
    Expression A3 = Expression.Creator.tanh(Expression.Creator.affineTransform(Arrays.asList(B3, W4, A)));

    Expression loss = Expression.Creator.pickNegLogSoftmax(A, new Vector<Integer>(Arrays.asList(0)));

    cg.gradientCheck();
    cg.forward();
    cg.backward();
    Assert.assertEquals(true, m.gradientCheck());
  }
}
