package cn.edu.hit.ir.JNN.Nodes;

import cn.edu.hit.ir.JNN.ComputationGraph;
import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Expression;
import cn.edu.hit.ir.JNN.Model;
import cn.edu.hit.ir.JNN.Trainers.SimpleSGDTrainer;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.Vector;

/**
 * Created by dancingsoul on 2016/5/23.
 */
public class ConcatenateTest {
  @Test
  public void testGradient() throws Exception {
    Model m = new Model();
    SimpleSGDTrainer sgd = new SimpleSGDTrainer(m);
    ComputationGraph cg = new ComputationGraph();

    Vector<Double> x1V = new Vector<Double>(Arrays.asList(1.0, 2.0));
    Vector<Double> x2V = new Vector<Double>(Arrays.asList(1.0, 2.0));
    Vector<Double> yV = new Vector<Double>(Arrays.asList(1.0, 2.0));
    Expression W = Expression.Creator.parameter(cg, m.addParameters(Dim.create(4, 2)));

    Expression x1 = Expression.Creator.input(cg, Dim.create(2, 1), x1V);
    Expression x2 = Expression.Creator.input(cg, Dim.create(2, 1), x2V);
    Expression y = Expression.Creator.input(cg, Dim.create(2, 1), yV);

    Expression t1 = Expression.Creator.multiply(W, x1);
    Expression t2 = Expression.Creator.concatenate(Arrays.asList(x2, y));
    Expression loss = Expression.Creator.squaredDistance(t1, t2);

    cg.gradientCheck();
    cg.forward();
    cg.backward();
    Assert.assertEquals(true, m.gradientCheck());

  }
}
