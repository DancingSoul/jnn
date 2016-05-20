package cn.edu.hit.ir.JNN.Nodes;

import cn.edu.hit.ir.JNN.*;
import cn.edu.hit.ir.JNN.Trainers.SimpleSGDTrainer;
import java.util.Vector;

import org.apache.commons.math3.analysis.function.Exp;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
/**
 * Created by dancingsoul on 2016/5/12.
 */
public class AffineTransformTest {
  @Test
  public void testGradient() throws Exception {
    Model m = new Model();
    SimpleSGDTrainer sgd = new SimpleSGDTrainer(m);
    ComputationGraph cg = new ComputationGraph();

    Vector<Double> x1V = new Vector<Double>(Arrays.asList(1.0, 2.0));
    Vector<Double> x2V = new Vector<Double>(Arrays.asList(1.0, 2.0));
    Vector<Double> yV = new Vector<Double>(Arrays.asList(1.0, 2.0));
    Expression W = Expression.Creator.parameter(cg, m.addParameters(Dim.create(2, 2)));
    Expression W2 = Expression.Creator.parameter(cg, m.addParameters(Dim.create(2, 2)));
    Expression W3 = Expression.Creator.parameter(cg, m.addParameters(Dim.create(2, 1)));
    Expression x1 = Expression.Creator.input(cg, Dim.create(2, 1), x1V);
    Expression x2 = Expression.Creator.input(cg, Dim.create(2, 1), x2V);
    Expression y = Expression.Creator.input(cg, Dim.create(2, 1), yV);

    Expression AT = Expression.Creator.affineTransform(Arrays.asList(x1, W, x2, W2, W3));
    Expression loss = Expression.Creator.squaredDistance(AT, y);



    cg.gradientCheck();
    cg.forward();
    cg.backward();
    Assert.assertEquals(true, m.gradientCheck());

  }
}
