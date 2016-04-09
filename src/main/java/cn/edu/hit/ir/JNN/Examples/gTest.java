package cn.edu.hit.ir.JNN.Examples;

import cn.edu.hit.ir.JNN.AtomicDouble;
import cn.edu.hit.ir.JNN.ComputationGraph;
import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Expression;
import cn.edu.hit.ir.JNN.Model;
import cn.edu.hit.ir.JNN.Trainer.SimpleSGDTrainer;

public class gTest {
  public static void main(String args[]) {
    Model m = new Model();
    SimpleSGDTrainer sgd = new SimpleSGDTrainer(m);
    ComputationGraph cg = new ComputationGraph();
    
    AtomicDouble xValues = new AtomicDouble(1.0);
    AtomicDouble yValues = new AtomicDouble(1.0);

    Expression x = Expression.Creator.input(cg, xValues);
    Expression y = Expression.Creator.input(cg, xValues);
    
    Expression a = Expression.Creator.parameter(cg, m.addParameters(Dim.create(1, 1)));
    Expression b = Expression.Creator.multiply(a, x);
    
    Expression loss = Expression.Creator.squaredDistance(b, y);
    
    
    cg.gradientCheck();
    cg.forward();
    cg.backward();
    System.out.println(m.gradientCheck());
    
    
    
    
  }
  
  
}
