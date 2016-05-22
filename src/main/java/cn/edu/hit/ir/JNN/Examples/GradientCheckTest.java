package cn.edu.hit.ir.JNN.Examples;

import cn.edu.hit.ir.JNN.AtomicDouble;
import cn.edu.hit.ir.JNN.ComputationGraph;
import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Expression;
import cn.edu.hit.ir.JNN.Model;
import cn.edu.hit.ir.JNN.Trainers.SimpleSGDTrainer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Date;
import java.util.Vector;

public class GradientCheckTest {
  public static void main(String args[]) {
    INDArray x = Nd4j.zeros(2, 2).addi(1.0);
    INDArray y = Nd4j.zeros(2, 3).addi(2.0);
    x.putScalar(3, 5);
    x.putScalar(0, 3);
    System.err.println(x.ordering());
    INDArray z = x.mmul(y);
    System.err.println(z.ordering());
    for (int i = 0; i < z.length(); ++i)
      System.err.println(z.getDouble(i));
    INDArray t = z.mul(1.0);
    System.err.println(t.ordering());
    for (int i = 0; i < t.length(); ++i)
      System.err.println(t.getDouble(i));
    z.addi(t);
    for (int i = 0; i < z.length(); ++i)
      System.err.println(z.getDouble(i));
  }
}
