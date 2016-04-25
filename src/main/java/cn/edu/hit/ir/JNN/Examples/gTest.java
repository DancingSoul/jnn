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

public class gTest {
  public static void main(String args[]) {
    INDArray a = Nd4j.zeros(2, 3);
    INDArray b = Nd4j.zeros(3, 2).addi(1.0);
    //INDArray c = a.add(b);
    //System.out.println(c.getDouble(1));
    INDArray d = a.getColumn(1);
    System.out.println(d.size(0) + " " + d.size(1));
  }
  
  
}
