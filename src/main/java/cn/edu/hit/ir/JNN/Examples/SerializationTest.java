package cn.edu.hit.ir.JNN.Examples;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Model;
import cn.edu.hit.ir.JNN.Utils.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by dancingsoul on 2016/4/27.
 */
public class SerializationTest {
  public static void main(String args[]) {
    INDArray a = Nd4j.zeros(5000, 5000).addi(1.0);
    INDArray b = Nd4j.zeros(1000, 1000);
    b.assign(a);
    b.addi(2.0);
    //a.addi(2.0);
    System.out.println(b.sumNumber().intValue());
  }
}
