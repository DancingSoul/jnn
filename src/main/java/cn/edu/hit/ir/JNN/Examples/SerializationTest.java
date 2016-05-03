package cn.edu.hit.ir.JNN.Examples;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Model;
import cn.edu.hit.ir.JNN.Utils.SerializationUtils;

/**
 * Created by dancingsoul on 2016/4/27.
 */
public class SerializationTest {
  public static void main(String args[]) {
    Model m = new Model();
    m.addParameters(Dim.create(1, 2));
    SerializationUtils.save("test.obj", m);
    Model t = (Model)SerializationUtils.load("test.obj");
    System.out.println(t.parametersList().size());
  }
}
