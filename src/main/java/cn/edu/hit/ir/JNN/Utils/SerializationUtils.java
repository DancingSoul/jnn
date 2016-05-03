package cn.edu.hit.ir.JNN.Utils;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Model;

import java.io.*;

/**
 * Created by dancingsoul on 2016/4/27.
 */
public class SerializationUtils {
  public static Object load(String fileName) {
    Object obj = new Object();
    try {
      ObjectInputStream oin = new ObjectInputStream(new FileInputStream(
              fileName));
      obj = oin.readObject();
      oin.close();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }
    return obj;
  }
  public static void loadModel(String fileName, Model m) {
    try {
      ObjectInputStream oin = new ObjectInputStream(new FileInputStream(
              fileName));
      Model t = (Model)oin.readObject();
      for (int i = 0; i < m.parametersList().size(); i++)
        m.parametersList().get(i).copy(t.parametersList().get(i));
      for (int i = 0; i < m.lookupParametersList().size(); i++)
        m.lookupParametersList().get(i).copy(t.lookupParametersList().get(i));
      oin.close();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void save(String fileName, Object obj) {
    try {
      ObjectOutputStream out = new ObjectOutputStream(
              new FileOutputStream(fileName));
      out.writeObject(obj);
      out.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
