package cn.edu.hit.ir.JNN.Examples;

import java.io.File;

import org.ojalgo.array.Array1D;
import org.ojalgo.array.Array2D;
import org.ojalgo.array.BufferArray;
import org.ojalgo.matrix.BasicMatrix;
import org.ojalgo.matrix.BasicMatrix.Factory;
import org.ojalgo.matrix.PrimitiveMatrix;

public class ojAlgoTest {
  public static void main(String args[]) {
    final Factory<PrimitiveMatrix> tmpFactory = PrimitiveMatrix.FACTORY;

    //final Array2D.Factory<Double> tmpFactory2 = Array2D.PRIMITIVE;
    //BasicMatrix t = tmpFactory.makeEye(5, 5);
    
    
    
    final File tmpFile = new File("BasicDemo.array");
    
    
    Array2D<Double> t = BufferArray.make(tmpFile, 5, 5);
    Array1D<Double> t2 = t.asArray1D();
    t.set(0, 1);
    t2.set(0, 2);
    BasicMatrix t3 = tmpFactory.rows(t);
    //t3 = tmpFactory.rows(t2);
    t2.set(0, 3);
    
    //Array2D<Double> t4 = tmpFactory2.copy(t3);
    
    
    t3 = t3.add(0, 0, 100);
    //System.out.println(t4.get(0));
    System.out.println(t3.get(0, 0));
    
  }
}
