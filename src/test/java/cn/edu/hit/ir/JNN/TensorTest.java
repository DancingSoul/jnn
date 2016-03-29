package cn.edu.hit.ir.JNN;

import org.ejml.data.DenseMatrix64F;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * Created by yijia_liu on 3/29/2016.
 */
public class TensorTest {
  @Test
  public void testCreate() throws Exception {
    Tensor d = new Tensor(new Dim(Arrays.asList(3, 4), 1));
    // In this way, the DenseMatrix only create memory
    System.out.println(d.d);
    System.out.println(d.v);
  }

  @Test
  public void testCreate1() throws Exception {
    Tensor d = new Tensor(new Dim(Arrays.asList(3, 4), 1), new DenseMatrix64F(3, 4));
    // In this way, the DenseMatrix is created with
    System.out.println(d.d);
    System.out.println(d.v);
  }

  @Test
  public void isValid() throws Exception {

  }

  @Test
  public void toScalar() throws Exception {

  }
}