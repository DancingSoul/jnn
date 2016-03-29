package test.java.cn.edu.hit.ir.JNN;

import java.util.Arrays;

import org.ejml.data.DenseMatrix64F;

import main.java.cn.edu.hit.ir.JNN.Dim;

/**
 * Created by yijia_liu on 3/29/2016.
 */
public class TensorUtilsTest {

  @Test
  public void constant() throws Exception {
    Tensor d = new Tensor(new Dim(Arrays.asList(100, 200), 1), new DenseMatrix64F(100, 200));
    TensorUtils.constant(d, 63d);
    // value of the tensor should be close to 63
    assertEquals(63d, d.v.get(20, 10), 1e-8);
  }

  @Test
  public void zero() throws Exception {
    Tensor d = new Tensor(new Dim(Arrays.asList(100, 200), 1), new DenseMatrix64F(100, 200));
    TensorUtils.zero(d);
    assertEquals(0d, d.v.get(20, 10), 1e-8);
  }

  @Test
  public void randomize() throws Exception {

  }

  @Test
  public void randomize1() throws Exception {

  }

  @Test
  public void randomBernoulli() throws Exception {

  }

  @Test
  public void randomizeNormal() throws Exception {

  }

  @Test
  public void accessElement() throws Exception {

  }

  @Test
  public void accessElement1() throws Exception {
    Tensor d = new Tensor(new Dim(Arrays.asList(100, 200), 1), new DenseMatrix64F(100, 200));
    TensorUtils.constant(d, 63d);
    assertEquals(63d, TensorUtils.accessElement(d, new Dim(Arrays.asList(99, 199), 1)), 1e-8);
    // following test is not adoptable because counting from 0
    // assertEquals(63d, TensorUtils.accessElement(d, new Dim(Arrays.asList(100, 200), 1)), 1e-8);
  }

  @Test
  public void setElement() throws Exception {
    Tensor d = new Tensor(new Dim(Arrays.asList(100, 200), 1), new DenseMatrix64F(100, 200));
    TensorUtils.zero(d);

    TensorUtils.setElement(d, 19, 63d);
    assertEquals(63d, d.v.get(19), 1e-8);

    TensorUtils.setElement(d, 0, 63d);
    assertEquals(63d, d.v.get(0, 0), 1e-8);

    TensorUtils.setElement(d, 1, 63d);
    assertEquals(63d, d.v.get(0, 1), 1e-8);
  }

  @Test
  public void setElements() throws Exception {

  }

  @Test
  public void copyElements() throws Exception {

  }
}