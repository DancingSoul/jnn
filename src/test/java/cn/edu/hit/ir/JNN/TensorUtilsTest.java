package cn.edu.hit.ir.JNN;

import java.util.Arrays;
import org.junit.Test;
import org.junit.Assert;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by yijia_liu on 3/29/2016.
 */
public class TensorUtilsTest {

  @Test
  public void testConstant() throws Exception {
    Tensor d = new Tensor(new Dim(Arrays.asList(100, 200), 1), Nd4j.zeros(100, 200));
    TensorUtils.constant(d, 63d);
    // value of the tensor should be close to 63
    Assert.assertEquals(63d, d.v.getDouble(20, 10), 1e-8);
  }

  @Test
  public void testZero() throws Exception {
    Tensor d = new Tensor(new Dim(Arrays.asList(100, 200), 1), Nd4j.zeros(100, 200));
    TensorUtils.zero(d);
    Assert.assertEquals(0d, d.v.getDouble(20, 10), 1e-8);
  }

  @Test
  public void testRandomize() throws Exception {
    Tensor d = new Tensor(new Dim(Arrays.asList(3, 4), 1), Nd4j.zeros(3, 4));
    System.out.println(d.v);
    TensorUtils.randomize(d);
    System.out.println(d.v);
  }

  @Test
  public void testRandomize1() throws Exception {
    // an empty tensor
    //Tensor d = new Tensor();
    //TensorUtils.randomize(d);
  }

  @Test
  public void testRandomBernoulli() throws Exception {

  }

  @Test
  public void testRandomizeNormal() throws Exception {

  }

  @Test
  public void testAccessElement() throws Exception {

  }

  @Test
  public void testAccessElement1() throws Exception {
    Tensor d = new Tensor(new Dim(Arrays.asList(100, 200), 1), Nd4j.zeros(100, 200));
    TensorUtils.constant(d, 63d);
    Assert.assertEquals(63d, TensorUtils.accessElement(d, new Dim(Arrays.asList(99, 199), 1)), 1e-8);
    // following test is not adoptable because counting from 0
    // assertEquals(63d, TensorUtils.accessElement(d, new Dim(Arrays.asList(100, 200), 1)), 1e-8);
  }

  @Test
  public void testSetElement() throws Exception {
    Tensor d = new Tensor(new Dim(Arrays.asList(100, 200), 1), Nd4j.zeros(100, 200));
    TensorUtils.zero(d);

    TensorUtils.setElement(d, 19, 63d);
    Assert.assertEquals(63d, d.v.getDouble(19), 1e-8);

    TensorUtils.setElement(d, 0, 63d);
    Assert.assertEquals(63d, d.v.getDouble(0, 0), 1e-8);

    TensorUtils.setElement(d, 1, 63d);
    Assert.assertEquals(63d, d.v.getDouble(0, 1), 1e-8);
  }

  @Test
  public void testSetElements() throws Exception {

  }

  @Test
  public void testCopyElements() throws Exception {
    Tensor d = new Tensor(new Dim(Arrays.asList(3, 4), 1), Nd4j.zeros(3 ,4));
    TensorUtils.constant(d, 10);

    Tensor d2 = new Tensor(new Dim(Arrays.asList(3, 4), 1), Nd4j.zeros(3 ,4));
    int code = System.identityHashCode(d2.v);
    TensorUtils.copyElements(d2, d);

    // copy should be performed.
    Assert.assertEquals(10d, d.v.getDouble(0), 1e-8);
    // copy should be done in place.
    Assert.assertEquals(code, System.identityHashCode(d2.v));
  }
}