package cn.edu.hit.ir.JNN;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

/**
 * Created by yijia_liu on 3/29/2016.
 */
public class ParametersTest {
  @Test
  public void testCreateParameters() {
    Parameters param = new Parameters(new Dim(Arrays.asList(3, 4), 1), 10);
    Assert.assertEquals(param.dim.at(0), 3);
    Assert.assertEquals(param.dim.at(1), 4);
    System.out.println(param.values.v);
    System.out.println(param.g.v);
  }

  @Test
  public void scaleParameters() throws Exception {

  }

  @Test
  public void squaredL2norm() throws Exception {

  }

  @Test
  public void gSquaredL2norm() throws Exception {

  }

  @Test
  public void size() throws Exception {

  }

  @Test
  public void copy() throws Exception {

  }

  @Test
  public void accumulateGrad() throws Exception {

  }

  @Test
  public void clear() throws Exception {

  }
}
