package cn.edu.hit.ir.JNN;

import org.junit.Test;
import org.junit.Assert;

import java.util.Arrays;

/**
 * Created by yijia_liu on 3/29/2016.
 */
public class LookupParametersTest {

  @Test
  public void testCreate() {
    LookupParameters param = new LookupParameters(10, new Dim(Arrays.asList(10, 1), 1));
    System.out.println(param);
  }

  @Test
  public void scaleParameters() throws Exception {
    Assert.assertEquals(true, true);
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
  public void initialize() throws Exception {

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