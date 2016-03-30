package cn.edu.hit.ir.JNN;

import java.util.Arrays;
import org.junit.Test;
import org.junit.Assert;

/**
 * Created by yijia_liu on 3/29/2016.
 */
public class DimTest {
  @Test
  public void testCreate1()throws Exception  {
    Dim dim = new Dim();
    Assert.assertEquals("Empty Dim, #dim: ", 0, dim.nd);
    Assert.assertEquals("Empty Dim, #batch: ", 1, dim.bd);
  }

  @Test
  public void testCreate2() throws Exception {
    // create a dim of (100, 200)
    Dim dim = new Dim(Arrays.asList(100, 200), 1);
    Assert.assertEquals("(100, 200) Dim #dim: ", 2, dim.nd);
    Assert.assertEquals("(100, 200) Dim #dim[0]: ", 100, dim.d[0]);
    Assert.assertEquals("(100, 200) Dim #dim[1]: ", 200, dim.d[1]);

    // copy a dim of (100, 200)
    Dim dim2 = new Dim(dim);
    Assert.assertEquals("(100, 200) Dim2 #dim: ", 2, dim2.nd);
    Assert.assertEquals("(100, 200) Dim2 #dim[0]: ", 100, dim2.d[0]);
    Assert.assertEquals("(100, 200) Dim2 #dim[1]: ", 200, dim2.d[1]);
  }

  @Test
  public void size() throws Exception {
    Dim dim = new Dim(Arrays.asList(100, 200), 1);
    Assert.assertEquals(dim.size(), 100 * 200);

    Dim dim2 = new Dim(Arrays.asList(100, 200), 2);
    Assert.assertEquals(dim2.size(), 100 * 200 * 2);
  }

  @Test
  public void batchSize() throws Exception {
    Dim dim = new Dim(Arrays.asList(100, 200), 1);
    Assert.assertEquals(dim.batchSize(), 100 * 200);
  }

  @Test
  public void getSumDimensions() throws Exception {
    Dim dim = new Dim(Arrays.asList(100, 200), 1);
    Assert.assertEquals(dim.getSumDimensions(), 300);
  }

  @Test
  public void truncate() throws Exception {

  }

  @Test
  public void singleBatch() throws Exception {

  }

  @Test
  public void resize() throws Exception {
  }

  @Test
  public void getNumDimensions() throws Exception {
    Dim dim = new Dim(Arrays.asList(100, 200), 1);
    Assert.assertEquals(dim.getNumDimensions(), 2);
  }

  @Test
  public void getNumRows() throws Exception {
    Dim dim = new Dim(Arrays.asList(100, 200), 1);
    Assert.assertEquals(dim.getNumRows(), 100);
  }

  @Test
  public void getNumCols() throws Exception {
    Dim dim = new Dim(Arrays.asList(100, 200), 1);
    Assert.assertEquals(dim.getNumCols(), 200);
  }

  @Test
  public void getNumBatchElements() throws Exception {
    Dim dim = new Dim(Arrays.asList(100, 200), 2);
    Assert.assertEquals(dim.getNumBatchElements(), 2);
  }

  @Test
  public void set() throws Exception {

  }

  @Test
  public void at() throws Exception {

  }

  @Test
  public void size1() throws Exception {

  }

  @Test
  public void transpose() throws Exception {

  }

  @Test
  public void equals() throws Exception {

  }
}