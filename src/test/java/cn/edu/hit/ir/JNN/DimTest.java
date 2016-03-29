package cn.edu.hit.ir.JNN;

import static org.junit.Assert.assertEquals;
import org.junit.Test;
import java.util.Arrays;

/**
 * Created by yijia_liu on 3/29/2016.
 */
public class DimTest {
  @Test
  public void testCreate1()throws Exception  {
    Dim dim = new Dim();
    assertEquals("Empty Dim, #dim: ", 0, dim.nd);
    assertEquals("Empty Dim, #batch: ", 1, dim.bd);
  }

  @Test
  public void testCreate2() throws Exception {
    // create a dim of (100, 200)
    Dim dim = new Dim(Arrays.asList(100, 200), 1);
    assertEquals("(100, 200) Dim #dim: ", 2, dim.nd);
    assertEquals("(100, 200) Dim #dim[0]: ", 100, dim.d[0]);
    assertEquals("(100, 200) Dim #dim[1]: ", 200, dim.d[1]);

    // copy a dim of (100, 200)
    Dim dim2 = new Dim(dim);
    assertEquals("(100, 200) Dim2 #dim: ", 2, dim.nd);
    assertEquals("(100, 200) Dim2 #dim[0]: ", 100, dim.d[0]);
    assertEquals("(100, 200) Dim2 #dim[1]: ", 200, dim.d[1]);
  }
}