package cn.edu.hit.ir.JNN.Nodes;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Parameters;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

public class ParameterNodeTest {
  @Test
  public void testCreate() throws Exception {
    Parameters param = new Parameters(new Dim(Arrays.asList(1, 2)), 1.);
    Node node = new ParameterNode(param);
    ParameterNode parameterNode = new ParameterNode(param);
    System.out.println(node.dim);
    System.out.println(parameterNode.dim);
  }

  @Test
  public void dimForward() throws Exception {

  }

  @Test
  public void asString() throws Exception {

  }

  @Test
  public void forwardImpl() throws Exception {

  }

  @Test
  public void backwardImpl() throws Exception {

  }

  @Test
  public void accumulateGrad() throws Exception {

  }
}