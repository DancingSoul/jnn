package cn.edu.hit.ir.JNN;

import java.util.Vector;

import cn.edu.hit.ir.JNN.Nodes.AbstractParameterNode;
import cn.edu.hit.ir.JNN.Nodes.Node;
import org.ejml.data.DenseMatrix64F;

class SimpleExecutionEngine extends AbstractExecutionEngine {
  private Vector<Tensor> nfxs;
  private Vector<Tensor> ndEdfs;
  int numNodesEvaluated;

  SimpleExecutionEngine() {
    nfxs = new Vector<Tensor>();
    ndEdfs = new Vector<Tensor>();
    numNodesEvaluated = 0;
  }

  public SimpleExecutionEngine(final ComputationGraph cg_) {
    cg = cg_;
    nfxs = new Vector<Tensor>();
    ndEdfs = new Vector<Tensor>();
    numNodesEvaluated = 0;
  }

  public void invalidate() {
    numNodesEvaluated = 0;
  }

  public Tensor forward() {
    final int nodeMaxIndex = cg.nodes.size() - 1;
    return forward(nodeMaxIndex);
  }

  public Tensor forward(int i) {
    invalidate();
    return incrementalForward(i);
  }

  public Tensor getValue(int i) {
    assert (i < cg.nodes.size());
    if (i >= numNodesEvaluated) {
      incrementalForward();
    }
    return nfxs.get(i);
  }

  public Tensor incrementalForward() {
    final int nodeMaxIndex = cg.nodes.size() - 1;
    return incrementalForward(nodeMaxIndex);
  }

  public Tensor incrementalForward(int i) {
    assert (i < cg.nodes.size());

    if (i >= numNodesEvaluated) {
      nfxs.setSize(i + 1);

      Vector<Tensor> xs = new Vector<Tensor>(16);
      for (; numNodesEvaluated <= i; ++numNodesEvaluated) {
        Node node = cg.nodes.get(numNodesEvaluated);
        xs.setSize(node.arity());
        int ai = 0;
        for (Integer arg : node.args) {
          xs.set(ai, nfxs.get(arg));
          ++ai;
        }
        nfxs.set(numNodesEvaluated, new Tensor(node.dim));
        // nfxs.get(numNodesEvaluated).d = node.dim;
        // nfxs.get(numNodesEvaluated).v = new DenseMatrix64F(node.dim.size());
        node.forward(xs, nfxs.get(numNodesEvaluated));
      }
    }
    return nfxs.get(i);
  }

  public void backward() {
    assert (nfxs.size() == cg.nodes.size());
    backward(cg.nodes.size() - 1);
  }

  public void backward(int fromWhere) {
    assert (fromWhere + 1 <= nfxs.size());
    assert (fromWhere + 1 <= cg.nodes.size());
    if (nfxs.get(fromWhere).d.size() != 1) {
      throw new RuntimeException(
          "called backward() on non-scalar node");
    }

    int numNodes = fromWhere + 1;
    ndEdfs.setSize(numNodes);
    for (int i = 0; i < numNodes; ++i) {
      Dim dim = nfxs.get(i).d;
      ndEdfs.set(i, new Tensor(dim));
      // ndEdfs.get(i).d = dim;
      // ndEdfs.get(i).v = new DenseMatrix64F(dim.size());
    }
    ndEdfs.lastElement().v = new DenseMatrix64F(1, 1);
    ndEdfs.lastElement().v.set(0, 1);

    Vector<Boolean> needsDerivative = new Vector<Boolean>(numNodes);
    needsDerivative.setSize(numNodes);
    for (int i = 0; i < numNodes; ++i)
      needsDerivative.set(i, false);
    for (Integer i : cg.parameterNodes)
      needsDerivative.set(i, true);

    for (int ni = 0; ni < numNodes; ++ni) {
      Boolean nd = needsDerivative.get(ni);
      for (Integer arg : cg.nodes.get(ni).args)
        nd |= needsDerivative.get(arg);
      needsDerivative.set(ni, nd);
    }

    Vector<Boolean> inComputation = new Vector<Boolean>(numNodes);
    inComputation.setSize(numNodes);
    for (int i = 0; i < numNodes; ++i) {
      inComputation.set(i, false);
    }
    inComputation.set(numNodes - 1, true);
    Vector<Tensor> xs = new Vector<Tensor>();

    for (int i = numNodes - 1; i >= 0; --i) {
      if (!inComputation.get(i)) continue;
      Node node = cg.nodes.get(i);
      xs.setSize(node.arity());
      int ai = 0;
      for (Integer arg : node.args) {
        inComputation.set(arg, true);
        xs.set(ai, nfxs.get(arg));
        ++ai;
      }
      ai = 0;
      for (Integer arg : node.args) {
        if (needsDerivative.get(arg)) {
          node.backward(xs, nfxs.get(i), ndEdfs.get(i), ai, ndEdfs.get(arg));
        }
        ++ai;
      }
    }

    for (Integer i : cg.parameterNodes) {
      ((AbstractParameterNode)(cg.nodes.get(i))).accumulateGrad(ndEdfs.get(i));
    }
  }
}
