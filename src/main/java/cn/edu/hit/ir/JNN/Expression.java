package cn.edu.hit.ir.JNN;

import java.util.Arrays;
import java.util.List;
import java.util.Vector;

import cn.edu.hit.ir.JNN.Nodes.AffineTransform;
import cn.edu.hit.ir.JNN.Nodes.ConstantMinusX;
import cn.edu.hit.ir.JNN.Nodes.CwiseMultiply;
import cn.edu.hit.ir.JNN.Nodes.Dropout;
import cn.edu.hit.ir.JNN.Nodes.LogisticSigmoid;
import cn.edu.hit.ir.JNN.Nodes.MatrixMultiply;
import cn.edu.hit.ir.JNN.Nodes.SquaredEuclideanDistance;
import cn.edu.hit.ir.JNN.Nodes.Sum;
import cn.edu.hit.ir.JNN.Nodes.Tanh;

public class Expression {
  public ComputationGraph pg;
  public int i;

  public Expression() {
    pg = null;
  }

  Expression(ComputationGraph pg_, int i_) {
    pg = pg_;
    i = i_;
  }

  public Tensor value() {
    return pg.getValue(i);
  }

  public static class Creator {
    public static Expression input(ComputationGraph g, AtomicDouble s) {
      return new Expression(g, g.addInput(s));
    }

    public static Expression input(ComputationGraph g, final Dim d, final Vector<Double> data) {
      return new Expression(g, g.addInput(d, data));
    }

    public static Expression parameter(ComputationGraph g, Parameters p) {
      return new Expression(g, g.addParameters(p));
    }

    public static Expression constParameter(ComputationGraph g, Parameters p) {
      return new Expression(g, g.addConstParameters(p));
    }

//    public static Expression lookup(ComputationGraph g, LookupParameters p, AtomicInteger index) {
//      return new Expression(g, g.addLookup(p, index));
//    }
//
//    public static Expression constLookup(ComputationGraph g, LookupParameters p, AtomicInteger index) {
//      return new Expression(g, g.addConstLookup(p, index));
//    }

    public static Expression lookup(ComputationGraph g, LookupParameters p,
                                    final Vector <Integer> indices) {
      return new Expression(g, g.addLookup(p, indices));
    }

    public static Expression constLookup(ComputationGraph g, LookupParameters p,
                                         final Vector <Integer> indices) {
      return new Expression(g, g.addConstLookup(p, indices));
    }
  /*public Expression zeroes(ComputationGraph g, final Dim d) {
    return new Expression(g, g.add(Zeroes, d));
  }*/

    public static Expression multiply(final Expression x, final Expression y) {
      return new Expression(x.pg, x.pg.addFunction(new MatrixMultiply(Arrays.asList(x.i, y.i))));
    }

    public static Expression add(final Expression x, final Expression y) {
      return new Expression(x.pg, x.pg.addFunction(new Sum(Arrays.asList(x.i, y.i))));
    }

    public static Expression tanh(final Expression x) {
      return new Expression(x.pg, x.pg.addFunction(new Tanh(Arrays.asList(x.i))));
    }

    public static Expression logistic(final Expression x) {
      return new Expression(x.pg, x.pg.addFunction(new LogisticSigmoid(Arrays.asList(x.i))));
    }
    
    
    public static Expression squaredDistance(final Expression x, final Expression y) {
      return new Expression(x.pg, x.pg.addFunction(
          new SquaredEuclideanDistance(Arrays.asList(x.i, y.i))));
    }
    public static Expression cwiseMultiply(final Expression x, final Expression y) {
      return new Expression(x.pg, x.pg.addFunction(
          new CwiseMultiply(Arrays.asList(x.i, y.i))));
    }
    public static Expression dropout(final Expression x, double p) {
      return new Expression(x.pg, x.pg.addFunction(
          new Dropout(Arrays.asList(x.i), p)));
    }
    public static Expression constantMinusX(double p, Expression x) {
      return new Expression(x.pg, x.pg.addFunction(
          new ConstantMinusX(Arrays.asList(x.i), p)));
    }
    public static Expression affineTransform(final List<Expression> xs) {
      assert(xs.size() == 0);
      return new Expression(xs.get(0).pg, xs.get(0).pg.addFunction(
          new AffineTransform(getIndices(xs))));
    }
    
    public static Vector <Integer> getIndices(List <Expression> xs) {
      Vector <Integer> res = new Vector<Integer>(xs.size());
      for (int i = 0; i < xs.size(); ++i) {
        res.set(i, xs.get(i).i);
      }
      return res;
    }
  }
}
