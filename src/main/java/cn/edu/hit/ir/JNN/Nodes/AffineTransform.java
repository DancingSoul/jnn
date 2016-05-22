package cn.edu.hit.ir.JNN.Nodes;

import java.util.List;
import java.util.Vector;


import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;

public class AffineTransform extends Node {
  public AffineTransform(List<Integer> x) {
    super(x);
  }

  public String asString(final Vector<String> argNames) {
    StringBuilder sb = new StringBuilder(argNames.get(0));
    for (int i = 1; i < argNames.size(); i += 2) {
      sb.append(" + ").append(argNames.get(i)).append(" * ").append(argNames.get(i + 1));
    }
    return sb.toString();
  }

  @Override
  public Dim dimForward(final Vector<Dim> xs) {
    if ((xs.size() - 1) % 2 != 0) {
      StringBuilder s = new StringBuilder(
              "Bad number of inputs for AffineTransform: ");
      throw new IllegalArgumentException(s.toString());
    }
    Dim d = xs.get(0);
    for (int i = 1; i < xs.size(); i += 2) {
      if (xs.get(i).getNumCols() != xs.get(i + 1).getNumRows() ||
              xs.get(0).getNumRows() != xs.get(i).getNumRows() ||
              xs.get(0).getNumCols() != xs.get(i + 1).getNumCols()) {
        StringBuilder s = new StringBuilder(
                "Bad dimensions for AffineTransform: ");
        throw new IllegalArgumentException(s.toString());
      }
      d.bd = Math.max(Math.max(d.bd, xs.get(i).bd), xs.get(i + 1).bd);
    }
    return d;
  }

  @Override
  public boolean supportsMultibatch() {
    return true;
  }

  @Override
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    assert(xs.size() % 2 == 1);
    if (xs.size() == 1) {
      fx.v = xs.get(0).v;
      return;
    } else {
      // Add, using broadcasting or not
      if (fx.d.bd > 1 && xs.get(0).d.bd == 1) {
        for (int i = 0; i < fx.v.size(1); i++) {
          fx.rowcolMatrix().getColumn(i).assign(xs.get(0).vec().transpose());
        }
      } else {
        if (fx.d.bd == 1) {
          fx.v.assign(xs.get(0).v);
        } else {
          for (int b = 0; b < fx.d.bd; ++b) {
            fx.setBatchMatrix(b, xs.get(0).getBatchMatrix(b));
          }
        }
      }
      //Multiply
      for (int i = 1; i < xs.size(); i += 2) {
        if (xs.get(i).d.bd == 1 && xs.get(i + 1).d.bd == fx.d.bd) {
          if (xs.get(i + 1).d.bd == 1) {
            fx.v.addi(xs.get(i).v.mmul(xs.get(i + 1).v));
          } else {
            fx.v.addi(xs.get(i).getBatchMatrix(0).mmul(xs.get(i + 1).colbatchMatrix()).reshape(fx.d.size(), 1));
          }
        } else {
          assert(xs.get(i + 1).d.bd == 1 || xs.get(i + 1).d.bd == xs.get(i).d.bd);
          for (int b = 0; b < fx.d.bd; ++b) {
            fx.addBatchMatrix(b, xs.get(i).getBatchMatrix(b).mmul(xs.get(i + 1).getBatchMatrix(b)));
          }
        }
      }
    }
  }

  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
    assert(i < xs.size());
    if (i == 0) {
      dEdxi.v.addi(dEdf.v);
    } else if (i % 2 == 1){
      int maxB = Math.max(dEdf.d.bd, xs.get(i + 1).d.bd);
      if (maxB == 1) {
        dEdxi.v.addi(dEdf.v.mmul(xs.get(i + 1).v.transpose()));
      } else {
        for (int b = 0; b < maxB; ++b)
          dEdxi.getBatchMatrix(b).addi(dEdf.getBatchMatrix(b).mmul(xs.get(i + 1).getBatchMatrix(b).transpose()));
      }
    } else {
      int maxB = Math.max(xs.get(i - 1).d.bd, dEdf.d.bd);
      if (xs.get(i - 1).d.bd == 1 && dEdxi.d.bd == dEdf.d.bd) {
        if (dEdf.d.bd == 1) {
          dEdxi.v.addi(xs.get(i - 1).v.transpose().mmul(dEdf.v));
        } else {
          dEdxi.colbatchMatrix().addi(xs.get(i - 1).getBatchMatrix(0).transpose().mmul(dEdf.colbatchMatrix()));
        }
      } else {
        if (maxB == 1) {
          dEdxi.v = xs.get(i - 1).v.transpose().mmul(dEdf.v);
        } else {
          for (int b = 0; b < maxB; ++b)
            dEdxi.getBatchMatrix(b).addi(xs.get(i - 1).getBatchMatrix(b).transpose().mmul(dEdf.getBatchMatrix(b)));
        }
      }
    }
  }
}