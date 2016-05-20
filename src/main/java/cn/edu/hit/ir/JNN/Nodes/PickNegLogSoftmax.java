package cn.edu.hit.ir.JNN.Nodes;

import cn.edu.hit.ir.JNN.Dim;
import cn.edu.hit.ir.JNN.Tensor;
import cn.edu.hit.ir.JNN.Utils.TensorUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Vector;

/**
 * Created by dancingsoul on 2016/5/4.
 */
public class PickNegLogSoftmax extends Node{
  public Vector<Integer> vals;
  public double[] logz;
  public PickNegLogSoftmax(final List<Integer> a, Vector<Integer> vals_) {
    super(a);
    vals = vals_;
  }
  @Override
  public String asString(final Vector<String> argNames) {return "";}

  @Override
  public Dim dimForward(final Vector<Dim> xs) {
    assert(xs.size() == 1);
    /*if (!LooksLikeVector(xs.get(0))) {
      throw new RuntimeException("Bad Input Dimensions in PickNegLogSoftmax");
    }*/
    return Dim.createBatches(1, 1, xs.get(0).bd);
  }

  public boolean supportsMultibatch() {
    return true;
  }


  private double logSumExp(INDArray x) {
    double sum = 0.0;
    double m = x.maxNumber().doubleValue();
    for (int i = 0; i < x.length(); ++i) {
      sum += Math.exp(x.getDouble(i) - m);
    }
    return Math.log(sum) + m;
  }
  @Override
  public void forwardImpl(final Vector<Tensor> xs, Tensor fx) {
    if (xs.get(0).d.getNumCols() == 1) {
      logz = new double[fx.d.getNumBatchElements()];
      if (vals.size() == 1) {
        INDArray x = xs.get(0).v;
        logz[0] = logSumExp(x);
        fx.v.putScalar(0, logz[0] - x.getDouble(vals.get(0)));
      } else {
        assert(vals.size() == fx.d.getNumBatchElements());
        for (int b = 0; b < vals.size(); ++b) {
          INDArray x = xs.get(0).getBatchMatrix(b).reshape(xs.get(0).d.batchSize(), 1);
          logz[b] = logSumExp(x);
          fx.v.putScalar(b, logz[b] - x.getDouble(vals.get(b)));
        }
      }
    } else {
      throw new RuntimeException("PickNegLogSoftmax :: forward not yet implemented for multiple columns");
    }
  }
  @Override
  public void backwardImpl(final Vector<Tensor> xs,
                           final Tensor fx, final Tensor dEdf, int i_, Tensor dEdxi) {
    if (xs.get(0).d.getNumCols() == 1) {
      if (vals.size() == 1) {
        int elem = vals.get(0);
        double err = dEdf.v.getDouble(0);
        INDArray x = xs.get(0).v;
        for (int i = 0; i < x.length(); i++)
          dEdxi.v.putScalar(i, Math.exp(x.getDouble(i) - logz[0]) * err + dEdxi.v.getDouble(i));
        dEdxi.v.putScalar(elem, dEdxi.v.getDouble(elem) - err);
      } else {
        assert(vals.size() == fx.d.getNumBatchElements());
        //TODO
      }
    } else {
      throw new RuntimeException("PickNegLogSoftmax :: forward not yet implemented for multiple columns");
    }
  }
}
