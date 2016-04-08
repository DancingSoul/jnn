package cn.edu.hit.ir.JNN.Trainer;

import java.util.Vector;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import cn.edu.hit.ir.JNN.LookupParameters;
import cn.edu.hit.ir.JNN.Model;
import cn.edu.hit.ir.JNN.Parameters;

public class SimpleSGDTrainer extends AbstractTrainer {


  SimpleSGDTrainer() {
  }

  public SimpleSGDTrainer(Model m) {
    this(m, 0.00000001, 0.1);
  }

  public SimpleSGDTrainer(Model m, double lam, double e0) {
    super(m, lam, e0);
  }

  public void update() {
    update(1.0);
  }

  public void update(double scale) {
    update(model.lookupParametersList(), model.paramtersList(), scale);
  }

  public void update(final Vector<LookupParameters> lookupParams,
                     final Vector<Parameters> params, double scale) {
    double gscale = clipGradients();
    for (Parameters p : params) {
      DenseMatrix64F reg = new DenseMatrix64F(p.values.v);
      CommonOps.scale(lambda, reg);
      DenseMatrix64F tmp = new DenseMatrix64F(p.g.v);
      CommonOps.scale(eta * scale * gscale, tmp);
      CommonOps.addEquals(tmp, reg);
      CommonOps.subtractEquals(p.values.v, tmp);
      p.clear();
    }
    for (LookupParameters p : lookupParams) {
      for (Integer i : p.nonZeroGrads) {
        DenseMatrix64F reg = new DenseMatrix64F(p.values.get(i).v);
        CommonOps.scale(lambda, reg);
        DenseMatrix64F tmp = new DenseMatrix64F(p.grads.get(i).v);
        CommonOps.scale(eta * scale * gscale, tmp);
        CommonOps.addEquals(tmp, reg);
        CommonOps.subtractEquals(p.values.get(i).v, tmp);
      }
      p.clear();
    }
    ++updates;
  }
}
