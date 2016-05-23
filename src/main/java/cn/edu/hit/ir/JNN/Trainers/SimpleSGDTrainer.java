package cn.edu.hit.ir.JNN.Trainers;

import java.util.Vector;
import cn.edu.hit.ir.JNN.LookupParameters;
import cn.edu.hit.ir.JNN.Model;
import cn.edu.hit.ir.JNN.Parameters;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SimpleSGDTrainer extends AbstractTrainer {
  SimpleSGDTrainer() {
  }

  public SimpleSGDTrainer(Model m) {
    this(m, 0.000001, 0.1);
  }

  public SimpleSGDTrainer(Model m, double lam, double e0) {
    super(m, lam, e0);
  }

  public void update() {
    update(1.0);
  }

  public void update(double scale) {
    update(model.lookupParametersList(), model.parametersList(), scale);
  }

  public void update(final Vector<LookupParameters> lookupParams,
                     final Vector<Parameters> params, double scale) {
    double gscale = clipGradients();
    for (Parameters p : params) {
      p.g.v.muli(eta * scale * gscale);
      p.g.v.addi(p.values.v.mul(lambda));
      p.values.v.subi(p.g.v);
      p.clear();
    }
    for (LookupParameters p : lookupParams) {
      for (Integer i : p.nonZeroGrads) {
        p.grads.get(i).v.muli(eta * scale * gscale);
        p.grads.get(i).v.addi(p.values.get(i).v.mul(lambda));
        p.values.get(i).v.subi(p.grads.get(i).v);
      }
      p.clear();
    }
    ++updates;
  }
}
