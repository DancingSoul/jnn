package cn.edu.hit.ir.JNN.Trainers;

import cn.edu.hit.ir.JNN.*;
import cn.edu.hit.ir.JNN.Utils.ShadowUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Vector;

/**
 * Created by dancingsoul on 2016/4/27.
 */
public class AdagradTrainer extends AbstractTrainer {
  public double epsilon;
  public boolean shadowParamsAllocated;
  public Vector<ShadowParameters> vp;
  public Vector<ShadowLookupParameters> vlp;

  public AdagradTrainer(Model m) {
    this(m, 1E-6, 0.1, 1e-20);
  }

  public AdagradTrainer(Model m, double lam, double e0, double eps) {
    super(m, lam, e0);
    epsilon = eps;
    shadowParamsAllocated = false;
  }

  public void update() {
    this.update(1.0);
  }

  public void update(double scale) {
    //executed on the first iteration to create vectors to store the velocity
    if (!shadowParamsAllocated) {
      vp = ShadowUtils.AllocateShadowParameters(model);
      vlp = ShadowUtils.AllocateShadowLookupParameters(model);
      shadowParamsAllocated = true;
    }
    int pi = 0;
    final double gscale = clipGradients();

    for (Parameters p : model.parametersList()) {
      Tensor v = vp.get(pi++).h;
      INDArray reg = p.values.v.mul(lambda);
      INDArray g = p.g.v.mul(scale * gscale);
      INDArray g2 = g.mul(g);
      v.v.addi(g2);
      //TODO
    }
    pi = 0;
    for (LookupParameters p : model.lookupParametersList()) {
      Vector<Tensor> vx = vlp.get(pi++).h;
      for (Integer i : p.nonZeroGrads) {
        //TODO
      }
    }
  }
}