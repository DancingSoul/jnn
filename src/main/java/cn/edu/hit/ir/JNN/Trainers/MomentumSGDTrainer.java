package cn.edu.hit.ir.JNN.Trainers;

import cn.edu.hit.ir.JNN.*;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Vector;

/**
 * Created by dancingsoul on 2016/4/25.
 */
public class MomentumSGDTrainer extends AbstractTrainer{
  public double momentum;
  public boolean velocityAllocated;
  //the following represent the current velocity
  public Vector<ShadowParameters> vp;
  public Vector<ShadowLookupParameters> vlp;

  public MomentumSGDTrainer(Model m) {
        this(m, 1E-6, 1E-2, 0.9);
    }
  public MomentumSGDTrainer(Model m, double lam, double e0, double mom) {
    super(m, lam, e0);
    momentum = mom;
    velocityAllocated = false;
  }
  public void update() {
        this.update(1.0);
    }
  public void update(double scale) {
    //executed on the first iteration to create vectors to store the velocity
    if (!velocityAllocated) {
      vp = ShadowUtils.AllocateShadowParameters(model);
      vlp = ShadowUtils.AllocateShadowLookupParameters(model);
      velocityAllocated = true;
    }
    final double gscale = clipGradients();
    int pi = 0;
    for (Parameters p : model.paramtersList()) {
      Tensor v = vp.get(pi++).h;
      INDArray reg = p.values.vec().mul(lambda);
      v.v = v.vec().mul(momentum).sub(p.g.vec().mul(eta * scale * gscale));
      p.values.vec().add(v.vec().sub(reg));
      p.clear();
    }
    pi = 0;
    for (LookupParameters p : model.lookupParametersList()) {
      Vector<Tensor> vx = vlp.get(pi++).h;
      for (Integer i : p.nonZeroGrads) {
        Tensor v = vx.get(i);
        INDArray reg = p.values.get(i).vec().mul(lambda);
        v.v = v.vec().mul(momentum).sub(p.grads.get(i).vec().mul(eta * scale * gscale));
        p.values.get(i).vec().add(v.vec().sub(reg));
      }
      p.clear();
    }
  }
}
