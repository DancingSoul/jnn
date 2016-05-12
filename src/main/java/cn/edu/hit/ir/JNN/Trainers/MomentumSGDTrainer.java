package cn.edu.hit.ir.JNN.Trainers;

import cn.edu.hit.ir.JNN.*;
import cn.edu.hit.ir.JNN.Utils.ShadowUtils;
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
    for (Parameters p : model.parametersList()) {
      Tensor v = vp.get(pi++).h;
      INDArray reg = p.values.v.mul(lambda);
      v.v = v.v.mul(momentum).sub(p.g.v.mul(eta * scale * gscale));
      p.values.v.addi(v.v.sub(reg));
      p.clear();
    }
    pi = 0;
    for (LookupParameters p : model.lookupParametersList()) {
      Vector<Tensor> vx = vlp.get(pi++).h;
      for (Integer i : p.nonZeroGrads) {
        Tensor v = vx.get(i);
        INDArray reg = p.values.get(i).v.mul(lambda);
        v.v = v.v.mul(momentum).sub(p.grads.get(i).v.mul(eta * scale * gscale));
        p.values.get(i).v.addi(v.v.sub(reg));
      }
      p.clear();
    }
  }
}
