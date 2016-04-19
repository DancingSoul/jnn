package cn.edu.hit.ir.JNN.Trainers;

import cn.edu.hit.ir.JNN.Model;

public abstract class AbstractTrainer {
  //learning rates
  public double eta0;
  public double eta;
  public double etaDecay;
  public double epoch;
  public double lambda; //weight regularization (l2)

  //clipping
  public boolean clippingEnabled;
  public double clipThreshold;
  public double clips;
  public double updates;

  public Model model; // parameters and gradients live here

  AbstractTrainer() {

  }

  public AbstractTrainer(Model m, double lam, double e0) {
    eta0 = eta = e0;
    lambda = lam;
    clippingEnabled = true;
    clipThreshold = 5;
    model = m;
  }

  public abstract void update();

  public abstract void update(double scale);

  public void updateEpoch() {
    this.updateEpoch(1.0);
  }

  public void updateEpoch(double r) {
    epoch += r;
    eta = eta0 / (1 + epoch * etaDecay);
  }

  /**
   * if clipping is enabled and the gradient is too big, return the amount to
   * scale the gradient by (otherwise 1)
   */
  public double clipGradients() {
    double gscale = 1.0;
    if (clippingEnabled) {
      double gg = model.gradientL2norm();
      if (Double.isNaN(gg) || Double.isInfinite(gg)) {
        throw new RuntimeException(
            "Magnitude of gradient is bad : " + gg);
      }
      if (gg > clipThreshold) {
        ++clips;
        gscale = clipThreshold / gg;
      }
    }
    return gscale;
  }

  public void status() {
    //...
  }

}
