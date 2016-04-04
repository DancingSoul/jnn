package cn.edu.hit.ir.JNN.Trainer;

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
    return 0.0;
  }

  public void status() {
    //...
  }

}
