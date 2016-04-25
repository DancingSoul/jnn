package cn.edu.hit.ir.JNN;

import java.io.Serializable;
import java.util.Vector;

public class Model implements Serializable {
  private static final long serialVersionUID = 83645026291628434L;
  
  private Vector<AbstractParameters> allParams;
  private Vector<Parameters> params;
  private Vector<LookupParameters> lookupParams;
  private Double[] gradientNormScratch;
  private static Double[] projectScratch;
  
  
  public Model() {
    allParams = new Vector<AbstractParameters>();
    params = new Vector<Parameters>();
    lookupParams = new Vector<LookupParameters>();
  }

  public double gradientL2norm() {
    if (gradientNormScratch == null) {
      gradientNormScratch = new Double[allParams.size()];
    }
    int pi = 0;
    for (AbstractParameters p : allParams) {
      gradientNormScratch[pi] = p.gSquaredL2norm();
      ++pi;
    }
    double gg = 0;
    for (int i = 0; i < pi; ++i) {
      gg += gradientNormScratch[i];
    }
    return Math.sqrt(gg);
  }

  public void resetGradient() {
    for (Parameters p : params) {
      p.clear();
    }
    for (LookupParameters p : lookupParams) {
      p.clear();
    }
  }

  public Parameters addParameters(final Dim d) {
    return this.addParameters(d, 0.0);
  }
  
  public Parameters addParameters(final Dim d, double scale) {
    Parameters p = new Parameters(d, scale);
    allParams.addElement(p);
    params.addElement(p);
    return p;
  }

  public LookupParameters addLookupParameters(int n, final Dim d) {
    LookupParameters p = new LookupParameters(n, d);
    allParams.addElement(p);
    lookupParams.addElement(p);
    return p;
  }

  public void projectWeights() {
    this.projectWeights(1.0);
  }
  
  public void projectWeights(double radius) {
    if (projectScratch == null) {
      projectScratch = new Double[allParams.size()];
    }
    int pi = 0;
    for (AbstractParameters p : allParams) {
      projectScratch[pi] = p.squaredL2norm();
      ++pi;
    }
    double gg = 0;
    for (int i = 0; i < pi; ++i) {
      gg += projectScratch[i];
    }
    System.out.println("NORM : " + Math.sqrt(gg));
  }

  public Vector<AbstractParameters> allParametersList() {
    return allParams;
  }

  public Vector<Parameters> paramtersList() {
    return params;
  }

  public Vector<LookupParameters> lookupParametersList() {
    return lookupParams;
  }

  public boolean gradientCheck() {
    for (int i = 0; i < params.size(); i++) {
      for (int j = 0; j < params.get(i).g.v.length(); ++j) {
          System.out.println(params.get(i).g.v.getDouble(j) + "       " + params.get(i).gCheck.v.getDouble(j));
          if (Math.abs(params.get(i).g.v.getDouble(j) - params.get(i).gCheck.v.getDouble(j)) > 1E-5) {
            return false;
        }
      }
    }
    for (int i = 0; i < lookupParams.size(); i++) {
      for (int j = 0; j < lookupParams.get(i).grads.size(); j++) {
        for (int k = 0; k < lookupParams.get(i).grads.get(j).v.length(); k++) {
          if (Math.abs(lookupParams.get(i).grads.get(j).v.getDouble(k) -
                  lookupParams.get(i).gradsCheck.get(j).v.getDouble(k)) > 1E-5) {
            return false;
          }
        }
      }
      
    } 
    return true;
  }
}
