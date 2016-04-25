package cn.edu.hit.ir.JNN;

import java.util.Vector;

/**
 * Created by dancingsoul on 2016/4/25.
 */
public class ShadowUtils {
  //one per element in model.parametersList
  public static Vector<ShadowParameters> AllocateShadowParameters(final Model m) {
    Vector<ShadowParameters> v = new Vector<ShadowParameters>(m.paramtersList().size());
    for (Parameters p : m.paramtersList()) {
      v.addElement(new ShadowParameters(p));
    }
    return v;
  }
  //one per element in model.lookupParametersList
  public static Vector<ShadowLookupParameters> AllocateShadowLookupParameters(final Model m) {
    Vector<ShadowLookupParameters> v = new Vector<ShadowLookupParameters>(m.lookupParametersList().size());
    for (LookupParameters p : m.lookupParametersList()) {
      v.addElement(new ShadowLookupParameters(p));
    }
    return v;
  }
}
