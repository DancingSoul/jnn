package cn.edu.hit.ir.JNN;


import java.util.Vector;

/**
 * Created by dancingsoul on 2016/4/25.
 */
public class ShadowLookupParameters {
    public Vector<Tensor> h;
    public ShadowLookupParameters(final LookupParameters lp) {
        h = new Vector<Tensor>(lp.values.size());
        for (Tensor t : lp.values) {
            h.addElement(new Tensor(t.d));
        }
    }
}
