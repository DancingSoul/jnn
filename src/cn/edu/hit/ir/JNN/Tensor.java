package cn.edu.hit.ir.JNN;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import cn.edu.hit.ir.JNN.Dim;

class Tensor implements Serializable {
  public Dim d;
  public float[] v;

  Tensor() {
  }

  Tensor(final Dim d_, float[] v_) {
    d = d_;
    v = v_;
  }

  public boolean isValid() {
    int s = d.size();
    for (int i = 0; i < s; ++i) {
      if (Float.isNaN(v[i]) || Float.isInfinite(v[i])) {
        return false;
      }
    }
    return true;
  }

  public static float toScalar(Tensor t) {
    assert (t.d.size() == 1);
    return t.v[0];
  }

  public static float[] toVector(Tensor t) {
    float[] ret = null;
    ret = Arrays.copyOf(t.v, t.d.size());
    return ret;
  }
}
