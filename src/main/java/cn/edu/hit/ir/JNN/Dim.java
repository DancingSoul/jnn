package cn.edu.hit.ir.JNN;

import java.util.Arrays;
import java.util.List;

public class Dim {
  /**
   * Dim is the class for storing the Dimension for a Tensor. Currently, only
   * two dimension is actually supported in JNN.
   */
  final int JNN_MAX_TENSOR_DIM = 7;
  public int d[] = new int[JNN_MAX_TENSOR_DIM];
  public int nd; // number of dimensions
  public int bd; // number of batches

  public Dim() {
    nd = 0;
    bd = 1;
  }

  public Dim(Dim d_) {
    nd = d_.nd;
    bd = d_.bd;
    System.arraycopy(d_.d, 0, d, 0, nd);
  }

  public Dim(List<Integer> x) {
    nd = 0;
    bd = 1;
    for (nd = 0; nd < x.size(); ++nd) {
      d[nd] = x.get(nd);
    }
  }

  public Dim(List<Integer> x, int b) {
    nd = 0;
    bd = b;
    for (nd = 0; nd < x.size(); ++nd) {
      d[nd] = x.get(nd);
    }
  }

  public String toString() {
    StringBuilder sb = new StringBuilder("(");
    if (nd > 0) {
      for (int i = 0; i < nd; ++i) {
        sb.append(d[i]);
        sb.append(i + 1 < nd ? "," : ")");
      }
    } else {
      sb.append(")");
    }
    sb.append("|").append(bd);
    return sb.toString();
  }

  public final int size() {
    return batchSize() * bd;
  }

  public final int batchSize() {
    int p = 1;
    for (int i = 0; i < nd; ++i) {
      p *= d[i];
    }
    return p;
  }

  public final int getSumDimensions() {
    int p = 0;
    for (int i = 0; i < nd; ++i) p += d[i];
    return p;
  }

  public final Dim truncate() {
    Dim r = new Dim(this);
    int m = 1;
    int s = size();
    for (int i = 1; i < s; ++i) {
      if (size(i) > 1) {
        m = i + 1;
      }
    }
    r.resize(m);
    return r;
  }

  public final Dim singleBatch() {
    Dim r = new Dim(this);
    r.bd = 1;
    return r;
  }

  public final void resize(int i) {
    nd = i;
  }

  public final int getNumDimensions() {
    return nd;
  }

  public final int getNumRows() {
    return d[0];
  }

  public final int getNumCols() {
    return nd > 1 ? d[1] : 1;
  }

  public final int getNumBatchElements() {
    return bd;
  }

  public final void set(int i, int s) {
    assert (i < nd);
    assert (s > 0);
    d[i] = s;
  }

  public final int at(int i) {
    return i < nd ? d[i] : 1;
  }

  public final int size(int i) {
    return this.at(i);
  }

  /**
   * @return
   */
  public final Dim transpose() {
    if (nd == 1) {
      // [Refactor] http://stackoverflow.com/questions/1005073/initialization-of-an-arraylist-in-one-line
      return new Dim(Arrays.asList(1, d[0]), bd);
    } else if (nd == 2) {
      return new Dim(Arrays.asList(d[1], d[0]), bd);
    }
    throw new IllegalArgumentException(
        "Cannot transpose Dim Object with more than 2 dimensions");
  }

  public final boolean equals(final Dim b) {
    if (nd != b.nd || bd != b.bd) {
      return false;
    }
    for (int i = 0; i < nd; i++) {
      if (d[i] != b.d[i]) {
        return false;
      }
    }
    return true;
  }

  // Creator
  public static Dim create(int d1) {
    return new Dim(Arrays.asList(d1));
  }

  public static Dim create(int d1, int d2) {
    return new Dim(Arrays.asList(d1, d2));
  }

  public static Dim createBatches(int d1, int n) {
    return new Dim(Arrays.asList(d1), n);
  }

  public static Dim createBatches(int d1, int d2, int n) {
    return new Dim(Arrays.asList(d1, d2), n);
  }
}
