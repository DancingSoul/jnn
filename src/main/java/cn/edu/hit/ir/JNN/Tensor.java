package cn.edu.hit.ir.JNN;

import java.io.Serializable;

import org.ejml.data.DenseMatrix64F;

public class Tensor implements Serializable {
  /**
   * Tensor is a wrapping class for access the Matrix.
   *
   * Tensor is used in:
   *  - Parameters
   *  - ComputationGraph
   *
   *  Currently, we suppose the number of dimension for the tensor is either 1 or 2
   */
  private static final long serialVersionUID = 2238574422776967031L;
  public Dim d;
	public DenseMatrix64F v;

	Tensor() {
	  d = new Dim();
	  v = new DenseMatrix64F();
	}

  Tensor(final Dim d_) {
    d = new Dim(d_);
    v = new DenseMatrix64F(d_.at(0), d_.at(1));
  }

	Tensor(final Dim d_, final DenseMatrix64F v_) {
		d = new Dim(d_);
		v = new DenseMatrix64F(v_);
	}

	public boolean isValid() {
    if (d == null || v == null) {
      // should pay attention to the uninitialized tensor.
      return false;
    }
    if (d.size() != v.getNumElements()) {
      return false;
    }

		int s = d.size();
		for (int i = 0; i < s; ++i) {
			if (Double.isNaN(v.get(i)) || Double.isInfinite(v.get(i))) {
				return false;
			}
		}
		return true;
	}


}
