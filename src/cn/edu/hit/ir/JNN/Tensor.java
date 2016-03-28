package cn.edu.hit.ir.JNN;

import java.io.Serializable;
import java.util.Arrays;

import org.ejml.data.DenseMatrix64F;

class Tensor implements Serializable {
	public Dim d;
	public DenseMatrix64F v;
	Tensor() {
	}

	Tensor(final Dim d_, final DenseMatrix64F v_) {
		d = d_;
		v = v_;
	}	
	
	public boolean isValid() {
		int s = d.size();
		for (int i = 0; i < s; ++i) {
			if (Double.isNaN(v.get(i)) || Double.isInfinite(v.get(i))) {
				return false;
			}
		}
		return true;
	}

	public double toScalar(Tensor t) {
		assert (t.d.size() == 1);
		return t.v.get(0);
	}

	/*public static double[] toVector(Tensor t) {
		float[] ret = null;
		ret = Arrays.copyOf(t.v, t.d.size());
		return ret;
	}*/
}
