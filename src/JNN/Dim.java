package JNN;

import java.util.ArrayList;

public class Dim {
	final int JNN_MAX_TENSOR_DIM = 7;
	Dim(){} {
		nd = 0;
		bd = 1;
	}
	Dim(ArrayList<Integer> x) {
		nd = 0;
		bd = 1;
		for (int v : x) d[nd++] = v;
	}
	Dim(ArrayList<Integer> x, int b) {
		nd = 0;
		bd = b;
		for (int v : x) d[nd++] = v;
	}
	
	public final int size() {
		return batchSize() * bd;
	}
	public final int batchSize() {
		int p = 1;
		for (int i = 0; i < nd; ++i) p *= d[i];
		return p;
	}
	public final int sumDims() {
		int p = 0;
		for (int i = 0; i < nd; ++i) p += d[i];
		return p;
	}
	public final Dim truncate() {
		Dim r = this;
		int m = 1;
		int s = size();
		for (int i = 1; i < s; ++i)
			if (size(i) > 1) m = i + 1;
		r.resize(m);
		return r;
	}
	public final Dim singleBatch(){
		Dim r = this;
		r.bd = 1;
		return r;
	}
	public final void resize(int i) { nd = i;}
	public final int nDims() { return nd;}
	public final int rows() { return d[0];}
	public final int cols() { return nd > 1 ? d[1] : 1;}
	public final int batchElems() { return bd;}
	public final void set(int i, int s) { assert(i < nd); assert(s > 0); d[i] = s;}
	public final int at(int i) { return i < nd ? d[i] : 1;} //жиди[]
	public final int size(int i) { return this.at(i);}
	public final Dim transpose() {
		if (nd == 1) { 
			return new Dim(new ArrayList<Integer>() {{add(1); add(d[0]);}}, bd);
		} else if (nd == 2) {
			return new Dim(new ArrayList<Integer>() {{add(d[1]); add(d[0]);}}, bd);
		} 
		throw new IllegalArgumentException("Cannot transpose Dim Object with more than 2 dimensions");
	}
	public final boolean equals(final Dim b) {
		if (nd != b.nd || bd != b.bd) return false;
		for (int i = 0; i < nd; i++)
			if (d[i] != b.d[i]) return false;
		return true;
	}
	int d[] = new int[JNN_MAX_TENSOR_DIM];
	int nd;
	int bd;
}
