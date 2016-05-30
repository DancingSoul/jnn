package cn.edu.hit.ir.JNN;

import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
	public INDArray v;

	public Tensor() {
		d = new Dim();
		v = Nd4j.zeros(1);
	}

	public Tensor(final Dim d_) {
		d = new Dim(d_);
		//v = Nd4j.zeros(d.size());
		//v.setOrder('f');
		//v = v.reshape(d.at(0), d.at(1) * d.bd);
		v = Nd4j.zeros(d.at(0), d.at(1) * d.bd);
	}

	public Tensor(final Dim d_, final INDArray v_) {
		d = new Dim(d_);
		v = v_.dup();
	}
	public Tensor(final Tensor t) {
		d = new Dim(t.d);
		v = t.v.dup();
	}

	public INDArray vec() {
		return v.reshape(d.size());
	}
	//Get the matrix for a particular batch
	public INDArray getBatchMatrix(int bid) {
		bid %= d.bd;
		return v.reshape(d.bd, d.batchSize()).getRow(bid).reshape(d.at(0), d.at(1));
	}
	public void setBatchMatrix(int bid, INDArray t) {
		bid %= d.bd;
		v.reshape(d.bd, d.batchSize()).getRow(bid).assign(t.reshape(t.length()));
	}
	public void addBatchMatrix(int bid, INDArray t) {
		bid %= d.bd;
		v.reshape(d.bd, d.batchSize()).getRow(bid).addi(t.reshape(t.length()));
	}
	//Get the data as a matrix, where each "row" is the concatenation of rows and columns, and each "column" is batches
	public INDArray rowcolMatrix() {
		return v.reshape(d.getNumRows() * d.getNumCols(), d.getNumBatchElements());
	}
	//Get the data as a matrix, where each "row" is the concatenation of rows
	//and each "column" is the concatenation of columns ans batches
	public INDArray colbatchMatrix() {
		return v.reshape(d.getNumRows(), d.getNumCols() * d.getNumBatchElements());
	}
	public boolean isValid() {
		if (d == null || v == null) {
			// should pay attention to the uninitialized tensor.
			return false;
		}
		if (d.size() != v.length()) {
			return false;
		}

		int s = d.size();
		for (int i = 0; i < s; ++i) {
			if (Double.isNaN(v.getDouble(i)) || Double.isInfinite(v.getDouble(i))) {
				return false;
			}
		}
		return true;
	}
}