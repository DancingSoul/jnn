package cn.edu.hit.ir.JNN;

import java.util.HashSet;
import java.util.Vector;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.NormOps;

public abstract class ParametersBase {
	public abstract void scaleParameters(double a);
	public abstract void squaredL2norm(AtomicDouble sqnorm);
	public abstract void gSquaredL2norm(AtomicDouble sqnorm);
	public abstract int size();
}

class Parameters extends ParametersBase{
	public Dim dim;
	public Tensor values;
	public Tensor g;
	
	Parameters() {
	}
	
	Parameters(Dim d, double scale){
		dim = d;
		values.d = g.d = d;
		values.v = new DenseMatrix64F(d.size());
		if (Math.abs(scale) < 1e-10 && Math.abs(scale) > -1e-10) {
			TensorUtils.randomize(values, scale);
		} else {
			TensorUtils.randomize(values);
		}
		g.v = new DenseMatrix64F(d.size());
		TensorUtils.zero(g);
	}
	
	public void scaleParameters(double a){
		CommonOps.scale(a, g.v);
	}
	
	public void squaredL2norm(AtomicDouble sqnorm){
		sqnorm.set(Math.pow(NormOps.normP2(values.v), 2));
	}
	
	public void gSquaredL2norm(AtomicDouble sqnorm){
		sqnorm.set(Math.pow(NormOps.normP2(g.v), 2));
	}
	
	public int size(){
		return dim.size();
	}
	
	public void copy(final Parameters param) {
		assert(dim.equals(param.dim));
		TensorUtils.copyElements(values, param.values);
	}
	
	public void accumulate_grad(final Tensor d){
		CommonOps.addEquals(g.v, d.v);
	}
	
	public void clear() {
		TensorUtils.zero(g);
	}
}

class LookupParameters extends ParametersBase{
	public Dim dim;
	public Vector <Tensor> values;
	public Vector <Tensor> grads;
	public HashSet <Integer> nonZeroGrads;
	
	LookupParameters() {
	}
	
	LookupParameters(int n, Dim d){
		dim = d;
		values.setSize(n);
		grads.setSize(n);
		for (int i = 0; i < n; i++){
			values.get(i).d = d;
			values.get(i).v = new DenseMatrix64F(d.size());
			TensorUtils.randomize(values.get(i));
			
			grads.get(i).d = d;
			grads.get(i).v = new DenseMatrix64F(d.size());
			TensorUtils.zero(grads.get(i));
		}
	}
	
	public void scaleParameters(double a){
		for (Tensor p : values)
			CommonOps.scale(a, p.v);
	}
	
	public void squaredL2norm(AtomicDouble sqnorm){
		double a = 0;
		for (int i = 0; i < values.size(); ++i)
			a += Math.pow(NormOps.normP2(values.get(i).v), 2);
		sqnorm.set(a);
	}
	
	public void gSquaredL2norm(AtomicDouble sqnorm){
		double a = 0;
		for (Integer i : nonZeroGrads)
			a += Math.pow(NormOps.normP2(grads.get(i).v), 2);
		sqnorm.set(a);
	}
	
	public int size(){
		return values.size() * dim.size();
	}
	
	public void initialize(int index, final Vector <Double> val) {
		double[] tmp = new double[val.size()];   
		for (int i = 0; i < val.size(); i++)
	      tmp[i] = val.get(i);
	    values.get(index).v = new DenseMatrix64F(val.size(), 1, true, tmp);
	}
	
	public void copy(final LookupParameters param) {
		assert(dim.equals(param.dim));
		for (int i = 0; i < param.values.size(); ++i) 
			TensorUtils.copyElements(values.get(i), param.values.get(i));
	}
	
	public void accumulateGrad(int index, final Tensor d) {
		CommonOps.addEquals(grads.get(index).v, d.v);
	}
	
	public void clear() {
		for (Integer i : nonZeroGrads)
			TensorUtils.zero(grads.get(i));
		nonZeroGrads.clear();
	}
	
}
