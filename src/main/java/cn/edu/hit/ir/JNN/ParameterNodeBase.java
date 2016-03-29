package main.java.cn.edu.hit.ir.JNN;

import java.util.Vector;

import org.ejml.data.DenseMatrix64F;

public abstract class ParameterNodeBase extends Node{
	public abstract void accumulateGrad(final Tensor g);
}

class ParameterNode extends ParameterNodeBase {
	ParameterNode(Parameters p) {
		
	}
	public String asString(final Vector<String> argNames) {
		//...
		return "";		
	}
	public Dim dimForward(final Vector<Dim> xs) {
		assert(xs.size() == 0);
		return dim;
	}
	public void forwardImpl(final Vector <Tensor> xs, Tensor fx) {
		assert(xs.size() == 0);
		fx.v = params.values.v;
	}
	
	public void backwardImpl(final Vector <Tensor> xs,
			final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
		//"called backward() on arity 0 node : i = ??
		//abort();
	}
	public void accumulateGrad(final Tensor g) {
		params.accumulate_grad(g);
	}
	public Dim dim;
	public Parameters params;
}

class ConstParameterNode extends Node {
	ConstParameterNode(Parameters p) {
		dim = new Dim(p.dim);
		params = p;
	}
	
	public String asString(final Vector <String> argNames) {
		//...
		return "";		
	}
	public Dim dimForward(final Vector <Dim> xs) {
		assert(xs.size() == 0);
		return dim;
	}
	public void forwardImpl(final Vector <Tensor> xs, Tensor fx) {
		assert(xs.size() == 0);
		fx.v = params.values.v;
	}
	
	public void backwardImpl(final Vector <Tensor> xs,
			final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {
		//"called backward() on arity 0 node : i = ??
		//abort();
	}
	
	public Dim dim;
	public Parameters params;
}

class InputNode extends Node {
	
	
}

