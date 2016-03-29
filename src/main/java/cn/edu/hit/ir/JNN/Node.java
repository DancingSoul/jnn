package cn.edu.hit.ir.JNN;

import java.util.List;
import java.util.Vector;

public abstract class Node {
	Node() {
		
	}
	Node(List<Integer> x) {
		args.setSize(x.size());
		for (int i = 0; i < x.size(); i++)
			args.setElementAt(x.get(i), i);
	}
	Node(Vector<Integer> x) {
		args.setSize(x.size());
		for (int i = 0; i < x.size(); i++)
			args.setElementAt(x.get(i), i);
	}
	
	
	public abstract Dim dimForward(final Vector <Dim> xs);
	public abstract String asString(final Vector <String> args);
	public int auxStorageSize() {return 0;};
	
	public abstract void forwardImpl(final Vector <Tensor> xs, Tensor fx);
	
	public abstract void backwardImpl(final Vector <Tensor> xs, 
			final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi);
	
	//?public abstract boolean supportsMultibatch(){return false;}
	
	public void forward(final Vector <Tensor> xs, Tensor fx) {}
	public void backward(final Vector <Tensor> xs, 
			final Tensor fx, final Tensor dEdf, int i, Tensor dEdxi) {}
	
	public final int arity(){return args.size();}
	
	public Dim dim;

	public Vector <Integer> args;
}


