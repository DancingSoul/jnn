package cn.edu.hit.ir.JNN;

import java.util.Vector;

public abstract class ExecutionEngine {
	
	ExecutionEngine() {
	
	}
	
	protected ComputationGraph cg;
	
	public abstract void invalidate();
	public abstract Tensor forward();
	public abstract Tensor forward(int i);
	public abstract Tensor incrementalForward();
	public abstract Tensor incrementalForward(int i);
	public abstract Tensor getValue(int i);
	public abstract void backward();
	public abstract void backward(int i);
	
	protected ExecutionEngine(ComputationGraph cg_) {
		cg = cg_;
	}
}

class SimpleExecutionEngine extends ExecutionEngine {
	private Vector <Tensor> nfxs;
	private Vector <Tensor> ndEdfs;
	int numNodesEvaluated;
	
	SimpleExecutionEngine() {
			
	}
	public SimpleExecutionEngine(final ComputationGraph cg_) {
		cg = cg_;
	}
	
	public void invalidate() {
		numNodesEvaluated = 0;
	}
	
	public Tensor forward() {
		final int nodeMaxIndex = cg.nodes.size() - 1;
		return forward(nodeMaxIndex);
	}
	public Tensor forward(int i) {
		invalidate();
		return incrementalForward(i);
	}
	
	public Tensor getValue(int i) {
		assert(i < cg.nodes.size());
		if (i >= numNodesEvaluated) {
			incrementalForward();
		}
		return nfxs.get(i);
	}
	
	public Tensor incrementalForward() {
		final int nodeMaxIndex = cg.nodes.size() - 1;
		return incrementalForward(nodeMaxIndex);
	}
	
	public Tensor incrementalForward(int i) {
		assert(i < cg.nodes.size());
		
		//...
	}
	
	public void backward() {
		assert(nfxs.size() == cg.nodes.size());
		backward(cg.nodes.size() - 1);
	}
	
	public void backward(int fromWhere) {
		assert(fromWhere + 1 <= nfxs.size());
		assert(fromWhere + 1 <= cg.nodes.size());
		//...
	}
		
}

