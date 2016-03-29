package main.java.cn.edu.hit.ir.JNN;

import java.util.Vector;

import org.ejml.data.DenseMatrix64F;

public class Model {
	Model() {
		
		
	}
	
	public double gradientL2norm() {
		if (gradientNormScratch.length == 0)
			gradientNormScratch = new AtomicDouble[allParams.size()];
		int pi = 0;
		for (ParametersBase p : allParams) {
			p.gSquaredL2norm(gradientNormScratch[pi]);
			++pi;
		}
		double gg = 0;
		for (int i = 0; i < pi; ++i)
			gg += gradientNormScratch[i].get();
		return Math.sqrt(gg);
	}
	
	public void reset_gradient() {
		for (Parameters p : params) {
			p.clear();
		}
		for (LookupParameters p : lookupParams) {
			p.clear();
		}
	}
	
	public Parameters addParameters(final Dim d, double scale) {
		Parameters p = new Parameters(d, scale);
		allParams.addElement(p);
		params.addElement(p);
		return p;
	}
	
	public LookupParameters addLookupParameters(int n, final Dim d) {
		LookupParameters p = new LookupParameters(n, d);
		allParams.addElement(p);
		lookupParams.addElement(p);
		return p;
	}
	
	public void projectWeights(double radius) {
		if (projectScratch.length == 0)
			projectScratch = new AtomicDouble[allParams.size()];
		int pi = 0;
		for (ParametersBase p : allParams) {
			p.squaredL2norm(projectScratch[pi]);
			++pi;
		}
		double gg = 0;
		for (int i = 0; i < pi; ++i)
			gg += projectScratch[i].get();
		System.out.println("NORM : " + Math.sqrt(gg));
	}
	
	public Vector <ParametersBase> allParametersLiset() {
		return allParams;
	}
	
	public Vector <Parameters> paramtersList() {
		return params;
	}
	
	public Vector <LookupParameters> lookupParametersList() {
		return lookupParams;
	}
	
	
	private Vector <ParametersBase> allParams;
	private Vector <Parameters> params;
	private Vector <LookupParameters> lookupParams;
	private AtomicDouble[] gradientNormScratch;
	private static AtomicDouble[] projectScratch;
}
