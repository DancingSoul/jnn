package cn.edu.hit.ir.JNN.Examples;

import cn.edu.hit.ir.JNN.*;
import cn.edu.hit.ir.JNN.Builders.AbstractRNNBuilder;
import cn.edu.hit.ir.JNN.Builders.LSTMBuilder;
import cn.edu.hit.ir.JNN.Trainers.AbstractTrainer;
import cn.edu.hit.ir.JNN.Trainers.MomentumSGDTrainer;
import cn.edu.hit.ir.JNN.Trainers.SimpleSGDTrainer;
import cn.edu.hit.ir.JNN.Utils.DictUtils;
import cn.edu.hit.ir.JNN.Utils.SerializationUtils;
import cn.edu.hit.ir.JNN.Utils.TensorUtils;
import java.util.Date;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Collections;
import java.util.Vector;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by dancingsoul on 2016/4/27.
 */

class RNNLanguageModel {
  private final static double pdrop = 0.5;
  private final static int LAYERS = 1;
  private final static int INPUT_DIM = 128;
  private final static int HIDDEN_DIM = 128;
  private final static int TAG_HIDDEN_DIM= 32;
  private final static int TAG_DIM = 32;
  public static int TAG_SIZE = 0;
  public static int VOCAB_SIZE = 0;
  private final static int kSOS = 0;
  private final static int kEOS = 1;
  private final static int kNONE = 0;
  private static boolean eval = false;

  private LookupParameters pW;
  private Parameters pL2th;
  private Parameters pR2th;
  private Parameters pThbias;
  private Parameters pTh2t;
  private Parameters pTbias;
  private LSTMBuilder l2rbuilder;
  private LSTMBuilder r2lbuilder;
  RNNLanguageModel(Model model) {
    l2rbuilder = new LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model);
    r2lbuilder = new LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model);
    pW = model.addLookupParameters(VOCAB_SIZE, Dim.create(INPUT_DIM));
    pL2th = model.addParameters(Dim.create(TAG_HIDDEN_DIM, HIDDEN_DIM));
    pR2th = model.addParameters(Dim.create(TAG_HIDDEN_DIM, HIDDEN_DIM));
    pThbias = model.addParameters(Dim.create(TAG_HIDDEN_DIM));

    pTh2t = model.addParameters(Dim.create(TAG_SIZE, TAG_HIDDEN_DIM));
    pTbias = model.addParameters(Dim.create(TAG_SIZE));
  }
  // return Expression of total loss
  Expression BuildTaggingGraph(final Vector<Integer> sent, final Vector<Integer> tags, ComputationGraph cg,
                               AtomicDouble cor, AtomicDouble nTagged) {
    final int slen = sent.size();
    l2rbuilder.newGraph(cg);
    l2rbuilder.startNewSequence();
    r2lbuilder.newGraph(cg);
    r2lbuilder.startNewSequence();
    Expression iL2th = Expression.Creator.parameter(cg, pL2th);
    Expression iR2th = Expression.Creator.parameter(cg, pR2th);
    Expression iThbias = Expression.Creator.parameter(cg, pThbias);
    Expression iTh2t = Expression.Creator.parameter(cg, pTh2t);
    Expression iTbias = Expression.Creator.parameter(cg, pTbias);
    Vector<Expression> errs = new Vector<Expression>();
    Vector<Expression> iWords = new Vector<Expression>();
    Vector<Expression> fwds = new Vector<Expression>();
    Vector<Expression> revs = new Vector<Expression>();

    iWords.setSize(slen);
    fwds.setSize(slen);
    revs.setSize(slen);
    //read sequence from left to reght
    l2rbuilder.addInput(Expression.Creator.lookup(cg, pW, new Vector<Integer>(Arrays.asList(kSOS))));
    for (int t = 0; t < slen; ++t) {
      iWords.set(t, Expression.Creator.lookup(cg, pW, new Vector<Integer>(Arrays.asList(sent.get(t)))));
      if (!eval) { iWords.set(t, Expression.Creator.noise(iWords.get(t), 0.1)); }
      fwds.set(t, l2rbuilder.addInput(iWords.get(t)));
    }

    //read sequence from right to left
    r2lbuilder.addInput(Expression.Creator.lookup(cg, pW, new Vector<Integer>(Arrays.asList(kEOS))));
    for (int  t = 0; t < slen; ++t)
      revs.set(slen - t - 1, r2lbuilder.addInput(iWords.get(slen - t - 1)));

    for (int t = 0; t < slen; ++t) {
      nTagged.add(1.0);
      Expression iTh = Expression.Creator.tanh(Expression.Creator.affineTransform(
              new Vector<Expression>(Arrays.asList(iThbias, iL2th, fwds.get(t), iR2th, revs.get(t)))));
      //if (!eval) { iTh = Expression.Creator.dropout(iTh, pDrop); }
      Expression iT = Expression.Creator.affineTransform(new Vector<Expression>(Arrays.asList(iTbias, iTh2t, iTh)));
      Vector<Double> dist = TensorUtils.toVector(cg.incrementalForward());
      double best = -1e100;
      int besti = -1;
      for (int i = 0; i < dist.size(); ++i) {
        if (dist.get(i) > best) {
          best = dist.get(i);
          besti = i;
        }
      }
      if (tags.get(t) == besti) cor.add(1.0);
      Expression iErr = Expression.Creator.pickNegLogSoftmax(iT, new Vector<Integer>(Arrays.asList(tags.get(t))));
      errs.addElement(iErr);
    }
    return Expression.Creator.sum(errs);
  }

  public static void setEval(boolean eval_) {
    eval = eval_;
  }
}


public class TagBilstm {
  static String trainName = "pku-train.pos";
  static String devName = "pku-test.pos";
  static Dict d = new Dict();
  static Dict td = new Dict();
  static int VOCAB_SIZE = 0;
  static int TAG_SIZE = 0;
  public static void readFile(String fileName, Vector<Vector<Integer>> x, Vector<Vector<Integer>> y) {
    int lc = 0;
    int toks = 0;
    try {
      BufferedReader reader = new BufferedReader(new FileReader(fileName));
      String line = null;
      while((line = reader.readLine()) != null){
        ++lc;
        Vector<Integer> tx = new Vector<Integer>();
        Vector<Integer> ty = new Vector<Integer>();
        DictUtils.readSentencePair(line, tx, d, ty, td);
        assert(x.size() == y.size());
        x.addElement(tx);
        y.addElement(ty);
        toks += x.size();
      }
      System.err.println(lc + " lines, " + toks + " tokens, " + d.size() + " types");
      System.err.println("Tags: " + td.size());
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public static void main(String args[]) {
    Vector<Vector<Integer>> trainX = new Vector<Vector<Integer>>();
    Vector<Vector<Integer>> trainY = new Vector<Vector<Integer>>();
    System.err.println("Reading training data from "  + trainName + "...") ;

    readFile(trainName, trainX, trainY);
    d.freeze();
    td.freeze();
    RNNLanguageModel.VOCAB_SIZE = d.size();
    RNNLanguageModel.TAG_SIZE = td.size();
    Vector<Vector<Integer>> devX = new Vector<Vector<Integer>>();
    Vector<Vector<Integer>> devY = new Vector<Vector<Integer>>();
    readFile(devName, devX, devY);

    double best = 1e100;
    Model model = new Model();
    boolean useMomentum = false;
    AbstractTrainer sgd = null;
    if (useMomentum)
      sgd = new MomentumSGDTrainer(model);
    else
      sgd = new SimpleSGDTrainer(model);

    RNNLanguageModel lm = new RNNLanguageModel(model);

    int reportEveryI = 50;
    int devEveryIReports = 25;
    int si = trainX.size();
    Vector<Integer> order = new Vector<Integer>();
    for (int i = 0; i < trainX.size(); ++i)
      order.addElement(i);
    boolean first = true;
    int report = 0;
    int lines = 0;
    Long startOfTraining = new Date().getTime();
    for (int t = 0; t < 60000; t++){
      double loss = 0.0;
      AtomicDouble correct = new AtomicDouble(0.0);
      AtomicDouble ttags = new AtomicDouble(0.0);
      for (int i = 0; i < reportEveryI; ++i) {
        if (si == trainX.size()) {
          si = 0;
          if (first) { first = false; } else { sgd.updateEpoch(); }
          System.err.println("SHUFFLE");
          Collections.shuffle(order);
        }
        //build graph for this instance
        ComputationGraph cg = new ComputationGraph();
        lm.BuildTaggingGraph(trainX.get(si), trainY.get(si), cg, correct, ttags);
        //cg.gradientCheck();
        ++si;
        loss += TensorUtils.toScalar(cg.forward());
        cg.backward();
        sgd.update(1.0);
        //System.out.println(model.gradientCheck());
        ++lines;
      }
      //sgd.status();
      System.err.println("E = " + (loss / ttags.doubleValue()) + " ppl = " + Math.exp(loss / ttags.doubleValue())
              + " (acc = " + (correct.doubleValue() / ttags.doubleValue()) + ")" + "lines : " + lines);

      //show socre on dev data
      report++;
      if (report % devEveryIReports == 0) {
        double dloss = 0.0;
        AtomicDouble dcorr = new AtomicDouble(0.0);
        AtomicDouble dtags = new AtomicDouble(0.0);
        lm.setEval(true);
        for (int i = 0; i < devX.size(); ++i) {
          ComputationGraph cg = new ComputationGraph();
          lm.BuildTaggingGraph(devX.get(i), devY.get(i), cg, dcorr, dtags);
          dloss += TensorUtils.toScalar(cg.forward());
        }
        lm.setEval(false);
        if (dloss < best) {
          best = dloss;
          SerializationUtils.save("tag.obj", model);
        }
        System.err.println("\n***DEV [epoch = " + (lines / (double) trainX.size()) + "] E = "
                + (dloss / dtags.doubleValue()) + " ppl = " + Math.exp(dloss / dtags.doubleValue())
                + " acc = " + (dcorr.doubleValue() / dtags.doubleValue()));
      }
    }
    System.err.println("consume: " + (new Date().getTime() - startOfTraining));
  }
}
