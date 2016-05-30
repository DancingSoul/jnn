package cn.edu.hit.ir.JNN.Examples;

import cn.edu.hit.ir.JNN.*;
import cn.edu.hit.ir.JNN.Builders.LSTMBuilder;
import cn.edu.hit.ir.JNN.Trainers.AbstractTrainer;
import cn.edu.hit.ir.JNN.Trainers.MomentumSGDTrainer;
import cn.edu.hit.ir.JNN.Trainers.SimpleSGDTrainer;
import cn.edu.hit.ir.JNN.Utils.DictUtils;
import cn.edu.hit.ir.JNN.Utils.SerializationUtils;
import cn.edu.hit.ir.JNN.Utils.TensorUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

/**
 * Created by dancingsoul on 2016/5/20.
 */
class LSTMLanguageModel {
  private final static double pdrop = 0.5;
  private final static int LAYERS = 1;
  public static int INPUT_DIM = 0;
  private final static int HIDDEN_DIM = 100;
  private final static int TAG_HIDDEN_DIM= 32;
  public static int TAG_SIZE = 0;
  public static int ALL_SIZE = 0;
  private static boolean eval = false;

  private LookupParameters pW;
  private LookupParameters pT;
  private Parameters pL2th;
  private Parameters pR2th;
  private Parameters pPreth;
  private Parameters pThbias;
  private Parameters pTh2t;
  private Parameters pTbias;
  private Parameters pW1;
  private Parameters pW2;
  private Parameters pB1;
  private Parameters pSOS;
  private LSTMBuilder l2rbuilder;
  private LSTMBuilder r2lbuilder;

  LSTMLanguageModel(Model model) {
    l2rbuilder = new LSTMBuilder(LAYERS, INPUT_DIM * 2 , HIDDEN_DIM, model);
    r2lbuilder = new LSTMBuilder(LAYERS, INPUT_DIM * 2, HIDDEN_DIM, model);
    pW = model.addLookupParameters(ALL_SIZE, Dim.create(INPUT_DIM));
    pT = model.addLookupParameters(TAG_SIZE, Dim.create(TAG_HIDDEN_DIM));
    pL2th = model.addParameters(Dim.create(TAG_HIDDEN_DIM, HIDDEN_DIM));
    pR2th = model.addParameters(Dim.create(TAG_HIDDEN_DIM, HIDDEN_DIM));
    pThbias = model.addParameters(Dim.create(TAG_HIDDEN_DIM));

    pTh2t = model.addParameters(Dim.create(TAG_SIZE, TAG_HIDDEN_DIM));
    pTbias = model.addParameters(Dim.create(TAG_SIZE));

    pW1 = model.addParameters(Dim.create(INPUT_DIM * 2 , INPUT_DIM));
    pW2 = model.addParameters(Dim.create(INPUT_DIM * 2 , INPUT_DIM));
    pB1 = model.addParameters(Dim.create(INPUT_DIM * 2));

    pPreth = model.addParameters(Dim.create(TAG_HIDDEN_DIM, TAG_HIDDEN_DIM));

    pSOS = model.addParameters(Dim.create(TAG_HIDDEN_DIM));
  }
  // return Expression of total loss
  Expression BuildTaggingGraph(final Vector<String> sw, final Vector<String> st, ComputationGraph cg,
                               AtomicDouble cor, AtomicDouble nTagged, HashMap<String, Vector<Double>> embeddings,
                               HashMap <String, Integer> tags, HashMap <String, Integer> all) {
    final int slen = sw.size();
    l2rbuilder.newGraph(cg);
    l2rbuilder.startNewSequence();
    r2lbuilder.newGraph(cg);
    r2lbuilder.startNewSequence();
    Expression iL2th = Expression.Creator.parameter(cg, pL2th);
    Expression iR2th = Expression.Creator.parameter(cg, pR2th);
    Expression iPreth = Expression.Creator.parameter(cg, pPreth);
    Expression iThbias = Expression.Creator.parameter(cg, pThbias);
    Expression iTh2t = Expression.Creator.parameter(cg, pTh2t);
    Expression iTbias = Expression.Creator.parameter(cg, pTbias);
    Expression iW1 = Expression.Creator.parameter(cg, pW1);
    Expression iW2 = Expression.Creator.parameter(cg, pW2);
    Expression iB1 = Expression.Creator.parameter(cg, pB1);

    Vector<Expression> errs = new Vector<Expression>();
    Vector<Expression> iWords = new Vector<Expression>();
    Vector<Expression> fwds = new Vector<Expression>();
    Vector<Expression> revs = new Vector<Expression>();

    iWords.setSize(slen);
    fwds.setSize(slen);
    revs.setSize(slen);
    //read sequence from left to reght

    Vector<Double> tmp = new Vector<Double>();
    for (int i = 0; i < INPUT_DIM; i++)
      tmp.addElement(0.0);
    Expression unkWord = Expression.Creator.input(cg, Dim.create(INPUT_DIM), tmp);

    l2rbuilder.addInput(Expression.Creator.rectify(Expression.Creator.affineTransform(Arrays.asList(
            iB1, iW1, unkWord, iW2,
            Expression.Creator.lookup(cg, pW, new Vector<Integer>(Arrays.asList(all.get("<s>"))))))));

    for (int t = 0; t < slen; ++t) {
      Vector<Double> in = embeddings.get(sw.get(t));
      if (in != null) iWords.set(t, Expression.Creator.rectify(Expression.Creator.affineTransform(Arrays.asList(
              iB1, iW1, Expression.Creator.input(cg, Dim.create(INPUT_DIM), in), iW2,
              Expression.Creator.lookup(cg, pW, new Vector<Integer>(Arrays.asList(all.get(sw.get(t)))))))));
      else {
        iWords.set(t, Expression.Creator.rectify(Expression.Creator.affineTransform(Arrays.asList(
                iB1, iW1, unkWord, iW2,
                Expression.Creator.lookup(cg, pW, new Vector<Integer>(Arrays.asList(all.get(sw.get(t)))))))));
        //if (!eval) { iWords.set(t, Expression.Creator.noise(iWords.get(t), 0.1)); }
      }
      fwds.set(t, l2rbuilder.addInput(iWords.get(t)));
    }

    //read sequence from right to left
    r2lbuilder.addInput(Expression.Creator.rectify(Expression.Creator.affineTransform(Arrays.asList(
            iB1, iW1, Expression.Creator.input(cg, Dim.create(INPUT_DIM), embeddings.get("</s>")), iW2,
            Expression.Creator.lookup(cg, pW, new Vector<Integer>(Arrays.asList(all.get("</s>"))))))));
    for (int  t = 0; t < slen; ++t)
      revs.set(slen - t - 1, r2lbuilder.addInput(iWords.get(slen - t - 1)));

    Expression preTag = Expression.Creator.parameter(cg, pSOS);
    for (int t = 0; t < slen; ++t) {
      nTagged.add(1.0);
      Expression iTh = Expression.Creator.rectify(Expression.Creator.affineTransform(
              new Vector<Expression>(Arrays.asList(iThbias, iL2th, fwds.get(t), iR2th, revs.get(t), iPreth, preTag))));
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
      preTag = Expression.Creator.lookup(cg, pT, new Vector<Integer>(Arrays.asList(besti)));
      if (tags.get(st.get(t)) == besti) cor.add(1.0);
      Expression iErr = Expression.Creator.pickNegLogSoftmax(iT, new Vector<Integer>(Arrays.asList(tags.get(st.get(t)))));
      errs.addElement(iErr);
    }
    return Expression.Creator.sum(errs);
  }

  public static void setEval(boolean eval_) {
    eval = eval_;
  }
}


public class CTB51 {
  static HashMap <String, Vector<Double> > embeddings = new HashMap<String, Vector<Double>>();
  static HashMap <String, Integer> tags = new HashMap<String, Integer>();
  static HashMap <String, Integer> all = new HashMap<String, Integer>();
  static double best;

  public static String transform(String str) {
    String res = "";
    int flag = 0;
    for (int i = 0; i < str.length(); i++) {
      if (str.charAt(i) >= '0' && str.charAt(i) <= '9' || (flag == 1 && str.charAt(i) == '.')) {
        flag = 1;
      } else {
        if (flag == 1){
          res += "##";
          flag = 0;
        }
        res += str.charAt(i);
      }
    }
    if (flag == 1) res += "##";
    return res;
  }
  public static void readFile(String fileName, Vector<Vector<String>> x, Vector<Vector<String>> y) {
    int lc = 0;
    int toks = 0;
    try {
      BufferedReader reader = new BufferedReader(new FileReader(fileName));
      String line = null;
      while((line = reader.readLine()) != null){
        ++lc;
        Vector<String> tx = new Vector<String>();
        Vector<String> ty = new Vector<String>();
        String[] item = line.split(" ");
        for (int i = 0; i < item.length; ++i) {
          tx.addElement(item[i].split("_")[0]);
          ty.addElement(item[i].split("_")[1]);

          String word = transform(tx.lastElement());
          tx.set(tx.size() - 1, word);
          if (all.get(word) == null) {
            all.put(word, all.size());
          }

          if (tags.get(ty.lastElement()) == null) {
            tags.put(ty.lastElement(), tags.size());
          }
        }
        assert(x.size() == y.size());
        x.addElement(tx);
        y.addElement(ty);
        toks += x.size();
      }
      System.err.println(lc + " lines, " + toks + " tokens, " + embeddings.size() + " types");
      System.err.println("Tags: " + tags.size());
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public static void readEmbedding(String fileName) {
    try {
      BufferedReader reader = new BufferedReader(new FileReader(fileName));
      String line = null;
      line = reader.readLine();
      int n = Integer.parseInt(line.split(" ")[0]);
      int dim = Integer.parseInt(line.split(" ")[1]) ;
      int cnt = 0;
      while((line = reader.readLine()) != null){
        cnt++;
        String word = transform(line.split(" ")[0]);
        Vector<Double> e = new Vector<Double>();
        e.setSize(dim);
        for (int i = 1; i <= dim; i++)
          e.set(i - 1, Double.parseDouble(line.split(" ")[i]));
        if (all.get(word) != null && embeddings.get(word) == null) embeddings.put(word, e);
        if (cnt % 1000 == 0) System.err.println("cur : " + cnt + " n : " + n + "  " + cnt * 100.0 / n + "%");
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
  public static void runOnDev(LSTMLanguageModel lm, Model model, Vector<Vector<String>> devX, Vector<Vector<String>> devY) {
    long last = new Date().getTime();
    double dloss = 0.0;
    AtomicDouble dcorr = new AtomicDouble(0.0);
    AtomicDouble dtags = new AtomicDouble(0.0);
    lm.setEval(true);
    for (int i = 0; i < devX.size(); ++i) {
      ComputationGraph cg = new ComputationGraph();
      lm.BuildTaggingGraph(devX.get(i), devY.get(i), cg, dcorr, dtags, embeddings, tags, all);
      dloss += TensorUtils.toScalar(cg.incrementalForward());
    }
    lm.setEval(false);
    if (dloss < best) {
      best = dloss;
      SerializationUtils.save("CTB51.obj", model);
    }
    System.err.println("\n***DEV E = "
            + (dloss / dtags.doubleValue()) + " ppl = " + Math.exp(dloss / dtags.doubleValue())
            + " acc = " + (dcorr.doubleValue() / dtags.doubleValue()) + " [consume = " + (new Date().getTime() - last) / 1000.0 + "s]");
  }


  public static void main(String args[]) {
    if (args.length != 4) {
      System.err.println("need input the file names : [train] [dev] [test] [embedding]");
      return;
    }

    String trainName = args[0];
    String devName = args[1];
    String testName = args[2];
    String embeddingName = args[3];


    Vector<Vector<String>> trainX = new Vector<Vector<String>>();
    Vector<Vector<String>> trainY = new Vector<Vector<String>>();
    System.err.println("Reading training data from "  + trainName + "...") ;

    readFile(trainName, trainX, trainY);

    Vector<Vector<String>> devX = new Vector<Vector<String>>();
    Vector<Vector<String>> devY = new Vector<Vector<String>>();
    System.err.println("Reading dev data from "  + devName + "...") ;
    readFile(devName, devX, devY);

    Vector<Vector<String>> testX = new Vector<Vector<String>>();
    Vector<Vector<String>> testY = new Vector<Vector<String>>();
    System.err.println("Reading test data from "  + testName + "...") ;
    readFile(testName, testX, testY);

    all.put("</s>", all.size());

    System.err.println("Reading embedding data from "  + embeddingName + "...") ;
    readEmbedding(embeddingName);

    int maxIteration = 10;
    int numInstances = trainX.size(); //Math.min(2000, trainX.size());
    if (args.length >= 5) {
      maxIteration = Integer.parseInt(args[4]);
    }
    if (args.length >= 6) {
      numInstances = Math.min(trainX.size(), Integer.parseInt(args[5]));
    }

    best = 1e100;
    Model model = new Model();
    boolean useMomentum = false;
    AbstractTrainer sgd = null;
    if (useMomentum)
      sgd = new MomentumSGDTrainer(model);
    else
      sgd = new SimpleSGDTrainer(model);

    all.put("<s>", all.size());
    LSTMLanguageModel.ALL_SIZE = all.size();
    LSTMLanguageModel.INPUT_DIM = embeddings.get("</s>").size();
    LSTMLanguageModel.TAG_SIZE = tags.size();

    LSTMLanguageModel lm = new LSTMLanguageModel(model);
    //SerializationUtils.loadModel("CTB51.obj", model);
    Vector<Integer> order = new Vector<Integer>();
    for (int i = 0; i < trainX.size(); i++)
      order.addElement(i);

    long last = new Date().getTime();
    long tot = last;
    for (int iteration = 0; iteration < maxIteration; ++iteration) {
      double loss = 0.0f;
      AtomicDouble correct = new AtomicDouble(0.0);
      AtomicDouble ttags = new AtomicDouble(0.0);
      Collections.shuffle(order);
      for (int i = 0; i < trainX.size(); i++) {
        int index = order.get(i);
        ComputationGraph cg = new ComputationGraph();
        lm.BuildTaggingGraph(trainX.get(index), trainY.get(index), cg, correct, ttags, embeddings, tags, all);
        loss += TensorUtils.toScalar(cg.incrementalForward());
        cg.backward();
        sgd.update(1.0);

        if (i + iteration > 0 && i % 1250 == 0) runOnDev(lm, model, devX, devY);
        if (i + iteration > 0 && i % 50 == 0) {
          System.err.println("E = " + (loss / ttags.doubleValue()) + " ppl = " + Math.exp(loss / ttags.doubleValue())
                  + " (acc = " + (correct.doubleValue() / ttags.doubleValue()) + ")" + " iterations : " + iteration
                  + " lines : " + i + "[consume = " + (new Date().getTime() - last) / 1000.0 + "s]");
          last = new Date().getTime();
        }
      }
      sgd.updateEpoch();
      System.out.println("Iteration Time : " + (new Date().getTime() - tot) / 1000.0 + "s]");
      tot = new Date().getTime();
    }

    double loss = 0.0;
    AtomicDouble correct = new AtomicDouble(0.0);
    AtomicDouble ttags = new AtomicDouble(0.0);
    SerializationUtils.loadModel("CTB51.obj", model);
    for (int i = 0; i < testX.size(); i++) {
      ComputationGraph cg = new ComputationGraph();
      lm.BuildTaggingGraph(testX.get(i), testY.get(i), cg, correct, ttags, embeddings, tags, all);
      loss += TensorUtils.toScalar(cg.incrementalForward());
    }
    System.err.println("E = " + (loss / ttags.doubleValue()) + " ppl = " + Math.exp(loss / ttags.doubleValue())
            + " (acc = " + (correct.doubleValue() / ttags.doubleValue()) + ")");
  }
}
