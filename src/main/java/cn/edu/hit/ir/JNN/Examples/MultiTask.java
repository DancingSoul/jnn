package cn.edu.hit.ir.JNN.Examples;

import cn.edu.hit.ir.JNN.*;
import cn.edu.hit.ir.JNN.Builders.LSTMBuilder;
import cn.edu.hit.ir.JNN.Trainers.AbstractTrainer;
import cn.edu.hit.ir.JNN.Trainers.MomentumSGDTrainer;
import cn.edu.hit.ir.JNN.Trainers.SimpleSGDTrainer;
import cn.edu.hit.ir.JNN.Utils.SerializationUtils;
import cn.edu.hit.ir.JNN.Utils.TensorUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

/**
 * Created by dancingsoul on 2016/5/29.
 */

class LSTMLanguageModelForMultiTask {
  private final static double pdrop = 0.5;
  private final static int LAYERS = 1;
  public static int INPUT_DIM = 0;
  private final static int HIDDEN_DIM = 100;
  private final static int TAG_HIDDEN_DIM= 32;
  public static int[] TAG_SIZE;
  public static int ALL_SIZE = 0;
  private static int TYPE_SIZE = 0;
  private static boolean eval = false;

  private LookupParameters pW;
  //private LookupParameters pT;
  private Parameters[] pL2th;
  private Parameters[] pR2th;
  //private Parameters pPreth;
  private Parameters[] pThbias;
  private Parameters[] pTh2t;
  private Parameters[] pTbias;
  private Parameters pW1;
  private Parameters pW2;
  private Parameters pB1;
  private Parameters pEOS;
  private Parameters pSOS;
  //private Parameters pUnk;
  private LSTMBuilder l2rbuilder;
  private LSTMBuilder r2lbuilder;

  LSTMLanguageModelForMultiTask(Model model) {
    l2rbuilder = new LSTMBuilder(LAYERS, INPUT_DIM , HIDDEN_DIM, model);
    r2lbuilder = new LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model);
    pW = model.addLookupParameters(ALL_SIZE, Dim.create(INPUT_DIM));

    //pUnk = model.addParameters(Dim.create(INPUT_DIM));

    TYPE_SIZE = TAG_SIZE.length;

    pL2th = new Parameters[TYPE_SIZE];
    pR2th = new Parameters[TYPE_SIZE];
    pThbias = new Parameters[TYPE_SIZE];
    pTh2t = new Parameters[TYPE_SIZE];
    pTbias = new Parameters[TYPE_SIZE];

    for (int i = 0; i < TYPE_SIZE; ++i) {
      pL2th[i] = model.addParameters(Dim.create(TAG_HIDDEN_DIM, HIDDEN_DIM));
      pR2th[i] = model.addParameters(Dim.create(TAG_HIDDEN_DIM, HIDDEN_DIM));
      pThbias[i] = model.addParameters(Dim.create(TAG_HIDDEN_DIM));
      pTh2t[i] = model.addParameters(Dim.create(TAG_SIZE[i], TAG_HIDDEN_DIM));
      pTbias[i] = model.addParameters(Dim.create(TAG_SIZE[i]));
    }

    pSOS = model.addParameters(Dim.create(INPUT_DIM));
    pEOS = model.addParameters(Dim.create(INPUT_DIM));

    pW1 = model.addParameters(Dim.create(INPUT_DIM , INPUT_DIM));
    pW2 = model.addParameters(Dim.create(INPUT_DIM , INPUT_DIM));
    pB1 = model.addParameters(Dim.create(INPUT_DIM));

    //pT = model.addLookupParameters(TAG_SIZE, Dim.create(TAG_HIDDEN_DIM));
    //pPreth = model.addParameters(Dim.create(TAG_HIDDEN_DIM, TAG_HIDDEN_DIM));
  }
  // return Expression of total loss
  Expression BuildTaggingGraph(final Vector<Integer> sw, final Vector<Integer> st, ComputationGraph cg,
                               AtomicDouble cor, AtomicDouble nTagged, HashMap<Integer, Vector<Double>> embeddings, int type) {
    final int slen = sw.size();
    l2rbuilder.newGraph(cg);
    l2rbuilder.startNewSequence();
    r2lbuilder.newGraph(cg);
    r2lbuilder.startNewSequence();

    Expression iL2th = Expression.Creator.parameter(cg, pL2th[type]);
    Expression iR2th = Expression.Creator.parameter(cg, pR2th[type]);
    Expression iThbias = Expression.Creator.parameter(cg, pThbias[type]);
    Expression iTh2t = Expression.Creator.parameter(cg, pTh2t[type]);
    Expression iTbias = Expression.Creator.parameter(cg, pTbias[type]);


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

    l2rbuilder.addInput(Expression.Creator.parameter(cg, pSOS));

    for (int t = 0; t < slen; ++t) {
      Vector<Double> in = embeddings.get(sw.get(t));
      if (in != null) iWords.set(t, Expression.Creator.rectify(Expression.Creator.affineTransform(Arrays.asList(
              iB1, iW1, Expression.Creator.input(cg, Dim.create(INPUT_DIM), in), iW2,
              Expression.Creator.lookup(cg, pW, new Vector<Integer>(Arrays.asList(sw.get(t))))))));
      else {
        iWords.set(t, Expression.Creator.rectify(
                Expression.Creator.lookup(cg, pW, new Vector<Integer>(Arrays.asList(sw.get(t))))));
        //if (!eval) { iWords.set(t, Expression.Creator.noise(iWords.get(t), 0.1)); }
      }
      fwds.set(t, l2rbuilder.addInput(iWords.get(t)));
    }

    //read sequence from right to left
    r2lbuilder.addInput(Expression.Creator.parameter(cg, pEOS));
    for (int  t = 0; t < slen; ++t)
      revs.set(slen - t - 1, r2lbuilder.addInput(iWords.get(slen - t - 1)));

    for (int t = 0; t < slen; ++t) {
      nTagged.add(1.0);
      Expression iTh = Expression.Creator.rectify(Expression.Creator.affineTransform(
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

      //System.out.println(t + "  " + st.get(t) + " " + dist.size());
      if (st.get(t) == besti) cor.add(1.0);
      Expression iErr = Expression.Creator.pickNegLogSoftmax(iT, new Vector<Integer>(Arrays.asList(st.get(t))));
      errs.addElement(iErr);
    }
    return Expression.Creator.sum(errs);
  }

  public static void setEval(boolean eval_) {
    eval = eval_;
  }
}


public class MultiTask {
  private static HashMap <Integer, Vector<Double> > embeddings = new HashMap<Integer, Vector<Double>>();
  private static HashMap <String, Integer> posTags = new HashMap<String, Integer>();
  private static HashMap <String, Integer> nerTags = new HashMap<String, Integer>();
  private static HashMap <String, Integer> allWords = new HashMap<String, Integer>();
  private static double best;

  public static void readPosFile(String fileName, Vector<Vector<Integer>> x, Vector<Vector<Integer>> y) {
    int lc = 0;
    int toks = 0;
    try {
      BufferedReader reader = new BufferedReader(new FileReader(fileName));
      String sentence = null;
      while((sentence = reader.readLine()) != null){
        ++lc;
        Vector<Integer> tx = new Vector<Integer>();
        Vector<Integer> ty = new Vector<Integer>();
        String[] words = sentence.split(" ");
        for (int i = 0; i < words.length; ++i) {
          String[] item = words[i].split("/");

          if (allWords.get(item[0]) == null) {
            allWords.put(item[0], allWords.size());
          }
          if (posTags.get(item[1]) == null) {
            posTags.put(item[1], posTags.size());
          }
          tx.addElement(allWords.get(item[0]));
          ty.addElement(posTags.get(item[1]));
        }
        assert(x.size() == y.size());
        x.addElement(tx);
        y.addElement(ty);
        toks += tx.size();
      }
      System.err.println(lc + " lines, " + toks + " tokens, ");
      System.err.println("Tags: " + posTags.size());
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public static void readNerFile(String fileName, Vector<Vector<Integer>> x, Vector<Vector<Integer>> y) {
    int lc = 0;
    int toks = 0;
    try {
      BufferedReader reader = new BufferedReader(new FileReader(fileName));
      String sentence = null;
      while((sentence = reader.readLine()) != null) {
        String str = null;
        while ((str = reader.readLine()).length() > 0) {
          sentence += '\n';
          sentence += str;
        }

        ++lc;
        Vector<Integer> tx = new Vector<Integer>();
        Vector<Integer> ty = new Vector<Integer>();
        String[] words = sentence.split("\n");
        for (int i = 0; i < words.length; ++i) {
          String[] item = words[i].split(" ");
          if (allWords.get(item[0]) == null) {
            allWords.put(item[0], allWords.size());
          }
          if (nerTags.get(item[3]) == null) {
            nerTags.put(item[3], nerTags.size());
          }
          tx.addElement(allWords.get(item[0]));
          ty.addElement(nerTags.get(item[3]));
        }
        assert(x.size() == y.size());
        x.addElement(tx);
        y.addElement(ty);
        toks += tx.size();
      }
      System.err.println(lc + " lines, " + toks + " tokens, ");
      System.err.println("Tags: " + nerTags.size());
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
        String[] item = line.split(" ");
        cnt++;
        String word = item[0];
        Vector<Double> e = new Vector<Double>();
        e.setSize(dim);
        for (int i = 1; i <= dim; i++)
          e.set(i - 1, Double.parseDouble(item[i]));
        if (allWords.get(word) != null && embeddings.get(word) == null) embeddings.put(allWords.get(word), e);
        if (cnt % 1000 == 0) System.err.println("cur : " + cnt + " n : " + n + "  " + cnt * 100.0 / n + "%");
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public static void runOnDev(LSTMLanguageModelForMultiTask lm, Model model, Vector<Vector<Integer>> devX,
                              Vector<Vector<Integer>> devY, int type){
    long last = new Date().getTime();
    double dloss = 0.0;
    AtomicDouble dcorr = new AtomicDouble(0.0);
    AtomicDouble dtags = new AtomicDouble(0.0);
    lm.setEval(true);
    for (int i = 0; i < devX.size(); ++i) {
      ComputationGraph cg = new ComputationGraph();
      lm.BuildTaggingGraph(devX.get(i), devY.get(i), cg, dcorr, dtags, embeddings, type);
      dloss += TensorUtils.toScalar(cg.incrementalForward());
    }
    lm.setEval(false);

    if (dloss < best) {
      String fileName = (type == 0 ? "pos" : "ner") + ".obj";
      best = dloss;
      SerializationUtils.save(fileName, model);
    }
    System.err.println("\n***DEV " +  (type == 0 ? "POS":  "NER") + " E = "
            + (dloss / dtags.doubleValue()) + " ppl = " + Math.exp(dloss / dtags.doubleValue())
            + " acc = " + (dcorr.doubleValue() / dtags.doubleValue()) + " [consume = " + (new Date().getTime() - last) / 1000.0 + "s]");
  }

  public static void runOnTest(LSTMLanguageModelForMultiTask lm, Model model, Vector<Vector<Integer>> devX,
                               Vector<Vector<Integer>> devY, int type) {

    String fileName = (type == 0 ? "pos" : "ner") + ".obj";
    SerializationUtils.loadModel(fileName, model);

    long last = new Date().getTime();
    double dloss = 0.0;
    AtomicDouble dcorr = new AtomicDouble(0.0);
    AtomicDouble dtags = new AtomicDouble(0.0);
    lm.setEval(true);
    for (int i = 0; i < devX.size(); ++i) {
      ComputationGraph cg = new ComputationGraph();
      lm.BuildTaggingGraph(devX.get(i), devY.get(i), cg, dcorr, dtags, embeddings, type);
      dloss += TensorUtils.toScalar(cg.incrementalForward());
    }
    lm.setEval(false);
    System.err.println("\n***TEST " +  (type == 0 ? "POS":  "NER") +"E = "
            + (dloss / dtags.doubleValue()) + " ppl = " + Math.exp(dloss / dtags.doubleValue())
            + " acc = " + (dcorr.doubleValue() / dtags.doubleValue()) + " [consume = " + (new Date().getTime() - last) / 1000.0 + "s]");
  }


  public static void main(String args[]) {
    if (args.length < 7) {
      System.err.println("need input the file names : [train_pos] [dev_pos] [test_pos] " +
              "[train_ner] [dev_ner] [test_ner] [embedding]");
      return;
    }

    String trainPosName = args[0];
    String devPosName = args[1];
    String testPosName = args[2];
    String trainNerName = args[3];
    String devNerName = args[4];
    String testNerName = args[5];
    String embeddingName = args[6];


    Vector<Vector<Integer>> trainPosX = new Vector<Vector<Integer>>();
    Vector<Vector<Integer>> trainPosY = new Vector<Vector<Integer>>();
    System.err.println("Reading training data from "  + trainPosName + "...") ;

    readPosFile(trainPosName, trainPosX, trainPosY);

    Vector<Vector<Integer>> devPosX = new Vector<Vector<Integer>>();
    Vector<Vector<Integer>> devPosY = new Vector<Vector<Integer>>();
    System.err.println("Reading dev data from "  + devPosName + "...") ;
    readPosFile(devPosName, devPosX, devPosY);

    Vector<Vector<Integer>> testPosX = new Vector<Vector<Integer>>();
    Vector<Vector<Integer>> testPosY = new Vector<Vector<Integer>>();
    System.err.println("Reading test data from "  + testPosName + "...") ;
    readPosFile(testPosName, testPosX, testPosY);

    Vector<Vector<Integer>> trainNerX = new Vector<Vector<Integer>>();
    Vector<Vector<Integer>> trainNerY = new Vector<Vector<Integer>>();
    System.err.println("Reading training data from "  + trainNerName + "...") ;

    readNerFile(trainNerName, trainNerX, trainNerY);

    Vector<Vector<Integer>> devNerX = new Vector<Vector<Integer>>();
    Vector<Vector<Integer>> devNerY = new Vector<Vector<Integer>>();
    System.err.println("Reading dev data from "  + devNerName + "...") ;
    readNerFile(devNerName, devNerX, devNerY);

    Vector<Vector<Integer>> testNerX = new Vector<Vector<Integer>>();
    Vector<Vector<Integer>> testNerY = new Vector<Vector<Integer>>();
    System.err.println("Reading test data from "  + testNerName + "...") ;
    readNerFile(testNerName, testNerX, testNerY);

    System.err.println("Reading embedding data from "  + embeddingName + "...") ;
    readEmbedding(embeddingName);

    int maxIteration = 10;
    int numInstances = trainPosX.size() + trainNerX.size(); //Math.min(2000, trainX.size());
    if (args.length >= 8) {
      maxIteration = Integer.parseInt(args[6]);
    }
    if (args.length >= 9) {
      numInstances = Math.min(numInstances, Integer.parseInt(args[7]));
    }

    best = 1e100;
    Model model = new Model();
    boolean useMomentum = false;
    AbstractTrainer sgd = null;
    if (useMomentum)
      sgd = new MomentumSGDTrainer(model);
    else
      sgd = new SimpleSGDTrainer(model);

    LSTMLanguageModelForMultiTask.ALL_SIZE = allWords.size();
    LSTMLanguageModelForMultiTask.INPUT_DIM = 100;
    LSTMLanguageModelForMultiTask.TAG_SIZE = new int[]{posTags.size(), nerTags.size()};
    LSTMLanguageModelForMultiTask lm = new LSTMLanguageModelForMultiTask(model);
    //SerializationUtils.loadModel("pos_ner.obj", model);
    Vector<Integer> order = new Vector<Integer>();
    for (int i = 0; i < trainPosX.size() + trainNerX.size(); i++)
      order.addElement(i);

    long last = new Date().getTime();
    long tot = last;
    for (int iteration = 0; iteration < maxIteration; ++iteration) {
      double loss = 0.0f;
      AtomicDouble correct = new AtomicDouble(0.0);
      AtomicDouble ttags = new AtomicDouble(0.0);
      Collections.shuffle(order);
      for (int i = 0; i < order.size(); i++) {
        int index = order.get(i);
        ComputationGraph cg = new ComputationGraph();
        if (index < trainPosX.size()) {
          lm.BuildTaggingGraph(trainPosX.get(index), trainPosY.get(index), cg, correct, ttags, embeddings, 0);
        } else {
          lm.BuildTaggingGraph(trainNerX.get(index - trainPosX.size()), trainNerY.get(index - trainPosX.size()), cg, correct, ttags, embeddings, 1);
        }
        loss += TensorUtils.toScalar(cg.incrementalForward());
        cg.backward();
        sgd.update(1.0);

        if (i + iteration > 0 && i % 3000 == 0) {
          runOnDev(lm, model, devPosX, devPosY, 0);
          runOnDev(lm, model, devNerX, devNerY, 1);
        }

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

    runOnTest(lm, model, testPosX, testPosY, 0);
    runOnTest(lm, model, testNerX, testNerY, 1);
  }
}
