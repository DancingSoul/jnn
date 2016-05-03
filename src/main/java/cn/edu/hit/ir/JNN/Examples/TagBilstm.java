package cn.edu.hit.ir.JNN.Examples;

import cn.edu.hit.ir.JNN.Dict;

import java.util.Vector;
/**
 * Created by dancingsoul on 2016/4/27.
 */
public class TagBilstm {
  static String trainName = "";
  static String devName = "";
  static Dict d = new Dict();
  static Dict td = new Dict();
  static int VOCAB_SIZE = 0;
  static int TAG_SIZE = 0;
  public static void readFile(String fileName, Vector<Vector<Integer>> x, Vector<Vector<Integer>> y) {

  }

  public static void main(String args[]) {
    Vector<Vector<Integer>> trainX = new Vector<Vector<Integer>>();
    Vector<Vector<Integer>> trainY = new Vector<Vector<Integer>>();
    System.err.println("Reading training data from "  + trainName + "...") ;

    readFile(trainName, trainX, trainY);
    d.freeze();
    td.freeze();
    VOCAB_SIZE = d.size();
    TAG_SIZE = td.size();
    Vector<Vector<Integer>> devX = new Vector<Vector<Integer>>();
    Vector<Vector<Integer>> devY = new Vector<Vector<Integer>>();

  }
}
