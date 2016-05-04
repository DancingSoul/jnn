package cn.edu.hit.ir.JNN.Utils;

import java.util.Vector;
import cn.edu.hit.ir.JNN.Dict;

/**
 * Created by dancingsoul on 2016/4/26.
 */
public class DictUtils {
  public static Vector<Integer> readSentence(final String line, Dict sd) {
    Vector<Integer> res = new Vector<Integer>();
    return res;
  }
  public static void readSentencePair(final String sentence, Vector<Integer> s, Dict sd, Vector<Integer> t, Dict td) {
    String[] words = sentence.split("\t");
    for (int i = 0; i < words.length; ++i) {
      String[] item = words[i].split("_");
      s.addElement(sd.convert(item[0]));
      t.addElement(td.convert(item[1]));
    }
  }
}
