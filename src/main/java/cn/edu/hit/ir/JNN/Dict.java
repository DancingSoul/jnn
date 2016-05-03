package cn.edu.hit.ir.JNN;

import java.util.HashMap;
import java.util.Vector;

/**
 * Created by dancingsoul on 2016/4/26.
 */
public class Dict {
  private boolean frozen;
  private boolean mapUnk;
  private int unkID;
  private Vector<String> words_;
  private HashMap<String, Integer> d_;

  public Dict() {
    words_ = new Vector<String>();
    d_ = new HashMap<String, Integer>();
    frozen = false;
    mapUnk = false;
    unkID = -1;
  }
  public final int size() {
    return words_.size();
  }
  public final boolean contains(final String words) {
    return d_.get(words) != null;
  }
  public void freeze() {
    frozen = true;
  }
  public boolean isFrozen() {
    return frozen;
  }
  public final int convert(final String word) {
    Integer i = d_.get(word);
    if (i == null) {
      if (frozen) {
        if (mapUnk) {
          return unkID;
        } else {
          System.err.println(mapUnk);
          System.err.println("Unknown word encounterer : " + word);
          throw new RuntimeException("Unknown word encounterer in frozen dictionary");
        }
      }
      words_.addElement(word);
      d_.put(word, words_.size() - 1);
      return words_.size() - 1;
    } else {
      return i.intValue();
    }
  }

  public final String convert(final int id) {
    assert(id < (int)words_.size());
    return words_.get(id);
  }
  public void setUnk(final String word) {
    if (!frozen)
      throw new RuntimeException("please call setUnk() only after dictionary is frozen");
    if (mapUnk)
      throw new RuntimeException("set UNK more than one time");
    //temporarily unfrozen the dictionary to allow the add of the UNK
    frozen = false;
    unkID = convert(word);
    frozen = true;
    mapUnk = true;
  }
  public void clear() {
    words_.clear();
    d_.clear();
  }
}
