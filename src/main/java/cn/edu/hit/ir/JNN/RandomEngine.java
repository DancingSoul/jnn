package cn.edu.hit.ir.JNN;

import java.util.Random;

/**
 * A singleton class for random generator.
 */
public class RandomEngine {
  private static RandomEngine INSTANCE = null;
  private static long seed = 1024;
  public Random rnd;

  public static RandomEngine getInstance() {
    if (INSTANCE == null) {
      INSTANCE = new RandomEngine();
    }
    return INSTANCE;
  }

  protected RandomEngine() {
    rnd = new Random(seed);
  }
}
