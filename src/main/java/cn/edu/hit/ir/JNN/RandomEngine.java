package cn.edu.hit.ir.JNN;

import org.nd4j.linalg.api.rng.DefaultRandom;

/**
 * A singleton class for random generator.
 */
public class RandomEngine {
  private static RandomEngine INSTANCE = null;
  private static long seed = 1024;
  public DefaultRandom rnd;

  public static RandomEngine getInstance() {
    if (INSTANCE == null) {
      INSTANCE = new RandomEngine();
    }
    return INSTANCE;
  }

  protected RandomEngine() {
    rnd = new DefaultRandom(seed);
  }
}
