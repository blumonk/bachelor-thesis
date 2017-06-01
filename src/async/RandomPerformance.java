package async;

import java.util.Arrays;
import java.util.Random;

public class RandomPerformance {
    private static double randomCube(int n, int dim, int times, int ndb, int nda, boolean silent) {
        double[][] points = new double[n][dim];
        int[] result = new int[n];

        Sorter sorter = Async.getSorter(n, dim, ndb, nda);

        long[] nanos = new long[times];
        Random random = new Random();
        System.gc();
        System.gc();
        for (int attempt = 0; attempt < times; ++attempt) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < dim; ++j) {
                    points[i][j] = random.nextDouble();
                }
            }
            if (dim > 3) {
                long t0 = System.nanoTime();
                sorter.sort(points, result);
                nanos[attempt] = System.nanoTime() - t0;
            } else {
                long t0 = System.nanoTime();
                for (int i = 0; i < 10; ++i) {
                    sorter.sort(points, result);
                }
                nanos[attempt] = (System.nanoTime() - t0) / 10;
            }
        }

        Arrays.sort(nanos);
        double sum = 0;
        for (long nano : nanos) sum += nano;
        double median = (nanos[times / 2] + nanos[times - 1 - times / 2]) / 2.0;
        if (!silent) {
            System.out.printf("dim = %2d ndb = %3d nda = %5d average %.2e, min %.2e, max %.2e, median %.2e,%n",
                    dim, ndb, nda, sum / times / 1e6, nanos[0] / 1e6, nanos[times - 1] / 1e6, median / 1e6);
        }
        return median/1e6;
    }

    public static void main(String[] args) {
        for (int i = 20; i <= 150; ++i) {
            randomCube(1000, i / 10, 10, 64, 64, true);
        }
        int threads = Integer.parseInt(System.getProperty("threads"));
        int points = Integer.parseInt(System.getProperty("points"));
        System.out.println("N = " + points + ", " + threads + " threads");
        for (int n : new int[]{ points }) {
            for (int d = 3; d <= 15; ++d) {
                for (int ndb: new int[]{64, 128}) {
                    for (int nda: new int[]{
                            points / 32 - 50,
                            points / 16 - 50,
                            points / 8 - 50,
                            points / 4 - 50,
                            points / 2 - 50,
                            points
                    }) {
                        randomCube(n, d, 25, ndb, nda, false);
                    }
                    System.out.println();
                }
                System.out.println();
            }
            System.out.println("    ------------------------------------");
        }
    }
}
