package nds;

import java.util.*;
import java.util.stream.Collectors;

public class RandomPerformance {
    private static double randomCube(int n, int dim, int times, int threshold, boolean silent) {
        double[][] points = new double[n][dim];
        int[] result = new int[n];

        ParallelNDS.Sorter sorter = ParallelNDS.getSorter(n, dim, threshold);
//        OriginalNDS.Sorter sorter = OriginalNDS.getSorter(n, dim);

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
            System.out.printf("    n = %d, dim = %2d, times = %d: average %.2e, min %.2e, max %.2e, median %.2e%n",
                    n, dim, times, sum / times / 1e6, nanos[0] / 1e6, nanos[times - 1] / 1e6, median / 1e6);
        }
        return median/1e6;
    }

    public static void main(String[] args) {
        System.out.println("randomCube:");
        for (int i = 1; i <= 100; ++i) {
            randomCube(1000, i / 10, 10, 64, true);
        }
        System.out.println("    warmed up");

//        int size = Integer.parseInt(System.getProperty("points"));
        for (int n : new int[]{ /*100, 1000,*/ 20000}) {
            List<Double> medians = new ArrayList<>();
            for (int d = 3; d <= 15; ++d) {
                medians.add(randomCube(n, d, 25, 64, false));
            }
            String s = (String) medians.stream()
                    .map(x -> String.format("%.2e", x))
                    .collect(Collectors.joining(", "));
            System.out.println("{\n    name: " + ",\n" + "    data: [" + s + "]\n},");
            System.out.println("    ------------------------------------");
        }
    }
}
