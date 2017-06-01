package nds;

import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.function.IntBinaryOperator;


public final class ParallelRedBlackTree {

    private static final IntBinaryOperator MAXIMUM = Math::max;

    public static Sorter getSorter(int size, int dim, int threshold) {
        if (dim < 0 || size < 0) {
            throw new IllegalArgumentException("Size or dimension is negative");
        }
        return new SorterXD(size, dim, threshold);
    }

    // XD sorter: the general case.
    private static final class SorterXD extends Sorter {
        private final int[] indices;
        private final int[] eqComp;
        private final MergeSorter sorter;
        private final int threshold;

        private double[][] input;
        private AtomicIntegerArray output;

        private final ThreadLocal<int[]> swap = ThreadLocal.withInitial(() -> new int[size]);

        private final ThreadLocal<int[]> fenwickData = ThreadLocal.withInitial(() -> new int[size]);
        private final ThreadLocal<double[]> fenwickPivots = ThreadLocal.withInitial(() -> new double[size]);
        private final ThreadLocal<Integer> fenwickSize = ThreadLocal.withInitial(() -> 0);

        private final ThreadLocal<RedBlackTree> tree = ThreadLocal.withInitial(() -> new RedBlackTree(size));

        private final ThreadLocal<Integer> lessThan = ThreadLocal.withInitial(() -> 0);
        private final ThreadLocal<Integer> equalTo = ThreadLocal.withInitial(() -> 0);
        private final ThreadLocal<Integer> greaterThan = ThreadLocal.withInitial(() -> 0);

        private static final int N_THREADS = 4;
        private static final ForkJoinPool pool = new ForkJoinPool(N_THREADS);
//        private static final ForkJoinPool pool = new ForkJoinPool(Integer.parseInt(System.getProperty("threads")));

        private void fenwickInit(int[] inIdx, int from, int until) {
            double[] fenwickPivots = this.fenwickPivots.get();
            int n = inIdx.length;
            for (int i = 0, j = from; j < until; ++i, ++j) {
                fenwickPivots[i] = input[inIdx[j]][1];
            }
            Arrays.sort(fenwickPivots, 0, until - from);
            int last = 0;
            for (int i = 1; i < until - from; ++i) {
                if (fenwickPivots[i] != fenwickPivots[last]) {
                    fenwickPivots[++last] = fenwickPivots[i];
                }
            }
            // first N (fenwickSize) unique elements sorted
            int size = last + 1;
            this.fenwickSize.set(size);
            Arrays.fill(fenwickData.get(), 0, size, -1);
        }

        // index of the Pivot <= key
        private int fenwickIndex(double key) {
            int left = -1, right = this.fenwickSize.get();
            double[] fenwickPivots = this.fenwickPivots.get();
            while (right - left > 1) {
                int mid = (left + right) >>> 1;
                if (fenwickPivots[mid] <= key) {
                    left = mid;
                } else {
                    right = mid;
                }
            }
            return left;
        }

        private void fenwickSet(double key, int value) {
            int fwi = fenwickIndex(key);
            int size = this.fenwickSize.get();
            int[] fenwickData = this.fenwickData.get();
            while (fwi < size) {
                fenwickData[fwi] = Math.max(fenwickData[fwi], value);
                fwi |= fwi + 1;
            }
        }

        private int fenwickQuery(double key) {
            int fwi = fenwickIndex(key);
            int[] fenwickData = this.fenwickData.get();
            if (fwi >= fenwickSize.get() || fwi < 0) {
                return -1;
            } else {
                int rv = -1;
                while (fwi >= 0) {
                    rv = Math.max(rv, fenwickData[fwi]);
                    fwi = (fwi & (fwi + 1)) - 1;
                }
                return rv;
            }
        }

        public SorterXD(int size, int dim, int threshold) {
            super(size, dim);
            this.threshold = threshold;
            indices = new int[size];
            eqComp = new int[size];
            sorter = new MergeSorter(size);
        }

        protected void sortImpl(double[][] input, int[] output) {
            for (int i = 0; i < size; ++i) {
                indices[i] = i;
            }
            Arrays.fill(output, 0);
            sorter.lexSort(indices, 0, size, input, eqComp);
            this.input = input;
            this.output = new AtomicIntegerArray(size);
            sort(0, size, dim - 1);
            for (int i = 0; i < size; ++i)
                output[i] = this.output.get(i);
            this.input = null;
            this.output = null;
        }

        // Median of medians algorithm
        private double medianInSwap(int from, int until, int dimension) {
            int to = until - 1;
            int med = (from + until) >>> 1;
            int[] swap = this.swap.get();
            ThreadLocalRandom random = ThreadLocalRandom.current();
            while (from <= to) {
                double pivot = input[swap[from + random.nextInt(to - from + 1)]][dimension];
                int ff = from, tt = to;
                while (ff <= tt) {
                    while (input[swap[ff]][dimension] < pivot) ++ff;
                    while (input[swap[tt]][dimension] > pivot) --tt;
                    if (ff <= tt) {
                        int tmp = swap[ff];
                        swap[ff] = swap[tt];
                        swap[tt] = tmp;
                        ++ff;
                        --tt;
                    }
                }
                if (med <= tt) {
                    to = tt;
                } else if (med >= ff) {
                    from = ff;
                } else {
                    return input[swap[med]][dimension];
                }
            }
            return input[swap[from]][dimension];
        }

        // Split to [less][eq][greater], O(N). Preserves relative lexicographic order in parts.
        private void split3(int[] inIdx, int from, int until, int dimension, double median) {
            int lt = 0, eq = 0, gt = 0;
            for (int i = from; i < until; ++i) {
                int cmp = Double.compare(input[inIdx[i]][dimension], median);
                if (cmp < 0) {
                    ++lt;
                } else if (cmp == 0) {
                    ++eq;
                } else {
                    ++gt;
                }
            }
            int lessThanPtr = 0, equalToPtr = lt, greaterThanPtr = lt + eq;
            int[] swap = this.swap.get();
            for (int i = from; i < until; ++i) {
                int cmp = Double.compare(input[inIdx[i]][dimension], median);
                if (cmp < 0) {
                    swap[lessThanPtr++] = inIdx[i];
                } else if (cmp == 0) {
                    swap[equalToPtr++] = inIdx[i];
                } else {
                    swap[greaterThanPtr++] = inIdx[i];
                }
            }
            lessThan.set(lt);
            equalTo.set(eq);
            greaterThan.set(gt);
            System.arraycopy(swap, 0, inIdx, from, until - from);
        }

        // Required to restore lexicographical order!!!
        private void merge(int[] idx, int from, int mid, int until) {
            int p0 = from, p1 = mid;
            int[] swap = this.swap.get();
            for (int i = from; i < until; ++i) {
                if (p0 == mid || p1 < until && eqComp[idx[p1]] < eqComp[idx[p0]]) {
                    swap[i] = idx[p1++];
                } else {
                    swap[i] = idx[p0++];
                }
            }
            System.arraycopy(swap, from, idx, from, until - from);
        }

        private int[] merge(int[] lIdx, int[] hIdx) {
            int ln = lIdx.length, hn = hIdx.length, total = ln + hn;
            int[] result = new int[total];
            int pl = 0, ph = 0;
            for (int i = 0; i < total; ++i) {
                if (pl == ln || ph < hn && eqComp[hIdx[ph]] < eqComp[lIdx[pl]]) {
                    result[i] = hIdx[ph++];
                } else {
                    result[i] = lIdx[pl++];
                }
            }
            return result;
        }

        // SweepB
        private void sortHighByLow2D(int[] lIndices, int[] hIndices) {
            int li = 0, ln = lIndices.length, hn = hIndices.length;
            RedBlackTree t = tree.get();
            t.clear();
            for (int hi = 0; hi < hn; ++hi) {
                int currH = hIndices[hi];
                int eCurrH = eqComp[currH];
                while (li < ln && eqComp[lIndices[li]] < eCurrH) {
                    int currL = lIndices[li++];
                    t.insert(input[currL][1], output.get(currL));
                }
                output.accumulateAndGet(currH, t.maxBefore(input[currH][1]) + 1, MAXIMUM);
            }
        }

        private void sortHighByLow2D(int[] idx, int lFrom, int lUntil, int hFrom, int hUntil) {
            fenwickInit(idx, lFrom, lUntil);
            int li = lFrom;
            for (int hi = hFrom; hi < hUntil; ++hi) {
                int currH = idx[hi];
                int eCurrH = eqComp[currH];
                while (li < lUntil && eqComp[idx[li]] < eCurrH) {
                    int currL = idx[li++];
                    fenwickSet(input[currL][1], output.get(currL));
                }
                output.accumulateAndGet(currH, fenwickQuery(input[currH][1]) + 1, MAXIMUM);
            }
        }

        private void sortHighByLow(int[] idx, int lFrom, int lUntil, int hFrom, int hUntil, int dimension) {
            int lSize = lUntil - lFrom, hSize = hUntil - hFrom;
            if (lSize == 0 || hSize == 0) {
                return;
            }
            if (lSize == 1) {
                for (int hi = hFrom; hi < hUntil; ++hi) {
                    if (dominatesEq(idx[lFrom], idx[hi], dimension)) {
                        updateFront(idx[hi], idx[lFrom]);
                    }
                }
            } else if (hSize == 1) {
                for (int li = lFrom; li < lUntil; ++li) {
                    if (dominatesEq(idx[li], idx[hFrom], dimension)) {
                        updateFront(idx[hFrom], idx[li]);
                    }
                }
            } else if (dimension == 1) {
                sortHighByLow2D(idx, lFrom, lUntil, hFrom, hUntil);
            } else {
                int[] indL = new int[lSize];
                int[] indH = new int[hSize];
                System.arraycopy(idx, lFrom, indL, 0, lSize);
                System.arraycopy(idx, hFrom, indH, 0, hSize);

                System.arraycopy(idx, lFrom, swap.get(), 0, lSize);
                System.arraycopy(idx, hFrom, swap.get(), lSize, hSize);
                double median = medianInSwap(0, lSize + hSize, dimension);

                split3(idx, lFrom, lUntil, dimension, median);
                int lMidL = lFrom + lessThan.get(), lMidR = lMidL + equalTo.get();

                split3(idx, hFrom, hUntil, dimension, median);
                int hMidL = hFrom + lessThan.get(), hMidR = hMidL + equalTo.get();

                sortHighByLow(idx, lFrom, lMidL, hFrom, hMidL, dimension);
                sortHighByLow(idx, lFrom, lMidL, hMidL, hMidR, dimension - 1);
                sortHighByLow(idx, lMidL, lMidR, hMidL, hMidR, dimension - 1);

                merge(idx, lFrom, lMidL, lMidR);

                sortHighByLow(idx, lFrom, lMidR, hMidR, hUntil, dimension - 1);
                sortHighByLow(idx, lMidR, lUntil, hMidR, hUntil, dimension);

                System.arraycopy(indL, 0, idx, lFrom, lSize);
                System.arraycopy(indH, 0, idx, hFrom, hSize);
            }
        }

        // NDHelperB
        class SortHighByLow extends RecursiveAction {
            final int[] lIndices, hIndices;
            final int dimension, lSize, hSize;

            public SortHighByLow(int[] lIndices, int[] hIndices, int dimension) {
                this.lIndices = lIndices;
                this.hIndices = hIndices;
                this.dimension = dimension;
                this.lSize = lIndices.length;
                this.hSize = hIndices.length;
            }

            @Override
            protected void compute() {
                if (lSize == 0 || hSize == 0) {
                    return;
                }
                if (lSize == 1) {
                    for (int hi = 0; hi < hSize; ++hi) {
                        if (dominatesEq(lIndices[0], hIndices[hi], dimension)) {
                            updateFront(hIndices[hi], lIndices[0]);
                        }
                    }
                } else if (hSize == 1) {
                    for (int li = 0; li < lSize; ++li) {
                        if (dominatesEq(lIndices[li], hIndices[0], dimension)) {
                            updateFront(hIndices[0], lIndices[li]);
                        }
                    }
                } else if (dimension == 1) {
                    sortHighByLow2D(lIndices, hIndices);
                } else if (lSize < threshold || hSize < threshold) {
                    int[] whole = new int[lSize + hSize];
                    System.arraycopy(lIndices, 0, whole, 0, lSize);
                    System.arraycopy(hIndices, 0, whole, lSize, hSize);
                    sortHighByLow(whole, 0, lSize, lSize, lSize + hSize, dimension);
                } else {
                    System.arraycopy(lIndices, 0, swap.get(), 0, lSize);
                    System.arraycopy(hIndices, 0, swap.get(), lSize, hSize);
                    double median = medianInSwap(0, lSize + hSize, dimension);

                    split3(lIndices, 0, lSize, dimension, median);
                    int lMidL = lessThan.get(), lMidR = lMidL + equalTo.get();
                    int[] lLChunk = new int[lMidL], lMChunk = new int[lMidR - lMidL], lRChunk = new int[lSize - lMidR];
                    System.arraycopy(lIndices, 0, lLChunk, 0, lMidL);
                    System.arraycopy(lIndices, lMidL, lMChunk, 0, lMidR - lMidL);
                    System.arraycopy(lIndices, lMidR, lRChunk, 0, lSize - lMidR);

                    split3(hIndices, 0, hSize, dimension, median);
                    int hMidL = lessThan.get(), hMidR = hMidL + equalTo.get();
                    int[] hLChunk = new int[hMidL], hMChunk = new int[hMidR - hMidL], hRChunk = new int[hSize - hMidR];
                    System.arraycopy(hIndices, 0, hLChunk, 0, hMidL);
                    System.arraycopy(hIndices, hMidL, hMChunk, 0, hMidR - hMidL);
                    System.arraycopy(hIndices, hMidR, hRChunk, 0, hSize - hMidR);

                    int[] lBigLChunk = merge(lLChunk, lMChunk);
                    SortHighByLow a = new SortHighByLow(lLChunk, hLChunk, dimension);
                    SortHighByLow b = new SortHighByLow(lLChunk.clone(), hMChunk, dimension - 1);
                    SortHighByLow c = new SortHighByLow(lMChunk, hMChunk.clone(), dimension - 1);
                    SortHighByLow d = new SortHighByLow(lBigLChunk, hRChunk, dimension - 1);
                    SortHighByLow e = new SortHighByLow(lRChunk, hRChunk.clone(), dimension);

                    a.fork(); b.fork(); c.fork(); d.fork();
                    e.compute();
                    a.join(); b.join(); c.join(); d.join();
                }
            }
        }

        // NDHelperA
        private void sort(int from, int until, int dimension) {
            int size = until - from;
            if (size == 2) {
                if (dominatesEq(indices[from], indices[from + 1], dimension)) {
                    updateFront(indices[from + 1], indices[from]);
                }
            } else if (size > 2) {
                if (dimension == 1) {
                    sort2D(from, until);
                } else {
                    if (allValuesEqual(from, until, dimension)) {
                        sort(from, until, dimension - 1);
                    } else {
                        System.arraycopy(indices, from, swap.get(), from, size);
                        double median = medianInSwap(from, until, dimension);

                        split3(indices, from, until, dimension, median);
                        int midL = from + lessThan.get(), midH = midL + equalTo.get();
                        final int lSize = midL - from, mSize = midH - midL, hSize = until - midH, lmSize = midH - from;
                        int[] lIndices, hIndices;

                        sort(from, midL, dimension);

                        if (lSize == 1) {
                            for (int hi = midL; hi < midH; ++hi)
                                if (dominatesEq(indices[from], indices[hi], dimension - 1))
                                    updateFront(indices[hi], indices[from]);
                        } else if (mSize == 1) {
                            for (int li = from; li < midL; ++li)
                                if (dominatesEq(indices[li], indices[midL], dimension - 1))
                                    updateFront(indices[midL], indices[li]);
                        } else if (dimension == 2) {
                            sortHighByLow2D(indices, from, midL, midL, midH);
                        } else {
                            sortHighByLow(indices, from, midL, midL, midH, dimension - 1);
                        }

                        sort(midL, midH, dimension - 1);
                        merge(indices, from, midL, midH);

                        if (lmSize == 1) {
                            for (int hi = midH; hi < until; ++hi)
                                if (dominatesEq(indices[from], indices[hi], dimension - 1))
                                    updateFront(indices[hi], indices[from]);
                        } else if (hSize == 1) {
                            for (int li = from; li < midH; ++li)
                                if (dominatesEq(indices[li], indices[midH], dimension - 1))
                                    updateFront(indices[midH], indices[li]);
                        } else if (dimension == 2) {
                            sortHighByLow2D(indices, from, midH, midH, until);
                        } else if (lmSize < threshold || hSize < threshold) {
                            sortHighByLow(indices, from, midH, midH, until, dimension - 1);
                        } else {
                            lIndices = new int[lmSize];
                            hIndices = new int[hSize];
                            System.arraycopy(indices, from, lIndices, 0, lmSize);
                            System.arraycopy(indices, midH, hIndices, 0, hSize);
                            pool.invoke(new SortHighByLow(lIndices, hIndices, dimension - 1));
                        }
                        sort(midH, until, dimension);
                        merge(indices, from, midH, until);
                    }
                }
            }
        }

        // SweepA
        private void sort2D(int from, int until) {
            RedBlackTree t = tree.get();
            t.clear();
            int curr = from;
            while (curr < until) {
                int currI = indices[curr];
                int next = curr + 1;
                while (next < until && eqComp[indices[next]] == eqComp[currI]) {
                    ++next;
                }
                int result = output.accumulateAndGet(currI, t.maxBefore(input[currI][1]) + 1, MAXIMUM);
                for (int i = curr; i < next; ++i) {
                    output.set(indices[i], result);
                }
                t.insert(input[currI][1], result);
                curr = next;
            }
        }

        // Update target's front considering source's front
        private void updateFront(int target, int source) {
            if (eqComp[target] == eqComp[source]) {
                output.set(target, output.get(source));
            } else {
                output.accumulateAndGet(target, output.get(source) + 1, MAXIMUM);
            }
        }

        // All the K-th coordinates of input[from, until) are equal
        private boolean allValuesEqual(int from, int until, int k) {
            double value = input[indices[from]][k];
            for (int i = from + 1; i < until; ++i) {
                if (input[indices[i]][k] != value) {
                    return false;
                }
            }
            return true;
        }

        // If l'th point dominates or equal to r'th point
        private boolean dominatesEq(int il, int ir, int k) {
            for (int i = 0; i <= k; ++i) {
                if (input[il][i] > input[ir][i]) {
                    return false;
                }
            }
            return true;
        }
    }

    private static class MergeSorter {
        final int[] scratch;
        int[] indices = null;
        int secondIndex = -1;
        double[][] reference = null;
        int[] eqComp = null;

        public MergeSorter(int size) {
            this.scratch = new int[size];
        }

        public void lexSort(int[] indices, int from, int until, double[][] reference, int[] eqComp) {
            this.indices = indices;
            this.reference = reference;
            this.eqComp = eqComp;
            lexSortImpl(from, until, 0, 0);
            this.eqComp = null;
            this.reference = null;
            this.indices = null;
        }

        private int lexSortImpl(int from, int until, int currIndex, int compSoFar) {
            if (from + 1 < until) {
                secondIndex = currIndex;
                sortImpl(from, until);
                secondIndex = -1;

                if (currIndex + 1 == reference[0].length) {
                    eqComp[indices[from]] = compSoFar;
                    for (int i = from + 1; i < until; ++i) {
                        int prev = indices[i - 1], curr = indices[i];
                        if (reference[prev][currIndex] != reference[curr][currIndex]) {
                            ++compSoFar;
                        }
                        eqComp[curr] = compSoFar;
                    }
                    return compSoFar + 1;
                } else {
                    int lastIndex = from;
                    for (int i = from + 1; i < until; ++i) {
                        if (reference[indices[lastIndex]][currIndex] != reference[indices[i]][currIndex]) {
                            compSoFar = lexSortImpl(lastIndex, i, currIndex + 1, compSoFar);
                            lastIndex = i;
                        }
                    }
                    return lexSortImpl(lastIndex, until, currIndex + 1, compSoFar);
                }
            } else {
                eqComp[indices[from]] = compSoFar;
                return compSoFar + 1;
            }
        }

        public void sort(int[] indices, int from, int until, double[][] reference, int secondIndex) {
            this.indices = indices;
            this.reference = reference;
            this.secondIndex = secondIndex;
            sortImpl(from, until);
            this.indices = null;
            this.reference = null;
            this.secondIndex = -1;
        }

        private void sortImpl(int from, int until) {
            if (from + 1 < until) {
                int mid = (from + until) >>> 1;
                sortImpl(from, mid);
                sortImpl(mid, until);
                int i = from, j = mid, k = 0, kMax = until - from;
                while (k < kMax) {
                    if (i == mid || j < until && reference[indices[j]][secondIndex] < reference[indices[i]][secondIndex]) {
                        scratch[k] = indices[j];
                        ++j;
                    } else {
                        scratch[k] = indices[i];
                        ++i;
                    }
                    ++k;
                }
                System.arraycopy(scratch, 0, indices, from, kMax);
            }
        }
    }
}
