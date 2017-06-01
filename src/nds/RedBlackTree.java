package nds;

import java.util.Arrays;

public class RedBlackTree {
    public int size;
    public Node root;
    public double[] stairs;
    public boolean[] isThere;

    public RedBlackTree(int maxSize) {
        this.size = 0;
        this.root = null;
        this.isThere = new boolean[maxSize];
        this.stairs = new double[maxSize];
    }

    public final class Node {
        public double key;
        public int value;
        public boolean red;
        public Node left;
        public Node right;
        public Node parent;
        public int index;

        public Node(double key, int value, boolean red, Node left, Node right, Node parent) {
            this.key = key;
            this.value = value;
            this.red = red;
            this.left = left;
            this.right = right;
            this.parent = parent;
            this.index = index;
        }
    }

    private static boolean isRed(Node node) {
        return (node != null && node.red);
    }

    private static boolean isBlack(Node node) {
        return (node == null || !node.red);
    }

    public void clear() {
        this.root = null;
        this.size = 0;
        Arrays.fill(this.isThere, false);
    }

    static Node getNode(Node node, double key) {
        while (true) {
            if (node == null)
                return null;
            double nk = node.key;
            if (key < nk)
                node = node.left;
            else if (key > nk)
                node = node.right;
            else
                return node;
        }
    }

    static Node minNodeNonNull(Node node) {
        while (true) {
            if (node.left == null)
                return node;
            node = node.left;
        }
    }

    static Node maxNodeNonNull(Node node) {
        while (true) {
            if (node.right == null)
                return node;
            node = node.right;
        }
    }

    static Node successor(Node node) {
        if (node.right != null)
            return minNodeNonNull(node.right);
        Node x = node;
        Node y = x.parent;
        while ((y != null) && (x == y.right)) {
            x = y;
            y = y.parent;
        }
        return y;
    }

    public static Node predecessor(Node node) {
        if (node.left != null)
            return maxNodeNonNull(node.left);
        Node x = node;
        Node y = x.parent;
        while ((y != null) && (x == y.left)) {
            x = y;
            y = y.parent;
        }
        return y;
    }

    public int maxBefore(double key) {
        Node node = this.root;
        if (node == null)
            return -1;
        else {
            Node y = null;
            Node x = node;
            int cmp = 1;
            while ((x != null) && cmp != 0) {
                y = x;
                cmp = Double.compare(key, x.key);
                x = cmp < 0 ? x.left : x.right;
            }
            if (cmp >= 0)
                return y.value;
            y = predecessor(y);
            return y == null ? -1 : y.value;
        }
    }

    public void insert(double key, int value) {
        if (isThere[value] && stairs[value] <= key) {
            return;
        }
        Node y = null;
        Node x = this.root;
        int cmp = 1;
        while ((x != null) && cmp != 0) {
            if (x.key <= key && x.value >= value)
                return;
            y = x;
            cmp = Double.compare(key, x.key);
            x = (cmp < 0) ? x.left : x.right;
        }
        if (cmp == 0) {
            int oldValue = y.value;
            if (oldValue < value) {
                y.value = value;
                isThere[oldValue] = false;
            }
            x = y;
        } else {
            Node z = new Node(key, value, true, null, null, y);
            if (y == null) this.root = z;
            else if (cmp < 0) y.left = z;
            else y.right = z;
            this.fixAfterInsert(z);
            this.size++;
            x = z;
        }
        Node next = successor(x);
        while ((next != null) && (next.value <= value)) {
            isThere[next.value] = false;
            delete(next);
            next = successor(x);
        }
        isThere[value] = true;
        stairs[value] = key;
    }

    private void fixAfterInsert(Node node) {
        Node z = node;
        while (isRed(z.parent)) {
            if (z.parent == z.parent.parent.left) {
                Node y = z.parent.parent.right;
                if (isRed(y)) {
                    z.parent.red = false;
                    y.red = false;
                    z.parent.parent.red = true;
                    z = z.parent.parent;
                } else {
                    if (z == z.parent.right) {
                        z = z.parent;
                        this.rotateLeft(z);
                    }
                    z.parent.red = false;
                    z.parent.parent.red = true;
                    this.rotateRight(z.parent.parent);
                }
            } else { // symmetric cases
                Node y = z.parent.parent.left;
                if (isRed(y)) {
                    z.parent.red = false;
                    y.red = false;
                    z.parent.parent.red = true;
                    z = z.parent.parent;
                } else {
                    if (z == z.parent.left) {
                        z = z.parent;
                        this.rotateRight(z);
                    }
                    z.parent.red = false;
                    z.parent.parent.red = true;
                    this.rotateLeft(z.parent.parent);
                }
            }
        }
        this.root.red = false;
    }

    public void delete(Node z) {
        if (z != null) {
            Node y = z;
            boolean yIsRed = y.red;
            Node x = null;
            Node xParent = null;

            if (z.left == null) {
                x = z.right;
                this.transplant(z, z.right);
                xParent = z.parent;
            } else if (z.right == null) {
                x = z.left;
                transplant(z, z.left);
                xParent = z.parent;
            } else {
                y = minNodeNonNull(z.right);
                yIsRed = y.red;
                x = y.right;

                if (y.parent == z)
                    xParent = y;
                else {
                    xParent = y.parent;
                    this.transplant(y, y.right);
                    y.right = z.right;
                    y.right.parent = y;
                }
                this.transplant(z, y);
                y.left = z.left;
                y.left.parent = y;
                y.red = z.red;
            }
//            this.deleteNode(z);
            z = null;
            if (!yIsRed)
                this.fixAfterDelete(x, xParent);
            this.size -= 1;
        }
    }

    private void fixAfterDelete(Node node, Node parent) {
        Node x = node;
        Node xParent = parent;
        while ((x != this.root) && isBlack(x)) {
            if (x == xParent.left) {
                Node w = xParent.right;

                if (w.red) {
                    w.red = false;
                    xParent.red = true;
                    this.rotateLeft(xParent);
                    w = xParent.right;
                }
                if (isBlack(w.left) && isBlack(w.right)) {
                    w.red = true;
                    x = xParent;
                } else {
                    if (isBlack(w.right)) {
                        w.left.red = false;
                        w.red = true;
                        this.rotateRight(w);
                        w = xParent.right;
                    }
                    w.red = xParent.red;
                    xParent.red = false;
                    w.right.red = false;
                    this.rotateLeft(xParent);
                    x = this.root;
                }
            } else { // symmetric cases
                Node w = xParent.left;

                if (w.red) {
                    w.red = false;
                    xParent.red = true;
                    this.rotateRight(xParent);
                    w = xParent.left;
                }
                if (isBlack(w.right) && isBlack(w.left)) {
                    w.red = true;
                    x = xParent;
                } else {
                    if (isBlack(w.left)) {
                        w.right.red = false;
                        w.red = true;
                        this.rotateLeft(w);
                        w = xParent.left;
                    }
                    w.red = xParent.red;
                    xParent.red = false;
                    w.left.red = false;
                    this.rotateRight(xParent);
                    x = this.root;
                }
            }
            xParent = x.parent;
        }
        if (x != null)
            x.red = false;
    }

    private void rotateLeft(Node x) {
        if (x == null)
            return;
        Node y = x.right;
        x.right = y.left;

        if (y.left != null)
            y.left.parent = x;
        y.parent = x.parent;

        if (x.parent == null)
            this.root = y;
        else if (x == x.parent.left)
            x.parent.left = y;
        else x.parent.right = y;

        y.left = x;
        x.parent = y;
    }

    private void rotateRight(Node x) {
        if (x == null)
            return;
        Node y = x.left;
        x.left = y.right;

        if (y.right != null)
            y.right.parent = x;
        y.parent = x.parent;

        if (x.parent == null) this.root = y;
        else if (x == x.parent.right) x.parent.right = y;
        else x.parent.left = y;

        y.right = x;
        x.parent = y;
    }

    private void transplant(Node to, Node from) {
        if (to.parent == null) this.root = from;
        else if (to == to.parent.left) to.parent.left = from;
        else to.parent.right = from;

        if (from != null)
            from.parent = to.parent;
    }
}