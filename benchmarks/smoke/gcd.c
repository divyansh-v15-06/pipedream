static int gcd(int left, int right) {
    while (right != 0) {
        int remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

int main(void) {
    return gcd(84, 30);
}
