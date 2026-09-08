static int maximum(int left, int right) {
    return left > right ? left : right;
}

static int minimum(int left, int right) {
    return left < right ? left : right;
}

int main(void) {
    return maximum(4, 9) + minimum(4, 9);
}
