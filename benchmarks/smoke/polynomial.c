static int polynomial(int value) {
    int square = value * value;
    int cube = square * value;
    return cube + (2 * square) + value + 1;
}

int main(void) {
    return polynomial(3);
}
