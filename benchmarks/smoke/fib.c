static int fibonacci(int count) {
    int previous = 0;
    int current = 1;
    for (int index = 0; index < count; ++index) {
        int next = previous + current;
        previous = current;
        current = next;
    }
    return previous;
}

int main(void) {
    return fibonacci(8);
}
