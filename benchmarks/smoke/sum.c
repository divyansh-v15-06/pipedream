static int sum_to(int limit) {
    int sum = 0;
    int value = 1;
    while (value <= limit) {
        sum += value;
        ++value;
    }
    return sum;
}

int main(void) {
    return sum_to(10);
}
