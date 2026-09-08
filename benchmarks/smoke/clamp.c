static int clamp(int value, int lower, int upper) {
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

int main(void) {
    return clamp(42, 0, 10);
}
