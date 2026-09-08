static int dot_product(const int left[3], const int right[3]) {
    int result = 0;
    for (int index = 0; index < 3; ++index) {
        result += left[index] * right[index];
    }
    return result;
}

int main(void) {
    const int left[3] = {1, 2, 3};
    const int right[3] = {4, 5, 6};
    return dot_product(left, right);
}
