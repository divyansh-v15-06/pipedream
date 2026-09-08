int main(void) {
    int values[4] = {1, 3, 5, 7};
    int total = 0;
    for (int index = 0; index < 4; ++index) {
        total += values[index];
    }
    return total;
}
