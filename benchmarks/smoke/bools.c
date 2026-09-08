static int is_even(int value) {
    return (value % 2) == 0;
}

int main(void) {
    return is_even(12) && !is_even(7);
}
