#include <stdio.h>
#include <stdlib.h>
#include <time.h>



int main() {
    FILE *file;
    int NUM_COUNT ;
    printf("Enter the number of digits to be generated:");
    scanf("%d",&NUM_COUNT);
    file = fopen("input.txt", "w");
    
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing\n");
        return 1;
    }
    fprintf(file, "%d\n", NUM_COUNT);
    // Seed the random number generator
    srand(time(NULL));

    for (int i = 0; i < NUM_COUNT; i++) {
        // Generate random single-digit number (0 to 9)
        int random_number = rand() % 10; // Random number between 0 and 9
        fprintf(file, "%d\n", random_number);
    }

    fclose(file);
    printf("Generated %d random single-digit numbers (0 to 9) and stored them in input.txt\n", NUM_COUNT);
    
    return 0;
}

