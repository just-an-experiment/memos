// generateStarShape generates a star shape pattern with a given number of lines.
// The pattern is represented as a slice of strings, where each string corresponds to a line in the pattern.
// The input `n` determines the number of lines in the pattern. Each line symmetrically contributes to the star shape.
func generateStarShape(n int) []string {
	if n <= 0 {
		return []string{}
	}

	// Initialize the slice to hold the pattern.
	lines := make([]string, n)

	// Generate the pattern.
	for i := 0; i < n; i++ {
		spaces := n - i - 1
		stars := 2*i + 1
		line := ""

		// Add leading spaces for left alignment.
		for j := 0; j < spaces; j++ {
			line += " "
		}

		// Add stars to form the line for the star shape.
		for j := 0; j < stars; j++ {
			line += "*"
		}

		// Assign the line to the slice.
		lines[i] = line
	}

	return lines
}

package main

import (
	"fmt"
	"reflect"
)

func test_generateStarShape() {
    // Define the test cases
    tests := []struct {
        input    int
        expected []string
    }{
        {1, []string{"*"}},
        {3, []string{"  *  ", " *** ", "*****"}},
        {5, []string{"    *    ", "   ***   ", "  *****  ", " ******* ", "*********"}},
    }

    // Execute the test cases
    for _, test := range tests {
        result := generateStarShape(test.input)
        if !reflect.DeepEqual(result, test.expected) {
            fmt.Printf("FAIL: Input: %d\nExpected: %v\nGot: %v\n", test.input, test.expected, result)
        } else {
            fmt.Printf("PASS: Input: %d\n", test.input)
        }
    }
}

// To implement by the user:
// func generateStarShape(n int) []string {
//     // Function implementation goes here
// }