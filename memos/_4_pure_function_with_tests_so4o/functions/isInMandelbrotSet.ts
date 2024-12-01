/**
 * Determines if a complex number is in the Mandelbrot set given a maximum number of iterations.
 * The function checks if the iterative sequence x_{n+1} = x_n^2 + c (where x is complex and c is the given complex number)
 * remains bounded within a circle of radius 2. If it does not exceed this threshold for the given number of iterations,
 * the function returns true indicating that the number is in the Mandelbrot set.
 * 
 * @param realPart - The real part of the complex number to check.
 * @param imaginaryPart - The imaginary part of the complex number to check.
 * @param maxIterations - The maximum number of iterations to perform.
 * @returns A boolean indicating whether the given complex number is within the Mandelbrot set.
 */
function isInMandelbrotSet(realPart: number, imaginaryPart: number, maxIterations: number): boolean {
    let zr = 0; // z real part
    let zi = 0; // z imaginary part
    let zrSquared = 0;
    let ziSquared = 0;
    
    for (let i = 0; i < maxIterations; i++) {
        zi = 2 * zr * zi + imaginaryPart;
        zr = zrSquared - ziSquared + realPart;

        zrSquared = zr * zr;
        ziSquared = zi * zi;

        if (zrSquared + ziSquared > 4) {
            return false;
        }
    }
    
    return true;
}

function test_isInMandelbrotSet() {
    // Test case 1
    console.assert(isInMandelbrotSet(0, 0, 1000) === true, 'Test Case 1 Failed');
    
    // Test case 2
    console.assert(isInMandelbrotSet(2, 0, 1000) === false, 'Test Case 2 Failed');
    
    // Test case 3
    console.assert(isInMandelbrotSet(-0.75, 0, 1000) === true, 'Test Case 3 Failed');
    
    // Test case 4
    console.assert(isInMandelbrotSet(-0.1, 0.651, 1000) === true, 'Test Case 4 Failed');
    
    // Test case 5
    console.assert(isInMandelbrotSet(0.5, 0.5, 1000) === false, 'Test Case 5 Failed');

    // Test case 6
    console.assert(isInMandelbrotSet(-2, 2, 1000) === false, 'Test Case 6 Failed');

    // Test case 7
    console.assert(isInMandelbrotSet(0.285, 0.01, 1000) === true, 'Test Case 7 Failed');

    // Test case 8
    console.assert(isInMandelbrotSet(-1.31, 0.1, 100) === true, 'Test Case 8 Failed');

    console.log('All test cases passed');
}

export { test_isInMandelbrotSet as tests, isInMandelbrotSet as default };