/**
 * addNumbers is a pure function that takes an array of numbers
 * and returns their sum. It emulates a Lisp-like addition 
 * function that can sum any number of numbers.
 *
 * @param numbers - An array of numbers to be summed.
 * @returns The sum of the input numbers.
 */
function addNumbers(numbers: number[]): number {
    return numbers.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
}