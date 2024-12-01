/**
 * Evaluates a simple arithmetic expression provided in a Lisp-like syntax (prefix notation).
 * The function supports basic arithmetic operators: +, -, *, and /.
 * 
 * @param expression - A string representing the arithmetic expression in prefix notation
 * @returns The result of the evaluated expression as a number
 */
function evaluateExpression(expression: string): number {
    const tokens: string[] = expression.trim().split(/\s+/);

    function evaluate(tokens: string[]): [number, string[]] {
        const operator: string = tokens[0];
        const operands: string[] = tokens.slice(1);

        if (operator === '+') {
            const [left, restTokens1] = evaluate(operands);
            const [right, restTokens2] = evaluate(restTokens1);
            return [left + right, restTokens2];
        } else if (operator === '-') {
            const [left, restTokens1] = evaluate(operands);
            const [right, restTokens2] = evaluate(restTokens1);
            return [left - right, restTokens2];
        } else if (operator === '*') {
            const [left, restTokens1] = evaluate(operands);
            const [right, restTokens2] = evaluate(restTokens1);
            return [left * right, restTokens2];
        } else if (operator === '/') {
            const [left, restTokens1] = evaluate(operands);
            const [right, restTokens2] = evaluate(restTokens1);
            return [left / right, restTokens2];
        } else {
            return [parseFloat(operator), operands];
        }
    }

    const [result] = evaluate(tokens);
    return result;
}