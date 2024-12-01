# Represents a new programming language's specification including name, version, keywords, data types, operators, syntax rules, and functionalities.

from pydantic import BaseModel, Field

class Keyword(BaseModel):
    keyword: str = Field(..., description="A reserved word in the language that has a predefined meaning and function in the language's syntax.")


class DataType(BaseModel):
    name: str = Field(..., description="The name of the data type.")
    description: str = Field(..., description="A brief explanation of what kind of values this data type can represent.")


class Operator(BaseModel):
    symbol: str = Field(..., description="The symbol representing an operation in the language.")
    description: str = Field(..., description="A description of what the operator does and how it is used.")


class SyntaxRule(BaseModel):
    rule: str = Field(..., description="A syntax rule that defines the structure of statements in the language.")


class Functionality(BaseModel):
    name: str = Field(..., description="The name of the functionality or feature provided by the language.")
    description: str = Field(..., description="A detailed description of how the functionality works within the language.")


class NewProgrammingLanguage(BaseModel):
    name: str = Field(..., description="The name of the new programming language.")
    version: str = Field(..., description="The version of this programming language specification.")
    keywords: list[Keyword] = Field(..., description="A list of keywords used in the programming language.")
    data_types: list[DataType] = Field(..., description="A list of data types supported by the language.")
    operators: list[Operator] = Field(..., description="A list of operators available in the language.")
    syntax_rules: list[SyntaxRule] = Field(..., description="A list of syntax rules that define the formal system of the language.")
    functionalities: list[Functionality] = Field(..., description="A list of unique functionalities or features offered by the language.")

# Example data that matches the model schema
examples = [
    {'name': 'QuantumScript', 'version': '1.2.0', 'keywords': [{'keyword': 'qvar'}, {'keyword': 'entangle'}, {'keyword': 'superposition'}, {'keyword': 'measure'}], 'data_types': [{'name': 'qubit', 'description': 'A quantum bit that can represent a 0, 1, or any quantum superposition of these states.'}, {'name': 'qregister', 'description': 'A collection of qubits that can be manipulated simultaneously.'}, {'name': 'probability', 'description': 'Represents the likelihood of a quantum state collapsing into a particular state.'}, {'name': 'wavefunction', 'description': 'A mathematical function describing the quantum state of a system.'}], 'operators': [{'symbol': '⊗', 'description': 'Tensor product operator for combining quantum states or gates.'}, {'symbol': '+', 'description': 'Addition of classical values or probability amplitudes.'}, {'symbol': '*', 'description': 'Multiplication of classical values or operation with scalars.'}, {'symbol': 'C⊗', 'description': 'Conditioned tensor product for applying operations based on certain conditions.'}], 'syntax_rules': [{'rule': 'qvar must be initialized before use: qvar qbit1 = |0⟩;'}, {'rule': 'entangle(qreg1, qreg2) allows for quantum entanglement between registers.'}, {'rule': 'measure(qreg) collapses the qubits in qreg to classical bits.'}], 'functionalities': [{'name': 'Entanglement Library', 'description': 'Provides built-in functions for generating entangled quantum states easily between qubits or quantum registers.'}, {'name': 'Quantum Circuit Simulation', 'description': 'Enables the simulation of quantum circuits to test quantum algorithms before applying them on actual quantum hardware.'}, {'name': 'Multiverse Debugging', 'description': 'A novel debugging technique which allows developers to visualize the superposition of states in 3D.'}]},
    {'name': 'ChronoScript', 'version': '0.9.5', 'keywords': [{'keyword': 'interval'}, {'keyword': 'timeline'}, {'keyword': 'revert'}, {'keyword': 'futurecast'}], 'data_types': [{'name': 'timestamp', 'description': 'Represents a moment in time with precision up to nanoseconds.'}, {'name': 'duration', 'description': 'Represents a span of time, useful for measuring intervals.'}, {'name': 'timeline', 'description': 'An ordered collection of timestamps that represent a chronological sequence of events.'}, {'name': 'event', 'description': 'Represents a specific occurrence within a timeline.'}], 'operators': [{'symbol': '↔', 'description': 'Bidirectional operator to create a temporal link between events.'}, {'symbol': '+', 'description': 'Add a duration to a timestamp to calculate a future timestamp.'}, {'symbol': '-', 'description': 'Subtract timestamps or durations to find the interval or difference.'}, {'symbol': '||', 'description': 'Temporal OR, to evaluate if any of the conditions happen within intersecting timelines.'}], 'syntax_rules': [{'rule': 'interval(duration) sets a repeating block of time.'}, {'rule': 'revert(timestamp) adjusts the current timeline to a past recorded state.'}, {'rule': 'futurecast(event) makes predictions based on existing timeline patterns and events.'}], 'functionalities': [{'name': 'Time Manipulation', 'description': 'Allows for complex manipulation of timelines, including rewinding, pausing, and fast-forwarding time sequences in simulations.'}, {'name': 'Predictive Modeling', 'description': 'Automates the generation of future events or data points based on historical timeline analysis.'}, {'name': 'Temporal Synchronization', 'description': 'Ensures all operations across different timelines stay consistent with each other in distributed environments.'}]},
]


export = {
    'default': NewProgrammingLanguage,
    'examples': examples
}