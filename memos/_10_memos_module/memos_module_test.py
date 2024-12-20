from memos_module import main

module_spec = {
    "description": """Create a module that analyzes text and provides statistics like word count, 
    character count, most common words, and readability scores. The module should handle different 
    text formats, support multiple languages, and provide visualization of the results.""",
    
    "module_context": [
        """Example text analyzer:
        def analyze_text(text: str) -> Dict[str, Any]:
            return {
                'word_count': len(text.split()),
                'char_count': len(text)
            }
        """,
        """Readability scoring:
        def calculate_readability(text: str) -> float:
            # Flesch reading ease score
            words = len(text.split())
            sentences = len(text.split('.'))
            syllables = count_syllables(text)
            return 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        """
    ],
    
    "test_context": [
        """Test basic text analysis:
        def test_word_and_char_count():
            text = "Hello world. This is a test."
            result = analyze_text(text)
            assert result['word_count'] == 6
            assert result['char_count'] == 27
        """,
        """Test empty input:
        def test_empty_text():
            result = analyze_text("")
            assert result['word_count'] == 0
            assert result['char_count'] == 0
        """
    ],
    
    "dependencies": [
        "import nltk",
        "from textblob import TextBlob"
    ]
}

# Generate the module
generated_module = main(module_spec)

print("\nGenerated Module Summary:")
print(f"Module Name: {generated_module.name}")
print(f"Files Generated: {len(generated_module.files)}")
for file in generated_module.files:
    print(f"- {file.path}")

print("\nTest Cases Generated:")
for test in generated_module.test_cases:
    print(f"- {test.name}: {test.description}")