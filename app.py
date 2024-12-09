from flask import Flask, render_template, Response, request, jsonify
import sys
import queue
import threading
import json
import time
from collections import deque
import contextlib
import os
from memos._1_structured_output_4o.structured_output_4o import main as structured_output_4o_main
from memos._2_structured_output_4o_with_examples.structured_output_4o_with_examples import main as structured_output_4o_with_examples_main
from memos._3_pure_function_so4o.pure_function_so4o import (
    main as pure_function_so4o_main,
    ProgrammingLanguage,
    Pure
)
from memos._4_pure_function_with_tests_so4o.pure_function_with_tests_so4o import (
    main as pure_function_with_tests_so4o_main
)
from memos._5_html_so4o.html_so4o import main as html_so4o_main, GenerationProgress
from memos._6_pure_function_with_tests_and_imports_so4o.pure_function_with_tests_and_imports_so4o import (
    main as pure_function_with_tests_and_imports_so4o_main,
    ProgrammingLanguage
)
from memos._7_knowledge_graph_4o.knowledge_graph_4o import main as knowledge_graph_4o_main
from memos._8_knowledge_graph_sonnet.knowledge_graph_sonnet import main as knowledge_graph_sonnet_main

app = Flask(__name__)

def run_flask():
    app.run(debug=False, threaded=True, port=5000)


# ~~~ ROUTES ~~~
@app.route('/')
def index():
    return render_template('index.html')

### 0

@app.route('/history')
def get_history():
    return json.dumps([entry for entry in terminal_history])

@app.route('/stream')
def stream():
    def generate():
        while True:
            try:
                output = output_queue.get(timeout=1)
                yield f"data: {json.dumps(output)}\n\n"
            except queue.Empty:
                yield f"data: \n\n"
            except Exception as e:
                print(f"Stream error: {str(e)}")
                break
    
    return Response(generate(), mimetype='text/event-stream')

### 1

@app.route('/structured-output-4o')
def structured_output_4o():
    return render_template('structured-output-4o.html')

@app.route('/structured-output-4o/generate', methods=['POST'])
def generate_model():
    query = request.json['query']
    generated_model = structured_output_4o_main(query)
    response = jsonify({
        'model_code': generated_model.model_code,
        'model_name': generated_model.model_name,
        'model_description': generated_model.model_description
    })
    return response

@app.route('/structured-output-4o/saved_models')
def get_saved_models():
    models_dir = 'structured_output_4o/so_models'
    models = []
    for f in os.listdir(models_dir):
        if f.endswith('.py'):
            model_name = f.split('.')[0]
            models.append(model_name)
    return jsonify(models)

### 2

@app.route('/structured-output-4o-with-examples')
def structured_output_4o_with_examples():
    return render_template('structured-output-4o-with-examples.html')

@app.route('/structured-output-4o-with-examples/generate', methods=['POST'])
def generate_model_with_examples():
    query = request.json['query']
    num_examples = request.json.get('num_examples', 0)
    generated_model = structured_output_4o_with_examples_main(query, num_examples)
    response = jsonify({
        'model_code': generated_model.model_code,
        'model_name': generated_model.model_name,
        'model_description': generated_model.model_description,
        'examples': generated_model.examples
    })
    return response

@app.route('/structured-output-4o-with-examples/saved_models')
def get_saved_models_with_examples():
    models_dir = '_2_structured_output_4o_with_examples/so_models'
    models = []
    for f in os.listdir(models_dir):
        if f.endswith('.py'):
            model_name = f.split('.')[0]
            models.append(model_name)
    return jsonify(models)

### 3

@app.route('/pure-function-so4o')
def pure_function_so4o():
    return render_template('pure-function-so4o.html')

@app.route('/pure-function-so4o/generate', methods=['POST'])
def generate_pure_function():
    try:
        data = request.json
        if not data or 'data' not in data or not isinstance(data['data'], list) or len(data['data']) < 2:
            return jsonify({
                "data": None,
                "error": "Invalid request format. Expected data array with query and language."
            }), 400
            
        query = data['data'][0]
        language = data['data'][1]
        
        if not query or not language:
            return jsonify({
                "data": None,
                "error": "Query and language are required."
            }), 400

        generated_function = pure_function_so4o_main(
            query=query, 
            language=ProgrammingLanguage(language)
        )
        
        # Format the response in markdown
        markdown_response = generated_function.code
        
        return jsonify({
            "data": [markdown_response],
            "error": None
        })
    except Exception as e:
        return jsonify({
            "data": None,
            "error": str(e)
        }), 500

### 4

@app.route('/pure-function-with-tests-so4o')
def pure_function_with_tests_so4o():
    return render_template('pure-function-with-tests-so4o.html')

@app.route('/pure-function-with-tests-so4o/generate', methods=['POST'])
def generate_pure_function_with_tests():
    try:
        data = request.json
        if not data or 'data' not in data or not isinstance(data['data'], list) or len(data['data']) < 2:
            return jsonify({
                "data": None,
                "error": "Invalid request format. Expected data array with query and language."
            }), 400
            
        query = data['data'][0]
        language = data['data'][1]
        max_attempts = data['data'][2] if len(data['data']) > 2 else 3
        num_tests = data['data'][3] if len(data['data']) > 3 else 3
        
        if not query or not language:
            return jsonify({
                "data": None,
                "error": "Query and language are required."
            }), 400

        generated_function, generated_tests = pure_function_with_tests_so4o_main(
            query=query, 
            language=ProgrammingLanguage(language),
            max_attempts=max_attempts,
            num_tests=num_tests
        )
        
        return jsonify({
            "data": [
                generated_function.code,           # The generated function code
                generated_tests.test_code,         # The generated tests
                generated_function.is_pure == Pure.YES  # Boolean indicating if function is pure
            ],
            "error": None
        })
    except Exception as e:
        return jsonify({
            "data": None,
            "error": str(e)
        }), 500

### 5

@app.route('/html-so4o')
def html_so4o():
    return render_template('html-so4o.html')

@app.route('/html-so4o/generate', methods=['POST'])
def generate_html_so4o():
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({
                "error": "Invalid request format. Query is required."
            }), 400
            
        query = data['query']
        max_retries = data.get('max_retries', 3)  # Default to 3 retries
        
        if not query:
            return jsonify({
                "error": "Query cannot be empty."
            }), 400

        if not isinstance(max_retries, int) or max_retries < 1:
            return jsonify({
                "error": "max_retries must be a positive integer"
            }), 400

        generated_html = html_so4o_main(query, max_retries)
        
        return jsonify({
            "html": generated_html.html,
            "css": generated_html.css,
            "description": generated_html.description,
            "error": None
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

### 6

@app.route('/pure-function-with-tests-and-imports-so4o')
def pure_function_with_tests_and_imports_so4o():
    return render_template('pure-function-with-tests-and-imports-so4o.html')

@app.route('/pure-function-with-tests-and-imports-so4o/generate', methods=['POST'])
def generate_pure_function_with_tests_and_imports():
    try:
        data = request.json
        if not data:
            return jsonify({
                "data": None,
                "error": "Invalid request format."
            }), 400
            
        query = data.get('query')
        language = data.get('language')
        max_attempts = data.get('max_attempts', 3)
        num_tests = data.get('num_tests', 3)
        folders = data.get('folders', [])
        
        if not query or not language or not folders:
            return jsonify({
                "data": None,
                "error": "Query, language, and folders are required."
            }), 400

        if not isinstance(folders, list):
            return jsonify({
                "data": None,
                "error": "Folders must be a list."
            }), 400
            
        folders = [os.path.normpath(f) for f in folders]
        
        for folder in folders:
            if '..' in folder or folder.startswith('/'):
                return jsonify({
                    "data": None,
                    "error": "Invalid folder path."
                }), 400

        generated_function, generated_tests = pure_function_with_tests_and_imports_so4o_main(
            query=query, 
            folders=folders,
            language=ProgrammingLanguage(language),
            max_attempts=max_attempts,
            num_tests=num_tests
        )
        
        return jsonify({
            "data": [
                generated_function.code,           # The generated function code
                generated_tests.test_code,         # The generated tests
                generated_function.is_pure == Pure.YES  # Boolean indicating if function is pure
            ],
            "error": None
        })
    except Exception as e:
        return jsonify({
            "data": None,
            "error": str(e)
        }), 500

### 7

@app.route('/knowledge-graph-4o')
def knowledge_graph_4o():
    return render_template('knowledge-graph-4o.html')

@app.route('/knowledge-graph-4o/generate', methods=['POST'])
def generate_graph():
    try:
        text = request.json.get('text')
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        kg = knowledge_graph_4o_main(text)
        
        return jsonify({
            "nodes": [node.model_dump() for node in kg.nodes],
            "edges": [edge.model_dump() for edge in kg.edges]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/knowledge-graph-4o/saved')
def get_saved_graphs():
    try:
        graphs_dir = os.path.join('memos', '_7_knowledge_graph_4o', 'graphs')
        if not os.path.exists(graphs_dir):
            return jsonify([])
            
        graphs = [f for f in os.listdir(graphs_dir) if f.endswith('.json')]
        return jsonify(graphs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/knowledge-graph-4o/load/<filename>')
def load_graph(filename):
    try:
        graph_path = os.path.join('memos', '_7_knowledge_graph_4o', 'graphs', filename)
        if not os.path.exists(graph_path):
            return jsonify({"error": "Graph not found"}), 404
            
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
            
        return jsonify(graph_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

### 8

@app.route('/knowledge-graph-sonnet')
def knowledge_graph_sonnet():
    return render_template('knowledge-graph-sonnet.html')

@app.route('/knowledge-graph-sonnet/generate', methods=['POST'])
def generate_graph_sonnet():
    try:
        text = request.json.get('text')
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        kg = knowledge_graph_sonnet_main(text)
        
        return jsonify({
            "nodes": [node.model_dump() for node in kg.nodes],
            "edges": [edge.model_dump() for edge in kg.edges]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/knowledge-graph-sonnet/saved')
def get_saved_graphs_sonnet():
    try:
        graphs_dir = os.path.join('memos', '_8_knowledge_graph_sonnet', 'graphs')
        if not os.path.exists(graphs_dir):
            return jsonify([])
            
        graphs = [f for f in os.listdir(graphs_dir) if f.endswith('.json')]
        return jsonify(graphs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/knowledge-graph-sonnet/load/<filename>')
def load_graph_sonnet(filename):
    try:
        graph_path = os.path.join('memos', '_8_knowledge_graph_sonnet', 'graphs', filename)
        if not os.path.exists(graph_path):
            return jsonify({"error": "Graph not found"}), 404
            
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
            
        return jsonify(graph_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ~~~ TERMINAL BROWSER ~~~

# Maintain terminal history using a deque (limited to last 1000 entries)
terminal_history = deque(maxlen=1000)
output_queue = queue.Queue()

class StreamCapture:
    def __init__(self, queue):
        self.queue = queue
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.current_input = ""

    def write(self, text):
        # Convert bytes to string if necessary
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
            
        self.queue.put({"type": "output", "text": text})
        terminal_history.append({"type": "output", "text": text, "timestamp": time.time()})
        self.original_stdout.write(text)
        self.original_stdout.flush()

    def flush(self):
        self.original_stdout.flush()

    def update_current_input(self, text):
        self.current_input = text
        self.queue.put({"type": "input_update", "text": text})

stream_capture = StreamCapture(output_queue)

def custom_input(prompt=""):
    if prompt:
        stream_capture.write(prompt)
    
    buffer = []
    while True:
        try:
            char = sys.stdin.read(1)
            if char == '\n':
                line = ''.join(buffer)
                stream_capture.write('\n')
                terminal_history.append({"type": "input", "text": line, "timestamp": time.time()})
                return line
            elif char in ('\b', '\x7f'):  # Backspace
                if buffer:
                    buffer.pop()
                    stream_capture.update_current_input(''.join(buffer))
            else:
                buffer.append(char)
                stream_capture.update_current_input(''.join(buffer))
        except KeyboardInterrupt:
            raise KeyboardInterrupt()

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    print("Terminal mirror started. Open http://localhost:5000 in your browser")
    print("Type 'exit()' to quit")
    
    with contextlib.redirect_stdout(stream_capture), \
         contextlib.redirect_stderr(stream_capture):
        while True:
            try:
                user_input = custom_input(">>> ")
                if user_input.strip() == 'exit()':
                    break
                
                try:
                    result = eval(user_input)
                    if result is not None:
                        print(result)
                except SyntaxError:
                    try:
                        exec(user_input)
                    except Exception as e:
                        print(f"Error: {str(e)}")
                except Exception as e:
                    print(f"Error: {str(e)}")
            except KeyboardInterrupt:
                print("^C")
                break
            except EOFError:
                break

    print("\nShutting down...")
    
    