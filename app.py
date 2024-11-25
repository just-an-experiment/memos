from flask import Flask, render_template, Response, request, jsonify
import sys
import queue
import threading
import json
import time
from collections import deque
import contextlib
import os
from structured_output_4o.structured_output_4o import main as structured_output_4o_main

app = Flask(__name__)

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

# ~~~ ROUTES ~~~
@app.route('/')
def index():
    return render_template('index.html')

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


def run_flask():
    app.run(debug=False, threaded=True, port=5000)

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