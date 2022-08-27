from flask import Flask, request, render_template
import time

# from settings import USE_8BIT

# if USE_8BIT:
#     from model_8bit import code_model
# else:
#     from model import code_model
from model_cpu import code_model

app = Flask(__name__)

code = code_model


@app.route('/generate', methods=['GET', 'POST'])
def handle_post():

    if request.method == 'POST':
        start = time.time()
        text = request.form['input']
        output = code.generate(text)
        end = time.time() - start
        print(output)
        print(f'elapsed time is {end}')
        return render_template('codegen.html', input=text, output=output)
    
    return render_template('codegen.html', input='', output='')


@app.route('/health')
def health():
    return "ok"


if __name__ == '__main__':
    app.run()