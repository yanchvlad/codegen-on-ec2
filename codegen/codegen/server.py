from flask import Flask, request, render_template
from model import code_model

app = Flask(__name__)


@app.route('/generate', methods=['GET', 'POST'])
def handle_post():

    if request.method == 'POST':
        text = request.form['input']
        output = code_model(text)
        print(output)
        return render_template('codegen.html', input=text, output=output)
    
    return render_template('codegen.html', input='', output='')


@app.route('/health')
def health():
    return "ok"


if __name__ == '__main__':
    app.run(debug=True)