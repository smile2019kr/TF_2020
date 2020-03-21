from flask import Flask
from flask import render_template
from machine.random_number_maker import RandomNumberMaker
from machine.boston import Boston


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move/<path>') #move뒤에 있는 경로로 이동
def move(path):
    return render_template(f'{path}.html') #위에서 지정한 경로를 반환


@app.route('/random_number')
def random_number():
    rnm = RandomNumberMaker()
    result = rnm.fetch_randomnumber()
    render_params = {} # 파라메터는 딕셔너리로 줘야함
    render_params['result'] = result
    return render_template('random_number.html', **render_params) # ** : 모두 가져오라는 것

@app.route('/neuron') #move뒤에 있는 경로로 이동
def create_neuron():
    rnm = RandomNumberMaker()
    result = rnm.create_neuron()
    render_params = {}  # 파라메터는 딕셔너리로 줘야함
    render_params['result'] = result
    return render_template('neuron.html', **render_params)

@app.route('/boston') #move뒤에 있는 경로로 이동
def boston():
    boston = Boston()
    #boston.show_dataset()
    boston.create_model()

    render_params = {}  # 파라메터는 딕셔너리로 줘야함
    render_params['result'] = None
    return render_template('boston.html', **render_params)



if __name__ == '__main__':
    app.run()

