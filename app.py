import newmodel
from flask import Flask, request, render_template
app = Flask(__name__)


def check(data):
    if len(data) < 5:
        return False
    if data[0] != '>':
        return False
    content = data.split('\n')
    content = [x.strip() for x in content]
    content = content[1:]
    pattern = []
    for i in range(0, len(content)):
        for j in range(0, len(content[i])):
            if content[i][j] != ' ':
                pattern.append(content[i][j])
    for i in range(0, len(pattern)):
        if pattern[i] != 'A' and pattern[i] != 'C' and pattern[i] != 'G' and pattern[i] != 'T':
            return False
    return True


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/readme.html')
def readme():
    return render_template('readme.html')


@app.route('/downloads.html')
def downloads():
    return render_template('downloads.html')


@app.route('/citation.html')
def citation():
    return render_template('citation.html')


@app.route('/contributors.html')
def contributors():
    return render_template('contributors.html')


@app.route('/server.html', methods=['GET', 'POST'])
def server():
    if request.method == 'POST':
        file = request.form['message']
        step_size = request.form['quantity']
        if check(file):
            content = file.split('\n')
            content = [x.strip() for x in content]
            content = content[1:]
            pattern = []
            for i in range(0, len(content)):
                for j in range(0, len(content[i])):
                    if content[i][j] != ' ':
                        pattern.append(content[i][j])
            prob, test_data = newmodel.run_model(pattern, int(step_size))
            sizes = []
            start = 0
            for i in range(0, len(test_data)):
                sizes.append(start+len(test_data[i]))
                start = start+len(test_data[i])
            size_range = []
            for i in range(0, len(sizes)):
                if i == 0:
                    size_range.append([1, sizes[i]])
                else:
                    size_range.append([sizes[i-1]+1, sizes[i]])
            return render_template('result.html', prob=prob, size_range=size_range, test_data=test_data)
            #return render_template('test.html', prob=prob, size_range=size_range, test_data=test_data)

        else:
            return render_template('error.html')

    return render_template('server.html')


@app.route('/result.html')
def result():
    return render_template('result.html')


@app.route('/test.html')
def test():
    return render_template('test.html')


if __name__ == '__main__':
    app.run()


