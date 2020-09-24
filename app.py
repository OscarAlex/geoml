from flask import Flask, render_template, request

app= Flask(__name__)

#Prueba, esta es la rama 2

@app.route('/')
def Index():
    return render_template('index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        return('archivo subido')


if __name__ == '__main__':
    app.run(port = 3000, debug = True)

