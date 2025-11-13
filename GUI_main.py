from flask import Flask, render_template, request
import GUI_class

app = Flask(__name__)

@app.route('/')
def function():
    return render_template('GUI.html')

@app.route('/results',methods=['POST','GET'])
def link():
    if request.method == 'POST':           # If data is being posted
        requests = request.form.to_dict()  # Convert all incoming data to dictionary
        tempType = requests['temp']        # Get the input in tempType variable

        obj = GUI_class.ChatBot(tempType)

        if tempType:

            generated_response = obj.Approach3()

            flag1 = True

            return render_template('GUI.html', generated_response=generated_response, flag1=flag1)

        else:
            error = "Please input your query!"

            return render_template('GUI.html', error=error)


if __name__ == '__main__':
    app.run(debug=True)

