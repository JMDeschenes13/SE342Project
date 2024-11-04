from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('page1.html')

@app.route('/generate_plan', methods=['POST'])
def generate_plan():
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    weight = request.form.get('weight')
    height = request.form.get('height')
    body_fat = request.form.get('bodyFat')
    bmi = request.form.get('bmi')
    bmi_case = request.form.get('bmiCase')
    
    
    return f"Workout plan generated for {name} ({age} years, {gender}). Height: {height} cm, Weight: {weight} kg, Body Fat: {body_fat}%, BMI: {bmi} ({bmi_case})."

if __name__ == "__main__":
    app.run(debug=True)
