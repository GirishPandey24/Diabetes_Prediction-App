from flask import Flask,render_template,request
import pickle
from sklearn.linear_model import LogisticRegression

app=Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        Pregnancies=request.form['Pregnancies']
        Glucose=request.form['Glucose']
        BloodPressure=request.form['BloodPressure']
        SkinThickness=request.form['SkinThickness']
        Insulin=request.form['Insulin']
        BMI=request.form['BMI']
        DiabetesPedigreeFunction=request.form['DiabetesPedigreeFunction']
        Age=request.form['Age']

        data=[[float(Pregnancies),float(Glucose),float(BloodPressure),float(SkinThickness),float(Insulin),float(BMI),float(DiabetesPedigreeFunction),float(Age)]]
        lr=pickle.load(open('model1.pkl','rb'))
        prediction=lr.predict(data)[0]

        if prediction==1:
            return render_template('index.html',label=1)
        else:
            return render_template('index.html',label=-1)

    return render_template('index.html',prediction=prediction)

if __name__=='__main__':
    app.run(debug=True)
