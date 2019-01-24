from flask import Flask, render_template, request, session 
from flask_session import Session 
from predict_squad import Inference
from whoosh_search import run

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Web app using Flask
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/bertmodel", methods=["GET", "POST"])
def bert_demo():
	infer_model = None
	answers = None
	if request.method == "POST":
		model = request.form.get('model')
		if (model == 'en'):
			infer_model = Inference('en')
			curr_model = 'en'
		else:
			infer_model = Inference('vi')
			curr_model = 'vi'

		passage = request.form.get('passage')
		question = request.form.get('question')
		if (passage is None or question is None or not passage or not question):
			warning = "Please provide the question and a paragraph"
		else:
			answers = infer_model.response([[passage, question]])
			answers = answers[0]	# Since we only have 1 QA pair
			warning = "Success"
	else:
		# Sample passage
		passage = ""
		question = ""
		warning = ""
		curr_model = 'en'

	return render_template("demo_bert.html", passage = passage, question = question, answers = answers, warning = warning, curr_model = curr_model)

@app.route("/uithelper", methods=["GET", "POST"])
def uit_helper():
	if request.method == "POST":
		question = request.form.get('question')
		isQueryExpand = request.form.get('isQueryExpand')
		paragraphs = run(question, isQueryExpand)
		inputs = []
		for para in paragraphs:
			inputs.append([para[0], question])
		infer_model = Inference('vi_uit')
		answers = infer_model.response(inputs)
		for i in range(len(paragraphs)):
			paragraphs[i].append(answers[i])
	else:
		paragraphs = []
		question = None
	return render_template("demo_uit.html", paragraphs = paragraphs, question = question)

if __name__ == "__main__":
	app.run()