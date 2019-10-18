from flask import Flask, render_template, request, session
from QASystem.predict_squad import Inference

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"


# Web app using Flask
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/bertmodel", methods=["GET", "POST"])
def bert_demo():
    answers = None
    if request.method == "POST":
        model = request.form.get('model')
        if model == 'en':
            infer_model = Inference('en')
            curr_model = 'en'
        else:
            infer_model = Inference('vi')
            curr_model = 'vi'

        passage = request.form.get('passage')
        question = request.form.get('question')
        if passage is None or question is None or not passage or not question:
            warning = "Please provide the question and a paragraph"
        else:
            answers = infer_model.response([[passage, question]])
            answers = answers[0]  # Since we only have 1 QA pair
            warning = "Success"
    else:
        # Sample passage
        passage = ""
        question = ""
        warning = ""
        curr_model = 'en'

    return render_template("demo_bert.html", passage=passage, question=question, answers=answers, warning=warning,
                           curr_model=curr_model)


if __name__ == "__main__":
    app.run()
