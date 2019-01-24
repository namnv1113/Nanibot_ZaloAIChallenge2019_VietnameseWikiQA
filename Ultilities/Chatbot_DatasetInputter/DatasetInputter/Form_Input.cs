using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DatasetInputter
{
    public partial class Form_Input : Form
    {
        private LoadData data_loader;
        public DataStatistics DataStatistics;
        private List<RadioButton> radioButtons;
        private List<CheckBox> checkBoxes;
        public int mode = 0;     // 0 = Modify, 1 = Add

        public Form_Input()
        {
            InitializeComponent();
            Application.CurrentInputLanguage = InputLanguage.FromCulture(new System.Globalization.CultureInfo("vi-VN"));

            if (mode == 0)
            {
                button_clear.Enabled = false;
                button_save.Enabled = false;
            }

            radioButtons = new List<RadioButton>();
            radioButtons.Add(radioButton_at_date);
            radioButtons.Add(radioButton_at_numberic);
            radioButtons.Add(radioButton_at_person);
            radioButtons.Add(radioButton_at_location);
            radioButtons.Add(radioButton_at_otherentity);
            radioButtons.Add(radioButton_at_nounphrase);
            radioButtons.Add(radioButton_at_adjectivephrase);
            radioButtons.Add(radioButton_at_verbphrase);
            radioButtons.Add(radioButton_at_clause);
            radioButtons.Add(radioButton_at_others);

            checkBoxes = new List<CheckBox>();
            checkBoxes.Add(checkBox_reasoning_synonymy);
            checkBoxes.Add(checkBox_reasoning_knowledge);
            checkBoxes.Add(checkBox_reasoning_syntactic);
            checkBoxes.Add(checkBox_reasoning_multiplesentences);
            checkBoxes.Add(checkBox_reasoning_ambiguous);
        }

        public void Clear()
        {
            textBox_question.Text = "";
            textBox_answer1.Text = "";
        }

        private void button_clear_Click(object sender, EventArgs e)
        {
            // DialogResult result = MessageBox.Show("Are you sure to clear all data?", "Warning", MessageBoxButtons.YesNo, MessageBoxIcon.Warning);
            //if (result == DialogResult.Yes)
            Clear();
        }
        private void button_save_Click(object sender, EventArgs e)
        {
            if (RequiredFieldEmpty())
            {
                MessageBox.Show("Please input required field before saving", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            string id = GenerateID();
            data_loader.SaveQA(id, textBox_question.Text, textBox_answer1.Text, "", "", textBox_txtStart1.Text, "", "");
            MessageBox.Show("Save complete", "Saved", MessageBoxButtons.OK, MessageBoxIcon.Information);
            Clear();
            UpdatePreviousQuestion();
            DataStatistics.question_count++;
            LoadGeneralDataStatistics();
            textBox_question.Focus();
        }

        private string GenerateID()
        {
            string id = "";
            // Encode answer type
            id += "A";
            int temp = 0;
            foreach (RadioButton button in radioButtons)
            {
                if (button.Checked)
                {
                    id += temp.ToString();
                    break;
                }
                temp++;
            }
            if (temp == 10)
                id += "?";
            // Encode reasoning required
            temp = 0;
            id += "R";
            foreach (CheckBox box in checkBoxes)
            {
                id += (box.Checked) ? "1" : "0";
                temp++;
            }
            // Encode time
            id += "T" + DateTime.Now.Ticks.ToString();
            return id;
        }

        private bool RequiredFieldEmpty()
        {
            if (textBox_question.Text == "")
                return true;
            if (textBox_answer1.Text == "")
                return true;
            /*
            if (textBox_answer2.Text == "")
                return true;
            if (textBox_answer3.Text == "")
                return true;
                */
            return false;
        }

        private void button_Next_Click(object sender, EventArgs e)
        {
            if (mode == 0)
            {
                data_loader.UpdateQA(textBox_question.Text, textBox_answer1.Text, textBox_txtStart1.Text);
                LoadQA();
            }
            else
            {
                DialogResult result = MessageBox.Show("Are you sure to continue? Be sure to save current question before continue.", "Warning", MessageBoxButtons.YesNo, MessageBoxIcon.Warning);
                if (result == DialogResult.Yes)
                {
                    Clear();
                    LoadSentence(data_loader.LoadNextSentence());
                }
            }
        }

        private void Form_Input_Load(object sender, EventArgs e)
        {
            data_loader = new LoadData();
            data_loader.LoadJson();
            DataStatistics = new DataStatistics(data_loader.article.data);
            LoadGeneralDataStatistics();
            LoadSentence(data_loader.LoadNextSentence());
            LoadGuide();
        }

        private void LoadGuide()
        {
            richTextBox_guide.Clear();

            richTextBox_guide.AppendText("Current progress: " + data_loader.curr_data_index + " / " + DataStatistics.sentences_count + " - 1" + Environment.NewLine);
            richTextBox_guide.AppendText("---------------- Hướng dẫn ---------------------" + Environment.NewLine);
            richTextBox_guide.AppendText("1/ Đặt câu hỏi ứng với đoạn văn bản bên trên" + Environment.NewLine);
            richTextBox_guide.AppendText("2/ Với câu hỏi đã đặt, đưa ra 3 câu trả lời tương ứng với câu hỏi bên trên. Câu trả lời phải được lấy (Copy Paste) từ đoạn văn bản ở trên" + Environment.NewLine);
            richTextBox_guide.AppendText("   Nếu chỉ đưa ra được 1 hoặc 2 câu trả lời thì câu trả lời còn lại copy từ câu trả lời gần nhất" + Environment.NewLine);
            richTextBox_guide.AppendText("   Thứ tự câu trả lời: Từ ngắn đến dài" + Environment.NewLine);
            richTextBox_guide.AppendText("3/ Sau khi trả lời, đánh giá câu trả lời và câu hỏi theo bảng bên phải" + Environment.NewLine);
            richTextBox_guide.AppendText("4/ Bấm Save để lưu lại kết quả và tiếp tục đặt câu hỏi với đoạn văn cũ. Next để qua đoạn văn mới" + Environment.NewLine);
            richTextBox_guide.AppendText("-------------------------------------" + Environment.NewLine);
            richTextBox_guide.AppendText("Lexical variation - synonymy: Major correspondences between the question and the answer sentence are synonyms" + Environment.NewLine);
            richTextBox_guide.AppendText("-------------------------------------" + Environment.NewLine);
            richTextBox_guide.AppendText("Lexical variation - world knowledge: Major correspondences between the question and the answer sentence require world knowledge to resolve." + Environment.NewLine);
            richTextBox_guide.AppendText("-------------------------------------" + Environment.NewLine);
            richTextBox_guide.AppendText("Syntactic variation: After the question is paraphrased into declarative form, its syntactic dependency structure does not match that of the answer sentence even after local modifications" + Environment.NewLine);
            richTextBox_guide.AppendText("-------------------------------------" + Environment.NewLine);
            richTextBox_guide.AppendText("Multiple sentence: There is anaphora, or higher-level fusion of multiple sentences is required" + Environment.NewLine);
            richTextBox_guide.AppendText("-------------------------------------" + Environment.NewLine);
            richTextBox_guide.AppendText("Ambiguous: The question does not have a unique answer" + Environment.NewLine);
        }

        private void LoadGeneralDataStatistics()
        {
            textBox_total_sentences.Text = DataStatistics.sentences_count.ToString();
            textBox_total_questions.Text = DataStatistics.question_count.ToString();
            textBox_question_count.Text = DataStatistics.question_count_in_sentences.ToString();
            LoadGuide();
        }

        private void LoadSentence(QuyChe_Data data)
        {
            richTextBox_title.Clear();
            richTextBox_title.AppendText(data.title);
            richTextBox_sentence.Clear();
            richTextBox_sentence.AppendText(data.paragraphs[0].context);
            UpdatePreviousQuestion();
            LoadGeneralDataStatistics();
            if (mode == 0)
            {
                LoadQA();
                LoadGuide();
            }
                
        }

        private void LoadQA()
        {
            QuestionAnswer qa = data_loader.GetNextQA();
            if (qa is null)
            {
                LoadSentence(data_loader.LoadNextSentence());
                return;
            }
            textBox_answer1.Text = qa.answers[0].text;
            textBox_question.Text = qa.question;
        }

        private void UpdatePreviousQuestion()
        {
            int count = 0;
            richTextBox_prev_question.Clear();
            foreach (QuestionAnswer qa in data_loader.LoadPreviousQuestionForCurrentSentence())
            {
                richTextBox_prev_question.AppendText(qa.question + Environment.NewLine);
                count++;
            }
            DataStatistics.question_count_in_sentences = count;
        }

        private void Form_Input_FormClosing(object sender, FormClosingEventArgs e)
        {
            data_loader.SaveJSONToFile();
            MessageBox.Show("Saved to JSON file", "Thanks for the hard work", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void textBox_answer1_TextChanged(object sender, EventArgs e)
        {
            int index = richTextBox_sentence.Text.IndexOf(textBox_answer1.Text);
            textBox_txtStart1.Text = index.ToString();
            textBox_ans1len.Text = textBox_answer1.Text.Length.ToString();
        }

        private void button_skipToSentence_Click(object sender, EventArgs e)
        {
            int skipJump = -1;
            int.TryParse(textBox_skipToSentence.Text, out skipJump);
            if (skipJump == - 1 || skipJump >= DataStatistics.sentences_count)
            {
                MessageBox.Show("Can't jump that far.");
                return;
            }
            data_loader.curr_data_index = skipJump;
            data_loader.curr_qa_index = -1;
            LoadSentence(data_loader.LoadCurrSentence());

        }
    }
}
