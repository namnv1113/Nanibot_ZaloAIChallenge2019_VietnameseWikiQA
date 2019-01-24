namespace DatasetInputter
{
    partial class Form_Input
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.label_sentence = new System.Windows.Forms.Label();
            this.richTextBox_sentence = new System.Windows.Forms.RichTextBox();
            this.label_question = new System.Windows.Forms.Label();
            this.label_answer1 = new System.Windows.Forms.Label();
            this.textBox_answer1 = new System.Windows.Forms.TextBox();
            this.button_save = new System.Windows.Forms.Button();
            this.textBox_txtStart1 = new System.Windows.Forms.TextBox();
            this.textBox_question = new System.Windows.Forms.TextBox();
            this.groupBox_qa = new System.Windows.Forms.GroupBox();
            this.richTextBox_title = new System.Windows.Forms.RichTextBox();
            this.label_title = new System.Windows.Forms.Label();
            this.groupBox_previous_question = new System.Windows.Forms.GroupBox();
            this.richTextBox_prev_question = new System.Windows.Forms.RichTextBox();
            this.groupBox_statistics = new System.Windows.Forms.GroupBox();
            this.button_skipToSentence = new System.Windows.Forms.Button();
            this.textBox_skipToSentence = new System.Windows.Forms.TextBox();
            this.label_currSentence = new System.Windows.Forms.Label();
            this.textBox_question_count = new System.Windows.Forms.TextBox();
            this.label_question_count = new System.Windows.Forms.Label();
            this.textBox_total_sentences = new System.Windows.Forms.TextBox();
            this.label_total_sentences = new System.Windows.Forms.Label();
            this.textBox_total_questions = new System.Windows.Forms.TextBox();
            this.label_total_questions = new System.Windows.Forms.Label();
            this.button_Next = new System.Windows.Forms.Button();
            this.button_clear = new System.Windows.Forms.Button();
            this.groupBox_answertype = new System.Windows.Forms.GroupBox();
            this.radioButton_at_others = new System.Windows.Forms.RadioButton();
            this.radioButton_at_clause = new System.Windows.Forms.RadioButton();
            this.radioButton_at_verbphrase = new System.Windows.Forms.RadioButton();
            this.radioButton_at_adjectivephrase = new System.Windows.Forms.RadioButton();
            this.radioButton_at_nounphrase = new System.Windows.Forms.RadioButton();
            this.radioButton_at_person = new System.Windows.Forms.RadioButton();
            this.radioButton_at_otherentity = new System.Windows.Forms.RadioButton();
            this.radioButton_at_numberic = new System.Windows.Forms.RadioButton();
            this.radioButton_at_location = new System.Windows.Forms.RadioButton();
            this.radioButton_at_date = new System.Windows.Forms.RadioButton();
            this.groupBox_guide = new System.Windows.Forms.GroupBox();
            this.richTextBox_guide = new System.Windows.Forms.RichTextBox();
            this.checkBox_reasoning_synonymy = new System.Windows.Forms.CheckBox();
            this.checkBox_reasoning_knowledge = new System.Windows.Forms.CheckBox();
            this.checkBox_reasoning_syntactic = new System.Windows.Forms.CheckBox();
            this.checkBox_reasoning_multiplesentences = new System.Windows.Forms.CheckBox();
            this.checkBox_reasoning_ambiguous = new System.Windows.Forms.CheckBox();
            this.groupBox_answer_reasoning = new System.Windows.Forms.GroupBox();
            this.textBox_ans1len = new System.Windows.Forms.TextBox();
            this.groupBox_qa.SuspendLayout();
            this.groupBox_previous_question.SuspendLayout();
            this.groupBox_statistics.SuspendLayout();
            this.groupBox_answertype.SuspendLayout();
            this.groupBox_guide.SuspendLayout();
            this.groupBox_answer_reasoning.SuspendLayout();
            this.SuspendLayout();
            // 
            // label_sentence
            // 
            this.label_sentence.AutoSize = true;
            this.label_sentence.Location = new System.Drawing.Point(19, 57);
            this.label_sentence.Name = "label_sentence";
            this.label_sentence.Size = new System.Drawing.Size(65, 18);
            this.label_sentence.TabIndex = 0;
            this.label_sentence.Text = "Sentence";
            // 
            // richTextBox_sentence
            // 
            this.richTextBox_sentence.Font = new System.Drawing.Font("Times New Roman", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.richTextBox_sentence.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(64)))), ((int)(((byte)(0)))), ((int)(((byte)(64)))));
            this.richTextBox_sentence.Location = new System.Drawing.Point(22, 89);
            this.richTextBox_sentence.Name = "richTextBox_sentence";
            this.richTextBox_sentence.ReadOnly = true;
            this.richTextBox_sentence.Size = new System.Drawing.Size(520, 284);
            this.richTextBox_sentence.TabIndex = 10;
            this.richTextBox_sentence.TabStop = false;
            this.richTextBox_sentence.Text = "";
            // 
            // label_question
            // 
            this.label_question.AutoSize = true;
            this.label_question.Location = new System.Drawing.Point(19, 387);
            this.label_question.Name = "label_question";
            this.label_question.Size = new System.Drawing.Size(66, 18);
            this.label_question.TabIndex = 2;
            this.label_question.Text = "Question";
            // 
            // label_answer1
            // 
            this.label_answer1.AutoSize = true;
            this.label_answer1.Location = new System.Drawing.Point(19, 523);
            this.label_answer1.Name = "label_answer1";
            this.label_answer1.Size = new System.Drawing.Size(68, 18);
            this.label_answer1.TabIndex = 4;
            this.label_answer1.Text = "Answer 1";
            // 
            // textBox_answer1
            // 
            this.textBox_answer1.Font = new System.Drawing.Font("Times New Roman", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBox_answer1.Location = new System.Drawing.Point(120, 516);
            this.textBox_answer1.Multiline = true;
            this.textBox_answer1.Name = "textBox_answer1";
            this.textBox_answer1.Size = new System.Drawing.Size(359, 105);
            this.textBox_answer1.TabIndex = 1;
            this.textBox_answer1.TextChanged += new System.EventHandler(this.textBox_answer1_TextChanged);
            // 
            // button_save
            // 
            this.button_save.Location = new System.Drawing.Point(231, 663);
            this.button_save.Name = "button_save";
            this.button_save.Size = new System.Drawing.Size(135, 38);
            this.button_save.TabIndex = 10;
            this.button_save.Text = "Save";
            this.button_save.UseVisualStyleBackColor = true;
            this.button_save.Click += new System.EventHandler(this.button_save_Click);
            // 
            // textBox_txtStart1
            // 
            this.textBox_txtStart1.Location = new System.Drawing.Point(497, 516);
            this.textBox_txtStart1.Name = "textBox_txtStart1";
            this.textBox_txtStart1.ReadOnly = true;
            this.textBox_txtStart1.Size = new System.Drawing.Size(45, 25);
            this.textBox_txtStart1.TabIndex = 12;
            this.textBox_txtStart1.TabStop = false;
            this.textBox_txtStart1.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // textBox_question
            // 
            this.textBox_question.Font = new System.Drawing.Font("Times New Roman", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBox_question.Location = new System.Drawing.Point(22, 423);
            this.textBox_question.Multiline = true;
            this.textBox_question.Name = "textBox_question";
            this.textBox_question.Size = new System.Drawing.Size(520, 65);
            this.textBox_question.TabIndex = 1;
            // 
            // groupBox_qa
            // 
            this.groupBox_qa.Controls.Add(this.textBox_ans1len);
            this.groupBox_qa.Controls.Add(this.richTextBox_title);
            this.groupBox_qa.Controls.Add(this.label_title);
            this.groupBox_qa.Controls.Add(this.textBox_question);
            this.groupBox_qa.Controls.Add(this.label_sentence);
            this.groupBox_qa.Controls.Add(this.richTextBox_sentence);
            this.groupBox_qa.Controls.Add(this.textBox_txtStart1);
            this.groupBox_qa.Controls.Add(this.label_question);
            this.groupBox_qa.Controls.Add(this.label_answer1);
            this.groupBox_qa.Controls.Add(this.textBox_answer1);
            this.groupBox_qa.Location = new System.Drawing.Point(12, 9);
            this.groupBox_qa.Name = "groupBox_qa";
            this.groupBox_qa.Size = new System.Drawing.Size(562, 648);
            this.groupBox_qa.TabIndex = 17;
            this.groupBox_qa.TabStop = false;
            this.groupBox_qa.Text = "Field";
            // 
            // richTextBox_title
            // 
            this.richTextBox_title.Font = new System.Drawing.Font("Times New Roman", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.richTextBox_title.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(0)))), ((int)(((byte)(192)))));
            this.richTextBox_title.Location = new System.Drawing.Point(66, 26);
            this.richTextBox_title.Name = "richTextBox_title";
            this.richTextBox_title.ReadOnly = true;
            this.richTextBox_title.Size = new System.Drawing.Size(476, 25);
            this.richTextBox_title.TabIndex = 17;
            this.richTextBox_title.TabStop = false;
            this.richTextBox_title.Text = "";
            // 
            // label_title
            // 
            this.label_title.AutoSize = true;
            this.label_title.Location = new System.Drawing.Point(22, 29);
            this.label_title.Name = "label_title";
            this.label_title.Size = new System.Drawing.Size(38, 18);
            this.label_title.TabIndex = 16;
            this.label_title.Text = "Title";
            // 
            // groupBox_previous_question
            // 
            this.groupBox_previous_question.Controls.Add(this.richTextBox_prev_question);
            this.groupBox_previous_question.Location = new System.Drawing.Point(605, 9);
            this.groupBox_previous_question.Name = "groupBox_previous_question";
            this.groupBox_previous_question.Size = new System.Drawing.Size(464, 211);
            this.groupBox_previous_question.TabIndex = 18;
            this.groupBox_previous_question.TabStop = false;
            this.groupBox_previous_question.Text = "Previous Questions";
            // 
            // richTextBox_prev_question
            // 
            this.richTextBox_prev_question.Font = new System.Drawing.Font("Times New Roman", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.richTextBox_prev_question.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(0)))), ((int)(((byte)(64)))));
            this.richTextBox_prev_question.Location = new System.Drawing.Point(15, 30);
            this.richTextBox_prev_question.Name = "richTextBox_prev_question";
            this.richTextBox_prev_question.ReadOnly = true;
            this.richTextBox_prev_question.Size = new System.Drawing.Size(432, 165);
            this.richTextBox_prev_question.TabIndex = 0;
            this.richTextBox_prev_question.TabStop = false;
            this.richTextBox_prev_question.Text = "";
            // 
            // groupBox_statistics
            // 
            this.groupBox_statistics.Controls.Add(this.button_skipToSentence);
            this.groupBox_statistics.Controls.Add(this.textBox_skipToSentence);
            this.groupBox_statistics.Controls.Add(this.label_currSentence);
            this.groupBox_statistics.Controls.Add(this.textBox_question_count);
            this.groupBox_statistics.Controls.Add(this.label_question_count);
            this.groupBox_statistics.Controls.Add(this.textBox_total_sentences);
            this.groupBox_statistics.Controls.Add(this.label_total_sentences);
            this.groupBox_statistics.Controls.Add(this.textBox_total_questions);
            this.groupBox_statistics.Controls.Add(this.label_total_questions);
            this.groupBox_statistics.Location = new System.Drawing.Point(1086, 12);
            this.groupBox_statistics.Name = "groupBox_statistics";
            this.groupBox_statistics.Size = new System.Drawing.Size(256, 208);
            this.groupBox_statistics.TabIndex = 19;
            this.groupBox_statistics.TabStop = false;
            this.groupBox_statistics.Text = "Statistics";
            // 
            // button_skipToSentence
            // 
            this.button_skipToSentence.Location = new System.Drawing.Point(175, 144);
            this.button_skipToSentence.Name = "button_skipToSentence";
            this.button_skipToSentence.Size = new System.Drawing.Size(38, 25);
            this.button_skipToSentence.TabIndex = 8;
            this.button_skipToSentence.Text = "Go";
            this.button_skipToSentence.UseVisualStyleBackColor = true;
            this.button_skipToSentence.Click += new System.EventHandler(this.button_skipToSentence_Click);
            // 
            // textBox_skipToSentence
            // 
            this.textBox_skipToSentence.Location = new System.Drawing.Point(95, 144);
            this.textBox_skipToSentence.Name = "textBox_skipToSentence";
            this.textBox_skipToSentence.Size = new System.Drawing.Size(65, 25);
            this.textBox_skipToSentence.TabIndex = 7;
            this.textBox_skipToSentence.TabStop = false;
            // 
            // label_currSentence
            // 
            this.label_currSentence.AutoSize = true;
            this.label_currSentence.Location = new System.Drawing.Point(16, 151);
            this.label_currSentence.Name = "label_currSentence";
            this.label_currSentence.Size = new System.Drawing.Size(73, 18);
            this.label_currSentence.TabIndex = 6;
            this.label_currSentence.Text = "Sentence: ";
            // 
            // textBox_question_count
            // 
            this.textBox_question_count.Location = new System.Drawing.Point(143, 105);
            this.textBox_question_count.Name = "textBox_question_count";
            this.textBox_question_count.ReadOnly = true;
            this.textBox_question_count.Size = new System.Drawing.Size(70, 25);
            this.textBox_question_count.TabIndex = 5;
            this.textBox_question_count.TabStop = false;
            this.textBox_question_count.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label_question_count
            // 
            this.label_question_count.AutoSize = true;
            this.label_question_count.Location = new System.Drawing.Point(16, 112);
            this.label_question_count.Name = "label_question_count";
            this.label_question_count.Size = new System.Drawing.Size(109, 18);
            this.label_question_count.TabIndex = 4;
            this.label_question_count.Text = "Question Count";
            // 
            // textBox_total_sentences
            // 
            this.textBox_total_sentences.Location = new System.Drawing.Point(143, 64);
            this.textBox_total_sentences.Name = "textBox_total_sentences";
            this.textBox_total_sentences.ReadOnly = true;
            this.textBox_total_sentences.Size = new System.Drawing.Size(70, 25);
            this.textBox_total_sentences.TabIndex = 3;
            this.textBox_total_sentences.TabStop = false;
            this.textBox_total_sentences.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label_total_sentences
            // 
            this.label_total_sentences.AutoSize = true;
            this.label_total_sentences.Location = new System.Drawing.Point(16, 71);
            this.label_total_sentences.Name = "label_total_sentences";
            this.label_total_sentences.Size = new System.Drawing.Size(113, 18);
            this.label_total_sentences.TabIndex = 2;
            this.label_total_sentences.Text = "Total Sentences:";
            // 
            // textBox_total_questions
            // 
            this.textBox_total_questions.Location = new System.Drawing.Point(143, 23);
            this.textBox_total_questions.Name = "textBox_total_questions";
            this.textBox_total_questions.ReadOnly = true;
            this.textBox_total_questions.Size = new System.Drawing.Size(70, 25);
            this.textBox_total_questions.TabIndex = 1;
            this.textBox_total_questions.TabStop = false;
            this.textBox_total_questions.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label_total_questions
            // 
            this.label_total_questions.AutoSize = true;
            this.label_total_questions.Location = new System.Drawing.Point(16, 30);
            this.label_total_questions.Name = "label_total_questions";
            this.label_total_questions.Size = new System.Drawing.Size(109, 18);
            this.label_total_questions.TabIndex = 0;
            this.label_total_questions.Text = "Total Quesions:";
            // 
            // button_Next
            // 
            this.button_Next.Location = new System.Drawing.Point(439, 663);
            this.button_Next.Name = "button_Next";
            this.button_Next.Size = new System.Drawing.Size(135, 38);
            this.button_Next.TabIndex = 20;
            this.button_Next.Text = "Next";
            this.button_Next.UseVisualStyleBackColor = true;
            this.button_Next.Click += new System.EventHandler(this.button_Next_Click);
            // 
            // button_clear
            // 
            this.button_clear.Location = new System.Drawing.Point(12, 663);
            this.button_clear.Name = "button_clear";
            this.button_clear.Size = new System.Drawing.Size(135, 38);
            this.button_clear.TabIndex = 21;
            this.button_clear.TabStop = false;
            this.button_clear.Text = "Clear";
            this.button_clear.UseVisualStyleBackColor = true;
            this.button_clear.Click += new System.EventHandler(this.button_clear_Click);
            // 
            // groupBox_answertype
            // 
            this.groupBox_answertype.Controls.Add(this.radioButton_at_others);
            this.groupBox_answertype.Controls.Add(this.radioButton_at_clause);
            this.groupBox_answertype.Controls.Add(this.radioButton_at_verbphrase);
            this.groupBox_answertype.Controls.Add(this.radioButton_at_adjectivephrase);
            this.groupBox_answertype.Controls.Add(this.radioButton_at_nounphrase);
            this.groupBox_answertype.Controls.Add(this.radioButton_at_person);
            this.groupBox_answertype.Controls.Add(this.radioButton_at_otherentity);
            this.groupBox_answertype.Controls.Add(this.radioButton_at_numberic);
            this.groupBox_answertype.Controls.Add(this.radioButton_at_location);
            this.groupBox_answertype.Controls.Add(this.radioButton_at_date);
            this.groupBox_answertype.Location = new System.Drawing.Point(605, 238);
            this.groupBox_answertype.Name = "groupBox_answertype";
            this.groupBox_answertype.Size = new System.Drawing.Size(737, 144);
            this.groupBox_answertype.TabIndex = 22;
            this.groupBox_answertype.TabStop = false;
            this.groupBox_answertype.Text = "Answer Type";
            // 
            // radioButton_at_others
            // 
            this.radioButton_at_others.AutoSize = true;
            this.radioButton_at_others.Location = new System.Drawing.Point(564, 67);
            this.radioButton_at_others.Name = "radioButton_at_others";
            this.radioButton_at_others.Size = new System.Drawing.Size(64, 22);
            this.radioButton_at_others.TabIndex = 25;
            this.radioButton_at_others.Text = "Other";
            this.radioButton_at_others.UseVisualStyleBackColor = true;
            // 
            // radioButton_at_clause
            // 
            this.radioButton_at_clause.AutoSize = true;
            this.radioButton_at_clause.Location = new System.Drawing.Point(356, 105);
            this.radioButton_at_clause.Name = "radioButton_at_clause";
            this.radioButton_at_clause.Size = new System.Drawing.Size(68, 22);
            this.radioButton_at_clause.TabIndex = 25;
            this.radioButton_at_clause.Text = "Clause";
            this.radioButton_at_clause.UseVisualStyleBackColor = true;
            // 
            // radioButton_at_verbphrase
            // 
            this.radioButton_at_verbphrase.AutoSize = true;
            this.radioButton_at_verbphrase.Location = new System.Drawing.Point(356, 69);
            this.radioButton_at_verbphrase.Name = "radioButton_at_verbphrase";
            this.radioButton_at_verbphrase.Size = new System.Drawing.Size(103, 22);
            this.radioButton_at_verbphrase.TabIndex = 24;
            this.radioButton_at_verbphrase.Text = "Verb Phrase";
            this.radioButton_at_verbphrase.UseVisualStyleBackColor = true;
            // 
            // radioButton_at_adjectivephrase
            // 
            this.radioButton_at_adjectivephrase.AutoSize = true;
            this.radioButton_at_adjectivephrase.Location = new System.Drawing.Point(356, 35);
            this.radioButton_at_adjectivephrase.Name = "radioButton_at_adjectivephrase";
            this.radioButton_at_adjectivephrase.Size = new System.Drawing.Size(133, 22);
            this.radioButton_at_adjectivephrase.TabIndex = 23;
            this.radioButton_at_adjectivephrase.Text = "Adjective Phrase";
            this.radioButton_at_adjectivephrase.UseVisualStyleBackColor = true;
            // 
            // radioButton_at_nounphrase
            // 
            this.radioButton_at_nounphrase.AutoSize = true;
            this.radioButton_at_nounphrase.Location = new System.Drawing.Point(173, 107);
            this.radioButton_at_nounphrase.Name = "radioButton_at_nounphrase";
            this.radioButton_at_nounphrase.Size = new System.Drawing.Size(108, 22);
            this.radioButton_at_nounphrase.TabIndex = 25;
            this.radioButton_at_nounphrase.Text = "Noun Phrase";
            this.radioButton_at_nounphrase.UseVisualStyleBackColor = true;
            // 
            // radioButton_at_person
            // 
            this.radioButton_at_person.AutoSize = true;
            this.radioButton_at_person.Location = new System.Drawing.Point(15, 110);
            this.radioButton_at_person.Name = "radioButton_at_person";
            this.radioButton_at_person.Size = new System.Drawing.Size(69, 22);
            this.radioButton_at_person.TabIndex = 2;
            this.radioButton_at_person.Text = "Person";
            this.radioButton_at_person.UseVisualStyleBackColor = true;
            // 
            // radioButton_at_otherentity
            // 
            this.radioButton_at_otherentity.AutoSize = true;
            this.radioButton_at_otherentity.Location = new System.Drawing.Point(173, 70);
            this.radioButton_at_otherentity.Name = "radioButton_at_otherentity";
            this.radioButton_at_otherentity.Size = new System.Drawing.Size(108, 22);
            this.radioButton_at_otherentity.TabIndex = 24;
            this.radioButton_at_otherentity.Text = "Other Entity";
            this.radioButton_at_otherentity.UseVisualStyleBackColor = true;
            // 
            // radioButton_at_numberic
            // 
            this.radioButton_at_numberic.AutoSize = true;
            this.radioButton_at_numberic.Location = new System.Drawing.Point(15, 73);
            this.radioButton_at_numberic.Name = "radioButton_at_numberic";
            this.radioButton_at_numberic.Size = new System.Drawing.Size(89, 22);
            this.radioButton_at_numberic.TabIndex = 1;
            this.radioButton_at_numberic.Text = "Numberic";
            this.radioButton_at_numberic.UseVisualStyleBackColor = true;
            // 
            // radioButton_at_location
            // 
            this.radioButton_at_location.AutoSize = true;
            this.radioButton_at_location.Location = new System.Drawing.Point(173, 32);
            this.radioButton_at_location.Name = "radioButton_at_location";
            this.radioButton_at_location.Size = new System.Drawing.Size(82, 22);
            this.radioButton_at_location.TabIndex = 23;
            this.radioButton_at_location.Text = "Location";
            this.radioButton_at_location.UseVisualStyleBackColor = true;
            // 
            // radioButton_at_date
            // 
            this.radioButton_at_date.AutoSize = true;
            this.radioButton_at_date.Location = new System.Drawing.Point(15, 35);
            this.radioButton_at_date.Name = "radioButton_at_date";
            this.radioButton_at_date.Size = new System.Drawing.Size(57, 22);
            this.radioButton_at_date.TabIndex = 0;
            this.radioButton_at_date.Text = "Date";
            this.radioButton_at_date.UseVisualStyleBackColor = true;
            // 
            // groupBox_guide
            // 
            this.groupBox_guide.Controls.Add(this.richTextBox_guide);
            this.groupBox_guide.Location = new System.Drawing.Point(886, 396);
            this.groupBox_guide.Name = "groupBox_guide";
            this.groupBox_guide.Size = new System.Drawing.Size(456, 292);
            this.groupBox_guide.TabIndex = 24;
            this.groupBox_guide.TabStop = false;
            this.groupBox_guide.Text = "Guide";
            // 
            // richTextBox_guide
            // 
            this.richTextBox_guide.Font = new System.Drawing.Font("Times New Roman", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.richTextBox_guide.Location = new System.Drawing.Point(16, 24);
            this.richTextBox_guide.Name = "richTextBox_guide";
            this.richTextBox_guide.ReadOnly = true;
            this.richTextBox_guide.Size = new System.Drawing.Size(384, 304);
            this.richTextBox_guide.TabIndex = 0;
            this.richTextBox_guide.TabStop = false;
            this.richTextBox_guide.Text = "";
            // 
            // checkBox_reasoning_synonymy
            // 
            this.checkBox_reasoning_synonymy.AutoSize = true;
            this.checkBox_reasoning_synonymy.Location = new System.Drawing.Point(15, 48);
            this.checkBox_reasoning_synonymy.Name = "checkBox_reasoning_synonymy";
            this.checkBox_reasoning_synonymy.Size = new System.Drawing.Size(214, 22);
            this.checkBox_reasoning_synonymy.TabIndex = 0;
            this.checkBox_reasoning_synonymy.TabStop = false;
            this.checkBox_reasoning_synonymy.Text = "Lexical Variation (Synonymy)";
            this.checkBox_reasoning_synonymy.UseVisualStyleBackColor = true;
            // 
            // checkBox_reasoning_knowledge
            // 
            this.checkBox_reasoning_knowledge.AutoSize = true;
            this.checkBox_reasoning_knowledge.Location = new System.Drawing.Point(15, 102);
            this.checkBox_reasoning_knowledge.Name = "checkBox_reasoning_knowledge";
            this.checkBox_reasoning_knowledge.Size = new System.Drawing.Size(260, 22);
            this.checkBox_reasoning_knowledge.TabIndex = 30;
            this.checkBox_reasoning_knowledge.TabStop = false;
            this.checkBox_reasoning_knowledge.Text = "Lexical Variation (World knowledge)";
            this.checkBox_reasoning_knowledge.UseVisualStyleBackColor = true;
            // 
            // checkBox_reasoning_syntactic
            // 
            this.checkBox_reasoning_syntactic.AutoSize = true;
            this.checkBox_reasoning_syntactic.Location = new System.Drawing.Point(15, 154);
            this.checkBox_reasoning_syntactic.Name = "checkBox_reasoning_syntactic";
            this.checkBox_reasoning_syntactic.Size = new System.Drawing.Size(148, 22);
            this.checkBox_reasoning_syntactic.TabIndex = 32;
            this.checkBox_reasoning_syntactic.TabStop = false;
            this.checkBox_reasoning_syntactic.Text = "Syntactic Variation";
            this.checkBox_reasoning_syntactic.UseVisualStyleBackColor = true;
            // 
            // checkBox_reasoning_multiplesentences
            // 
            this.checkBox_reasoning_multiplesentences.AutoSize = true;
            this.checkBox_reasoning_multiplesentences.Location = new System.Drawing.Point(15, 203);
            this.checkBox_reasoning_multiplesentences.Name = "checkBox_reasoning_multiplesentences";
            this.checkBox_reasoning_multiplesentences.Size = new System.Drawing.Size(216, 22);
            this.checkBox_reasoning_multiplesentences.TabIndex = 34;
            this.checkBox_reasoning_multiplesentences.TabStop = false;
            this.checkBox_reasoning_multiplesentences.Text = "Multiple Sentences Reasoning";
            this.checkBox_reasoning_multiplesentences.UseVisualStyleBackColor = true;
            // 
            // checkBox_reasoning_ambiguous
            // 
            this.checkBox_reasoning_ambiguous.AutoSize = true;
            this.checkBox_reasoning_ambiguous.Location = new System.Drawing.Point(15, 247);
            this.checkBox_reasoning_ambiguous.Name = "checkBox_reasoning_ambiguous";
            this.checkBox_reasoning_ambiguous.Size = new System.Drawing.Size(99, 22);
            this.checkBox_reasoning_ambiguous.TabIndex = 36;
            this.checkBox_reasoning_ambiguous.TabStop = false;
            this.checkBox_reasoning_ambiguous.Text = "Ambiguous";
            this.checkBox_reasoning_ambiguous.UseVisualStyleBackColor = true;
            // 
            // groupBox_answer_reasoning
            // 
            this.groupBox_answer_reasoning.Controls.Add(this.checkBox_reasoning_ambiguous);
            this.groupBox_answer_reasoning.Controls.Add(this.checkBox_reasoning_multiplesentences);
            this.groupBox_answer_reasoning.Controls.Add(this.checkBox_reasoning_syntactic);
            this.groupBox_answer_reasoning.Controls.Add(this.checkBox_reasoning_knowledge);
            this.groupBox_answer_reasoning.Controls.Add(this.checkBox_reasoning_synonymy);
            this.groupBox_answer_reasoning.Location = new System.Drawing.Point(605, 396);
            this.groupBox_answer_reasoning.Name = "groupBox_answer_reasoning";
            this.groupBox_answer_reasoning.Size = new System.Drawing.Size(275, 292);
            this.groupBox_answer_reasoning.TabIndex = 23;
            this.groupBox_answer_reasoning.TabStop = false;
            this.groupBox_answer_reasoning.Text = "Reasoning";
            // 
            // textBox_ans1len
            // 
            this.textBox_ans1len.Location = new System.Drawing.Point(497, 559);
            this.textBox_ans1len.Name = "textBox_ans1len";
            this.textBox_ans1len.ReadOnly = true;
            this.textBox_ans1len.Size = new System.Drawing.Size(45, 25);
            this.textBox_ans1len.TabIndex = 18;
            this.textBox_ans1len.TabStop = false;
            this.textBox_ans1len.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // Form_Input
            // 
            this.AcceptButton = this.button_save;
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 18F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1357, 709);
            this.Controls.Add(this.groupBox_guide);
            this.Controls.Add(this.groupBox_answer_reasoning);
            this.Controls.Add(this.groupBox_answertype);
            this.Controls.Add(this.button_clear);
            this.Controls.Add(this.button_Next);
            this.Controls.Add(this.groupBox_statistics);
            this.Controls.Add(this.groupBox_previous_question);
            this.Controls.Add(this.groupBox_qa);
            this.Controls.Add(this.button_save);
            this.Font = new System.Drawing.Font("Garamond", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Margin = new System.Windows.Forms.Padding(4);
            this.Name = "Form_Input";
            this.Text = "Simple Dataset Input";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Form_Input_FormClosing);
            this.Load += new System.EventHandler(this.Form_Input_Load);
            this.groupBox_qa.ResumeLayout(false);
            this.groupBox_qa.PerformLayout();
            this.groupBox_previous_question.ResumeLayout(false);
            this.groupBox_statistics.ResumeLayout(false);
            this.groupBox_statistics.PerformLayout();
            this.groupBox_answertype.ResumeLayout(false);
            this.groupBox_answertype.PerformLayout();
            this.groupBox_guide.ResumeLayout(false);
            this.groupBox_answer_reasoning.ResumeLayout(false);
            this.groupBox_answer_reasoning.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Label label_sentence;
        private System.Windows.Forms.RichTextBox richTextBox_sentence;
        private System.Windows.Forms.Label label_question;
        private System.Windows.Forms.Label label_answer1;
        private System.Windows.Forms.TextBox textBox_answer1;
        private System.Windows.Forms.Button button_save;
        private System.Windows.Forms.TextBox textBox_txtStart1;
        private System.Windows.Forms.TextBox textBox_question;
        private System.Windows.Forms.GroupBox groupBox_qa;
        private System.Windows.Forms.GroupBox groupBox_previous_question;
        private System.Windows.Forms.RichTextBox richTextBox_prev_question;
        private System.Windows.Forms.GroupBox groupBox_statistics;
        private System.Windows.Forms.TextBox textBox_total_sentences;
        private System.Windows.Forms.Label label_total_sentences;
        private System.Windows.Forms.TextBox textBox_total_questions;
        private System.Windows.Forms.Label label_total_questions;
        private System.Windows.Forms.TextBox textBox_question_count;
        private System.Windows.Forms.Label label_question_count;
        private System.Windows.Forms.Button button_Next;
        private System.Windows.Forms.Button button_clear;
        private System.Windows.Forms.GroupBox groupBox_answertype;
        private System.Windows.Forms.RadioButton radioButton_at_others;
        private System.Windows.Forms.RadioButton radioButton_at_clause;
        private System.Windows.Forms.RadioButton radioButton_at_verbphrase;
        private System.Windows.Forms.RadioButton radioButton_at_adjectivephrase;
        private System.Windows.Forms.RadioButton radioButton_at_nounphrase;
        private System.Windows.Forms.RadioButton radioButton_at_person;
        private System.Windows.Forms.RadioButton radioButton_at_otherentity;
        private System.Windows.Forms.RadioButton radioButton_at_numberic;
        private System.Windows.Forms.RadioButton radioButton_at_location;
        private System.Windows.Forms.RadioButton radioButton_at_date;
        private System.Windows.Forms.GroupBox groupBox_guide;
        private System.Windows.Forms.RichTextBox richTextBox_guide;
        private System.Windows.Forms.RichTextBox richTextBox_title;
        private System.Windows.Forms.Label label_title;
        private System.Windows.Forms.CheckBox checkBox_reasoning_synonymy;
        private System.Windows.Forms.CheckBox checkBox_reasoning_knowledge;
        private System.Windows.Forms.CheckBox checkBox_reasoning_syntactic;
        private System.Windows.Forms.CheckBox checkBox_reasoning_multiplesentences;
        private System.Windows.Forms.CheckBox checkBox_reasoning_ambiguous;
        private System.Windows.Forms.GroupBox groupBox_answer_reasoning;
        private System.Windows.Forms.Button button_skipToSentence;
        private System.Windows.Forms.TextBox textBox_skipToSentence;
        private System.Windows.Forms.Label label_currSentence;
        private System.Windows.Forms.TextBox textBox_ans1len;
    }
}

