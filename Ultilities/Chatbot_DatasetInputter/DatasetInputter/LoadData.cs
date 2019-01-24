using DatasetInputter.Class;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DatasetInputter
{
    class LoadData
    {
        string file_location = "OurGorgeousData.json";
        public Article article;
        public int curr_data_index = -1;
        public int curr_qa_index = -1;
        public void LoadJson()
        {
            using (StreamReader r = new StreamReader(file_location))
            {
                string json = r.ReadToEnd();
                article = JsonConvert.DeserializeObject<Article>(json);
                //data = unsorted_data.OrderBy(o => o.paragraphs.qas.Count).ToList();
            }
        }

        public QuyChe_Data LoadNextSentence()
        {
            curr_data_index += 1;
            curr_qa_index = -1;
            if (curr_data_index >= article.data.Count)
                curr_data_index = 1;
            DataStatistics.question_count_in_sentences = article.data[curr_data_index].paragraphs[0].qas.Count;
            return article.data[curr_data_index];
        }

        public QuyChe_Data LoadCurrSentence()
        {
            return article.data[curr_data_index];
        }

        public List<QuestionAnswer> LoadPreviousQuestionForCurrentSentence()
        {
            return article.data[curr_data_index].paragraphs[0].qas;
        }

        public void SaveQA(string id, string ques, string anw1, string anw2, string anw3, string startPos1, string startPos2, string startPos3)
        {
            QuestionAnswer new_qa = new QuestionAnswer();
            new_qa.id = id;
            new_qa.question = ques;
            new_qa.answers.Add(new Answer(anw1, startPos1));
            // new_qa.answer.Add(new Answer(anw2, startPos2));
            // new_qa.answer.Add(new Answer(anw3, startPos3));
            article.data[curr_data_index].paragraphs[0].qas.Add(new_qa);
        }

        internal void SaveJSONToFile()
        {
            string json = JsonConvert.SerializeObject(article);

            //write string to file
            System.IO.File.WriteAllText(file_location, json);
        }

        internal QuestionAnswer GetNextQA()
        {
            try
            {
                curr_qa_index++;
                QuestionAnswer res = article.data[curr_data_index].paragraphs[0].qas[curr_qa_index];
                return res;
            }
            catch (ArgumentOutOfRangeException)
            {
                return null;
            }
            
        }

        internal void UpdateQA(string question, string answer, string txtStart)
        {
            article.data[curr_data_index].paragraphs[0].qas[curr_qa_index].question = question;
            article.data[curr_data_index].paragraphs[0].qas[curr_qa_index].answers[0].text = answer;

            int temp = 0;
            Int32.TryParse(txtStart, out temp);
            article.data[curr_data_index].paragraphs[0].qas[curr_qa_index].answers[0].answer_start = temp;
        }
    }
}
