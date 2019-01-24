using System.Collections.Generic;

namespace DatasetInputter
{
    public class QuestionAnswer
    {
        public QuestionAnswer()
        {
            answers = new List<Answer>();
        }

        public string question { get; set; }
        public string id { get; set; }
        public List<Answer> answers { get; set; }
    }
}