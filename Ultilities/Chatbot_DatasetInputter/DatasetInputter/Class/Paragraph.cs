using System.Collections.Generic;

namespace DatasetInputter
{
    public class Paragraph
    {
        public Paragraph()
        {
            qas = new List<QuestionAnswer>();
        }

        public string context { get; set; }
        public List<QuestionAnswer> qas { get; set; }

    }
}