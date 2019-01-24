using System;

namespace DatasetInputter
{
    public class Answer
    {
        public Answer(string anw1, string start_pos)
        {
            this.text = anw1;
            int temp = 0;
            Int32.TryParse(start_pos, out temp);
            answer_start = temp;
        }

        public string text { get; set; }
        public int answer_start { get; set; }
    }
}