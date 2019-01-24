using System.Collections.Generic;

namespace DatasetInputter
{
    public class DataStatistics
    {
        public static int sentences_count;
        public static int question_count;
        public static int question_count_in_sentences;
        public static int[] answer_type_count;
        public static int[] reason_required_count;

        public DataStatistics(List<QuyChe_Data> data)
        {
            sentences_count = 0;
            question_count = 0;
            question_count_in_sentences = 0;

            sentences_count = data.Count;
            foreach (QuyChe_Data quyche in data)
                foreach (Paragraph paragraph in quyche.paragraphs)
                    question_count += paragraph.qas.Count;
        }
    }
}