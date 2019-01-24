using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DatasetInputter
{
    public class QuyChe_Data
    {
        public QuyChe_Data()
        {
            paragraphs = new List<Paragraph>();
        }

        public string title { get; set; }
        public List<Paragraph> paragraphs { get; set; }
    }
}
