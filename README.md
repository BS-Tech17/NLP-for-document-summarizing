# NLP-for-document-summarizing
This AI model uses NLTK for summarising any PDF. It firstly cleans the PDF using the regular expressions and then it tokenizes the text into sentences and then words. 
This project was specifically made for summarising PDFs regarding wells during my internship at ONGC, Dehradun
The arrays declared inside the code such as focus_terms and exclude_terms can be modified according to the the document to improve the quality of the text in the summary.
This program gives the output in the form of a text file which presents the summary of the document in a number of points specified in the program(15 by default but it can be changed by changing the value of the variable t). 
The predefined phrases can also be modified further.
