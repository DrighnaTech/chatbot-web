from flask import Flask, request, render_template, jsonify
import os
from dotenv import load_dotenv
import sqlite3
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from aaa import openai_key

app = Flask(__name__)

# Load environment variables
os.environ["OPENAI_API_KEY"] = openai_key
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
Model = 'gpt-3.5-turbo'

# Set up the chat model and parser
model = ChatOpenAI(api_key=API_KEY, model=Model)
parser = StrOutputParser()

# Define the question template
question_template = """You are an AI assistant dedicated to supporting educators in their teaching efforts. Your primary objective is to provide detailed, accurate, and contextually relevant responses to assist teachers effectively.

---

**Context:**

{context}

**Question:**

{question}

---

Your response should be:

- **Comprehensive:** Thoroughly cover all aspects of the query.
- **Clear:** Ensure that the information is easy to understand and well-organized.
- **Tailored:** Specifically address the needs of educators based on the provided context.
- **Professional:** Maintain a formal and respectful tone throughout the response.

Please proceed with the response."""
prompt = PromptTemplate.from_template(template=question_template)

# Database connection function
def get_db_connection():
    conn = sqlite3.connect('aidrighna11.db')
    conn.row_factory = sqlite3.Row
    return conn

def create_database():
    conn = sqlite3.connect('aidrighna11.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Lesson_plan_maker (
            S_no INTEGER PRIMARY KEY AUTOINCREMENT,
            Name VARCHAR(255),
            Input VARCHAR(255),
            Output VARCHAR(255),
            Start_Time DATETIME DEFAULT CURRENT_TIMESTAMP,
            Latency FLOAT,
            Tokens INT,
            Cost FLOAT
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()

create_database()

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/ask-question', methods=['POST'])
def ask_question():
    # Get the uploaded PDF file
    pdf_file = request.files['pdf_file']
    question = request.form['question']
    
    if pdf_file and question:
        # Save the uploaded PDF
        pdf_path = os.path.join('uploads', pdf_file.filename)
        pdf_file.save(pdf_path)

        # Load the PDF and split it into pages
        file_loader = PyPDFLoader(pdf_path)
        pages = file_loader.load_and_split()

        # Split pages into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        chunks = splitter.split_documents(pages)

        # Create vector storage and retriever
        vector_storage = FAISS.from_documents(pages, OpenAIEmbeddings())
        retriever = vector_storage.as_retriever()

        # Create the runnable chain
        result = RunnableParallel(context=retriever, question=RunnablePassthrough())
        chain = result | prompt | model | parser

        # Get the answer
        answer = chain.invoke(question)
        formatted_answer = format_answer_as_html(answer)

        # Save the question and answer to the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Lesson_plan_maker (Name, Input, Output, Latency, Tokens, Cost) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (pdf_file.filename, question, answer, 0, 0, 0))  # Adjust Latency, Tokens, and Cost as needed
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'answer': formatted_answer})
    else:
        return jsonify({'error': 'Invalid input'}), 400

@app.route('/lessons', methods=['GET'])
def get_lessons():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Lesson_plan_maker")
    lessons = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # Convert the lessons to a list of dictionaries
    lessons_list = [dict(row) for row in lessons]
    return jsonify(lessons_list)

@app.route('/view-lessons')
def view_lessons():
    return render_template('lessons.html')

def format_answer_as_html(answer):
    # Convert plain text to HTML
    html_content = f"<div style='font-family: Arial, sans-serif; background-color: transparent; padding: 20px; border-radius: 10px;'>"
    for line in answer.split('\n'):
        if line.startswith('**'):
            html_content += f"<h3 style='color: #333;'>{line.strip('**').strip()}</h3>"
        else:
            html_content += f"<p>{line}</p>"
    html_content += "</div>"
    return html_content

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
