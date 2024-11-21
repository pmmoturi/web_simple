#!/usr/bin/env python3

from flask import Flask, render_template, request, redirect, url_for, flash, Response
import fitz  # PyMuPDF
import psycopg2
import re
from datetime import datetime
import pytesseract
from PIL import Image
import os
from werkzeug.utils import secure_filename
import json
from openai import OpenAI
from openai import OpenAIError
import openai
from collections import defaultdict
import csv
import pandas as pd
from io import StringIO
from config import DB_CONFIG, OPENAI_API_KEY, FLASK_SECRET_KEY, UPLOAD_FOLDER

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def ask_chatgpt(context, question):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with access to MPESA statement data."},
                {"role": "user", "content": f"Context: {context}"},
                {"role": "user", "content": f"Question: {question}"}
            ],
            max_tokens=150,
            temperature=0.5
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        print(f"OpenAI error occurred: {e}")
        return "Sorry, the service is currently experiencing high demand or quota issues. Please try again later."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred. Please try again later."

def extract_pdf_details(pdf_path, password):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    pdf_document = fitz.open(pdf_path)

    if pdf_document.authenticate(password):
        print("Password is correct, decrypting PDF.")
    else:
        print("Failed to authenticate PDF. Exiting...")
        pdf_document.close()
        return None

    first_page = pdf_document[0]
    pix = first_page.get_pixmap(dpi=300)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    first_page_text = pytesseract.image_to_string(image)

    user_name_match = re.search(r"Customer Name:\s*([A-Z\s]+)", first_page_text)
    user_phone_number_match = re.search(r"Mobile Number:\s*([\d\s]+)", first_page_text)
    email_address_match = re.search(r"Email Address:\s*([\w\.-]+@[\w\.-]+)", first_page_text)
    period_start_date_match = re.search(r"Statement Period:\s*([\d]{2} [A-Za-z]{3} [\d]{4})", first_page_text)
    period_end_date_match = re.search(r"Statement Period:.*?([\d]{2} [A-Za-z]{3} [\d]{4})", first_page_text)
    request_date_match = re.search(r"Request Date:\s*([\d]{2} [A-Za-z]{3} [\d]{4})", first_page_text)

    user_name = user_name_match.group(1).strip() if user_name_match else None
    user_name = re.sub(r'[^A-Z\s]', '', user_name) if user_name else None

    user_phone_number = user_phone_number_match.group(1).replace(" ", "").strip() if user_phone_number_match else None
    email_address = email_address_match.group(1).strip() if email_address_match else None
    period_start_date = datetime.strptime(period_start_date_match.group(1), "%d %b %Y").date() if period_start_date_match else None
    period_end_date = datetime.strptime(period_end_date_match.group(1), "%d %b %Y").date() if period_end_date_match else None
    request_date = datetime.strptime(request_date_match.group(1), "%d %b %Y") if request_date_match else datetime.now()

    insert_file_query = '''
    INSERT INTO mpesa_statement_files (file_name, user_phone_number, user_name, user_email, request_date, period_start_date, period_end_date)
    VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
    '''
    file_name = os.path.basename(pdf_path)
    cursor.execute(insert_file_query, (file_name, user_phone_number, user_name, email_address, request_date, period_start_date, period_end_date))
    mpesa_file_id = cursor.fetchone()[0]
    conn.commit()

    transaction_records = []

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        jsontext = page.get_text("json")

        if jsontext.strip():
            pdf_text = json.loads(jsontext)
            blocks = pdf_text.get("blocks", [])

            transaction_data = []
            for block in blocks:
                block_text = ""

                if "lines" in block:
                    for line in block["lines"]:
                        line_text = " ".join(span["text"] for span in line["spans"])
                        block_text += line_text + " "

                transaction_data.extend(block_text.split())

            i = 0
            while i < len(transaction_data):
                if len(transaction_data[i]) == 10 and transaction_data[i].isalnum() and transaction_data[i].isupper():
                    receipt_no = transaction_data[i]
                    completion_time = transaction_data[i+1] + " " + transaction_data[i+2]
                    details_end_idx = i + 3
                    while transaction_data[details_end_idx] != "Completed":
                        details_end_idx += 1
                    details = " ".join(transaction_data[i+3:details_end_idx])
                    transaction_status = transaction_data[details_end_idx]
                    amount = transaction_data[details_end_idx + 1].replace(",", "")
                    balance = transaction_data[details_end_idx + 2].replace(",", "")

                    paid_in = ""
                    withdrawn = ""

                    if amount.startswith("-"):
                        withdrawn = amount
                    else:
                        paid_in = amount

                    paid_in = float(paid_in) if paid_in else None
                    withdrawn = float(withdrawn) if withdrawn else None
                    balance = float(balance) if balance else None

                    insert_record_query = '''
                    INSERT INTO mpesa_records (receipt_no, completion_time, details, transaction_status, paid_in, withdrawn, balance, mpesa_file)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    '''
                    cursor.execute(insert_record_query, (receipt_no, completion_time, details, transaction_status, paid_in, withdrawn, balance, mpesa_file_id))
                    conn.commit()

                    i = details_end_idx + 3
                else:
                    i += 1

    print("Data successfully inserted into mpesa_statement_files and mpesa_records tables.")

    cursor.close()
    conn.close()
    pdf_document.close()

    return {
        "user_name": user_name,
        "user_phone_number": user_phone_number,
        "email_address": email_address,
        "period_start_date": period_start_date,
        "period_end_date": period_end_date,
        "request_date": request_date,
        "mpesa_file_id": mpesa_file_id
    }

def fetch_transaction_records(file_id):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM mpesa_records WHERE mpesa_file = %s", (file_id,))
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    return records

def fetch_statement_details(file_id):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM mpesa_statement_files WHERE id = %s", (file_id,))
    details = cursor.fetchone()
    cursor.close()
    conn.close()
    return details

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download/<int:file_id>/<file_format>')
def download_file(file_id, file_format):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT receipt_no, completion_time, details, transaction_status, paid_in, withdrawn, balance FROM mpesa_records WHERE mpesa_file = %s", (file_id,))
    records = cursor.fetchall()
    cursor.close()
    conn.close()

    if file_format == 'csv':
        si = StringIO()
        writer = csv.writer(si)
        writer.writerow(["Receipt No", "Completion Time", "Details", "Transaction Status", "Paid In", "Withdrawn", "Balance"])
        writer.writerows(records)
        output = si.getvalue().encode('utf-8')

        response = Response(output, mimetype='text/csv')
        response.headers["Content-Disposition"] = f"attachment; filename=mpesa_records_{file_id}.csv"
        return response

    elif file_format == 'xlsx':
        df = pd.DataFrame(records, columns=["Receipt No", "Completion Time", "Details", "Transaction Status", "Paid In", "Withdrawn", "Balance"])
        output = StringIO()
        df.to_excel(f"mpesa_records_{file_id}.xlsx", index=False)

        response = Response(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response.headers["Content-Disposition"] = f"attachment; filename=mpesa_records_{file_id}.xlsx"
        return response

    else:
        flash("Invalid file format.")
        return redirect(url_for('index'))

@app.route('/results/<file_id>')
def results(file_id):
    records = fetch_transaction_records(file_id)
    details = fetch_statement_details(file_id)

    monthly_fees = defaultdict(lambda: {'peer_to_peer': 0, 'paybill': 0, 'till': 0})
    for record in records:
        if "Customer Transfer of Funds Charge" in record[2]:
            monthly_fees[record[1][:7]]['peer_to_peer'] += float(record[5] or 0)
        elif "Pay Bill Charge" in record[2]:
            monthly_fees[record[1][:7]]['paybill'] += float(record[5] or 0)
        elif "Pay Merchant Charge" in record[2]:
            monthly_fees[record[1][:7]]['till'] += float(record[5] or 0)

    if 'monthly_fees' not in locals() or monthly_fees is None:
        monthly_fees = {}

    return render_template('result.html', details=details, records=records, monthly_fees=monthly_fees)

@app.route('/query_chatgpt/<int:file_id>', methods=['POST'])
def query_chatgpt(file_id):
    question = request.form['question']

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT receipt_no, completion_time, details, transaction_status, paid_in, withdrawn, balance 
        FROM mpesa_records WHERE mpesa_file = %s
    """, (file_id,))
    records = cursor.fetchall()
    cursor.close()
    conn.close()

    context = ""
    for record in records:
        context += f"Receipt No: {record[0]}, Completion Time: {record[1]}, Details: {record[2]}, "
        context += f"Transaction Status: {record[3]}, Paid In: {record[4]}, Withdrawn: {record[5]}, Balance: {record[6]}\n"

    response = ask_chatgpt(context, question)

    if 'monthly_fees' not in locals() or monthly_fees is None:
        monthly_fees = {}

    return render_template('result.html', details={"mpesa_file_id": file_id}, records=records, query_response=response)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'password' not in request.form:
        flash('No file or password provided!')
        return redirect(url_for('index'))

    file = request.files['file']
    password = request.form['password']

    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    details = extract_pdf_details(filepath, password)
    if details:
        mpesa_file_id = details['mpesa_file_id']

        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT receipt_no, completion_time, details, transaction_status, paid_in, withdrawn, balance FROM mpesa_records WHERE mpesa_file = %s", (mpesa_file_id,))
        records = cursor.fetchall()
        cursor.close()
        conn.close()

        if 'monthly_fees' not in locals() or monthly_fees is None:
            monthly_fees = {}

        return render_template('result.html', details=details, records=records, query_response=None)
    else:
        flash('Failed to process PDF. Check password or try again.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)