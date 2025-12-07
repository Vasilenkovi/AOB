import pandas as pd
import requests
from PyPDF2 import PdfReader
import pdfplumber
import io
import time
import random

# Функция для скачивания и парсинга PDF
# def download_and_parse_pdf(pdf_url):
#     try:
#         response = requests.get(pdf_url)
#         if response.status_code == 200:
#             pdf_content = io.BytesIO(response.content)
#             pdf_reader = PdfReader(pdf_content)
#             full_text = "\n".join(
#                 [page.extract_text() for page in pdf_reader.pages]
#             )
#             return full_text
#         else:
#             print(f"Ошибка скачивания PDF: {pdf_url}")
#             return None
#     except Exception as e:
#         print(f"Ошибка при обработке PDF {pdf_url}: {e}")
#         return None


def download_and_parse_pdf(pdf_url):
    try:
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                full_text = "\n".join([page.extract_text() for page in pdf.pages])
            return full_text
        else:
            print(f"Ошибка скачивания PDF: {pdf_url}")
            return None
    except Exception as e:
        print(f"Ошибка при обработке PDF {pdf_url}: {e}")
        return None


# Основной цикл
def process_batch(**kwargs):
    # Загружаем данные
    data_arxiv = pd.read_csv(kwargs["data_arxiv_path"])
    try:
        results = pd.read_csv(kwargs["results_path"])
        last_index = results['id'].max()  # Последний индекс в results.csv
    except FileNotFoundError:
        results = pd.DataFrame(columns=['id', 'full_text'])
        last_index = -1  # Если файл пустой, начинаем с 0

    batch_size = random.randint(kwargs["batch_min"], kwargs["batch_max"])
    # Определяем следующий индекс для обработки
    start_index = last_index + 1
    end_index = start_index + batch_size

    # Обрабатываем партию
    batch = data_arxiv.iloc[start_index:end_index]
    for index, row in batch.iterrows():
        pdf_url = row['pdf_url']
        entry_id = index
        print(f"Обработка: {entry_id} ({pdf_url})")

        full_text = download_and_parse_pdf(pdf_url)
        if full_text:
            new_row = pd.DataFrame({'id': [entry_id], 'full_text': [full_text]})
            results = pd.concat([results, new_row], ignore_index=True)
            results.to_csv(kwargs["results_path"], index=False)
        delay_between_links = random.randint(
            kwargs["delay_between_links_min"],
            kwargs["delay_between_links_max"]
        )
        # Задержка между ссылками
        #time.sleep(random.randint(delay_between_links["delay_between_links_min"], delay_between_links["delay_between_links_max"]))

    # Задержка между партиями
    delay_between_batches = random.randint(
        kwargs["delay_between_batches_min"],
        kwargs["delay_between_batches_max"]
    )

    print(
        f"Обработано с {start_index} по {end_index}. "
        f"Следующая партия через {delay_between_batches // 60} минут."
    )

    time.sleep(delay_between_batches)


while True:
    process_batch(
        data_arxiv_path="arxiv_ai.csv",
        results_path="results.csv",
        batch_min=20,
        batch_max=30,  # Количество ссылок за подход
        delay_between_links_min=1,
        delay_between_links_max=2,  # Задержка между ссылками (секунды)
        delay_between_batches_min=50,
        delay_between_batches_max=100  # Задержка между партиями (секунды)
    )
