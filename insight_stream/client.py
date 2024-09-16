from typing import Dict, List, Optional
import json
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition
import requests
import urllib.parse
import socket

EMBED_MODEL_URL = os.getenv('EMBED_MODEL_URL')
EMBED_MODEL_API_KEY = os.getenv('EMBED_MODEL_API_KEY')
EMBED_MODEL_NAME = os.getenv('EMBED_MODEL_NAME')

ENCODING_FORMAT = os.getenv('ENCODING_FORMAT')
TIKTOKEN_MODEL = os.getenv('TIKTOKEN_MODEL')
TIKTOKEN_ENABLED = bool(os.getenv('TIKTOKEN_ENABLED'))

CHAT_MODEL_URL = os.getenv('CHAT_MODEL_URL')
CHAT_MODEL_API_KEY = os.getenv('CHAT_MODEL_API_KEY')
CHAT_MODEL_NAME = os.getenv('CHAT_MODEL_NAME')

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_KEY = os.getenv('QDRANT_KEY')
QDRANT_VECTOR_SIZE = int(os.getenv('QDRANT_VECTOR_SIZE', 1536))

SERVER_NAME = os.getenv('SERVER_NAME')
TOKEN = os.getenv('TOKEN')

embeddings = OpenAIEmbeddings(model=EMBED_MODEL_NAME,
                              openai_api_base=EMBED_MODEL_URL,
                              openai_api_key=EMBED_MODEL_API_KEY,
                              model_kwargs={"encoding_format": ENCODING_FORMAT},
                              tiktoken_enabled=TIKTOKEN_ENABLED,
                              tiktoken_model_name=TIKTOKEN_MODEL,
                              )

llm = ChatOpenAI(
    openai_api_key=CHAT_MODEL_API_KEY,
    model=CHAT_MODEL_NAME,
    temperature=0,
    openai_api_base=CHAT_MODEL_URL,
	request_timeout=120,
    max_retries=5  
)

clientQdrant=QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)

def ask(
		bot_id: str,
		question: str
	) -> Optional[Dict[str, any]]:
	""""Получение ответа бота на вопрос пользователя"""
	request = requests.post(
		f'{SERVER_NAME}/v.1.0/{bot_id}',
		data=json.dumps({ "question": question }),
		headers={
			'Content-Type': 'application/json',
			'Authorization': f'Bearer {TOKEN}'
		}
	)
	if request.status_code == 200:
		return request.json()
	else:
		return None
	
def _add_documents(index_id: str, documents: List[Document], file_path: str) -> List[Document]:
	"""Добавляем в квадрант документы, являющиеся чанками для одного файла из папки"""
	try:
		clientQdrant.create_collection(
				collection_name=index_id,
				vectors_config=models.VectorParams(size=QDRANT_VECTOR_SIZE, distance=models.Distance.COSINE)
			)
	except:
		pass
	qdrant = Qdrant(client=clientQdrant, collection_name=index_id, embeddings=embeddings)

	docs_for_qdrant = []

	prompt_kw = ChatPromptTemplate.from_template("Выбери абревиатуры из текста {chunk} для которых есть расшифровка значения, в виде списка, разделенного запятыми. Каждая абревиатура нужна только один раз без повторений. Не выделяй абревиатуры ковычками. Абревиатуры типа ст., ч., п., РФ и стандартные сокращения (г., ..) не нужны, только специфические для данного стандарта. В ответ дай только список уникальных абревиатур. Используй русский язык и именительный падеж, не используй вступительные фразы а-ля 'Вот список ключевых слов и абревиатур' или 'Аббревиатуры:' Давай в ответ сами абревиатуры а не их номера. Не давай никаких примечаний")
	prompt_qa = ChatPromptTemplate.from_template("Сгенерируй основные вопросы, которые лучше отражают содержание текста {chunk} (учти, что текст это часть документа {name}). Дай только сам список вопросов на русском языке, разделяй вопросы с помощью символа переноса строки, нумеровать вопросы не нужно, учти, что далее по данным вопросам будет производится поиск нужного текса, следовательно вопросы должны передавать специфику текста. убедись, что вопросы не повторяются. если ссылаешься внутри вопроса на приказ или политику, указывай его номер. не используй вступительные фразы а-ля 'Вот основные вопросы на которые отвечает текст'.")
	
	chain_kw = prompt_kw | llm | StrOutputParser()
	chain_qa = prompt_qa | llm | StrOutputParser()

	for doc in documents:

		keywords = chain_kw.invoke({"chunk": doc.page_content})
		print("keywords:")
		print(keywords)

		questions = chain_qa.invoke({"chunk": doc.page_content, "name": os.path.splitext(os.path.basename(file_path))[0]})
		print("questions:")
		print(questions)


		new_doc = Document(page_content=keywords.replace('"', '') + "\n" + questions, metadata = {
			"topic": os.path.splitext(os.path.basename(file_path))[0], 
			"url": urllib.parse.quote(f'{SERVER_NAME}/documents/{os.path.basename(file_path)}', safe=":/"),
			"extension": os.path.splitext(os.path.basename(file_path))[1][1:].lower(),
			"keywords": keywords,
			"questions": questions,
			"page_content": doc.page_content},
			)
		docs_for_qdrant.append(new_doc)
		
	try:
		qdrant.add_documents(docs_for_qdrant)
	except Exception as e:
		print(e)
	return docs_for_qdrant


def delete_documents(index_id: str, documents: List[Document]):
	"""Удаление документов с сервера и из коллекции квадранта"""

	#удаление коллекции целиком
	clientQdrant.delete_collection(collection_name=index_id)

	#удаление соотвестующих документов на сервере
	urls_for_del = []
	for doc in documents:
		url = doc.metadata.get("url")
		if url and url not in urls_for_del:
			urls_for_del.append(url)

	for url in urls_for_del:
		_del_file_from_server(url)


def upload_doc(index_id: str, path: str) -> List[Document]:
	"""Загрузка документов в индекс квадранта и на сервер"""


	elements = partition(path)
	chunks = chunk_by_title(elements)


	#объединим чанки в более крупные
	final_chunks = []
	cur_text = ''

	l=1

	for chunk in chunks:
		text = str(chunk)
 
		if len(cur_text + text) <= 10000:
			cur_text += '\n' + text if cur_text else text
		else:
			print(len(cur_text))
			doc = Document(page_content=cur_text)
			final_chunks.append(doc)
			l+=1
			cur_text = text

	if cur_text:
		print(len(cur_text))
		doc = Document(page_content=cur_text)
		final_chunks.append(doc)

	#загрузка чанков документа в квадрант
	for chunk in final_chunks:
		try:
			docs = _add_documents(index_id, [chunk], path)
		except Exception as err:
			print(err)
			docs = _add_documents(index_id, [chunk], path)

	
	print(f"Файл {os.path.basename(path)} загружен в квадрант, index_id {index_id}")

	#загрузка файла на сервер
	url = _load_file_to_server(path)
	
	return docs

def upload_dir(index_id: str, path: str) -> List[Document]:
	"""Множественная загрузка документов в индекс квадранта и на сервер"""

	docs = []

	for file in os.listdir(path):
		file_path = os.path.join(path, file)
		if os.path.isfile(file_path):
			chunks = upload_doc(index_id, file_path)
			docs.extend(chunks)

	return docs

def _check_server_availability() -> bool:
	"Проверка доступности сервера"

	server_url = urllib.parse.urlparse(SERVER_NAME)
	host = server_url.hostname
	port = server_url.port

	if not port:
		if server_url.scheme == 'https':
			port = 443
		elif server_url.scheme == 'http':
			port = 80
			
	try:
		socket.create_connection((host, port), timeout=5)
		return True
	except socket.error:
		return False

def _load_file_to_server(file_path: str) -> str:
	"""Добавление файла на сервера"""

	if not _check_server_availability():
		print("Сервер для загрузки документов недоступен")
		return ''

	url = f"{SERVER_NAME}/documents/{os.path.basename(file_path)}"
	file_url = urllib.parse.quote(url, safe=":/")

	headers = {'Authorization': f'Bearer {TOKEN}'}
	files = {'file': open(file_path, 'rb')}
    
	response = requests.put(url, headers=headers, files=files)
    
	if response.status_code == 201:
		print(f"Успешно загружен файл {os.path.basename(file_path)} на сервер!")
		return file_url
	else:
		print(f"Загрузка файла {os.path.basename(file_path)} на сервер не удалась")
		return ''
	
def _del_file_from_server(url: str):
	"""Удаление файла с сервера"""

	if not _check_server_availability():
		print("Сервер недоступен")
		return ''

	headers = {'Authorization': f'Bearer {TOKEN}'}
	response = requests.delete(url, headers=headers)
    
	if response.status_code == 204:
		print(f"Успешно удален файл {url}!")
	else:
		print("Ошибка при удалении")
