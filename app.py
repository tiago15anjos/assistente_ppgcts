!pip install -q -U google-generativeai
import google.generativeai as genai
import numpy as np
import json

GOOGLE_API_KEY = "AIzaSyAjb_gLNCOIQKrxbldarbbPqQFDtcGbDso"
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "candidate_count": 1,
    "temperature": 0.7,
}
with open('perguntas_respostas.json', 'r') as file:
    dados = json.load(file)

embeddings = []
for item in dados:
    # Embedding para a pergunta
    embedding_pergunta = genai.embed_content(
        model="models/embedding-001",
        content=item['pergunta'],
        title=f"Pergunta: {item['pergunta']}",
        task_type="RETRIEVAL_DOCUMENT"
    )["embedding"]
    embeddings.append(embedding_pergunta)

def gerar_e_buscar_consulta(consulta, base, model):
    embedding_consulta = genai.embed_content(
        model=model,
        content=consulta,
        task_type="RETRIEVAL_QUERY"
    )["embedding"]

    # Calcular o produto escalar entre o embedding da consulta e os embeddings das perguntas
    produtos_escalares = np.dot(np.stack(base), embedding_consulta)

    # Encontrar o índice da pergunta mais próxima
    indice_pergunta_mais_proxima = np.argmax(produtos_escalares)

    return dados[indice_pergunta_mais_proxima]['resposta']

consulta = "Quem é o coordenador do curso?"
trecho = gerar_e_buscar_consulta(consulta, embeddings, "models/embedding-001")
print(trecho)

prompt = f"Reescreva esse texto de uma forma natural como se fosse uma assistente virtual do programa PPGCTS. sem adicionar informações que não façam parte do banco de dados. Em sua resposta, não apresente o comentário, somente a resposta direto: {trecho}"

model_2 = genai.GenerativeModel("gemini-1.0-pro")
response = model_2.generate_content(prompt)
print(response.text)

