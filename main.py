import os
# Biblioteca para separar variáveis de ambiente do código / pode ser alterado
from decouple import config

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

#Alterar SECRET_KEY conforme necessário para que se tenha a chave de acesso a API desejada
SECRET_KEY = config('GEMINI_API_KEY')

#print(SECRET_KEY)

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash',api_key=SECRET_KEY)

#llm.invoke("Sing a ballad of LangChain.")

#Dois possíveis exemplos básicos, um de texto e um de texto e imagem

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Tell me a joke.",
        }, 
    ]
)

test = llm.invoke([message])

print(test)

'''
Referências:
https://python.langchain.com/docs/integrations/platforms/google/
https://python.langchain.com/docs/integrations/chat/google_generative_ai/
https://youtu.be/M7EcMyjUfPk?list=PLbGui_ZYuhiii0nA9nPo_O01qVpoR7PsT
'''