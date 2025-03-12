import os
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
import logging
from dotenv import load_dotenv

# Bibliotecas para LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import langchain

# Bibliotecas para Gemini/Vertex AI
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Part
import vertexai

# Bibliotecas para exportação
from docx import Document
from fpdf import FPDF

load_dotenv()


# Inicializar Vertex AI
def init_vertex_ai(project_id: str, location: str = "us-central1"):
    """Inicializa a conexão com o Vertex AI."""
    vertexai.init(project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION"))
    return GenerativeModel("gemini-1.5-pro")

# Definição dos estados do grafo
class BookState(dict):
    """Classe para armazenar o estado do processo de criação do livro."""
    
    def __init__(self):
        super().__init__()
        self.update({
            "title": "",
            "genre": "",
            "target_audience": "",
            "outline": [],
            "chapters": {},
            "current_chapter": 0,
            "status": "start",
            "feedback": "",
        })

# Funções para cada etapa do processo
def get_book_info(state: BookState, model) -> BookState:
    """Obtém informações básicas sobre o livro se não estiverem definidas."""
    if not state["title"]:
        prompt = """
        Você é um assistente especializado em ajudar autores a planejar livros.
        Por favor, sugira um título para um novo livro, um gênero e o público-alvo.
        Responda em formato JSON com as chaves: "title", "genre", "target_audience".
        """
        
        response = model.generate_content(prompt)
        try:
            info = json.loads(response.text)
            state["title"] = info.get("title", "Livro sem título")
            state["genre"] = info.get("genre", "Ficção")
            state["target_audience"] = info.get("target_audience", "Adultos")
        except json.JSONDecodeError:
            # Fallback caso o modelo não retorne um JSON válido
            state["title"] = "Meu Novo Livro"
            state["genre"] = "Ficção"
            state["target_audience"] = "Geral"
    
    state["status"] = "book_info_collected"
    return state

def create_outline(state: BookState, model) -> BookState:
    """Cria o sumário do livro baseado nas informações fornecidas."""
    prompt = f"""
    Você é um especialista em estrutura narrativa. 
    Baseado nas seguintes informações, crie um sumário detalhado para um livro:
    
    Título: {state["title"]}
    Gênero: {state["genre"]}
    Público-Alvo: {state["target_audience"]}
    
    Inclua entre 5 e 10 capítulos com títulos e uma breve descrição para cada um.
    Responda em formato JSON com uma lista de objetos contendo "chapter_number", "chapter_title" e "chapter_description".
    """
    
    response = model.generate_content(prompt)
    try:
        outline_data = json.loads(response.text)
        state["outline"] = outline_data
        # Inicializa dicionário de capítulos vazio
        state["chapters"] = {item["chapter_number"]: {"title": item["chapter_title"], 
                                                   "description": item["chapter_description"],
                                                   "content": ""} 
                          for item in outline_data}
    except json.JSONDecodeError:
        # Fallback em caso de erro
        state["outline"] = [{"chapter_number": 1, "chapter_title": "Introdução", 
                          "chapter_description": "Capítulo introdutório do livro."}]
        state["chapters"] = {1: {"title": "Introdução", 
                              "description": "Capítulo introdutório do livro.",
                              "content": ""}}
    
    state["current_chapter"] = 1
    state["status"] = "outline_created"
    return state

def write_chapter(state: BookState, model) -> BookState:
    """Escreve o conteúdo para o capítulo atual."""
    current = state["current_chapter"]
    
    if current > len(state["chapters"]):
        state["status"] = "all_chapters_written"
        return state
    
    chapter_info = state["chapters"][current]
    prev_content = ""
    
    # Se houver um capítulo anterior, pega um resumo do seu conteúdo para contexto
    if current > 1 and state["chapters"].get(current-1, {}).get("content"):
        prev_chapter = state["chapters"][current-1]
        prev_content = f"""
        Resumo do capítulo anterior ({current-1}: {prev_chapter['title']}):
        {prev_chapter['content'][:500]}... (resumido)
        """
    
    prompt = f"""
    Você é um escritor profissional escrevendo um livro intitulado "{state['title']}" no gênero "{state['genre']}" 
    para o público "{state['target_audience']}".
    
    Agora você precisa escrever o Capítulo {current}: "{chapter_info['title']}".
    
    Descrição do capítulo: {chapter_info['description']}
    
    {prev_content}
    
    Escreva o conteúdo completo deste capítulo. Seja criativo, use diálogos quando apropriado, 
    descrições vívidas e mantenha consistência com a narrativa geral.
    O capítulo deve ter pelo menos 1500 palavras.
    """
    
    response = model.generate_content(prompt)
    state["chapters"][current]["content"] = response.text
    
    # Avança para o próximo capítulo
    state["current_chapter"] += 1
    
    # Verifica se todos os capítulos foram escritos
    if state["current_chapter"] > len(state["chapters"]):
        state["status"] = "all_chapters_written"
    else:
        state["status"] = "chapter_written"
    
    return state

def review_and_edit(state: BookState, model) -> BookState:
    """Revisa e edita o livro completo para garantir consistência."""
    # Criamos um resumo do livro para o modelo fazer sua revisão
    book_summary = f"""
    Título: {state['title']}
    Gênero: {state['genre']}
    Público-alvo: {state['target_audience']}
    
    Sumário:
    """
    
    for chapter_num, chapter_data in sorted(state["chapters"].items()):
        book_summary += f"\nCapítulo {chapter_num}: {chapter_data['title']} - {chapter_data['description'][:100]}..."
    
    prompt = f"""
    Você é um editor profissional revisando o seguinte livro:
    
    {book_summary}
    
    Por favor, forneça um feedback geral sobre a estrutura do livro e sugestões de melhoria.
    Concentre-se em aspectos como fluxo narrativo, consistência temática e apelo ao público-alvo.
    """
    
    response = model.generate_content(prompt)
    state["feedback"] = response.text
    state["status"] = "reviewed"
    return state

def export_book(state: BookState) -> BookState:
    """Exporta o livro para arquivos DOCX e PDF."""
    # Criar documento Word
    doc = Document()
    doc.add_heading(state["title"], 0)
    
    # Adicionar informações do livro
    doc.add_paragraph(f"Gênero: {state['genre']}")
    doc.add_paragraph(f"Público-alvo: {state['target_audience']}")
    
    # Adicionar cada capítulo
    for chapter_num, chapter_data in sorted(state["chapters"].items()):
        doc.add_heading(f"Capítulo {chapter_num}: {chapter_data['title']}", 1)
        doc.add_paragraph(chapter_data["content"])
    
    # Salvar arquivo Word
    doc_path = f"{state['title'].replace(' ', '_')}.docx"
    doc.save(doc_path)
    
    # Criar PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, state["title"], 0, 1, "C")
    pdf.set_font("Arial", size=12)
    
    # Adicionar informações do livro
    pdf.cell(0, 10, f"Gênero: {state['genre']}", 0, 1)
    pdf.cell(0, 10, f"Público-alvo: {state['target_audience']}", 0, 1)
    
    # Adicionar cada capítulo (versão simplificada por limitações da FPDF)
    for chapter_num, chapter_data in sorted(state["chapters"].items()):
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Capítulo {chapter_num}: {chapter_data['title']}", 0, 1)
        pdf.set_font("Arial", size=12)
        
        # Dividir o conteúdo em parágrafos para melhor formatação
        paragraphs = chapter_data["content"].split("\n\n")
        for paragraph in paragraphs:
            pdf.multi_cell(0, 10, paragraph)
    
    # Salvar PDF
    pdf_path = f"{state['title'].replace(' ', '_')}.pdf"
    pdf.output(pdf_path)
    
    state["status"] = "exported"
    state["export_paths"] = {"docx": doc_path, "pdf": pdf_path}
    return state

# Função para decidir o próximo estado
def router(state: BookState) -> str:
    """Decide qual deve ser o próximo estado baseado no status atual."""
    if state["status"] == "start":
        return "get_book_info"
    elif state["status"] == "book_info_collected":
        return "create_outline"
    elif state["status"] == "outline_created":
        return "write_chapter"
    elif state["status"] == "chapter_written":
        return "write_chapter"
    elif state["status"] == "all_chapters_written":
        return "review_and_edit"
    elif state["status"] == "reviewed":
        return "export_book"
    elif state["status"] == "exported":
        return END
    else:
        return END

# Montagem do grafo de estados
def create_book_agent(model):
    """Cria e retorna o agente de geração de livros."""
    # Definir o grafo
    workflow = StateGraph(BookState)
    
    # Adicionar nós
    workflow.add_node("get_book_info", lambda s: get_book_info(s, model))
    workflow.add_node("create_outline", lambda s: create_outline(s, model))
    workflow.add_node("write_chapter", lambda s: write_chapter(s, model))
    workflow.add_node("review_and_edit", lambda s: review_and_edit(s, model))
    workflow.add_node("export_book", export_book)
    
    # Adicionar arestas
    workflow.add_conditional_edges("", router)
    workflow.add_conditional_edges("get_book_info", router)
    workflow.add_conditional_edges("create_outline", router)
    workflow.add_conditional_edges("write_chapter", router)
    workflow.add_conditional_edges("review_and_edit", router)
    workflow.add_conditional_edges("export_book", router)
    
    # Compilar o grafo
    return workflow.compile()

# Função principal
def main(custom_title: str = "", 
         custom_genre: str = "", 
         custom_audience: str = ""):
    """Função principal para executar o agente de geração de livros."""
    # Inicializar o modelo
    model = init_vertex_ai(os.getenv("PROJECT_ID"))
    
    # Criar o agente
    book_agent = create_book_agent(model)
    
    # Estado inicial
    initial_state = BookState()
    if custom_title:
        initial_state["title"] = custom_title
    if custom_genre:
        initial_state["genre"] = custom_genre
    if custom_audience:
        initial_state["target_audience"] = custom_audience
    
    # Checkpoint para salvar o estado entre execuções
    memory = MemorySaver()
    
    # Executar o fluxo
    for output in book_agent.stream(initial_state, checkpointer=memory):
        if output.get("status") != "start":
            stage = output.get("status", "desconhecido")
            print(f"Concluído: {stage}")
            
            # Fornecer informações adicionais sobre o progresso
            if stage == "book_info_collected":
                print(f"Título: {output['title']}")
                print(f"Gênero: {output['genre']}")
                print(f"Público-alvo: {output['target_audience']}")
            elif stage == "outline_created":
                print(f"Sumário criado com {len(output['outline'])} capítulos")
            elif stage == "chapter_written":
                print(f"Capítulo {output['current_chapter']-1} concluído")
            elif stage == "exported":
                print(f"Livro exportado para:")
                print(f"- Word: {output['export_paths']['docx']}")
                print(f"- PDF: {output['export_paths']['pdf']}")
    
    # Obter o estado final
    final_state = memory.get_checkpoint()
    return final_state

# Exemplo de uso
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Agente de geração de livros usando Gemini e LangGraph")
    parser.add_argument("--title", default="", help="Título do livro (opcional)")
    parser.add_argument("--genre", default="", help="Gênero do livro (opcional)")
    parser.add_argument("--audience", default="", help="Público-alvo do livro (opcional)")
    
    args = parser.parse_args()
    
    result = main(args.title, args.genre, args.audience)
    print("Processo de geração de livro concluído!")