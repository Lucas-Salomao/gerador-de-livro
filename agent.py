import os
from typing import Dict, List, Tuple, Any, TypedDict, Optional
import json
from pathlib import Path
import logging
import re
from dotenv import load_dotenv
import time

# Bibliotecas para LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import langchain

# Bibliotecas para Gemini/Vertex AI
# from google.cloud import aiplatform
# from vertexai.generative_models import GenerativeModel, Part
# import vertexai

import google.generativeai as genai

# Biblioteca para exportação
import docx
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

load_dotenv()

# # Configuração de logs
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("book_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inicializar Gemini API
def init_gemin_api():
    """Inicializa a conexão com o Gemini API."""
    logger.info("Inicializando Gemini API...")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai.GenerativeModel('gemini-2.5-flash')

# def init_vertex_ai():
#     """Inicializa a conexão com o Vertex AI usando credenciais do ambiente."""
#     logger.info("Inicializando Vertex AI...")
#     project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
#     if not project_id:
#         raise ValueError("A variável de ambiente GOOGLE_CLOUD_PROJECT não foi definida.")
    
#     vertexai.init(project=project_id)
    
#     # IMPORTANTE: O nome do modelo no Vertex AI pode ser um pouco diferente.
#     # Verifique na documentação do Vertex AI o identificador correto para o modelo que deseja.
#     # "gemini-1.5-flash-001" é um exemplo comum e robusto.
#     return GenerativeModel("gemini-2.0-flash")

# Função auxiliar para parsing seguro de JSON
def safe_json_parse(response_text: str, fallback: Any) -> Any:
    """Tenta decodificar JSON e retorna um fallback em caso de erro."""
    if response_text.startswith("```json"):
        response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
    elif response_text.startswith("```"):
        response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        logger.error(f"Erro ao decodificar JSON: {response_text[:100]}... Usando fallback.")
        return fallback

# Definição dos estados do grafo
class BookState(TypedDict, total=False):
    theme: str
    title: str
    area_tecnologica: str
    target_audience: str
    num_chapters: int
    outline: List[Dict[str, Any]]
    chapters: Dict[int, Dict[str, str]]
    current_chapter: int
    status: str
    feedback: str
    export_path: str
    feedback_path: str

# Funções para cada etapa do processo
def get_book_info(state: BookState, model) -> Dict[str, Any]:
    """Obtém informações básicas e gera o título com base no tema."""
    logger.info(f"Estado recebido em get_book_info: {state}")
    logger.info("Coletando informações básicas do livro...")
    updates = {}
    
    theme = state.get("theme", "Um tema genérico")
    area_tecnologica = state.get("area_tecnologica", "Não especificada")
    target_audience = state.get("target_audience", "Adultos")
    
    updates["theme"] = theme
    updates["area_tecnologica"] = area_tecnologica
    updates["target_audience"] = target_audience
    
    prompt = f"""
    Você é um especialista em redação técnica. Baseado no seguinte tema, área tecnológica e público-alvo, sugira um título formal e técnico que reflita um enfoque analítico e informativo:
    Tema: {theme}
    Área Tecnológica: {area_tecnologica}
    Público-Alvo: {target_audience}
    Responda SOMENTE em formato JSON com a chave "title", sem texto adicional. Exemplo: {{"title": "Fundamentos de Exploração Espacial"}}. Não inclua bloco de código, ou seja ```json```
    O Título deve ter no máximo 80 caracteres. Caracteres inválidos para o título: , \\ / : * ? " < > |
    """
    
    logger.info("Gerando título com base no tema...")
    response = generate_with_retry(model, prompt)
    logger.debug(f"Resposta bruta do modelo: {response.text}")
    info = safe_json_parse(response.text, {"title": f"Livro sobre {theme}"})
    updates["title"] = info.get("title", f"Livro sobre {theme}")
    logger.info(f"Título gerado: {updates['title']}")
    
    updates["status"] = "book_info_collected"
    logger.debug(f"Atualizações de get_book_info: {updates}")
    return updates

def create_outline(state: BookState, model) -> Dict[str, Any]:
    """Cria o sumário do livro baseado nas informações fornecidas."""
    logger.info("Criando sumário do livro...")
    
    prompt = f"""
    Você é um especialista técnico elaborando um livro técnico para estudo de um determinado tema. 
    Baseado nas seguintes informações, crie um sumário detalhado e com foco em aspectos técnicos e práticos:
    
    Tema: {state['theme']}
    Título sugerido: {state['title']}
    Área Tecnológica: {state.get('area_tecnologica', 'Não especificada')}
    Público-Alvo: {state['target_audience']}
    
    Cada capítulo deve ter uma numeração inteira e sequencial (exemplo: 1, 2, 3, 4, 5, etc.)

    O sumário deve conter exatamente {state['num_chapters']} capítulos, cada um abordando um aspecto técnico ou prático do tema, com títulos objetivos e descrições que detalhem o conteúdo analítico a ser explorado.
    Responda SOMENTE em formato JSON com uma lista de objetos contendo "chapter_number", "chapter_title" e "chapter_description".
    Exemplo: [{{"chapter_number": 1, "chapter_title": "Princípios de Propulsão Espacial", "chapter_description": "Análise dos sistemas de propulsão usados em missões espaciais"}}]
    Não inclua bloco de código, ou seja ```json```
    """

    response = generate_with_retry(model, prompt)
    if not response:
        logger.error("Falha ao gerar sumário.")
        return {"status": "error", "message": "Falha ao gerar sumário."}

    logger.debug(f"Resposta bruta do modelo: {response.text}")
    outline_data = safe_json_parse(response.text, [
        {"chapter_number": 1, "chapter_title": "Introdução", 
         "chapter_description": f"Exploração inicial do tema {state['theme']}."}
    ])
    
    if len(outline_data) < 5:
        logger.warning("Sumário com menos de 5 capítulos. Adicionando capítulos extras.")
        for i in range(len(outline_data) + 1, 6):
            outline_data.append({
                "chapter_number": i,
                "chapter_title": f"Capítulo {i}",
                "chapter_description": f"Continuação da exploração de {state['theme']}."
            })
    
    updates = {
        "outline": outline_data,
        "chapters": {item["chapter_number"]: {"title": item["chapter_title"], 
                                              "description": item["chapter_description"],
                                              "content": ""} 
                     for item in outline_data},
        "current_chapter": 1,
        "status": "outline_created"
    }
    logger.info(f"Sumário criado com {len(outline_data)} capítulos.")
    logger.info(f"Capítulos gerados: {updates['chapters']}")
    return updates

def write_chapter(state: BookState, model, st_session=None) -> Dict[str, Any]:
    """Escreve o conteúdo para o capítulo atual."""
    current = state["current_chapter"]
    updates = {}
    if current > len(state["chapters"]):
        updates["status"] = "all_chapters_written"
        logger.info("Todos os capítulos foram escritos.")
        return updates
    
    chapter_info = state["chapters"][current]
    logger.info(f"Escrevendo Capítulo {current}: {chapter_info['title']}...")
        
    prev_content = ""
    if current > 1 and state["chapters"].get(current-1, {}).get("content"):
        prev_chapter = state["chapters"][current-1]
        prev_content = f"""
        Resumo do capítulo anterior ({current-1}: {prev_chapter['title']}):
        {prev_chapter['content'][:500]}... (resumido)
        """
    
    prompt = f"""
    Você é um especialista técnico escrevendo um livro intitulado "{state['title']}" com o tema "{state['theme']}".
    A área tecnológica do livro é "{state.get('area_tecnologica', 'Não especificada')}", direcionado para o público "{state['target_audience']}"
    
    Escreva o Capítulo {current}: "{chapter_info['title']}".
    
    Descrição do capítulo: {chapter_info['description']}
    
    {prev_content}
    
    Escreva um texto técnico e analítico, com linguagem formal e objetiva. Inclua informações técnicas detalhadas, exemplos contextualizados (reais ou hipotéticos), dados relevantes e explicações claras. Evite diálogos narrativos ou descrições literárias excessivas. Estruture o conteúdo com seções claras (ex.: introdução, análise, exemplos, conclusão). O capítulo deve ter pelo menos 3000 palavras. Seja o mais detalhista possível e aborde o tema do capítulo com profundidade e bastante exemplo.
    Estruture o capítulo com títulos e subtítulos para facilitar a leitura e compreensão do conteúdo. Siga a numeração do capítulo e estruture os subtítulos com base na numeração do capítulo.
    """

    response = generate_with_retry(model, prompt)

    if not response:
        logger.error("Falha ao gerar conteúdo para o capítulo.")
        return {"status": "error", "message": "Falha ao gerar conteúdo para o capítulo."}

    updated_chapters = state["chapters"].copy()
    updated_chapters[current]["content"] = response.text
    updates["chapters"] = updated_chapters
    logger.info(f"Capítulo {current} concluído com sucesso.")
    
    # Exibir o conteúdo gerado no Streamlit
    if st_session:
        # st_session.write(f"### Capítulo {current}: {chapter_info['title']}")
        st_session.write(response.text)
    time.sleep(5)
    updates["current_chapter"] = current + 1
    updates["status"] = "chapter_written" if updates["current_chapter"] <= len(state["chapters"]) else "all_chapters_written"
    return updates

def review_and_edit(state: BookState, model) -> Dict[str, Any]:
    """Revisa e edita o livro completo."""
    logger.info("Revisando e editando o livro...")
    book_summary = f"""
    Tema: {state['theme']}
    Título: {state['title']}
    Área Tecnológica: {state.get('area_tecnologica', 'Não especificada')}
    Público-alvo: {state['target_audience']}
    
    Sumário:
    """
    for chapter_num, chapter_data in sorted(state["chapters"].items()):
        book_summary += f"\nCapítulo {chapter_num}: {chapter_data['title']} - {chapter_data['description'][:100]}..."
    
    prompt = f"""
    Você é um editor revisando o livro:
    
    {book_summary}
    
    Forneça feedback sobre estrutura, fluxo narrativo, consistência com o tema "{state['theme']}" 
    e apelo ao público-alvo. Sugira melhorias. Revise tecnicamente o livro e verifique se há alguma inconsistência.
    """

    response = generate_with_retry(model, prompt)

    if not response:
        logger.error("Falha ao gerar feedback.")
        return {"status": "error", "message": "Falha ao gerar feedback."}

    updates = {
        "feedback": response.text,
        "status": "reviewed"
    }
    logger.info("Revisão concluída. Feedback gerado.")
    
    return updates

def export_feedback(state: BookState) -> Dict[str, Any]:
    """Exporta o feedback para um arquivo TXT."""
    logger.info("Exportando feedback para TXT...")
    feedback_path = f"{state['title'].replace(' ', '_')}_feedback.txt"
    with open(feedback_path, "w", encoding="utf-8") as f:
        f.write(state["feedback"])
    updates = {
        "feedback_path": feedback_path,
        "status": "feedback_exported"
    }
    logger.info(f"Feedback exportado com sucesso para: {feedback_path}")
    return updates

# def export_book(state: BookState) -> Dict[str, Any]:
#     """Exporta o livro apenas para DOCX."""
#     logger.info("Exportando livro para DOCX...")
#     doc = Document()
#     doc.add_heading(state["title"], 0)
#     doc.add_paragraph(f"Tema: {state['theme']}")
#     doc.add_paragraph(f"Gênero: {state['genre']}")
#     doc.add_paragraph(f"Público-alvo: {state['target_audience']}")
    
#     for chapter_num, chapter_data in sorted(state["chapters"].items()):
#         doc.add_heading(f"Capítulo {chapter_num}: {chapter_data['title']}", 1)
#         doc.add_paragraph(chapter_data["content"])
    
#     doc_path = f"{state['title'].replace(' ', '_')}.docx"
#     doc.save(doc_path)
#     updates = {
#         "export_path": doc_path,
#         "status": "exported"
#     }
#     logger.info(f"Livro exportado com sucesso para: {doc_path}")
#     return updates


def export_book(state: BookState) -> Dict[str, Any]:
    """Exporta o livro para DOCX com formatação completa e suporte total a Markdown."""
    logger.info("Exportando livro para DOCX com formatação completa...")
    doc = Document()

    # Configuração de estilos
    styles = doc.styles
    for style in styles:
        if hasattr(style, 'font') and style.font:
            style.font.name = 'Arial'

    title_style = styles['Title']
    title_style.font.name = 'Arial'
    title_style.font.size = Pt(24)
    title_style.font.bold = True
    title_style.font.color.rgb = RGBColor(0, 0, 0)

    heading_styles = {
        1: styles['Heading 1'], 2: styles['Heading 2'], 3: styles['Heading 3'],
        4: styles['Heading 4'], 5: styles['Heading 5'], 6: styles['Heading 6']
    }
    for level, style in heading_styles.items():
        style.font.name = 'Arial'
        style.font.size = Pt(16 - level * 2)  # Diminui o tamanho conforme o nível
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 51, 102)

    normal_style = styles['Normal']
    normal_style.font.name = 'Arial'
    normal_style.font.size = Pt(12)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(10)

    # Configurar margens e rodapé
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

        footer = section.footer
        footer_paragraph = footer.paragraphs[0]
        footer_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        footer_run = footer_paragraph.add_run()
        footer_run.font.name = 'Arial'
        footer_run.font.size = Pt(10)

        def add_field(run, field_code):
            field = OxmlElement('w:fldChar')
            field.set(qn('w:fldCharType'), 'begin')
            run._element.append(field)
            instr = OxmlElement('w:instrText')
            instr.set(qn('xml:space'), 'preserve')
            instr.text = field_code
            run._element.append(instr)
            sep = OxmlElement('w:fldChar')
            sep.set(qn('w:fldCharType'), 'separate')
            run._element.append(sep)
            end = OxmlElement('w:fldChar')
            end.set(qn('w:fldCharType'), 'end')
            run._element.append(end)

        footer_run.add_text('Página ')
        add_field(footer_run, 'PAGE')
        footer_run.add_text(' de ')
        add_field(footer_run, 'NUMPAGES')

    # Adicionar título
    title_paragraph = doc.add_paragraph(state["title"], style='Title')
    title_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    # Informações iniciais
    info_paragraph = doc.add_paragraph()
    info_paragraph.add_run("Tema: ").bold = True
    info_paragraph.add_run(state["theme"])
    info_paragraph.add_run("\nÁrea Tecnológica: ").bold = True
    info_paragraph.add_run(state["area_tecnologica"])
    info_paragraph.add_run("\nPúblico-alvo: ").bold = True
    info_paragraph.add_run(state["target_audience"])
    doc.add_page_break()

    # Adicionar sumário
    doc.add_heading("Sumário", level=1)
    toc_paragraph = doc.add_paragraph()
    toc_run = toc_paragraph.add_run()
    toc_field = OxmlElement('w:fldChar')
    toc_field.set(qn('w:fldCharType'), 'begin')
    toc_run._element.append(toc_field)
    toc_instr = OxmlElement('w:instrText')
    toc_instr.set(qn('xml:space'), 'preserve')
    toc_instr.text = 'TOC \\o "1-3" \\h \\z \\u'  # Níveis 1 a 3
    toc_run._element.append(toc_instr)
    toc_sep = OxmlElement('w:fldChar')
    toc_sep.set(qn('w:fldCharType'), 'separate')
    toc_run._element.append(toc_sep)
    toc_end = OxmlElement('w:fldChar')
    toc_end.set(qn('w:fldCharType'), 'end')
    toc_run._element.append(toc_end)
    doc.add_paragraph("Clique com o botão direito e escolha 'Atualizar campo' para gerar o sumário.", style='Caption')
    doc.add_page_break()

    # Parser Markdown completo
    def process_markdown(text, doc):
        lines = text.split('\n')
        in_code_block = False
        code_lines = []
        in_table = False
        table_rows = []

        for i, line in enumerate(lines):
            line = line.rstrip()
            if not line and not in_code_block and not in_table:
                continue

            # Blocos de código
            if line.strip() == '```':
                if not in_code_block:
                    in_code_block = True
                    code_lines = []
                else:
                    in_code_block = False
                    code_paragraph = doc.add_paragraph()
                    code_paragraph.paragraph_format.left_indent = Inches(0.5)
                    code_paragraph.paragraph_format.right_indent = Inches(0.5)
                    run = code_paragraph.add_run('\n'.join(code_lines))
                    run.font.name = 'Courier New'
                    run.font.size = Pt(10)
                    run.font.color.rgb = RGBColor(50, 50, 50)
                continue
            if in_code_block:
                code_lines.append(line)
                continue

            # Títulos
            if line.startswith('#'):
                level = min(line.count('#', 0, 6), 6)
                title_text = line.lstrip('#').strip()
                doc.add_heading(title_text, level=min(level, 6))
                continue

            # Linha horizontal
            if re.match(r'^\s*[-*_]{3,}\s*$', line):
                doc.add_paragraph().add_run().add_break(docx.enum.text.WD_BREAK.LINE)
                continue

            # Citações
            if line.startswith('>'):
                quote_paragraph = doc.add_paragraph()
                quote_paragraph.paragraph_format.left_indent = Inches(0.5)
                apply_inline_formatting(line.lstrip('>').strip(), quote_paragraph)
                continue

            # Listas ordenadas e não ordenadas
            if re.match(r'^\s*(\d+\.|-|\*|\+)\s+', line):
                indent_level = len(re.match(r'^\s*', line).group()) // 2
                list_item = re.sub(r'^\s*(\d+\.|-|\*|\+)\s+', '', line).strip()
                style = 'List Number' if re.match(r'^\s*\d+\.\s+', line) else 'List Bullet'
                paragraph = doc.add_paragraph(style=style)
                paragraph.paragraph_format.left_indent = Inches(0.25 * (indent_level + 1))
                apply_inline_formatting(list_item, paragraph)
                continue

            # Tabelas
            if line.startswith('|') and '|' in line[1:]:
                if not in_table:
                    in_table = True
                    table_rows = []
                row = [cell.strip() for cell in line.split('|')[1:-1]]
                if row and not re.match(r'^\s*-+\s*$', row[0]):  # Ignora linha de separação
                    table_rows.append(row)
                elif table_rows:  # Fim da tabela após separador
                    in_table = False
                    # ADICIONE ESTA VERIFICAÇÃO
                    if len(table_rows) > 0 and len(table_rows[0]) > 0:
                        table = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
                        table.style = 'Table Grid'
                        for r_idx, row_data in enumerate(table_rows):
                            # Garante que todas as linhas tenham o mesmo número de colunas
                            if len(row_data) == len(table_rows[0]):
                                for c_idx, cell_text in enumerate(row_data):
                                    try:
                                        cell = table.rows[r_idx].cells[c_idx]
                                        apply_inline_formatting(cell_text, cell.paragraphs[0])
                                    except IndexError:
                                        logger.warning(f"Ignorando célula fora do alcance na tabela. Linha {r_idx}, Coluna {c_idx}")
                    else:
                        logger.warning("Ignorando tentativa de criar uma tabela vazia ou malformada.")
                    table_rows = [] # Limpa para a próxima tabela
                continue

            # Parágrafo simples
            paragraph = doc.add_paragraph()
            apply_inline_formatting(line, paragraph)

    # Função para aplicar formatação inline
    def apply_inline_formatting(text, paragraph):
        # Regex para capturar negrito, itálico, tachado e links
        pattern = r'(\*\*\*.*?\*\*\*|\*\*.*?\*\*|\*.*?\*|~~.*?~~|$$ .*? $$$$ .*? $$|_.*?_)'
        parts = re.split(pattern, text)
        
        for part in parts:
            if not part:
                continue
            run = paragraph.add_run()
            if part.startswith('***') and part.endswith('***'):
                run.text = part[3:-3]
                run.bold = True
                run.italic = True
            elif part.startswith('**') and part.endswith('**'):
                run.text = part[2:-2]
                run.bold = True
            elif part.startswith('*') and part.endswith('*'):
                run.text = part[1:-1]
                run.italic = True
            elif part.startswith('_') and part.endswith('_'):
                run.text = part[1:-1]
                run.italic = True
            elif part.startswith('~~') and part.endswith('~~'):
                run.text = part[2:-2]
                run.font.strike = True
            elif part.startswith('[') and ']' in part and '(' in part and part.endswith(')'):
                link_text = part[1:part.index(']')]
                link_url = part[part.index('(')+1:-1]
                run.text = link_text
                run.font.underline = True
                run.font.color.rgb = RGBColor(0, 0, 255)
                # python-docx não suporta hiperlinks diretamente; URL é apenas visual
            else:
                run.text = part

    # Adicionar capítulos
    for chapter_num, chapter_data in sorted(state["chapters"].items()):
        doc.add_heading(f"Capítulo {chapter_num}: {chapter_data['title']}", level=1)
        process_markdown(chapter_data["content"], doc)
        if chapter_num < len(state["chapters"]):
            doc.add_page_break()

    if state.get("feedback"):
        doc.add_page_break()
        doc.add_heading("Feedback da Revisão", level=1)
        process_markdown(state["feedback"], doc)

    doc_path = f"{state['title'].replace(' ', '_')}.docx"
    doc.save(doc_path)

    updates = {
        "export_path": doc_path,
        "status": "exported"
    }
    logger.info(f"Livro exportado com sucesso para: {doc_path}")
    return updates


def router(state: BookState) -> str:
    """Decide o próximo estado."""
    status_map = {
        "start": "get_book_info",
        "book_info_collected": "create_outline",
        "outline_created": "write_chapter",
        "chapter_written": "write_chapter",
        "all_chapters_written": "review_and_edit",
        "reviewed": "export_feedback",
        "feedback_exported": "export_book",
        "exported": END
    }
    next_state = status_map.get(state["status"], END)
    logger.debug(f"Transição de estado: {state['status']} -> {next_state}")
    return next_state

def create_book_agent(model, st_session=None):
    """Cria o agente de geração de livros."""
    logger.info("Criando agente de geração de livros...")
    workflow = StateGraph(BookState)
    
    workflow.add_node("get_book_info", lambda state: get_book_info(state, model))
    workflow.add_node("create_outline", lambda state: create_outline(state, model))
    workflow.add_node("write_chapter", lambda state: write_chapter(state, model, st_session))
    workflow.add_node("review_and_edit", lambda state: review_and_edit(state, model))
    workflow.add_node("export_feedback", export_feedback)
    workflow.add_node("export_book", export_book)
    
    workflow.set_entry_point("get_book_info")
    
    workflow.add_conditional_edges("get_book_info", router)
    workflow.add_conditional_edges("create_outline", router)
    workflow.add_conditional_edges("write_chapter", router)
    workflow.add_conditional_edges("review_and_edit", router)
    workflow.add_conditional_edges("export_feedback", router)
    workflow.add_conditional_edges("export_book", router)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

def generate_with_retry(model, prompt, retries=3, delay=5):
    for i in range(retries):
        try:
            response = model.generate_content(prompt)
            return response
        except Exception as e:
            logger.warning(f"Erro na chamada da API (tentativa {i+1}/{retries}): {e}")
            time.sleep(delay)
    logger.error("Falha ao gerar conteúdo após múltiplas tentativas.")
    return None # Ou lançar uma exceção

def agent_book_generator(area_tecnologica: str = "", custom_audience: str = "", custom_theme: str = "", custom_num_chapters: int = 5):
    """Executa o agente de geração de livros e emite atualizações de progresso."""
    logger.info("Iniciando processo de geração de livro...")
    try:
        # model = init_vertex_ai()
        model = init_gemin_api()
        book_agent = create_book_agent(model) 
        
        initial_state = BookState(status="start")
        # ... (código de preenchimento do initial_state) ...
        if custom_theme: initial_state["theme"] = custom_theme
        if area_tecnologica: initial_state["area_tecnologica"] = area_tecnologica
        if custom_audience: initial_state["target_audience"] = custom_audience
        initial_state["num_chapters"] = custom_num_chapters

        config = {"configurable": {"thread_id": "1"}, "recursion_limit": 1000}
        
        for output in book_agent.stream(initial_state, config=config):
            node_name = list(output.keys())[0] if output else "unknown"
            node_output = output.get(node_name, {})
            stage = node_output.get("status", "desconhecido")
            
            # --- INÍCIO DA LÓGICA DA BARRA DE PROGRESSO ---
            # Verifica se estamos na etapa de escrita e se temos os dados necessários
            if node_name == "write_chapter" and "chapters" in node_output:
                # O 'current_chapter' no estado já foi incrementado, então subtraímos 1 para pegar o que acabou de ser escrito.
                current = node_output.get("current_chapter", 1) - 1
                if current > 0: # Garante que não tentemos acessar o capítulo 0
                    total = len(node_output["chapters"])
                    title = node_output["chapters"][current]["title"]
                    
                    # Emite um dicionário especial para o progresso
                    yield {
                        "type": "progress",
                        "text": f"Capítulo {current}/{total}: {title}",
                        "value": int((current / total) * 100)
                    }
            # --- FIM DA LÓGICA DA BARRA DE PROGRESSO ---

            # O restante do código de 'yield' para o conteúdo de texto continua o mesmo
            yield f"**Etapa Concluída:** {stage}"

            if stage == "book_info_collected":
                yield f"**Título gerado:** {node_output.get('title', 'N/A')}"
            
            elif stage == "outline_created":
                yield f"Sumário criado com {len(node_output.get('outline', []))} capítulos."

            # Modificação para exibir o último capítulo
            elif stage == "chapter_written" or (node_name == "write_chapter" and stage == "all_chapters_written"):
                # O 'current_chapter' no estado já foi incrementado, então subtraímos 1 para pegar o que acabou de ser escrito.
                current_chap_num = node_output.get('current_chapter', 0) - 1
                if current_chap_num > 0:
                    chapters_data = node_output.get("chapters", {})
                    chapter_info = chapters_data.get(current_chap_num)

                    # Garante que a informação do capítulo e o conteúdo existam
                    if chapter_info and chapter_info.get("content"):
                        yield f"### Capítulo {current_chap_num}: {chapter_info.get('title', '')}"
                        yield chapter_info["content"]
        
        final_state = book_agent.checkpointer.get(config)
        logger.info("Processo de geração de livro concluído!")
        
        yield {"final_state": final_state}

    except Exception as e:
        logger.error(f"Erro durante a geração do livro: {e}")
        yield {"final_state": {"status": "error", "message": str(e)}}