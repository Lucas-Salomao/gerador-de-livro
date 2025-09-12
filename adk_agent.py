import os
import json
import logging
import re
from typing import Dict, List, Any, Callable, Optional
from pydantic import BaseModel, Field
import docx
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from adk.tools import tool
from adk.agents import LlmAgent
import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_json_parse(response_text: str, fallback: Any) -> Any:
    """Tries to decode JSON and returns a fallback on error."""
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
        logger.error(f"Error decoding JSON: {response_text[:100]}... Using fallback.")
        return fallback

class BookState(BaseModel):
    theme: str
    title: Optional[str] = None
    genre: str
    main_category: str
    target_audience: str
    num_chapters: int
    outline: Optional[List[Dict[str, Any]]] = None
    chapters: Dict[int, Dict[str, str]] = Field(default_factory=dict)
    current_chapter: int = 1
    feedback: Optional[str] = None
    export_path: Optional[str] = None
    feedback_path: Optional[str] = None

class BookGeneratorAgent:
    def __init__(self, search_tool: Callable[[str], str]):
        self.search_tool = search_tool
        self._setup_llm()
        self.llm_agent.add_tools([self.web_search])


    def _setup_llm(self):
        """Initializes the Gemini API and the LLM agent."""
        logger.info("Initializing Gemini API...")
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            # Note: The model name might need to be adjusted based on availability.
            model = genai.GenerativeModel('gemini-1.5-flash')
            self.llm_agent = LlmAgent(llm=model)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            raise

    @tool
    def web_search(self, query: str) -> str:
        """
        Performs a web search using the provided search tool and returns the results.
        This tool is intended to be used to gather information for writing book chapters.
        """
        logger.info(f"Performing web search for: {query}")
        return self.search_tool(query=query)

    def get_book_info(self, state: BookState) -> BookState:
        """Gets basic info and generates the title."""
        logger.info("Generating book title...")
        prompt = f"""
        You are an expert in technical writing. Based on the following theme, genre, and target audience, suggest a formal and technical title that reflects an analytical and informative approach:
        Theme: {state.theme}
        Main Category: {state.main_category}
        Specific Genre: {state.genre}
        Target Audience: {state.target_audience}
        Respond ONLY in JSON format with the key "title", without additional text. Example: {{"title": "Fundamentals of Space Exploration"}}. Do not include a code block, i.e. ```json```
        The title must have a maximum of 80 characters. Invalid characters for the title: , \\ / : * ? " < > |
        """
        response = self.llm_agent.run(prompt=prompt)
        info = safe_json_parse(response.content, {"title": f"Book on {state.theme}"})
        state.title = info.get("title", f"Book on {state.theme}")
        logger.info(f"Generated title: {state.title}")
        return state

    def create_outline(self, state: BookState) -> BookState:
        """Creates the book's outline."""
        logger.info("Creating book outline...")
        prompt = f"""
        You are a technical expert drafting a technical book for studying a specific theme.
        Based on the following information, create a detailed outline with a focus on technical and practical aspects:

        Theme: {state.theme}
        Suggested Title: {state.title}
        Main Category: {state.main_category}
        Specific Genre: {state.genre}
        Target Audience: {state.target_audience}

        Each chapter must have a sequential integer numbering (e.g., 1, 2, 3, 4, 5, etc.)

        The outline must contain exactly {state.num_chapters} chapters, each addressing a technical or practical aspect of the theme, with objective titles and descriptions that detail the analytical content to be explored.
        Respond ONLY in JSON format with a list of objects containing "chapter_number", "chapter_title", and "chapter_description".
        Example: [{{"chapter_number": 1, "chapter_title": "Principles of Space Propulsion", "chapter_description": "Analysis of propulsion systems used in space missions"}}]
        Do not include a code block, i.e. ```json```
        """
        response = self.llm_agent.run(prompt=prompt)
        outline_data = safe_json_parse(response.content, [])

        if not outline_data or len(outline_data) < state.num_chapters:
            logger.warning("Outline has fewer chapters than requested. Generating a default outline.")
            outline_data = [
                {"chapter_number": i, "chapter_title": f"Chapter {i}", "chapter_description": f"Exploration of {state.theme}"}
                for i in range(1, state.num_chapters + 1)
            ]

        state.outline = outline_data
        state.chapters = {
            item["chapter_number"]: {
                "title": item["chapter_title"],
                "description": item["chapter_description"],
                "content": ""
            }
            for item in outline_data
        }
        logger.info(f"Outline created with {len(outline_data)} chapters.")
        return state

    def write_chapter(self, state: BookState) -> BookState:
        """Writes a single chapter, using web search to gather information."""
        chapter_info = state.chapters[state.current_chapter]
        logger.info(f"Writing Chapter {state.current_chapter}: {chapter_info['title']}...")

        # Use the web_search tool
        search_query = f"{chapter_info['title']} - {chapter_info['description']}"
        search_results = self.web_search(query=search_query)

        prev_content = ""
        if state.current_chapter > 1:
            prev_chapter = state.chapters[state.current_chapter - 1]
            prev_content = f"Summary of the previous chapter ({state.current_chapter - 1}: {prev_chapter['title']}):\n{prev_chapter['content'][:500]}... (summarized)"

        prompt = f"""
        You are a technical expert writing a book titled "{state.title}" on the theme "{state.theme}".
        The book's main category is "{state.main_category}" and the specific genre is "{state.genre}", aimed at "{state.target_audience}".

        Write Chapter {state.current_chapter}: "{chapter_info['title']}".
        Chapter description: {chapter_info['description']}

        {prev_content}

        Here are some web search results to inform your writing:
        <search_results>
        {search_results}
        </search_results>

        Write a technical and analytical text, with formal and objective language. Include detailed technical information, contextualized examples (real or hypothetical), relevant data, and clear explanations. Avoid narrative dialogues or excessive literary descriptions. Structure the content with clear sections (e.g., introduction, analysis, examples, conclusion). The chapter should be at least 3000 words. Be as detailed as possible and address the chapter's theme with depth and plenty of examples.
        Structure the chapter with titles and subtitles to facilitate reading and understanding. Follow the chapter numbering and structure the subtitles based on the chapter number.
        """
        response = self.llm_agent.run(prompt=prompt)
        state.chapters[state.current_chapter]["content"] = response.content
        logger.info(f"Chapter {state.current_chapter} completed successfully.")
        return state

    def review_and_edit(self, state: BookState) -> BookState:
        """Reviews and edits the complete book."""
        logger.info("Reviewing and editing the book...")
        book_summary = f"Theme: {state.theme}\nTitle: {state.title}\nGenre: {state.genre}\nTarget Audience: {state.target_audience}\n\nOutline:\n"
        for chapter_num, chapter_data in sorted(state.chapters.items()):
            book_summary += f"\nChapter {chapter_num}: {chapter_data['title']} - {chapter_data['description'][:100]}..."

        prompt = f"""
        You are an editor reviewing the book:
        {book_summary}

        Provide feedback on structure, narrative flow, consistency with the theme "{state.theme}", and appeal to the target audience. Suggest improvements. Technically review the book and check for any inconsistencies.
        """
        response = self.llm_agent.run(prompt=prompt)
        state.feedback = response.content
        logger.info("Review completed. Feedback generated.")
        return state

    def export_book(self, state: BookState) -> BookState:
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
        title_paragraph = doc.add_paragraph(state.title, style='Title')
        title_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

        # Informações iniciais
        info_paragraph = doc.add_paragraph()
        info_paragraph.add_run("Tema: ").bold = True
        info_paragraph.add_run(state.theme)
        info_paragraph.add_run("\nGênero: ").bold = True
        info_paragraph.add_run(state.genre)
        info_paragraph.add_run("\nPúblico-alvo: ").bold = True
        info_paragraph.add_run(state.target_audience)
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
        for chapter_num, chapter_data in sorted(state.chapters.items()):
            doc.add_heading(f"Capítulo {chapter_num}: {chapter_data['title']}", level=1)
            process_markdown(chapter_data["content"], doc)
            if chapter_num < len(state.chapters):
                doc.add_page_break()

        if state.feedback:
            doc.add_page_break()
            doc.add_heading("Feedback da Revisão", level=1)
            process_markdown(state.feedback, doc)

        doc_path = f"{state.title.replace(' ', '_')}.docx"
        doc.save(doc_path)

        state.export_path = doc_path
        logger.info(f"Livro exportado com sucesso para: {doc_path}")
        return state

def adk_book_generator(search_tool: Callable[[str], str], **kwargs):
    """Executes the ADK book generation agent and yields progress updates."""
    logger.info("Starting ADK book generation process...")
    agent = BookGeneratorAgent(search_tool=search_tool)

    try:
        state = BookState(**kwargs)

        # 1. Get book info
        state = agent.get_book_info(state)
        yield {"type": "info", "title": state.title}

        # 2. Create outline
        state = agent.create_outline(state)
        yield {"type": "outline", "outline": state.outline}

        # 3. Write chapters
        total_chapters = len(state.chapters)
        for i in range(1, total_chapters + 1):
            state.current_chapter = i
            state = agent.write_chapter(state)
            yield {
                "type": "progress",
                "text": f"Chapter {i}/{total_chapters}: {state.chapters[i]['title']}",
                "value": int((i / total_chapters) * 100)
            }
            yield {
                "type": "chapter",
                "chapter_number": i,
                "title": state.chapters[i]['title'],
                "content": state.chapters[i]['content']
            }

        # 4. Review and edit
        state = agent.review_and_edit(state)
        yield {"type": "review", "feedback": state.feedback}

        # 5. Export book
        state = agent.export_book(state)
        yield {"type": "export", "path": state.export_path}

        logger.info("ADK book generation process completed!")
        yield {"final_state": state.dict()}

    except Exception as e:
        logger.error(f"Error during ADK book generation: {e}")
        yield {"final_state": {"status": "error", "message": str(e)}}
