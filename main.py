import streamlit as st
from agent import agent_book_generator  # Importa a função main do seu script
import os

st.set_page_config(
        page_title="Gerador de Livros e Apostilas",  # Define o título da página
        page_icon="📚",  # Define o ícone da página
        menu_items={'About': "SENAI São Paulo - Gerência de Educação\n\nSupervisão de Tecnologias Educacionais\n\nDesenvolvido por Lucas Salomão"},
        layout="wide"
    )

# Isso garante que as variáveis existem desde o início, evitando erros.
if 'book_generated' not in st.session_state:
    st.session_state.book_generated = False
if 'result' not in st.session_state:
    st.session_state.result = None
if 'book_path' not in st.session_state:
    st.session_state.book_path = None
    
def sidebar():

    st.logo("https://www.fiema.org.br/uploads/area/19602/thumb_rBHr9q6LNEE4hOdcW7tzTCZadVauW7me.png", size="large")
    # Sidebar para inserir os argumentos
    st.sidebar.header("⚙️ Configurações")

    # Passo 1: Organizar todos os gêneros em um dicionário
    # As chaves são as categorias principais e os valores são as listas de gêneros específicos.
    generos_agrupados = {
        "Ficção": [
            "Aventura", "Contos", "Distopia", "Fantasia", "Ficção Científica",
            "Ficção Histórica", "Ficção Policial", "Humor", "Infantojuvenil",
            "Literatura Clássica", "Mistério", "Romance", "Suspense", "Terror"
        ],
        "Não Ficção Geral": [
            "Arte e Fotografia", "Autoajuda", "Autobiografia e Memórias", "Biografia",
            "Ciências Sociais", "Culinária", "Ensaios", "Esportes", "Filosofia",
            "História", "Política", "Psicologia", "Religião e Espiritualidade", "Viagem"
        ],
        "Técnico": [
            "Administração e Negócios", "Arquitetura e Design", "Ciência de Dados",
            "Ciências Exatas (Física, Matemática, Química)", "Ciências Biológicas e da Saúde",
            "Computação e Programação", "Direito", "Economia e Finanças",
            "Educação e Pedagogia", "Engenharia", "Marketing e Vendas", "Medicina",
            "Tecnologia da Informação (TI)"
        ]
    }

    # --- Lógica dos Seletores em Cascata ---

    # Passo 2: Criar o seletor da categoria principal (Nível 1)
    categorias_principais = ["Selecione uma categoria"] + list(generos_agrupados.keys())

    categoria_principal = st.sidebar.selectbox(
        "1. Categoria Principal",
        options=categorias_principais,
        index=0
    )

    # Inicializar variáveis para armazenar as seleções
    genero_especifico = None

    # Passo 3: Criar o seletor de gênero específico (Nível 2)
    # Este seletor só aparecerá se uma categoria principal for escolhida.
    if categoria_principal != "Selecione uma categoria":
        # Pega a lista de gêneros correspondente à categoria escolhida
        opcoes_genero = ["Selecione um gênero"] + sorted(generos_agrupados[categoria_principal])

        genero_especifico = st.sidebar.selectbox(
            "2. Gênero Específico",
            options=opcoes_genero,
            index=0
        )

    if categoria_principal != "Selecione uma categoria":
        if genero_especifico and genero_especifico != "Selecione um gênero":
            st.success(f"Gênero Específico selecionado: **{genero_especifico}**")
        else:
            st.info("Agora, por favor, selecione um gênero específico na barra lateral.")
    else:
        st.warning("Comece selecionando uma categoria principal na barra lateral.")
    
    
    
    audience = st.sidebar.text_input("Público-Alvo", placeholder="Público-Alvo")
    theme = st.sidebar.text_area("Tema do Livro", placeholder="Tema do Livro",height=150)
    chapters = st.sidebar.number_input("Número de Capítulos", min_value=5, value=5,step=1,max_value=50)

    # Botão para iniciar a geração do livro
    if st.sidebar.button("🌟 Gerar Livro"):
        # Inicia a geração do livro
        st.write("Iniciando a geração do livro...")
        
        # Executa a função main com os argumentos fornecidos
        result = agent_book_generator(genero_especifico, categoria_principal, audience, theme, chapters, st_session=st)
        
        st.session_state.book_generated = True
        st.session_state.result = result
        
        # Obtém o caminho do livro exportado
        st.session_state.book_path=result["channel_values"]["export_path"]

def frontend():
    # Título da aplicação
    st.title("📚 Gerador de Livros e Apostilas")
    
    sidebar()

    if st.session_state.book_generated:
        # st.header("Resultado da Geração")
        
        result = st.session_state.result
        
        # Tratamento seguro do resultado
        if result and result.get("status") != "error":
            book_path = st.session_state.book_path
            if book_path and os.path.exists(book_path):
                st.success("Livro gerado com sucesso!")
                
                with open(book_path, "rb") as file:
                    st.download_button(
                        label="Baixar Livro (DOCX)",
                        data=file,
                        file_name=os.path.basename(book_path),
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            else:
                st.error("Geração concluída, mas não foi possível encontrar o arquivo do livro.")
        elif result:
            st.error(f"Ocorreu um erro durante a geração do livro: {result.get('message', 'Erro desconhecido')}")
            
if __name__ == "__main__":
    frontend()
