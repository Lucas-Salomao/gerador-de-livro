import streamlit as st
from app import main  # Importa a função main do seu script

st.set_page_config(
        page_title="Gerador de Livros Técnicos",  # Define o título da página
        page_icon="📚",  # Define o ícone da página
        menu_items={'About': "SENAI São Paulo - Gerência de Educação\n\nSupervisão de Tecnologias Educacionais\n\nDesenvolvido por Lucas Salomão"},
        layout="wide"
    )

# Título da aplicação
st.title("📚 Gerador de Livros Técnicos")

# Sidebar para inserir os argumentos
st.sidebar.header("Configurações do Livro")
theme = st.sidebar.text_input("Tema do Livro", placeholder="Tema do Livro")
genre = st.sidebar.text_input("Gênero do Livro", placeholder="Gênero do Livro")
audience = st.sidebar.text_input("Público-Alvo", placeholder="Público-Alvo")

# Botão para iniciar a geração do livro
if st.sidebar.button("Gerar Livro"):
    # Inicia a geração do livro
    st.write("Iniciando a geração do livro...")
    
    # Executa a função main com os argumentos fornecidos
    result = main(theme, genre, audience, st_session=st)
    
    # Obtém o caminho do livro exportado
    book_path=result["channel_values"]["export_path"]
    
    # Verifica se o livro foi exportado com sucesso
    if book_path:
        # Botão de download do DOCX
        st.write("### Download do Livro")
        with open(book_path, "rb") as file:
            btn = st.download_button(
                label="Baixar Livro (DOCX)",
                data=file,
                file_name=book_path,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        
        st.success("Livro gerado com sucesso! Clique no botão acima para baixar.")
    else:
        st.error("Ocorreu um erro durante a geração do livro. Verifique os logs para mais detalhes.")