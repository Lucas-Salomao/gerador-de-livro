import streamlit as st
from agent import agent_book_generator
import os

st.set_page_config(
    page_title="Gerador de Apostila",
    page_icon="📚",
    menu_items={'About': "SENAI São Paulo - Gerência de Educação\n\nSupervisão de Tecnologias Educacionais\n\nDesenvolvido por Lucas Salomão"},
    layout="wide"
)

# Inicializa o estado da sessão para guardar o log e o resultado final
if 'generation_log' not in st.session_state:
    st.session_state.generation_log = []
if 'final_result' not in st.session_state:
    st.session_state.final_result = None
if 'book_path' not in st.session_state:
    st.session_state.book_path = None
    
def sidebar():
    """Cria a barra lateral e retorna os valores dos widgets."""
    with st.sidebar:
        st.logo("https://www.imagemhost.com.br/images/2024/11/22/Logo-novo-SENAI_-sem-slogan_755X325.md.png", size="large")
        st.title("Bem vindo, "+st.user.given_name+"!")
        st.header("⚙️ Configurações")

        # generos_agrupados = {
        #     "Ficção": ["Aventura", "Contos", "Distopia", "Fantasia", "Ficção Científica", "Ficção Histórica", "Ficção Policial", "Humor", "Infantojuvenil", "Literatura Clássica", "Mistério", "Romance", "Suspense", "Terror"],
        #     "Não Ficção Geral": ["Arte e Fotografia", "Autoajuda", "Autobiografia e Memórias", "Biografia", "Ciências Sociais", "Culinária", "Ensaios", "Esportes", "Filosofia", "História", "Política", "Psicologia", "Religião e Espiritualidade", "Viagem"],
        #     "Técnico": ["Administração e Negócios", "Arquitetura e Design", "Ciência de Dados", "Ciências Exatas (Física, Matemática, Química)", "Ciências Biológicas e da Saúde", "Computação e Programação", "Direito", "Economia e Finanças", "Educação e Pedagogia", "Engenharia", "Marketing e Vendas", "Medicina", "Tecnologia da Informação (TI)"]
        # }
        
        # categorias_principais = ["Selecione uma categoria"] + list(generos_agrupados.keys())
        # categoria_principal = st.selectbox("1. Categoria Principal", options=categorias_principais, index=0)
        
        # genero_especifico = None
        # if categoria_principal != "Selecione uma categoria":
        #     opcoes_genero = ["Selecione um gênero"] + sorted(generos_agrupados[categoria_principal])
        #     genero_especifico = st.selectbox("2. Gênero Específico", options=opcoes_genero, index=0)
        
        areas_tecnologicas = [
            "COMERCIAL",
            "COMUNICAÇÃO MIDIÁTICA",
            "CONSTRUÇÃO DE OBRAS",
            "DESENVOLVIMENTO DE SISTEMAS",
            "DESIGN",
            "ELETRÔNICA E AUTOMAÇÃO",
            "GERENCIAL",
            "GESTÃO E PROMOÇÃO DA SAÚDE E BEM-ESTAR",
            "GESTÃO E SEGURANÇA",
            "INFRAESTRUTURA DE INFORMAÇÃO E COMUNICAÇÃO",
            "MANUFATURA",
            "MANUTENÇÃO E OPERAÇÃO",
            "MATERIAIS",
            "METALMECÂNICA",
            "MINERAÇÃO E EXTRAÇÃO",
            "OPERAÇÕES DE TRANSPORTE",
            "OPERAÇÕES FINANCEIRAS",
            "PRODUÇÃO ALIMENTÍCIA",
            "PROTEÇÃO E REABILITAÇÃO DE ECOSSISTEMAS",
            "QUÍMICA",
            "SEGURANÇA",
            "SISTEMAS DE ENERGIA",
            "TÊXTIL E VESTUÁRIO"
        ]
        area_selecionada = st.selectbox(
            label="Escolha uma área técnologica",
            options=areas_tecnologicas
        )
        

        audience = st.text_input("Público-Alvo", placeholder="Ex: Estudantes de Curso Técnico")
        theme = st.text_area("Tema da apostila", placeholder="Ex: Uma introdução à Inteligência Artificial", height=150)
        chapters = st.number_input("Número de Capítulos", min_value=5, value=5, step=1, max_value=100)

        # O botão agora só dispara a ação, a lógica fica no frontend
        generate_button = st.button("🌟 Gerar Apostila")

        # Placeholders para a barra de progresso
        st.markdown("---")
        st.subheader("Progresso da geração")
        progress_label = st.empty()
        progress_bar = st.empty()
            
        st.markdown("---")

    return area_selecionada, audience, theme, chapters, generate_button, progress_label, progress_bar

def frontend():
    """Renderiza a interface principal e gerencia a lógica da aplicação."""
    st.title("📚 Gerador de Apostila")
    
    content_container = st.container()
    
    # Obtém os valores e widgets da sidebar
    area_selecionada, audience, theme, chapters, generate_button, progress_label, progress_bar = sidebar()

    with st.sidebar:
        if st.button("Limpar Tela", icon='🧹'):
            content_container.empty()
            st.session_state.generation_log = []
            st.session_state.final_result = None
            
        if st.button("Sair", icon='🚪'):
            st.logout()
    
    # --- Lógica de Exibição PERSISTENTE (sempre executa) ---
    # Este container garante que o conteúdo permaneça na tela em qualquer interação
    # (como clicar no botão de download).
    
    with content_container:
        for log_entry in st.session_state.generation_log:
            st.markdown(log_entry, unsafe_allow_html=True)

    # --- Lógica de Geração (executada apenas quando o botão é clicado) ---
    if generate_button:
        if not all([theme, audience, area_selecionada]):
            st.warning("Por favor, preencha todos os campos para iniciar.")
        else:
            # Limpa resultados antigos da memória
            st.session_state.generation_log = []
            st.session_state.final_result = None
            
            # Limpa o conteúdo VISUAL da tela para a nova geração
            content_container.empty() 
            
            progress_label.text("Iniciando...")
            progress_bar.progress(0)

            # Itera sobre o gerador do agente
            for message in agent_book_generator(area_tecnologica=area_selecionada, custom_audience=audience, custom_theme=theme, custom_num_chapters=chapters):
                if isinstance(message, dict):
                    if message.get("type") == "progress":
                        progress_label.text(message["text"])
                        progress_bar.progress(message["value"])
                    elif "final_state" in message:
                        st.session_state.final_result = message["final_state"]
                        progress_label.text("Geração Concluída!")
                        progress_bar.progress(100)
                        break
                else:
                    # 1. Salva a mensagem na memória para persistência
                    st.session_state.generation_log.append(str(message))
                    # 2. Escreve a mesma mensagem no container para exibição em tempo real
                    content_container.markdown(str(message), unsafe_allow_html=True)

            # Força um rerodamento final para garantir que o botão de download apareça
            st.rerun()

    # --- Lógica de Exibição do Resultado Final (botão de download) ---
    if st.session_state.final_result:
        st.markdown("---")
        
        result = st.session_state.final_result
        st.session_state.book_path=result["channel_values"]["export_path"]
        if result.get("status") != "error":
            book_path = st.session_state.book_path
            if book_path and os.path.exists(book_path):
                st.success("Sua apostila está pronta para download!")
                with open(book_path, "rb") as file:
                    st.download_button(
                        label="Baixar Livro (DOCX)",
                        data=file,
                        file_name=os.path.basename(book_path),
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            else:
                st.error(f"Arquivo não encontrado no caminho: {book_path}")
        else:
            st.error(f"Ocorreu um erro no agente: {result.get('message')}")

if __name__ == "__main__":
    if not st.user.is_logged_in:
        st.markdown(
            """
            <div style="text-align: center;">
                <img src="https://staticportaldaindustria.azureedge.net/static/img/logos/atualizado_2/senai.svg" alt="Logo SENAI" height="70">
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            "<h1 style='text-align: center;'>Gerador de Apostila</h1>",
            unsafe_allow_html=True
        )
            
        st.markdown(
            "<h2 style='text-align: center;'>👤 Login</h2>",
            unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns([5, 1.5, 5]) # Cria 3 colunas para empurrar o botão para o centro
        with col2:
            if st.button("Entrar com Meu Senai", use_container_width=True):
                st.login()
    else:
        frontend()