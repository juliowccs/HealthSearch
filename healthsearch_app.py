import re
import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# ==============================================================================
# CONFIGURAÇÃO GERAL DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="HealthSearch | Motor de Busca Híbrido",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# BASE DE DADOS DO CORPUS MÉDICO (HARDCODED)
# ==============================================================================
CORPUS = [
    {
        "id": "Doc 1",
        "titulo": "Protocolo Emergência ECG",
        "conteudo": "Pacientes com dor precordial aguda e suspeita de síndrome coronariana devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos."
    },
    {
        "id": "Doc 2",
        "titulo": "Guia de Farmacologia Cardíaca",
        "conteudo": "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz a mortalidade no infarto agudo do miocárdio."
    },
    {
        "id": "Doc 3",
        "titulo": "Diretriz de Hipertensão Arterial",
        "conteudo": "A crise hipertensiva severa requer administração de anti-hipertensivos venosos e monitoramento contínuo da pressão arterial na UTI."
    },
    {
        "id": "Doc 4",
        "titulo": "Manual de AVC Isquêmico",
        "conteudo": "O acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos venosos em até quatro horas e meia do início dos sintomas."
    },
    {
        "id": "Doc 5",
        "titulo": "Protocolo de Reanimação RCR",
        "conteudo": "Parada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta qualidade e desfibrilação precoce no código azul."
    },
    {
        "id": "Doc 6",
        "titulo": "Procedimentos de UTI Geral",
        "conteudo": "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a monitorização cardíaca contínua por telemetria."
    }
]

# Stopwords em português para filtragem léxica
STOPWORDS_PT = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "até", 
    "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "depois", "do", 
    "dos", "e", "ela", "elas", "ele", "eles", "em", "entre", "era", "eram", "essa", 
    "essas", "esse", "esses", "esta", "estas", "este", "estes", "eu", "foi", "fomos", 
    "foram", "isso", "isto", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", 
    "meus", "minha", "minhas", "na", "nas", "não", "no", "nos", "nossa", "nossas", 
    "nosso", "nossos", "num", "numa", "o", "os", "ou", "para", "pela", "pelas", "pelo", 
    "pelos", "por", "qual", "quando", "que", "quem", "se", "sem", "seu", "seus", "só", 
    "sua", "suas", "também", "te", "tem", "têm", "temos", "tenho", "teu", "teus", "tu", 
    "tua", "tuas", "um", "uma", "você", "vocês"
}

# ==============================================================================
# MODELOS & CACHING
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_bi_encoder():
    # Modelo multilíngue ideal para consultas em português
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource(show_spinner=False)
def load_cross_encoder():
    # Modelo Cross-Encoder para re-ranking profundo
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_data(show_spinner=False)
def get_corpus_embeddings(_model, texts):
    return _model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

# ==============================================================================
# FASE 1: PRÉ-PROCESSAMENTO LÉXICO
# ==============================================================================
def preprocess_text(text: str) -> list[str]:
    """
    Normaliza o texto para minúsculas, preserva identificadores compostos 
    alfanuméricos com hífen (ex: CÓD-ECG-12D) e remove stopwords.
    """
    text = text.lower()
    # Captura palavras e códigos técnicos com hífen
    tokens = re.findall(r"\b[a-zá-ú0-9]+(?:-[a-zá-ú0-9]+)*\b", text)
    return [t for t in tokens if t not in STOPWORDS_PT and len(t) > 1]

# ==============================================================================
# PIPELINE DE BUSCA E FUSÃO
# ==============================================================================
def run_pipeline(query: str, k1: float, b: float, alpha: float, k_rrf: int, apply_cross_encoder: bool):
    # Carregamento de modelos
    bi_encoder = load_bi_encoder()
    corpus_texts = [f"{d['titulo']}. {d['conteudo']}" for d in CORPUS]
    corpus_embeddings = get_corpus_embeddings(bi_encoder, tuple(corpus_texts))

    # 1. MOTOR LÉXICO (BM25)
    tokenized_corpus = [preprocess_text(doc) for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    tokenized_query = preprocess_text(query)
    bm25_scores = bm25.get_scores(tokenized_query) if tokenized_query else np.zeros(len(CORPUS))

    # 2. MOTOR SEMÂNTICO (EMBEDDINGS)
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    semantic_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()

    # Construção de Ranks Base
    df = pd.DataFrame({
        "id": [d["id"] for d in CORPUS],
        "titulo": [d["titulo"] for d in CORPUS],
        "conteudo": [d["conteudo"] for d in CORPUS],
        "score_bm25": bm25_scores,
        "score_semantico": semantic_scores
    })

    # Atribuição de Ranks ordinais (1 = melhor posição)
    # Empates resolvidos por ordem estável
    df["rank_bm25"] = df["score_bm25"].rank(ascending=False, method="min").astype(int)
    df["rank_semantico"] = df["score_semantico"].rank(ascending=False, method="min").astype(int)

    # 3. FUSÃO RECÍPROCA (RRF)
    # Fórmula: Score_RRF = α * [1 / (k_rrf + Rank_BM25)] + (1 - α) * [1 / (k_rrf + Rank_Semantico)]
    df["rrf_bm25_component"] = 1.0 / (k_rrf + df["rank_bm25"])
    df["rrf_sem_component"] = 1.0 / (k_rrf + df["rank_semantico"])
    df["score_rrf"] = (alpha * df["rrf_bm25_component"]) + ((1.0 - alpha) * df["rrf_sem_component"])

    df = df.sort_values(by="score_rrf", ascending=False).reset_index(drop=True)
    df["rank_rrf"] = df.index + 1

    # 4. BÔNUS: CROSS-ENCODER RE-RANKING (SOBRE O TOP-3)
    if apply_cross_encoder:
        cross_encoder = load_cross_encoder()
        top_3_indices = df.head(3).index
        pairs = [[query, f"{df.loc[i, 'titulo']}: {df.loc[i, 'conteudo']}"] for i in top_3_indices]
        cross_scores = cross_encoder.predict(pairs)
        
        df["score_cross_encoder"] = np.nan
        df.loc[top_3_indices, "score_cross_encoder"] = cross_scores
        
        # Cria dataframe ordenado pós re-ranking do top-3
        top_3_reranked = df.loc[top_3_indices].sort_values(by="score_cross_encoder", ascending=False).copy()
        top_3_reranked["rank_cross_encoder"] = range(1, 4)
        df.loc[top_3_indices, "rank_cross_encoder"] = top_3_reranked["rank_cross_encoder"]

    return df

# ==============================================================================
# BARRA LATERAL: CONTROLES E HIPERPARÂMETROS
# ==============================================================================
with st.sidebar:
    st.title("⚙️ Painel de Calibração")
    st.markdown("Ajuste fino dos motores léxico, semântico e do algoritmo de fusão.")

    st.subheader("1. Parâmetros BM25")
    k1 = st.slider(
        "Saturação de Frequência ($k_1$)",
        min_value=0.0,
        max_value=3.0,
        value=1.2,
        step=0.1,
        help="Controla a velocidade com que a frequência do termo atinge o limite de saturação."
    )
    b = st.slider(
        "Normalização por Tamanho ($b$)",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Controla o grau de penalização para documentos longos (0.0 = sem normalização, 1.0 = normalização total)."
    )

    st.divider()

    st.subheader("2. Parâmetros da Fusão RRF")
    alpha = st.slider(
        "Peso Léxico ($α$)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="1.0 = Busca puramente léxica | 0.0 = Busca puramente semântica."
    )
    k_rrf = st.number_input(
        "Constante de Suavização ($k_{RRF}$)",
        min_value=1,
        max_value=120,
        value=60,
        step=1,
        help="Atenua a influência de posições de topo discrepantes no cálculo recíproco."
    )

    st.divider()

    st.subheader("3. Módulo Bônus")
    apply_cross = st.checkbox(
        "Habilitar Re-Ranking Cross-Encoder",
        value=False,
        help="Aplica modelo de atenção cruzada (MiniLM-L6) sobre os 3 melhores documentos do RRF."
    )

    st.markdown("---")
    st.caption("**HealthSearch")

# ==============================================================================
# CORPO PRINCIPAL: CONSULTAS E VISUALIZAÇÃO
# ==============================================================================
st.title("🏥 HealthSearch: Motor de Busca Híbrido")
st.markdown(
    "Recuperação de diretrizes clínicas hospitalares combinando a precisão cirúrgica do "
    "**Okapi BM25** com a contextualização de **Embeddings Neurais** via **Reciprocal Rank Fusion (RRF)**."
)

# Seletor de exemplos rápidos para validação pedagógica
exemplos = [
    "Personalizada",
    "CÓD-ECG-12D",
    "infarto agudo e dor precordial",
    "trombolítico para acidente vascular cerebral",
    "pressão alta crítica na UTI"
]
escolha_exemplo = st.selectbox("💡 Consultas de teste recomendadas:", exemplos)

if escolha_exemplo != "Personalizada":
    query_input = st.text_input("Termo de Busca:", value=escolha_exemplo)
else:
    query_input = st.text_input("Termo de Busca:", value="infarto")

if query_input.strip():
    resultados = run_pipeline(
        query=query_input,
        k1=k1,
        b=b,
        alpha=alpha,
        k_rrf=k_rrf,
        apply_cross_encoder=apply_cross
    )

    # Organização das visualizações em abas obrigatórias
    tab_rrf, tab_lexico, tab_semantico, tab_comparativo, tab_bonus = st.tabs([
        "🔀 Híbrido RRF",
        "🔤 Léxico (BM25)",
        "🧠 Semântico (Vetorial)",
        "📊 Matriz Comparativa",
        "⚡ Bônus: Cross-Encoder"
    ])

    # --------------------------------------------------------------------------
    # ABA 1: HÍBRIDO RRF
    # --------------------------------------------------------------------------
    with tab_rrf:
        st.subheader("Resultados Consolidados via Reciprocal Rank Fusion")
        st.markdown(f"Equilíbrio atual: **{alpha*100:.0f}% Léxico** vs **{(1-alpha)*100:.0f}% Semântico** ($k_{{rrf}} = {k_rrf}$)")

        for _, row in resultados.iterrows():
            with st.container():
                st.markdown(f"#### **#{row['rank_rrf']} — {row['titulo']}** `({row['id']})`")
                st.write(row["conteudo"])
                col1, col2, col3 = st.columns(3)
                col1.metric("Score RRF Final", f"{row['score_rrf']:.5f}")
                col2.metric("Posição Léxica", f"#{row['rank_bm25']}")
                col3.metric("Posição Semântica", f"#{row['rank_semantico']}")
                st.divider()

    # --------------------------------------------------------------------------
    # ABA 2: MOTOR LÉXICO (BM25)
    # --------------------------------------------------------------------------
    with tab_lexico:
        st.subheader("Ranqueamento por Termos Exatos (Okapi BM25)")
        tokens_identificados = preprocess_text(query_input)
        st.info(f"Tokens processados após limpeza e remoção de stopwords: `{tokens_identificados}`")

        df_bm25 = resultados.sort_values(by="score_bm25", ascending=False).reset_index(drop=True)
        for idx, row in df_bm25.iterrows():
            rank = idx + 1
            st.markdown(f"**Rank #{rank}** | {row['titulo']} — `{row['id']}`")
            st.caption(f"Score BM25: `{row['score_bm25']:.4f}`")
            st.write(row["conteudo"])
            st.markdown("---")

    # --------------------------------------------------------------------------
    # ABA 3: MOTOR SEMÂNTICO (VETORIAL)
    # --------------------------------------------------------------------------
    with tab_semantico:
        st.subheader("Ranqueamento por Proximidade Conceitual (Cosine Similarity)")
        st.caption("Modelo: `paraphrase-multilingual-MiniLM-L12-v2` | Normalização $L_2$ aplicada aos vetores.")

        df_sem = resultados.sort_values(by="score_semantico", ascending=False).reset_index(drop=True)
        for idx, row in df_sem.iterrows():
            rank = idx + 1
            st.markdown(f"**Rank #{rank}** | {row['titulo']} — `{row['id']}`")
            st.caption(f"Similaridade Cosseno: `{row['score_semantico']:.4f}`")
            st.write(row["conteudo"])
            st.markdown("---")

    # --------------------------------------------------------------------------
    # ABA 4: MATRIZ COMPARATIVA DE RANKS
    # --------------------------------------------------------------------------
    with tab_comparativo:
        st.subheader("Diagnóstico Cruzado de Recuperação")
        st.markdown(
            "Esta tabela evidencia discrepâncias onde um motor isolado falha e como a fusão "
            "equilibra as posições."
        )

        display_df = resultados[[
            "id", "titulo", "rank_rrf", "rank_bm25", "rank_semantico", 
            "score_rrf", "score_bm25", "score_semantico"
        ]].copy()

        display_df.columns = [
            "ID", "Título da Diretriz", "Rank RRF", "Rank BM25", "Rank Semântico",
            "Score RRF", "Score BM25", "Similaridade Cosseno"
        ]

        st.dataframe(
            display_df.style.highlight_min(subset=["Rank RRF", "Rank BM25", "Rank Semântico"], color="#d4edda"),
            use_container_width=True
        )

        st.markdown("##### 📌 Análise de Desempenho dos Casos Limite:")
        st.markdown(
            "- **Consultas com Códigos Exatos (ex: `CÓD-ECG-12D`)**: O BM25 atribui pontuação destacada devido à alta especificidade do token, enquanto modelos de embedding podem diluir a atenção entre conceitos cardíacos gerais.\n"
            "- **Consultas Semânticas/Sinônimos (ex: `infarto`)**: O modelo vetorial recupera o Doc 1 (síndrome coronariana) e o Doc 2 com alta proximidade, enquanto o BM25 zera a pontuação caso o termo literal não ocorra no documento."
        )

    # --------------------------------------------------------------------------
    # ABA 5: BÔNUS — RE-RANKING COM CROSS-ENCODER
    # --------------------------------------------------------------------------
    with tab_bonus:
        st.subheader("Re-Ranking com Cross-Encoder sobre o Top-3 RRF")
        if not apply_cross:
            st.warning("Marque o checkbox **'Habilitar Re-Ranking Cross-Encoder'** na barra lateral para ativar este módulo.")
        else:
            st.success("Modelo Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) computado com sucesso sobre a tupla (Query, Documento) dos 3 melhores colocados.")
            
            top_3_df = resultados.head(3).copy()
            top_3_sorted = top_3_df.sort_values(by="score_cross_encoder", ascending=False).reset_index(drop=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("##### 🥉 Entrada (Top-3 via RRF)")
                for _, r in top_3_df.iterrows():
                    st.info(f"**Posição #{r['rank_rrf']}** — {r['titulo']}\n\n*Score RRF:* `{r['score_rrf']:.5f}`")

            with col_b:
                st.markdown("##### 🏆 Saída Reordenada (Cross-Encoder)")
                for idx, r in top_3_sorted.iterrows():
                    nova_pos = idx + 1
                    st.success(f"**Nova Posição #{nova_pos}** — {r['titulo']}\n\n*Logits Cross-Encoder:* `{r['score_cross_encoder']:.4f}`")
else:
    st.info("Digite uma consulta acima ou selecione um dos exemplos para visualizar o diagnóstico de recuperação.")