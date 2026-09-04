# 🏥 HealthSearch: Motor de Busca Híbrido Clínico

Projeto desenvolvido para o **Laboratório Prático 05 - Desafio Integrador** da disciplina de **Tendências em Ciência da Computação (Recuperação de Informação / Processamento de Linguagem Natural)** do Centro Universitário de João Pessoa (**UNIPÊ**), sob orientação do **Prof. Me. Ricardo Roberto de Lima**.

---

## 📌 Visão Geral do Problema

Sistemas hospitalares enfrentam dilemas críticos de recuperação da informação médica:
* **Falha Léxica Pura:** Consultas como *"ataque cardíaco"* não encontram diretrizes cadastradas estritamente sob termos técnicos formais como *"síndrome coronariana aguda"* ou *"isquemia miocárdica"*.
* **Falha Semântica Pura:** Códigos de exames específicos (ex: `CÓD-ECG-12D`) ou dosagens têm sua precisão diluída em espaços vetoriais, trazendo apenas documentos genéricos sobre cardiologia[cite: 1].

O **HealthSearch** soluciona esses pontos cegos integrando o algoritmo léxico **Okapi BM25** e representações densas vetoriais (**Sentence-Transformers**) por meio do algoritmo **Reciprocal Rank Fusion (RRF)**[cite: 1].

---

## 🏗️ Arquitetura do Sistema

A aplicação é dividida em 4 fases sequenciais e uma camada bônus de refinamento[cite: 1]:

1. **Fase 1: Ingestão e Pré-processamento Léxico:** Normalização, preservação de tokens alfanuméricos compostos com hífen (ex: `CÓD-ECG-12D`) e eliminação de stopwords em português[cite: 1].
2. **Fase 2: Motor Léxico (BM25):** Indexação com o algoritmo Okapi BM25, permitindo controle interativo dos hiperparâmetros de saturação ($k_1$) e normalização de tamanho ($b$)[cite: 1].
3. **Fase 3: Motor Semântico Vetorial:** Geração de embeddings densos normalizados via modelo multilíngue (`paraphrase-multilingual-MiniLM-L12-v2`) e cálculo de Similaridade de Cosseno[cite: 1].
4. **Fase 4: Fusão Híbrida RRF:** Combinação dos ranks calculada via fórmula[cite: 1]:
   $$\text{Score}_{RRF}(D) = \alpha \cdot \left[ \frac{1}{k_{rrf} + \text{Rank}_{BM25}} \right] + (1 - \alpha) \cdot \left[ \frac{1}{k_{rrf} + \text{Rank}_{Semantico}} \right]$$
   *(com $k_{rrf} = 60$ por padrão e $\alpha$ ajustável de $0.0$ a $1.0$)*[cite: 1].
5. **Módulo Bônus (Cross-Encoder Re-Ranking):** Aplicação do modelo `cross-encoder/ms-marco-MiniLM-L-6-v2` sobre os 3 documentos mais bem avaliados pelo RRF para inferência profunda par a par $(\text{query}, \text{documento})$[cite: 1].

---

## 🛠️ Tecnologias e Bibliotecas

* **Linguagem:** Python 3.10+[cite: 1]
* **Interface Web:** Streamlit[cite: 1]
* **Algoritmo Léxico:** `rank-bm25`[cite: 1]
* **Modelagem Semântica e Bônus:** `sentence-transformers`, `torch`[cite: 1]
* **Manipulação de Dados:** `pandas`, `numpy`[cite: 1]

---

## 🚀 Instalação e Execução

### 1. Clonar o repositório
```bash
git clone [https://github.com/seu-usuario/healthsearch.git](https://github.com/seu-usuario/healthsearch.git)
cd healthsearch