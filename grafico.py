import matplotlib.pyplot as plt
import numpy as np

# Dados simulados para a query: "infarto"
docs = ['Doc 1\n(ECG/SCA)', 'Doc 2\n(Infarto/AAS)', 'Doc 3\n(Pressão)', 'Doc 4\n(AVC)', 'Doc 5\n(Parada)', 'Doc 6\n(UTI/ECG)']
rank_bm25 = [6, 1, 6, 6, 6, 6]       # BM25 só pontua o Doc 2 pelo termo exato; empata os demais
rank_sem = [2, 1, 4, 3, 5, 6]        # Semântico recupera Doc 2 e Doc 1 no topo
rank_rrf = [2, 1, 5, 3, 4, 6]        # RRF equilibra

x = np.arange(len(docs))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
rects1 = ax.bar(x - width, rank_bm25, width, label='Rank BM25 (Léxico)', color='#e74c3c')
rects2 = ax.bar(x, rank_sem, width, label='Rank Semântico (Cosseno)', color='#3498db')
rects3 = ax.bar(x + width, rank_rrf, width, label='Rank RRF (Híbrido)', color='#2ecc71')

ax.set_ylabel('Posição no Ranking (Menor = Melhor)')
ax.set_title('Comparação de Ranks por Método — Consulta: "infarto"')
ax.set_xticks(x)
ax.set_xticklabels(docs, fontsize=9)
ax.set_ylim(0.5, 6.5)
ax.invert_yaxis()  # Rank 1 no topo
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig("grafico_comparacao_ranks.png")
print("Gráfico salvo como grafico_comparacao_ranks.png para o PDF.")